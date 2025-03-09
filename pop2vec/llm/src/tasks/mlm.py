import copy
import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from random import shuffle
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import cast
from xml.dom.minidom import Document
import numpy as np
import pandas as pd
import torch
from pop2vec.llm.src.data_new.types import Background
from pop2vec.llm.src.data_new.types import EncodedDocument
from pop2vec.llm.src.data_new.types import PersonDocument
from pop2vec.llm.src.new_code.constants import INF
from pop2vec.llm.src.tasks.base import Task
from pop2vec.llm.src.tasks.sentence_masking import find_sentences_to_mask
from pop2vec.llm.src.tasks.sentence_masking import mask_tokens_in_sentences

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

T = TypeVar("T")

min_event_threshold = 5


@dataclass
class MLM(Task):
    """Task for used with Masked language modelling.

    .. todo::
        Describe MLM

    :param mask_ratio: Fraction of tokens to mask.
    :param smart_masking: Whether to apply smart masking (use tokens from the same group when choosing randoms).
    :param event_masking: Whether to apply masking at the event level. Default is False.
    """

    # MLM Specific params
    mask_ratio: float = 0.30
    smart_masking: bool = False
    vocabulary = None
    found_max_len = -1
    found_min_len = 1000000000
    time_range = [-INF, INF]
    masking: str = "random"

    def __post_init__(self):
        if self.masking not in ["random", "event"]:
            raise NotImplementedError

    def set_time_range(self, time_range: Tuple[int, int]):
        self.time_range = time_range

    def set_vocabulary(self, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.datamodule.vocabulary
        self.vocabulary = vocabulary

    def slice_by_time(self, document):
        if self.time_range == (-INF, +INF):
            return document
        lower_bound = np.searchsorted(document.abspos, self.time_range[0], side="left")
        upper_bound = np.searchsorted(document.abspos, self.time_range[1], side="right")
        document.sentences = document.sentences[lower_bound:upper_bound]
        document.age = document.age[lower_bound:upper_bound]
        document.abspos = document.abspos[lower_bound:upper_bound]
        document.segment = document.segment[lower_bound:upper_bound]
        return document

    def encode_document(
        self,
        document: PersonDocument,
        do_log: bool = False,
        do_mlm: bool = True,
    ) -> "MLMEncodedDocument":
        if do_log:
            logger.debug(
                f"RINPERSOON = {document.person_id}\n"
                f"first year active = {int(2017 - (16408 - np.min(document.abspos)) / 365)}\n"
                f"last year active = {int(2017 - (16408 - np.max(document.abspos)) / 365)}\n"
                f"min time = {np.min(document.abspos)}, max time = {np.max(document.abspos)}, threshold = {self.time_range}\n"
                f"min event age = {np.min(document.age)}, max event age = {np.max(document.age)}\n"
                f"background\n{document.background}\n"
                f"all events\n{document.sentences}\n"
                f"all ages\n{document.age}"
            )


        # Slice document by time range
        len_before = len(document.sentences)
        document = self.slice_by_time(document)
        len_after = len(document.sentences)

        if do_log:
            logger.debug(
                f"len_before {len_before} & len_after {len_after}\n"
                f"Sentences after time slicing\n{document.sentences}\n"
                f"all ages\n{document.age}\n"
            )

        # Get rid of all documents who have less than threshold # of events after slicing by time
        if len(document.sentences) < min_event_threshold:
            return None

        prefix_sentence = ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]

        # Apply CLS task transformations
        document, targ_cls = self.cls_task(document, do_mlm)

        # Construct sentences with [SEP] tokens
        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = np.array([len(s) for s in sentences])

        # Calculate cumulative lengths in reverse order to determine THRESHOLD
        reversed_lengths = sentence_lengths[1:][::-1]
        cumsum_lengths = np.cumsum(reversed_lengths)
        total_lengths = len(prefix_sentence) + cumsum_lengths

        # Determine how many sentences can fit within max_length
        indices = np.where(total_lengths < self.max_length)[0]
        THRESHOLD = indices[-1] + 1 if len(indices) > 0 else 0

        if do_log:
            logger.debug(
                f"total sentences = {len(sentence_lengths)}, ok = {THRESHOLD}\n"
                f"lengths of sentences = {[(i, sentence_lengths[i]) for i in range(len(sentence_lengths))]}\n"
            )
        # Slice the document to include only the sentences that fit
        if THRESHOLD > 0:
            document.sentences = document.sentences[-THRESHOLD:]
            document.age = document.age[-THRESHOLD:]
            document.abspos = document.abspos[-THRESHOLD:]
            document.segment = document.segment[-THRESHOLD:]
            sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
            sentence_lengths = np.array([len(s) for s in sentences])
        else:
            document.sentences = []
            document.age = []
            document.abspos = []
            document.segment = []
            sentences = [prefix_sentence]
            sentence_lengths = np.array([len(s) for s in sentences])

        if do_log:
            logger.debug(
                f"Sentences after thresholding due to max_len\n{document.sentences}\n"
                f"all ages\n{document.age}\n"
            )
        # Efficiently expand properties using numpy.repeat
        x_abspos = np.array([0] + document.abspos)
        abspos_expanded = np.repeat(x_abspos, sentence_lengths)

        x_age = np.array([0.0] + document.age)
        age_expanded = np.repeat(x_age, sentence_lengths)

        x_segment = np.array([0] + document.segment)
        segment_expanded = np.repeat(x_segment, sentence_lengths)

        # Concatenate sentences into a flat array
        flat_sentences = np.concatenate(sentences)

        # Efficient token to index mapping using pandas for vectorization
        token2index = self.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        flat_sentences_series = pd.Series(flat_sentences)
        token_ids = flat_sentences_series.map(token2index).fillna(unk_id).astype(int).values

        length = len(token_ids)
        self.found_max_len = max(self.found_max_len, length)
        self.found_min_len = min(self.found_min_len, length)

        if do_log:
            logger.debug(
                f"length = {length}, max = {self.found_max_len}, min = {self.found_min_len}"
            )

        # Create padding mask
        padding_mask = np.zeros(self.max_length, dtype=bool)
        padding_mask[:length] = True

        # Prepare input_ids and original_sequence
        original_sequence = np.zeros(self.max_length, dtype=int)
        original_sequence[:length] = token_ids

        sequence_id = np.array(document.person_id)
        input_ids = np.zeros((4, self.max_length), dtype=float)
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        if do_mlm:
            masked_sentences, masked_indx, masked_tokens = self.mlm_mask(token_ids.copy())
            input_ids[0, :length] = masked_sentences

            return MLMEncodedDocument(
                sequence_id=sequence_id,
                input_ids=input_ids,
                padding_mask=padding_mask,
                target_tokens=masked_tokens,
                target_pos=masked_indx,
                target_cls=targ_cls,
                original_sequence=original_sequence,
            )

        input_ids[0, :length] = original_sequence[:length]
        return SimpleEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            original_sequence=original_sequence,
        )

    # These could (maybe should?) also be calculated in the __post_init__.
    # Accessing the serialized methods in a parallel context may give problems down
    # the line.
    @cached_property
    def token_groups(self) -> List[Tuple[int, int]]:
        """Return pairs of first and last index for each token category in the
        vocabulary excluding GENERAL.
        """
        vocab = self.vocabulary.vocab()

        no_general = vocab.CATEGORY != "GENERAL"
        token_groups = (
            vocab.loc[no_general]
            .groupby("CATEGORY")
            .ID.agg(["first", "last"])
            .sort_values("first")
            .to_records(index=False)
            .tolist()
        )
        return cast(List[Tuple[int, int]], token_groups)

    def cls_task(self, document: PersonDocument, do_mlm: bool = True, permute_prob: float = 0.05):
        """Convert sequence for CLS task.

        Arguments:
            document: PersonDocument to encode.
            do_mlm: boolean indicating if MLM encoding or not. Default
            is True. If False, document is returned as it is.
            permute_prob: Probability of permuting sequences. For `permute_prob`
            of sequences, the sentences are reversed. For `permute_prob` of
            sequences, the sentences are randomly shuffled. This means that
            a fraction of (1 - 2*permute_prob) sequences are kept as they are.
        """
        max_prob = 0.5

        if not do_mlm:
            return document, 0

        if permute_prob >= max_prob:
            msg = "Set permutation probability to less than 0.5"
            raise ValueError(msg)

        p = np.random.rand(1)
        if p < permute_prob:
            document.sentences.reverse()
            targ_cls = 1
        elif p > 1 - permute_prob:
            shuffle(document.sentences)
            targ_cls = 2
        else:
            targ_cls = 0

        return document, targ_cls

    def mlm_mask(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mask out the tokens for mlm training."""
        vocabulary = self.vocabulary
        token2index = vocabulary.token2index

        unk_id = token2index["[UNK]"]
        mask_id = token2index["[MASK]"]
        sep_id = token2index["[SEP]"]

        max_masked_num = np.floor(self.mask_ratio * self.max_length).astype(np.int32)

        if self.masking == "event":
            ignore_tokens = vocabulary.general_tokens
            ignore_tokens = [token2index[x] for x in ignore_tokens]
            return self._mask_events(
                    token_ids, mask_id, sep_id, ignore_tokens, max_masked_num)

        vocab_size = len(token2index)
        n_general_tokens = len(vocabulary.general_tokens)
        return self._mask_single_tokens(
                token_ids, mask_id, sep_id, unk_id,
                vocab_size, n_general_tokens, max_masked_num)

    def _mask_events(self, token_ids, mask_id, sep_id, ignore_tokens, len_target_array):
        sentence_start_pos, sentences_to_mask = find_sentences_to_mask(
            token_ids, sep_id, self.mask_ratio, ignore_tokens
        )
        return mask_tokens_in_sentences(
                token_ids, sentence_start_pos, sentences_to_mask, mask_id,
                len_target_array, int(self.max_length - 1)
                )


    def _mask_single_tokens(
            self,
            token_ids,
            mask_id,
            sep_id,
            unk_id,
            vocab_size,
            n_general_tokens,
            max_masked_num):
        """Mask tokens individually. Copied from old code."""
        # limit is length of an actual sequence
        n_tokens = len(token_ids)

        num_tokens_to_mask = np.floor(n_tokens * self.mask_ratio).astype(np.int32)
        # first 10% of tokens won't be changed
        pos_unchange = np.floor(num_tokens_to_mask * 0.1).astype(np.int32)
        # last 10% of tokens would be random ; the rest will be changed
        pos_random = num_tokens_to_mask - pos_unchange

        # we do not mask SEP and UNK
        legal_mask = (token_ids[1:] != sep_id) & (token_ids[1:] != unk_id)
        legal_indx = np.arange(start=1, stop=n_tokens)[legal_mask]

        indx_to_mask = np.random.choice(a=legal_indx, size=num_tokens_to_mask, replace=False)

        #max_masked_num = np.floor(self.mask_ratio * self.max_length).astype(np.int32)

        # positions of the masked tokens
        y_indx = np.full(shape=max_masked_num, fill_value=int(self.max_length - 1))
        y_indx[: len(indx_to_mask)] = indx_to_mask.copy()

        # remember the actual tokens on positions
        y_token = np.zeros(shape=max_masked_num)
        y_token[: len(indx_to_mask)] = token_ids[indx_to_mask].copy()

        # masked token_ids #rather change the sampling domain for accurate masking
        # ratio?
        token_ids[indx_to_mask[pos_unchange:pos_random]] = mask_id

        if self.smart_masking:
            smart_edge = int(pos_random + int(pos_unchange * 0.3))

            # Random 7% of all random cases
            token_ids[indx_to_mask[smart_edge:]] = np.random.randint(
                # low we do not mask any special tokens
                low=n_general_tokens,
                high=vocab_size,
                size=(1, len(indx_to_mask[smart_edge:])),
            )

            # Smart Random 3% of all the cases
            smart_values = token_ids[indx_to_mask[pos_random:smart_edge]]

            for i, j in self.token_groups:
                smart_values = self.smart_masked(smart_values, i, j + 1)  # background

            token_ids[indx_to_mask[pos_random:smart_edge]] = smart_values

        else:
            token_ids[indx_to_mask[pos_random:]] = torch.randint(
                # low we do not mask any special tokens
                low=n_general_tokens,
                high=vocab_size,
                size=(1, len(indx_to_mask[pos_random:])),
            )
        return token_ids, y_indx, y_token

    @staticmethod
    def smart_masked(x: np.ndarray, min_i: int, max_i: int) -> np.ndarray:
        """Applies the smart_masking scheme."""
        ix = np.argwhere((x >= min_i) & (x < max_i))
        if len(ix) > 0:
            x[ix] = np.random.randint(low=min_i, high=max_i, size=(len(ix), 1))
        return x


@dataclass
class MLMEncodedDocument(EncodedDocument[MLM]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target_tokens: np.ndarray
    target_pos: np.ndarray
    target_cls: np.ndarray
    original_sequence: np.ndarray


@dataclass
class SimpleEncodedDocument(EncodedDocument[MLM]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    original_sequence: np.ndarray
