from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def find_sentences_to_mask(
    token_sequence: NDArray[np.int_],
    sep_token: int,
    mask_frac: float,
    ignore_tokens: list[int],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Determine random set of sentences to be masked.

    Args:
        token_sequence (np.ndarray): the sequence of tokens to mask.
        sep_token (int): the identifier of the SEP token.
        mask_frac (float): the fraction of tokens to be masked. Must be between 0 and 1.
        ignore_tokens (list): token ids to be ignored at the start of the sequence.
            Concerns the CLS token at the start of the sequence, and possibly others.

    Returns:
        A tuple of numpy arrays.
        - The first entry are the start/end positions of sentence: The position of the `sep_tokens` and the
            last `ignore_token` before the first sentence starts.
        - The second are the sentences selected for masking.
    """
    rng = np.random.default_rng()
    n_tokens = len(token_sequence)
    ignore_tokens_arr = np.asarray(ignore_tokens)

    idx_sentence_sep = np.nonzero(token_sequence == sep_token)[0]
    sentence_lengths = np.hstack((idx_sentence_sep[0], np.diff(idx_sentence_sep)))

    sentence_ids = np.arange(len(idx_sentence_sep) - 1)  # the last SEP token is the end of the sequence
    rng.shuffle(sentence_ids)

    lengths_shuffled = sentence_lengths[sentence_ids]
    lengths_cumsum = np.cumsum(lengths_shuffled)
    sentences_to_mask = lengths_cumsum / n_tokens <= mask_frac
    if not np.any(sentences_to_mask):
        sentences_to_mask[0] = True

    # we'll ignore the sep tokens for masking
    pos_first_relevant_token = np.min(np.nonzero(~np.isin(token_sequence, ignore_tokens_arr))) - 1
    sentence_starts_and_ends = np.hstack((pos_first_relevant_token, idx_sentence_sep))

    return sentence_starts_and_ends, np.sort(sentence_ids[sentences_to_mask])


def mask_tokens_in_sentences(
    token_sequence: NDArray[np.int_],
    sentence_starts_and_ends: NDArray[np.int_],
    sentences_to_mask: NDArray[np.int_],
    mask_id: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Mask all tokens belonging to the same sentences.

    Args:
        token_sequence (np.ndarray): sequence of tokens to be masked.
        sentence_starts_and_ends (np.ndarray): the indices of non-sentence tokens surrounding sentences.
            These are SEP tokens, and any general token before the first sentence. Masking a whole
            sentence will mask all tokens *between* two surrounding, non-sentence tokens.
        sentences_to_mask (np.ndarray): indices of sentences, among all sentences, to be masked.
        mask_id (int): the token ID for the mask.

    Returns: A tuple of np.ndarrays:
    - The first entry is the masked sequence
    - The second entry are the indices of the masked tokens
    - The third entry are the true values of the masked tokens
    """
    masking_idx_start = sentence_starts_and_ends[sentences_to_mask]
    masking_idx_stop = sentence_starts_and_ends[sentences_to_mask + 1]
    token_indices = np.arange(len(token_sequence))

    mask = (token_indices[:, None] > masking_idx_start) & (token_indices[:, None] < masking_idx_stop)
    mask = mask.any(axis=1)

    masked_sequence = token_sequence.copy()
    masked_sequence[mask] = mask_id
    target_tokens = token_sequence[mask]
    target_idx = np.nonzero(mask)[0]

    return masked_sequence, target_idx, target_tokens
