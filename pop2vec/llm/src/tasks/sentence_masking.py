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
        token_sequence (np.ndarray): the sequence of tokens to mask. The last
            token needs to be a `sep_token`. The start of the first sentence
            is identified by the first token that is not a member of `ignore_tokens`.
        sep_token (int): the identifier of the SEP token.
        mask_frac (float): the fraction of tokens to be masked. Must be between 0 and 1.
        ignore_tokens (list): token ids to be ignored at the start of the sequence.
            Concerns the CLS token at the start of the sequence, and possibly others.

    Returns:
        A tuple of numpy arrays.
        - The first entry are the start/end positions of sentence:
            The position of the `sep_tokens` and the last `ignore_token`
            before the first sentence starts. Note these are the tokens
            that *surround* any given sentence.
        - The second are the sentences selected for masking.

    Raises:
        - a ValueError if the `mask_frac` is out of bounds
        - a RuntimeError if the sequence is not ended with a `sep_token`.

    Notes:
        The function always returns at least one sentence to mask. This means
        that the effect masking fraction of tokens masked is larger than specified
        through `mask_frac`. For larger sequences with many tokens, however, this
        is unlikely to happen.
    """
    if mask_frac <= 0 or mask_frac >= 1:
        msg = "mask frac needs to be strictly between 0 and 1."
        raise ValueError(msg)

    if not isinstance(token_sequence, np.ndarray):
        raise TypeError

    if token_sequence[-1] != sep_token:
        msg = "Last token needs to be a `sep_token`."
        raise RuntimeError(msg)

    rng = np.random.default_rng()
    n_tokens = len(token_sequence)
    ignore_tokens_arr = np.asarray(ignore_tokens)

    idx_sentence_sep = np.nonzero(token_sequence == sep_token)[0]
    idx_first_relevant_token = np.min(
        np.nonzero(~np.isin(token_sequence, ignore_tokens_arr) & ~np.isin(token_sequence, sep_token))
    )

    # If there are any sep tokens before the first relevant tokens, we ignore them
    idx_sentence_sep = idx_sentence_sep[idx_sentence_sep > idx_first_relevant_token]
    sentence_ids = np.arange(len(idx_sentence_sep))
    rng.shuffle(sentence_ids)

    sentence_lengths = np.diff(np.hstack((idx_first_relevant_token - 1, idx_sentence_sep)))
    # deduct one to get the number of relevant tokens in each sentence
    sentence_lengths -= 1

    lengths_shuffled = sentence_lengths[sentence_ids]
    lengths_cumsum = np.cumsum(lengths_shuffled)
    sentences_to_mask = lengths_cumsum / n_tokens <= mask_frac

    sentence_starts_and_ends = np.hstack((idx_first_relevant_token - 1, idx_sentence_sep))
    return sentence_starts_and_ends, np.sort(sentence_ids[sentences_to_mask])


def mask_tokens_in_sentences(  # noqa: PLR0913
    token_sequence: NDArray[np.int_],
    sentence_starts_and_ends: NDArray[np.int_],
    sentences_to_mask: NDArray[np.int_],
    mask_id: int,
    len_target_array: int,
    fill_val_target_idx: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Mask all tokens belonging to the same sentences.

    Args:
        token_sequence (np.ndarray): sequence of tokens to be masked.
        sentence_starts_and_ends (np.ndarray): the indices of non-sentence tokens surrounding sentences.
            These are SEP tokens, and any general token before the first sentence. Masking a whole
            sentence will mask all tokens *between* two surrounding, non-sentence tokens.
        sentences_to_mask (np.ndarray): indices of sentences, among all sentences, to be masked.
        mask_id (int): the token ID for the mask.
        len_target_array (int): length of the array returned for the indices and true
            values of the masked tokens. Keeping this constant across samples
            is important for the training algorithm.
        fill_val_target_idx (int): the value with which to pad the array of
            target tokens.

    Returns: A tuple of np.ndarrays:
    - The first entry is the masked sequence
    - The second entry are the indices of the masked tokens
    - The third entry are the true values of the masked tokens
    """
    if len(sentences_to_mask) == 0:  # TODO: this will also have to be fixed
        masked_sequence = token_sequence
        target_idx = np.array([])
        target_tokens = np.array([])

    else:
        masking_idx_start = sentence_starts_and_ends[sentences_to_mask]
        masking_idx_stop = sentence_starts_and_ends[sentences_to_mask + 1]
        token_indices = np.arange(len(token_sequence))

        mask = (token_indices[:, None] > masking_idx_start) & (token_indices[:, None] < masking_idx_stop)
        mask = mask.any(axis=1)

        masked_sequence = token_sequence.copy()
        masked_sequence[mask] = mask_id
        target_tokens = token_sequence[mask]
        target_idx = np.nonzero(mask)[0]

    target_idx = _pad_array(target_idx, len_target_array, fill_val_target_idx)
    target_tokens = _pad_array(target_tokens, len_target_array, 0)

    return masked_sequence, target_idx, target_tokens


def _pad_array(arr: NDArray, length: int, fill_value: int) -> NDArray:
    """Right-pad an array up to length with fill_value."""
    if len(arr) > length:
        raise ValueError

    out = np.full(shape=length, fill_value=fill_value)
    out[: len(arr)] = arr.copy()
    return out
