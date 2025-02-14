import numpy as np
import pytest
from pop2vec.llm.src.tasks.sentence_masking import find_sentences_to_mask
from pop2vec.llm.src.tasks.sentence_masking import mask_tokens_in_sentences

MASK_ID = 99
LEN_TARGET_ARRAY = 10
FILL_VAL_TARGET_IDX = 511


def test_find_sentences_basic():
    token_sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5])
    sep_token = 5
    mask_frac = 0.5
    ignore_tokens = [1]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    assert isinstance(starts_and_ends, np.ndarray)
    assert isinstance(sentences_to_mask, np.ndarray)
    assert len(sentences_to_mask) > 0


def test_find_sentences_invalid_inputs():
    token_sequence = np.array([1, 2, 3, 4, 5])
    sep_token = 5
    ignore_tokens = [1]

    with pytest.raises(ValueError):
        find_sentences_to_mask(token_sequence, sep_token, 0, ignore_tokens)

    with pytest.raises(ValueError):
        find_sentences_to_mask(token_sequence, sep_token, 1, ignore_tokens)

    with pytest.raises(TypeError):
        find_sentences_to_mask(list(token_sequence), sep_token, 0.4, ignore_tokens)  # pyright: ignore[reportArgumentType]

    token_sequence = np.array([1, 2, 3, 4, 5, 7])
    with pytest.raises(RuntimeError):
        find_sentences_to_mask(token_sequence, sep_token, 0.4, ignore_tokens)


def test_find_sentences_all_sentences_masked():
    token_sequence = np.array([1, 2, 3, 5, 6, 7, 5, 8, 9, 5])
    sep_token = 5
    mask_frac = 0.99
    ignore_tokens = [1]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    n_sentences = np.nonzero(token_sequence == sep_token)[0].shape[0]
    assert len(sentences_to_mask) == n_sentences, "Does not mask all sentences"


def test_find_sentences_no_sentences_masked():
    token_sequence = np.array([1, 2, 3, 5, 6, 7, 5, 8, 9, 5])
    sep_token = 5
    mask_frac = 0.1
    ignore_tokens = [1]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    assert len(sentences_to_mask) == 0, "no sentence should be masked"


def test_find_sentences_ignore_tokens():
    token_sequence = np.array([1, 2, 3, 4, 5, 6, 7, 5, 8, 9, 5])
    sep_token = 5
    mask_frac = 0.5
    ignore_tokens = [1, 2, 3]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    assert starts_and_ends[0] == len(ignore_tokens) - 1, "does not ignore tokens correctly"


def test_find_sentences_edge_cases():
    # single sentence
    token_sequence = np.array([1, 2, 3, 4, 5])
    sep_token = 5
    mask_frac = 0.5
    ignore_tokens = [1]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    assert len(sentences_to_mask) == 0, "no sentences should be masked"

    # ignore token followed by sep token
    token_sequence = np.array([1, 5, 2, 3, 4, 5, 6, 7, 5])
    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)
    expected_starts_and_ends = np.array([1, 5, 8])
    assert np.all(starts_and_ends == expected_starts_and_ends)

    # ignore sep token followed by ignore token
    token_sequence = np.array([5, 1, 2, 3, 4, 5, 6, 7, 5])
    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)
    expected_starts_and_ends = np.array([1, 5, 8])
    assert np.all(starts_and_ends == expected_starts_and_ends)


def test_find_sentences_output_types():
    token_sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 5])
    sep_token = 5
    mask_frac = 0.5
    ignore_tokens = [1]

    starts_and_ends, sentences_to_mask = find_sentences_to_mask(token_sequence, sep_token, mask_frac, ignore_tokens)

    assert isinstance(starts_and_ends, np.ndarray)
    assert starts_and_ends.dtype == np.int_
    assert isinstance(sentences_to_mask, np.ndarray)
    assert sentences_to_mask.dtype == np.int_


def test_mask_tokens_in_sentences_basic():
    # single sentence
    token_sequence = np.array([1, 2, 3, 5, 8, 2, 5, 10, 15, 3, 5])
    sentence_starts_and_ends = np.array([-1, 3, 6, 10])
    sentences_to_mask = np.array([1])  # Mask the second sentence

    expected_masked_sequence = np.array([1, 2, 3, 5, 99, 99, 5, 10, 15, 3, 5])
    expected_target_idx = np.hstack((np.array([4, 5]), np.full(8, FILL_VAL_TARGET_IDX)))
    expected_target_tokens = np.hstack((np.array([8, 2]), np.zeros(8)))

    masked_sequence, target_idx, target_tokens = mask_tokens_in_sentences(
        token_sequence, sentence_starts_and_ends, sentences_to_mask, MASK_ID, LEN_TARGET_ARRAY, FILL_VAL_TARGET_IDX
    )

    np.testing.assert_array_equal(masked_sequence, expected_masked_sequence)
    np.testing.assert_array_equal(target_idx, expected_target_idx)
    np.testing.assert_array_equal(target_tokens, expected_target_tokens)

    # multiple sentences
    sentences_to_mask = np.array([0, 2])

    expected_masked_sequence = np.array([99, 99, 99, 5, 8, 2, 5, 99, 99, 99, 5])
    expected_target_idx = np.hstack((np.array([0, 1, 2, 7, 8, 9]), np.full(4, FILL_VAL_TARGET_IDX)))
    expected_target_tokens = np.hstack((np.array([1, 2, 3, 10, 15, 3]), np.zeros(4)))

    masked_sequence, target_idx, target_tokens = mask_tokens_in_sentences(
        token_sequence, sentence_starts_and_ends, sentences_to_mask, MASK_ID, LEN_TARGET_ARRAY, FILL_VAL_TARGET_IDX
    )

    np.testing.assert_array_equal(masked_sequence, expected_masked_sequence)
    np.testing.assert_array_equal(target_idx, expected_target_idx)
    np.testing.assert_array_equal(target_tokens, expected_target_tokens)


def test_mask_tokens_in_sentences_no_masking():
    token_sequence = np.array([1, 2, 3, 4, 5])
    sentence_starts_and_ends = np.array([0, 5])
    sentences_to_mask = np.array([])

    masked_sequence, target_idx, target_tokens = mask_tokens_in_sentences(
        token_sequence, sentence_starts_and_ends, sentences_to_mask, MASK_ID, LEN_TARGET_ARRAY, FILL_VAL_TARGET_IDX
    )

    np.testing.assert_array_equal(masked_sequence, token_sequence)
    assert target_idx.size == LEN_TARGET_ARRAY
    assert target_tokens.size == LEN_TARGET_ARRAY


def test_mask_tokens_in_sentences_all_masking():
    token_sequence = np.array([1, 2, 3, 4, 2, 5])  # 1 is an ignore token
    sentence_starts_and_ends = np.array([0, 5])
    sentences_to_mask = np.array([0])

    expected_masked_sequence = np.array([1, 99, 99, 99, 99, 5])
    expected_target_idx = np.hstack((np.array([1, 2, 3, 4]), np.full(6, FILL_VAL_TARGET_IDX)))
    expected_target_tokens = np.hstack((np.array([2, 3, 4, 2]), np.zeros(6)))

    masked_sequence, target_idx, target_tokens = mask_tokens_in_sentences(
        token_sequence, sentence_starts_and_ends, sentences_to_mask, MASK_ID, LEN_TARGET_ARRAY, FILL_VAL_TARGET_IDX
    )

    np.testing.assert_array_equal(masked_sequence, expected_masked_sequence)
    np.testing.assert_array_equal(target_idx, expected_target_idx)
    np.testing.assert_array_equal(target_tokens, expected_target_tokens)


def test_mask_tokens_in_sentences_input_validation():
    with pytest.raises(TypeError):
        mask_tokens_in_sentences(
            [1, 2, 3],  # pyright: ignore[reportArgumentType]
            np.array([0, 3]),
            np.array([0]),
            99,
            10,
            511,
        )


def test_mask_tokens_in_sentences_empty_input():
    token_sequence = np.array([])
    sentence_starts_and_ends = np.array([0])
    sentences_to_mask = np.array([])

    masked_sequence, target_idx, target_tokens = mask_tokens_in_sentences(
        token_sequence, sentence_starts_and_ends, sentences_to_mask, MASK_ID, LEN_TARGET_ARRAY, FILL_VAL_TARGET_IDX
    )

    assert masked_sequence.size == 0
    assert target_idx.size == LEN_TARGET_ARRAY
    assert target_tokens.size == LEN_TARGET_ARRAY
