import pytest
import torch

from gpt1.data import LanguageModelDataset


def test_dataset_length_uses_complete_sequences() -> None:
    token_ids = list(range(13))

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        seq_len=4,
    )

    assert len(dataset) == 3


def test_first_sample_has_shifted_labels() -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(13)),
        seq_len=4,
    )

    sample = dataset[0]

    assert torch.equal(sample["input_ids"], torch.tensor([0, 1, 2, 3]))

    assert torch.equal(
        sample["labels"],
        torch.tensor([1, 2, 3, 4]),
    )


def test_last_sample_starts_at_next_chunk() -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(13)),
        seq_len=4,
    )

    sample = dataset[1]

    assert torch.equal(sample["input_ids"], torch.tensor([4, 5, 6, 7]))

    assert torch.equal(
        sample["labels"],
        torch.tensor([5, 6, 7, 8]),
    )


def test_labels_are_shifted_for_every_sample() -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(13)),
        seq_len=4,
    )

    for sample in dataset:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        assert torch.equal(input_ids[1:], labels[:-1])


def test_sample_has_expected_key_shape_and_dtype() -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(9)),
        seq_len=4,
    )

    sample = dataset[0]

    assert set(sample) == {"input_ids", "labels"}
    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long
    assert sample["input_ids"].shape == (4,)
    assert sample["labels"].shape == (4,)


def test_dataset_accepts_integer_tensor() -> None:
    token_ids = torch.arange(
        9,
        dtype=torch.int32,
    )

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        seq_len=4,
    )
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long


def test_incomplete_final_sequence_is_ignored() -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(10)),
        seq_len=4,
    )

    assert len(dataset) == 2

    final_sample = dataset[1]

    assert torch.equal(
        final_sample["labels"],
        torch.tensor([5, 6, 7, 8]),
    )


@pytest.mark.parametrize(
    ("seq_len", "expected_exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_invalid_sequence_length_is_rejected(
    seq_len: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        LanguageModelDataset(
            token_ids=list(range(10)),
            seq_len=seq_len,
        )


def test_too_few_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="至少"):
        LanguageModelDataset(
            token_ids=[0, 1, 2, 3],
            seq_len=4,
        )


@pytest.mark.parametrize(
    ("token_ids", "expected_exception"),
    [
        ([0, 1, -1], ValueError),
        ([0, True, 2], TypeError),
        ([0, 1.5, 2], TypeError),
    ],
)
def test_invalid_token_ids_are_rejected(
    token_ids: list[object],
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        LanguageModelDataset(
            token_ids=token_ids,
            seq_len=2,
        )


def test_multidimensional_tensor_is_rejected() -> None:
    token_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ]
    )

    with pytest.raises(ValueError, match="一维"):
        LanguageModelDataset(
            token_ids=token_ids,
            seq_len=2,
        )


def test_boolean_tensor_is_rejected() -> None:
    token_ids = torch.tensor(
        [True, False, True]
    )

    with pytest.raises(TypeError, match="整数"):
        LanguageModelDataset(
            token_ids=token_ids,
            seq_len=2,
        )


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_sample_index_is_rejected(
    index: int,
) -> None:
    dataset = LanguageModelDataset(
        token_ids=list(range(9)),
        seq_len=4,
    )

    with pytest.raises(IndexError):
        dataset[index]
