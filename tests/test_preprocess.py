from pathlib import Path

import pytest
import torch

from gpt1.tokenizer import CharacterTokenizer
from scripts.preprocess import (
    encode_text_files,
    find_text_files,
    main,
    save_token_ids,
    split_token_ids,
)


def test_find_text_files_recursively_and_in_sorted_order(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)

    (input_dir / "b.txt").write_text("b", encoding="utf-8")
    (input_dir / "a.txt").write_text("a", encoding="utf-8")
    (nested_dir / "c.txt").write_text("c", encoding="utf-8")
    (input_dir / "ignored.md").write_text("ignored", encoding="utf-8")

    text_files = find_text_files(input_dir, "*.txt")

    assert text_files == sorted(
        [
            input_dir / "a.txt",
            input_dir / "b.txt",
            nested_dir / "c.txt",
        ]
    )


def test_find_text_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        find_text_files(tmp_path / "missing", "*.txt")


def test_find_text_files_rejects_file_as_input_directory(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("text", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="不是一个目录"):
        find_text_files(input_path, "*.txt")


def test_find_text_files_rejects_empty_match(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "corpus.md").write_text("text", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="未找到"):
        find_text_files(input_dir, "*.txt")


def test_encode_text_files_adds_eos_after_each_document(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "a.txt"
    second_path = tmp_path / "b.txt"
    first_path.write_text("ab", encoding="utf-8")
    second_path.write_text("cd", encoding="utf-8")

    tokenizer = CharacterTokenizer.train(["ab", "cd"])

    token_ids = encode_text_files(
        [first_path, second_path],
        tokenizer,
    )

    expected = [
        *tokenizer.encode("ab"),
        tokenizer.eos_token_id,
        *tokenizer.encode("cd"),
        tokenizer.eos_token_id,
    ]

    assert token_ids.dtype == torch.int32
    assert token_ids.tolist() == expected


def test_encode_text_files_uses_unk_for_unknown_characters(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "unknown.txt"
    text_path.write_text("z", encoding="utf-8")
    tokenizer = CharacterTokenizer.train(["abc"])

    token_ids = encode_text_files([text_path], tokenizer)

    assert token_ids.tolist() == [
        tokenizer.unk_token_id,
        tokenizer.eos_token_id,
    ]


def test_encode_text_files_rejects_empty_file_list() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])

    with pytest.raises(ValueError, match="没有生成任何 token"):
        encode_text_files([], tokenizer)


def test_encode_text_files_rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "invalid.txt"
    text_path.write_bytes(b"\xff\xfe")
    tokenizer = CharacterTokenizer.train(["abc"])

    with pytest.raises(ValueError, match="无法读取"):
        encode_text_files([text_path], tokenizer)


def test_split_token_ids_uses_requested_ratio() -> None:
    token_ids = torch.arange(10, dtype=torch.int32)

    train_ids, validation_ids = split_token_ids(
        token_ids,
        validation_ratio=0.2,
    )

    assert train_ids.tolist() == list(range(8))
    assert validation_ids.tolist() == [8, 9]
    assert train_ids.dtype == token_ids.dtype
    assert validation_ids.dtype == token_ids.dtype


@pytest.mark.parametrize(
    ("validation_ratio", "expected_exception"),
    [
        (True, TypeError),
        ("0.1", TypeError),
        (0.0, ValueError),
        (1.0, ValueError),
        (-0.1, ValueError),
        (1.1, ValueError),
    ],
)
def test_split_token_ids_rejects_invalid_ratio(
    validation_ratio: object,
    expected_exception: type[Exception],
) -> None:
    token_ids = torch.arange(10)

    with pytest.raises(expected_exception):
        split_token_ids(token_ids, validation_ratio)


def test_split_token_ids_rejects_multidimensional_tensor() -> None:
    token_ids = torch.arange(12).reshape(3, 4)

    with pytest.raises(ValueError, match="一维"):
        split_token_ids(token_ids, validation_ratio=0.1)


def test_split_token_ids_requires_at_least_two_tokens() -> None:
    token_ids = torch.tensor([1])

    with pytest.raises(ValueError, match="至少为 2"):
        split_token_ids(token_ids, validation_ratio=0.1)


def test_save_token_ids_creates_parent_and_round_trips(
    tmp_path: Path,
) -> None:
    token_ids = torch.arange(8, dtype=torch.int32)
    output_path = tmp_path / "processed" / "train.bin"

    save_token_ids(token_ids, output_path)
    loaded = torch.load(output_path, weights_only=True)

    assert output_path.is_file()
    assert torch.equal(loaded, token_ids)


def test_main_rejects_same_relative_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="不能相同"):
        main(
            [
                "--train-output",
                "processed.bin",
                "--validation-output",
                "processed.bin",
            ]
        )


def test_main_runs_complete_preprocessing_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    first_text = "hello"
    second_text = "world"
    (input_dir / "a.txt").write_text(first_text, encoding="utf-8")
    (input_dir / "b.txt").write_text(second_text, encoding="utf-8")

    tokenizer = CharacterTokenizer.train([first_text, second_text])
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    train_output = tmp_path / "processed" / "train.bin"
    validation_output = tmp_path / "processed" / "validation.bin"

    main(
        [
            "--input-dir",
            str(input_dir),
            "--tokenizer-path",
            str(tokenizer_path),
            "--train-output",
            str(train_output),
            "--validation-output",
            str(validation_output),
            "--validation-ratio",
            "0.25",
        ]
    )

    train_ids = torch.load(train_output, weights_only=True)
    validation_ids = torch.load(validation_output, weights_only=True)
    output = capsys.readouterr().out

    assert train_ids.ndim == 1
    assert validation_ids.ndim == 1
    assert train_ids.dtype == torch.int32
    assert validation_ids.dtype == torch.int32
    assert train_ids.numel() + validation_ids.numel() == 12
    assert "语料预处理完成" in output
    assert "语料文件数: 2" in output
