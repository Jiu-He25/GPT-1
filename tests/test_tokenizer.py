import json
from pathlib import Path

import pytest

from gpt1.tokenizer import CharacterTokenizer


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    """为多个测试提供一个小型 tokenizer。"""
    return CharacterTokenizer.train(
        [
            "hello",
            "GPT",
            "你好",
        ]
    )


def test_special_token_ids_are_fixed(
    tokenizer: CharacterTokenizer,
) -> None:
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.sep_token_id == 4


def test_training_is_deterministic() -> None:
    first = CharacterTokenizer.train(
        [
            "cab",
            "你好",
        ]
    )
    second = CharacterTokenizer.train(
        [
            "你好",
            "cab",
        ]
    )

    assert first.token_to_id == second.token_to_id


def test_encode_decode_round_trip(
    tokenizer: CharacterTokenizer,
) -> None:
    text = "hello你好"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_unknown_character_uses_unk_token() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])

    token_ids = tokenizer.encode("z")

    assert token_ids == [tokenizer.unk_token_id]
    assert tokenizer.decode(token_ids) == "<unk>"


def test_unknown_id_decodes_to_unk_token(
    tokenizer: CharacterTokenizer,
) -> None:
    assert tokenizer.decode([999]) == "<unk>"


def test_encode_can_add_bos_and_eos(
    tokenizer: CharacterTokenizer,
) -> None:
    token_ids = tokenizer.encode(
        "hello",
        add_bos=True,
        add_eos=True,
    )

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids) == "<bos>hello<eos>"


def test_decode_can_skip_special_tokens(
    tokenizer: CharacterTokenizer,
) -> None:
    token_ids = [
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        *tokenizer.encode("hello"),
        tokenizer.sep_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    ]

    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )

    assert decoded == "hello"


def test_save_and_load_round_trip(
    tokenizer: CharacterTokenizer,
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"

    tokenizer.save(tokenizer_path)
    loaded = CharacterTokenizer.load(tokenizer_path)

    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.id_to_token == tokenizer.id_to_token
    assert loaded.encode("hello你好") == tokenizer.encode("hello你好")


def test_saved_tokenizer_has_expected_metadata(
    tokenizer: CharacterTokenizer,
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    data = json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    )

    assert data["tokenizer_type"] == "character"
    assert data["token_to_id"] == tokenizer.token_to_id


def test_empty_training_data_creates_special_tokens_only() -> None:
    tokenizer = CharacterTokenizer.train([])

    assert tokenizer.vocab_size == 5
    assert tokenizer.token_to_id == {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
    }


def test_train_rejects_non_string_text() -> None:
    with pytest.raises(TypeError, match="字符串"):
        CharacterTokenizer.train(
            [
                "valid text",
                123,
            ]
        )


def test_encode_rejects_non_string_input(
    tokenizer: CharacterTokenizer,
) -> None:
    with pytest.raises(TypeError, match="字符串"):
        tokenizer.encode(123)


@pytest.mark.parametrize(
    "token_id",
    [
        True,
        1.5,
        "1",
    ],
)
def test_decode_rejects_non_integer_ids(
    tokenizer: CharacterTokenizer,
    token_id: object,
) -> None:
    with pytest.raises(TypeError, match="整数"):
        tokenizer.decode([token_id])


def test_vocab_rejects_duplicate_ids() -> None:
    token_to_id = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
        "a": 4,
    }

    with pytest.raises(ValueError, match="重复"):
        CharacterTokenizer(token_to_id)


def test_vocab_ids_must_be_continuous() -> None:
    token_to_id = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
        "a": 6,
    }

    with pytest.raises(ValueError, match="连续"):
        CharacterTokenizer(token_to_id)


def test_special_token_ids_cannot_change() -> None:
    token_to_id = {
        "<pad>": 1,
        "<unk>": 0,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
    }

    with pytest.raises(ValueError, match="特殊 Token"):
        CharacterTokenizer(token_to_id)


def test_load_rejects_missing_file(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        CharacterTokenizer.load(tokenizer_path)


def test_load_rejects_wrong_tokenizer_type(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "tokenizer_type": "bpe",
                "token_to_id": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="字符级"):
        CharacterTokenizer.load(tokenizer_path)