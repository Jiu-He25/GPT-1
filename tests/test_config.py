import json
from pathlib import Path

import pytest

from gpt1.config import FinetuneConfig, ModelConfig, PretrainConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_load_model_config() -> None:
    config = ModelConfig.from_json(CONFIG_DIR / "model.json")

    assert config.vocab_size == 8000
    assert config.max_seq_len == 256
    assert config.hidden_size == 256
    assert config.num_heads == 4
    assert config.hidden_size % config.num_heads == 0


def test_load_pretrain_config() -> None:
    config = PretrainConfig.from_json(CONFIG_DIR / "pretrain.json")

    assert config.batch_size == 8
    assert config.gradient_accumulation_steps == 4
    assert config.learning_rate == pytest.approx(0.00025)
    assert config.precision == "fp32"


def test_load_finetune_config() -> None:
    config = FinetuneConfig.from_json(CONFIG_DIR / "finetune.json")

    assert config.task_type == "classification"
    assert config.num_labels == 2
    assert config.learning_rate == pytest.approx(0.0000625)
    assert config.precision == "fp32"


def test_model_config_save_and_load_round_trip(
    tmp_path: Path,
) -> None:
    original = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        num_layers=2,
        hidden_size=128,
        num_heads=4,
        ffn_size=512,
        dropout=0.2,
    )

    config_path = tmp_path / "model.json"
    original.save_json(config_path)

    loaded = ModelConfig.from_json(config_path)

    assert loaded == original


def test_hidden_size_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        ModelConfig(
            hidden_size=250,
            num_heads=4,
        )


@pytest.mark.parametrize(
    "config_class",
    [
        PretrainConfig,
        FinetuneConfig,
    ],
)
def test_invalid_precision_is_rejected(config_class) -> None:
    with pytest.raises(ValueError, match="precision"):
        config_class(precision="int8")


@pytest.mark.parametrize(
    ("config_class", "field_name"),
    [
        (ModelConfig, "vocab_size"),
        (PretrainConfig, "batch_size"),
        (FinetuneConfig, "num_labels"),
    ],
)
def test_positive_integer_fields_reject_booleans(
    config_class,
    field_name: str,
) -> None:
    with pytest.raises(ValueError):
        config_class(**{field_name: True})


@pytest.mark.parametrize(
    ("config_class", "field_name"),
    [
        (ModelConfig, "dropout"),
        (FinetuneConfig, "classification_dropout"),
    ],
)
def test_dropout_accepts_zero(
    config_class,
    field_name: str,
) -> None:
    config_class(**{field_name: 0})


@pytest.mark.parametrize(
    ("config_class", "field_name"),
    [
        (ModelConfig, "dropout"),
        (FinetuneConfig, "classification_dropout"),
    ],
)
def test_dropout_rejects_one(
    config_class,
    field_name: str,
) -> None:
    with pytest.raises(ValueError):
        config_class(**{field_name: 1.0})


def test_missing_config_file_is_rejected(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        ModelConfig.from_json(missing_path)


def test_unknown_json_field_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "unknown-field.json"
    config_path.write_text(
        json.dumps({"unknown_field": 123}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="参数不匹配"):
        ModelConfig.from_json(config_path)


def test_json_root_must_be_an_object(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "list.json"
    config_path.write_text(
        json.dumps([1, 2, 3]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSON对象"):
        ModelConfig.from_json(config_path)