import json
from pathlib import Path

import pytest
import torch

from gpt1.config import ModelConfig, PretrainConfig
from gpt1.model import GPTModel
from gpt1.trainer import Trainer, build_optimizer
from scripts.pretrain import (
    load_token_ids,
    main,
    parse_args,
    run_pretraining,
    save_training_checkpoint,
    select_device,
)


def test_parse_args_uses_expected_defaults() -> None:
    """测试未传命令行参数时会使用默认配置路径和自动设备。"""
    args = parse_args([])

    assert args.config == Path("configs/pretrain.json")
    assert args.device == "auto"
    assert args.resume is None


def test_parse_args_accepts_custom_values() -> None:
    """测试命令行可以指定自定义配置路径和训练设备。"""
    args = parse_args(
        [
            "--config",
            "custom/pretrain.json",
            "--device",
            "cpu",
            "--resume",
            "checkpoints/latest.pt",
        ]
    )

    assert args.config == Path("custom/pretrain.json")
    assert args.device == "cpu"
    assert args.resume == Path("checkpoints/latest.pt")


def test_select_device_accepts_cpu() -> None:
    """测试显式选择 CPU 时返回 CPU 设备。"""
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 CUDA 和 MPS 都不可用时，自动模式会选择 CPU。"""
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(
            mps_backend,
            "is_available",
            lambda: False,
        )

    assert select_device("auto") == torch.device("cpu")


def test_select_device_rejects_invalid_name() -> None:
    """测试不受支持的设备名称会被拒绝。"""
    with pytest.raises(ValueError, match="不支持"):
        select_device("tpu")


def test_select_device_rejects_non_string_name() -> None:
    """测试设备名称不是字符串时会报类型错误。"""
    with pytest.raises(TypeError, match="字符串"):
        select_device(123)


def test_unavailable_cuda_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试当前环境没有 CUDA 时不能强制选择 CUDA。"""
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")


def test_load_token_ids_returns_cpu_long_clone(
    tmp_path: Path,
) -> None:
    """测试整数 token 文件会加载为独立的 CPU long 张量。"""
    token_path = tmp_path / "tokens.bin"
    original = torch.tensor(
        [1, 2, 3, 4],
        dtype=torch.int32,
    )
    torch.save(original, token_path)

    loaded = load_token_ids(token_path)

    assert loaded.device.type == "cpu"
    assert loaded.dtype == torch.long
    assert torch.equal(
        loaded,
        torch.tensor([1, 2, 3, 4]),
    )
    assert loaded.data_ptr() != original.data_ptr()


def test_load_token_ids_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """测试 token 文件不存在时会报错。"""
    with pytest.raises(FileNotFoundError, match="不存在"):
        load_token_ids(tmp_path / "missing.bin")


def test_load_token_ids_rejects_non_tensor_file(
    tmp_path: Path,
) -> None:
    """测试文件中保存的对象不是 Tensor 时会被拒绝。"""
    token_path = tmp_path / "tokens.bin"
    torch.save([1, 2, 3], token_path)

    with pytest.raises(ValueError, match="Tensor"):
        load_token_ids(token_path)


@pytest.mark.parametrize(
    ("token_ids", "expected_exception", "message"),
    [
        (
            torch.tensor([[1, 2], [3, 4]]),
            ValueError,
            "一维",
        ),
        (
            torch.tensor([1.0, 2.0]),
            TypeError,
            "整数",
        ),
        (
            torch.tensor([], dtype=torch.long),
            ValueError,
            "不能为空",
        ),
        (
            torch.tensor([1, -1]),
            ValueError,
            "负数",
        ),
    ],
)
def test_load_token_ids_rejects_invalid_tensors(
    tmp_path: Path,
    token_ids: torch.Tensor,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """测试非一维、非整数、空张量和负数 token 会被拒绝。"""
    token_path = tmp_path / "tokens.bin"
    torch.save(token_ids, token_path)

    with pytest.raises(expected_exception, match=message):
        load_token_ids(token_path)


def test_main_loads_config_and_token_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """测试预训练入口能够联合读取模型配置、训练配置和 token 文件。"""
    model_path = tmp_path / "model.json"
    pretrain_path = tmp_path / "pretrain.json"
    train_path = tmp_path / "train.bin"
    validation_path = tmp_path / "validation.bin"

    model_config = ModelConfig(
        vocab_size=20,
        max_seq_len=8,
        num_layers=2,
        hidden_size=16,
        num_heads=4,
        ffn_size=32,
        dropout=0.0,
    )
    model_config.save_json(model_path)

    pretrain_config = PretrainConfig(
        model_config_path=str(model_path),
        train_data_path=str(train_path),
        validation_data_path=str(validation_path),
        output_dir=str(tmp_path / "checkpoints"),
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=1,
        log_interval=1,
        eval_interval=1,
        save_interval=1,
    )
    pretrain_path.write_text(
        json.dumps(pretrain_config.to_dict()),
        encoding="utf-8",
    )

    torch.save(torch.arange(17), train_path)
    torch.save(torch.arange(9), validation_path)

    main(
        [
            "--config",
            str(pretrain_path),
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    assert "预训练配置和数据加载成功" in output
    assert "训练设备：cpu" in output
    assert "训练 token 数：17" in output
    assert "验证 token 数：9" in output
    assert "最终 step=1" in output


def test_save_training_checkpoint_returns_history_path(
    tmp_path: Path,
) -> None:
    """测试保存函数返回历史 checkpoint 路径并同时写入 latest.pt。"""
    model_config = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    pretrain_config = PretrainConfig(
        output_dir=str(tmp_path),
        warmup_steps=0,
        max_steps=1,
    )
    model = GPTModel(model_config)
    optimizer = build_optimizer(
        model=model,
        learning_rate=pretrain_config.learning_rate,
        weight_decay=pretrain_config.weight_decay,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )
    trainer.global_step = 3

    checkpoint_path = save_training_checkpoint(
        output_dir=tmp_path,
        trainer=trainer,
        model_config=model_config,
        pretrain_config=pretrain_config,
        epoch=0,
    )

    expected_path = tmp_path / "step_00000003.pt"
    assert checkpoint_path == expected_path
    assert expected_path.is_file()
    assert (tmp_path / "latest.pt").is_file()


def test_run_pretraining_can_log_without_evaluating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """测试日志间隔与验证间隔不同时不会读取未定义的验证损失。"""
    model_config = ModelConfig(
        vocab_size=20,
        max_seq_len=4,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    pretrain_config = PretrainConfig(
        output_dir=str(tmp_path / "checkpoints"),
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=1,
        log_interval=1,
        eval_interval=2,
        save_interval=1,
    )

    trainer = run_pretraining(
        model_config=model_config,
        pretrain_config=pretrain_config,
        train_token_ids=torch.arange(13),
        validation_token_ids=torch.arange(9),
        device=torch.device("cpu"),
    )

    output = capsys.readouterr().out
    assert trainer.global_step == 1
    assert "step=1 loss=" in output
    assert "validation_loss" not in output
    assert (
        tmp_path / "checkpoints" / "latest.pt"
    ).is_file()


def test_run_pretraining_resumes_from_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """测试预训练能从 step 1 的 checkpoint 恢复并继续到 step 2。"""
    checkpoint_dir = tmp_path / "checkpoints"
    model_config = ModelConfig(
        vocab_size=20,
        max_seq_len=4,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    first_config = PretrainConfig(
        output_dir=str(checkpoint_dir),
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=1,
        log_interval=10,
        eval_interval=10,
        save_interval=1,
    )
    train_token_ids = torch.arange(13)
    validation_token_ids = torch.arange(9)

    first_trainer = run_pretraining(
        model_config=model_config,
        pretrain_config=first_config,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        device=torch.device("cpu"),
    )

    assert first_trainer.global_step == 1
    latest_path = checkpoint_dir / "latest.pt"
    assert latest_path.is_file()
    capsys.readouterr()

    resumed_config = PretrainConfig(
        output_dir=str(checkpoint_dir),
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=2,
        log_interval=10,
        eval_interval=10,
        save_interval=1,
    )

    resumed_trainer = run_pretraining(
        model_config=model_config,
        pretrain_config=resumed_config,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        device=torch.device("cpu"),
        resume_path=latest_path,
    )

    output = capsys.readouterr().out
    assert "从 step 1 继续" in output
    assert resumed_trainer.global_step == 2
    assert (checkpoint_dir / "step_00000002.pt").is_file()
