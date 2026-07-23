import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer


PathLike = Union[str, Path]


def _validate_non_negative_integer(
    name: str,
    value: int,
) -> None:
    """检查参数是不是非负整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")

    if value < 0:
        raise ValueError(f"{name} 不能小于 0")


def save_checkpoint(
    path: PathLike,
    *,
    model: nn.Module,
    step: int,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    config: Optional[Mapping[str, Any]] = None,
    tokenizer_identifier: Optional[str] = None,
) -> None:
    """保存模型及训练状态。"""
    _validate_non_negative_integer("step", step)
    _validate_non_negative_integer("epoch", epoch)

    if config is not None and not isinstance(
        config,
        Mapping,
    ):
        raise TypeError("config 必须是映射类型")

    if (
        tokenizer_identifier is not None
        and not isinstance(tokenizer_identifier, str)
    ):
        raise TypeError(
            "tokenizer_identifier 必须是字符串或 None"
        )

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "step": step,
        "epoch": epoch,
        "config": (
            dict(config)
            if config is not None
            else None
        ),
        "tokenizer_identifier": tokenizer_identifier,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: PathLike,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Any = "cpu",
    restore_rng_state: bool = True,
) -> dict[str, Any]:
    """加载模型及训练状态，并返回训练元数据。"""
    if not isinstance(restore_rng_state, bool):
        raise TypeError(
            "restore_rng_state 必须是布尔值"
        )

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 文件不存在: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "checkpoint 内容必须是字典"
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "checkpoint 缺少 model_state_dict"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )
    if (
        optimizer is not None
        and optimizer_state is not None
    ):
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )
    if (
        scheduler is not None
        and scheduler_state is not None
    ):
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )
    if (
        scaler is not None
        and scaler_state is not None
    ):
        scaler.load_state_dict(scaler_state)

    if restore_rng_state:
        python_rng_state = checkpoint.get(
            "python_rng_state"
        )
        if python_rng_state is not None:
            random.setstate(python_rng_state)

        torch_rng_state = checkpoint.get(
            "torch_rng_state"
        )
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)

        cuda_rng_state = checkpoint.get(
            "cuda_rng_state"
        )
        if (
            cuda_rng_state is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                cuda_rng_state
            )

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "config": checkpoint.get("config"),
        "tokenizer_identifier": checkpoint.get(
            "tokenizer_identifier"
        ),
    }
