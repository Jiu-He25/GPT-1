import math
import random
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from gpt1.objectives import causal_language_model_loss


LossFunction = Callable[[Tensor, Tensor], Tensor]
Device = Union[str, torch.device]


def _validate_positive_number(
    name: str,
    value: float,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(f"{name} 必须是数值")

    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _validate_non_negative_number(
    name: str,
    value: float,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(f"{name} 必须是数值")

    if value < 0:
        raise ValueError(f"{name} 不能小于 0")


def set_seed(seed: int) -> None:
    """固定 Python 和 PyTorch 的随机种子。"""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """为所有可训练的数据创建 AdamW optimizer"""
    _validate_positive_number(
        "learning_rate",
        learning_rate,
    )
    _validate_non_negative_number(
        "weight_decay",
        weight_decay,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("模型中没有可训练参数")

    return torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay)
    )

def build_scheduler(
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
) -> LambdaLR:
    """创建线性 warmup 加 cosine decay 调度器"""
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
    ):
        raise TypeError("warmup_steps 必须是整数")

    if warmup_steps < 0:
        raise ValueError("warmup_steps 不能小于 0")

    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
    ):
        raise TypeError("max_steps 必须是整数")

    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")

    if warmup_steps > max_steps:
        raise ValueError(
            "warmup_steps 不能大于 max_steps"
        )

    def learning_rate_multiplier(
            current_step: int,
    ) -> float:
        if (
            warmup_steps > 0
            and current_step < warmup_steps
        ):
            return current_step / warmup_steps

        if current_step >= max_steps:
            return 0.0

        decay_step_count = max_steps - warmup_steps

        if decay_step_count == 0:
            return 0.0

        decay_progress = (
            current_step - warmup_steps
        ) / decay_step_count

        return 0.5 * (
            1.0
            + math.cos(math.pi * decay_progress)
        )
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=learning_rate_multiplier,
    )

class Trainer:
    """GPT 模型的 FP32 训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: Device = "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        loss_function: LossFunction = (
            causal_language_model_loss
        ),
    ) -> None:
        if (
            isinstance(
                gradient_accumulation_steps,
                bool,
            )
            or not isinstance(
                gradient_accumulation_steps,
                int,
            )
        ):
            raise TypeError(
                "gradient_accumulation_steps 必须是整数"
            )

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps 必须大于 0"
            )

        _validate_positive_number(
            "max_grad_norm",
            max_grad_norm,
        )

        if not callable(loss_function):
            raise TypeError(
                "loss_function 必须是可调用对象"
            )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.max_grad_norm = float(max_grad_norm)
        self.loss_function = loss_function

        self.global_step = 0
        self.micro_step = 0

        self.optimizer.zero_grad(set_to_none=True)

    def _prepare_batch(
        self,
        batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch 必须是映射类型")

        if "input_ids" not in batch:
            raise KeyError("batch 缺少 input_ids")

        if "labels" not in batch:
            raise KeyError("batch 缺少 labels")

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if not isinstance(input_ids, Tensor):
            raise TypeError(
                "batch['input_ids'] 必须是 Tensor"
            )

        if not isinstance(labels, Tensor):
            raise TypeError(
                "batch['labels'] 必须是 Tensor"
            )

        return (
            input_ids.to(self.device),
            labels.to(self.device),
        )

    def train_step(
        self,
        batch: Mapping[str, Tensor],
    ) -> float:
        """处理一个 micro batch，并更新参数"""
        self.model.train()

        input_ids, labels = self._prepare_batch(batch)

        logits = self.model(input_ids)
        loss = self.loss_function(logits, labels)

        scaled_loss = (
            loss / self.gradient_accumulation_steps
        )
        scaled_loss.backward()

        self.micro_step += 1

        should_update = (
            self.micro_step % self.gradient_accumulation_steps == 0
        )

        if should_update:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )
            self.global_step += 1

        return float(loss.detach().item())

    def evaluate(
        self,
        dataloader: Iterable[Mapping[str, Tensor]],
    ) -> float:
        """计算验证集的平均 loss"""
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        try:
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, labels = (
                        self._prepare_batch(batch)
                    )

                    logits = self.model(input_ids)

                    loss = self.loss_function(
                        logits,
                        labels,
                    )

                    total_loss += float(
                        loss.detach().item()
                    )
                    batch_count += 1
        finally:
            self.model.train(was_training)

        if batch_count == 0:
            raise ValueError("验证 dataloader 不能为空")

        return total_loss / batch_count