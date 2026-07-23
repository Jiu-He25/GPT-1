import math
import random
from typing import Dict

import pytest
import torch
from torch import Tensor, nn

from gpt1.config import ModelConfig
from gpt1.model import GPTModel
from gpt1.trainer import (
    Trainer,
    build_optimizer,
    build_scheduler,
    set_seed,
)


def make_tiny_model() -> GPTModel:
    config = ModelConfig(
        vocab_size=16,
        max_seq_len=4,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=True,
    )
    return GPTModel(config)


def make_batch() -> Dict[str, Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [
                [2, 3, 4, 5],
                [2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
    }


def clone_parameters(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def parameters_changed(
    before: Dict[str, Tensor],
    model: nn.Module,
) -> bool:
    return any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_set_seed_makes_python_and_torch_reproducible() -> None:
    set_seed(123)
    first_python_value = random.random()
    first_torch_value = torch.rand(4)

    set_seed(123)
    second_python_value = random.random()
    second_torch_value = torch.rand(4)

    assert first_python_value == second_python_value
    assert torch.equal(first_torch_value, second_torch_value)


@pytest.mark.parametrize("seed", [True, 1.5, "123"])
def test_set_seed_rejects_non_integer_values(seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        set_seed(seed)


def test_build_optimizer_uses_only_trainable_parameters() -> None:
    model = nn.Linear(3, 2)
    model.weight.requires_grad_(False)

    optimizer = build_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.1,
    )

    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimized_parameters == [model.bias]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("learning_rate", "weight_decay", "expected_exception"),
    [
        (0.0, 0.0, ValueError),
        (-0.1, 0.0, ValueError),
        (True, 0.0, TypeError),
        (0.01, -0.1, ValueError),
        (0.01, True, TypeError),
    ],
)
def test_build_optimizer_rejects_invalid_arguments(
    learning_rate: object,
    weight_decay: object,
    expected_exception: type[Exception],
) -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(expected_exception):
        build_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )


def test_build_optimizer_rejects_model_without_trainable_parameters() -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="可训练参数"):
        build_optimizer(
            model,
            learning_rate=0.01,
            weight_decay=0.0,
        )


def test_scheduler_performs_warmup_and_cosine_decay() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=2,
        max_steps=6,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    learning_rates = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(
            optimizer.param_groups[0]["lr"]
        )

    expected = [
        0.5,
        1.0,
        0.5 * (1.0 + math.cos(math.pi * 0.25)),
        0.5,
        0.5 * (1.0 + math.cos(math.pi * 0.75)),
        0.0,
    ]

    assert learning_rates == pytest.approx(expected)


@pytest.mark.parametrize(
    ("warmup_steps", "max_steps", "expected_exception"),
    [
        (-1, 10, ValueError),
        (True, 10, TypeError),
        (0, 0, ValueError),
        (11, 10, ValueError),
        (0, 1.5, TypeError),
    ],
)
def test_scheduler_rejects_invalid_arguments(
    warmup_steps: object,
    max_steps: object,
    expected_exception: type[Exception],
) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with pytest.raises(expected_exception):
        build_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
        )


def test_train_step_updates_parameters_and_global_step() -> None:
    set_seed(7)
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )
    before = clone_parameters(model)

    loss = trainer.train_step(make_batch())

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert parameters_changed(before, model)
    assert trainer.global_step == 1


def test_gradient_accumulation_delays_optimizer_update() -> None:
    set_seed(11)
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )
    before = clone_parameters(model)

    trainer.train_step(make_batch())

    assert not parameters_changed(before, model)
    assert trainer.global_step == 0

    trainer.train_step(make_batch())

    assert parameters_changed(before, model)
    assert trainer.global_step == 1


def test_scheduler_steps_only_when_optimizer_updates() -> None:
    class CountingScheduler:
        def __init__(self) -> None:
            self.step_count = 0

        def step(self) -> None:
            self.step_count += 1

    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0,
    )
    scheduler = CountingScheduler()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )

    trainer.train_step(make_batch())
    assert scheduler.step_count == 0

    trainer.train_step(make_batch())
    assert scheduler.step_count == 1


def test_train_step_applies_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        max_grad_norm=0.25,
    )
    original_clip = torch.nn.utils.clip_grad_norm_
    observed = {}

    def recording_clip_grad_norm_(
        parameters,
        max_norm: float,
        *args,
        **kwargs,
    ):
        observed["max_norm"] = max_norm
        return original_clip(
            parameters,
            max_norm,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        recording_clip_grad_norm_,
    )

    trainer.train_step(make_batch())

    assert observed["max_norm"] == pytest.approx(0.25)


def test_evaluate_does_not_change_parameters_or_training_mode() -> None:
    set_seed(13)
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )
    model.train()
    before = clone_parameters(model)

    loss = trainer.evaluate([make_batch(), make_batch()])

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert model.training
    assert not parameters_changed(before, model)
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_tiny_repeated_batch_can_be_overfit() -> None:
    set_seed(17)
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        max_grad_norm=1.0,
    )
    batch = make_batch()

    initial_loss = trainer.evaluate([batch])
    for _ in range(50):
        trainer.train_step(batch)
    final_loss = trainer.evaluate([batch])

    assert final_loss < initial_loss * 0.5


@pytest.mark.parametrize(
    ("trainer_kwargs", "expected_exception"),
    [
        ({"gradient_accumulation_steps": 0}, ValueError),
        ({"gradient_accumulation_steps": True}, TypeError),
        ({"max_grad_norm": 0.0}, ValueError),
        ({"max_grad_norm": True}, TypeError),
    ],
)
def test_trainer_rejects_invalid_arguments(
    trainer_kwargs: Dict[str, object],
    expected_exception: type[Exception],
) -> None:
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=0.01,
        weight_decay=0.0,
    )

    with pytest.raises(expected_exception):
        Trainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            **trainer_kwargs,
        )
