import random
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from gpt1.checkpoint import load_checkpoint, save_checkpoint


class StatefulStub:
    """用于模拟 AMP scaler 这类带 state_dict 的对象。"""

    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


def make_components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    torch.manual_seed(7)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )

    inputs = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=torch.float32,
    )
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return model, optimizer, scheduler


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor)
        assert torch.equal(left, right)
        return

    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
        return

    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_nested_equal(left_item, right_item)
        return

    assert left == right


def test_save_checkpoint_creates_parent_directories(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "nested" / "run" / "latest.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        step=0,
    )

    assert checkpoint_path.is_file()


def test_model_only_checkpoint_round_trip(tmp_path: Path) -> None:
    source_model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "model.pt"
    expected_state = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
    }

    save_checkpoint(
        str(checkpoint_path),
        model=source_model,
        step=3,
        epoch=1,
    )

    target_model = nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.zero_()

    metadata = load_checkpoint(
        str(checkpoint_path),
        model=target_model,
        map_location="cpu",
    )

    assert_nested_equal(target_model.state_dict(), expected_state)
    assert metadata["step"] == 3
    assert metadata["epoch"] == 1


def test_checkpoint_restores_all_training_components(
    tmp_path: Path,
) -> None:
    source_model, source_optimizer, source_scheduler = make_components()
    source_scaler = StatefulStub(17)
    checkpoint_path = tmp_path / "training.pt"

    save_checkpoint(
        checkpoint_path,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        scaler=source_scaler,
        step=12,
        epoch=4,
        config={"learning_rate": 0.01, "max_steps": 100},
        tokenizer_identifier="artifacts/tokenizer/tokenizer.json",
    )

    target_model, target_optimizer, target_scheduler = make_components()
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.add_(10.0)
    target_scaler = StatefulStub(0)

    metadata = load_checkpoint(
        checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        scaler=target_scaler,
        map_location=torch.device("cpu"),
    )

    assert_nested_equal(
        target_model.state_dict(),
        source_model.state_dict(),
    )
    assert_nested_equal(
        target_optimizer.state_dict(),
        source_optimizer.state_dict(),
    )
    assert_nested_equal(
        target_scheduler.state_dict(),
        source_scheduler.state_dict(),
    )
    assert target_scaler.value == 17
    assert metadata["step"] == 12
    assert metadata["epoch"] == 4
    assert metadata["config"] == {
        "learning_rate": 0.01,
        "max_steps": 100,
    }
    assert metadata["tokenizer_identifier"] == (
        "artifacts/tokenizer/tokenizer.json"
    )


def test_checkpoint_restores_python_and_torch_rng_state(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "rng.pt"
    random.seed(123)
    torch.manual_seed(123)

    save_checkpoint(
        checkpoint_path,
        model=model,
        step=0,
    )
    expected_python_value = random.random()
    expected_torch_value = torch.rand(4)

    random.seed(999)
    torch.manual_seed(999)
    load_checkpoint(
        checkpoint_path,
        model=model,
        restore_rng_state=True,
    )

    assert random.random() == expected_python_value
    assert torch.equal(torch.rand(4), expected_torch_value)


def test_rng_restoration_can_be_disabled(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "rng.pt"
    random.seed(123)
    torch.manual_seed(123)
    save_checkpoint(checkpoint_path, model=model, step=0)

    random.seed(321)
    torch.manual_seed(321)
    expected_python_value = random.random()
    expected_torch_value = torch.rand(4)

    random.seed(321)
    torch.manual_seed(321)
    load_checkpoint(
        checkpoint_path,
        model=model,
        restore_rng_state=False,
    )

    assert random.random() == expected_python_value
    assert torch.equal(torch.rand(4), expected_torch_value)


def test_load_checkpoint_forwards_map_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model=model, step=0)
    original_load = torch.load
    observed: dict[str, Any] = {}

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        observed["map_location"] = kwargs.get("map_location")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
    )

    assert observed["map_location"] == "cpu"


def test_load_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            tmp_path / "missing.pt",
            model=nn.Linear(2, 2),
        )


@pytest.mark.parametrize("step", [-1, True, 1.5, "1"])
def test_save_checkpoint_rejects_invalid_step(
    tmp_path: Path,
    step: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="step"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            model=nn.Linear(2, 2),
            step=step,
        )


@pytest.mark.parametrize("epoch", [-1, True, 1.5, "1"])
def test_save_checkpoint_rejects_invalid_epoch(
    tmp_path: Path,
    epoch: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="epoch"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            model=nn.Linear(2, 2),
            step=0,
            epoch=epoch,
        )


def test_save_checkpoint_rejects_non_mapping_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="config"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            model=nn.Linear(2, 2),
            step=0,
            config=["not", "a", "mapping"],
        )


def test_save_checkpoint_rejects_invalid_tokenizer_identifier(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="tokenizer_identifier"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            model=nn.Linear(2, 2),
            step=0,
            tokenizer_identifier=123,
        )
