import pytest
import torch
from torch.nn import functional as F

from gpt1.objectives import causal_language_model_loss


def test_loss_matches_pytorch_cross_entropy() -> None:
    """测试自定义语言模型损失与 PyTorch 交叉熵结果一致。"""
    torch.manual_seed(0)

    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 4))

    actual_loss = causal_language_model_loss(
        logits,
        labels,
    )
    expected_loss = F.cross_entropy(
        logits.reshape(-1, 10),
        labels.reshape(-1),
    )

    assert torch.allclose(actual_loss, expected_loss)


def test_loss_supports_backward() -> None:
    """测试语言模型损失能够反向传播并产生有限梯度。"""
    logits = torch.randn(
        2,
        4,
        10,
        requires_grad=True,
    )
    labels = torch.randint(0, 10, (2, 4))

    loss = causal_language_model_loss(
        logits,
        labels,
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ignore_index_is_not_included_in_loss() -> None:
    """测试标签为 ignore_index 的位置不会参与损失计算。"""
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[1, -100, 3]])

    actual_loss = causal_language_model_loss(
        logits,
        labels,
    )
    expected_loss = F.cross_entropy(
        logits.reshape(-1, 5),
        labels.reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(actual_loss, expected_loss)


def test_all_ignored_labels_return_zero_loss() -> None:
    """测试所有标签都被忽略时返回可反向传播的零损失。"""
    logits = torch.randn(
        1,
        3,
        5,
        requires_grad=True,
    )
    labels = torch.full(
        (1, 3),
        fill_value=-100,
    )

    loss = causal_language_model_loss(
        logits,
        labels,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert logits.grad is not None
    assert torch.equal(
        logits.grad,
        torch.zeros_like(logits.grad),
    )


@pytest.mark.parametrize(
    "invalid_label",
    [
        -1,
        5,
    ],
)
def test_invalid_labels_are_rejected(
    invalid_label: int,
) -> None:
    """测试负数标签和等于词表大小的越界标签会被拒绝。"""
    logits = torch.randn(1, 1, 5)
    labels = torch.tensor([[invalid_label]])

    with pytest.raises(ValueError):
        causal_language_model_loss(
            logits,
            labels,
        )


def test_mismatched_shapes_are_rejected() -> None:
    """测试 logits 与 labels 的批次或序列形状不一致时会报错。"""
    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 3))

    with pytest.raises(ValueError):
        causal_language_model_loss(
            logits,
            labels,
        )


def test_non_floating_logits_are_rejected() -> None:
    """测试使用整数类型的 logits 时会报错。"""
    logits = torch.ones(
        2,
        4,
        10,
        dtype=torch.long,
    )
    labels = torch.ones(
        2,
        4,
        dtype=torch.long,
    )

    with pytest.raises(TypeError):
        causal_language_model_loss(
            logits,
            labels,
        )


def test_non_integer_labels_are_rejected() -> None:
    """测试使用浮点数类型的 labels 时会报错。"""
    logits = torch.randn(2, 4, 10)
    labels = torch.ones(2, 4)

    with pytest.raises(TypeError):
        causal_language_model_loss(
            logits,
            labels,
        )
