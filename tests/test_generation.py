from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from torch import Tensor, nn

from gpt1.config import ModelConfig
from gpt1.generation import generate
from gpt1.model import GPTModel


class ScriptedModel(nn.Module):
    """按照预先给定的 token 顺序返回可控 logits 的测试模型。"""

    def __init__(
        self,
        scripted_tokens: List[List[int]],
        vocab_size: int = 10,
        max_seq_len: int = 4,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.scripted_tokens = scripted_tokens
        self.call_count = 0
        self.seen_inputs: List[Tensor] = []
        self.grad_enabled_states: List[bool] = []
        self.training_states: List[bool] = []

    def forward(self, input_ids: Tensor) -> Tensor:
        """记录模型调用状态，并让指定 token 获得最高分。"""
        if self.call_count >= len(self.scripted_tokens):
            raise AssertionError("模型被调用的次数超过测试脚本")

        predicted_tokens = self.scripted_tokens[
            self.call_count
        ]

        if len(predicted_tokens) != input_ids.shape[0]:
            raise AssertionError("测试脚本的 batch 大小不匹配")

        self.seen_inputs.append(input_ids.detach().clone())
        self.grad_enabled_states.append(
            torch.is_grad_enabled()
        )
        self.training_states.append(self.training)

        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
        )

        for batch_index, token_id in enumerate(
            predicted_tokens
        ):
            logits[
                batch_index,
                -1,
                token_id,
            ] = 10.0

        self.call_count += 1
        return logits


def test_generate_greedily_selects_highest_logit() -> None:
    """测试 greedy decoding 每一步都会选择 logits 最大的 token。"""
    model = ScriptedModel(
        scripted_tokens=[
            [4],
            [5],
            [6],
        ]
    )
    prompt = torch.tensor([[1, 2]])

    generated = generate(
        model=model,
        input_ids=prompt,
        max_new_tokens=3,
    )

    assert generated.tolist() == [
        [1, 2, 4, 5, 6]
    ]
    assert prompt.tolist() == [[1, 2]]


def test_generate_stops_when_eos_is_generated() -> None:
    """测试模型生成 EOS 后会提前结束，不再生成多余 token。"""
    model = ScriptedModel(
        scripted_tokens=[
            [4],
            [3],
            [5],
        ]
    )

    generated = generate(
        model=model,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=5,
        eos_token_id=3,
    )

    assert generated.tolist() == [[1, 2, 4, 3]]
    assert model.call_count == 2


def test_generate_does_not_continue_after_prompt_eos() -> None:
    """测试提示词已经以 EOS 结尾时不会调用模型继续生成。"""
    model = ScriptedModel(
        scripted_tokens=[[4]],
    )
    prompt = torch.tensor([[1, 3]])

    generated = generate(
        model=model,
        input_ids=prompt,
        max_new_tokens=5,
        eos_token_id=3,
    )

    assert torch.equal(generated, prompt)
    assert model.call_count == 0


def test_generate_crops_model_context() -> None:
    """测试提示词超过上下文上限后，模型只接收最近的 token。"""
    model = ScriptedModel(
        scripted_tokens=[
            [5],
            [6],
        ],
        max_seq_len=3,
    )

    generated = generate(
        model=model,
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        max_new_tokens=2,
    )

    assert generated.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert model.seen_inputs[0].tolist() == [[2, 3, 4]]
    assert model.seen_inputs[1].tolist() == [[3, 4, 5]]


def test_generate_handles_finished_items_in_batch() -> None:
    """测试批量生成时，先完成的样本保持 EOS，直到整个 batch 完成。"""
    model = ScriptedModel(
        scripted_tokens=[
            [3, 4],
            [5, 3],
        ]
    )

    generated = generate(
        model=model,
        input_ids=torch.tensor(
            [
                [1, 2],
                [5, 6],
            ]
        ),
        max_new_tokens=5,
        eos_token_id=3,
    )

    assert generated.tolist() == [
        [1, 2, 3, 3],
        [5, 6, 4, 3],
    ]
    assert model.call_count == 2


def test_generate_disables_gradients_and_restores_training_mode() -> None:
    """测试生成期间关闭梯度和训练模式，并在结束后恢复原状态。"""
    model = ScriptedModel(
        scripted_tokens=[[4]],
    )
    model.train()

    generate(
        model=model,
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=1,
    )

    assert model.grad_enabled_states == [False]
    assert model.training_states == [False]
    assert model.training is True


def test_generate_with_zero_new_tokens_returns_prompt() -> None:
    """测试生成数量为零时直接返回提示词副本且不调用模型。"""
    model = ScriptedModel(
        scripted_tokens=[[4]],
    )
    prompt = torch.tensor([[1, 2]])

    generated = generate(
        model=model,
        input_ids=prompt,
        max_new_tokens=0,
    )

    assert torch.equal(generated, prompt)
    assert generated.data_ptr() != prompt.data_ptr()
    assert model.call_count == 0


def test_generate_works_with_real_gpt_model() -> None:
    """测试生成函数能够与项目中的真实 GPTModel 接口正确连接。"""
    config = ModelConfig(
        vocab_size=20,
        max_seq_len=8,
        num_layers=1,
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    model = GPTModel(config)

    generated = generate(
        model=model,
        input_ids=torch.tensor([[2, 5, 6]]),
        max_new_tokens=2,
    )

    assert generated.shape == (1, 5)
    assert generated.dtype == torch.long


@pytest.mark.parametrize(
    ("input_ids", "max_new_tokens", "eos_token_id", "error"),
    [
        (
            torch.tensor([1, 2]),
            1,
            None,
            ValueError,
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            1,
            None,
            TypeError,
        ),
        (
            torch.tensor([[1, 2]]),
            -1,
            None,
            ValueError,
        ),
        (
            torch.tensor([[1, 2]]),
            True,
            None,
            TypeError,
        ),
        (
            torch.tensor([[1, 2]]),
            1,
            10,
            ValueError,
        ),
        (
            torch.tensor([[-1, 2]]),
            1,
            None,
            ValueError,
        ),
    ],
)
def test_generate_rejects_invalid_arguments(
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    error: type[Exception],
) -> None:
    """测试错误形状、类型、生成数量、EOS 和 token ID 会被拒绝。"""
    model = ScriptedModel(
        scripted_tokens=[[4]],
    )

    with pytest.raises(error):
        generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
