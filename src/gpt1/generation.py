from typing import Optional

import torch
from torch import Tensor, nn

from gpt1.model import GPTModel

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

def generate(
    model: GPTModel,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    """使用 greedy decoding 自回归生成 token"""
    if not isinstance(model, nn.Module):
        raise TypeError("model 必须是 PyTorch 模型")

    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids 必须是 PyTorch Tensor")

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids 必须是二维 Tensor："
            "[batch, sequence]"
        )

    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("input_ids 必须使用整数类型")

    if input_ids.shape[0] == 0:
        raise ValueError("input_ids 的 batch 不能为空")

    if input_ids.shape[1] == 0:
        raise ValueError("input_ids 的序列不能为空")

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
    ):
        raise TypeError("max_new_tokens 必须是整数")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能小于 0")
    if eos_token_id is not None:
        if (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
        ):
            raise TypeError(
                "eos_token_id 必须是整数或 None"
            )

        if not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError(
                "eos_token_id 必须在模型词表范围内"
            )
    minimum_token_id = int(input_ids.min().item())
    maximum_token_id = int(input_ids.max().item())

    if minimum_token_id < 0:
        raise ValueError("input_ids 中不能包含负数")

    if maximum_token_id >= model.config.vocab_size:
        raise ValueError(
            "input_ids 中的 token ID 超出模型词表范围"
        )

    first_parameter = next(
        model.parameters(),
        None,
    )

    if first_parameter is None:
        raise ValueError("model 中没有参数")

    model_device = first_parameter.device
    generated_ids = input_ids.to(
        device=model_device,
        dtype=torch.long,
    ).clone()

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            if eos_token_id is None:
                finished = torch.zeros(
                    generated_ids.shape[0],
                    dtype=torch.bool,
                    device=model_device,
                )
            else:
                finished = (
                    generated_ids[:, -1] == eos_token_id
                )

            for _ in range(max_new_tokens):
                if torch.all(finished).item():
                    break

                model_input = generated_ids[:, -model.config.max_seq_len:,]

                logits = model(model_input)

                next_token_ids = torch.argmax(
                    logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

                if eos_token_id is not None:
                    eos_values = torch.full_like(
                        next_token_ids,
                        fill_value=eos_token_id,
                    )

                    next_token_ids = torch.where(
                        finished.unsqueeze(-1),
                        eos_values,
                        next_token_ids,
                    )

                generated_ids = torch.cat(
                    [
                        generated_ids,
                        next_token_ids,
                    ],
                    dim=1,
                )

                if eos_token_id is not None:
                    finished = (
                        finished
                        | next_token_ids.squeeze(-1).eq(eos_token_id)
                    )
    finally:
        model.train(was_training)

    return generated_ids
