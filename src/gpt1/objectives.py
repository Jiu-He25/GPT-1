import torch
from torch import Tensor
from torch.nn import functional as F


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

def causal_language_model_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """计算 GPT 模型的下一个 token 预测损失"""
    if not isinstance(logits, Tensor):
        raise TypeError("logits 必须是 PyTorch Tensor")

    if not isinstance(labels, Tensor):
        raise TypeError("labels 必须是 PyTorch Tensor")

    if logits.ndim != 3:
        raise ValueError(
            "logits 必须是三维 Tensor："
            "[batch, sequence, vocab]"
        )

    if labels.ndim != 2:
        raise ValueError(
            "labels 必须是二维 Tensor："
            "[batch, sequence]"
        )
    if not logits.is_floating_point():
        raise TypeError("logits 必须使用浮点数类型")

    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("labels 必须使用整数类型")

    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "logits 的前两维必须与 labels 形状相同，"
            f"当前分别是 {tuple(logits.shape[:2])} "
            f"和 {tuple(labels.shape)}"
        )

    if logits.shape[0] == 0:
        raise ValueError("batch 不能为空")

    if logits.shape[1] == 0:
        raise ValueError("sequence 不能为空")

    if logits.shape[2] == 0:
        raise ValueError("vocab 不能为空")

    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index 必须是整数")

    labels = labels.to(
        device=logits.device,
        dtype=torch.long,
    )

    valid_positions = labels != ignore_index

    if not torch.any(valid_positions).item():
        return logits.sum() * 0.0
    
    valid_labels = labels[valid_positions]
    minimum_label = int(valid_labels.min().item())
    maximum_label = int(valid_labels.max().item())
    vocab_size = logits.shape[-1]

    if minimum_label < 0:
        raise ValueError(
            "除 ignore_index 外，label 不能为负数"
        )
    
    if maximum_label >= vocab_size:
        raise ValueError(
            f"label 必须小于 vocab_size={vocab_size}，"
            f"当前最大值是 {maximum_label}"
        )
    
    flattened_logits = logits.reshape(
        -1,
        vocab_size,
    )
    flattened_labels = labels.reshape(-1)

    return F.cross_entropy(
        flattened_logits,
        flattened_labels,
        ignore_index=ignore_index,
    )
