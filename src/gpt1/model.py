import math
from typing import Dict

import torch
from torch import Tensor, nn

from gpt1.config import ModelConfig


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class CausalSelfAttention(nn.Module):
    """带因果的多头注意力"""

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = (
            config.hidden_size // config.num_heads
        )
        self.max_seq_len = config.max_seq_len

        self.qkv_projection = nn.Linear(
            config.hidden_size
        )
