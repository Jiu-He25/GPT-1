import math

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
            config.hidden_size,
            3 * config.hidden_size,
        )
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.attention_dropout = nn.Dropout(
            config.dropout
        )
        self.output_dropout = nn.Dropout(
            config.dropout
        )

        causal_mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool,
            )
        )
        causal_mask = causal_mask.view(
            1,
            1,
            config.max_seq_len,
            config.max_seq_len,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states必须是3维 Tensor"
            )
        
        batch_size, seq_len, hidden_size = (
            hidden_states.shape
        )

        if hidden_size != self.hidden_size:
            raise ValueError(
                "hidden_states 最后一维必须等于 "
                f"hidden_size={self.hidden_size}"
            )
        
        if seq_len == 0:
            raise ValueError("注意力序列不能为空")
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"序列长度{seq_len}超过 "
                f"max_seq_len={self.max_seq_len}"
            )
        

        query_key_value = self.qkv_projection(
            hidden_states
        )

        query, key, value = query_key_value.chunk(
            3,
            dim=-1,
        )

        query = query.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        value = value.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        mask = self.causal_mask[
            :,
            :,
            :seq_len,
            :seq_len,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        context = torch.matmul(
            attention_weights,
            value,
        )
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.hidden_size,
            )
        )

        output = self.output_projection(context)
        output = self.output_dropout(output)

        return output
    
class MLP(nn.Module):
    """Transformer Block 中的前馈网络"""

    def __init__(
        self,
        config: ModelConfig
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            config.hidden_size,
            config.ffn_size,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(
            config.ffn_size,
            config.hidden_size,
        )
        self.dropout = nn.Dropout(config.dropout)
    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        hidden_states = self.input_projection(
            hidden_states
        )
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output_projection(
            hidden_states
        )
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
class TransformerBlock(nn.Module):
    """一个带残差连接的 Transformer Block"""
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attention = CausalSelfAttention(config)

        self.mlp_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.mlp = MLP(config)
    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        attention_input = self.attention_layer_norm(
            hidden_states
        )
        hidden_states = hidden_states + self.attention(
            attention_input
        )

        mlp_input = self.mlp_layer_norm(
            hidden_states
        )
        hidden_states = hidden_states + self.mlp(
            mlp_input
        )

        return hidden_states
    

class GPTModel(nn.Module):
    """Decoder-only GPT 语言模型"""

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.config = config
        
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.hidden_size,
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_module)

        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

    def _initialize_module(
        self,
        module: nn.Module,
    ) -> None:
        """按照 GPT 风格初始化模型参数"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_input_ids(
        self,
        input_ids: Tensor,
    ) -> None:
        if not isinstance(input_ids, Tensor):
            raise TypeError(
                "input_ids 必须是 PyTorch Tensor"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是二维 Tensor："
                "[batch, sequence]"
            )

        if input_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "input_ids 必须使用整数类型"
            )

        batch_size, seq_len = input_ids.shape

        if batch_size == 0:
            raise ValueError("input_ids batch 不能为空")

        if seq_len == 0:
            raise ValueError("input_ids 序列不能为空")

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"序列长度 {seq_len} 超过 "
                f"max_seq_len={self.config.max_seq_len}"
            )

        minimum_token_id = int(
            input_ids.min().item()
        )
        maximum_token_id = int(
            input_ids.max().item()
        )

        if minimum_token_id < 0:
            raise ValueError(
                f"token ID 不能为负数，"
                f"当前最小值为 {minimum_token_id}"
            )

        if maximum_token_id >= self.config.vocab_size:
            raise ValueError(
                f"token ID 必须小于 vocab_size="
                f"{self.config.vocab_size}，"
                f"当前最大值为 {maximum_token_id}"
            )
        
    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        self._validate_input_ids(input_ids)

        input_ids = input_ids.to(dtype=torch.long)
        _, seq_len = input_ids.shape

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(
            input_ids
        )
        position_embeddings = self.position_embedding(
            position_ids
        )
        hidden_states = (
            token_embeddings + position_embeddings
        )
        hidden_states = self.embedding_dropout(
            hidden_states
        )

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_layer_norm(
            hidden_states
        )
        logits = self.lm_head(hidden_states)

        return logits
