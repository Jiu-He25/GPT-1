import pytest
import torch

from gpt1.config import ModelConfig
from gpt1.model import (
    CausalSelfAttention,
    GPTModel,
    MLP,
    TransformerBlock,
)


def make_tiny_config(**overrides: object) -> ModelConfig:
    values = {
        "vocab_size": 32,
        "max_seq_len": 8,
        "num_layers": 2,
        "hidden_size": 16,
        "num_heads": 4,
        "ffn_size": 32,
        "dropout": 0.0,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.fixture
def tiny_config() -> ModelConfig:
    return make_tiny_config()


def test_causal_self_attention_preserves_shape(
    tiny_config: ModelConfig,
) -> None:
    attention = CausalSelfAttention(tiny_config)
    hidden_states = torch.randn(
        2,
        5,
        tiny_config.hidden_size,
    )

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_mlp_preserves_shape(
    tiny_config: ModelConfig,
) -> None:
    mlp = MLP(tiny_config)
    hidden_states = torch.randn(
        2,
        5,
        tiny_config.hidden_size,
    )

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_transformer_block_preserves_shape(
    tiny_config: ModelConfig,
) -> None:
    block = TransformerBlock(tiny_config)
    hidden_states = torch.randn(
        2,
        5,
        tiny_config.hidden_size,
    )

    output = block(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_gpt_model_returns_expected_logits_shape(
    tiny_config: ModelConfig,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        5,
        tiny_config.vocab_size,
    )
    assert logits.dtype == torch.float32


def test_gpt_model_accepts_shorter_sequences(
    tiny_config: ModelConfig,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (
        1,
        3,
        tiny_config.vocab_size,
    )


@pytest.mark.parametrize(
    "input_ids",
    [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.ones((1, 2, 3), dtype=torch.long),
    ],
)
def test_gpt_model_rejects_non_matrix_input(
    tiny_config: ModelConfig,
    input_ids: torch.Tensor,
) -> None:
    model = GPTModel(tiny_config)

    with pytest.raises(ValueError, match="二维"):
        model(input_ids)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bool,
    ],
)
def test_gpt_model_rejects_non_integer_input(
    tiny_config: ModelConfig,
    dtype: torch.dtype,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.ones((1, 3), dtype=dtype)

    with pytest.raises(TypeError, match="整数"):
        model(input_ids)


@pytest.mark.parametrize(
    "invalid_token_id",
    [
        -1,
        32,
    ],
)
def test_gpt_model_rejects_out_of_range_token_ids(
    tiny_config: ModelConfig,
    invalid_token_id: int,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.tensor(
        [[1, invalid_token_id, 2]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="token ID"):
        model(input_ids)


def test_gpt_model_rejects_empty_sequence(
    tiny_config: ModelConfig,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.empty(
        (1, 0),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="不能为空"):
        model(input_ids)


def test_gpt_model_rejects_sequence_over_maximum_length(
    tiny_config: ModelConfig,
) -> None:
    model = GPTModel(tiny_config)
    input_ids = torch.ones(
        (1, tiny_config.max_seq_len + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids)


def test_future_tokens_do_not_affect_past_logits(
    tiny_config: ModelConfig,
) -> None:
    torch.manual_seed(7)
    model = GPTModel(tiny_config)
    model.eval()

    original = torch.tensor(
        [[1, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    changed_future = torch.tensor(
        [[1, 2, 3, 8, 9]],
        dtype=torch.long,
    )

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed_future)

    assert torch.allclose(
        original_logits[:, :3],
        changed_logits[:, :3],
        atol=1e-6,
        rtol=0.0,
    )


def test_lm_head_and_token_embedding_share_weights(
    tiny_config: ModelConfig,
) -> None:
    model = GPTModel(tiny_config)

    assert (
        model.lm_head.weight.data_ptr()
        == model.token_embedding.weight.data_ptr()
    )


def test_weight_tying_can_be_disabled() -> None:
    config = make_tiny_config(
        tie_word_embeddings=False,
    )
    model = GPTModel(config)

    assert (
        model.lm_head.weight.data_ptr()
        != model.token_embedding.weight.data_ptr()
    )


def test_backward_produces_embedding_gradients(
    tiny_config: ModelConfig,
) -> None:
    torch.manual_seed(11)
    model = GPTModel(tiny_config)
    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    logits = model(input_ids)
    loss = logits.square().mean()
    loss.backward()

    gradient = model.token_embedding.weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_eval_mode_is_deterministic() -> None:
    config = make_tiny_config(dropout=0.5)
    model = GPTModel(config)
    model.eval()
    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    with torch.no_grad():
        first_logits = model(input_ids)
        second_logits = model(input_ids)

    assert torch.equal(first_logits, second_logits)
