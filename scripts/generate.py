import argparse
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

import torch

from gpt1.checkpoint import load_checkpoint
from gpt1.config import ModelConfig
from gpt1.generation import generate
from gpt1.model import GPTModel
from gpt1.tokenizer import CharacterTokenizer


DEFAULT_MODEL_CONFIG_PATH = Path(
    "configs/model.json"
)
DEFAULT_TOKENIZER_PATH = Path(
    "artifacts/tokenizer/tokenizer.json"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/latest.pt"
)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """读取文本生成命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用训练好的 GPT 模型生成文本"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="提供给模型的提示词",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="tokenizer 文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="最多生成多少个新 token",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="推理设备，auto 会自动选择",
    )

    return parser.parse_args(argv)


def select_device(device_name: str) -> torch.device:
    """选择文本生成使用的计算设备。"""
    if not isinstance(device_name, str):
        raise TypeError("device_name 必须是字符串")

    valid_device_names = {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }

    if device_name not in valid_device_names:
        raise ValueError(
            f"不支持的设备：{device_name}"
        )

    mps_backend = getattr(
        torch.backends,
        "mps",
        None,
    )
    mps_is_available = (
        mps_backend is not None
        and mps_backend.is_available()
    )

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if mps_is_available:
            return torch.device("mps")

        return torch.device("cpu")

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("当前环境不支持 CUDA")

    if (
        device_name == "mps"
        and not mps_is_available
    ):
        raise RuntimeError("当前环境不支持 MPS")

    return torch.device(device_name)


def load_model(
    model_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[GPTModel, Dict[str, Any]]:
    """加载模型配置和 checkpoint。"""
    model_config = ModelConfig.from_json(
        model_config_path
    )
    model = GPTModel(model_config)

    metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
        restore_rng_state=False,
    )

    model.to(device)
    model.eval()

    return model, metadata


def generate_text(
    model: GPTModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """将提示词编码，生成 token，再解码成文本。"""
    if not isinstance(prompt, str):
        raise TypeError("prompt 必须是字符串")

    prompt_token_ids = tokenizer.encode(
        prompt,
        add_bos=True,
    )

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
    )

    generated_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
    )

    return generated_text


def main(
    argv: Optional[Sequence[str]] = None,
) -> None:
    """加载所有生成组件并输出文本。"""
    args = parse_args(argv)
    device = select_device(args.device)

    tokenizer = CharacterTokenizer.load(
        args.tokenizer
    )
    model, metadata = load_model(
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            "tokenizer 词表大小与模型 vocab_size 不一致："
            f"{tokenizer.vocab_size} != "
            f"{model.config.vocab_size}"
        )

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"使用设备：{device}")
    print(
        "checkpoint step："
        f"{metadata.get('step', 0)}"
    )
    print("生成结果：")
    print(generated_text)


if __name__ == "__main__":
    main()
