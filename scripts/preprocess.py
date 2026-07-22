import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from gpt1.tokenizer import CharacterTokenizer


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_TOKENIZER_PATH = Path(
    "artifacts/tokenizer/tokenizer.json"
)
DEFAULT_TRAIN_OUTPUT = Path(
    "data/processed/train.bin"
)
DEFAULT_VALIDATION_OUTPUT = Path(
    "data/processed/validation.bin"
)
DEFAULT_PATTERN = "*.txt"
DEFAULT_VALIDATION_RATIO = 0.1

def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将原始文本预处理为连续 token IDs。"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="原始文本文件所在目录。",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="已训练 tokenizer 的文件路径。",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="训练 token IDs 的输出路径。",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT,
        help="验证 token IDs 的输出路径。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="验证集占全部 token 的比例。",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="语料文件匹配模式。",
    )

    return parser.parse_args(argv)


def find_text_files(
    input_dir: Path,
    pattern: str,
) -> List[Path]:
    """在指定目录中查找匹配模式的文本文件。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录 {input_dir} 不存在。")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径 {input_dir} 不是一个目录。")

    text_files = sorted(
        path
        for path in input_dir.rglob(pattern)
        if path.is_file()
    )

    if not text_files:
        raise FileNotFoundError(
            f"在目录 {input_dir} 中未找到匹配模式 '{pattern}' 的文本文件。"
        )

    return text_files


def encode_text_files(
    text_files: Sequence[Path],
    tokenizer: CharacterTokenizer,
) -> Tensor:
    """编码所有文本，并在每个文档末尾添加 EOS。"""
    token_ids: List[int] = []

    for text_file in text_files:
        try:
            text = text_file.read_text(
                encoding="utf-8"
            )
        except Exception as e:
            raise ValueError(
                f"无法读取文件 {text_file}。请确保文件存在且可读。"
            ) from e

        document_token_ids = tokenizer.encode(
            text,
            add_eos=True
        )
        token_ids.extend(document_token_ids)

    if not token_ids:
        raise ValueError(
            "语料编码后没有生成任何 token"
        )

    return torch.tensor(
        token_ids,
        dtype=torch.int32
    )

def split_token_ids(
    token_ids: Tensor,
    validation_ratio: float,
) -> Tuple[Tensor, Tensor]:
    """将 token IDs 拆分为训练集和验证集。"""
    if (
        isinstance(validation_ratio, bool)
        or not isinstance(
            validation_ratio,
            (int, float),
        )
    ):
        raise TypeError(
            "validation_ratio 必须是数值"
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio 必须在 0 和 1 之间"
        )

    if token_ids.ndim != 1:
        raise ValueError("token_ids 必须是一维 Tensor")

    total_token_count = token_ids.numel()

    if total_token_count < 2:
        raise ValueError(
            "token_ids 中的 token 数量必须至少为 2"
        )

    validation_token_count = max(
        1,
        int(total_token_count * validation_ratio)
    )
    train_token_count = total_token_count - validation_token_count


    train_token_ids = token_ids[:train_token_count].clone()
    validation_token_ids = token_ids[train_token_count:].clone()

    return train_token_ids, validation_token_ids


def save_token_ids(
    token_ids: Tensor,
    output_path: Path,
) -> None:
    """保存 token IDs 到指定路径。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(token_ids, output_path)


def main(
    argv: Optional[Sequence[str]] = None,
) -> None:
    args = parse_args(argv)

    if (
        args.train_output.resolve() ==
        args.validation_output.resolve()
    ):
        raise ValueError(
            "训练输出路径和验证输出路径不能相同。"
        )

    tokenizer = CharacterTokenizer.load(
        args.tokenizer_path
    )
    text_files = find_text_files(
        input_dir=args.input_dir,
        pattern=args.pattern
    )

    all_token_ids = encode_text_files(
        text_files=text_files,
        tokenizer=tokenizer
    )
    train_token_ids, validation_token_ids = split_token_ids(
        token_ids=all_token_ids,
        validation_ratio=args.validation_ratio
    )

    save_token_ids(
        token_ids=train_token_ids,
        output_path=args.train_output
    )
    save_token_ids(
        token_ids=validation_token_ids,
        output_path=args.validation_output
    )

    unknown_token_count = int(
        (all_token_ids == tokenizer.unk_token_id).sum().item()
    )

    print("语料预处理完成")
    print(f"语料文件数: {len(text_files)}")
    print(f"Tokenizer 词表大小: {tokenizer.vocab_size}")
    print(f"全部 token 数: {all_token_ids.numel()}")
    print(f"训练 token 数: {train_token_ids.numel()}")
    print(
        "验证 token 数: "
        f"{validation_token_ids.numel()}"
    )
    print(f"未知 token 数: {unknown_token_count}")
    print(f"训练集保存位置: {args.train_output}")
    print(
        "验证集保存位置: "
        f"{args.validation_output}"
    )

if __name__ == "__main__":
    main()
