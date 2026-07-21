import argparse
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from gpt1.tokenizer import CharacterTokenizer


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/tokenizer/tokenizer.json"
)
DEFAULT_PATTERN = "*.txt"


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练字符级分词器"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="输入文本文件所在的目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出分词器文件的路径",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="输入文本文件的匹配模式",
    )
    
    return parser.parse_args(argv)


def find_text_files(input_dir: Path, pattern: str) -> List[Path]:
    """查找输入目录下的文本文件并排序"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录 {input_dir} 不存在")
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径 {input_dir} 不是一个目录")
    
    text_files = sorted(
        path
        for path in input_dir.rglob(pattern)
        if path.is_file()
    )

    if not text_files:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到匹配模式 {pattern} 的文本文件")
    
    return text_files


def load_texts(
        text_files: Iterable[Path]
) -> Iterator[str]:
    """逐个读取UTF-8编码的文本文件内容"""
    for text_file in text_files:
        try:
            yield text_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise UnicodeDecodeError(
                f"无法解码文件 {text_file}，请确保它是UTF-8编码的文本文件"
            ) from error

def main(
    argv: Optional[Sequence[str]] = None,
) -> None:
    args = parse_args(argv)

    text_files = find_text_files(args.input_dir, args.pattern)

    tokenizer = CharacterTokenizer.train(
        load_texts(text_files)
    )
    tokenizer.save(args.output)

    print("Tokenizer 训练完成")
    print(f"语料目录: {args.input_dir}")
    print(f"语料文件数: {len(text_files)}")
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"保存位置: {args.output}")
    print(
        "模型配置中的 vocab_size 应设置为: "
        f"{tokenizer.vocab_size}"
    )

if __name__ == "__main__":
    main()