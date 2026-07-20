import json
from pathlib import Path
from typing import Dict, List, Union, Iterable


SPETIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>"
)


class CharacterTokenizer:
    def __init__(self, token_to_id:Dict[str, int])-> None:
        self.token_to_id = token_to_id
        self._validate_vocab()

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    def _validate_vocab(self) -> None:
        if not self.token_to_id:
            raise ValueError("词表不能为空。")
        for token, token_id in self.token_to_id.items():
            if not isinstance(token, str):
                raise TypeError(f"Token '{token}' 必须是字符串类型。")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"Token ID '{token_id}' 必须是整数类型。")
            if token_id < 0:
                raise ValueError(f"Token ID '{token_id}' 必须是非负整数。")
            
        token_ids = list(self.token_to_id.values())

        if len(token_ids) != len(set(token_ids)):
            raise ValueError("词表中存在重复的 Token ID。")
        
        expected_ids = set(range(len(token_ids)))

        if set(token_ids) != expected_ids:
            raise ValueError("词表中的 Token ID 必须是连续的整数，从 0 开始。")
        
        for expected_id, sepcial_token in enumerate(SPETIAL_TOKENS):
            actual_id = self.token_to_id.get(sepcial_token)

            if actual_id != expected_id:
                raise ValueError(
                    f"特殊 Token '{sepcial_token}' 的 ID 必须是 {expected_id}，"
                    f"但实际为 {actual_id}。"
                    )
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
    ) -> "CharacterTokenizer":
        
        character = set()

        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"输入文本 '{text}' 必须是字符串类型。")
            
            character.update(text)

        token_to_id = {
            token: idx
            for idx, token in enumerate(SPETIAL_TOKENS)
        }

        for character in sorted(character):
            if character not in token_to_id:
                token_to_id[character] = len(token_to_id)

        return cls(token_to_id=token_to_id)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]
    
    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]
    
    @property
    def sep_token_id(self) -> int:
        return self.token_to_id["<sep>"]
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,        
    ) -> List[int]:
        """将字符串转化为 token id 列表。"""
        if not isinstance(text, str):
            raise TypeError(f"输入文本 '{text}' 必须是字符串类型。")
        
        token_ids = []

        if add_bos:
            token_ids.append(self.bos_token_id)

        for character in text:
            token_id = self.token_to_id.get(
                character,
                self.unk_token_id,
            )
            token_ids.append(token_id)

        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """将 token id 列表转化为字符串。"""
        tokens = []

        for token_id in token_ids:
            if isinstance(token_id,bool) or not isinstance(token_id, int):
                raise TypeError(f"Token ID '{token_id}' 必须是整数类型。")
            
            token = self.id_to_token.get(token_id, "<unk>")

            if skip_special_tokens and token in SPETIAL_TOKENS:
                continue

            tokens.append(token)

        return "".join(tokens)
    
    def save(
        self,
        path: Union[str, Path],
    ) -> None:
        """将词表保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tokenizer_type": "CharacterTokenizer",
            "token_to_id": self.token_to_id,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
    ) ->"CharacterTokenizer":
        """从 JSON 文件加载词表。"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"文件 '{path}' 不存在。")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"文件 '{path}' 的内容不是有效的 JSON 对象。")
        
        if data.get("tokenizer_type") != "character":
            raise ValueError("这不是字符级分词器文件")

        token_to_id = data.get("token_to_id")

        if not isinstance(token_to_id, dict):
            raise ValueError("分词器文件中缺少 token_to_id")

        return cls(token_to_id)