from typing import Dict, Iterable, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


TokenIds = Union[Iterable[int], Tensor]


class LanguageModelDataset(Dataset):
    """将连续 token ID 切成固定长度的语言模型样本"""

    def __init__(
        self,
        token_ids: TokenIds,
        seq_len: int,
    ) -> None:
        self._validate_seq_len(seq_len)

        self.seq_len = seq_len
        self.token_ids = self._prepare_token_ids(token_ids)

        minimum_tokens = self.seq_len + 1

        if self.token_ids.numel() < minimum_tokens:
            raise ValueError(
                f"至少需要 {minimum_tokens} 个 token. "
                f"当前只有 {self.token_ids.numel()} 个"
            )
        
    @staticmethod
    def _validate_seq_len(seq_len: int) -> None:
        """检查序列长度"""
        if isinstance(seq_len, bool) or not isinstance(seq_len, int):
            raise TypeError("seq_len 必须是整数")
        
        if seq_len <= 0:
            raise ValueError("seq_len 必须大于 0")
        
    @staticmethod
    def _prepare_token_ids(
        token_ids: TokenIds,
    ) -> Tensor:
        if isinstance(token_ids, Tensor):
            if token_ids.ndim != 1:
                raise ValueError("token_ids 必须是一维张量")
            
            if (
                token_ids.dtype == torch.bool
                or token_ids.is_floating_point()
                or token_ids.is_complex()
            ):
                raise TypeError("token_ids 必须包含整数")
            
            prepared_token_ids = (
                token_ids.detach().to(device="cpu", dtype=torch.long).clone()
            )
        else:
            try:
                token_ids_list = list(token_ids)
            except TypeError as error:
                raise TypeError(
                    "token_ids 必须是整数序列或一维张量"
                ) from error
            
            for token_id in token_ids_list:
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                ):
                    raise TypeError("token_ids 必须包含整数")
                
                if token_id < 0:
                    raise ValueError("token ID 不能为负")
                
            prepared_token_ids = torch.tensor(
                token_ids_list,
                dtype=torch.long,
            )

        if torch.any(prepared_token_ids < 0).item():
            raise ValueError("token ID 不能小于0")
        
        return prepared_token_ids
    
    def __len__(self) -> int:
        """返回能够切出的完整样本数量"""
        available_predictions = self.token_ids.numel() - 1
        return available_predictions // self.seq_len
    
    def __getitem__(self, index) -> Dict[str, Tensor]:
        """根据样本编号返回输入右移一位的标签"""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("样本索引必须是整数")

        if index < 0 or index >= len(self):
            raise IndexError("样本索引超出范围")

        start = index * self.seq_len
        end = start + self.seq_len + 1

        token_window = self.token_ids[start:end]

        input_ids = token_window[:-1].clone()
        labels = token_window[1:].clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """根据 Dataset 创建 PyTorch DataLoader"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size 必须是整数")
    
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于0")
    
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers 必须是整数")
    
    if num_workers < 0:
        raise ValueError("num_workers 不能小于0")
    
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed 必须是整数")
    
    boolean_fields = {
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }

    for field_name, value in boolean_fields.items():
        if not isinstance(value, bool):
            raise TypeError(f"{field_name} 必须是布尔值")
        
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )
