import json
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any, Dict, Union


@dataclass
class ModelConfig:
    """保存GPT模型的结构参数"""

    vocab_size: int = 8000
    max_seq_len: int = 256
    num_layers: int = 4
    hidden_size: int = 256
    num_heads: int = 4
    ffn_size: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self)->None:
        """在初始化后验证参数的有效性"""
        self.validate()
    
    def validate(self)->None:
        """验证模型参数的有效性"""
        positive_integer_fields = (
            "vocab_size",
            "max_seq_len",
            "num_layers",
            "hidden_size",
            "num_heads",
            "ffn_size",
        )
        for field in positive_integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field}必须是正整数, 当前为 {value}")
            
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size必须能被num_heads整除, 当前为 hidden_size={self.hidden_size}, num_heads={self.num_heads}")
        
        if not isinstance(self.dropout, float) or not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout必须是介于0和1之间的浮点数, 当前为 {self.dropout}")
        
        if not isinstance(self.layer_norm_epsilon, float) or self.layer_norm_epsilon <= 0:
            raise ValueError(f"layer_norm_epsilon必须是正浮点数, 当前为 {self.layer_norm_epsilon}")
        
        if not isinstance(self.initializer_range, float) or self.initializer_range <= 0:
            raise ValueError(f"initializer_range必须是正浮点数, 当前为 {self.initializer_range}")
        
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError(f"tie_word_embeddings必须是布尔值, 当前为 {self.tie_word_embeddings}")
        
    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
    ) -> "ModelConfig":
        """从JSON文件加载模型配置"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件未找到: {path}")
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"配置文件必须包含一个JSON对象, 当前为 {type(data)}")
        
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"配置文件中的参数不匹配: {error}") from error
    
    def to_dict(self) -> Dict[str, Any]:
        """将模型配置转换为字典"""
        return asdict(self)
    
    def save_json(self, path: Union[str, Path]) -> None:
        """将模型配置保存为JSON文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=2,
                ensure_ascii=False
            )

@dataclass
class PretrainConfig:
    """保存GPT模型的预训练参数"""

    model_config_path: str = "configs/model.json"
    train_data_path: str = "data/processed/train.bin"
    validation_data_path: str = "data/processed/validation.bin"
    output_dir: str = "artifacts/pretrain"

    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    learning_rate: float = 0.00025
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    max_grad_norm: float = 1.0

    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    num_workers: int = 0
    seed: int = 42
    precision: str = "fp32"

    def __post_init__(self) -> None:
        """在初始化后验证参数的有效性"""
        self.validate()

    def validate(self) -> None:
        """验证预训练参数的有效性"""
        positive_integer_fields = (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "log_interval",
            "eval_interval",
            "save_interval",
        )
        for field in positive_integer_fields:
            value = getattr(self, field)

            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field}必须是正整数, 当前为 {value}")
            
        non_negative_float_fields = (
            "warmup_steps",
            "num_workers",
        )

        for field in non_negative_float_fields:
            value = getattr(self, field)

            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field}必须是非负整数, 当前为 {value}")
            
        if type(self.seed) is not int:
            raise ValueError(f"seed必须是整数, 当前为 {self.seed}")
        
        if self.warmup_steps > self.max_steps:
            raise ValueError(f"warmup_steps不能大于max_steps, 当前为 warmup_steps={self.warmup_steps}, max_steps={self.max_steps}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须是正浮点数, 当前为 {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay必须是非负浮点数, 当前为 {self.weight_decay}")
        
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm必须是正浮点数, 当前为 {self.max_grad_norm}")
        
        valid_precisions = ("fp32", "fp16", "bf16")

        if self.precision not in valid_precisions:
            raise ValueError(f"precision必须是以下之一: {valid_precisions}, 当前为 {self.precision}")
    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
    ) -> "PretrainConfig":
        """从JSON文件加载预训练配置"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件未找到: {path}")
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"配置文件必须包含一个JSON对象, 当前为 {type(data)}")
        
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"配置文件中的参数不匹配: {error}") from error
