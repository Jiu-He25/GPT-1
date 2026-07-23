import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

from gpt1.config import ModelConfig, PretrainConfig
from gpt1.checkpoint import load_checkpoint, save_checkpoint
from gpt1.data import LanguageModelDataset, build_dataloader
from gpt1.model import GPTModel
from gpt1.trainer import Trainer, build_optimizer, build_scheduler, set_seed


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(
        description="预训练 GPT 语言模型"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain.json"),
        help="预训练配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="需要恢复的 checkpoint 路径"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="训练设备，auto 会自动选择",
    )

    return parser.parse_args(argv)


def select_device(device_name: str) -> torch.device:
    """选择用于训练模型的设备。"""
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


def load_token_ids(
    path: Path,
) -> Tensor:
    """从预处理文件中加载一维 token ID 张量。"""
    token_path = Path(path)

    if not token_path.is_file():
        raise FileNotFoundError(
            f"token 文件不存在：{token_path}"
        )

    token_ids = torch.load(
        token_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(token_ids, Tensor):
        raise ValueError(
            "token 文件必须保存一个 PyTorch Tensor"
        )

    if token_ids.ndim != 1:
        raise ValueError(
            "token 文件中的 Tensor 必须是一维"
        )

    if token_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "token 文件中的 Tensor 必须使用整数类型"
        )

    if token_ids.numel() == 0:
        raise ValueError("token 文件不能为空")

    if int(token_ids.min().item()) < 0:
        raise ValueError("token ID 不能为负数")

    return token_ids.to(
        dtype=torch.long,
        device="cpu",
    ).clone()

def save_training_checkpoint(
    output_dir: Path,
    trainer: Trainer,
    model_config: ModelConfig,
    pretrain_config: PretrainConfig,
    epoch: int,
) -> Path:
    """保存带步数名称的 checkpoint 和 latest.pt。"""
    step = trainer.global_step

    checkpoint_path = output_dir / f"step_{step:08d}.pt"
    latest_path = output_dir / "latest.pt"

    saved_config = {
        "model": model_config.to_dict(),
        "pretrain": pretrain_config.to_dict(),
    }

    checkpoint_arguments = {
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "scheduler": trainer.scheduler,
        "step": step,
        "epoch": epoch,
        "config": saved_config,
    }

    save_checkpoint(checkpoint_path, **checkpoint_arguments)
    save_checkpoint(latest_path, **checkpoint_arguments)

    print(f"checkpoint 已保存：{checkpoint_path}")
    return checkpoint_path

def run_pretraining(
    model_config: ModelConfig,
    pretrain_config: PretrainConfig,
    train_token_ids: Tensor,
    validation_token_ids: Tensor,
    device: torch.device,
    resume_path: Optional[Path] = None,
) -> Trainer:
    """组装训练组件并进行预训练"""
    set_seed(pretrain_config.seed)

    train_dataset = LanguageModelDataset(train_token_ids, model_config.max_seq_len)
    validation_dataset = LanguageModelDataset(token_ids=validation_token_ids, seq_len=model_config.max_seq_len)

    use_pin_memory = device.type == "cuda"

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=True,
        num_workers=pretrain_config.num_workers,
        seed=pretrain_config.seed,
        drop_last=False,
        pin_memory=use_pin_memory,
    )
    validation_dataloader = build_dataloader(
        dataset=validation_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=False,
        num_workers=pretrain_config.num_workers,
        seed=pretrain_config.seed,
        drop_last=False,
        pin_memory=use_pin_memory,
    )
    model = GPTModel(model_config)

    optimizer = build_optimizer(
        model=model,
        learning_rate=pretrain_config.learning_rate,
        weight_decay=pretrain_config.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=pretrain_config.warmup_steps,
        max_steps=pretrain_config.max_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=(
            pretrain_config.gradient_accumulation_steps
        ),
        max_grad_norm=pretrain_config.max_grad_norm,
    )
    epoch = 0

    if resume_path is not None:
        metadata = load_checkpoint(
            resume_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            map_location=device,
        )

        trainer.global_step = int(metadata["step"])
        trainer.micro_step = (trainer.global_step * trainer.gradient_accumulation_steps)
        epoch = int(metadata["epoch"])

        print(
            "已恢复 check point，"
            f"从 step {trainer.global_step} 继续"
        )
    
    output_dir = Path(pretrain_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recent_losses = []
    last_saved_step = -1

    print("开始预训练")
    print(f"训练样本数：{len(train_dataset)}")
    print(f"验证样本数：{len(validation_dataset)}")
    print(
        "有效 batch size："
        f"{pretrain_config.batch_size * pretrain_config.gradient_accumulation_steps}"
    )

    while trainer.global_step < pretrain_config.max_steps:
        for batch in train_dataloader:
            previous_global_step = trainer.global_step

            loss = trainer.train_step(batch)
            recent_losses.append(loss)

            optimizer_was_updated = (trainer.global_step > previous_global_step)

            if not optimizer_was_updated:
                continue

            step = trainer.global_step

            if step % pretrain_config.log_interval == 0:
                average_loss = sum(recent_losses) / len(recent_losses)
                learning_rate = trainer.optimizer.param_groups[0]["lr"]
                print(
                    f"step={step} "
                    f"loss={average_loss:.6f} "
                    f"lr={learning_rate:.8f}"
                )
                recent_losses.clear()
            if step % pretrain_config.eval_interval == 0:
                validation_loss = trainer.evaluate(
                    validation_dataloader
                )

                print(
                    f"step={step} "
                    f"validation_loss="
                    f"{validation_loss:.6f}"
                )

            if step % pretrain_config.save_interval == 0:
                save_training_checkpoint(
                    output_dir=output_dir,
                    trainer=trainer,
                    model_config=model_config,
                    pretrain_config=pretrain_config,
                    epoch=epoch,
                )
                last_saved_step = step

            if step >= pretrain_config.max_steps:
                break

        else:
            epoch += 1
            continue

        break
    if last_saved_step != trainer.global_step:
        save_training_checkpoint(
            output_dir=output_dir,
            trainer=trainer,
            model_config=model_config,
            pretrain_config=pretrain_config,
            epoch=epoch,
        )

    print(
        "预训练完成，"
        f"最终 step={trainer.global_step}"
    )

    return trainer

def main(
    argv: Optional[Sequence[str]] = None,
) -> None:
    """读取并检查预训练所需的配置与数据。"""
    args = parse_args(argv)

    pretrain_config = PretrainConfig.from_json(
        args.config
    )
    model_config = ModelConfig.from_json(
        pretrain_config.model_config_path
    )

    if pretrain_config.precision != "fp32":
        raise ValueError(
            "当前 Trainer 只支持 fp32 训练"
        )

    device = select_device(args.device)

    train_token_ids = load_token_ids(
        Path(pretrain_config.train_data_path)
    )
    validation_token_ids = load_token_ids(
        Path(pretrain_config.validation_data_path)
    )

    print("预训练配置和数据加载成功")
    print(f"训练设备：{device}")
    print(f"模型层数：{model_config.num_layers}")
    print(f"隐藏维度：{model_config.hidden_size}")
    print(f"训练 token 数：{train_token_ids.numel()}")
    print(
        "验证 token 数："
        f"{validation_token_ids.numel()}"
    )
    run_pretraining(
        model_config=model_config,
        pretrain_config=pretrain_config,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        device=device,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()