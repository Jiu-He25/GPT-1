# GPT-1 from Scratch

这是一个用于**从零实现 GPT-1** 的学习项目。目标是只依赖 Python 标准库和 PyTorch，不使用 Hugging Face 的 `transformers`、`tokenizers`、`datasets` 或 `Trainer`。当前仓库只搭好了目录和空文件；除本 README 外，所有实现都留给你自己完成。

项目建议分成两条路径：

1. 先实现一个缩小版 decoder-only Transformer，把分词、预训练、保存、加载和生成完整跑通。
2. 再把配置切换到论文中的 GPT-1 规格，并按需要加入下游任务微调。

GPT-1 的重点不只是一个能续写文本的 Transformer，还包括“无监督语言模型预训练 → 有监督任务微调”这两阶段流程。

## 项目结构

```text
GPT-1/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── model.json
│   ├── pretrain.json
│   └── finetune.json
├── src/
│   └── gpt1/
│       ├── __init__.py
│       ├── config.py
│       ├── tokenizer.py
│       ├── data.py
│       ├── model.py
│       ├── objectives.py
│       ├── trainer.py
│       ├── checkpoint.py
│       └── generation.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── preprocess.py
│   ├── pretrain.py
│   ├── finetune.py
│   └── generate.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
└── artifacts/
    ├── tokenizer/
    │   └── .gitkeep
    ├── checkpoints/
    │   └── .gitkeep
    └── logs/
        └── .gitkeep
```

## 每个文件用来写什么

### 根目录与配置

| 文件 | 用途 |
| --- | --- |
| `README.md` | 项目说明、模块边界、实现顺序和运行约定。 |
| `.gitignore` | 忽略原始语料、预处理数据、checkpoint、日志、虚拟环境和 Python 缓存，避免把大文件或机器本地产物提交到 Git。当前留空，由你填写。 |
| `requirements.txt` | 记录依赖及固定版本。至少需要 `torch`；如果坚持最小依赖，其余功能可用标准库完成。Windows、macOS 和 Linux 可以共用纯 Python 依赖版本；Linux 上的 CUDA wheel 应按 PyTorch 官网选择器单独安装。 |
| `configs/model.json` | 模型结构参数，例如词表大小、上下文长度、层数、隐藏维度、注意力头数、FFN 维度和 dropout。 |
| `configs/pretrain.json` | 预训练参数，例如数据路径、batch size、梯度累积、学习率、warmup、训练步数、精度、随机种子和保存间隔。 |
| `configs/finetune.json` | 下游任务微调参数，例如任务类型、标签数、分类头 dropout、辅助 LM loss 权重和训练轮数。 |

### `src/gpt1`：可复用的核心代码

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 把 `gpt1` 标记为 Python 包；以后可在这里暴露最常用的类，但一开始保持空文件即可。 |
| `config.py` | 定义配置数据类，读取 JSON，检查诸如 `hidden_size % num_heads == 0`、序列长度不超过位置嵌入上限等约束。 |
| `tokenizer.py` | 自己实现 BPE 的训练、编码、解码、词表保存和加载，以及 `<unk>`、`<pad>`、`<bos>`、`<eos>`、`<sep>` 等特殊 token。不要调用 Hugging Face tokenizer。 |
| `data.py` | 读取原始文本/已编码 token，定义自己的 `Dataset`，切成定长连续序列并生成 causal LM 的输入和目标；也负责 `collate_fn` 和创建 PyTorch `DataLoader`。 |
| `model.py` | 实现 token/位置嵌入、masked multi-head self-attention、MLP、GELU、LayerNorm、残差连接、Transformer block、GPT 主体、LM head 和下游分类 head。 |
| `objectives.py` | 写 causal language-model loss、下游分类 loss、二者联合 loss，以及 GPT-1 微调所需的结构化输入变换。 |
| `trainer.py` | 通用训练循环：前向、反向、梯度累积、梯度裁剪、混合精度、优化器、学习率调度、验证和日志。这里不依赖 Hugging Face Trainer。 |
| `checkpoint.py` | 保存/恢复模型、优化器、调度器、AMP scaler、当前 step/epoch、配置和随机数状态，使训练可断点续跑，并能跨 Windows、macOS 与 Linux 加载。 |
| `generation.py` | 自回归生成，处理 temperature、top-k、top-p、最大长度、EOS 停止；先实现无 KV cache 的正确版本。 |

### `scripts`：命令行入口

| 文件 | 用途 |
| --- | --- |
| `train_tokenizer.py` | 读取 `data/raw` 中的训练语料，训练 BPE，并将词表和 merge 规则写入 `artifacts/tokenizer`。 |
| `preprocess.py` | 清洗语料、调用 tokenizer 编码、拼接 token，并把适合高效读取的结果写入 `data/processed`。 |
| `pretrain.py` | 组装配置、数据、GPT 模型、优化器和 trainer，启动 next-token prediction 预训练。 |
| `finetune.py` | 加载预训练 checkpoint，添加任务 head，按标注数据进行有监督微调。 |
| `generate.py` | 加载 tokenizer 与 checkpoint，从命令行接收 prompt 并生成文本。 |

### `tests`：先验证正确性，再烧 GPU

| 文件 | 用途 |
| --- | --- |
| `test_tokenizer.py` | 测试 encode/decode、未知字符、特殊 token、保存后再加载是否一致。 |
| `test_data.py` | 测试切片长度、输入与右移目标是否对应，以及 padding/attention mask 是否正确。 |
| `test_model.py` | 测试张量形状、参数共享、causal mask，并确认未来 token 不会影响过去位置的输出。 |
| `test_training.py` | 用极小语料做 overfit 测试，检查 loss 能下降、梯度存在、checkpoint 恢复后结果一致。 |

### 数据与产物目录

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | 原始语料和下游任务原始数据。大数据不要提交到 Git。 |
| `data/processed/` | tokenizer 编码后的 token、索引文件和 train/validation 切分。 |
| `artifacts/tokenizer/` | 词表、BPE merge 规则和 tokenizer 元数据。 |
| `artifacts/checkpoints/` | `.pt` checkpoint。 |
| `artifacts/logs/` | loss、学习率、吞吐量、显存使用等训练日志。 |
| `.gitkeep` | 空目录不会被 Git 跟踪，这些空文件只用于保留目录，不需要写任何内容。 |

## 模块之间的数据流

```text
原始文本
  → train_tokenizer.py → 词表与 BPE merges
  → preprocess.py      → 连续 token IDs
  → pretrain.py        → 预训练 checkpoint
  ├→ generate.py       → 文本续写
  └→ finetune.py       → 下游任务 checkpoint
```

建议固定以下接口，后面各模块会比较容易对接：

- tokenizer：`encode(text) -> list[int]`，`decode(ids) -> str`，并支持 `save(path)` / `load(path)`。
- dataset：每个 LM batch 至少返回 `input_ids` 和 `labels`，形状都是 `[batch, sequence]`。
- model：输入 `LongTensor[batch, sequence]`，输出 `logits[batch, sequence, vocab]`。
- LM loss：使用 `logits[:, :-1]` 预测 `input_ids[:, 1:]`，padding 位置必须忽略。
- checkpoint：至少保存模型、optimizer、scheduler、训练位置、配置和 tokenizer 标识。

## GPT-1 论文参考规格

论文中的模型是 decoder-only Transformer，使用 masked self-attention 和学习式位置嵌入。参考配置如下：

| 参数 | 论文设置 |
| --- | ---: |
| Transformer 层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| FFN 中间维度 | 3072 |
| 上下文长度 | 512 tokens |
| BPE | 40,000 merges |
| dropout | 0.1 |
| 激活函数 | GELU |
| 参数初始化 | 正态分布，标准差 0.02 |
| 优化器 | Adam |
| 峰值学习率 | 2.5e-4 |
| warmup | 前 2,000 updates 线性升温 |
| 学习率下降 | cosine decay 到 0 |

注意：论文写的是 **40,000 次 BPE merges**，不应直接理解成“最终词表恰好 40,000”。最终词表大小还取决于初始符号集合和特殊 token。论文使用 BooksCorpus 做预训练；你应该使用自己有权使用的语料，不要把来源不明的数据直接放进仓库。

原始 GPT-1 论文与官方代码：

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [OpenAI GPT-1 官方历史代码](https://github.com/openai/finetune-transformer-lm)

## 分词器选择

如果目标是尽量忠实地复现 2018 年论文，可以实现文本预分词后再做 BPE。如果主要训练中文或中英混合语料，基于空格的预分词会把中文处理得很差，建议自己实现 byte-level BPE 或直接从 Unicode 字符开始做 BPE。

byte-level BPE 更适合中文和任意字符，但它属于工程上的调整，不是对原论文分词流程的逐项复刻。无论选哪一种，都要把方案、特殊 token ID、词表和 merges 一起写入 tokenizer 元数据；训练和推理必须加载同一份文件。

## 推荐实现顺序

1. 写 `config.py` 与三个 JSON 配置，先把所有参数集中管理。
2. 写 `tokenizer.py` 和 `train_tokenizer.py`，完成编码/解码 round trip。
3. 写 `data.py` 和 `preprocess.py`，检查 next-token 标签右移是否正确。
4. 写 `model.py`，先只跑随机输入的形状测试和 causal mask 测试。
5. 写 `objectives.py`、`trainer.py` 和 `checkpoint.py`，用几 KB 文本刻意 overfit。
6. 写 `pretrain.py`，在小配置上打通完整训练与断点恢复。
7. 写 `generation.py` 和 `generate.py`，先 greedy，再加 temperature/top-k/top-p。
8. 最后写微调数据格式、分类 head 与 `finetune.py`。

不要一开始就在 4090 上跑论文配置。先用下面这种调试规格验证代码：

```json
{
  "vocab_size": 8000,
  "max_seq_len": 256,
  "num_layers": 4,
  "hidden_size": 256,
  "num_heads": 4,
  "ffn_size": 1024,
  "dropout": 0.1
}
```

小模型能稳定 overfit 一个很小的数据集、保存恢复一致、生成不报错后，再逐步放大到 12 层。单张 4090 运行 GPT-1 规模的模型本身并不夸张，真正消耗时间的是语料规模、训练 token 数和反复实验。

## 需要自己写 DataLoader 吗

需要写**数据加载部分**，但不需要重新实现 PyTorch 的 `DataLoader` 类。直接使用 `torch.utils.data.DataLoader`，你负责实现它所需要的数据规则即可。

在 `src/gpt1/data.py` 中建议实现：

1. `LanguageModelDataset(torch.utils.data.Dataset)`：保存连续 token IDs，并按位置返回长度为 `T` 的 `input_ids` 和右移一位的 `labels`。
2. `FineTuneDataset`：把下游任务样本转换成 GPT-1 所需的 token 序列、标签和 attention mask。
3. `collate_fn`：预训练样本都是固定长度时可以不用自定义；微调样本长度不同时，用它完成 padding 和 mask。
4. `build_dataloader(...)`：集中设置 batch size、shuffle、`num_workers`、`pin_memory`、sampler 和随机种子。

第一版预训练 Dataset 的核心关系应当是：

```text
tokens = [t0, t1, t2, t3, t4, ...]
input  = [t0, t1, t2, t3]
label  = [t1, t2, t3, t4]
```

单卡训练不需要自己写分布式 sampler。Windows 和 macOS 的多进程 DataLoader 通常使用 `spawn`，所以创建 DataLoader 和启动训练的入口要放在 `if __name__ == "__main__":` 保护之下。调试阶段先使用 `num_workers=0`，逻辑正确后再逐渐增加。

## 大概需要写多少行代码

下面按“不计算空行和大段注释、代码清晰但不过度封装”估算：

| 模块 | 预计代码量 |
| --- | ---: |
| 配置读取与校验 | 60～100 行 |
| 自己实现 BPE tokenizer | 250～450 行 |
| Dataset、DataLoader 与预处理 | 150～250 行 |
| GPT-1 模型 | 250～400 行 |
| 预训练/微调目标 | 80～150 行 |
| trainer、优化器与调度器 | 250～450 行 |
| checkpoint | 80～140 行 |
| 文本生成 | 100～180 行 |
| 五个命令行脚本合计 | 250～450 行 |
| 测试合计 | 250～450 行 |

一个能完成分词、预训练、保存恢复和生成的最小版本，大约需要 **1,200～1,800 行**。把微调、健壮的 checkpoint、跨平台处理、混合精度和完整测试都做好，整个项目大约是 **2,000～3,000 行**。如果加入高性能数据格式、KV cache、分布式训练、详细日志和更多下游任务，可能增长到 4,000 行以上。

建议把 **2,000 行左右** 当作第一阶段目标。代码量不是最终指标；一个 300 行但 causal mask 或标签右移错误的模型，比一个简单但经过测试的实现更难排查。

## Windows/macOS 开发、Linux 训练约定

- 代码和配置使用相对路径，不要写 `D:\\...` 或 `/home/...` 这样的机器专属绝对路径。
- Python 读写文本时显式使用 UTF-8；路径使用 `pathlib.Path`。
- 在 Windows 或 macOS 上运行单元测试和 tiny overfit，在 Linux 4090 上进行正式预训练。
- checkpoint 使用 `state_dict` 加配置保存，不要直接 pickle 整个模型对象。
- 加载 checkpoint 时显式指定 `map_location`，这样 CUDA checkpoint 也能在 Windows/macOS CPU 或 MPS 上检查。
- 在 Linux 上按 [PyTorch 官方安装选择器](https://docs.pytorch.org/get-started/locally/) 选择匹配驱动的 CUDA wheel；安装后先确认 `torch.cuda.is_available()`。
- macOS 同样从 [PyTorch 官方安装页](https://docs.pytorch.org/get-started/locally/) 获取安装方式，不要安装 CUDA wheel。Apple Silicon 可以在 `torch.backends.mps.is_available()` 为真时使用 [MPS 后端](https://docs.pytorch.org/docs/stable/notes/mps.html) 做小规模测试；Intel Mac 使用 CPU。
- 设备选择建议统一为 `cuda → mps → cpu`，不要在模型、Dataset 或 checkpoint 中写死 `.cuda()`；统一使用 `.to(device)`。
- 不要假定所有 CUDA 优化在 MPS 上都可用。AMP、fused optimizer、`pin_memory` 和异步拷贝应按设备分别启用，并保留 CPU 回退路径。
- 4090 上可以考虑 PyTorch AMP、BF16/FP16、梯度累积和 fused AdamW，但先保证 FP32 小测试正确。
- 固定 Python/PyTorch 版本和随机种子；Windows、macOS 与 Linux 的浮点结果不保证逐位完全一致。

实现完成后，可以从仓库根目录设置源码路径：

```powershell
# Windows PowerShell
$env:PYTHONPATH = "src"
python scripts/pretrain.py --config configs/pretrain.json
```

```bash
# macOS 或 Linux
export PYTHONPATH=src
python scripts/pretrain.py --config configs/pretrain.json
```

以上命令只是预定的命令行接口；当前脚本是空文件，需要你实现参数解析后才能运行。

## 第一轮完成标准

- 不导入任何 Hugging Face 包。
- tokenizer 保存再加载后，编码结果完全一致。
- 模型输出形状正确，并通过 causal leakage 测试。
- 在极小语料上能明显 overfit。
- 中断后恢复训练，global step、优化器和学习率连续。
- 同一个 checkpoint 能在 Windows/macOS CPU 上加载，也能在 Linux CUDA 上继续训练；Apple Silicon 还可以额外验证 MPS 加载。
- 固定随机种子时，生成脚本可以复现实验结果。
