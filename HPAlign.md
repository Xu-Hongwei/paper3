# HPAlign

HPAlign 是一个面向**遥感图像-文本检索（Remote Sensing Image-Text Retrieval, RSITR）**的实验工程。

当前阶段首先实现一个干净、可复现的 **CLIP Fine-tuning Baseline**，在 RSICD、RSITMD 等遥感图文检索数据集上进行训练和评测，为后续细粒度对齐方法提供统一基线。

目前 Baseline 的核心流程为：

```text
Remote Sensing Image
        │
        ▼
 CLIP Vision Encoder
        │
  Global Image Feature
        │
        ├───────────────┐
        │               │
        │     Cosine Similarity
        │               │
        └───────────────┤
                        │
  Global Text Feature   │
        ▲               │
        │               │
 CLIP Text Encoder      │
        ▲               │
        │               │
      Caption           │
                        ▼
             Bidirectional InfoNCE
                        │
                        ▼
                Image-Text Retrieval
```

评测采用遥感图文检索常用指标：

- Image-to-Text：R@1 / R@5 / R@10
- Text-to-Image：R@1 / R@5 / R@10
- Mean Recall（mR）

当前工程重点是先保证 **Vanilla CLIP Full Fine-tuning Baseline** 的训练、测试和评测流程稳定可靠。

后续将在相同的 CLIP 主干、数据划分和评测协议上逐步扩展实体级、原型级和细粒度图文对齐模块。

---

# 项目结构

```text
HPAlign/
│
├── train.py
├── test.py
├── requirements.txt
│
├── configs/
│   └── baseline/
│       ├── rsicd.yaml
│       └── rsitmd.yaml
│
├── data/
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── rsicd.py
│   ├── rsitmd.py
│   └── transforms.py
│
├── models/
│   ├── __init__.py
│   └── clip/
│       ├── __init__.py
│       ├── backbone.py
│       └── clip_model.py
│
├── losses/
│   ├── __init__.py
│   └── clip_loss.py
│
├── engine/
│   ├── trainer.py
│   └── evaluator.py
│
├── evaluation/
│   ├── retrieval.py
│   └── metrics.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── seed.py
│   └── checkpoint.py
│
├── scripts/
│   ├── train_clip_rsicd.sh
│   └── train_clip_rsitmd.sh
│
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── results/
```

---

# 目录说明

## `train.py`

项目统一训练入口。

主要负责：

- 读取实验配置；
- 构建 Dataset 和 DataLoader；
- 初始化 CLIP 模型；
- 初始化损失函数；
- 构建 optimizer 和 scheduler；
- 调用 `engine/trainer.py` 完成训练；
- 保存模型 checkpoint 和训练日志。

原则上，具体训练逻辑不直接堆积在 `train.py` 中，主文件仅负责实验流程的组织。

---

## `test.py`

模型测试与最终检索评测入口。

主要负责：

- 加载训练完成的 checkpoint；
- 提取测试集图像特征和文本特征；
- 构造图文相似度矩阵；
- 执行 Image-to-Text 和 Text-to-Image 检索；
- 输出 R@1、R@5、R@10 和 mR。

---

# `configs/`

存放实验配置文件。

```text
configs/
└── baseline/
    ├── rsicd.yaml
    └── rsitmd.yaml
```

不同数据集和实验设置分别使用独立配置文件。

配置内容主要包括：

```text
Dataset
Backbone
Pretrained Weight
Batch Size
Epoch
Learning Rate
Optimizer
Scheduler
Random Seed
Checkpoint Path
Evaluation Settings
```

例如：

```text
configs/baseline/rsicd.yaml
```

用于定义 RSICD 上的纯 CLIP baseline 实验。

后续加入完整模型后，可继续扩展：

```text
configs/
├── baseline/
└── hpalign/
```

从而保证 baseline 与完整方法使用一致的数据和训练协议。

---

# `data/`

存放数据集及数据划分文件。

例如：

```text
data/
├── rsicd/
│   ├── images/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
└── rsitmd/
    ├── images/
    ├── train.json
    ├── val.json
    └── test.json
```

该目录只负责保存原始数据及预处理后的数据索引，不包含模型代码。

---

# `datasets/`

负责所有与数据读取相关的逻辑。

## `base_dataset.py`

定义不同遥感图文数据集共享的基础 Dataset 接口。

例如统一返回：

```python
{
    "image": image,
    "caption": caption,
    "image_id": image_id,
    "caption_id": caption_id
}
```

统一 Dataset 接口可以避免不同数据集使用不同训练逻辑。

---

## `rsicd.py`

RSICD 数据集读取与组织。

主要负责：

- 加载 RSICD 图像；
- 读取对应 captions；
- 管理 image-caption 对应关系；
- 生成训练、验证和测试样本。

---

## `rsitmd.py`

RSITMD 数据集读取与组织。

整体接口尽量与 `rsicd.py` 保持一致，从而保证模型和 Trainer 无需针对具体数据集修改。

---

## `transforms.py`

统一管理图像数据增强和预处理。

例如：

```text
Resize
Random Crop
Normalization
CLIP Image Transform
```

训练集和测试集可以使用不同的数据增强策略。

---

# `models/`

存放模型结构。

当前阶段只有 CLIP baseline：

```text
models/
└── clip/
```

后续所有新方法都应建立在统一模型接口上。

---

## `models/clip/backbone.py`

负责最基础的 CLIP Encoder。

主要功能：

```text
Image
  ↓
CLIP Vision Encoder
  ↓
Image Embedding
```

以及：

```text
Caption
  ↓
CLIP Text Encoder
  ↓
Text Embedding
```

该文件只负责 CLIP 特征提取，不负责：

- loss；
- evaluation；
- Dataset；
- prototype；
- OT。

后续完整模型如果需要 patch/token feature，也优先在这里扩展统一的 CLIP 特征接口。

---

## `models/clip/clip_model.py`

定义完整的 CLIP Retrieval Baseline。

负责组合：

```text
CLIP Vision Encoder
+
CLIP Text Encoder
```

并向训练框架输出统一格式的模型结果，例如：

```python
{
    "image_feat": image_feat,
    "text_feat": text_feat,
    "logit_scale": logit_scale
}
```

该模型应保持尽可能纯净，作为整个项目后续实验的基础对照。

---

# `losses/`

存放各种训练目标。

当前阶段只有：

```text
clip_loss.py
```

---

## `clip_loss.py`

实现标准 CLIP 双向对比学习损失。

包括：

```text
Image → Text Contrastive Loss
Text → Image Contrastive Loss
```

总损失：

```text
L_CLIP = (L_I2T + L_T2I) / 2
```

当前 baseline 中不加入：

```text
Prototype Loss
OT Loss
Reconstruction Loss
Triplet Loss
Fine-grained Loss
```

以保证结果能够代表纯 CLIP Fine-tuning 性能。

---

# `engine/`

负责训练与验证过程的统一控制。

---

## `trainer.py`

核心训练循环。

主要负责：

```text
Forward
↓
Loss Calculation
↓
Backward
↓
Optimizer Step
↓
Scheduler Step
↓
Logging
↓
Checkpoint Saving
```

`trainer.py` 不负责具体模型结构，因此后续模型发生变化时，尽量不修改训练框架。

---

## `evaluator.py`

负责训练过程中的验证和模型选择。

例如：

- 每若干 epoch 执行 retrieval evaluation；
- 计算验证集 mR；
- 保存 best checkpoint。

最终目标是使用统一 Evaluator 比较 baseline 与后续完整方法。

---

# `evaluation/`

负责标准遥感图文检索评测。

---

## `retrieval.py`

负责构建图文检索过程。

主要流程：

```text
Extract all image embeddings
            +
Extract all text embeddings
            ↓
     Similarity Matrix
            ↓
      Ranking / Retrieval
```

支持：

```text
Image → Text
Text → Image
```

---

## `metrics.py`

负责计算检索指标：

```text
R@1
R@5
R@10
mR
```

其中：

```text
mR =
平均(
    I2T R@1,
    I2T R@5,
    I2T R@10,
    T2I R@1,
    T2I R@5,
    T2I R@10
)
```

所有模型必须共用同一套 metrics 实现，保证实验公平。

---

# `utils/`

存放与具体算法无关的公共工具。

---

## `config.py`

负责读取和解析 YAML 配置。

---

## `logger.py`

负责训练日志记录，例如：

```text
Epoch
Learning Rate
Training Loss
Validation mR
Best Performance
```

---

## `seed.py`

统一设置随机种子，包括：

```text
Python
NumPy
PyTorch
CUDA
```

提高实验可复现性。

---

## `checkpoint.py`

负责：

- checkpoint 保存；
- checkpoint 加载；
- best model 管理；
- resume training。

---

# `scripts/`

保存常用实验启动脚本。

例如：

```text
train_clip_rsicd.sh
```

用于运行 RSICD baseline。

```text
train_clip_rsitmd.sh
```

用于运行 RSITMD baseline。

这样可以避免每次手动输入大量训练参数，同时方便保存不同实验的固定配置。

---

# `outputs/`

统一保存实验输出。

```text
outputs/
├── checkpoints/
├── logs/
└── results/
```

## `checkpoints/`

保存：

```text
last.pth
best.pth
epoch_xx.pth
```

等模型权重。

## `logs/`

保存训练和验证日志。

## `results/`

保存最终实验结果，例如：

```text
RSICD_CLIP_results.json
RSITMD_CLIP_results.json
```

以及后续论文表格所需的各项 retrieval metrics。

---

# 当前开发阶段

当前优先完成：

```text
Dataset
   ↓
CLIP Backbone
   ↓
CLIP Retrieval Model
   ↓
CLIP Loss
   ↓
Trainer
   ↓
Retrieval Evaluation
```

即：

```text
RSICD / RSITMD
        ↓
Vanilla CLIP Fine-tuning
        ↓
I2T / T2I Retrieval
        ↓
R@1 / R@5 / R@10 / mR
```

在 CLIP baseline 结果稳定之后，再逐步加入后续模块。

预期扩展方向包括：

```text
CLIP Global Retrieval
        +
Entity Grounding
        +
Hierarchical Prototype Learning
        +
GMM Prototype Refinement
        +
Prototype-level Optimal Transport
        +
Token/Patch Fine-grained Alignment
```

整个项目始终保持：

```text
相同 Dataset
相同 Data Split
相同 CLIP Backbone
相同 Evaluation
相同 Training Framework
```

只改变新增模型模块和对应损失，从而保证后续消融实验与 Baseline 的公平可比性。