# RSITR 实体原型流程整理

> 项目：论文3 / 遥感图文检索（RSITR）  
> 当前阶段：文本结构化与实体原型准备  
> 当前原则：**Raw 保留 + Identity 仅做描述剥离 + 单复数保留 + 不删除实体 + 语义聚合交给 Prototype**

---

## 1. 当前整体目标

当前方法希望在 CLIP 全局图文检索的基础上，引入实体级的细粒度对齐能力。

整体主线固定为：

```text
Caption
  ↓
Structured Semantics / EAR
  ↓
Raw Entity Vocabulary
  ↓
Entity Identity Normalization
  ↓
Entity Embedding
  ↓
Entity Prototype
  ↓
Patch Grounding
  ↓
Prototype Alignment
  ↓
RSITR Joint Training
```

当前暂时不引入 GMM、OT、Relation Prototype、Attribute Prototype、EMA Prototype Memory、自适应 Margin、复杂可靠性模块等额外结构，优先验证最小实体原型模块是否有效。

---

# 2. 第一阶段：CLIP Baseline

CLIP Baseline 是后续所有实验的参考基线。

## 2.1 模型配置

```text
Model      : OpenCLIP
Backbone   : ViT-B-32-quickgelu
Pretrained : openai
Dataset    : RSICD
```

数据规模：

```text
Train images : 7,862
Train pairs  : 39,310
Captions/image: 5
```

## 2.2 当前标准 Baseline 结果

```text
I2T
R@1  = 17.02
R@5  = 35.22
R@10 = 48.86

T2I
R@1  = 11.36
R@5  = 34.78
R@10 = 50.58

mR = 32.97
```

后续所有新增模块均与：

\[
\boxed{\text{Baseline mR}=32.97}
\]

进行比较。

---

# 3. Baseline 相关主要文件

## `train.py`

训练入口。

主要流程：

```text
读取 YAML 配置
↓
构建 Dataset / DataLoader
↓
构建 CLIP
↓
构建 Optimizer / Scheduler
↓
训练
↓
Validation
↓
保存 best.pth
```

---

## `test.py`

测试入口。

主要负责：

```text
加载 checkpoint
↓
提取 image feature / text feature
↓
构建相似度矩阵
↓
计算 I2T / T2I Retrieval Metrics
```

输出主要包括：

```text
I2T R@1 / R@5 / R@10
T2I R@1 / R@5 / R@10
mR
```

---

## `configs/baseline/rsicd.yaml`

实验配置文件。

主要保存：

```yaml
image_root:
train_file:
val_file:
test_file:

model:
pretrained:

batch_size:
epochs:
lr:
weight_decay:
```

作用是将实验参数与程序代码分离。

---

# 4. 第二阶段：Structured Semantics / EAR

## 4.1 目标

利用 LLM 将原始 caption 转换成结构化语义：

\[
\boxed{\text{Entity} + \text{Attribute} + \text{Relation}}
\]

例如：

```text
Caption:
many green trees surround several white buildings
```

可得到：

```text
Entity:
    green trees
    white buildings

Attribute:
    green
    white
    many
    several

Relation:
    surround
```

这里 LLM 只负责文本语义解析，不接触图像。

---

# 5. `tools/structured_semantics/` 文件说明

目录：

```text
tools/structured_semantics/
│
├── __init__.py
├── schema.py
├── prompts.py
├── llm_client.py
├── extract.py
├── validator.py
├── sanitizer.py
├── statistics.py
└── cache.py
```

---

## `schema.py`

定义结构化语义输出的数据格式。

例如：

```json
{
  "entities": [],
  "attributes": [],
  "relations": []
}
```

职责：

\[
\boxed{\text{定义 EAR 输出结构}}
\]

---

## `prompts.py`

保存 LLM 的 EAR Prompt。

主要规定：

- 什么属于 Entity；
- 什么属于 Attribute；
- 什么属于 Relation；
- 输出 JSON 的格式；
- 边界情况的处理规则。

当前 EAR Prompt 已冻结，不再频繁修改。

---

## `llm_client.py`

负责调用阿里云百炼 Qwen API。

流程：

```text
Prompt
↓
HTTP API
↓
Qwen
↓
JSON response
```

主要管理：

- Model；
- Base URL；
- API Key；
- Temperature；
- JSON Response；
- Thinking 配置；
- Retry 等。

---

## `extract.py`

EAR 主执行脚本。

流程：

```text
RSICD Captions
↓
Caption 去重
↓
调用 LLM
↓
获得 EAR 结构化语义
```

注意：

Caption 去重仅用于减少 API 调用，训练数据本身并没有被删除或改变。

---

## `validator.py`

负责检查 LLM 输出是否合法。

例如：

- `entities` 是否为 list；
- Entity 字段是否完整；
- Relation 是否引用有效 Entity；
- 字段类型是否合法。

作用：

\[
\boxed{\text{发现结构性错误}}
\]

---

## `sanitizer.py`

负责对部分可安全修复的结构问题做清理。

例如：

```text
缺少 attributes → 补 []
重复 relation   → 去重
错误 entity_id  → 重分配
非法 relation   → 删除
```

注意：Sanitizer 不重新解释 Caption 语义。

---

## `statistics.py`

负责生成 EAR 统计报告。

例如：

- Entity 总数；
- Attribute 总数；
- Relation 总数；
- Valid Rate；
- 平均每条 Caption 的实体数等。

主要用于实验记录和质量检查。

---

## `cache.py`

缓存已经完成的 LLM 调用结果。

目的：

- 避免重复 API 请求；
- 降低调用成本；
- 支持中断恢复；
- 保证重复实验一致性。

---

# 6. EAR 当前结果

RSICD 全量训练 Caption：

```text
Train pairs             : 39,310
Normalized unique caption: 12,296
```

结构化语义统计：

```text
Total entities   : 32,667
Attributes       : 25,376
Relations        : 18,333
Final valid      : 100%
```

因此 EAR 阶段已经完成并冻结。

---

# 7. 第三阶段：Entity Prototype Preparation

当前主要目录：

```text
tools/entity_prototype/
```

正式主线最终建议只保留：

```text
tools/entity_prototype/
│
├── __init__.py
├── entity_vocab.py
├── clean_entities_with_llm.py
├── extract_entity_identity_embeddings.py
└── init_entity_prototypes.py
```

其中后两个分别属于下一阶段和 Prototype 初始化阶段。

---

# 8. `entity_vocab.py`

## 8.1 输入

EAR 结构化语义结果。

## 8.2 输出

```text
data/rsicd/entity_prototype/rsicd_entity_vocab.json
```

## 8.3 作用

从所有 Caption 的 EAR Entity 中构建：

\[
\boxed{\text{Raw Entity Vocabulary}}
\]

当前得到：

```text
Raw unique entity phrases = 2260
```

即：

\[
\boxed{|V_{raw}|=2260}
\]

---

## 8.4 Raw Entity 阶段的固定原则

Raw Entity Vocabulary 必须尽量保留原始语言表达。

例如：

```text
tree
 trees
green trees
rows of trees
white buildings
storage tanks
```

这里全部独立存在。

这一阶段不做：

- Lemmatization；
- Singularization；
- Synonym Merging；
- Ontology Mapping。

因此：

```text
tree != trees
green trees != trees
house != building
```

---

# 9. 为什么 Raw Entity 必须保留？

例如：

```text
green trees
```

它实际上同时包含：

```text
Entity identity = trees
Attribute       = green
```

Raw phrase 可以为后续细粒度 Patch Grounding 提供更丰富的上下文。

因此不能在最开始直接把所有信息压缩成 `tree`。

---

# 10. `clean_entities_with_llm.py`

虽然文件名仍然叫 `clean_entities_with_llm.py`，但正式定义已经改变。

它不再负责“清除无效 Entity”，而是：

\[
\boxed{\text{Entity Identity Normalization}}
\]

---

## 10.1 输入

```text
rsicd_entity_vocab.json
+
EAR structured semantics
```

## 10.2 输出

建议正式输出：

```text
rsicd_entity_identity_vocab.json
```

---

## 10.3 固定职责

将：

```text
Raw Entity Phrase
```

转换为：

```text
Entity Identity Phrase
```

例如：

```text
green trees
→ trees
```

其中：

```text
green = Attribute
trees = Entity Identity
```

---

## 10.4 Identity Normalization 的固定规则

### 允许

#### 去除颜色

```text
green trees
→ trees
```

#### 去除数量描述

```text
many trees
→ trees
```

#### 去除排列/包装表达

```text
rows of trees
→ trees
```

#### 去除简单 Shape/Size 描述

```text
square building
→ building
```

#### 修正明显 Typo

```text
storagetanks
→ storage tanks

passenger termial building
→ passenger terminal building
```

---

## 10.5 必须保留单复数

正式规则：

\[
\boxed{tree\neq trees}
\]

例如：

```text
tree
→ tree

trees
→ trees
```

禁止：

```text
trees
→ tree
```

同理：

```text
buildings → building  ×
roads     → road      ×
cars      → car       ×
```

Morphology 的语义差异留给后续 CLIP + Prototype 自己学习。

---

# 11. 必须保留 Entity-defining Compound

以下表达不允许被过度简化：

```text
storage tanks
basketball fields
tennis courts
football fields
residential buildings
industrial buildings
teaching buildings
apartment buildings
high-rise buildings
parking lots
swimming pools
railway station
riverbank
```

例如：

```text
storage tanks
→ storage tanks
```

不能：

```text
storage tanks
→ tanks
```

---

# 12. 禁止语义替换

LLM Identity Normalization 不负责语义聚类。

所以禁止：

```text
badminton fields → badminton courts
tennis fields    → tennis courts
house            → building
viaduct          → bridge
pond             → water
bank             → riverbank
```

固定原则：

\[
\boxed{\text{不要改变 lexical entity identity}}
\]

---

# 13. 禁止补充 Raw Phrase 中不存在的信息

例如：

```text
tanks
→ tanks
```

不能根据 Caption 上下文自动补成：

```text
tanks
→ storage tanks
```

同样：

```text
residential → residential
industrial  → industrial
commercial  → commercial
```

而不是：

```text
residential → residential area
industrial  → industrial area
commercial  → commercial area
```

---

# 14. 最终取消 Entity Deletion

之前曾尝试使用：

```text
valid / invalid
```

但是出现了如下边界问题：

```text
residential
commercial
industrial
center
```

这些词虽然形式上可能是形容词或泛化词，但在部分 Caption 中确实被当作名词化区域使用。

为了让流程稳定、可迁移，并避免误删有效语义，最终规则修改为：

\[
\boxed{\text{Entity deletion disabled}}
\]

因此即使出现：

```text
it
one
something
place
center
```

也不在 Identity Normalization 阶段删除。

原因：

> 少量噪声可以由后续表示学习吸收；但错误删除有效 Entity 是不可逆的信息损失。

因此当前原则是：

\[
\boxed{\text{保守保留 > 激进删除}}
\]

---

# 15. Raw Entity 与 Identity Entity 的区别

这是当前方法最重要的概念之一。

例如：

```text
raw_text      = green trees
identity_text = trees
```

两者都保留。

不是使用 `trees` 覆盖 `green trees`。

---

## 15.1 Raw Entity 的用途

\[
\boxed{\text{Raw Entity} \rightarrow \text{Patch Grounding}}
\]

例如：

```text
green trees
↓
CLIP Text Encoder
↓
与 Image Patch 比较
```

Raw phrase 包含更完整的实例级描述，因此更适合局部视觉定位。

---

## 15.2 Identity Entity 的用途

\[
\boxed{\text{Identity Entity} \rightarrow \text{Prototype Learning}}
\]

例如：

```text
trees
↓
CLIP Text Encoder
↓
Entity Prototype Routing
```

Prototype 更关注：

```text
“是什么实体”
```

而不是：

```text
“什么颜色”
“多少个”
“如何排列”
```

---

# 16. `tree` 和 `trees` 的最终定义

Raw 层：

\[
\boxed{tree\neq trees}
\]

Identity 层同样：

\[
\boxed{tree\neq trees}
\]

例如：

```text
tree        → tree
trees       → trees
green trees → trees
```

至于 `tree` 和 `trees` 在语义空间中是否接近，不由 LLM 预处理决定，而交给：

\[
\boxed{\text{CLIP Embedding + Prototype}}
\]

这形成一个清晰的方法边界：

> **LLM 负责结构整理，Prototype 负责语义聚合。**

---

# 17. 当前 Identity Normalization 测试结果

前 200 个 Raw Entity 的测试结果：

```text
Raw phrases       : 200
Changed phrases   : 11
Unchanged phrases : 189
Final identities  : 190
```

典型结果：

```text
green trees
→ trees

green plants
→ plants

green meadows
→ meadows

rows of houses
→ houses

rows of trees
→ trees

storagetanks
→ storage tanks
```

说明当前最终规范符合设计目标。

---

# 18. 探索阶段文件

之前尝试过：

```text
canonicalize_entities.py
extract_canonical_entity_embeddings.py
audit_canonical_entities.py
```

这些已经不属于正式 Pipeline。

建议统一移动到：

```text
tools/entity_prototype/legacy/
```

---

## `canonicalize_entities.py`

旧思路：

```text
green trees → tree
trees       → tree
rows of trees → tree
```

也就是说会把：

```text
tree
 trees
```

强行做 Morphology / Semantic Canonicalization。

现在已经放弃，因为这会让 LLM 过早执行本应由 Prototype 学习的语义聚合。

---

## `extract_canonical_entity_embeddings.py`

旧 Canonical Pipeline 的 Embedding 脚本。

曾经生成：

```text
Canonical entities = 997
Embedding shape     = 997 × 512
```

这份结果只保留为探索实验，不进入正式模型。

---

## `audit_canonical_entities.py`

曾计划用于检测：

```text
use
rule
their
this
rad
```

等疑似噪声。

后来发现继续增加 Audit / Repair / Filtering 会导致流程过度复杂，因此取消。

正式 Pipeline：

\[
\boxed{\text{No additional audit stage}}
\]

---

# 19. 下一阶段：Entity Embedding

Identity Normalization 全量完成后，下一步新增：

```text
tools/entity_prototype/
    extract_entity_identity_embeddings.py
```

建议这个脚本一次性产生两种表示。

---

## 19.1 输入

```text
rsicd_entity_vocab.json
rsicd_entity_identity_vocab.json
```

## 19.2 输出

建议统一保存：

```text
rsicd_entity_embeddings.pt
```

内部结构建议：

```python
{
    "raw_phrases": ...,
    "raw_embeddings": ...,

    "identity_phrases": ...,
    "identity_embeddings": ...,

    "raw_to_identity": ...
}
```

---

# 20. 双表示 Entity Embedding

对于：

```text
raw_text      = green trees
identity_text = trees
```

计算：

\[
z_e^{raw}=E_T(\text{"green trees"})
\]

以及：

\[
z_e^{id}=E_T(\text{"trees"})
\]

两种表示用途不同：

```text
Raw embedding
→ Patch Grounding

Identity embedding
→ Prototype Initialization / Routing
```

---

# 21. `init_entity_prototypes.py`

已有旧版 Spherical K-means 实现。

旧实验曾在 Raw Entity 上尝试：

```text
K = 8
K = 16
K = 32
```

发现 Raw Entity 容易受：

```text
white
green
rows
circle
square
```

等描述因素影响。

因此正式版本改成：

\[
\boxed{\text{Identity Embedding} \rightarrow \text{Spherical K-means}}
\]

并且第一版只使用：

```text
K = 16
```

---

# 22. Prototype 初始化

Spherical K-means 得到：

\[
P^{(0)}=\{p_1,p_2,\ldots,p_{16}\}
\]

这些 Prototype Center 只作为初始化。

不是固定语义类别，也不是人工标签。

之后模型中：

```python
self.prototypes = nn.Parameter(...)
```

Prototype 可通过反向传播更新。

因此：

\[
\boxed{\text{K-means = initialization only}}
\]

---

# 23. Patch Grounding

CLIP ViT-B/32 输入 224 × 224 图像时：

```text
224 × 224
↓
7 × 7 patch grid
↓
49 patch tokens
```

对于 Raw Entity：

```text
green trees
```

得到文本表示：

\[
z_e^{raw}
\]

Image Patch Feature：

\[
v_1,\ldots,v_{49}
\]

计算：

\[
s_j=\cos(z_e^{raw},v_j)
\]

再做 Soft Grounding：

\[
a_j=\operatorname{softmax}(s_j/\tau_g)
\]

最终视觉实体表示：

\[
\tilde v_e=\sum_j a_jv_j
\]

得到：

\[
\boxed{\text{Visual Entity Feature}}
\]

---

# 24. Entity Prototype Routing

Identity Entity：

```text
trees
```

文本表示：

\[
z_e^{id}
\]

与 Prototype Bank：

\[
P=[p_1,\ldots,p_{16}]
\]

计算文本 Prototype 分布：

\[
q^t=
\operatorname{softmax}
\left(
\frac{z_e^{id}P^\top}{\tau_p}
\right)
\]

视觉实体表示：

\[
\tilde v_e
\]

得到视觉 Prototype 分布：

\[
q^v=
\operatorname{softmax}
\left(
\frac{\tilde v_eP^\top}{\tau_p}
\right)
\]

然后约束：

\[
q^t\approx q^v
\]

这就是：

\[
\boxed{\text{Prototype Alignment}}
\]

---

# 25. 第一版最小训练目标

暂时只使用：

\[
\boxed{
L=L_{CLIP}+\lambda_eL_{entity}+\lambda_pL_{proto}
}
\]

---

## 25.1 Global CLIP Loss

保持原始 CLIP Baseline：

\[
L_{CLIP}
\]

不改变全局检索主干。

---

## 25.2 Entity Grounding Loss

可先采用：

\[
L_{entity}
=
1-\cos(z_e^{raw},\tilde v_e)
\]

用于约束 Raw Entity 与对应局部视觉区域。

---

## 25.3 Prototype Alignment Loss

使用文本和视觉实体的 Prototype Routing Distribution：

\[
L_{proto}=D_{JS}(q^t,q^v)
\]

用于保证文本实体和视觉实体具有一致的 Prototype 语义归属。

---

# 26. 当前暂缓的模块

第一版暂不加入：

```text
GMM
Optimal Transport
Relation Prototype
Attribute Prototype
EMA Prototype Memory
Adaptive Margin
Reliability Estimation
Sample Filtering
Prototype Diversity Loss
```

原则：

\[
\boxed{\text{先验证最小 Entity + Prototype 模块是否有效}}
\]

如果能够稳定超过 Baseline，再逐步增加复杂模块。

---

# 27. 正式代码目录建议

```text
tools/
│
├── structured_semantics/
│   ├── __init__.py
│   ├── schema.py
│   ├── prompts.py
│   ├── llm_client.py
│   ├── extract.py
│   ├── validator.py
│   ├── sanitizer.py
│   ├── statistics.py
│   └── cache.py
│
└── entity_prototype/
    ├── __init__.py
    ├── entity_vocab.py
    ├── clean_entities_with_llm.py
    ├── extract_entity_identity_embeddings.py
    ├── init_entity_prototypes.py
    │
    └── legacy/
        ├── canonicalize_entities.py
        ├── extract_canonical_entity_embeddings.py
        └── audit_canonical_entities.py
```

---

# 28. 数据目录建议

```text
data/rsicd/
│
├── rsicd_train.json
├── rsicd_val.json
├── rsicd_test.json
│
└── entity_prototype/
    ├── rsicd_entity_vocab.json
    ├── rsicd_entity_identity_vocab.json
    ├── rsicd_entity_embeddings.pt
    │
    └── prototypes/
        └── entity_k16.pt
```

文件含义：

| 文件 | 作用 |
|---|---|
| `rsicd_entity_vocab.json` | Raw Entity Vocabulary |
| `rsicd_entity_identity_vocab.json` | Raw Entity → Identity Entity 映射 |
| `rsicd_entity_embeddings.pt` | Raw + Identity 两类 CLIP Embedding |
| `entity_k16.pt` | K=16 Prototype Initialization |

---

# 29. 最终方法流程图

```text
                         RSICD Caption
                              │
                              ▼
                     Structured Semantics
                        LLM EAR Parsing
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
                Entity    Attribute    Relation
                  │
                  ▼
            Raw Entity Phrase
             "green trees"
                  │
                  │ LLM Identity
                  │ Normalization
                  ▼
               "trees"
                  │
          ┌───────┴────────┐
          │                │
          │                │
Raw phrase embedding   Identity embedding
 "green trees"             "trees"
          │                │
          ▼                ▼
     Patch Grounding    Prototype Bank
          │                │
          ▼                ▼
    Visual Entity      Prototype Route
          │                │
          └───────┬────────┘
                  ▼
           Prototype Alignment
                  │
                  ▼
              RSITR Training
                  │
                  ▼
      CLIP + Entity + Prototype
```

---

# 30. 当前开发进度

```text
CLIP Baseline               ✅ 完成
EAR                         ✅ 完成
Raw Entity Vocabulary       ✅ 完成
Identity Normalization      🟡 Final 版已确定，需全量跑完
Entity Embedding            ⏭ 下一阶段
Prototype K=16              ⏭ 后续
Patch Grounding             ⏭ 后续
Joint Training              ⏭ 后续
```

---

# 31. 当前正式方法边界

以后统一遵循：

\[
\boxed{
\text{Raw 保留}
+
\text{Identity 仅去描述}
+
\text{单复数保留}
+
\text{不删除实体}
+
\text{不做语义替换}
+
\text{语义聚合交给 Prototype}
}
\]

一句话概括：

> **LLM 负责结构化语言，Prototype 负责学习语义聚合，Patch Grounding 负责建立实体与局部视觉区域之间的对应。**

---

# 32. 下一步

当前只需要完成 Final Identity Normalization 全量运行。

随后新增：

```text
extract_entity_identity_embeddings.py
```

一次性生成：

```text
Raw Entity Embeddings
Identity Entity Embeddings
Raw → Identity Mapping
```

完成后进入：

```text
Identity Embedding
↓
K=16 Spherical K-means
↓
Entity Prototype Initialization
↓
Patch Grounding + Prototype Alignment Training
```

从这里开始，不再修改实体清洗流程。
