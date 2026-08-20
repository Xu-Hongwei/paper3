# C0：Clean CLIP Region–Entity Grounding 诊断实验设计

## 1. 实验目标

C0 被定义为一个**纯诊断实验**，而不是新的训练模型。

它只回答一个问题：

> 将视觉匹配单位从 Raw Patch 改为真实的 Region Crop 后，当前经过 RSICD 微调的 Clean CLIP 是否已经具备 Entity 级空间辨别能力？

核心假设：

\[
\boxed{
\text{Entity}\rightarrow\text{Raw Patch 失败}
\not\Rightarrow
\text{CLIP 不具备 Entity 语义能力}
}
\]

更可能的问题是：

\[
\boxed{
\text{单个 ViT Patch 并不是一个完整、稳定的视觉语义单位}
}
\]

因此 C0 将视觉单位替换为：

\[
\text{Region Crop}
\xrightarrow{\text{完整 CLIP Vision Encoder}}
\text{Region Feature}
\]

文本侧则使用独立 Entity 文本：

\[
\text{Entity String}
\xrightarrow{\text{CLIP Text Encoder}}
\text{Entity Feature}
\]

最后直接计算：

\[
S_{ik}=\cos(e_i,r_k)
\]

其中：

- \(e_i\)：第 \(i\) 个 Entity 的文本特征；
- \(r_k\)：第 \(k\) 个 Region Crop 的视觉特征。

---

## 2. 当前实验基础

当前已经完成：

- Clean CLIP 主干清理；
- B1/B1b、Adapter、Local Distillation 等旧分支移除；
- Entity Index v2 构建完成；
- Entity 文本与 Entity token span 均可恢复；
- Clean CLIP 10ep checkpoint 测试通过；
- 当前 RSICD 10ep baseline：

| Metric | I2T | T2I |
|---|---:|---:|
| R@1 | 13.45 | 12.33 |
| R@5 | 32.39 | 34.66 |
| R@10 | 45.38 | 50.87 |
| Mean | 30.41 | 32.62 |

\[
\boxed{\text{mR}=31.51}
\]

后续 C0 直接基于：

```text
configs/baseline/rsicd.yaml
outputs/clip_rsicd_10ep/best.pth
```

进行，不修改训练主链。

---

## 3. C0 整体流程

```text
                    RSICD Image
                         │
                 Eval Image 224×224
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       32×32          64×64          96×96 ...
       Regions         Regions         Regions
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                Resize Region → 224
                         ↓
                Clean CLIP Vision
                         ↓
                  Region Features
                     R × 512


Caption
  │
  ↓
Entity Index v2
  │
  ↓
["stadium", "football field", "buildings"]
  │
  ↓
Independent CLIP Text Encoding
  │
  ↓
Entity Features
     E × 512


Entity Features × Region Featuresᵀ
                 │
                 ↓
             E × R Scores
                 │
                 ↓
          每个 Entity 的 Top-K Regions
```

---

## 4. Region 生成方案

### 4.1 基本尺度

第一版固定采用：

\[
\boxed{32,\ 64,\ 96,\ 128}
\]

所有尺寸均定义在当前 CLIP 的 `224×224` eval 输入空间中。

### 4.2 Stride

采用：

\[
stride=\frac{window}{2}
\]

近似得到：

| Window | Stride | Region 数量 |
|---:|---:|---:|
| 32 | 16 | 169 |
| 64 | 32 | 36 |
| 96 | 48 | 16 |
| 128 | 64 | 9 |

总计约：

\[
169+36+16+9=230
\]

个局部 Region。

### 4.3 Global Control

额外保留完整：

```text
224 × 224 Whole Image
```

作为全局对照，但不把它视为普通局部 Region。

因此每张图约包含：

```text
230 local regions
+ 1 global image
```

### 4.4 Region 编码原则

每个 crop 必须重新经过完整的 CLIP Vision Encoder：

```python
region_features = model.backbone.encode_image(
    crops,
    normalize=True,
)
```

**C0 不使用：**

```python
encode_image_with_patches()
```

因为本实验的核心变量就是：

\[
\boxed{
\text{Raw Patch}
\rightarrow
\text{Full Vision Encoded Region}
}
\]

---

## 5. Entity 文本处理

Entity 直接来自 Entity Index v2：

```python
entities = train_dataset.get_entity_texts(index)
```

例如：

```text
caption:
several buildings and green trees are around a stadium
with a football field in it

entities:
[
    "buildings",
    "trees",
    "stadium",
    "football field"
]
```

文本侧直接独立编码：

```python
entity_features = model.backbone.encode_text(
    entities,
    normalize=True,
)
```

### 第一版明确不加入

- Contextual token span pooling；
- Caption + Entity prompt；
- Attribute；
- Relation；
- Prototype；
- Prompt Ensemble；
- `"a satellite image of ..."` 等额外 prompt。

目的就是控制变量，只验证：

\[
\boxed{
\text{Independent Entity Text}
\leftrightarrow
\text{Region Crop}
}
\]

---

## 6. 相似度计算

Region 和 Entity 特征均做 L2 Normalize，因此：

\[
S_{ik}=e_i^\top r_k
\]

即 cosine similarity。

对于每个 Entity：

\[
S_i^{global}
=
\cos(e_i,I)
\]

\[
S_i^{region}
=
\max_k \cos(e_i,r_k)
\]

定义：

\[
\Delta_i
=
S_i^{region}
-
S_i^{global}
\]

作为 **Region Gain**。

建议每个 Entity 至少记录：

```text
Entity
Global similarity
Best region similarity
Region gain
Best region scale
Best region box
Top-K regions
```

示例：

| Entity | Global | Best Region | Gain |
|---|---:|---:|---:|
| stadium | 0.22 | 0.34 | +0.12 |
| football field | 0.18 | 0.37 | +0.19 |
| buildings | 0.24 | 0.31 | +0.07 |

如果 Region Crop 后 Entity similarity 明显提高，并且 Top Region 位于合理位置，则说明：

> CLIP 本身已经具备一定实体语义能力，之前失败的主要问题可能是 Raw Patch Interface，而不是 CLIP 不理解这些实体。

---

## 7. 第一版不使用 NMS

C0 v1：

\[
\boxed{\text{No NMS}}
\]

原因：

C0 当前不是 object detection，而是在观察模型的原始 Region response。

如果：

```text
stadium Top-1
stadium Top-2
stadium Top-3
```

对应多个高度重叠的 Region，这本身就是有意义的信息，说明模型对某个位置的响应具有稳定性。

提前使用 NMS 反而可能掩盖真实行为。

---

## 8. C0 的评价方式

RSICD 没有 Entity Bounding Box、Region Annotation 或 Segmentation Mask，因此：

\[
\boxed{
\text{当前不能定义真正的 Grounding Accuracy}
}
\]

### 8.1 主要评价：定性观察

重点观察：

```text
Original Image
Entity
Top-1 Region
Top-5 Regions
Region Similarities
```

例如：

```text
stadium
→ 是否覆盖 stadium

football field
→ 是否集中到 stadium 内部 field

buildings
→ 是否偏向周围建筑区域

trees
→ 是否偏向植被区域
```

### 8.2 辅助诊断指标

可以计算，但不能称为 accuracy：

#### Best Region Similarity

\[
\max_k S_{ik}
\]

#### Region Gain

\[
\Delta_i
=
S_i^{region}
-
S_i^{global}
\]

#### Top1–Top2 Margin

\[
M_i
=
S_i^{top1}
-
S_i^{top2}
\]

只表示模型对某个 Region 的相对偏好强弱，不表示定位是否正确。

#### 不同 Entity Top-1 Region IoU

\[
IoU(B_i^{top1},B_j^{top1})
\]

可用于辅助判断多个 Entity 是否总是落在同一 Region。

但不能简单认为：

\[
IoU越低越好
\]

因为例如：

```text
stadium
football field
```

天然存在空间嵌套关系，高 IoU 可能完全合理。

---

## 9. 第一轮样本选择

第一版不需要随机跑全数据集。

先人工选择：

\[
\boxed{10\sim20\text{ 个空间结构明显的样本}}
\]

覆盖以下类型。

### 小尺度实体

```text
airplane
ship
storage tanks
```

### 中尺度实体

```text
buildings
pond
football field
```

### 大尺度实体

```text
stadium
airport
river
runway
```

### 相邻实体

```text
pond + buildings
ship + port
```

### 嵌套实体

```text
stadium + football field
airport + runway
```

### 多实体复杂场景

例如：

```text
several buildings and green trees are around a stadium
with a football field in it
```

对应：

```text
buildings
trees
stadium
football field
```

这是一个非常适合 C0 的典型案例，因为其空间结构明显：

\[
\text{buildings}
\neq
\text{trees}
\neq
\text{stadium}
\supset
\text{football field}
\]

---

## 10. 第一阶段使用 Train Sample

C0 当前是内部表征诊断，而不是正式泛化性能实验。

因此第一阶段允许使用：

```text
Train Sample
+
Entity Index v2
```

因为当前目的是验证：

> 经过 RSICD 微调后的 CLIP 内部，是否已经存在 Entity → Region 的可匹配结构？

并不声称：

```text
RSICD test grounding accuracy
```

如果后续需要在论文中正式声称模型具备 grounding ability，则必须进一步准备：

- Val/Test EAR；
- 或少量人工 Entity–Region GT；
- 或独立 grounding benchmark。

这不属于 C0 v1 范围。

---

## 11. C0 输出设计

### 11.1 JSON

每个 sample 保存完整诊断信息，例如：

```json
{
  "index": 14628,
  "caption": "several buildings and green trees are around a stadium with a football field in it",
  "entities": [
    {
      "text": "stadium",
      "global_similarity": 0.231,
      "best_region_similarity": 0.357,
      "region_gain": 0.126,
      "top_regions": [
        {
          "scale": 128,
          "box": [48, 48, 176, 176],
          "similarity": 0.357
        }
      ]
    }
  ]
}
```

这样以后可以直接做统计，而不需要重新运行 CLIP。

### 11.2 PNG

第一版建议：

```text
----------------------------------------------------
Caption: ...
----------------------------------------------------

Entity: stadium
Original | Top1 | Top2 | Top3 | Top4 | Top5

Entity: football field
Original | Top1 | Top2 | Top3 | Top4 | Top5

Entity: buildings
Original | Top1 | Top2 | Top3 | Top4 | Top5
```

或者在原图上直接绘制 Top-K Region boxes。

第一版不需要复杂 heatmap。

---

## 12. C0 结果解释

### 情况 A：Region Grounding 明显改善

例如：

```text
stadium        → stadium region
football field → field
buildings      → surrounding buildings
trees          → vegetation
```

说明：

\[
\boxed{
\text{主要瓶颈是 Raw Patch Interface}
}
\]

而不是：

\[
\text{CLIP 完全没有实体语义能力}
\]

后续可以继续研究：

```text
Region Proposal
Region–Entity Alignment
Multi-scale Region Fusion
Prototype Learning
```

---

### 情况 B：部分实体成功，细粒度仍然混淆

例如：

```text
stadium   → 正确
buildings → 正确

stadium / football field → 高度重叠
ship / port               → 混淆
```

说明：

\[
\boxed{
\text{Region 表征具备基础，但 Fine-grained Discrimination 不足}
}
\]

后续重点应转向：

```text
Region-level discrimination
Entity-aware region refinement
Structured semantic supervision
```

这将为后续方法设计提供明确依据。

---

### 情况 C：Region Crop 后仍全面失效

例如：

```text
不同 Entity 基本选择同一位置
不同 Entity 的 Region score 分布高度相似
Entity 改变后 Top Region 几乎不变化
```

说明问题不仅仅来自 Raw Patch 粒度。

此时才有必要进一步考虑：

```text
Visual Local Adaptation
Attention Refinement
Region Proposal Network
更强局部视觉编码器
```

而不是提前堆叠 Prototype、OT 等复杂模块。

---

## 13. C0 v1 的严格边界

最终 C0 v1 固定为：

\[
\boxed{
\begin{aligned}
&\text{Clean CLIP 10ep Checkpoint}\\
+&\text{EAR Entity Text}\\
+&\text{32/64/96/128 Multi-scale Crops}\\
+&\text{Full CLIP Region Encoding}\\
+&\text{Cosine Similarity}\\
+&\text{Top-K Visualization}
\end{aligned}
}
\]

明确不加入：

```text
PACL
Raw Patch Grounding
Adapter
Teacher
Distillation
Contextual Entity
Attribute
Relation
Prototype
OT
NMS
额外训练
```

---

## 14. 代码组织

第一版只新增一个独立诊断脚本：

```text
test/
└── c0_region_entity_grounding.py
```

直接读取：

```text
configs/baseline/rsicd.yaml
```

以及：

```text
outputs/clip_rsicd_10ep/best.pth
```

不修改：

```text
train.py
models/
engine/
losses/
evaluation/
```

从而保证：

\[
\boxed{
\text{C0 始终是一个与训练主链解耦的纯诊断实验}
}
\]

---

## 15. C0 最终要回答的问题

C0 的最终结论只围绕一句话展开：

\[
\boxed{
\text{当视觉单位从 Raw Patch 换成完整编码的 Region Crop 后，}
\text{Clean CLIP 能否表现出稳定、Entity-specific 的空间响应？}
}
\]

如果答案是“能”，则后续研究应沿 Region-level Fine-grained Alignment 推进。

如果答案是“部分能”，则需要研究 Region-level Discrimination。

如果答案仍然是“不能”，则说明视觉局部表征本身仍需增强。
