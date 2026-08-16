# 2026-08-16：FarSLIP 启发下的 RSITR 后续路线整理

## 1. 今日结论概述

经过对 CLIPSelf、FarSLIP 以及当前 RSICD 实验路线的重新梳理，当前项目后续不应继续沿着“Entity 直接拉 Patch”的方式推进，而应重新组织为一条更清晰的技术主线：

\[
\boxed{
\text{Global Retrieval Anchor}
\rightarrow
\text{Local Visual Adaptation}
\rightarrow
\text{Structured Region Grounding}
\rightarrow
\text{Prototype / OT Fine Reranking}
}
\]

核心思想是：

1. **保留 CLIP 已有的全局图文检索空间；**
2. **先单独提升视觉局部表征质量；**
3. **再进行结构化跨模态语义对齐；**
4. **最后利用 Prototype / OT 解决同类别内部的细粒度排序问题。**

这比最初直接采用

\[
\text{CLIP}+\text{Entity}+\text{GMM}+\text{OT}
\]

的串联方式更完整，因为每个阶段都有明确的前置问题、优化目标和下游用途。

---

## 2. FarSLIP 给出的关键启发

FarSLIP 最重要的启发不是简单照抄两阶段训练，而是验证了一个关键原则：

\[
\boxed{
\text{局部视觉表征学习}
\quad\text{和}\quad
\text{局部跨模态语义对齐}
\text{最好解耦}
}
\]

其基本经验是：

- 视觉局部能力增强时，更适合使用 **Patch-to-Patch**；
- 跨模态语义监督时，更适合使用 **Crop CLS-to-Text**；
- 不建议直接使用 Raw Patch 强行对齐文本类别；
- 必须注意局部适配对 CLIP 原有 semantic coherence 和 global retrieval space 的破坏。

可以概括为：

```text
Stage 1
视觉局部学习：
Patch ↔ Patch

同时：
Whole Image ↔ Caption
维持并提升 Global Retrieval

Stage 2
区域跨模态学习：
BBox Crop CLS ↔ Category Text

同时：
Whole Image ↔ Caption
继续 Global Retrieval
```

其核心思想是：

\[
\boxed{
\text{Patch 负责局部视觉结构}
+
\text{CLS 负责稳定跨模态语义}
}
\]

---

## 3. FarSLIP Stage 1 对我们的意义

FarSLIP Stage 1 使用：

\[
\boxed{
L_{\text{stage1}}
=
L_{\text{glo}}
+
L_{\text{dis}}
}
\]

其中：

- \(L_{\text{glo}}\)：标准 Image-Caption CLIP InfoNCE；
- \(L_{\text{dis}}\)：Patch-to-Patch Self-Distillation。

FarSLIP Stage 1 的 global retrieval 能力明显提高，但这种提升不能全部归因于局部蒸馏，因为它同时进行了大规模遥感域 Global Contrastive Learning。

因此其提升来自：

```text
大规模 RS5M 图文适配
        +
Patch-to-Patch Local Distillation
```

而不是单独一个 Local Loss。

---

## 4. 当前 Clean CLIP B1b 的定位

当前正式实验采用：

```text
RSICD-finetuned CLIP
        │
        ├── Frozen Teacher Vision
        │
        └── Student Vision
              Block 1~9 frozen
              Block 10~12 trainable
```

训练目标：

\[
\boxed{
L_{\text{B1b}}
=
L_{\text{local}}
+
10L_{\text{preserve}}
}
\]

其中：

\[
L_{\text{local}}
=
1-\cos(r_R^S,t_R^T)
\]

用于增强 Local Representation；

\[
L_{\text{preserve}}
=
1-\cos(z_I^S,z_I^T)
\]

用于约束 Student 全局视觉语义不要明显偏离已经训练好的 RSICD CLIP。

当前 B1b 的核心目标不是直接提高最终检索 mR，而是：

\[
\boxed{
\text{Local plasticity}
+
\text{Global semantic stability}
}
\]

也就是在尽量保持原有 CLIP 全局检索空间的同时，增强视觉编码器的局部区域表征能力。

---

## 5. 当前 B1b Smoke Test 已验证的结论

Clean CLIP B1b smoke test 已通过。

关键结构验证：

```text
Teacher frozen
Vision Blocks 10~12 active
Other CLIP parameters frozen

Entity span index : disabled
Global retrieval  : raw Clean CLIP features
```

10-step smoke 中：

\[
LocalCos:
0.2905
\rightarrow
0.3720
\]

说明最后三个 Vision Transformer Blocks 已经开始学习局部监督。

平均：

\[
GlobalCos
=
0.999162
\]

说明 \(\lambda_{preserve}=10\) 对全局视觉表示具有较强保护作用。

Validation：

\[
30.03
\rightarrow
29.90
\]

仅下降约 0.13 mR。

因此当前结论是：

\[
\boxed{
\text{Clean CLIP B1b pipeline PASS}
}
\]

正式 10 epoch 实验继续保留：

```yaml
local_trainable_blocks: 3
local_distill_weight: 1.0
global_preserve_weight: 10.0
lr: 1e-5
batch_size: 64
warmup_ratio: 0.05
```

---

## 6. B1b 应该如何在论文中描述

B1b 不应被描述成“一个用于直接提升图文检索的模块”。

更合理的描述是：

> A stability-aware local visual adaptation stage that improves local visual representations while preserving the original global CLIP semantic space.

候选名称：

\[
\boxed{
\text{Global-Preserving Local Adaptation (GPLA)}
}
\]

或：

\[
\boxed{
\text{Semantic-Preserving Local Distillation (SPLD)}
}
\]

其作用是为后续 Structured Grounding 提供可靠的视觉局部空间。

---

## 7. 下一步最值得做的改进：Patch-to-Patch

当前 B1b 的 Local Teacher target 仍然更接近 CLIPSelf：

```text
Full Image
    ↓ Student
Patch Map
    ↓
Region Pool
    ↓
r_student

      ↕

Crop Image
    ↓ Frozen Teacher
Crop CLS
    ↓
t_teacher
```

即：

\[
\text{Region Patch}
\leftrightarrow
\text{Crop CLS}
\]

潜在问题是 Crop CLS 属于高度压缩的全局语义表示，可能迫使不同局部 Patch 向同一 Crop 语义中心收缩，从而造成局部表示均质化。

因此建议下一实验改为：

```text
Full Image
    ↓ Student
Patch Map
    ↓
Region Pool
    ↓
r_student

      ↕

Crop Image
    ↓ Teacher
Crop Patch Map
    ↓
Patch Pool
    ↓
r_teacher
```

即：

\[
\boxed{
L_{\text{P2P}}
=
1-
\cos
\left(
r_{\text{full-region}},
r_{\text{crop-patch}}
\right)
}
\]

该实验可以与当前 B1b 构成清楚的消融：

| 方法 | Student | Teacher target |
|---|---|---|
| 当前 B1b | Full-image Region Patch | Crop CLS |
| 改进版 | Full-image Region Patch | **Crop Pooled Patches** |

如果 P2P 在局部诊断上明显优于 Crop-CLS，则可以得到较强证据：

\[
\boxed{
\text{Local visual target 的设计比单纯增加 Preserve 权重更重要}
}
\]

---

## 8. Local Representation 的评价不能只看 Local Loss

后续评价 Local Adaptation 时，不能只看：

\[
L_{\text{local}}\downarrow
\]

还需要固定诊断集进行 Patch/Entity-level 分析。

当前已有固定：

```text
8 images
19 entities
```

建议继续比较：

```text
RSICD CLIP baseline
vs
B1b Crop-CLS
vs
P2P Local
```

指标包括：

\[
Entropy\downarrow
\]

\[
Top5\uparrow
\]

\[
SimStd\uparrow
\]

\[
Max-Mean\uparrow
\]

\[
Max-Min\uparrow
\]

同时注意：相似度更集中并不等于 Grounding 更正确。最好后续增加少量人工可视化，用于判断高响应区域是否真的落在对应实体上。

---

## 9. 是否重新加入 Global InfoNCE

FarSLIP Stage 1 的另一个启发是：

\[
\boxed{
L_{\text{global}}
\text{可以和}
L_{\text{local}}
\text{共同训练}
}
\]

未来可以考虑：

\[
\boxed{
L
=
\lambda_l L_{\text{P2P}}
+
\lambda_p L_{\text{preserve}}
+
\lambda_g L_{\text{global}}
}
\]

三个 loss 分别负责：

- \(L_{P2P}\)：局部视觉质量；
- \(L_{preserve}\)：防止全局语义漂移；
- \(L_{global}\)：主动优化 Retrieval discriminability。

但当前不建议立刻加入。

原因是：

- FarSLIP 使用大规模 RS5M；
- 我们 RSICD 只有 7862 张训练图像；
- 当前实验首先应该回答一个更干净的问题：

\[
\boxed{
\text{小数据条件下，能否单独增强 Local 而保持 Global？}
}
\]

因此合理实验顺序为：

```text
B1b Crop-CLS
      ↓
P2P Local
      ↓
P2P + Global InfoNCE
```

---

## 10. 对旧 Entity Grounding 的重新判断

旧 Entity Grounding 的基本逻辑是：

\[
e_i
\leftrightarrow
p_j
\]

例如：

```text
"airplane"
    ↓
Entity embedding
    ↓
与 49 个 Patch 算 similarity
    ↓
Top-K 聚合
    ↓
强制 Entity-Patch 对齐
```

此前实验结果出现：

\[
Entropy\uparrow
\]

\[
Top5\downarrow
\]

\[
SimStd\downarrow
\]

说明 Patch 表征存在明显 homogenization / attention diffusion。

结合 FarSLIP 的分析，可以进一步解释为：

\[
\boxed{
\text{Raw Patch 并不是 CLIP 原始训练中最稳定的语言对齐接口}
}
\]

因此后续不建议再使用简单：

\[
L
=
1-\cos(e,p)
\]

这样的强 Entity-Patch alignment。

---

## 11. Structured Grounding 应该如何改

我们没有 FarSLIP Stage 2 那样真实的：

```text
BBox + Category
```

但我们有：

\[
\boxed{
\text{Entity}
+
\text{Attribute}
+
\text{Relation}
}
\]

因此可以构建一种无 bbox 的弱监督 Structured Region Grounding。

推荐流程：

```text
Text Caption
    ↓
LLM Structured Semantics
    ↓
Entity + Attribute + Relation


Image
    ↓
Enhanced Patch Map
    ↓
Candidate Region Generation
    ↓
Region Crop
    ↓
CLIP Vision
    ↓
Region CLS
    ↓
Structured Semantic Matching
```

也就是：

\[
\boxed{
\text{Patch 不直接承担强语言监督}
}
\]

而是：

\[
\boxed{
\text{Patch 用于定位 Candidate Region}
}
\]

真正的跨模态语义匹配采用：

\[
\boxed{
CLS(R_k)
\leftrightarrow
e_i
}
\]

---

## 12. Structured Grounding 示例

Caption：

> “two white airplanes beside a long runway”

LLM 输出：

```text
Entity:
- airplanes
- runway

Attribute:
- airplanes → count=two
- airplanes → color=white
- runway → extent=long

Relation:
- airplanes → beside → runway
```

图像侧产生候选：

```text
R1 = airplane region
R2 = airplane region
R3 = runway region
R4 = terminal region
```

分别计算：

\[
S_E(e_i,R_k)
\]

解决“R1 是不是 airplane？”；

\[
S_A(a_i,R_k)
\]

解决“R1 是否满足 white / two 等属性？”；

\[
S_R(r_{ij},R_k,R_l)
\]

解决“airplane region 和 runway region 是否满足 beside？”。

最终：

\[
\boxed{
S_{\text{struct}}
=
S_E
+
\alpha S_A
+
\beta S_R
}
\]

这才是后续真正意义上的 Structured Semantic Grounding。

---

## 13. Prototype 应该往后移动

之前路线偏向：

```text
Patch
 ↓
Entity Grounding
 ↓
Cluster
 ↓
GMM Prototype
```

现在建议改成：

```text
Patch
 ↓
Candidate Region
 ↓
Region Semantic Feature
 ↓
Structured Grounding
 ↓
Reliable Region-Entity Pair
 ↓
Prototype Learning
```

也就是：

\[
\boxed{
\text{Prototype 不直接聚 Raw Patch}
}
\]

而应该聚：

\[
\boxed{
\text{已经获得结构化语义的 Region representations}
}
\]

这样 Prototype 才有明确语义，例如：

```text
airplane
├── isolated airplane
├── multiple airplanes
├── airplanes near runway
├── airplanes near terminal
└── airplanes on apron
```

Prototype 的任务因此从“场景类别中心”转变为：

\[
\boxed{
\text{Attribute / Relation-conditioned intra-category modes}
}
\]

这与项目要解决的：

\[
\boxed{
\text{intra-category fine-grained discrimination}
}
\]

保持一致。

---

## 14. GMM 的重新定位

GMM 不应作为框架中必须存在的模块。

建议把它降级为：

\[
\boxed{
\text{可验证的细粒度 Prototype refinement 方法}
}
\]

实验顺序：

```text
Single Prototype
      ↓
K-means / EMA Prototype
      ↓
GMM Prototype
```

只有当 grounded region features 确实呈现明显多峰分布，并且 GMM 比简单 Prototype 明显更好时，才保留 GMM。

否则不应为了方法复杂度强行加入 GMM。

---

## 15. OT 的重新定位

OT 同样不应该提前参与 Grounding。

合理顺序为：

\[
\boxed{
Grounding
\rightarrow
Prototype
\rightarrow
OT
}
\]

等图文两侧已经得到：

\[
\mu_1^t,\mu_2^t,\ldots
\]

和：

\[
\mu_1^v,\mu_2^v,\ldots
\]

之后，再解决图文细粒度模式不是严格一一对应的问题。

定义：

\[
C_{ij}
=
1-
\cos(\mu_i^t,\mu_j^v)
\]

求：

\[
\Pi^*
\]

此时 OT 的职责才是：

\[
\boxed{
\text{Soft many-to-many prototype correspondence}
}
\]

---

## 16. 最终 Retrieval 推荐改为两阶段

之前计划：

\[
S_{\text{final}}
=
S_{\text{global}}
+
\lambda S_{\text{fine}}
\]

对所有 Image-Text pair 都计算 Fine score。

但随着后续加入 Entity、Attribute、Relation、Region、Prototype、OT，全库两两计算会非常昂贵。

而当前项目的主要问题本来就是：

> CLIP 已经能找到大致正确的场景，但在同类别内部排序不够精确。

因此更适合采用：

### Stage A：Global Retrieval

\[
S_g
=
\cos(z_I,z_T)
\]

使用 Clean CLIP 先召回 Top-50 或 Top-100 候选图像。

### Stage B：Fine-grained Reranking

只对 Top-M 计算：

\[
S_{\text{struct}}
\]

\[
S_{\text{proto}}
\]

\[
S_{\text{OT}}
\]

最终：

\[
\boxed{
S_{\text{rerank}}
=
S_g
+
\lambda_1S_{\text{struct}}
+
\lambda_2S_{\text{proto}}
+
\lambda_3S_{\text{OT}}
}
\]

这样更加符合：

\[
\boxed{
\text{专门解决 intra-category ranking}
}
\]

这一研究目标。

---

## 17. 最终推荐路线

| 阶段 | 核心目的 | 方法 |
|---|---|---|
| A. Clean CLIP | 全局检索基座 | Global CLIP |
| B1. 当前 B1b | 验证 Last-3 local plasticity | Region Patch → Crop CLS |
| B2. Local 改进 | 提高局部视觉质量 | **Patch → Patch** |
| B3. 可选 Global CL | 主动维护/提升 retrieval | \(L_{P2P}+L_{preserve}+L_g\) |
| C. Structured Grounding | 无 bbox 的 region semantics | Candidate Region CLS ↔ E/A/R |
| D. Prototype | 建模同类别内部语义模式 | Grounded Region Prototypes |
| E. OT | 处理图文非一一对应 | Prototype-level OT |
| F. Retrieval | 最终细粒度排序 | Global Retrieve → Fine Rerank |

---

## 18. 当前最合理的实验顺序

### Experiment 1：当前正式 B1b

继续跑：

```text
Clean CLIP
+
Region Patch → Crop CLS
+
Global Preserve
```

观察：

- LocalCos；
- GlobalCos；
- Val mR；
- 固定 Patch/Entity diagnostic。

### Experiment 2：Patch-to-Patch

只修改 Local Teacher target：

```text
Crop CLS
→
Crop Pooled Patches
```

其他参数保持尽量一致。

目标：

> 判断 P2P 是否比 Crop-CLS 更适合 RSICD 小数据局部适配。

### Experiment 3：Global InfoNCE Ablation

在 P2P 稳定后增加 \(L_g\)：

```text
P2P + Preserve
vs
P2P + Preserve + Global InfoNCE
```

判断小规模 RSICD 条件下继续 Global CL 是否真正带来 retrieval improvement。

### Experiment 4：Structured Region Grounding

不再采用：

\[
Entity\rightarrow Raw Patch
\]

而改为：

```text
Entity / Attribute / Relation
        ↓
Candidate Region
        ↓
Region CLS
        ↓
Structured Matching
```

### Experiment 5：Prototype

先简单 Prototype，再判断是否需要 GMM。

### Experiment 6：Prototype OT

只在 Prototype 语义可靠以后加入。

### Experiment 7：Fine-grained Reranking

最终使用：

```text
Global CLIP Retrieval
        ↓
Top-M
        ↓
Structured Fine Reranking
```

评估是否真正改善同类别内部的正确实例排序。

---

## 19. 最终方法逻辑

当前建议的最终论文逻辑可以概括为：

\[
\boxed{
\underbrace{\text{Global Alignment}}_{\text{Clean CLIP}}
\rightarrow
\underbrace{\text{Reliable Local Visual Structure}}_{\text{P2P Adaptation}}
\rightarrow
\underbrace{\text{Structured Region Semantics}}_{\text{Entity+Attribute+Relation}}
\rightarrow
\underbrace{\text{Intra-class Semantic Modes}}_{\text{Prototype}}
\rightarrow
\underbrace{\text{Soft Cross-modal Correspondence}}_{\text{OT}}
\rightarrow
\underbrace{\text{Fine-grained Reranking}}_{\text{Final Retrieval}}
}
\]

更简洁地说：

\[
\boxed{
\text{先保 Global}
+
\text{再学 Local}
+
\text{再做 Structured Grounding}
+
\text{最后解决同类排序}
}
\]

---

## 20. 当前研究判断

当前阶段最重要的不是继续增加模块数量，而是把两个基础问题真正解决：

\[
\boxed{
1.\ \text{Local representation 是否可靠？}
}
\]

\[
\boxed{
2.\ \text{没有 bbox 时，如何建立可靠 Structured Region Grounding？}
}
\]

只要这两点建立起来，Prototype、GMM、OT 才具有明确作用。

因此当前路线优先级为：

```text
当前 B1b 正式实验
        ↓
Patch-to-Patch
        ↓
Structured Region Grounding
        ↓
Prototype
        ↓
OT
        ↓
Fine-grained Reranking
```

而不再建议：

```text
Entity-Patch 强拉
        ↓
直接 Prototype
        ↓
直接 OT
```

---

**日期：2026-08-16**

**当前项目：Remote Sensing Image-Text Retrieval / Hierarchical Prototype Alignment**

**当前阶段：Clean CLIP B1b Local Visual Adaptation**
