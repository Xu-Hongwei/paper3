# 第一阶段 Entity Grounding 存在的问题

> 当前阶段：LLM Entity Extraction → CLIP Token Span → Intra-image Entity-Patch Grounding  
> 当前性质：**弱监督图像内实体定位（Weakly-supervised Intra-image Entity Grounding）**

---

# 1. 当前 Entity 使用方式概述

当前 Entity 流程为：

\[
\boxed{
\text{训练 Caption}
\rightarrow
\text{LLM 提取 Entity}
\rightarrow
\text{映射到 CLIP Token Span}
\rightarrow
\text{只作用于该 Caption 正配对的图像}
}
\]

具体而言：

```text
RSICD Train Caption
        ↓
Qwen 离线提取
        ↓
Entity / Attribute / Relation
        ↓
当前阶段只使用 Entity
        ↓
Entity String 匹配回原 Caption
        ↓
CLIP Token Span
        ↓
从 contextualized Token Features 中池化
        ↓
Entity Feature
        ↓
只与该 Caption 对应图像的 49 个 Patch 匹配
```

训练时不会在线调用 LLM，也不会让某个 Entity 去其他图像中搜索对应区域。

因此当前 Grounding 的本质是：

\[
\boxed{
\text{Entity} \leftrightarrow \text{OWN IMAGE Patches}
}
\]

而不是：

\[
\text{Entity} \leftrightarrow \text{Cross-image Regions}
\]

---

# 2. 最核心的问题：我们只有“Entity 属于这张图”的监督

当前已知信息只有：

\[
(\text{Image}, \text{Caption})
\]

以及 LLM 告诉我们：

> Caption 中包含 `airplane`、`bridge`、`river`、`runway` 等 Entity。

因此我们可以较有把握地认为：

\[
\boxed{
\text{该 Entity 应该与这张正配对图像有关}
}
\]

但我们实际上并不知道：

\[
\boxed{
\text{该 Entity 在图像中的具体位置}
}
\]

即：

\[
\boxed{
\text{“Entity 属于这张图”}
\neq
\text{“我们知道 Entity 在这张图哪里”}
}
\]

这就是第一阶段最根本的监督缺口。

---

# 3. 当前 Grounding 实际是在“自己找位置，再监督自己”

对于某个 Entity \(e_i\)，当前做法是：

\[
s_{ij}
=
\cos(e_i,p_j)
\]

其中：

\[
p_j
\]

是该正配对图像中的第 \(j\) 个 Patch。

然后：

\[
a_{ij}
=
\operatorname{softmax}
\left(
\frac{s_{ij}}{\tau}
\right)
\]

通过 Entity 自己决定 Patch 权重，再得到：

\[
v_i
=
\operatorname{Norm}
\left(
\sum_j a_{ij}p_j
\right)
\]

最后优化：

\[
L_e
=
1-\cos(e_i,v_i)
\]

因此实际闭环是：

```text
Entity
   ↓
自己计算 Entity-Patch similarity
   ↓
自己决定哪些 Patch 重要
   ↓
形成 Visual Entity
   ↓
再要求 Visual Entity 靠近自己
```

即：

\[
\boxed{
\text{Entity}
\rightarrow
\text{自己预测位置}
\rightarrow
\text{形成 Visual Entity}
\rightarrow
\text{再监督自己}
}
\]

这是当前第一阶段最明显的结构性风险。

---

# 4. 缺少真实 Region Ground Truth

当前训练没有：

- Bounding Box；
- Segmentation Mask；
- Region Annotation；
- Entity-Region Ground Truth Correspondence。

因此模型无法直接知道：

```text
airplane → 哪些 Patch 是飞机
bridge   → 哪些 Patch 是桥
river    → 哪些 Patch 是河流
```

我们实际上只是做了一个弱假设：

\[
\boxed{
\text{Caption 里的 Entity 应该能够在正配对图像中找到视觉 evidence}
}
\]

因此当前任务本质上属于：

\[
\boxed{
\textbf{Weakly-supervised Intra-image Entity Grounding}
}
\]

而不是有 Region GT 的 supervised grounding。

---

# 5. 当前 Entity Loss 存在 Self-referential Shortcut

由于 Visual Entity 本身就是由 Entity 选择 Patch 得到的：

\[
e_i
\rightarrow
a_{ij}
\rightarrow
v_i
\]

然后又使用：

\[
1-\cos(e_i,v_i)
\]

进行监督，因此当前 Loss 并不一定要求模型真正找到正确区域。

理想情况是：

```text
plane
   ↓
飞机 Patch 相似度明显最高
   ↓
Attention 集中到飞机区域
   ↓
Visual Plane Feature
```

但模型还可能找到更容易的退化解：

```text
plane
   ↓
让很多 Patch 都逐渐和 plane 相似
   ↓
各 Patch similarity 差距缩小
   ↓
Attention 变得均匀
   ↓
全图平均后的 Visual Entity 仍然接近 plane
   ↓
Entity Loss 继续下降
```

也就是说：

\[
\boxed{
L_e \downarrow
\not\Rightarrow
\text{Grounding Quality} \uparrow
}
\]

当前 Loss 可以被优化得很好，但并不保证定位变得更准确。

---

# 6. 当前 Entity Loss 只有 Positive Alignment

当前局部监督主要是：

\[
L_e
=
1-\cos(e_i,v_i)
\]

它只要求：

\[
\boxed{
\text{Entity 和当前正图像中的聚合区域更接近}
}
\]

但没有要求：

\[
S(e,I^+)
>
S(e,I^-)
\]

也没有要求：

```text
正确 Patch
>
错误 Patch
```

因此它更接近：

\[
\boxed{
\text{Local Positive Consistency}
}
\]

而不是：

\[
\boxed{
\text{Local Discriminative Alignment}
}
\]

这意味着模型只需要把正样本内部“拉近”，却不需要建立真正的局部判别边界。

---

# 7. 当前 Grounding 不包含跨图像竞争

对于 batch 中第 \(b\) 个 image-caption pair：

\[
I_b,T_b
\]

Entity：

\[
E_b=\{e_1,\dots,e_n\}
\]

Patch：

\[
P_b=\{p_1,\dots,p_{49}\}
\]

当前只计算：

\[
E_b\times P_b
\]

而不会计算：

\[
e_i^{(b)}
\]

和：

\[
P_{b'},\quad b'\neq b
\]

之间的对应关系。

因此不存在：

\[
\boxed{
\text{Cross-image Entity Grounding}
}
\]

这使得当前 Entity supervision 无法直接回答：

> 当前 Entity 为什么应该匹配这张图，而不是其他图？

所以它与最终图文检索任务需要的跨样本判别能力之间存在明显缺口。

---

# 8. 同一张图的 5 个 Caption 没有被显式联合利用

RSICD 中一张图有 5 条 Caption。

例如同一张机场图可能出现：

```text
Caption 1:
several airplanes are parked at an airport

Caption 2:
many aircraft are standing on an apron

Caption 3:
planes are parked beside airport buildings
```

对应 Entity 可能包括：

```text
airplanes
airport
aircraft
apron
planes
buildings
```

这些描述天然提供：

\[
\boxed{
\text{同一图像的多视角语言监督}
}
\]

但是当前 B2 中：

```text
airplane
aircraft
plane
```

不会因为来自同一图像而被显式建立语义联系。

每个 Entity 都只是独立执行：

\[
e_i\leftrightarrow v_i
\]

因此没有充分利用 RSICD 本身已有的多 Caption 结构。

---

# 9. 当前所有 Entity 都被强制 Ground 到 49 个 Patch

这是另一个重要问题。

并不是所有 LLM Entity 都适合在 7×7 Patch Grid 中形成可靠视觉区域。

例如：

```text
airport staff
cars
vessels
```

可能尺寸非常小。

而：

```text
river
buildings
port
```

又可能覆盖多个 Patch，甚至是分散区域。

但当前训练逻辑没有：

```text
no-match
low-confidence
unresolvable
not-visible
```

等机制。

只要 Entity span 有效，就要求它：

\[
\boxed{
\text{必须从当前图像的 49 个 Patch 中找到对应视觉表示}
}
\]

这会对不可见、小目标、模糊或大范围 Entity 产生错误监督。

---

# 10. 7×7 Patch 分辨率本身存在局部定位上限

当前使用 ViT-B/32，在 224×224 输入下：

\[
7\times7=49
\]

个 Patch。

每个 Patch 大约覆盖：

\[
32\times32
\]

像素。

对于遥感图像中的：

- 小车辆；
- 小船；
- 人员；
- 密集小目标；
- 狭长道路；
- 边界复杂区域；

7×7 Grid 本身可能过于粗糙。

因此：

\[
\boxed{
\text{CLIP Patch Feature 可以作为局部表示起点，
但不能直接等同于高质量 Region Feature}
}
\]

这不是当前 collapse 的唯一原因，但会限制 Grounding 上限。

---

# 11. Local Grounding 与最终 Retrieval Score 脱节

当前训练时：

```text
Image + Caption
      ↓
Global Retrieval Loss
+
Entity Grounding Loss
```

但 Val/Test 时只使用：

\[
S_{\text{global}}(I,T)
=
z_v^\top z_t
\]

最终检索不使用：

- Entity Feature；
- Patch Feature；
- Entity-Patch Similarity；
- Visual Entity Feature。

因此当前 Entity 分支实际上只是：

\[
\boxed{
\text{Training-only Auxiliary Supervision}
}
\]

它只能通过 Adapter 参数变化**间接**影响 Global Retrieval。

于是存在：

```text
Local Grounding 学到信息
        ↓
必须通过共享 Adapter
        ↓
间接改变 Global Embedding
        ↓
才可能改变 Retrieval Ranking
```

这条信息传递路径较弱。

---

# 12. Global 与 Local 共用 Adapter 可能进一步放大问题

当前：

```text
Image Global
Patch
    ↓
共享 Visual Adapter
```

以及：

```text
Text Global
Entity
    ↓
共享 Text Adapter
```

所以：

\[
L_{\text{global}}
\]

和：

\[
L_{\text{entity}}
\]

会同时更新同一组 Adapter 参数。

如果 Entity Grounding objective 本身产生错误或非判别性梯度，它就可能影响 Global Retrieval 表示。

当前实验已经观察到：

```text
Global-only Adapter
Test mR = 32.06

Global + Entity
Test mR = 31.52
```

说明加入当前 Entity supervision 后，Global-only Adapter 原本获得的收益基本被抵消。

不过需要强调：

\[
\boxed{
\text{共享 Adapter 更可能是放大问题的因素，
不是当前 Grounding 失败的最根本原因}
}
\]

即使完全拆分 Global / Local Adapter，当前 Self-referential Entity Loss 仍然可能得到错误局部解。

---

# 13. 当前问题不主要出在 LLM Entity Extraction

目前已有结果表明：

- Entity span coverage 很高；
- 常见实体能够正常提取；
- `plane`、`bridge`、`river`、`buildings` 等都能成功定位到 Token Span；
- Entity Feature 来自完整 Caption 的 contextualized Token Feature。

因此当前没有明显证据说明：

\[
\boxed{
\text{LLM Entity Extraction 是主要瓶颈}
}
\]

当前真正失败的是：

```text
WHAT
Entity 是什么
    ↓
WHERE
它在图像哪里
```

即：

\[
\boxed{
\text{Entity} \rightarrow \text{Visual Region}
}
\]

这一段弱监督映射机制。

---

# 14. 当前第一阶段的核心矛盾

我们目前能够可靠知道：

\[
\boxed{
\text{Entity 与当前正图像有关}
}
\]

但是不知道：

\[
\boxed{
\text{Entity 在当前图像中的具体对应区域}
}
\]

第一版设计实际上直接假设：

```text
Entity 属于正图像
        ↓
Entity 自己选择 Patch
        ↓
选出的 Patch 就可以作为正确 Visual Entity
```

这个假设过强。

因此第一阶段真正需要解决的问题应该重新表述为：

\[
\boxed{
\text{如何在只有 Image-Caption 弱监督的情况下，
可靠地从图像中发现 Entity 对应的视觉 evidence？}
}
\]

---

# 15. 当前问题优先级

| 问题 | 影响程度 |
|---|---:|
| Entity 自己选 Patch、再监督自己的 Self-referential 闭环 | **高** |
| Positive-only Loss 缺少判别竞争 | **高** |
| “Entity 属于图像”被直接当作“知道 Entity 在哪里” | **高** |
| 所有 Entity 强制 Ground，无 no-match / confidence 机制 | **高** |
| Local Grounding 不直接参与 Retrieval Ranking | **高** |
| 同图 5 Caption 的 Entity 没有显式联合利用 | 中高 |
| Global / Local 共用 Adapter | 中高 |
| 7×7 Patch 空间分辨率较低 | 中 |
| LLM Entity Extraction | 当前证据下不是主要问题 |

---

# 16. 第一阶段需要重新定义

当前第一版可以概括为：

\[
\boxed{
\text{Entity Positive Alignment}
}
\]

但下一版应该升级为：

\[
\boxed{
\text{Entity-aware Discriminative Grounding}
}
\]

新的设计至少应该满足：

1. Entity 可以在正图像中寻找局部视觉 evidence；
2. 不能仅靠“所有 Patch 都变得与 Entity 相似”降低 Loss；
3. 正确局部 evidence 应该比错误 Patch / 错误图像更有竞争力；
4. 对不可见、低置信度 Entity 允许不强制匹配；
5. Local similarity 应与最终 Retrieval objective 建立更直接联系；
6. 需要考虑同图多 Caption 带来的跨描述语义结构。

---

# 17. 当前阶段结论

当前 Entity 提取链路本身基本成立：

```text
Caption
→ LLM Entity
→ CLIP Token Span
→ Contextualized Entity Feature
```

真正需要重构的是：

```text
Entity Feature
        ↓
如何可靠选择视觉区域
        ↓
如何形成具有判别性的 Local Alignment
        ↓
如何把 Local 信息有效用于 Retrieval
```

因此第一阶段暴露出的根本问题可以概括为：

\[
\boxed{
\text{我们把“Entity 属于这张图”的弱监督，
过早地转化成了“Entity 已经知道应该看哪些 Patch”的局部监督。}
}
\]

而实际上：

\[
\boxed{
\text{从 WHAT 到 WHERE 的推断本身，
就是第一阶段真正需要解决的问题。}
}
\]
