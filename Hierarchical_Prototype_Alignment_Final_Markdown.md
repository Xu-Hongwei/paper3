# Hierarchical Prototype Alignment with GMM and Optimal Transport for Remote Sensing Image-Text Retrieval

## 1. Overall Framework

本文提出一种层级化原型驱动的遥感图文检索框架：

$$
\boxed{
CLIP\ Global
+
LLM\ Entity\ Grounding
+
Hierarchical\ Prototype\ Learning
+
GMM\ Fine\ Prototype\ Refinement
+
Prototype\ OT
+
Token/Patch\ Fine\text{-}grained\ Alignment
}
$$

目标：

在保持 CLIP
全局检索能力的基础上，引入实体级、层级化、原型驱动的细粒度对齐机制，解决遥感图文检索中的：

-   文本描述粒度不一致；
-   图像区域无语义标签；
-   细粒度类别过多；
-   图文细类非一一对应；
-   正样本噪声污染。

------------------------------------------------------------------------

# 2. Overall Pipeline

``` text
                Image                         Text
                  |                            |
             CLIP Vision                  CLIP Text
                  |                            |
        CLS + Patch Features          EOS + Token Features
                  |                            |
                  |
        -----------------------------
        |                           |
     Global Branch             Entity Branch
        |                           |
    L_global                  LLM Entity Extraction
                                    |
                            Text Entity Prototype
                                    |
              (当前图文 pair 内部实体-区域匹配)
                                    |
                         Top-K Visual Patch Mining
                                    |
                         Visual Entity Prototype
                                    |
                    Entity Prototype Dataset Construction
                                    |
                              Entity Graph
                                    |
                              Coarse Category
                                    |
                    -------------------------------
                    |                             |
              Text Coarse Pool              Visual Coarse Pool
                    |                             |
              GMM Refinement              GMM Refinement
                    |                             |
          Text Fine Prototype          Visual Fine Prototype
                    |
                    |
              Prototype-level OT
                    |
                    |
          Token/Patch Prototype Assignment
                    |
                    |
                 KL Alignment
```

------------------------------------------------------------------------

# 3. CLIP Global Retrieval Branch

## 3.1 Feature Extraction

图像：

$$
z_{cls}^{v}
$$

文本：

$$
z_{eos}^{t}
$$

图像 patch：

$$
V=\{v_1,v_2,\cdots,v_N\}
$$

文本 token：

$$
E=\{e_1,e_2,\cdots,e_M\}
$$

------------------------------------------------------------------------

## 3.2 Global Contrastive Learning

图文相似度：

$$
S_g=
cos(z^v,z^t)
$$

采用 InfoNCE：

$$
L_g=InfoNCE(S_g)
$$

作用：

保持经典 CLIP 的全局图文匹配能力。

------------------------------------------------------------------------

# 4. LLM Text Entity Extraction

## 4.1 Motivation

一个 caption 通常包含多个实体。

例如：

> Several fighter jets are parked near runway.

LLM 提取：

$$
E=
\{
fighter\ jet,
runway
\}
$$

这里的 Entity 不是单个 token，而是具有完整语义的实例。

------------------------------------------------------------------------

# 5. Entity-level Fine Prototype Construction

## 5.1 Text Entity Prototype

对于实体：

$$
e_i
$$

编码：

$$
p_i^t=f_t(e_i)
$$

得到文本实体原型。

------------------------------------------------------------------------

## 5.2 Intra-pair Visual Patch Matching

注意：

该过程不是从整个数据集寻找图像，而是在当前图文 pair
内完成实体-区域匹配。

当前图像：

$$
(I,T)
$$

视觉 patch：

$$
V=\{v_1,\cdots,v_N\}
$$

计算：

$$
s_{ij}=cos(p_i^t,v_j)
$$

选择：

$$
R_i=TopK(s_{ij})
$$

得到实体对应视觉区域。

------------------------------------------------------------------------

## 5.3 Visual Entity Prototype

聚合：

$$
p_i^v
=
\sum_{j\in R_i}\alpha_jv_j
$$

得到视觉实体原型。

此时：

$$
\boxed{
Text\ Entity
\leftrightarrow
Visual\ Entity
}
$$

------------------------------------------------------------------------

# 6. Entity Graph Coarse Category Construction

## 6.1 Motivation

实体数量过多，例如：

``` text
fighter jet
military aircraft
airplane
drone
```

需要进行语义合并。

------------------------------------------------------------------------

## 6.2 Entity Graph

节点：

$$
V_e=\{p_i\}
$$

边：

$$
A_{ij}=cos(p_i,p_j)
$$

利用图聚类获得粗类别。

例如：

$$
\{
fighter\ jet,
airplane,
drone
\}
\rightarrow
Aircraft
$$

------------------------------------------------------------------------

# 7. Coarse-conditioned GMM Fine Prototype Refinement

## 7.1 Motivation

原始 entity 过于碎片化，因此在粗类别内部重新建模。

GMM 的作用不是产生粗类，而是在粗类别内部重新发现稳定细粒度模式。

$$
Many\ Entities
\rightarrow
Few\ Stable\ Fine\ Prototypes
$$

------------------------------------------------------------------------

## 7.2 Text-side GMM

对于粗类别：

$$
C_k^t
$$

收集：

$$
X_k^t=
\{p_i^t|e_i\in C_k\}
$$

拟合：

$$
P_t(z|C_k)
=
\sum_m
\pi_m^t
\mathcal{N}(z|\mu_m^t,\Sigma_m^t)
$$

得到：

$$
\mu_m^t
$$

------------------------------------------------------------------------

## 7.3 Visual-side GMM

视觉侧：

$$
X_k^v=
\{p_i^v|e_i\in C_k\}
$$

拟合：

$$
P_v(z|C_k)
=
\sum_m
\pi_m^v
\mathcal{N}(z|\mu_m^v,\Sigma_m^v)
$$

得到：

$$
\mu_m^v
$$

------------------------------------------------------------------------

# 8. Prototype-level Optimal Transport

文本细粒度原型：

$$
\{\mu_i^t\}
$$

视觉细粒度原型：

$$
\{\mu_j^v\}
$$

二者不是严格一一对应。

------------------------------------------------------------------------

## 8.1 Cost Matrix

$$
C_{ij}
=
1-cos(\mu_i^t,\mu_j^v)
$$

------------------------------------------------------------------------

## 8.2 OT Optimization

$$
\Pi^*
=
\arg\min_\Pi
\sum_{ij}
\Pi_{ij}C_{ij}
$$

得到：

$$
\Pi
$$

表示文本细类与视觉细类之间的软对应关系。

------------------------------------------------------------------------

# 9. Token/Patch Fine-grained Alignment

## 9.1 Token Assignment

文本 token 得到：

$$
q^t
$$

表示其属于不同文本细粒度 prototype 的概率。

------------------------------------------------------------------------

## 9.2 Patch Assignment

视觉 patch 得到：

$$
q^v
$$

表示其属于不同视觉细粒度 prototype 的概率。

------------------------------------------------------------------------

## 9.3 OT Mapping

$$
\hat q^v=\Pi q^t
$$

------------------------------------------------------------------------

## 9.4 KL Alignment

$$
L_{KL}
=
D_{KL}
(q^v||\Pi q^t)
$$

------------------------------------------------------------------------

# 10. Prototype Separation Regularization

为了避免所有 token 和 patch 坍缩到同一个 prototype：

$$
L_{sep}
=
\sum_{i\neq j}
max(0,m-||\mu_i-\mu_j||)
$$

最终：

$$
L_{fine}
=
L_{KL}
+
\lambda L_{sep}
$$

------------------------------------------------------------------------

# 11. Final Objective

$$
\boxed{
L=
L_g
+
\lambda_1L_{entity}
+
\lambda_2L_{OT}
+
\lambda_3L_{fine}
}
$$

其中：

-   $L_g$：CLIP 全局检索损失；
-   $L_{entity}$：文本实体与视觉实体一致性；
-   $L_{OT}$：细粒度 prototype 跨模态匹配；
-   $L_{fine}$：token/patch 最小粒度监督。

------------------------------------------------------------------------

# 12. Main Contributions

## 1. LLM-guided Entity Grounding

利用 LLM 获得明确文本实体，而不是粗糙 token。

------------------------------------------------------------------------

## 2. Pair-wise Entity-to-Patch Matching

在单个图文 pair 内：

$$
Entity\rightarrow Patch
$$

获得视觉实体。

------------------------------------------------------------------------

## 3. Hierarchical Prototype Learning

建立：

$$
Entity
\rightarrow
Coarse
\rightarrow
Fine
$$

层级结构。

------------------------------------------------------------------------

## 4. Coarse-conditioned GMM Refinement

粗类别约束下重新发现细粒度模式。

------------------------------------------------------------------------

## 5. Prototype OT Alignment

解决文本细类和视觉细类不严格对应。

------------------------------------------------------------------------

## 6. Prototype-guided Token/Patch KL

实现：

$$
Token
\leftrightarrow
Patch
$$

最细粒度监督。

------------------------------------------------------------------------

# 13. Summary

一句话总结：

$$
\boxed{
利用 LLM 提取文本实体，在单个图文样本内部定位对应视觉区域，
构建实体级跨模态原型；通过图结构获得粗粒度语义类别，
再利用粗类别约束的 GMM 重新学习稳定细粒度原型，
通过原型级最优传输建立跨模态对应，
最终在原型空间监督 token 与 patch 细粒度对齐。
}
$$
