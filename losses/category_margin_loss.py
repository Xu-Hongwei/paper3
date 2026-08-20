import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossCategoryMarginLoss(nn.Module):
    """
    双向跨类别 batch-hard margin loss。

    仅使用已知类别(category_id >= 0)：
        T2I: 文本查询，从不同类别图像中选择最相似 hard negative；
        I2T: 图像查询，从不同类别文本中选择最相似 hard negative。

    当前版本只验证纯 category-aware hard negative，不使用 LLM Entity 过滤。
    输入特征会在内部 L2 normalize，因此 margin 直接作用于 cosine similarity。
    """

    def __init__(self, margin=0.05):
        super().__init__()

        self.margin = float(margin)
        if self.margin < 0:
            raise ValueError("margin must be >= 0.")

    @staticmethod
    def _check_inputs(
        image_features,
        text_features,
        category_ids,
    ):
        if image_features.ndim != 2:
            raise ValueError(
                "image_features must have shape [B, D]."
            )

        if text_features.ndim != 2:
            raise ValueError(
                "text_features must have shape [B, D]."
            )

        if image_features.shape != text_features.shape:
            raise ValueError(
                "image_features and text_features must have "
                "the same shape. "
                f"Got {tuple(image_features.shape)} and "
                f"{tuple(text_features.shape)}."
            )

        batch_size = image_features.shape[0]

        if category_ids.ndim != 1:
            raise ValueError(
                "category_ids must have shape [B]."
            )

        if category_ids.shape[0] != batch_size:
            raise ValueError(
                f"category_ids must have shape [{batch_size}], "
                f"got {tuple(category_ids.shape)}."
            )

    @staticmethod
    def _zero_loss(image_features, text_features):
        """返回保持计算图连接的标量 0。"""
        return (
            image_features.sum()
            + text_features.sum()
        ) * 0.0

    def _direction_loss(
        self,
        similarities,
        category_ids,
        image_features,
        text_features,
    ):
        """
        similarities[i, j]:
            anchor i 与 candidate j 的 cosine similarity。

        正样本固定为当前 pair 的对角线；
        负样本只允许 category_j != category_i 且双方类别已知。
        """
        batch_size = similarities.shape[0]
        device = similarities.device

        known = category_ids >= 0

        negative_mask = (
            known[:, None]
            & known[None, :]
            & (
                category_ids[:, None]
                != category_ids[None, :]
            )
        )

        has_negative = negative_mask.any(dim=1)
        valid_mask = known & has_negative
        valid_count = int(valid_mask.sum().item())

        pos_sim = similarities.diagonal()

        masked_sim = similarities.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
        hard_neg_sim, hard_neg_index = masked_sim.max(dim=1)

        # 无合法负样本的行不参与 loss，避免后续统计出现 inf。
        safe_hard_neg_sim = torch.where(
            valid_mask,
            hard_neg_sim,
            torch.zeros_like(hard_neg_sim),
        )

        per_anchor_loss = F.relu(
            self.margin
            - pos_sim
            + safe_hard_neg_sim
        )

        active_mask = (
            valid_mask
            & (per_anchor_loss > 0)
        )

        if valid_count > 0:
            loss = per_anchor_loss[valid_mask].mean()
            mean_pos_sim = pos_sim[valid_mask].mean()
            mean_hard_neg_sim = (
                safe_hard_neg_sim[valid_mask].mean()
            )
        else:
            loss = self._zero_loss(
                image_features,
                text_features,
            )
            mean_pos_sim = similarities.new_zeros(())
            mean_hard_neg_sim = similarities.new_zeros(())

        # 无合法负样本时 index 统一设为 -1，便于调试。
        hard_neg_index = torch.where(
            valid_mask,
            hard_neg_index,
            torch.full(
                (batch_size,),
                -1,
                dtype=torch.long,
                device=device,
            ),
        )

        return {
            "loss": loss,
            "valid_mask": valid_mask,
            "active_mask": active_mask,
            "valid_count": valid_count,
            "active_count": int(active_mask.sum().item()),
            "pos_sim": pos_sim,
            "hard_neg_sim": safe_hard_neg_sim,
            "hard_neg_index": hard_neg_index,
            "mean_pos_sim": mean_pos_sim,
            "mean_hard_neg_sim": mean_hard_neg_sim,
        }

    def forward(
        self,
        image_features,
        text_features,
        category_ids,
    ):
        self._check_inputs(
            image_features,
            text_features,
            category_ids,
        )

        category_ids = category_ids.to(
            image_features.device,
            dtype=torch.long,
        )

        # margin 始终定义在 raw cosine similarity 上，
        # 不乘 CLIP logit_scale。
        image_features = F.normalize(
            image_features,
            dim=-1,
        )
        text_features = F.normalize(
            text_features,
            dim=-1,
        )

        sim_i2t = image_features @ text_features.t()
        sim_t2i = sim_i2t.t()

        i2t = self._direction_loss(
            similarities=sim_i2t,
            category_ids=category_ids,
            image_features=image_features,
            text_features=text_features,
        )

        t2i = self._direction_loss(
            similarities=sim_t2i,
            category_ids=category_ids,
            image_features=image_features,
            text_features=text_features,
        )

        return {
            # 仅作为便利输出；正式训练建议分别乘 t2i/i2t 权重。
            "loss": 0.5 * (
                t2i["loss"]
                + i2t["loss"]
            ),
            "loss_t2i": t2i["loss"],
            "loss_i2t": i2t["loss"],
            "t2i_valid_count": t2i["valid_count"],
            "i2t_valid_count": i2t["valid_count"],
            "t2i_active_count": t2i["active_count"],
            "i2t_active_count": i2t["active_count"],
            "t2i_mean_pos_sim": t2i["mean_pos_sim"],
            "i2t_mean_pos_sim": i2t["mean_pos_sim"],
            "t2i_mean_hard_neg_sim": t2i[
                "mean_hard_neg_sim"
            ],
            "i2t_mean_hard_neg_sim": i2t[
                "mean_hard_neg_sim"
            ],
            # 以下字段主要供单测 / debug 使用。
            "t2i_hard_neg_index": t2i[
                "hard_neg_index"
            ],
            "i2t_hard_neg_index": i2t[
                "hard_neg_index"
            ],
            "t2i_hard_neg_sim": t2i[
                "hard_neg_sim"
            ],
            "i2t_hard_neg_sim": i2t[
                "hard_neg_sim"
            ],
            "t2i_valid_mask": t2i["valid_mask"],
            "i2t_valid_mask": i2t["valid_mask"],
            "t2i_active_mask": t2i["active_mask"],
            "i2t_active_mask": i2t["active_mask"],
            "similarity_i2t": sim_i2t,
        }
