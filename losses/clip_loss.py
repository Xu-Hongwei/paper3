import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Multi-positive symmetric CLIP loss.

    同一 image_id 的样本互为正样本。
    当一个 anchor 对应多个正样本时，对这些正样本的 log-prob 等权平均。
    """

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        image_ids,
    ):
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("image/text features must have shape [B, D].")

        if image_features.shape != text_features.shape:
            raise ValueError(
                "image_features and text_features must have the same shape: "
                f"{image_features.shape} vs {text_features.shape}"
            )

        batch_size = image_features.shape[0]
        if image_ids.ndim != 1 or image_ids.shape[0] != batch_size:
            raise ValueError(
                f"image_ids must have shape [{batch_size}], "
                f"got {tuple(image_ids.shape)}"
            )

        image_ids = image_ids.to(image_features.device)

        # CLIP cosine similarity（输入特征已 L2 normalize）。
        logits_i2t = logit_scale * image_features @ text_features.t()
        logits_t2i = logits_i2t.t()

        # 同一 image_id 均视为正样本。
        positive_mask = (image_ids[:, None] == image_ids[None, :]).float()
        positive_targets = positive_mask / positive_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        loss_i2t = -(
            positive_targets * F.log_softmax(logits_i2t, dim=1)
        ).sum(dim=1).mean()

        loss_t2i = -(
            positive_targets.t() * F.log_softmax(logits_t2i, dim=1)
        ).sum(dim=1).mean()

        loss = 0.5 * (loss_i2t + loss_t2i)

        return {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logits_i2t": logits_i2t,
            "positive_mask": positive_mask,
        }
