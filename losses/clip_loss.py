import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Multi-positive symmetric CLIP contrastive loss.

    Samples sharing the same image_id are treated as
    positives rather than negatives.

    For multiple positives, their log-probabilities are
    averaged equally.

    Example:
        image_ids = [0, 0, 1, 1]

        means:
            image_0 <-> text_0, text_1
            image_1 <-> text_2, text_3
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        image_ids,
    ):
        """
        Args:
            image_features:
                Tensor [B, D], L2-normalized.

            text_features:
                Tensor [B, D], L2-normalized.

            logit_scale:
                Positive scalar, normally exp(CLIP.logit_scale).

            image_ids:
                Tensor [B].

                Samples with the same image_id are regarded
                as positive image-text pairs.

                Example:
                    [0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1]

                means each image has 5 captions.

        Returns:
            dict:
                loss
                loss_i2t
                loss_t2i
                logits_i2t
                positive_mask
        """

        # -------------------------------------------------
        # Input checks
        # -------------------------------------------------

        if image_features.ndim != 2:
            raise ValueError(
                "image_features must have shape [B, D]"
            )

        if text_features.ndim != 2:
            raise ValueError(
                "text_features must have shape [B, D]"
            )

        if image_features.shape != text_features.shape:
            raise ValueError(
                "image_features and text_features must "
                "have the same shape. "
                f"Got {image_features.shape} and "
                f"{text_features.shape}"
            )

        batch_size = image_features.shape[0]

        if image_ids.ndim != 1:
            raise ValueError(
                "image_ids must have shape [B]"
            )

        if image_ids.shape[0] != batch_size:
            raise ValueError(
                "image_ids must have the same batch size "
                f"as features. Got {image_ids.shape[0]} "
                f"and {batch_size}"
            )

        image_ids = image_ids.to(image_features.device)

        # -------------------------------------------------
        # Similarity matrix
        #
        # [B, D] @ [D, B] -> [B, B]
        # -------------------------------------------------

        logits_i2t = (
            logit_scale
            * image_features
            @ text_features.t()
        )

        logits_t2i = logits_i2t.t()

        # -------------------------------------------------
        # Multi-positive mask
        #
        # positive_mask[i, j] = 1
        # if sample i and sample j belong to the same image
        #
        # Example:
        #
        # image_ids = [0, 0, 1, 1]
        #
        # mask =
        #
        # 1 1 0 0
        # 1 1 0 0
        # 0 0 1 1
        # 0 0 1 1
        # -------------------------------------------------

        positive_mask = (
            image_ids[:, None]
            == image_ids[None, :]
        ).float()

        # -------------------------------------------------
        # Normalize positives
        #
        # Example:
        #
        # 1 1 0 0
        #
        # becomes
        #
        # 0.5 0.5 0 0
        #
        # Therefore each positive contributes equally.
        # -------------------------------------------------

        positive_targets = (
            positive_mask
            /
            positive_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
        )

        # -------------------------------------------------
        # Image -> Text
        #
        # log_softmax over all candidate texts
        # then average over all positives
        # -------------------------------------------------

        log_prob_i2t = F.log_softmax(
            logits_i2t,
            dim=1,
        )

        loss_i2t = -(
            positive_targets
            * log_prob_i2t
        ).sum(dim=1).mean()

        # -------------------------------------------------
        # Text -> Image
        #
        # Since the positive relationship is symmetric,
        # transpose the target matrix.
        # -------------------------------------------------

        log_prob_t2i = F.log_softmax(
            logits_t2i,
            dim=1,
        )

        loss_t2i = -(
            positive_targets.t()
            * log_prob_t2i
        ).sum(dim=1).mean()

        # -------------------------------------------------
        # Symmetric CLIP loss
        # -------------------------------------------------

        loss = (
            loss_i2t
            +
            loss_t2i
        ) / 2.0

        return {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logits_i2t": logits_i2t,
            "positive_mask": positive_mask,
        }