import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Standard symmetric CLIP contrastive loss.

    Assumption:
        The positive image-text pairs are located on the
        diagonal of the batch similarity matrix.

        image_i <-> text_i
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
    ):
        """
        Args:
            image_features:
                Tensor [B, D], L2-normalized.

            text_features:
                Tensor [B, D], L2-normalized.

            logit_scale:
                Positive scalar, normally exp(CLIP.logit_scale).

        Returns:
            dict:
                loss
                loss_i2t
                loss_t2i
                logits_i2t
        """

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

        # ---------------------------------------------
        # Similarity matrix
        #
        # [B, D] @ [D, B] -> [B, B]
        # ---------------------------------------------

        logits_i2t = (
            logit_scale
            * image_features
            @ text_features.t()
        )

        logits_t2i = logits_i2t.t()

        # ---------------------------------------------
        # Diagonal positives:
        #
        # image 0 <-> text 0
        # image 1 <-> text 1
        # ...
        # ---------------------------------------------

        labels = torch.arange(
            batch_size,
            device=image_features.device,
        )

        # ---------------------------------------------
        # Image -> Text
        # ---------------------------------------------

        loss_i2t = F.cross_entropy(
            logits_i2t,
            labels,
        )

        # ---------------------------------------------
        # Text -> Image
        # ---------------------------------------------

        loss_t2i = F.cross_entropy(
            logits_t2i,
            labels,
        )

        # ---------------------------------------------
        # Symmetric CLIP loss
        # ---------------------------------------------

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
        }