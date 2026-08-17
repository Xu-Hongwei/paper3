import torch.nn as nn

from .backbone import CLIPBackbone


class CLIPRetrieval(nn.Module):
    """
    Clean CLIP retrieval model for RSITR.

    Responsibilities:
        1. Build the CLIP backbone.
        2. Encode the full image into a normalized global visual feature.
        3. Encode the full caption into a normalized global text feature.
        4. Return the learnable CLIP similarity scale.

    This module intentionally does NOT contain:
        - Frozen Teacher / Student
        - Local Self-Distillation
        - Global Preservation
        - Local Patch Head
        - Adapter
        - Region pooling
        - Entity span pooling
        - Entity grounding loss
        - Prototype / OT modules
    """

    def __init__(self, config):
        super().__init__()

        self.backbone = CLIPBackbone(
            model_name=config["backbone"],
            pretrained=config["pretrained"],
            pretrained_local_path=config.get(
                "pretrained_local_path"
            ),
            prefer_local_pretrained=config.get(
                "prefer_local_pretrained",
                True,
            ),
        )

    def forward(
        self,
        images,
        captions,
    ):
        """
        Standard global CLIP forward.

        Args:
            images:
                Tensor [B, 3, H, W]

            captions:
                list[str] / tuple[str] / token tensor

        Returns:
            dict:
                image_feat: [B, D], L2-normalized
                text_feat:  [B, D], L2-normalized
                logit_scale: positive CLIP similarity scale
        """

        image_features = self.backbone.encode_image(
            images,
            normalize=True,
        )

        text_features = self.backbone.encode_text(
            captions,
            normalize=True,
        )

        return {
            "image_feat": image_features,
            "text_feat": text_features,
            "logit_scale": self.backbone.logit_scale,
        }
