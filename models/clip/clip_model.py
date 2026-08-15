import torch.nn as nn

from .backbone import CLIPBackbone


class CLIPRetrieval(nn.Module):
    """
    Vanilla CLIP retrieval model.
    """

    def __init__(self, config):
        super().__init__()

        self.backbone = CLIPBackbone(
            model_name=config["backbone"],
            pretrained=config["pretrained"],
            pretrained_local_path=config.get(
                "pretrained_local_path",
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

        image_features = (
            self.backbone.encode_image(
                images
            )
        )

        text_features = (
            self.backbone.encode_text(
                captions
            )
        )

        return {
            "image_feat": image_features,
            "text_feat": text_features,
            "logit_scale": self.backbone.logit_scale,
        }
