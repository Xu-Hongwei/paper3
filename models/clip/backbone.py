import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip


class CLIPBackbone(nn.Module):
    """
    CLIP backbone for remote sensing image-text retrieval.

    Responsibilities:
        1. Load pretrained CLIP.
        2. Encode images.
        3. Tokenize and encode captions.
        4. Return normalized global embeddings.

    This module does NOT contain:
        - contrastive loss
        - optimizer
        - retrieval metrics
        - prototype / OT / entity modules
    """

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
    ):
        super().__init__()

        # --------------------------------------------------
        # Load pretrained CLIP and its official transforms
        # --------------------------------------------------
        (
            self.model,
            self.preprocess_train,
            self.preprocess_val,
        ) = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        # --------------------------------------------------
        # CLIP tokenizer
        # --------------------------------------------------
        self.tokenizer = open_clip.get_tokenizer(
            model_name
        )

        self.model_name = model_name
        self.pretrained = pretrained

    def tokenize(self, captions, device=None):
        """
        Convert raw captions to CLIP token ids.

        Args:
            captions:
                list[str] / tuple[str]

            device:
                target torch device

        Returns:
            token tensor
        """

        tokens = self.tokenizer(captions)

        if device is not None:
            tokens = tokens.to(device)

        return tokens

    def encode_image(
        self,
        images,
        normalize=True,
    ):
        """
        Encode images into CLIP global image embeddings.
        """

        image_features = self.model.encode_image(
            images
        )

        if normalize:
            image_features = F.normalize(
                image_features,
                dim=-1,
            )

        return image_features

    def encode_text(
        self,
        captions,
        normalize=True,
    ):
        """
        Encode raw captions or already-tokenized text.

        captions can be:
            list[str]
            tuple[str]
            Tensor
        """

        if isinstance(
            captions,
            (list, tuple)
        ):
            device = next(
                self.model.parameters()
            ).device

            text_tokens = self.tokenize(
                captions,
                device=device,
            )

        elif torch.is_tensor(captions):
            text_tokens = captions

        else:
            raise TypeError(
                "captions must be list[str], "
                "tuple[str], or torch.Tensor"
            )

        text_features = self.model.encode_text(
            text_tokens
        )

        if normalize:
            text_features = F.normalize(
                text_features,
                dim=-1,
            )

        return text_features

    @property
    def logit_scale(self):
        """
        CLIP learnable similarity scale.
        """

        return self.model.logit_scale.exp()