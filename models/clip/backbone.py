import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

# ----------------------------------------------------------------------
# Offline / local-pretrained support.
#
# `pretrained="openai"` for ViT-B-32 normally downloads weights from
# HuggingFace (timm repo), which fails when the network is unavailable.
# As a fallback we load the official OpenAI CLIP checkpoint cached by the
# openai/clip package (default: ~/.cache/clip/ViT-B-32.pt), which is the
# exact same weight set, into the native open_clip architecture.
#
# Local candidate resolution order:
#   1. `pretrained_local_path` (config)  -> expanded with ~;
#   2. env var OPEN_CLIP_PRETRAINED      -> path to a local checkpoint;
#   3. `pretrained` value is itself an existing file path;
#   4. default openai/clip cache file, only valid for ViT-B-32 family
#      with pretrained="openai".
# ----------------------------------------------------------------------
DEFAULT_LOCAL_OPENAI_VITB32 = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "clip",
    "ViT-B-32.pt",
)


def _resolve_local_checkpoint(model_name, pretrained, explicit_path):
    """Return the first existing local checkpoint candidate, or None."""

    candidates = []

    if explicit_path:
        candidates.append(str(Path(explicit_path).expanduser()))

    env_path = os.environ.get("OPEN_CLIP_PRETRAINED", "")
    if env_path:
        candidates.append(env_path)

    if isinstance(pretrained, str) and os.path.isfile(pretrained):
        candidates.append(pretrained)

    if (
        pretrained == "openai"
        and model_name.startswith("ViT-B-32")
    ):
        candidates.append(DEFAULT_LOCAL_OPENAI_VITB32)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    return None


def _load_local_checkpoint(model, checkpoint_path):
    """Load an OpenAI-CLIP-style checkpoint into an open_clip model.

    Supports:
        - OpenAI CLIP TorchScript archives (the .pt files cached by the
          openai/clip package, e.g. ~/.cache/clip/ViT-B-32.pt);
        - open_clip / plain state-dict checkpoints (.pt / .pth / .bin);
        - safetensors checkpoints (.safetensors).
    """

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        try:
            # OpenAI CLIP archives are TorchScript modules.
            ckpt = torch.jit.load(
                checkpoint_path,
                map_location="cpu",
            )
            state_dict = ckpt.state_dict()
        except Exception:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            state_dict = (
                ckpt.get("state_dict", ckpt)
                if isinstance(ckpt, dict)
                else ckpt
            )

    # Drop non-tensor metadata (e.g. input_resolution / vocab_size).
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if torch.is_tensor(v)
    }

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Local CLIP checkpoint is missing weights for: "
            f"{missing}"
        )

    if unexpected:
        print(
            "[CLIPBackbone] ignoring unexpected keys from "
            f"{checkpoint_path}: {unexpected}"
        )

    return model


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
        pretrained_local_path=None,
        prefer_local_pretrained=True,
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained

        local_path = _resolve_local_checkpoint(
            model_name,
            pretrained,
            pretrained_local_path,
        )
        self.pretrained_local_path = local_path

        def _build_without_weights():
            previous_disable_level = (
                logging.root.manager.disable
            )
            logging.disable(logging.WARNING)
            try:
                return open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=None,
                )
            finally:
                logging.disable(
                    previous_disable_level
                )

        if (
            local_path is not None
            and prefer_local_pretrained
        ):
            # Local-first: no network access required.
            (
                self.model,
                self.preprocess_train,
                self.preprocess_val,
            ) = _build_without_weights()
            _load_local_checkpoint(
                self.model,
                local_path,
            )
            print(
                "[CLIPBackbone] loaded pretrained weights from "
                f"local file: {local_path}"
            )
        else:
            try:
                (
                    self.model,
                    self.preprocess_train,
                    self.preprocess_val,
                ) = open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=pretrained,
                )
            except Exception as exc:
                # Network unavailable: fall back to a local checkpoint
                # if one exists, otherwise re-raise with guidance.
                if local_path is None:
                    raise RuntimeError(
                        "Failed to load pretrained CLIP weights "
                        f"({exc}). To run offline, set "
                        "`pretrained_local_path` in the config or the "
                        "env var OPEN_CLIP_PRETRAINED to a local "
                        "checkpoint file."
                    ) from exc
                (
                    self.model,
                    self.preprocess_train,
                    self.preprocess_val,
                ) = _build_without_weights()
                _load_local_checkpoint(
                    self.model,
                    local_path,
                )
                print(
                    "[CLIPBackbone] download failed; loaded weights "
                    f"from local file: {local_path}"
                )

        # --------------------------------------------------
        # CLIP tokenizer
        # --------------------------------------------------
        self.tokenizer = open_clip.get_tokenizer(
            model_name
        )

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
