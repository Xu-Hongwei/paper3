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

    def encode_image_with_patches(
        self,
        images,
        normalize=True,
    ):
        """
        Encode images once and return both:

            global image features:
                [B, D]

            spatial patch features:
                [B, N_patch, D]

        For ViT-B/32 with 224x224 input:
            N_patch = 7 * 7 = 49
            D = 512

        The final Transformer block is taken through
        OpenCLIP's forward_intermediates() interface.

        Intermediate patch tokens are normalized by the
        visual tower's final LayerNorm, then explicitly
        projected through visual.proj into the CLIP joint
        embedding space.
        """

        visual = self.model.visual

        if not hasattr(
            visual,
            "forward_intermediates",
        ):
            raise RuntimeError(
                "Current visual encoder does not expose "
                "forward_intermediates()."
            )

        outputs = visual.forward_intermediates(
            images,
            indices=1,
            stop_early=False,
            normalize_intermediates=True,
            intermediates_only=False,
            output_fmt="NLC",
            output_extra_tokens=False,
        )

        if "image_features" not in outputs:
            raise RuntimeError(
                "forward_intermediates() did not return "
                "'image_features'."
            )

        intermediates = outputs.get(
            "image_intermediates"
        )

        if (
            not isinstance(intermediates, list)
            or len(intermediates) == 0
        ):
            raise RuntimeError(
                "forward_intermediates() did not return "
                "a valid image_intermediates list."
            )

        image_features = outputs[
            "image_features"
        ]

        patch_features = intermediates[-1]

        if patch_features.ndim != 3:
            raise RuntimeError(
                "Expected patch features in NLC format "
                "[B, N_patch, width], got "
                f"{tuple(patch_features.shape)}"
            )

        # --------------------------------------------------
        # Project patch tokens from visual width
        # (e.g. 768) into CLIP joint embedding dimension
        # (e.g. 512), matching global text/entity features.
        # --------------------------------------------------

        visual_proj = getattr(
            visual,
            "proj",
            None,
        )

        if visual_proj is not None:
            patch_features = (
                patch_features
                @ visual_proj
            )

        if normalize:
            image_features = F.normalize(
                image_features,
                dim=-1,
            )

            patch_features = F.normalize(
                patch_features,
                dim=-1,
            )

        return (
            image_features,
            patch_features,
        )

    def encode_image_intermediate_patches(
        self,
        images,
        layers=(6, 8, 10, 12),
        normalize=True,
    ):
        """
        一次 Vision Transformer 前向，同时返回多层 Patch 特征。

        Args:
            images:
                [B, C, H, W]

            layers:
                1-based Transformer layer numbers.
                ViT-B/32 共 12 层，默认取第 6/8/10/12 层。

            normalize:
                是否对最终全局特征和各层 Patch 特征做 L2 Normalize。

        Returns:
            image_features:
                [B, D]

            patch_features:
                dict[int, Tensor]
                例如：
                    {
                        6:  [B, 49, 512],
                        8:  [B, 49, 512],
                        10: [B, 49, 512],
                        12: [B, 49, 512],
                    }

        Notes:
            1. OpenCLIP 内部 block index 为 0-based，因此 layer 6 对应 index 5。
            2. 中间层 token 先经过 visual.ln_post，再通过 visual.proj
               映射到 CLIP joint embedding space。
            3. 该接口仅用于 Local/Patch 诊断，不改变现有
               encode_image_with_patches() 的行为。
        """
        visual = self.model.visual

        if not hasattr(visual, "forward_intermediates"):
            raise RuntimeError(
                "Current visual encoder does not expose "
                "forward_intermediates()."
            )

        if not isinstance(layers, (list, tuple)):
            raise TypeError("layers must be a list or tuple of integers.")
        if len(layers) == 0:
            raise ValueError("layers must not be empty.")
        if any(not isinstance(layer, int) for layer in layers):
            raise TypeError("Every layer number must be an integer.")
        if len(set(layers)) != len(layers):
            raise ValueError(
                f"Duplicate layer numbers are not allowed: {layers}"
            )

        transformer = getattr(visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None)
        if resblocks is None:
            raise RuntimeError(
                "Current visual encoder does not expose transformer.resblocks."
            )

        num_layers = len(resblocks)
        invalid_layers = [
            layer for layer in layers
            if layer < 1 or layer > num_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid visual Transformer layers {invalid_layers}; "
                f"valid range is [1, {num_layers}]."
            )

        # 用户接口采用 1-based layer number；
        # OpenCLIP forward_intermediates 使用 0-based block index。
        indices = [layer - 1 for layer in layers]

        outputs = visual.forward_intermediates(
            images,
            indices=indices,
            stop_early=False,
            normalize_intermediates=True,
            intermediates_only=False,
            output_fmt="NLC",
            output_extra_tokens=False,
        )

        image_features = outputs.get("image_features")
        intermediates = outputs.get("image_intermediates")

        if image_features is None:
            raise RuntimeError(
                "forward_intermediates() did not return 'image_features'."
            )

        if (
            not isinstance(intermediates, list)
            or len(intermediates) != len(layers)
        ):
            actual = 0 if intermediates is None else len(intermediates)
            raise RuntimeError(
                "Unexpected number of image intermediates: "
                f"expected {len(layers)}, got {actual}."
            )

        visual_proj = getattr(visual, "proj", None)
        patch_features = {}

        for layer, features in zip(layers, intermediates):
            if features.ndim != 3:
                raise RuntimeError(
                    f"Layer {layer}: expected NLC Patch features "
                    f"[B, N_patch, width], got {tuple(features.shape)}."
                )

            # 中间层 token 已通过 visual.ln_post；
            # 再映射到与 CLIP global/text 相同的 joint space。
            if visual_proj is not None:
                features = features @ visual_proj

            if normalize:
                features = F.normalize(
                    features,
                    dim=-1,
                )

            patch_features[layer] = features

        if normalize:
            image_features = F.normalize(
                image_features,
                dim=-1,
            )

        return image_features, patch_features

    def encode_text_with_tokens(self, captions, normalize=True):
        """
        Caption 只经过一次 Text Transformer，同时返回全局文本特征和 token 特征。

        Returns:
            text_features:  [B, D]
            token_features: [B, L, D]
        """
        if isinstance(captions, (list, tuple)):
            device = next(self.model.parameters()).device
            text_tokens = self.tokenize(captions, device=device)
        elif torch.is_tensor(captions):
            text_tokens = captions
        else:
            raise TypeError(
                "captions must be list[str], tuple[str], or torch.Tensor"
            )

        if not hasattr(self.model, "forward_intermediates"):
            raise RuntimeError(
                "Current OpenCLIP model does not expose forward_intermediates()."
            )

        outputs = self.model.forward_intermediates(
            text=text_tokens,
            text_indices=1,
            normalize=normalize,
            normalize_intermediates=True,
            intermediates_only=False,
        )

        text_features = outputs.get("text_features")
        intermediates = outputs.get("text_intermediates")

        if text_features is None:
            raise RuntimeError("forward_intermediates() did not return text_features.")
        if not isinstance(intermediates, list) or not intermediates:
            raise RuntimeError(
                "forward_intermediates() did not return text_intermediates."
            )

        # 最后一层 token 已经过 ln_final，再使用 CLIP 原文本投影映射到联合空间。
        token_features = intermediates[-1]
        text_projection = getattr(self.model, "text_projection", None)

        if text_projection is not None:
            if isinstance(text_projection, nn.Linear):
                token_features = text_projection(token_features)
            else:
                token_features = token_features @ text_projection

        if normalize:
            token_features = F.normalize(token_features, dim=-1)

        return text_features, token_features

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
