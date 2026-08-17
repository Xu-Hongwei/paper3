import logging
import os
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


# OpenAI CLIP 官方 ViT-B/32 的默认本地缓存位置。
DEFAULT_LOCAL_OPENAI_VITB32 = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "clip",
    "ViT-B-32.pt",
)


def _resolve_local_checkpoint(model_name, pretrained, explicit_path=None):
    """按优先级查找可用的本地 CLIP 权重。"""
    candidates = []

    if explicit_path:
        candidates.append(str(Path(explicit_path).expanduser()))

    env_path = os.environ.get("OPEN_CLIP_PRETRAINED")
    if env_path:
        candidates.append(env_path)

    if isinstance(pretrained, str) and os.path.isfile(pretrained):
        candidates.append(pretrained)

    if pretrained == "openai" and model_name.startswith("ViT-B-32"):
        candidates.append(DEFAULT_LOCAL_OPENAI_VITB32)

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def _load_local_checkpoint(model, checkpoint_path):
    """将本地 OpenAI CLIP / OpenCLIP 权重加载到当前模型。"""
    checkpoint_path = str(checkpoint_path)

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        try:
            # OpenAI CLIP 官方 .pt 通常是 TorchScript archive。
            checkpoint = torch.jit.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.state_dict()
        except Exception:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )

    # 去掉 input_resolution、vocab_size 等非 Tensor 元数据。
    state_dict = {
        name: value
        for name, value in state_dict.items()
        if torch.is_tensor(value)
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

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
    RSITR 使用的 CLIP Backbone。

    当前保留三类接口：
        1. Global image/text encoding：Clean CLIP 主线；
        2. Final patch features：后续局部诊断；
        3. Token features：后续 Entity span / structured semantics 使用。

    不包含 loss、optimizer、Teacher/Student、Adapter、Prototype 或 OT。
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

        def build_without_weights():
            # OpenCLIP 在 pretrained=None 时可能输出无关 warning，这里局部屏蔽。
            previous_level = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                return open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=None,
                )
            finally:
                logging.disable(previous_level)

        if local_path and prefer_local_pretrained:
            self.model, self.preprocess_train, self.preprocess_val = (
                build_without_weights()
            )
            _load_local_checkpoint(self.model, local_path)
            print(
                "[CLIPBackbone] loaded pretrained weights from "
                f"local file: {local_path}"
            )
        else:
            try:
                self.model, self.preprocess_train, self.preprocess_val = (
                    open_clip.create_model_and_transforms(
                        model_name=model_name,
                        pretrained=pretrained,
                    )
                )
            except Exception as exc:
                if local_path is None:
                    raise RuntimeError(
                        "Failed to load pretrained CLIP weights "
                        f"({exc}). Set `pretrained_local_path` or "
                        "OPEN_CLIP_PRETRAINED for offline use."
                    ) from exc

                self.model, self.preprocess_train, self.preprocess_val = (
                    build_without_weights()
                )
                _load_local_checkpoint(self.model, local_path)
                print(
                    "[CLIPBackbone] download failed; loaded weights "
                    f"from local file: {local_path}"
                )

        self.tokenizer = open_clip.get_tokenizer(model_name)

    def tokenize(self, captions, device=None):
        """将文本转换为 CLIP token ids。"""
        tokens = self.tokenizer(captions)
        return tokens.to(device) if device is not None else tokens

    def encode_image(self, images, normalize=True):
        """提取全局图像特征。"""
        features = self.model.encode_image(images)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, captions, normalize=True):
        """提取全局文本特征，支持原始字符串或已 tokenized Tensor。"""
        if isinstance(captions, (list, tuple)):
            device = next(self.model.parameters()).device
            tokens = self.tokenize(captions, device=device)
        elif torch.is_tensor(captions):
            tokens = captions
        else:
            raise TypeError(
                "captions must be list[str], tuple[str], or torch.Tensor"
            )

        features = self.model.encode_text(tokens)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_image_with_patches(self, images, normalize=True):
        """
        一次 Vision 前向，同时返回全局特征和最终层 Patch 特征。

        ViT-B/32 + 224×224 时：
            image_features: [B, 512]
            patch_features: [B, 49, 512]

        Patch token 会经过 visual.proj，映射到 CLIP joint space，
        便于后续和文本 / Entity 特征直接比较。
        """
        visual = self.model.visual

        if not hasattr(visual, "forward_intermediates"):
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

        image_features = outputs.get("image_features")
        intermediates = outputs.get("image_intermediates")

        if image_features is None:
            raise RuntimeError(
                "forward_intermediates() did not return image_features."
            )
        if not isinstance(intermediates, list) or not intermediates:
            raise RuntimeError(
                "forward_intermediates() did not return image_intermediates."
            )

        patch_features = intermediates[-1]
        if patch_features.ndim != 3:
            raise RuntimeError(
                "Expected patch features [B, N_patch, width], got "
                f"{tuple(patch_features.shape)}."
            )

        visual_proj = getattr(visual, "proj", None)
        if visual_proj is not None:
            patch_features = patch_features @ visual_proj

        if normalize:
            image_features = F.normalize(image_features, dim=-1)
            patch_features = F.normalize(patch_features, dim=-1)

        return image_features, patch_features

    def encode_text_with_tokens(self, captions, normalize=True):
        """
        一次 Text Transformer 前向，同时返回全局文本特征和最终层 token 特征。

        Returns:
            text_features:  [B, D]
            token_features: [B, L, D]
        """
        if isinstance(captions, (list, tuple)):
            device = next(self.model.parameters()).device
            tokens = self.tokenize(captions, device=device)
        elif torch.is_tensor(captions):
            tokens = captions
        else:
            raise TypeError(
                "captions must be list[str], tuple[str], or torch.Tensor"
            )

        if not hasattr(self.model, "forward_intermediates"):
            raise RuntimeError(
                "Current OpenCLIP model does not expose "
                "forward_intermediates()."
            )

        outputs = self.model.forward_intermediates(
            text=tokens,
            text_indices=1,
            normalize=normalize,
            normalize_intermediates=True,
            intermediates_only=False,
        )

        text_features = outputs.get("text_features")
        intermediates = outputs.get("text_intermediates")

        if text_features is None:
            raise RuntimeError(
                "forward_intermediates() did not return text_features."
            )
        if not isinstance(intermediates, list) or not intermediates:
            raise RuntimeError(
                "forward_intermediates() did not return text_intermediates."
            )

        # 最终 token 通过原 CLIP text_projection 进入 joint space。
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

    @property
    def logit_scale(self):
        """CLIP 可学习的相似度缩放系数。"""
        return self.model.logit_scale.exp()
