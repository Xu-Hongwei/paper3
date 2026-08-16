import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CLIPBackbone
from .local_region import (
    sample_region_boxes,
    crop_regions,
    pool_region_patches,
)


class CLIPRetrieval(nn.Module):
    """
    Clean CLIP Retrieval。

    当前主线：
        1. Global Retrieval：直接使用 RSICD-finetuned CLIP；
        2. Frozen Teacher：训练开始时由 Student Vision 精确复制；
        3. Student Vision：仅解冻最后 N 个 Transformer Block；
        4. Local Self-Distillation：Region Student 对齐 Crop Teacher；
        5. Global Preservation：Student 全图语义对齐 Frozen Teacher。

    当前不使用：
        - Global Adapter
        - Local Head
        - Entity loss
    """

    def __init__(self, config):
        super().__init__()

        self.backbone = CLIPBackbone(
            model_name=config["backbone"],
            pretrained=config["pretrained"],
            pretrained_local_path=config.get("pretrained_local_path"),
            prefer_local_pretrained=config.get("prefer_local_pretrained", True),
        )

        self.freeze_backbone = bool(
            config.get("freeze_backbone", False)
        )

        # --------------------------------------------------
        # Local Self-Distillation 配置
        # --------------------------------------------------
        self.local_image_size = int(
            config.get("local_image_size", 224)
        )
        self.local_grid_size = int(
            config.get("local_grid_size", 7)
        )
        self.local_num_regions = int(
            config.get("local_num_regions", 2)
        )
        self.local_min_scale = float(
            config.get("local_min_scale", 0.20)
        )
        self.local_max_scale = float(
            config.get("local_max_scale", 0.60)
        )
        self.local_trainable_blocks = int(
            config.get("local_trainable_blocks", 0)
        )

        if self.local_num_regions <= 0:
            raise ValueError(
                "local_num_regions 必须 > 0"
            )

        if not (
            0
            < self.local_min_scale
            <= self.local_max_scale
            <= 1
        ):
            raise ValueError(
                "local scale 必须满足 "
                "0 < min_scale <= max_scale <= 1"
            )

        # --------------------------------------------------
        # 检查 Vision Transformer 结构
        # --------------------------------------------------
        visual = self.backbone.model.visual
        transformer = getattr(
            visual,
            "transformer",
            None,
        )
        resblocks = getattr(
            transformer,
            "resblocks",
            None,
        )

        if resblocks is None:
            raise RuntimeError(
                "Current CLIP visual tower does not expose "
                "transformer.resblocks."
            )

        self.num_visual_blocks = len(
            resblocks
        )

        if not (
            0
            <= self.local_trainable_blocks
            <= self.num_visual_blocks
        ):
            raise ValueError(
                "local_trainable_blocks 必须位于 "
                f"[0, {self.num_visual_blocks}]，"
                f"当前为 {self.local_trainable_blocks}"
            )

        # --------------------------------------------------
        # 独立 Frozen Teacher Vision
        #
        # train.py 在加载 RSICD CLIP baseline 后必须调用：
        #     model.sync_local_teacher()
        #
        # 之后 Teacher 永久冻结。
        # --------------------------------------------------
        self.local_teacher_visual = copy.deepcopy(
            self.backbone.model.visual
        )
        self._freeze_local_teacher()

        # 默认冻结整个 Backbone，
        # 再显式解冻 Student Vision 最后 N 个 Block。
        if self.local_trainable_blocks > 0:
            self._configure_partial_visual_student()
        elif self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _freeze_local_teacher(self):
        """Frozen Teacher 永久保持 eval 且不参与梯度。"""
        self.local_teacher_visual.eval()

        for param in self.local_teacher_visual.parameters():
            param.requires_grad = False

    def sync_local_teacher(self):
        """
        将 Teacher 同步为训练开始前的 Student Vision。

        调用时机：
            1. 创建模型；
            2. 加载 RSICD CLIP baseline；
            3. 调用 sync_local_teacher()；
            4. 开始 B1b 训练。
        """
        self.local_teacher_visual.load_state_dict(
            self.backbone.model.visual.state_dict(),
            strict=True,
        )
        self._freeze_local_teacher()

    def _configure_partial_visual_student(self):
        """
        冻结整个 CLIP，只解冻 Vision 最后 N 个 Transformer Block。

        例如 ViT-B/32 + local_trainable_blocks=3：
            Block 1~9   frozen
            Block 10~12 trainable
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        resblocks = (
            self.backbone.model.visual
            .transformer
            .resblocks
        )

        start = (
            self.num_visual_blocks
            - self.local_trainable_blocks
        )

        for block in resblocks[start:]:
            for param in block.parameters():
                param.requires_grad = True

    def get_local_student_parameter_names(self):
        """
        返回 B1b 唯一允许训练的 Student 参数名。

        Clean CLIP 主线不再支持 Local Head-only B1a。
        """
        if self.local_trainable_blocks <= 0:
            return []

        start = (
            self.num_visual_blocks
            - self.local_trainable_blocks
        )

        prefix = (
            "backbone.model.visual."
            "transformer.resblocks."
        )

        allowed_prefixes = tuple(
            f"{prefix}{index}."
            for index in range(
                start,
                self.num_visual_blocks,
            )
        )

        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith(
                allowed_prefixes
            )
        ]

    def train(self, mode=True):
        """
        Teacher 永远 eval。

        B1b 中：
            Backbone 整体保持 eval；
            仅最后 N 个可训练 Vision Block 设置为 train。
        """
        super().train(mode)

        self.local_teacher_visual.eval()

        if (
            self.freeze_backbone
            or self.local_trainable_blocks > 0
        ):
            self.backbone.eval()

        if (
            mode
            and self.local_trainable_blocks > 0
        ):
            resblocks = (
                self.backbone.model.visual
                .transformer
                .resblocks
            )

            start = (
                self.num_visual_blocks
                - self.local_trainable_blocks
            )

            for block in resblocks[start:]:
                block.train()

        return self

    @staticmethod
    def _pool_entity_spans(
        token_features,
        entity_spans,
        entity_sample_ids,
    ):
        """
        保留 contextualized Entity span pooling 接口，
        供后续 Structured Grounding 使用。

        当前 B1b 不调用该函数。
        """
        if entity_spans.shape[0] == 0:
            return token_features.new_empty(
                (
                    0,
                    token_features.shape[-1],
                )
            )

        starts = entity_spans[:, 0].long()
        ends = entity_spans[:, 1].long()
        sample_ids = (
            entity_sample_ids.long()
        )
        seq_len = token_features.shape[1]

        if (
            starts.min() < 0
            or ends.max() > seq_len
            or torch.any(ends <= starts)
            or sample_ids.min() < 0
            or sample_ids.max()
            >= token_features.shape[0]
        ):
            raise ValueError(
                "Entity span 或 entity_sample_ids 非法"
            )

        zero = token_features.new_zeros(
            (
                token_features.shape[0],
                1,
                token_features.shape[-1],
            )
        )

        prefix = torch.cat(
            (
                zero,
                token_features.cumsum(
                    dim=1
                ),
            ),
            dim=1,
        )

        pooled = (
            prefix[sample_ids, ends]
            - prefix[sample_ids, starts]
        )

        lengths = (
            (ends - starts)
            .to(token_features.dtype)
            .unsqueeze(1)
        )

        return pooled / lengths

    def _encode_teacher_images(
        self,
        images,
    ):
        """
        Frozen Teacher：
            任意 224×224 图像 -> normalized CLIP global visual feature。

        用于：
            1. Region crop teacher；
            2. Full-image global preservation teacher。
        """
        with torch.no_grad():
            teacher_features = (
                self.local_teacher_visual(
                    images
                )
            )

            teacher_features = F.normalize(
                teacher_features,
                dim=-1,
            )

        return teacher_features

    def _build_local_distillation(
        self,
        images,
        patch_features,
        region_boxes=None,
    ):
        """
        Local Self-Distillation。

        Teacher:
            Crop R
            -> Frozen Teacher Vision
            -> t_R^T

        Student:
            Full Image
            -> Student L12 Patches
            -> Region Pool
            -> r_R^S
        """
        if region_boxes is None:
            region_boxes = sample_region_boxes(
                batch_size=images.shape[0],
                image_size=self.local_image_size,
                num_regions=self.local_num_regions,
                min_scale=self.local_min_scale,
                max_scale=self.local_max_scale,
                device=images.device,
            )
        else:
            region_boxes = region_boxes.to(
                device=images.device,
                dtype=torch.float32,
            )

        crops = crop_regions(
            images,
            region_boxes,
            output_size=self.local_image_size,
        )

        teacher_features = (
            self._encode_teacher_images(
                crops
            )
        )

        student_features, _ = (
            pool_region_patches(
                patch_features,
                region_boxes,
                image_size=self.local_image_size,
                grid_size=self.local_grid_size,
            )
        )

        student_features = F.normalize(
            student_features,
            dim=-1,
        )

        return {
            "local_student_feat": student_features,
            "local_teacher_feat": teacher_features,
            "local_region_boxes": region_boxes,
        }

    @torch.no_grad()
    def encode_local_patches(
        self,
        images,
        normalize=True,
    ):
        """
        Grounding 诊断接口。

        直接输出 B1b Student Vision 的 L12 Patch，
        不经过 Adapter，也不经过 Local Head。
        """
        _, patch_features = (
            self.backbone.encode_image_with_patches(
                images,
                normalize=False,
            )
        )

        if normalize:
            patch_features = F.normalize(
                patch_features,
                dim=-1,
            )

        return patch_features

    def forward(
        self,
        images,
        captions,
        entity_spans=None,
        entity_sample_ids=None,
        entity_counts=None,
        local_distill=False,
        region_boxes=None,
    ):
        # ==================================================
        # B1b：Clean CLIP Local Self-Distillation
        # ==================================================
        if local_distill:
            if entity_spans is not None:
                raise ValueError(
                    "B1b 暂不与 Entity Grounding 同时启用。"
                )

            # Student full-image forward。
            image_features, patch_features = (
                self.backbone.encode_image_with_patches(
                    images,
                    normalize=False,
                )
            )

            # Student global representation：
            # 直接使用 raw CLIP visual embedding。
            student_global_features = F.normalize(
                image_features,
                dim=-1,
            )

            # Frozen Teacher full-image representation。
            teacher_global_features = (
                self._encode_teacher_images(
                    images
                )
            )

            # Global retrieval loss 当前仅用于监控。
            # Text Tower 完全冻结，无需保存反向图。
            with torch.no_grad():
                text_features = (
                    self.backbone.encode_text(
                        captions,
                        normalize=True,
                    )
                )

            outputs = {
                "image_feat": student_global_features.detach(),
                "text_feat": text_features,
                "logit_scale": self.backbone.logit_scale,
                "local_student_global_feat": student_global_features,
                "local_teacher_global_feat": teacher_global_features,
            }

            outputs.update(
                self._build_local_distillation(
                    images,
                    patch_features,
                    region_boxes=region_boxes,
                )
            )

            return outputs

        # ==================================================
        # Validation / Test：纯 CLIP Global Retrieval
        # ==================================================
        if entity_spans is None:
            image_features = (
                self.backbone.encode_image(
                    images,
                    normalize=True,
                )
            )

            text_features = (
                self.backbone.encode_text(
                    captions,
                    normalize=True,
                )
            )

            return {
                "image_feat": image_features,
                "text_feat": text_features,
                "logit_scale": self.backbone.logit_scale,
            }

        # ==================================================
        # 兼容接口：Clean CLIP Entity features
        #
        # 当前 B1b 不使用，仅为后续 Structured Grounding 保留。
        # 不包含旧 Adapter，也不实现旧 naive visual grounding loss。
        # ==================================================
        if entity_sample_ids is None:
            raise ValueError(
                "提供 entity_spans 时必须同时提供 "
                "entity_sample_ids。"
            )

        image_features, patch_features = (
            self.backbone.encode_image_with_patches(
                images,
                normalize=False,
            )
        )

        text_features, token_features = (
            self.backbone.encode_text_with_tokens(
                captions,
                normalize=False,
            )
        )

        device = token_features.device

        entity_spans = entity_spans.to(
            device=device,
            non_blocking=True,
        )
        entity_sample_ids = (
            entity_sample_ids.to(
                device=device,
                non_blocking=True,
            )
        )

        entity_features = (
            self._pool_entity_spans(
                token_features,
                entity_spans,
                entity_sample_ids,
            )
        )

        return {
            "image_feat": F.normalize(
                image_features,
                dim=-1,
            ),
            "text_feat": F.normalize(
                text_features,
                dim=-1,
            ),
            "patch_feat": F.normalize(
                patch_features,
                dim=-1,
            ),
            "entity_feat": F.normalize(
                entity_features,
                dim=-1,
            ),
            "entity_sample_ids": entity_sample_ids,
            "entity_counts": entity_counts,
            "logit_scale": self.backbone.logit_scale,
        }
