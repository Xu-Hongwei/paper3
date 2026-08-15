import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import ResidualAdapter
from .backbone import CLIPBackbone


class CLIPRetrieval(nn.Module):
    """CLIP + 非对称残差 Adapter + Entity Grounding。"""

    def __init__(self, config):
        super().__init__()
        self.backbone = CLIPBackbone(
            model_name=config["backbone"],
            pretrained=config["pretrained"],
            pretrained_local_path=config.get("pretrained_local_path"),
            prefer_local_pretrained=config.get("prefer_local_pretrained", True),
        )

        embed_dim = int(config.get("embed_dim", 512))
        self.visual_adapter = ResidualAdapter(
            dim=embed_dim,
            bottleneck_dim=int(config.get("visual_adapter_dim", 128)),
        )
        self.text_adapter = ResidualAdapter(
            dim=embed_dim,
            bottleneck_dim=int(config.get("text_adapter_dim", 64)),
        )

        self.grounding_temperature = float(
            config.get("grounding_temperature", 0.02)
        )
        if self.grounding_temperature <= 0:
            raise ValueError("grounding_temperature 必须 > 0")

        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _pool_entity_spans(token_features, entity_spans, entity_sample_ids):
        """用前缀和完成变长 Entity span mean pooling。"""
        if entity_spans.shape[0] == 0:
            return token_features.new_empty((0, token_features.shape[-1]))

        starts = entity_spans[:, 0].long()
        ends = entity_spans[:, 1].long()
        sample_ids = entity_sample_ids.long()
        seq_len = token_features.shape[1]

        if (
            starts.min() < 0
            or ends.max() > seq_len
            or torch.any(ends <= starts)
            or sample_ids.min() < 0
            or sample_ids.max() >= token_features.shape[0]
        ):
            raise ValueError("Entity span 或 entity_sample_ids 非法")

        zero = token_features.new_zeros(
            (token_features.shape[0], 1, token_features.shape[-1])
        )
        prefix = torch.cat((zero, token_features.cumsum(dim=1)), dim=1)
        pooled = prefix[sample_ids, ends] - prefix[sample_ids, starts]
        lengths = (ends - starts).to(token_features.dtype).unsqueeze(1)
        return pooled / lengths

    def _ground_entities(self, entity_features, patch_features, entity_counts):
        """按图计算 Entity-Patch Grounding，避免复制整批 patch。"""
        if entity_features.shape[0] == 0:
            return entity_features.new_empty((0, entity_features.shape[-1]))

        counts = entity_counts.tolist() if torch.is_tensor(entity_counts) else list(entity_counts)
        if len(counts) != patch_features.shape[0]:
            raise ValueError("entity_counts 与图像 batch 不匹配")
        if sum(counts) != entity_features.shape[0]:
            raise ValueError("entity_counts 与 Entity 数量不匹配")

        visual_entities = []
        offset = 0

        for sample_index, count in enumerate(counts):
            if count <= 0:
                continue

            entities = entity_features[offset:offset + count]
            patches = patch_features[sample_index]
            similarity = entities @ patches.transpose(0, 1)
            weights = F.softmax(
                similarity / self.grounding_temperature,
                dim=-1,
            )
            visual_entities.append(weights @ patches)
            offset += count

        return F.normalize(torch.cat(visual_entities, dim=0), dim=-1)

    def forward(
        self,
        images,
        captions,
        entity_spans=None,
        entity_sample_ids=None,
        entity_counts=None,
    ):
        # 验证/测试只走全局双塔：raw CLIP -> residual Adapter -> normalize。
        if entity_spans is None:
            image_features = self.backbone.encode_image(images, normalize=False)
            text_features = self.backbone.encode_text(captions, normalize=False)

            image_features = F.normalize(
                image_features + self.visual_adapter(image_features),
                dim=-1,
            )
            text_features = F.normalize(
                text_features + self.text_adapter(text_features),
                dim=-1,
            )

            return {
                "image_feat": image_features,
                "text_feat": text_features,
                "logit_scale": self.backbone.logit_scale,
            }

        if entity_sample_ids is None or entity_counts is None:
            raise ValueError(
                "entity_spans、entity_sample_ids、entity_counts 必须同时提供"
            )

        # Backbone 每个模态只前向一次，先保留未 L2 归一化特征。
        image_features, patch_features = (
            self.backbone.encode_image_with_patches(images, normalize=False)
        )
        text_features, token_features = (
            self.backbone.encode_text_with_tokens(captions, normalize=False)
        )

        device = token_features.device
        entity_spans = entity_spans.to(device=device, non_blocking=True)
        entity_sample_ids = entity_sample_ids.to(device=device, non_blocking=True)

        entity_features = self._pool_entity_spans(
            token_features,
            entity_spans,
            entity_sample_ids,
        )

        # Global 与 Local 共用同一模态 Adapter。
        image_features = F.normalize(
            image_features + self.visual_adapter(image_features),
            dim=-1,
        )
        patch_features = F.normalize(
            patch_features + self.visual_adapter(patch_features),
            dim=-1,
        )
        text_features = F.normalize(
            text_features + self.text_adapter(text_features),
            dim=-1,
        )
        entity_features = F.normalize(
            entity_features + self.text_adapter(entity_features),
            dim=-1,
        )

        visual_entity_features = self._ground_entities(
            entity_features,
            patch_features,
            entity_counts,
        )

        return {
            "image_feat": image_features,
            "text_feat": text_features,
            "entity_feat": entity_features,
            "visual_entity_feat": visual_entity_features,
            "entity_sample_ids": entity_sample_ids,
            "entity_counts": entity_counts,
            "logit_scale": self.backbone.logit_scale,
        }
