import json
from pathlib import Path


FINAL_SEMANTIC_KEYS = (
    "sanitized_structured_semantics",
    "final_structured_semantics",
    "structured_semantics",
    "sanitized",
    "semantics",
)


class StructuredSemanticsReader:
    """读取冻结 EAR，并按原始训练 pair_index 提供 Entity / Attribute / Relation。"""

    def __init__(self, path):
        self.path = str(path)
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples")
        if not isinstance(samples, list):
            raise ValueError("EAR 文件缺少有效 samples 列表。")

        self.samples = samples
        self.pair_to_sample = {}
        for sample_index, sample in enumerate(samples):
            for pair_index in self._source_indices(sample):
                if pair_index in self.pair_to_sample:
                    raise ValueError(f"pair_index={pair_index} 被重复映射。")
                self.pair_to_sample[pair_index] = sample_index

    @staticmethod
    def _source_indices(sample):
        value = sample.get("source_indices")
        if isinstance(value, list) and value:
            return [int(x) for x in value]

        value = sample.get("source_index")
        if isinstance(value, int):
            return [value]

        raise ValueError("EAR sample 缺少 source_indices/source_index。")

    @staticmethod
    def _semantics(sample):
        for key in FINAL_SEMANTIC_KEYS:
            value = sample.get(key)
            if isinstance(value, dict) and (
                "entities" in value
                or "attributes" in value
                or "relations" in value
            ):
                return value

        if any(key in sample for key in ("entities", "attributes", "relations")):
            return {
                key: sample[key]
                for key in ("entities", "attributes", "relations")
                if key in sample
            }

        raise ValueError("找不到冻结后的 Structured Semantics。")

    @staticmethod
    def _attributes(entity):
        result = []
        for attribute in entity.get("attributes") or []:
            if isinstance(attribute, str):
                value = attribute.strip()
                if value:
                    result.append({"type": "unknown", "value": value})
                continue

            if not isinstance(attribute, dict):
                continue

            attr_type = str(attribute.get("type", "unknown")).strip().lower()
            value = str(
                attribute.get("value", attribute.get("text", ""))
            ).strip()
            if value:
                result.append({"type": attr_type or "unknown", "value": value})

        return result

    def get_by_pair(self, pair_index):
        sample_index = self.pair_to_sample.get(int(pair_index))
        if sample_index is None:
            raise KeyError(f"EAR 未覆盖 pair_index={pair_index}")

        sample = self.samples[sample_index]
        semantics = self._semantics(sample)

        entities = []
        for entity in semantics.get("entities") or []:
            if not isinstance(entity, dict):
                continue

            text = entity.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            entities.append(
                {
                    "id": str(entity.get("id", "")).strip(),
                    "text": text.strip(),
                    "attributes": self._attributes(entity),
                }
            )

        return {
            "sample_index": sample_index,
            "caption": str(sample.get("caption", "")).strip(),
            "entities": entities,
            "relations": list(semantics.get("relations") or []),
        }

    def __len__(self):
        return len(self.samples)
