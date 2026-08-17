import argparse
import json
from pathlib import Path

import open_clip
import torch

from datasets.utils import pre_caption


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("EAR 顶层必须是 JSON object。")

    return data


def get_samples(data):
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("EAR 文件缺少有效的 samples 列表。")
    return samples


def get_caption(sample):
    for key in ("caption", "text", "normalized_caption", "raw_caption", "sentence"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("EAR sample 中找不到 caption。")


def get_source_indices(sample):
    value = sample.get("source_indices")
    if isinstance(value, list) and value:
        if not all(isinstance(x, int) and x >= 0 for x in value):
            raise ValueError("source_indices 必须是非负整数列表。")
        return sorted(set(value))

    value = sample.get("source_index")
    if isinstance(value, int) and value >= 0:
        return [value]

    raise ValueError("EAR sample 缺少 source_indices/source_index。")


def get_entities(sample):
    """优先读取 sanitizer 后的最终 Entity，不做二次语义修改。"""
    semantics = None

    for key in (
        "sanitized_structured_semantics",
        "final_structured_semantics",
        "structured_semantics",
        "sanitized",
        "semantics",
    ):
        value = sample.get(key)
        if isinstance(value, dict) and isinstance(value.get("entities"), list):
            semantics = value
            break

    if semantics is None and isinstance(sample.get("entities"), list):
        semantics = {"entities": sample["entities"]}

    if semantics is None:
        raise ValueError("找不到最终 EAR entities。")

    entities = []
    for entity in semantics["entities"]:
        if isinstance(entity, str):
            text = entity
        elif isinstance(entity, dict):
            text = entity.get("text", "")
        else:
            continue

        if isinstance(text, str) and text.strip():
            entities.append(text.strip())

    return entities


def get_special_token_ids(tokenizer):
    """用空字符串推断 SOT / EOT，避免写死 token id。"""
    tokens = tokenizer([""])[0].tolist()
    nonzero = [token for token in tokens if token != 0]

    if len(nonzero) < 2:
        raise RuntimeError("无法推断 CLIP SOT/EOT token id。")

    return int(nonzero[0]), int(nonzero[1])


def get_content_tokens(tokenizer, text, sot_id, eot_id):
    tokens = tokenizer([text])[0].tolist()

    try:
        sot_pos = tokens.index(sot_id)
        eot_pos = tokens.index(eot_id, sot_pos + 1)
    except ValueError as exc:
        raise RuntimeError(f"文本中找不到 SOT/EOT: {text}") from exc

    positions = list(range(sot_pos + 1, eot_pos))
    ids = [tokens[pos] for pos in positions]
    return ids, positions


def find_unique_span(caption_ids, caption_positions, entity_ids):
    if not entity_ids or len(entity_ids) > len(caption_ids):
        return "unmatched", None

    width = len(entity_ids)
    starts = [
        start
        for start in range(len(caption_ids) - width + 1)
        if caption_ids[start:start + width] == entity_ids
    ]

    if not starts:
        return "unmatched", None
    if len(starts) > 1:
        return "multiple", None

    start = starts[0]
    clip_start = caption_positions[start]
    clip_end = caption_positions[start + width - 1] + 1
    return "unique", (clip_start, clip_end)


def build_compact_index(ear_data, model_name, max_words, expected_pairs):
    """
    构建 Entity 索引 v2。

    同时保存：
        1. 每条 caption 的全部 EAR Entity 文本；
        2. 可唯一匹配到 caption token 的 Entity spans。

    这样训练热路径仍以整数 Tensor 为主，
    后续需要时也能直接恢复每条描述中的 Entity 字符串。
    """
    samples = get_samples(ear_data)
    tokenizer = open_clip.get_tokenizer(model_name)
    sot_id, eot_id = get_special_token_ids(tokenizer)

    pair_assignment = {}

    # 有效 token-span 索引。
    semantic_offsets = [0]
    span_start = []
    span_end = []
    span_entity_ids = []

    # 全量 Entity 文本索引。
    entity_vocab = []
    entity_to_id = {}
    semantic_entity_offsets = [0]
    semantic_entity_ids = []

    total_entities = 0
    unique_entities = 0
    multiple_entities = 0
    unmatched_entities = 0
    invalid_entities = 0
    max_pair_index = -1

    for semantic_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"EAR sample {semantic_index} 不是字典。")

        caption = pre_caption(get_caption(sample), max_words)
        caption_ids, caption_positions = get_content_tokens(
            tokenizer, caption, sot_id, eot_id
        )

        entities = get_entities(sample)

        for entity in entities:
            total_entities += 1

            # 先完整保存 Entity 文本，不因 span 匹配失败而丢失。
            if entity not in entity_to_id:
                entity_to_id[entity] = len(entity_vocab)
                entity_vocab.append(entity)

            entity_id = entity_to_id[entity]
            semantic_entity_ids.append(entity_id)

            try:
                entity_clean = pre_caption(entity, 100000)
                entity_ids, _ = get_content_tokens(
                    tokenizer, entity_clean, sot_id, eot_id
                )
            except ValueError:
                invalid_entities += 1
                continue

            status, span = find_unique_span(
                caption_ids,
                caption_positions,
                entity_ids,
            )

            if status == "unique":
                unique_entities += 1
                span_start.append(span[0])
                span_end.append(span[1])
                span_entity_ids.append(entity_id)
            elif status == "multiple":
                multiple_entities += 1
            else:
                unmatched_entities += 1

        semantic_offsets.append(len(span_start))
        semantic_entity_offsets.append(len(semantic_entity_ids))

        for pair_index in get_source_indices(sample):
            if pair_index in pair_assignment:
                raise ValueError(
                    f"pair_index={pair_index} 被重复映射到多个 semantic sample。"
                )

            pair_assignment[pair_index] = semantic_index
            max_pair_index = max(max_pair_index, pair_index)

    num_pairs = expected_pairs if expected_pairs is not None else max_pair_index + 1

    if max_pair_index >= num_pairs:
        raise ValueError(
            f"source index 越界: max={max_pair_index}, expected_pairs={num_pairs}"
        )

    pair_to_semantic = [-1] * num_pairs
    for pair_index, semantic_index in pair_assignment.items():
        pair_to_semantic[pair_index] = semantic_index

    missing = [i for i, semantic_index in enumerate(pair_to_semantic)
               if semantic_index < 0]
    if missing:
        raise ValueError(
            f"EAR 未覆盖全部训练 pair，共缺失 {len(missing)} 个，"
            f"前 20 个: {missing[:20]}"
        )

    return {
        "metadata": {
            "format": "entity_span_index_v2",
            "model_name": model_name,
            "max_words": max_words,
            "span_rule": "unique_exact_clip_token_span_only",
            "span_format": "[start, end), CLIP token absolute position",
        },
        "statistics": {
            "num_semantic_samples": len(samples),
            "num_original_pairs": num_pairs,
            "total_entities": total_entities,
            "entity_vocab_size": len(entity_vocab),
            "valid_unique_entities": unique_entities,
            "multiple_entities_skipped": multiple_entities,
            "unmatched_entities_skipped": unmatched_entities,
            "invalid_entities_skipped": invalid_entities,
            "valid_entity_coverage": unique_entities / max(total_entities, 1),
        },
        "pair_to_semantic": torch.tensor(pair_to_semantic, dtype=torch.int32),

        # 每个 semantic sample 的有效 span。
        "semantic_offsets": torch.tensor(semantic_offsets, dtype=torch.int32),
        "span_start": torch.tensor(span_start, dtype=torch.int16),
        "span_end": torch.tensor(span_end, dtype=torch.int16),
        "span_entity_ids": torch.tensor(span_entity_ids, dtype=torch.int32),

        # 每个 semantic sample 的完整 Entity 文本。
        "entity_vocab": entity_vocab,
        "semantic_entity_offsets": torch.tensor(
            semantic_entity_offsets,
            dtype=torch.int32,
        ),
        "semantic_entity_ids": torch.tensor(
            semantic_entity_ids,
            dtype=torch.int32,
        ),
    }


def print_report(result):
    stats = result["statistics"]

    print("\n" + "=" * 72)
    print("Compact Entity Index v2")
    print("=" * 72)
    print(f"Semantic captions       : {stats['num_semantic_samples']}")
    print(f"Original train pairs    : {stats['num_original_pairs']}")
    print(f"Total EAR entities      : {stats['total_entities']}")
    print(f"Entity vocab size       : {stats['entity_vocab_size']}")
    print(f"Valid unique spans      : {stats['valid_unique_entities']}")
    print(f"Multiple skipped        : {stats['multiple_entities_skipped']}")
    print(f"Unmatched skipped       : {stats['unmatched_entities_skipped']}")
    print(f"Invalid skipped         : {stats['invalid_entities_skipped']}")
    print(f"Valid coverage          : {stats['valid_entity_coverage'] * 100:.4f}%")
    print("=" * 72)


def run(input_path, output_path, model_name, max_words, expected_pairs):
    ear_data = load_json(input_path)
    result = build_compact_index(
        ear_data=ear_data,
        model_name=model_name,
        max_words=max_words,
        expected_pairs=expected_pairs,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)

    print_report(result)
    print(f"\nSaved: {output}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="从冻结 EAR 构建 Entity 文本 + token-span 紧凑索引。"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32-quickgelu",
    )
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=0,
        help="原始训练 pair 数；0 表示根据 source_indices 推断。",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    expected_pairs = args.expected_pairs if args.expected_pairs > 0 else None

    run(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        max_words=args.max_words,
        expected_pairs=expected_pairs,
    )


if __name__ == "__main__":
    main()
