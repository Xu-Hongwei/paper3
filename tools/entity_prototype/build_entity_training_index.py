import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import open_clip
import torch

from datasets.utils import pre_caption


def load_json(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("EAR 顶层必须是 JSON object。")
    return data


def get_samples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("EAR 文件缺少有效的 samples 列表。")
    return samples


def get_caption(sample: Dict[str, Any]) -> str:
    for key in ("caption", "text", "normalized_caption", "raw_caption", "sentence"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("EAR sample 中找不到 caption。")


def get_source_indices(sample: Dict[str, Any]) -> List[int]:
    value = sample.get("source_indices")
    if isinstance(value, list) and value:
        if not all(isinstance(x, int) and x >= 0 for x in value):
            raise ValueError("source_indices 必须是非负整数列表。")
        return sorted(set(value))

    value = sample.get("source_index")
    if isinstance(value, int) and value >= 0:
        return [value]

    raise ValueError("EAR sample 缺少 source_indices/source_index。")


def get_entities(sample: Dict[str, Any]) -> List[str]:
    # 优先使用 sanitizer 后的最终 EAR 结果，不做二次语义修改。
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


def get_special_token_ids(tokenizer) -> Tuple[int, int]:
    # 用空字符串推断 SOT/EOT，避免写死具体 token id。
    tokens = tokenizer([""])[0].tolist()
    nonzero = [x for x in tokens if x != 0]
    if len(nonzero) < 2:
        raise RuntimeError("无法推断 CLIP SOT/EOT token id。")
    return int(nonzero[0]), int(nonzero[1])


def get_content_tokens(
    tokenizer,
    text: str,
    sot_id: int,
    eot_id: int,
) -> Tuple[List[int], List[int]]:
    tokens = tokenizer([text])[0].tolist()
    try:
        sot_pos = tokens.index(sot_id)
        eot_pos = tokens.index(eot_id, sot_pos + 1)
    except ValueError as exc:
        raise RuntimeError(f"文本中找不到 SOT/EOT: {text}") from exc

    positions = list(range(sot_pos + 1, eot_pos))
    ids = [tokens[pos] for pos in positions]
    return ids, positions


def find_unique_span(
    caption_ids: List[int],
    caption_positions: List[int],
    entity_ids: List[int],
) -> Tuple[str, Optional[Tuple[int, int]]]:
    if not entity_ids or len(entity_ids) > len(caption_ids):
        return "unmatched", None

    starts = []
    width = len(entity_ids)
    for start in range(len(caption_ids) - width + 1):
        if caption_ids[start:start + width] == entity_ids:
            starts.append(start)

    if not starts:
        return "unmatched", None
    if len(starts) > 1:
        return "multiple", None

    start = starts[0]
    clip_start = caption_positions[start]
    clip_end = caption_positions[start + width - 1] + 1
    return "unique", (clip_start, clip_end)


def build_compact_index(
    ear_data: Dict[str, Any],
    model_name: str,
    max_words: int,
    expected_pairs: Optional[int],
) -> Dict[str, Any]:
    samples = get_samples(ear_data)
    tokenizer = open_clip.get_tokenizer(model_name)
    sot_id, eot_id = get_special_token_ids(tokenizer)

    pair_assignment: Dict[int, int] = {}
    semantic_offsets = [0]
    span_start: List[int] = []
    span_end: List[int] = []

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

        for entity in get_entities(sample):
            total_entities += 1
            try:
                entity_clean = pre_caption(entity, 100000)
                entity_ids, _ = get_content_tokens(
                    tokenizer, entity_clean, sot_id, eot_id
                )
            except ValueError:
                invalid_entities += 1
                continue

            status, span = find_unique_span(
                caption_ids, caption_positions, entity_ids
            )
            if status == "unique":
                unique_entities += 1
                span_start.append(span[0])
                span_end.append(span[1])
            elif status == "multiple":
                multiple_entities += 1
            else:
                unmatched_entities += 1

        semantic_offsets.append(len(span_start))

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

    missing = [i for i, x in enumerate(pair_to_semantic) if x < 0]
    if missing:
        raise ValueError(
            f"EAR 未覆盖全部训练 pair，共缺失 {len(missing)} 个，"
            f"前 20 个: {missing[:20]}"
        )

    # 训练热路径只保留整数 Tensor，避免完整 EAR dict/string 进入 DataLoader worker。
    result = {
        "metadata": {
            "format": "entity_span_index_v1",
            "model_name": model_name,
            "max_words": max_words,
            "span_rule": "unique_exact_clip_token_span_only",
            "span_format": "[start, end), CLIP token absolute position",
        },
        "statistics": {
            "num_semantic_samples": len(samples),
            "num_original_pairs": num_pairs,
            "total_entities": total_entities,
            "valid_unique_entities": unique_entities,
            "multiple_entities_skipped": multiple_entities,
            "unmatched_entities_skipped": unmatched_entities,
            "invalid_entities_skipped": invalid_entities,
            "valid_entity_coverage": unique_entities / max(total_entities, 1),
        },
        "pair_to_semantic": torch.tensor(pair_to_semantic, dtype=torch.int32),
        "semantic_offsets": torch.tensor(semantic_offsets, dtype=torch.int32),
        "span_start": torch.tensor(span_start, dtype=torch.int16),
        "span_end": torch.tensor(span_end, dtype=torch.int16),
    }
    return result


def print_report(result: Dict[str, Any]) -> None:
    stats = result["statistics"]
    print("\n" + "=" * 72)
    print("Compact Entity Span Index")
    print("=" * 72)
    print(f"Semantic captions       : {stats['num_semantic_samples']}")
    print(f"Original train pairs    : {stats['num_original_pairs']}")
    print(f"Total EAR entities      : {stats['total_entities']}")
    print(f"Valid unique spans      : {stats['valid_unique_entities']}")
    print(f"Multiple skipped        : {stats['multiple_entities_skipped']}")
    print(f"Unmatched skipped       : {stats['unmatched_entities_skipped']}")
    print(f"Invalid skipped         : {stats['invalid_entities_skipped']}")
    print(f"Valid coverage          : {stats['valid_entity_coverage'] * 100:.4f}%")
    print()
    print(f"pair_to_semantic shape  : {tuple(result['pair_to_semantic'].shape)}")
    print(f"semantic_offsets shape  : {tuple(result['semantic_offsets'].shape)}")
    print(f"span_start shape        : {tuple(result['span_start'].shape)}")
    print(f"span_end shape          : {tuple(result['span_end'].shape)}")
    print("=" * 72)


def run(
    input_path: str,
    output_path: str,
    model_name: str,
    max_words: int,
    expected_pairs: Optional[int],
) -> None:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结 EAR 构建紧凑的 Entity token-span 训练索引。"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="ViT-B-32-quickgelu")
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=0,
        help="原始训练 pair 数，0 表示根据 source_indices 推断。",
    )
    return parser


def main() -> None:
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
