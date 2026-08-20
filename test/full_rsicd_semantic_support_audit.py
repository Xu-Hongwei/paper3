import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PROBE_TYPES = ("color", "size", "shape", "state")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full RSICD EAR semantic support audit: Entity × Attribute Type × Attribute Value."
    )
    parser.add_argument(
        "--ear-file",
        type=str,
        default="E:/paper3/data/structured_semantics/rsicd_train_qwen37_v30_open.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rsicd_semantic_support_audit",
    )
    parser.add_argument(
        "--probe-attribute-types",
        nargs="+",
        default=list(DEFAULT_PROBE_TYPES),
        help="用于筛选视觉 decodability probe 候选的属性类型。count 仍会统计，但默认不作为局部属性 probe。",
    )
    parser.add_argument(
        "--min-entity-support",
        type=int,
        default=50,
        help="候选 Entity 至少需要多少个 unique EAR sample occurrences。",
    )
    parser.add_argument(
        "--min-values",
        type=int,
        default=2,
        help="某 Entity-Attribute 至少需要多少个有充分支持的不同 value。",
    )
    parser.add_argument(
        "--min-value-support",
        type=int,
        default=10,
        help="每个 attribute value 至少需要多少个 unique EAR sample occurrences。",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="终端打印多少个高支持候选。",
    )
    parser.add_argument(
        "--include-absent",
        action="store_true",
        help="默认排除显式 presence=absent 的 Entity；打开后纳入统计。",
    )
    return parser.parse_args()


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_samples(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("EAR 顶层必须是 dict 或 list。")

    for key in ("samples", "results", "items", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return value

    # 少数导出格式可能把 sample_index 直接作为 key。
    values = list(data.values())
    if values and all(isinstance(x, dict) for x in values):
        return values

    raise ValueError(
        f"无法定位 EAR samples。顶层 keys={list(data.keys())[:30]}"
    )


def get_semantics(sample):
    for key in (
        "sanitized_structured_semantics",
        "final_structured_semantics",
        "structured_semantics",
        "sanitized",
        "semantics",
    ):
        value = sample.get(key)
        if isinstance(value, dict):
            return value

    if "entities" in sample or "relations" in sample:
        return {
            "entities": sample.get("entities", []),
            "relations": sample.get("relations", []),
        }

    return None


def get_pair_weight(sample):
    source_indices = sample.get("source_indices")
    if isinstance(source_indices, list) and source_indices:
        return len(source_indices)

    num_occurrences = sample.get("num_occurrences")
    if isinstance(num_occurrences, int) and num_occurrences > 0:
        return num_occurrences

    return 1


def is_absent(attributes):
    absent_values = {"absent", "no", "none", "without", "not present"}
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        attr_type = normalize_text(attr.get("type"))
        attr_value = normalize_text(attr.get("value"))
        if attr_type == "presence" and attr_value in absent_values:
            return True
    return False


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    data = load_json(args.ear_file)
    samples = get_samples(data)
    probe_types = {normalize_text(x) for x in args.probe_attribute_types}

    # Entity 统计：默认按 lower + whitespace normalization，故 building/buildings 仍保持不同。
    entity_occ = Counter()
    entity_weighted = Counter()
    entity_sample_ids = defaultdict(set)
    entity_image_ids = defaultdict(set)

    attr_occ = Counter()
    attr_weighted = Counter()
    attr_sample_ids = defaultdict(set)

    value_occ = Counter()
    value_weighted = Counter()
    value_sample_ids = defaultdict(set)

    relation_occ = Counter()
    relation_weighted = Counter()

    global_attr_types = Counter()
    global_attr_values = defaultdict(Counter)

    valid_semantic_samples = 0
    skipped_no_semantics = 0
    skipped_absent_entities = 0
    total_entity_occurrences = 0
    total_attribute_occurrences = 0
    total_relation_occurrences = 0

    for semantic_index, sample in enumerate(samples):
        semantics = get_semantics(sample)
        if not semantics:
            skipped_no_semantics += 1
            continue

        valid_semantic_samples += 1
        pair_weight = get_pair_weight(sample)
        image_id = sample.get("image_id")
        sample_uid = sample.get("sample_index", semantic_index)

        entities = semantics.get("entities", [])
        relations = semantics.get("relations", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        id_to_text = {}

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_text = normalize_text(entity.get("text"))
            if not entity_text:
                continue

            entity_id = entity.get("id")
            if entity_id is not None:
                id_to_text[str(entity_id)] = entity_text

            attributes = entity.get("attributes", [])
            if not isinstance(attributes, list):
                attributes = []

            if not args.include_absent and is_absent(attributes):
                skipped_absent_entities += 1
                continue

            total_entity_occurrences += 1
            entity_occ[entity_text] += 1
            entity_weighted[entity_text] += pair_weight
            entity_sample_ids[entity_text].add(sample_uid)
            if image_id is not None:
                entity_image_ids[entity_text].add(image_id)

            # 同一 Entity occurrence 内完全重复 attribute 去重，避免 LLM 重复项抬高支持度。
            seen_attributes = set()
            for attr in attributes:
                if not isinstance(attr, dict):
                    continue

                attr_type = normalize_text(attr.get("type"))
                attr_value = normalize_text(attr.get("value"))
                if not attr_type or not attr_value:
                    continue

                key = (attr_type, attr_value)
                if key in seen_attributes:
                    continue
                seen_attributes.add(key)

                total_attribute_occurrences += 1
                global_attr_types[attr_type] += 1
                global_attr_values[attr_type][attr_value] += 1

                attr_key = (entity_text, attr_type)
                value_key = (entity_text, attr_type, attr_value)

                attr_occ[attr_key] += 1
                attr_weighted[attr_key] += pair_weight
                attr_sample_ids[attr_key].add(sample_uid)

                value_occ[value_key] += 1
                value_weighted[value_key] += pair_weight
                value_sample_ids[value_key].add(sample_uid)

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            predicate = normalize_text(relation.get("predicate"))
            if not predicate:
                continue

            subject_id = relation.get("subject")
            object_id = relation.get("object")
            subject_text = id_to_text.get(str(subject_id), normalize_text(subject_id))
            object_text = id_to_text.get(str(object_id), normalize_text(object_id))

            relation_key = (subject_text, predicate, object_text)
            relation_occ[relation_key] += 1
            relation_weighted[relation_key] += pair_weight
            total_relation_occurrences += 1

    # ------------------------------------------------------------
    # Entity support
    # ------------------------------------------------------------
    entity_rows = []
    for entity, count in entity_occ.most_common():
        entity_rows.append({
            "entity": entity,
            "occurrence_count": count,
            "unique_sample_count": len(entity_sample_ids[entity]),
            "unique_image_count": len(entity_image_ids[entity]),
            "pair_weighted_count": entity_weighted[entity],
            "num_attribute_types": len({
                attr_type for e, attr_type in attr_occ if e == entity
            }),
        })

    # ------------------------------------------------------------
    # Entity × Attribute support
    # ------------------------------------------------------------
    attr_rows = []
    candidate_rows = []

    for (entity, attr_type), count in sorted(
        attr_occ.items(), key=lambda x: (-x[1], x[0])
    ):
        values = [
            (value, value_occ[(entity, attr_type, value)])
            for e, t, value in value_occ
            if e == entity and t == attr_type
        ]
        values.sort(key=lambda x: (-x[1], x[0]))

        supported_values = [
            (value, support)
            for value, support in values
            if support >= args.min_value_support
        ]

        entity_support = entity_occ[entity]
        coverage = count / max(entity_support, 1)
        eligible = (
            attr_type in probe_types
            and entity_support >= args.min_entity_support
            and len(supported_values) >= args.min_values
        )

        attr_rows.append({
            "entity": entity,
            "attribute_type": attr_type,
            "entity_support": entity_support,
            "attribute_occurrence_count": count,
            "attribute_coverage": coverage,
            "unique_sample_count": len(attr_sample_ids[(entity, attr_type)]),
            "pair_weighted_count": attr_weighted[(entity, attr_type)],
            "num_unique_values": len(values),
            "num_supported_values": len(supported_values),
            "supported_values": " | ".join(
                f"{value}:{support}" for value, support in supported_values
            ),
            "probe_eligible": int(eligible),
        })

        if eligible:
            top_values = supported_values[:8]
            candidate_rows.append({
                "entity": entity,
                "attribute_type": attr_type,
                "entity_support": entity_support,
                "attribute_occurrence_count": count,
                "attribute_coverage": coverage,
                "num_supported_values": len(supported_values),
                "supported_values": " | ".join(
                    f"{value}:{support}" for value, support in top_values
                ),
                "min_value_support": min(x[1] for x in supported_values),
                "total_supported_value_occurrences": sum(
                    x[1] for x in supported_values
                ),
            })

    candidate_rows.sort(
        key=lambda x: (
            -x["num_supported_values"],
            -x["total_supported_value_occurrences"],
            -x["attribute_coverage"],
        )
    )

    # ------------------------------------------------------------
    # Entity × Attribute × Value support
    # ------------------------------------------------------------
    value_rows = []
    for (entity, attr_type, value), count in sorted(
        value_occ.items(), key=lambda x: (-x[1], x[0])
    ):
        value_rows.append({
            "entity": entity,
            "attribute_type": attr_type,
            "attribute_value": value,
            "entity_support": entity_occ[entity],
            "value_occurrence_count": count,
            "unique_sample_count": len(
                value_sample_ids[(entity, attr_type, value)]
            ),
            "pair_weighted_count": value_weighted[
                (entity, attr_type, value)
            ],
            "meets_min_value_support": int(
                count >= args.min_value_support
            ),
        })

    relation_rows = []
    for (subject, predicate, obj), count in relation_occ.most_common():
        relation_rows.append({
            "subject_entity": subject,
            "predicate": predicate,
            "object_entity": obj,
            "occurrence_count": count,
            "pair_weighted_count": relation_weighted[
                (subject, predicate, obj)
            ],
        })

    global_attr_rows = []
    for attr_type, count in global_attr_types.most_common():
        values = global_attr_values[attr_type]
        global_attr_rows.append({
            "attribute_type": attr_type,
            "occurrence_count": count,
            "num_unique_values": len(values),
            "top_values": " | ".join(
                f"{value}:{support}"
                for value, support in values.most_common(20)
            ),
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "entity_support.csv",
        entity_rows,
        [
            "entity", "occurrence_count", "unique_sample_count",
            "unique_image_count", "pair_weighted_count",
            "num_attribute_types",
        ],
    )
    write_csv(
        output_dir / "entity_attribute_support.csv",
        attr_rows,
        [
            "entity", "attribute_type", "entity_support",
            "attribute_occurrence_count", "attribute_coverage",
            "unique_sample_count", "pair_weighted_count",
            "num_unique_values", "num_supported_values",
            "supported_values", "probe_eligible",
        ],
    )
    write_csv(
        output_dir / "entity_attribute_value_support.csv",
        value_rows,
        [
            "entity", "attribute_type", "attribute_value",
            "entity_support", "value_occurrence_count",
            "unique_sample_count", "pair_weighted_count",
            "meets_min_value_support",
        ],
    )
    write_csv(
        output_dir / "probe_candidates.csv",
        candidate_rows,
        [
            "entity", "attribute_type", "entity_support",
            "attribute_occurrence_count", "attribute_coverage",
            "num_supported_values", "supported_values",
            "min_value_support", "total_supported_value_occurrences",
        ],
    )
    write_csv(
        output_dir / "relation_support.csv",
        relation_rows,
        [
            "subject_entity", "predicate", "object_entity",
            "occurrence_count", "pair_weighted_count",
        ],
    )
    write_csv(
        output_dir / "global_attribute_support.csv",
        global_attr_rows,
        [
            "attribute_type", "occurrence_count",
            "num_unique_values", "top_values",
        ],
    )

    summary = {
        "metadata": {
            "ear_file": args.ear_file,
            "probe_attribute_types": sorted(probe_types),
            "min_entity_support": args.min_entity_support,
            "min_values": args.min_values,
            "min_value_support": args.min_value_support,
            "include_absent": args.include_absent,
            "normalization": (
                "lowercase + whitespace normalization only; "
                "no synonym/plural merging"
            ),
            "support_basis": (
                "candidate eligibility uses unique EAR entity occurrences; "
                "pair_weighted_count is reported separately"
            ),
        },
        "dataset": {
            "raw_samples": len(samples),
            "valid_semantic_samples": valid_semantic_samples,
            "skipped_no_semantics": skipped_no_semantics,
            "total_entity_occurrences": total_entity_occurrences,
            "total_attribute_occurrences": total_attribute_occurrences,
            "total_relation_occurrences": total_relation_occurrences,
            "skipped_absent_entities": skipped_absent_entities,
            "unique_entities": len(entity_occ),
            "unique_entity_attribute_pairs": len(attr_occ),
            "unique_entity_attribute_values": len(value_occ),
            "probe_candidate_pairs": len(candidate_rows),
        },
        "global_attribute_types": {
            row["attribute_type"]: {
                "occurrence_count": row["occurrence_count"],
                "num_unique_values": row["num_unique_values"],
                "top_values": row["top_values"],
            }
            for row in global_attr_rows
        },
        "top_probe_candidates": candidate_rows[: args.top],
    }

    with (output_dir / "semantic_support_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 116)
    print("FULL RSICD STRUCTURED SEMANTIC SUPPORT AUDIT")
    print("=" * 116)
    print(f"EAR file                : {args.ear_file}")
    print(f"Raw semantic samples    : {len(samples)}")
    print(f"Valid semantic samples  : {valid_semantic_samples}")
    print(f"Entity occurrences      : {total_entity_occurrences}")
    print(f"Attribute occurrences   : {total_attribute_occurrences}")
    print(f"Relation occurrences    : {total_relation_occurrences}")
    print(f"Unique entities         : {len(entity_occ)}")
    print(f"Probe attribute types   : {sorted(probe_types)}")
    print(
        "Eligibility             : "
        f"entity>={args.min_entity_support}, "
        f"supported_values>={args.min_values}, "
        f"each value>={args.min_value_support}"
    )
    print(f"Eligible E-A pairs      : {len(candidate_rows)}")

    print("\nGLOBAL ATTRIBUTE TYPES")
    print("-" * 116)
    for row in global_attr_rows:
        print(
            f"{row['attribute_type']:<12} "
            f"occ={row['occurrence_count']:<6} "
            f"values={row['num_unique_values']:<5} "
            f"top={row['top_values'][:85]}"
        )

    print("\nTOP PROBE CANDIDATES")
    print("-" * 116)
    if not candidate_rows:
        print("<none under current thresholds>")
    else:
        for i, row in enumerate(candidate_rows[: args.top], start=1):
            print(
                f"{i:>2}. {row['entity']:<28} "
                f"{row['attribute_type']:<8} "
                f"E={row['entity_support']:<5} "
                f"A={row['attribute_occurrence_count']:<5} "
                f"cov={row['attribute_coverage']:.1%} "
                f"values={row['supported_values']}"
            )

    print("\nIMPORTANT")
    print("-" * 116)
    print("1. 当前不合并 building/buildings、gray/grey、large/big 等近义/形态变体。")
    print("2. probe_candidates.csv 是高精度起点，不是最终语义 taxonomy。")
    print("3. 若候选很少，先检查实体/属性同义表达碎片化，再决定是否做人工可审计的 canonicalization。")
    print("4. count 已完整统计，但默认不参与 local attribute probe；后续更适合 set-level / multi-instance 分支。")
    print("5. relation_support.csv 同时保留，用于后续判断 Relation 是否比 Attribute 更有数据支撑。")
    print("-" * 116)
    print(f"Output dir: {output_dir}")
    print("=" * 116)


if __name__ == "__main__":
    main()
