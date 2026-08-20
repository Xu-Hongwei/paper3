import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


TARGET_ATTRS = ("color", "size", "shape", "state")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit Aircraft explicit EAR attribute coverage before attribute-wise linear probing."
    )
    parser.add_argument("--samples-csv", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["aircraft"])
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-area", type=float, default=0.35)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/attribute_distribution_audit_aircraft",
    )
    return parser.parse_args()


def cache_key(class_name, image_id):
    return f"{class_name}:{int(image_id)}"


def parse_python_or_json(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).strip().lower()
    return " ".join(text.split())


def extract_type_value(item):
    """
    兼容常见 EAR attribute 结构：
    {"type": "color", "value": "white"}
    {"attribute_type": "color", "attribute_value": "white"}
    {"name": "color", "value": "white"}
    """
    if not isinstance(item, dict):
        return None, None

    attr_type = (
        item.get("type")
        or item.get("attribute_type")
        or item.get("name")
        or item.get("category")
    )
    attr_value = (
        item.get("value")
        if "value" in item
        else item.get("attribute_value")
    )

    attr_type = normalize_text(attr_type)
    if attr_type not in TARGET_ATTRS:
        return None, None

    if isinstance(attr_value, (list, tuple, set)):
        values = [normalize_text(v) for v in attr_value if normalize_text(v)]
        return attr_type, values

    value = normalize_text(attr_value)
    return attr_type, [value] if value else []


def parse_attributes(raw):
    """
    输出：
    {
      "color": ["white"],
      "size": ["large"],
      ...
    }
    不在这里做 large/big、grey/gray 等语义合并，先保留 EAR 原始分布。
    """
    parsed = parse_python_or_json(raw)
    result = {name: [] for name in TARGET_ATTRS}

    if parsed is None:
        return result

    if isinstance(parsed, dict):
        # 结构1：{"color": "white", "size": "large"}
        direct_hit = False
        for attr in TARGET_ATTRS:
            if attr in parsed:
                direct_hit = True
                value = parsed[attr]
                if isinstance(value, (list, tuple, set)):
                    result[attr].extend(
                        normalize_text(v) for v in value if normalize_text(v)
                    )
                else:
                    value = normalize_text(value)
                    if value:
                        result[attr].append(value)

        # 结构2：{"type": "...", "value": "..."}
        attr_type, values = extract_type_value(parsed)
        if attr_type:
            result[attr_type].extend(values)

        # 结构3：嵌套 attributes
        if "attributes" in parsed:
            nested = parse_attributes(parsed["attributes"])
            for attr in TARGET_ATTRS:
                result[attr].extend(nested[attr])

        # 其他嵌套 dict/list 再递归一次
        if not direct_hit and not attr_type:
            for value in parsed.values():
                if isinstance(value, (dict, list, tuple)):
                    nested = parse_attributes(value)
                    for attr in TARGET_ATTRS:
                        result[attr].extend(nested[attr])

    elif isinstance(parsed, (list, tuple)):
        for item in parsed:
            if isinstance(item, dict):
                attr_type, values = extract_type_value(item)
                if attr_type:
                    result[attr_type].extend(values)
                else:
                    nested = parse_attributes(item)
                    for attr in TARGET_ATTRS:
                        result[attr].extend(nested[attr])
            elif isinstance(item, (list, tuple)):
                nested = parse_attributes(item)
                for attr in TARGET_ATTRS:
                    result[attr].extend(nested[attr])

    # 单样本同一属性同一值去重，但保留多个不同值
    for attr in TARGET_ATTRS:
        result[attr] = list(dict.fromkeys(v for v in result[attr] if v))

    return result


def load_samples(path, classes):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        attr_column = None
        for candidate in ("attributes", "attrs", "attribute"):
            if candidate in fieldnames:
                attr_column = candidate
                break

        if attr_column is None:
            raise RuntimeError(
                f"CSV 中未找到 attributes/attrs/attribute 列。当前列：{fieldnames}"
            )

        for row in reader:
            if row.get("class") not in classes:
                continue
            rows.append({
                "class": row["class"],
                "image_id": int(row["image_id"]),
                "dataset_index": int(row["dataset_index"]),
                "entity": row.get("entity", ""),
                "phrase": row.get("phrase", ""),
                "caption": row.get("caption", ""),
                "attributes_raw": row.get(attr_column, ""),
                "attributes": parse_attributes(row.get(attr_column, "")),
            })

    return rows, attr_column


def has_valid_object_proposal(record, min_area, max_area):
    if record is None:
        return False
    areas = record.get("area_ratios", [])
    return any(min_area <= float(area) <= max_area for area in areas)


def summarize_subset(samples, subset_name):
    attr_value_sample_counts = {
        attr: Counter() for attr in TARGET_ATTRS
    }
    attr_value_occurrence_counts = {
        attr: Counter() for attr in TARGET_ATTRS
    }
    attr_present_samples = Counter()
    samples_with_any = 0
    combinations = Counter()

    for sample in samples:
        attrs = sample["attributes"]
        active = []

        for attr in TARGET_ATTRS:
            values = attrs[attr]
            if values:
                attr_present_samples[attr] += 1
                active.append(attr)

            for value in set(values):
                attr_value_sample_counts[attr][value] += 1
            for value in values:
                attr_value_occurrence_counts[attr][value] += 1

        if active:
            samples_with_any += 1
        combinations["+".join(active) if active else "none"] += 1

    total = len(samples)
    summary = {
        "subset": subset_name,
        "num_samples": total,
        "samples_with_any_target_attribute": samples_with_any,
        "coverage_any": samples_with_any / max(total, 1),
        "attribute_coverage": {},
        "attribute_combinations": dict(combinations.most_common()),
    }

    for attr in TARGET_ATTRS:
        present = attr_present_samples[attr]
        summary["attribute_coverage"][attr] = {
            "samples_present": present,
            "coverage": present / max(total, 1),
            "num_unique_values": len(attr_value_sample_counts[attr]),
            "value_sample_counts": dict(
                attr_value_sample_counts[attr].most_common()
            ),
            "value_occurrence_counts": dict(
                attr_value_occurrence_counts[attr].most_common()
            ),
        }

    return summary


def print_subset(summary):
    print("\n" + "=" * 100)
    print(f"[{summary['subset']}] samples={summary['num_samples']}")
    print(
        "Any target attr: "
        f"{summary['samples_with_any_target_attribute']}/{summary['num_samples']} "
        f"({summary['coverage_any']:.1%})"
    )
    print("=" * 100)

    for attr in TARGET_ATTRS:
        item = summary["attribute_coverage"][attr]
        print(
            f"\n{attr.upper():<7} coverage="
            f"{item['samples_present']}/{summary['num_samples']} "
            f"({item['coverage']:.1%}), "
            f"unique_values={item['num_unique_values']}"
        )

        counts = item["value_sample_counts"]
        if not counts:
            print("  <none>")
            continue

        for value, count in counts.items():
            print(f"  {value:<24} {count:>4}")

    print("\nAttribute combinations:")
    for combo, count in summary["attribute_combinations"].items():
        print(f"  {combo:<28} {count:>4}")


def main():
    args = parse_args()

    samples, attr_column = load_samples(
        args.samples_csv,
        set(args.classes),
    )
    cache_blob = torch.load(
        args.cache,
        map_location="cpu",
        weights_only=False,
    )
    cache = cache_blob["records"]

    grounded = []
    object_valid = []

    for sample in samples:
        record = cache.get(cache_key(sample["class"], sample["image_id"]))
        if record is None:
            continue

        if int(record.get("num_boxes", 0)) > 0:
            grounded.append(sample)

        if has_valid_object_proposal(
            record,
            args.min_area,
            args.max_area,
        ):
            object_valid.append(sample)

    summaries = {
        "grounded": summarize_subset(grounded, "grounded"),
        "object_valid": summarize_subset(object_valid, "object_valid"),
    }

    print("=" * 100)
    print("ATTRIBUTE DISTRIBUTION AUDIT")
    print("=" * 100)
    print(f"Samples CSV : {args.samples_csv}")
    print(f"Attr column : {attr_column}")
    print(f"Cache       : {args.cache}")
    print(f"Classes     : {args.classes}")
    print(f"Area range  : [{args.min_area}, {args.max_area}]")
    print(f"CSV samples : {len(samples)}")

    print_subset(summaries["grounded"])
    print_subset(summaries["object_valid"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "attribute_distribution_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "samples_csv": args.samples_csv,
                    "cache": args.cache,
                    "classes": args.classes,
                    "attribute_column": attr_column,
                    "target_attributes": list(TARGET_ATTRS),
                    "min_area": args.min_area,
                    "max_area": args.max_area,
                    "note": (
                        "Values are kept close to raw EAR output; no synonym merging "
                        "such as gray/grey or large/big is applied."
                    ),
                },
                "subsets": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    value_rows = []
    for subset_name, summary in summaries.items():
        for attr in TARGET_ATTRS:
            item = summary["attribute_coverage"][attr]
            for value, count in item["value_sample_counts"].items():
                value_rows.append({
                    "subset": subset_name,
                    "attribute": attr,
                    "value": value,
                    "sample_count": count,
                    "subset_samples": summary["num_samples"],
                    "sample_ratio": count / max(summary["num_samples"], 1),
                })

    counts_path = output_dir / "attribute_value_counts.csv"
    with counts_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "subset",
            "attribute",
            "value",
            "sample_count",
            "subset_samples",
            "sample_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(value_rows)

    sample_path = output_dir / "sample_attribute_audit.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "class",
            "image_id",
            "dataset_index",
            "entity",
            "phrase",
            "color",
            "size",
            "shape",
            "state",
            "is_grounded",
            "is_object_valid",
            "caption",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        grounded_ids = {
            (s["class"], s["image_id"]) for s in grounded
        }
        valid_ids = {
            (s["class"], s["image_id"]) for s in object_valid
        }

        for sample in samples:
            key = (sample["class"], sample["image_id"])
            writer.writerow({
                "class": sample["class"],
                "image_id": sample["image_id"],
                "dataset_index": sample["dataset_index"],
                "entity": sample["entity"],
                "phrase": sample["phrase"],
                "color": " | ".join(sample["attributes"]["color"]),
                "size": " | ".join(sample["attributes"]["size"]),
                "shape": " | ".join(sample["attributes"]["shape"]),
                "state": " | ".join(sample["attributes"]["state"]),
                "is_grounded": int(key in grounded_ids),
                "is_object_valid": int(key in valid_ids),
                "caption": sample["caption"],
            })

    print("\n" + "=" * 100)
    print("NEXT DECISION")
    print("=" * 100)
    print("1. 先看 grounded=112 左右整体属性覆盖，判断 Aircraft 文本是否真的有足够 fine semantics。")
    print("2. 再看 object_valid≈66 子集，判断经过 box 过滤后是否进一步丢失属性多样性。")
    print("3. 某属性至少要有多个可解释 value 且每个 value 有足够样本，才值得做 attribute-wise probe。")
    print("4. 先不要合并近义值；先人工查看 counts，再决定 white/gray、large/big 等是否需要规范化。")
    print("-" * 100)
    print(f"Summary : {summary_path}")
    print(f"Counts  : {counts_path}")
    print(f"Samples : {sample_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
