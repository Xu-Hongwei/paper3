import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查结构化语义 JSON 的 source_indices 是否完整、唯一覆盖原始训练 pair。"
    )
    parser.add_argument("--semantic-file", type=str, required=True)
    parser.add_argument("--expected-count", type=int, default=39310)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/structured_semantics/source_indices_check.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    semantic_path = Path(args.semantic_file)
    with semantic_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("JSON 中未找到有效的 samples 列表。")

    sample_indices = []
    flat_source_indices = []
    empty_source_samples = []
    occurrence_mismatch = []
    invalid_source_samples = []

    for pos, sample in enumerate(samples):
        sample_index = sample.get("sample_index")
        source_indices = sample.get("source_indices")
        num_occurrences = sample.get("num_occurrences")

        sample_indices.append(sample_index)

        if not isinstance(source_indices, list):
            invalid_source_samples.append(sample_index)
            continue

        if len(source_indices) == 0:
            empty_source_samples.append(sample_index)

        if num_occurrences is not None and num_occurrences != len(source_indices):
            occurrence_mismatch.append({
                "sample_index": sample_index,
                "num_occurrences": num_occurrences,
                "source_indices_len": len(source_indices),
            })

        flat_source_indices.extend(source_indices)

    expected = set(range(args.expected_count))
    counts = Counter(flat_source_indices)
    actual = set(flat_source_indices)

    duplicate_source_indices = sorted(
        idx for idx, count in counts.items() if count > 1
    )
    missing_source_indices = sorted(expected - actual)
    out_of_range_source_indices = sorted(
        idx for idx in actual if not isinstance(idx, int) or idx < 0 or idx >= args.expected_count
    )

    valid_int_indices = [
        idx for idx in flat_source_indices
        if isinstance(idx, int)
    ]

    sample_index_counts = Counter(sample_indices)
    duplicate_sample_indices = sorted(
        idx for idx, count in sample_index_counts.items() if count > 1
    )

    exact_coverage = (
        len(flat_source_indices) == args.expected_count
        and len(actual) == args.expected_count
        and not duplicate_source_indices
        and not missing_source_indices
        and not out_of_range_source_indices
    )

    report = {
        "semantic_file": str(semantic_path),
        "num_semantic_samples": len(samples),
        "sample_index_unique_count": len(set(sample_indices)),
        "duplicate_sample_indices": duplicate_sample_indices,
        "expected_raw_pair_count": args.expected_count,
        "flattened_source_index_count": len(flat_source_indices),
        "unique_source_index_count": len(actual),
        "min_source_index": min(valid_int_indices) if valid_int_indices else None,
        "max_source_index": max(valid_int_indices) if valid_int_indices else None,
        "duplicate_source_index_count": len(duplicate_source_indices),
        "duplicate_source_indices_preview": duplicate_source_indices[:50],
        "missing_source_index_count": len(missing_source_indices),
        "missing_source_indices_preview": missing_source_indices[:50],
        "out_of_range_source_index_count": len(out_of_range_source_indices),
        "out_of_range_source_indices_preview": out_of_range_source_indices[:50],
        "empty_source_samples_count": len(empty_source_samples),
        "empty_source_samples_preview": empty_source_samples[:50],
        "invalid_source_samples_count": len(invalid_source_samples),
        "invalid_source_samples_preview": invalid_source_samples[:50],
        "num_occurrences_mismatch_count": len(occurrence_mismatch),
        "num_occurrences_mismatch_preview": occurrence_mismatch[:50],
        "exact_coverage_pass": exact_coverage,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("STRUCTURED SEMANTICS source_indices CHECK")
    print("=" * 100)
    print(f"Semantic samples               : {len(samples)}")
    print(f"Unique sample_index            : {len(set(sample_indices))}")
    print(f"Expected raw pairs             : {args.expected_count}")
    print(f"Flattened source_indices       : {len(flat_source_indices)}")
    print(f"Unique source_indices          : {len(actual)}")
    print(
        f"Source index range             : "
        f"{report['min_source_index']} ~ {report['max_source_index']}"
    )
    print(f"Duplicate source_indices       : {len(duplicate_source_indices)}")
    print(f"Missing source_indices         : {len(missing_source_indices)}")
    print(f"Out-of-range source_indices    : {len(out_of_range_source_indices)}")
    print(f"Empty source_indices samples   : {len(empty_source_samples)}")
    print(f"Invalid source_indices samples : {len(invalid_source_samples)}")
    print(f"num_occurrences mismatches     : {len(occurrence_mismatch)}")
    print("-" * 100)
    print(f"EXACT COVERAGE                 : {'PASS' if exact_coverage else 'FAIL'}")
    print(f"Report                         : {output_path}")
    print("=" * 100)

    if duplicate_source_indices:
        print("Duplicate preview:", duplicate_source_indices[:20])
    if missing_source_indices:
        print("Missing preview  :", missing_source_indices[:20])
    if out_of_range_source_indices:
        print("Out-of-range preview:", out_of_range_source_indices[:20])


if __name__ == "__main__":
    main()
