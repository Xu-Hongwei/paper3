import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RSICD_CLASSES = [
    "airport", "bareland", "baseballfield", "beach", "bridge", "center",
    "church", "commercial", "denseresidential", "desert", "farmland",
    "forest", "industrial", "meadow", "mediumresidential", "mountain",
    "park", "parking", "playground", "pond", "port", "railwaystation",
    "resort", "river", "school", "sparseresidential", "square", "stadium",
    "storagetanks", "viaduct",
]
CANONICAL_BY_NORMALIZED = {
    re.sub(r"[^a-z0-9]", "", name.lower()): name
    for name in RSICD_CLASSES
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查 RSICD 官方 txtclasses_rsicd 与 dataset.json 的类别映射覆盖。"
    )
    parser.add_argument("--class-dir", type=str, required=True)
    parser.add_argument("--dataset-json", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rsicd_official_class_check",
    )
    return parser.parse_args()


def normalize_class_file_name(name):
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return CANONICAL_BY_NORMALIZED.get(key)


def load_dataset_filenames(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        images = data.get("images")
        if not isinstance(images, list):
            raise ValueError(
                "dataset JSON 为 dict，但没有有效的 images 列表。"
            )
    elif isinstance(data, list):
        images = data
    else:
        raise ValueError("dataset JSON 顶层必须是 dict 或 list。")

    filenames = []
    for idx, item in enumerate(images):
        if not isinstance(item, dict):
            raise ValueError(f"images[{idx}] 不是 dict。")

        value = None
        for key in ("filename", "image"):
            if key in item:
                value = item[key]
                break

        if value is None:
            raise KeyError(
                f"images[{idx}] 没有 filename/image 字段。"
            )

        filenames.append(Path(str(value)).name)

    return filenames


def read_lines(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding) as f:
                return [
                    Path(line.strip()).name
                    for line in f
                    if line.strip()
                ]
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"无法解码文件: {path}"
    )


def main():
    args = parse_args()

    class_dir = Path(args.class_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not class_dir.is_dir():
        raise FileNotFoundError(f"class-dir 不存在: {class_dir}")

    txt_files = sorted(class_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"{class_dir} 下没有找到 *.txt"
        )

    dataset_filenames = load_dataset_filenames(args.dataset_json)
    dataset_counter = Counter(dataset_filenames)
    dataset_set = set(dataset_filenames)

    raw_class_files = []
    unexpected_class_files = []
    missing_canonical_classes = set(RSICD_CLASSES)

    filename_to_classes = defaultdict(list)
    canonical_counts = Counter()
    raw_counts = Counter()

    for txt_path in txt_files:
        raw_name = txt_path.stem
        canonical_name = normalize_class_file_name(raw_name)

        raw_class_files.append(raw_name)
        if canonical_name is None:
            unexpected_class_files.append(raw_name)
        else:
            missing_canonical_classes.discard(canonical_name)

        filenames = read_lines(txt_path)
        raw_counts[raw_name] = len(filenames)

        if canonical_name is not None:
            canonical_counts[canonical_name] += len(filenames)

        for filename in filenames:
            filename_to_classes[filename].append(
                canonical_name if canonical_name is not None else raw_name
            )

    mapped_set = set(filename_to_classes)
    duplicate_mappings = {
        filename: classes
        for filename, classes in filename_to_classes.items()
        if len(classes) > 1
    }

    missing_from_class_files = sorted(dataset_set - mapped_set)
    extra_in_class_files = sorted(mapped_set - dataset_set)

    numeric_dataset = sorted(
        filename
        for filename in dataset_set
        if Path(filename).stem.isdigit()
    )
    numeric_mapped = [
        filename for filename in numeric_dataset
        if filename in filename_to_classes
    ]
    numeric_missing = [
        filename for filename in numeric_dataset
        if filename not in filename_to_classes
    ]

    recognized_mapping = {}
    for filename, classes in filename_to_classes.items():
        if len(classes) != 1:
            continue
        category = classes[0]
        if category in RSICD_CLASSES:
            recognized_mapping[filename] = category

    exact_dataset_unique = len(dataset_set) == len(dataset_filenames)
    exact_coverage = (
        not duplicate_mappings
        and not missing_from_class_files
        and not extra_in_class_files
        and len(mapped_set) == len(dataset_set)
    )
    canonical_30_pass = (
        not unexpected_class_files
        and not missing_canonical_classes
        and len(canonical_counts) == 30
    )

    report = {
        "class_dir": str(class_dir),
        "dataset_json": str(args.dataset_json),
        "num_txt_files": len(txt_files),
        "raw_class_files": raw_class_files,
        "unexpected_class_files": unexpected_class_files,
        "missing_canonical_classes": sorted(missing_canonical_classes),
        "dataset_image_count": len(dataset_filenames),
        "dataset_unique_filename_count": len(dataset_set),
        "dataset_filenames_unique": exact_dataset_unique,
        "mapped_unique_filename_count": len(mapped_set),
        "duplicate_mapping_count": len(duplicate_mappings),
        "duplicate_mapping_preview": dict(
            list(sorted(duplicate_mappings.items()))[:50]
        ),
        "missing_from_class_files_count": len(missing_from_class_files),
        "missing_from_class_files_preview": missing_from_class_files[:50],
        "extra_in_class_files_count": len(extra_in_class_files),
        "extra_in_class_files_preview": extra_in_class_files[:50],
        "numeric_dataset_image_count": len(numeric_dataset),
        "numeric_mapped_count": len(numeric_mapped),
        "numeric_missing_count": len(numeric_missing),
        "numeric_missing_preview": numeric_missing[:50],
        "raw_class_counts": dict(sorted(raw_counts.items())),
        "canonical_class_counts": {
            name: canonical_counts[name]
            for name in RSICD_CLASSES
        },
        "canonical_30_pass": canonical_30_pass,
        "exact_filename_coverage_pass": exact_coverage,
    }

    with (output_dir / "report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with (output_dir / "filename_to_category.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            recognized_mapping,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print("=" * 108)
    print("RSICD OFFICIAL CLASS MAPPING CHECK")
    print("=" * 108)
    print(f"TXT files                     : {len(txt_files)}")
    print(f"Dataset images                : {len(dataset_filenames)}")
    print(f"Dataset unique filenames      : {len(dataset_set)}")
    print(f"Mapped unique filenames       : {len(mapped_set)}")
    print(f"Duplicate class mappings      : {len(duplicate_mappings)}")
    print(f"Missing from class files      : {len(missing_from_class_files)}")
    print(f"Extra in class files          : {len(extra_in_class_files)}")
    print(f"Numeric filenames in dataset  : {len(numeric_dataset)}")
    print(f"Numeric filenames mapped      : {len(numeric_mapped)}")
    print(f"Numeric filenames missing     : {len(numeric_missing)}")
    print("-" * 108)
    print(f"Unexpected class txt files    : {unexpected_class_files}")
    print(f"Missing canonical classes     : {sorted(missing_canonical_classes)}")
    print("-" * 108)
    print("CLASS COUNTS")
    for name in RSICD_CLASSES:
        print(f"{name:20s}: {canonical_counts[name]}")
    print("-" * 108)
    print(
        f"CANONICAL 30 CLASSES          : "
        f"{'PASS' if canonical_30_pass else 'CHECK NEEDED'}"
    )
    print(
        f"EXACT FILENAME COVERAGE       : "
        f"{'PASS' if exact_coverage else 'CHECK NEEDED'}"
    )
    print(f"Report                        : {output_dir / 'report.json'}")
    print(
        f"Mapping                       : "
        f"{output_dir / 'filename_to_category.json'}"
    )
    print("=" * 108)

    if unexpected_class_files:
        print("注意：发现非标准类名 txt，请先不要自动用于训练。")
    if duplicate_mappings:
        print("注意：存在图像同时出现在多个类别 txt 中。")
    if missing_from_class_files:
        print("注意：dataset 中存在没有被类别 txt 覆盖的图像。")


if __name__ == "__main__":
    main()
