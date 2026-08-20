import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.re_dataset import (
    RSICD_CLASSES,
    get_rsicd_category_id,
    get_rsicd_category_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查 RSICD train annotation 的 category_id 解析结果。"
    )
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--preview", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_file, "r", encoding="utf-8") as f:
        ann = json.load(f)

    if not isinstance(ann, list):
        raise ValueError("Training annotation must be a list.")

    pair_counter = Counter()
    image_to_category = {}
    unknown_examples = []

    for idx, item in enumerate(ann):
        image = item["image"]
        category_id = get_rsicd_category_id(image)
        category_name = get_rsicd_category_name(category_id)

        pair_counter[category_name] += 1
        image_to_category.setdefault(image, category_name)

        if category_id < 0 and len(unknown_examples) < args.preview:
            unknown_examples.append((idx, image))

    image_counter = Counter(image_to_category.values())

    known_pairs = sum(
        count
        for name, count in pair_counter.items()
        if name != "unknown"
    )
    unknown_pairs = pair_counter["unknown"]

    known_images = sum(
        count
        for name, count in image_counter.items()
        if name != "unknown"
    )
    unknown_images = image_counter["unknown"]

    print("=" * 100)
    print("RSICD TRAIN CATEGORY CHECK")
    print("=" * 100)
    print(f"Training pairs        : {len(ann)}")
    print(f"Unique images         : {len(image_to_category)}")
    print(f"Known category pairs  : {known_pairs}")
    print(f"Unknown category pairs: {unknown_pairs}")
    print(f"Known category images : {known_images}")
    print(f"Unknown category imgs : {unknown_images}")
    print("-" * 100)

    print("KNOWN IMAGE COUNTS")
    for name in RSICD_CLASSES:
        print(f"{name:20s}: {image_counter[name]}")

    print("-" * 100)
    print("UNKNOWN EXAMPLES")
    for idx, image in unknown_examples:
        print(f"pair={idx:5d}  image={image}")

    parsed_known_classes = {
        name
        for name in image_counter
        if name != "unknown" and image_counter[name] > 0
    }
    missing_classes = [
        name
        for name in RSICD_CLASSES
        if name not in parsed_known_classes
    ]

    print("-" * 100)
    print(f"Parsed known classes  : {len(parsed_known_classes)}/30")
    print(f"Missing known classes : {missing_classes}")
    print("=" * 100)

    if len(parsed_known_classes) != 30:
        raise RuntimeError(
            "Not all 30 RSICD categories were parsed."
        )


if __name__ == "__main__":
    main()
