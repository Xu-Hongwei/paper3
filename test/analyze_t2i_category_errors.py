import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset, create_loader
from evaluation import evaluate_retrieval
from models import CLIPRetrieval
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计 RSICD T2I Top-1 检索错误中的类别错误比例。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/t2i_category_error_analysis",
    )
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def get_image_labels(dataset):
    """
    直接使用 evaluation annotation 自带的 numeric `label`。
    不从文件名、caption 或 label_name 推断类别。
    """
    labels = []
    label_names = []

    for image_id, ann in enumerate(dataset.ann):
        if "label" not in ann:
            raise KeyError(
                f"Evaluation annotation image_id={image_id} 缺少 `label`。"
                f"当前 keys={list(ann.keys())}"
            )

        labels.append(ann["label"])
        label_names.append(ann.get("label_name"))

    return labels, label_names


def build_label_name_map(labels, label_names):
    """
    label_name 仅用于辅助显示，不参与类别是否相同的判定。
    若同一 numeric label 对应多个名字，则显示最常见名字并标记数量。
    """
    names_by_label = defaultdict(Counter)

    for label, name in zip(labels, label_names):
        if name is None:
            continue
        names_by_label[label][str(name)] += 1

    display = {}
    for label in set(labels):
        counter = names_by_label.get(label, Counter())
        if not counter:
            display[label] = str(label)
            continue

        name, count = counter.most_common(1)[0]
        if len(counter) == 1:
            display[label] = f"{label}:{name}"
        else:
            display[label] = f"{label}:{name} (+{len(counter)-1} names)"

    return display


def analyze_t2i(scores_i2t, dataset, labels):
    """
    scores_i2t: [num_images, num_texts]
    对每条 text 在 image 维做 Top-1。
    """
    scores_t2i = scores_i2t.T

    rows = []
    exact_correct = 0
    same_class_top1 = 0
    exact_errors = 0
    class_errors_within_exact_errors = 0
    same_class_wrong = 0

    confusion = Counter()
    per_gt_class = defaultdict(lambda: {
        "queries": 0,
        "exact_correct": 0,
        "same_class_top1": 0,
        "same_class_wrong": 0,
        "cross_class_wrong": 0,
    })

    for text_id, scores in enumerate(scores_t2i):
        gt_image_id = int(dataset.txt2img[text_id])
        pred_image_id = int(np.argmax(scores))

        gt_label = labels[gt_image_id]
        pred_label = labels[pred_image_id]

        exact_match = pred_image_id == gt_image_id
        class_match = pred_label == gt_label

        stats = per_gt_class[gt_label]
        stats["queries"] += 1
        stats["exact_correct"] += int(exact_match)
        stats["same_class_top1"] += int(class_match)

        if exact_match:
            exact_correct += 1
        else:
            exact_errors += 1
            if class_match:
                same_class_wrong += 1
                stats["same_class_wrong"] += 1
            else:
                class_errors_within_exact_errors += 1
                stats["cross_class_wrong"] += 1
                confusion[(gt_label, pred_label)] += 1

        same_class_top1 += int(class_match)

        rows.append({
            "text_id": text_id,
            "query": dataset.text[text_id],
            "gt_image_id": gt_image_id,
            "pred_image_id": pred_image_id,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "exact_match": int(exact_match),
            "class_match": int(class_match),
            "gt_score": float(scores[gt_image_id]),
            "pred_score": float(scores[pred_image_id]),
            "pred_minus_gt": float(scores[pred_image_id] - scores[gt_image_id]),
        })

    total = len(rows)

    summary = {
        "num_queries": total,
        "exact_top1_correct": exact_correct,
        "exact_top1_accuracy": exact_correct / total,
        "exact_top1_errors": exact_errors,
        "same_class_top1": same_class_top1,
        "category_top1_accuracy": same_class_top1 / total,
        "category_top1_error_rate": 1.0 - same_class_top1 / total,
        "same_class_wrong": same_class_wrong,
        "cross_class_wrong": class_errors_within_exact_errors,
        "same_class_wrong_fraction_among_retrieval_errors": (
            same_class_wrong / exact_errors if exact_errors else 0.0
        ),
        "cross_class_wrong_fraction_among_retrieval_errors": (
            class_errors_within_exact_errors / exact_errors if exact_errors else 0.0
        ),
    }

    return rows, summary, confusion, per_gt_class


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )

    labels, label_names = get_image_labels(dataset)
    label_display = build_label_name_map(labels, label_names)

    image_batch_size = int(
        args.image_batch_size
        or config["training"].get("eval_batch_size", 128)
    )
    text_batch_size = int(
        args.text_batch_size
        or config["training"].get("text_batch_size", 256)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else config["training"].get("num_workers", 4)
    )

    loader = create_loader(
        dataset,
        batch_size=image_batch_size,
        num_workers=num_workers,
        is_train=False,
        pin_memory=True,
    )

    print("=" * 104)
    print("RSICD T2I CATEGORY ERROR ANALYSIS")
    print("=" * 104)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Images     : {len(dataset.image)}")
    print(f"Captions   : {len(dataset.text)}")
    print(f"Classes    : {len(set(labels))}")
    print("Class key  : annotation['label'] (numeric, authoritative)")
    print("=" * 104)

    metrics, scores_i2t = evaluate_retrieval(
        model=model,
        data_loader=loader,
        dataset=dataset,
        device=device,
        text_batch_size=text_batch_size,
    )

    rows, summary, confusion, per_gt_class = analyze_t2i(
        scores_i2t,
        dataset,
        labels,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "t2i_all_queries_with_category.csv",
        rows,
        list(rows[0].keys()),
    )

    mismatch_rows = []
    for (gt_label, pred_label), count in confusion.most_common():
        mismatch_rows.append({
            "gt_label": gt_label,
            "gt_display": label_display.get(gt_label, str(gt_label)),
            "pred_label": pred_label,
            "pred_display": label_display.get(pred_label, str(pred_label)),
            "count": count,
        })

    write_csv(
        output_dir / "cross_category_confusion.csv",
        mismatch_rows,
        ["gt_label", "gt_display", "pred_label", "pred_display", "count"],
    )

    per_class_rows = []
    for label, stats in sorted(
        per_gt_class.items(),
        key=lambda x: (-x[1]["queries"], str(x[0])),
    ):
        n = stats["queries"]
        per_class_rows.append({
            "label": label,
            "display": label_display.get(label, str(label)),
            **stats,
            "exact_top1_accuracy": stats["exact_correct"] / n,
            "category_top1_accuracy": stats["same_class_top1"] / n,
            "cross_class_error_rate": stats["cross_class_wrong"] / n,
        })

    write_csv(
        output_dir / "per_class_category_accuracy.csv",
        per_class_rows,
        list(per_class_rows[0].keys()),
    )

    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "retrieval_metrics": metrics,
        "category_analysis": summary,
        "num_classes": len(set(labels)),
        "top_cross_category_confusions": mismatch_rows[:30],
    }

    with (output_dir / "category_error_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 104)
    print("CORE RESULT")
    print("=" * 104)
    print(
        f"Exact T2I Top-1 accuracy                 : "
        f"{summary['exact_top1_accuracy']:.2%} "
        f"({summary['exact_top1_correct']}/{summary['num_queries']})"
    )
    print(
        f"Exact T2I Top-1 errors                   : "
        f"{summary['exact_top1_errors']}"
    )
    print(
        f"Category Top-1 accuracy                  : "
        f"{summary['category_top1_accuracy']:.2%} "
        f"({summary['same_class_top1']}/{summary['num_queries']})"
    )
    print(
        f"Category Top-1 error rate                : "
        f"{summary['category_top1_error_rate']:.2%}"
    )
    print("-" * 104)
    print(
        f"Among exact retrieval errors -> same class wrong : "
        f"{summary['same_class_wrong_fraction_among_retrieval_errors']:.2%} "
        f"({summary['same_class_wrong']}/{summary['exact_top1_errors']})"
    )
    print(
        f"Among exact retrieval errors -> cross class wrong: "
        f"{summary['cross_class_wrong_fraction_among_retrieval_errors']:.2%} "
        f"({summary['cross_class_wrong']}/{summary['exact_top1_errors']})"
    )

    print("\nTOP CROSS-CATEGORY CONFUSIONS")
    print("-" * 104)
    for row in mismatch_rows[:20]:
        print(
            f"{row['gt_display']:<35} -> "
            f"{row['pred_display']:<35} "
            f"{row['count']:>4}"
        )

    print("\n" + "=" * 104)
    print("OUTPUTS")
    print("=" * 104)
    print(f"Summary    : {output_dir / 'category_error_summary.json'}")
    print(f"All queries: {output_dir / 't2i_all_queries_with_category.csv'}")
    print(f"Confusions : {output_dir / 'cross_category_confusion.csv'}")
    print(f"Per class  : {output_dir / 'per_class_category_accuracy.csv'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
