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
        description="统计 RSICD T2I Top-1 检索错误中的已知类别错检比例，并排除纯数字文件名样本。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/t2i_category_error_analysis_known_unknown",
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


def parse_category_from_ann(ann):
    """
    类别来源：
    1) 若 annotation 有显式 label，直接使用；
    2) 否则从文件名解析类别前缀：
       school_57.jpg -> school
       dense_residential_12.jpg -> dense_residential
    3) 纯数字文件名如 00001.jpg -> unknown，不猜类别。
    """
    if "label" in ann:
        label = ann["label"]
        return label, ann.get("label_name", str(label)), "annotation_label"

    image_ref = ann.get("image")
    if not image_ref:
        return None, "unknown", "unknown"

    stem = Path(str(image_ref)).stem
    if stem.isdigit():
        return None, "unknown", "unknown_numeric"

    prefix, sep, suffix = stem.rpartition("_")
    if sep and prefix and suffix.isdigit():
        label = prefix.lower()
        return label, label, "filename_prefix"

    return None, "unknown", "unknown_unparsed"


def get_image_labels(dataset):
    labels = []
    label_names = []
    source_counter = Counter()

    for ann in dataset.ann:
        label, name, source = parse_category_from_ann(ann)
        labels.append(label)
        label_names.append(name)
        source_counter[source] += 1

    return labels, label_names, dict(source_counter)


def build_label_name_map(labels, label_names):
    names_by_label = defaultdict(Counter)
    for label, name in zip(labels, label_names):
        if label is None:
            continue
        names_by_label[label][str(name)] += 1

    display = {}
    for label, counter in names_by_label.items():
        name, _ = counter.most_common(1)[0]
        display[label] = f"{label}:{name}" if str(label) != name else name
    return display


def analyze_t2i(scores_i2t, dataset, labels):
    scores_t2i = scores_i2t.T

    rows = []
    exact_correct = 0
    exact_errors = 0

    known_pair_queries = 0
    category_correct_known_pairs = 0

    same_class_wrong = 0
    cross_class_wrong = 0
    unknown_category_errors = 0

    confusion = Counter()
    per_gt_class = defaultdict(lambda: {
        "queries": 0,
        "exact_correct": 0,
        "known_pair_queries": 0,
        "same_class_top1": 0,
        "same_class_wrong": 0,
        "cross_class_wrong": 0,
        "unknown_pred_category": 0,
    })

    for text_id, scores in enumerate(scores_t2i):
        gt_image_id = int(dataset.txt2img[text_id])
        pred_image_id = int(np.argmax(scores))

        gt_label = labels[gt_image_id]
        pred_label = labels[pred_image_id]

        exact_match = pred_image_id == gt_image_id
        resolvable = gt_label is not None and pred_label is not None
        class_match = resolvable and gt_label == pred_label

        gt_key = gt_label if gt_label is not None else "__unknown__"
        stats = per_gt_class[gt_key]
        stats["queries"] += 1

        if exact_match:
            exact_correct += 1
            stats["exact_correct"] += 1
        else:
            exact_errors += 1

        if resolvable:
            known_pair_queries += 1
            stats["known_pair_queries"] += 1
            if class_match:
                category_correct_known_pairs += 1
                stats["same_class_top1"] += 1

            if not exact_match:
                if class_match:
                    same_class_wrong += 1
                    stats["same_class_wrong"] += 1
                else:
                    cross_class_wrong += 1
                    stats["cross_class_wrong"] += 1
                    confusion[(gt_label, pred_label)] += 1
        elif not exact_match:
            unknown_category_errors += 1
            stats["unknown_pred_category"] += 1

        rows.append({
            "text_id": text_id,
            "query": dataset.text[text_id],
            "gt_image_id": gt_image_id,
            "pred_image_id": pred_image_id,
            "gt_label": gt_label if gt_label is not None else "unknown",
            "pred_label": pred_label if pred_label is not None else "unknown",
            "exact_match": int(exact_match),
            "category_resolvable": int(resolvable),
            "class_match": int(class_match) if resolvable else "",
            "gt_score": float(scores[gt_image_id]),
            "pred_score": float(scores[pred_image_id]),
            "pred_minus_gt": float(scores[pred_image_id] - scores[gt_image_id]),
        })

    total = len(rows)
    resolvable_errors = same_class_wrong + cross_class_wrong

    return rows, {
        "num_queries": total,
        "exact_top1_correct": exact_correct,
        "exact_top1_accuracy": exact_correct / total,
        "exact_top1_errors": exact_errors,
        "known_pair_queries": known_pair_queries,
        "known_pair_query_coverage": known_pair_queries / total,
        "category_top1_accuracy_on_known_pairs": (
            category_correct_known_pairs / known_pair_queries
            if known_pair_queries else 0.0
        ),
        "resolvable_retrieval_errors": resolvable_errors,
        "same_class_wrong": same_class_wrong,
        "cross_class_wrong": cross_class_wrong,
        "unknown_category_errors": unknown_category_errors,
        "same_class_wrong_fraction_among_resolvable_errors": (
            same_class_wrong / resolvable_errors if resolvable_errors else 0.0
        ),
        "cross_class_wrong_fraction_among_resolvable_errors": (
            cross_class_wrong / resolvable_errors if resolvable_errors else 0.0
        ),
        "unknown_fraction_among_retrieval_errors": (
            unknown_category_errors / exact_errors if exact_errors else 0.0
        ),
    }, confusion, per_gt_class


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

    labels, label_names, label_source = get_image_labels(dataset)
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
    print("RSICD T2I CATEGORY ERROR ANALYSIS — KNOWN / UNKNOWN AWARE")
    print("=" * 104)
    print(f"Split        : {args.split}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Epoch        : {checkpoint.get('epoch', 'unknown')}")
    print(f"Images       : {len(dataset.image)}")
    print(f"Captions     : {len(dataset.text)}")
    print(f"Known classes: {len({x for x in labels if x is not None})}")
    print(f"Label source : {label_source}")
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
        known = stats["known_pair_queries"]
        per_class_rows.append({
            "label": label,
            "display": label_display.get(label, str(label)),
            **stats,
            "exact_top1_accuracy": stats["exact_correct"] / n if n else 0.0,
            "category_top1_accuracy_on_known_pairs": (
                stats["same_class_top1"] / known if known else ""
            ),
            "cross_class_error_rate_all_queries": (
                stats["cross_class_wrong"] / n if n else 0.0
            ),
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
        "label_source": label_source,
        "num_known_classes": len({x for x in labels if x is not None}),
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
        f"Exact T2I Top-1 accuracy                  : "
        f"{summary['exact_top1_accuracy']:.2%} "
        f"({summary['exact_top1_correct']}/{summary['num_queries']})"
    )
    print(
        f"Exact T2I Top-1 errors                    : "
        f"{summary['exact_top1_errors']}"
    )
    print(
        f"Known-category pair coverage              : "
        f"{summary['known_pair_query_coverage']:.2%} "
        f"({summary['known_pair_queries']}/{summary['num_queries']})"
    )
    print(
        f"Category Top-1 accuracy on known pairs    : "
        f"{summary['category_top1_accuracy_on_known_pairs']:.2%}"
    )
    print("-" * 104)
    print(
        f"Resolvable retrieval errors               : "
        f"{summary['resolvable_retrieval_errors']}"
    )
    print(
        f"Among resolvable errors -> same class     : "
        f"{summary['same_class_wrong_fraction_among_resolvable_errors']:.2%} "
        f"({summary['same_class_wrong']}/{summary['resolvable_retrieval_errors']})"
    )
    print(
        f"Among resolvable errors -> cross class    : "
        f"{summary['cross_class_wrong_fraction_among_resolvable_errors']:.2%} "
        f"({summary['cross_class_wrong']}/{summary['resolvable_retrieval_errors']})"
    )
    print(
        f"Unknown-category errors / all errors       : "
        f"{summary['unknown_fraction_among_retrieval_errors']:.2%} "
        f"({summary['unknown_category_errors']}/{summary['exact_top1_errors']})"
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
