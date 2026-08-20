import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset, create_loader
from models import CLIPRetrieval
from utils import load_config, set_seed
from evaluation.retrieval import (
    extract_image_features,
    extract_text_features,
    compute_similarity_matrix,
)


CATEGORY_ALIASES = {
    "airport": ["airport", "airports"],
    "bareland": ["bare land", "bareland", "barren land"],
    "baseballfield": ["baseball field", "baseball fields", "baseballfield", "baseballfields"],
    "beach": ["beach", "beaches"],
    "bridge": ["bridge", "bridges"],
    "center": ["center", "centre"],
    "church": ["church", "churches"],
    "commercial": ["commercial area", "commercial areas", "commercial district", "commercial districts"],
    "denseresidential": ["dense residential", "dense residential area", "dense residential areas"],
    "desert": ["desert", "deserts"],
    "farmland": ["farmland", "farmlands", "farm land", "farm lands", "farm", "farms"],
    "forest": ["forest", "forests"],
    "industrial": ["industrial area", "industrial areas", "industrial zone", "industrial zones"],
    "meadow": ["meadow", "meadows"],
    "mediumresidential": ["medium residential", "medium residential area", "medium residential areas"],
    "mountain": ["mountain", "mountains", "mountainous area", "mountainous areas"],
    "park": ["park", "parks"],
    "parkinglot": ["parking lot", "parking lots", "parkinglot", "parkinglots"],
    "playground": ["playground", "playgrounds"],
    "pond": ["pond", "ponds"],
    "port": ["port", "ports", "harbor", "harbour", "harbors", "harbours"],
    "railwaystation": ["railway station", "railway stations", "train station", "train stations"],
    "resort": ["resort", "resorts"],
    "river": ["river", "rivers"],
    "school": ["school", "schools", "campus", "campuses"],
    "sparseresidential": ["sparse residential", "sparse residential area", "sparse residential areas"],
    "square": ["square", "squares", "plaza", "plazas"],
    "stadium": ["stadium", "stadiums", "stadia"],
    "storagetanks": ["storage tank", "storage tanks", "storagetank", "storagetanks"],
    "viaduct": ["viaduct", "viaducts", "overpass", "overpasses"],
}

CATEGORY_PROMPTS = {
    "airport": "airport",
    "bareland": "bare land",
    "baseballfield": "baseball field",
    "beach": "beach",
    "bridge": "bridge",
    "center": "city center",
    "church": "church",
    "commercial": "commercial area",
    "denseresidential": "dense residential area",
    "desert": "desert",
    "farmland": "farmland",
    "forest": "forest",
    "industrial": "industrial area",
    "meadow": "meadow",
    "mediumresidential": "medium residential area",
    "mountain": "mountain",
    "park": "park",
    "parkinglot": "parking lot",
    "playground": "playground",
    "pond": "pond",
    "port": "port",
    "railwaystation": "railway station",
    "resort": "resort",
    "river": "river",
    "school": "school",
    "sparseresidential": "sparse residential area",
    "square": "square",
    "stadium": "stadium",
    "storagetanks": "storage tanks",
    "viaduct": "viaduct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 explicit category anchor rank 对 RSICD 跨类别错误的可分性。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/anchor_rank_discriminability",
    )
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def phrase_present(text, phrase):
    text = normalize_text(text)
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def extract_explicit_categories(caption):
    hits = []
    for category, aliases in CATEGORY_ALIASES.items():
        if any(phrase_present(caption, alias) for alias in aliases):
            hits.append(category)
    return hits


def parse_image_category(ann):
    if "label" in ann:
        return ann["label"]

    image_ref = ann.get("image", "")
    stem = Path(str(image_ref)).stem
    if stem.isdigit():
        return None

    prefix, sep, suffix = stem.rpartition("_")
    if sep and prefix and suffix.isdigit():
        return prefix.lower()

    return None


def baseline_orders(scores_i2t):
    return [np.argsort(scores)[::-1] for scores in scores_i2t.T]


def anchor_rank_in_topn(base_order, anchor_scores, topn):
    n = min(topn, len(base_order))
    top_ids = np.asarray(base_order[:n], dtype=np.int64)
    base_top1 = int(top_ids[0])
    anchor_order = top_ids[np.argsort(anchor_scores[top_ids])[::-1]]
    return int(np.where(anchor_order == base_top1)[0][0]) + 1


def safe_div(a, b):
    return float(a / b) if b else 0.0


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def compute_pr_curve(rows, positive_key, topn):
    """
    Trigger: anchor_rank > T
    在 category-resolvable single-anchor queries 上评估。
    """
    eval_rows = [r for r in rows if r["category_resolvable"] == 1]
    positives_total = sum(int(r[positive_key]) for r in eval_rows)
    curve = []

    for threshold in range(1, topn):
        tp = fp = fn = tn = 0

        for row in eval_rows:
            pred = row["anchor_rank"] > threshold
            target = bool(row[positive_key])

            if pred and target:
                tp += 1
            elif pred and not target:
                fp += 1
            elif not pred and target:
                fn += 1
            else:
                tn += 1

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        curve.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "triggered": tp + fp,
            "positives_total": positives_total,
        })

    return curve


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histogram(rows, output_path, topn):
    groups = {
        "Exact correct": [r["anchor_rank"] for r in rows if r["cohort"] == "exact_correct"],
        "Same-category wrong": [r["anchor_rank"] for r in rows if r["cohort"] == "same_category_wrong"],
        "Cross-category wrong": [r["anchor_rank"] for r in rows if r["cohort"] == "cross_category_wrong"],
        "Strict GT-supported cross": [r["anchor_rank"] for r in rows if r["strict_target"] == 1],
    }

    plt.figure(figsize=(10, 6))
    bins = np.arange(0.5, topn + 1.5, 5)

    for name, values in groups.items():
        if values:
            plt.hist(values, bins=bins, histtype="step", linewidth=2, density=True, label=f"{name} (n={len(values)})")

    plt.xlabel("Anchor rank of baseline Top-1 within baseline Top-N")
    plt.ylabel("Density")
    plt.title("Anchor-rank distribution by baseline retrieval outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_cdf(rows, output_path, topn):
    groups = {
        "Exact correct": [r["anchor_rank"] for r in rows if r["cohort"] == "exact_correct"],
        "Same-category wrong": [r["anchor_rank"] for r in rows if r["cohort"] == "same_category_wrong"],
        "Cross-category wrong": [r["anchor_rank"] for r in rows if r["cohort"] == "cross_category_wrong"],
        "Strict GT-supported cross": [r["anchor_rank"] for r in rows if r["strict_target"] == 1],
    }

    plt.figure(figsize=(10, 6))
    xs = np.arange(1, topn + 1)

    for name, values in groups.items():
        if not values:
            continue
        values = np.asarray(values)
        ys = np.asarray([(values <= x).mean() for x in xs])
        plt.plot(xs, ys, linewidth=2, label=f"{name} (n={len(values)})")

    plt.xlabel("Anchor rank threshold")
    plt.ylabel("CDF: P(anchor rank <= threshold)")
    plt.title("CDF of anchor rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pr(curve, title, output_path):
    plt.figure(figsize=(7, 6))
    recall = [x["recall"] for x in curve]
    precision = [x["precision"] for x in curve]
    plt.plot(recall, precision, marker="o", markersize=3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_rank_margin_scatter(rows, output_path):
    valid = [r for r in rows if r["category_resolvable"] == 1]

    plt.figure(figsize=(10, 7))
    cohorts = ["exact_correct", "same_category_wrong", "cross_category_wrong"]

    for cohort in cohorts:
        subset = [r for r in valid if r["cohort"] == cohort]
        if not subset:
            continue

        x = [r["anchor_rank"] for r in subset]
        y = [r["global_top1_top2_margin"] for r in subset]
        plt.scatter(x, y, s=18, alpha=0.45, label=f"{cohort} (n={len(subset)})")

    plt.xlabel("Anchor rank of baseline Top-1")
    plt.ylabel("Global Top1 - Top2 similarity margin")
    plt.title("Anchor inconsistency vs. full-caption confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


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

    print("=" * 118)
    print("ANCHOR RANK DISCRIMINABILITY ANALYSIS")
    print("=" * 118)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Top-N      : {args.topn}")
    print("=" * 118)

    print("Extracting image features...")
    image_features = extract_image_features(model, loader, device)

    print("Extracting full-caption text features...")
    text_features = extract_text_features(
        model,
        dataset.text,
        device,
        batch_size=text_batch_size,
    )

    print("Computing baseline similarity...")
    scores_i2t = compute_similarity_matrix(
        image_features,
        text_features,
        device,
    )
    scores_t2i = scores_i2t.T
    base_orders = baseline_orders(scores_i2t)

    query_categories = [
        extract_explicit_categories(text)
        for text in dataset.text
    ]
    single_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) == 1
    ]

    used_categories = sorted({
        query_categories[i][0]
        for i in single_anchor_queries
    })
    prompts = [CATEGORY_PROMPTS[c] for c in used_categories]

    print(f"Encoding {len(prompts)} category anchors...")
    anchor_text_features = extract_text_features(
        model,
        prompts,
        device,
        batch_size=text_batch_size,
    )

    anchor_scores_matrix = compute_similarity_matrix(
        image_features,
        anchor_text_features,
        device,
    ).T
    category_to_anchor_scores = {
        category: anchor_scores_matrix[i]
        for i, category in enumerate(used_categories)
    }

    image_labels = [
        parse_image_category(ann)
        for ann in dataset.ann
    ]

    rows = []

    for text_id in single_anchor_queries:
        gt_id = int(dataset.txt2img[text_id])
        order = base_orders[text_id]
        pred_id = int(order[0])

        gt_label = image_labels[gt_id]
        pred_label = image_labels[pred_id]
        resolvable = gt_label is not None and pred_label is not None

        if pred_id == gt_id:
            cohort = "exact_correct"
        elif resolvable and gt_label == pred_label:
            cohort = "same_category_wrong"
        elif resolvable and gt_label != pred_label:
            cohort = "cross_category_wrong"
        else:
            cohort = "unknown_category"

        category = query_categories[text_id][0]
        anchor_scores = category_to_anchor_scores[category]
        anchor_rank = anchor_rank_in_topn(
            base_order=order,
            anchor_scores=anchor_scores,
            topn=args.topn,
        )

        top1_score = float(scores_t2i[text_id, order[0]])
        top2_score = float(scores_t2i[text_id, order[1]])
        gt_score = float(scores_t2i[text_id, gt_id])

        strict_target = int(
            cohort == "cross_category_wrong"
            and category == gt_label
            and category != pred_label
        )

        rows.append({
            "text_id": text_id,
            "query": dataset.text[text_id],
            "explicit_category": category,
            "gt_image_id": gt_id,
            "pred_image_id": pred_id,
            "gt_label": gt_label if gt_label is not None else "unknown",
            "pred_label": pred_label if pred_label is not None else "unknown",
            "category_resolvable": int(resolvable),
            "cohort": cohort,
            "strict_target": strict_target,
            "anchor_rank": anchor_rank,
            "global_top1_score": top1_score,
            "global_top2_score": top2_score,
            "global_gt_score": gt_score,
            "global_top1_top2_margin": top1_score - top2_score,
            "global_top1_gt_margin": top1_score - gt_score,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "single_anchor_query_diagnostic.csv", rows)

    cohort_names = [
        "exact_correct",
        "same_category_wrong",
        "cross_category_wrong",
        "unknown_category",
    ]

    cohort_summary = {}
    for cohort in cohort_names:
        subset = [r for r in rows if r["cohort"] == cohort]
        cohort_summary[cohort] = {
            "count": len(subset),
            "anchor_rank": percentile_summary([r["anchor_rank"] for r in subset]),
            "top1_top2_margin": percentile_summary([r["global_top1_top2_margin"] for r in subset]),
        }

    strict_rows = [r for r in rows if r["strict_target"] == 1]
    cohort_summary["strict_target"] = {
        "count": len(strict_rows),
        "anchor_rank": percentile_summary([r["anchor_rank"] for r in strict_rows]),
        "top1_top2_margin": percentile_summary([r["global_top1_top2_margin"] for r in strict_rows]),
    }

    # 为 PR 补两个布尔字段，不写回主 CSV。
    pr_rows = []
    for row in rows:
        x = dict(row)
        x["is_cross_error"] = int(row["cohort"] == "cross_category_wrong")
        x["is_strict_target"] = int(row["strict_target"] == 1)
        pr_rows.append(x)

    cross_pr = compute_pr_curve(pr_rows, "is_cross_error", args.topn)
    strict_pr = compute_pr_curve(pr_rows, "is_strict_target", args.topn)

    write_csv(output_dir / "cross_error_pr_curve.csv", cross_pr)
    write_csv(output_dir / "strict_target_pr_curve.csv", strict_pr)

    best_cross = max(cross_pr, key=lambda x: x["f1"])
    best_strict = max(strict_pr, key=lambda x: x["f1"])

    plot_histogram(
        rows,
        output_dir / "anchor_rank_histogram.png",
        args.topn,
    )
    plot_cdf(
        rows,
        output_dir / "anchor_rank_cdf.png",
        args.topn,
    )
    plot_pr(
        cross_pr,
        "Cross-category error detection by anchor-rank threshold",
        output_dir / "cross_error_pr_curve.png",
    )
    plot_pr(
        strict_pr,
        "Strict GT-supported cross-error detection by anchor-rank threshold",
        output_dir / "strict_target_pr_curve.png",
    )
    plot_rank_margin_scatter(
        rows,
        output_dir / "anchor_rank_vs_global_margin.png",
    )

    # 额外输出若干常用阈值的浓度统计。
    threshold_rows = []
    for threshold in [10, 15, 20, 25, 30, 35, 40, 45]:
        if threshold >= args.topn:
            continue

        triggered = [r for r in rows if r["anchor_rank"] > threshold]
        resolvable_triggered = [r for r in triggered if r["category_resolvable"] == 1]

        cross = sum(r["cohort"] == "cross_category_wrong" for r in resolvable_triggered)
        strict = sum(r["strict_target"] == 1 for r in resolvable_triggered)
        exact = sum(r["cohort"] == "exact_correct" for r in resolvable_triggered)
        same = sum(r["cohort"] == "same_category_wrong" for r in resolvable_triggered)

        threshold_rows.append({
            "threshold": threshold,
            "triggered_all_single_anchor": len(triggered),
            "triggered_resolvable": len(resolvable_triggered),
            "cross_count": cross,
            "cross_fraction_among_resolvable_triggered": safe_div(cross, len(resolvable_triggered)),
            "strict_count": strict,
            "strict_fraction_among_resolvable_triggered": safe_div(strict, len(resolvable_triggered)),
            "exact_correct_count": exact,
            "same_category_wrong_count": same,
        })

    write_csv(output_dir / "threshold_concentration.csv", threshold_rows)

    summary = {
        "metadata": {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "topn": args.topn,
            "num_queries": len(dataset.text),
            "num_single_anchor_queries": len(rows),
            "test_label_used_by_signal": False,
            "labels_used_for_offline_cohort_analysis_only": True,
        },
        "cohort_summary": cohort_summary,
        "best_cross_error_detector_by_f1": best_cross,
        "best_strict_target_detector_by_f1": best_strict,
        "threshold_concentration": threshold_rows,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nCOHORT SUMMARY")
    print("-" * 118)
    for cohort in [
        "exact_correct",
        "same_category_wrong",
        "cross_category_wrong",
        "strict_target",
        "unknown_category",
    ]:
        item = cohort_summary[cohort]
        ar = item["anchor_rank"]
        if not ar:
            continue

        print(
            f"{cohort:<28} n={item['count']:4d} | "
            f"anchor mean={ar['mean']:.2f} med={ar['median']:.1f} "
            f"p75={ar['p75']:.1f} p90={ar['p90']:.1f}"
        )

    print("\nBEST ANCHOR-RANK DETECTION THRESHOLDS")
    print("-" * 118)
    print(
        f"Cross-category error | T={best_cross['threshold']:2d} "
        f"P={best_cross['precision']:.3f} R={best_cross['recall']:.3f} "
        f"F1={best_cross['f1']:.3f} triggered={best_cross['triggered']}"
    )
    print(
        f"Strict target        | T={best_strict['threshold']:2d} "
        f"P={best_strict['precision']:.3f} R={best_strict['recall']:.3f} "
        f"F1={best_strict['f1']:.3f} triggered={best_strict['triggered']}"
    )

    print("\nKEY OUTPUTS")
    print("-" * 118)
    print(f"Summary      : {output_dir / 'summary.json'}")
    print(f"Per-query    : {output_dir / 'single_anchor_query_diagnostic.csv'}")
    print(f"Thresholds   : {output_dir / 'threshold_concentration.csv'}")
    print(f"Cross PR     : {output_dir / 'cross_error_pr_curve.csv'}")
    print(f"Strict PR    : {output_dir / 'strict_target_pr_curve.csv'}")
    print(f"Histogram    : {output_dir / 'anchor_rank_histogram.png'}")
    print(f"CDF          : {output_dir / 'anchor_rank_cdf.png'}")
    print(f"2D scatter   : {output_dir / 'anchor_rank_vs_global_margin.png'}")
    print("=" * 118)


if __name__ == "__main__":
    main()
