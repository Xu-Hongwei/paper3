import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset, create_loader
from models import CLIPRetrieval
from utils import load_config, set_seed
from evaluation.retrieval import extract_image_features, extract_text_features, compute_similarity_matrix


# 只保留高精度、可直接从 caption 中显式识别的 RSICD 类别。
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
        description="Explicit category anchor rerank diagnostic for RSICD T2I."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/explicit_category_anchor_rerank",
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


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def baseline_orders(scores_i2t):
    scores_t2i = scores_i2t.T
    return [np.argsort(scores)[::-1] for scores in scores_t2i]


def rerank_order(base_scores, base_order, anchor_scores, topn, lam):
    """
    只重排 baseline Top-N，Top-N 之外保持原始顺序。
    这样不会把融合分数与 Top-N 外的原始 cosine 直接混用。
    """
    n = min(topn, len(base_order))
    candidate_ids = np.asarray(base_order[:n], dtype=np.int64)

    g = zscore(base_scores[candidate_ids])
    a = zscore(anchor_scores[candidate_ids])
    fused = g + lam * a
    local_order = candidate_ids[np.argsort(fused)[::-1]]

    if n == len(base_order):
        return local_order

    return np.concatenate([local_order, np.asarray(base_order[n:], dtype=np.int64)])


def rank_of(order, image_id):
    return int(np.where(order == image_id)[0][0]) + 1


def retrieval_metrics_from_orders(orders, txt2img):
    ranks = np.asarray(
        [rank_of(order, int(txt2img[text_id])) for text_id, order in enumerate(orders)],
        dtype=np.int32,
    )
    return {
        "r1": float(np.mean(ranks <= 1) * 100.0),
        "r5": float(np.mean(ranks <= 5) * 100.0),
        "r10": float(np.mean(ranks <= 10) * 100.0),
        "medr": float(np.median(ranks)),
        "meanr": float(np.mean(ranks)),
    }, ranks


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

    print("=" * 112)
    print("EXPLICIT CATEGORY ANCHOR RERANK DIAGNOSTIC")
    print("=" * 112)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Top-N      : {args.topn}")
    print(f"Lambdas    : {args.lambdas}")
    print("Rule       : rerank only when caption contains EXACTLY ONE explicit RSICD category")
    print("No test label is used by the reranker.")
    print("=" * 112)

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
    base_metrics, base_ranks = retrieval_metrics_from_orders(
        base_orders,
        dataset.txt2img,
    )

    # ------------------------------------------------------------
    # Query 中显式类别抽取：推理时只看 caption，不看 GT label。
    # ------------------------------------------------------------
    query_categories = [extract_explicit_categories(text) for text in dataset.text]
    single_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) == 1
    ]
    multi_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) > 1
    ]
    no_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) == 0
    ]

    used_categories = sorted({
        query_categories[i][0] for i in single_anchor_queries
    })
    prompts = [CATEGORY_PROMPTS[c] for c in used_categories]

    print(f"Encoding {len(prompts)} category anchor prompts...")
    anchor_text_features = extract_text_features(
        model,
        prompts,
        device,
        batch_size=text_batch_size,
    )
    category_to_feature = {
        category: anchor_text_features[i]
        for i, category in enumerate(used_categories)
    }

    # [category, image]
    image_features_cpu = image_features.float()
    category_anchor_scores = {}
    for category, feature in category_to_feature.items():
        category_anchor_scores[category] = (
            image_features_cpu @ feature.float()
        ).numpy()

    image_labels = [parse_image_category(ann) for ann in dataset.ann]

    # ------------------------------------------------------------
    # 定义最干净的诊断目标：
    # baseline Top1 跨类别错误 + GT 类别显式出现 + baseline Pred 类别未显式出现。
    # 注意：这些 label 只用于评估/分组，不参与 rerank。
    # ------------------------------------------------------------
    target_queries = []
    for text_id in range(len(dataset.text)):
        gt_id = int(dataset.txt2img[text_id])
        pred_id = int(base_orders[text_id][0])
        if pred_id == gt_id:
            continue

        gt_label = image_labels[gt_id]
        pred_label = image_labels[pred_id]
        if gt_label is None or pred_label is None or gt_label == pred_label:
            continue

        cats = query_categories[text_id]
        if gt_label in cats and pred_label not in cats:
            target_queries.append(text_id)

    # 更严格的第一步：只对“唯一显式类别 == GT”的 target 看效果。
    target_single_queries = [
        i for i in target_queries
        if len(query_categories[i]) == 1
        and query_categories[i][0] == image_labels[int(dataset.txt2img[i])]
    ]

    print("\nQUERY COVERAGE")
    print("-" * 112)
    print(f"No explicit category      : {len(no_anchor_queries):4d} / {len(dataset.text)} = {len(no_anchor_queries)/len(dataset.text):.2%}")
    print(f"Exactly one category      : {len(single_anchor_queries):4d} / {len(dataset.text)} = {len(single_anchor_queries)/len(dataset.text):.2%}")
    print(f"Multiple categories       : {len(multi_anchor_queries):4d} / {len(dataset.text)} = {len(multi_anchor_queries)/len(dataset.text):.2%}")
    print(f"GT_yes / Pred_no baseline : {len(target_queries):4d}")
    print(f"Strict single-GT target   : {len(target_single_queries):4d}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_result = None

    print("\nBASELINE")
    print("-" * 112)
    print(
        f"T2I R@1={base_metrics['r1']:.2f} "
        f"R@5={base_metrics['r5']:.2f} "
        f"R@10={base_metrics['r10']:.2f} "
        f"MedR={base_metrics['medr']:.1f} "
        f"MeanR={base_metrics['meanr']:.2f}"
    )

    for lam in args.lambdas:
        new_orders = []
        for text_id, base_order in enumerate(base_orders):
            cats = query_categories[text_id]
            if len(cats) != 1:
                new_orders.append(base_order)
                continue

            category = cats[0]
            new_orders.append(
                rerank_order(
                    base_scores=scores_t2i[text_id],
                    base_order=base_order,
                    anchor_scores=category_anchor_scores[category],
                    topn=args.topn,
                    lam=lam,
                )
            )

        metrics, new_ranks = retrieval_metrics_from_orders(
            new_orders,
            dataset.txt2img,
        )

        # 严格 GT image rescue / corruption
        exact_rescue = 0
        exact_corruption = 0
        for text_id in single_anchor_queries:
            gt_id = int(dataset.txt2img[text_id])
            base_top1 = int(base_orders[text_id][0])
            new_top1 = int(new_orders[text_id][0])

            if base_top1 != gt_id and new_top1 == gt_id:
                exact_rescue += 1
            if base_top1 == gt_id and new_top1 != gt_id:
                exact_corruption += 1

        # 类别级 rescue / corruption：只看类别已知样本
        category_rescue = 0
        category_corruption = 0
        for text_id in single_anchor_queries:
            gt_id = int(dataset.txt2img[text_id])
            gt_label = image_labels[gt_id]
            if gt_label is None:
                continue

            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])
            base_label = image_labels[base_pred]
            new_label = image_labels[new_pred]
            if base_label is None or new_label is None:
                continue

            if base_label != gt_label and new_label == gt_label:
                category_rescue += 1
            if base_label == gt_label and new_label != gt_label:
                category_corruption += 1

        # 我们最关心的 strict target cohort。
        target_exact_rescue = 0
        target_category_rescue = 0
        target_top1_changed = 0
        target_gt_rank_improved = 0

        for text_id in target_single_queries:
            gt_id = int(dataset.txt2img[text_id])
            gt_label = image_labels[gt_id]
            base_top1 = int(base_orders[text_id][0])
            new_top1 = int(new_orders[text_id][0])

            target_top1_changed += int(new_top1 != base_top1)
            target_exact_rescue += int(new_top1 == gt_id)
            target_category_rescue += int(
                image_labels[new_top1] is not None
                and image_labels[new_top1] == gt_label
            )
            target_gt_rank_improved += int(new_ranks[text_id] < base_ranks[text_id])

        result = {
            "lambda": lam,
            "t2i_r1": metrics["r1"],
            "t2i_r5": metrics["r5"],
            "t2i_r10": metrics["r10"],
            "t2i_medr": metrics["medr"],
            "t2i_meanr": metrics["meanr"],
            "exact_rescue_single_anchor": exact_rescue,
            "exact_corruption_single_anchor": exact_corruption,
            "exact_net_single_anchor": exact_rescue - exact_corruption,
            "category_rescue_single_anchor": category_rescue,
            "category_corruption_single_anchor": category_corruption,
            "category_net_single_anchor": category_rescue - category_corruption,
            "target_single_count": len(target_single_queries),
            "target_exact_rescue": target_exact_rescue,
            "target_category_rescue": target_category_rescue,
            "target_top1_changed": target_top1_changed,
            "target_gt_rank_improved": target_gt_rank_improved,
        }
        rows.append(result)

        if best_result is None or result["t2i_r1"] > best_result["t2i_r1"]:
            best_result = result

        print(
            f"\nλ={lam:.2f} | "
            f"R1={metrics['r1']:.2f} R5={metrics['r5']:.2f} R10={metrics['r10']:.2f}"
        )
        print(
            f"  exact    rescue={exact_rescue:3d} corruption={exact_corruption:3d} "
            f"net={exact_rescue-exact_corruption:+4d}"
        )
        print(
            f"  category rescue={category_rescue:3d} corruption={category_corruption:3d} "
            f"net={category_rescue-category_corruption:+4d}"
        )
        print(
            f"  target({len(target_single_queries)}) "
            f"exact_rescue={target_exact_rescue:3d} "
            f"category_rescue={target_category_rescue:3d} "
            f"top1_changed={target_top1_changed:3d} "
            f"gt_rank_improved={target_gt_rank_improved:3d}"
        )

    # ------------------------------------------------------------
    # 保存 target case 的 baseline anchor diagnostic：
    # 不带 lambda，先看独立 anchor 是否本来就更偏 GT。
    # ------------------------------------------------------------
    target_rows = []
    for text_id in target_single_queries:
        category = query_categories[text_id][0]
        gt_id = int(dataset.txt2img[text_id])
        pred_id = int(base_orders[text_id][0])
        anchor_scores = category_anchor_scores[category]

        target_rows.append({
            "text_id": text_id,
            "query": dataset.text[text_id],
            "anchor_category": category,
            "gt_image_id": gt_id,
            "baseline_pred_image_id": pred_id,
            "baseline_gt_rank": int(base_ranks[text_id]),
            "global_gt_score": float(scores_t2i[text_id, gt_id]),
            "global_pred_score": float(scores_t2i[text_id, pred_id]),
            "global_pred_minus_gt": float(
                scores_t2i[text_id, pred_id] - scores_t2i[text_id, gt_id]
            ),
            "anchor_gt_score": float(anchor_scores[gt_id]),
            "anchor_pred_score": float(anchor_scores[pred_id]),
            "anchor_gt_minus_pred": float(
                anchor_scores[gt_id] - anchor_scores[pred_id]
            ),
            "anchor_prefers_gt": int(anchor_scores[gt_id] > anchor_scores[pred_id]),
            "gt_label": image_labels[gt_id],
            "pred_label": image_labels[pred_id],
        })

    if target_rows:
        with (output_dir / "strict_target_anchor_diagnostic.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()))
            writer.writeheader()
            writer.writerows(target_rows)

    with (output_dir / "lambda_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    anchor_prefers_gt = sum(x["anchor_prefers_gt"] for x in target_rows)
    summary = {
        "metadata": {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "topn": args.topn,
            "lambdas": args.lambdas,
            "rerank_rule": "only exactly-one-explicit-category queries",
            "fusion": "candidate-wise zscore(global) + lambda * zscore(anchor)",
            "test_label_used_by_reranker": False,
        },
        "coverage": {
            "num_queries": len(dataset.text),
            "no_anchor": len(no_anchor_queries),
            "single_anchor": len(single_anchor_queries),
            "multi_anchor": len(multi_anchor_queries),
            "gt_yes_pred_no": len(target_queries),
            "strict_single_gt_target": len(target_single_queries),
        },
        "baseline": base_metrics,
        "strict_target_anchor_prefers_gt": {
            "count": anchor_prefers_gt,
            "total": len(target_rows),
            "fraction": anchor_prefers_gt / max(len(target_rows), 1),
        },
        "lambda_results": rows,
        "best_by_overall_r1": best_result,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 112)
    print("ANCHOR-ONLY DIAGNOSTIC")
    print("=" * 112)
    print(
        f"On strict target cases, independent category anchor prefers GT over baseline wrong Top1: "
        f"{anchor_prefers_gt}/{len(target_rows)} = "
        f"{anchor_prefers_gt/max(len(target_rows),1):.2%}"
    )
    print("如果这个比例明显 > 50%，说明“完整 caption 稀释显式类别词”假设有直接证据。")
    print("如果接近/低于 50%，说明仅强调类别词也解决不了，问题更偏视觉类别表示/边界。")
    print("-" * 112)
    print(f"Summary : {output_dir / 'summary.json'}")
    print(f"Sweep   : {output_dir / 'lambda_sweep.csv'}")
    print(f"Targets : {output_dir / 'strict_target_anchor_diagnostic.csv'}")
    print("=" * 112)


if __name__ == "__main__":
    main()
