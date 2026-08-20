import argparse
import csv
import json
import re
import sys
from pathlib import Path

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


# 仅用于从 caption 中高精度识别显式类别；推理时不读取测试标签。
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
        description="Anchor-rank-triggered Category Gate diagnostic for RSICD T2I."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument(
        "--gate-ms",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="触发后，在 baseline Top-N 中保留 anchor Top-M 候选。",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[25, 30, 35, 40, 45],
        help="若 baseline Top1 的 anchor rank > T，则触发 gate。",
    )
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/anchor_rank_triggered_category_gate",
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


def rank_of(order, image_id):
    return int(np.where(order == image_id)[0][0]) + 1


def retrieval_metrics_from_orders(orders, txt2img):
    ranks = np.asarray(
        [
            rank_of(order, int(txt2img[text_id]))
            for text_id, order in enumerate(orders)
        ],
        dtype=np.int32,
    )
    return {
        "r1": float(np.mean(ranks <= 1) * 100.0),
        "r5": float(np.mean(ranks <= 5) * 100.0),
        "r10": float(np.mean(ranks <= 10) * 100.0),
        "medr": float(np.median(ranks)),
        "meanr": float(np.mean(ranks)),
    }, ranks


def anchor_rank_in_topn(base_order, anchor_scores, topn):
    """
    baseline Top1 在该类别 anchor 下、baseline Top-N 候选中的相对排名。
    返回 1..N；越大说明 full-caption Top1 与显式类别越不一致。
    """
    n = min(topn, len(base_order))
    top_ids = np.asarray(base_order[:n], dtype=np.int64)
    base_top1 = int(top_ids[0])

    anchor_order = top_ids[np.argsort(anchor_scores[top_ids])[::-1]]
    return int(np.where(anchor_order == base_top1)[0][0]) + 1


def category_gate_order(base_order, anchor_scores, topn, gate_m):
    """
    触发后：
    1) baseline Top-N 内按 anchor score 选 Top-M；
    2) 选中候选整体前移；
    3) gate 内部、gate 外部均保持原 full-caption baseline 顺序。
    """
    n = min(topn, len(base_order))
    m = min(max(int(gate_m), 1), n)

    top_ids = np.asarray(base_order[:n], dtype=np.int64)
    anchor_local = anchor_scores[top_ids]

    selected_pos = np.argpartition(-anchor_local, m - 1)[:m]
    selected_ids = set(top_ids[selected_pos].tolist())

    selected = [x for x in top_ids if int(x) in selected_ids]
    remaining = [x for x in top_ids if int(x) not in selected_ids]
    reranked_top = np.asarray(selected + remaining, dtype=np.int64)

    if n == len(base_order):
        return reranked_top

    return np.concatenate([
        reranked_top,
        np.asarray(base_order[n:], dtype=np.int64),
    ])


def build_trigger_cache(base_orders, query_categories, category_to_anchor_scores, topn):
    """
    每个 query 只计算一次 baseline Top1 的 anchor rank。
    非单一显式类别 query 记为 None，不参与触发。
    """
    trigger_ranks = [None] * len(base_orders)

    for text_id, base_order in enumerate(base_orders):
        cats = query_categories[text_id]
        if len(cats) != 1:
            continue

        category = cats[0]
        trigger_ranks[text_id] = anchor_rank_in_topn(
            base_order=base_order,
            anchor_scores=category_to_anchor_scores[category],
            topn=topn,
        )

    return trigger_ranks


def main():
    args = parse_args()

    for m in args.gate_ms:
        if m < 1 or m > args.topn:
            raise ValueError(f"gate M={m} 必须满足 1 <= M <= topn={args.topn}")
    for t in args.thresholds:
        if t < 1 or t > args.topn:
            raise ValueError(f"threshold T={t} 必须满足 1 <= T <= topn={args.topn}")

    valid_pairs = [(m, t) for m in args.gate_ms for t in args.thresholds if t > m]
    if not valid_pairs:
        raise ValueError("没有合法参数组合；要求 threshold T > gate M。")

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

    print("=" * 122)
    print("ANCHOR-RANK-TRIGGERED CATEGORY GATE")
    print("=" * 122)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Top-N      : {args.topn}")
    print(f"Gate M     : {args.gate_ms}")
    print(f"Thresholds : {args.thresholds}")
    print("Trigger    : baseline Top1 anchor-rank > T")
    print("Gate       : anchor selects Top-M; original full-caption order is preserved within groups")
    print("No test label is used by trigger or reranker.")
    print("=" * 122)

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
    base_orders = baseline_orders(scores_i2t)
    base_metrics, base_ranks = retrieval_metrics_from_orders(
        base_orders,
        dataset.txt2img,
    )

    query_categories = [
        extract_explicit_categories(text)
        for text in dataset.text
    ]
    single_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) == 1
    ]
    no_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) == 0
    ]
    multi_anchor_queries = [
        i for i, cats in enumerate(query_categories) if len(cats) > 1
    ]

    used_categories = sorted({
        query_categories[i][0]
        for i in single_anchor_queries
    })
    prompts = [CATEGORY_PROMPTS[c] for c in used_categories]

    print(f"Encoding {len(prompts)} category anchor prompts...")
    anchor_text_features = extract_text_features(
        model,
        prompts,
        device,
        batch_size=text_batch_size,
    )

    # [num_categories, num_images]
    anchor_scores_matrix = compute_similarity_matrix(
        image_features,
        anchor_text_features,
        device,
    ).T
    category_to_anchor_scores = {
        category: anchor_scores_matrix[i]
        for i, category in enumerate(used_categories)
    }

    print("Computing anchor-rank trigger signal...")
    trigger_ranks = build_trigger_cache(
        base_orders=base_orders,
        query_categories=query_categories,
        category_to_anchor_scores=category_to_anchor_scores,
        topn=args.topn,
    )

    # 以下标签仅用于离线诊断，不参与 trigger / rerank。
    image_labels = [parse_image_category(ann) for ann in dataset.ann]

    target_queries = []
    target_single_queries = []

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

            if len(cats) == 1 and cats[0] == gt_label:
                target_single_queries.append(text_id)

    print("\nQUERY COVERAGE")
    print("-" * 122)
    print(
        f"No explicit category      : {len(no_anchor_queries):4d} / {len(dataset.text)} "
        f"= {len(no_anchor_queries)/len(dataset.text):.2%}"
    )
    print(
        f"Exactly one category      : {len(single_anchor_queries):4d} / {len(dataset.text)} "
        f"= {len(single_anchor_queries)/len(dataset.text):.2%}"
    )
    print(
        f"Multiple categories       : {len(multi_anchor_queries):4d} / {len(dataset.text)} "
        f"= {len(multi_anchor_queries)/len(dataset.text):.2%}"
    )
    print(f"GT_yes / Pred_no baseline : {len(target_queries):4d}")
    print(f"Strict single-GT target   : {len(target_single_queries):4d}")

    print("\nBASELINE")
    print("-" * 122)
    print(
        f"T2I R@1={base_metrics['r1']:.2f} "
        f"R@5={base_metrics['r5']:.2f} "
        f"R@10={base_metrics['r10']:.2f} "
        f"MedR={base_metrics['medr']:.1f} "
        f"MeanR={base_metrics['meanr']:.2f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    trigger_rows = []

    for gate_m, threshold in valid_pairs:
        new_orders = []
        triggered_queries = []

        for text_id, base_order in enumerate(base_orders):
            cats = query_categories[text_id]
            trigger_rank = trigger_ranks[text_id]

            if len(cats) != 1 or trigger_rank is None or trigger_rank <= threshold:
                new_orders.append(base_order)
                continue

            category = cats[0]
            triggered_queries.append(text_id)

            new_orders.append(
                category_gate_order(
                    base_order=base_order,
                    anchor_scores=category_to_anchor_scores[category],
                    topn=args.topn,
                    gate_m=gate_m,
                )
            )

        metrics, new_ranks = retrieval_metrics_from_orders(
            new_orders,
            dataset.txt2img,
        )

        exact_rescue = 0
        exact_corruption = 0
        category_rescue = 0
        category_corruption = 0

        triggered_exact_correct = 0
        triggered_cross_category_wrong = 0
        triggered_same_category_wrong = 0
        triggered_unknown_category = 0

        for text_id in triggered_queries:
            gt_id = int(dataset.txt2img[text_id])
            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])

            gt_label = image_labels[gt_id]
            base_label = image_labels[base_pred]
            new_label = image_labels[new_pred]

            if base_pred == gt_id:
                triggered_exact_correct += 1
            elif (
                gt_label is not None
                and base_label is not None
                and gt_label != base_label
            ):
                triggered_cross_category_wrong += 1
            elif (
                gt_label is not None
                and base_label is not None
                and gt_label == base_label
            ):
                triggered_same_category_wrong += 1
            else:
                triggered_unknown_category += 1

            if base_pred != gt_id and new_pred == gt_id:
                exact_rescue += 1
            if base_pred == gt_id and new_pred != gt_id:
                exact_corruption += 1

            if (
                gt_label is not None
                and base_label is not None
                and new_label is not None
            ):
                if base_label != gt_label and new_label == gt_label:
                    category_rescue += 1
                if base_label == gt_label and new_label != gt_label:
                    category_corruption += 1

        target_triggered = 0
        target_exact_rescue = 0
        target_category_rescue = 0
        target_gt_rank_improved = 0
        target_gt_in_gate = 0
        target_wrong_top1_filtered = 0
        target_both = 0

        for text_id in target_single_queries:
            if trigger_ranks[text_id] is None or trigger_ranks[text_id] <= threshold:
                continue

            target_triggered += 1

            gt_id = int(dataset.txt2img[text_id])
            gt_label = image_labels[gt_id]
            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])

            target_exact_rescue += int(new_pred == gt_id)
            target_category_rescue += int(
                image_labels[new_pred] is not None
                and image_labels[new_pred] == gt_label
            )
            target_gt_rank_improved += int(new_ranks[text_id] < base_ranks[text_id])

            category = query_categories[text_id][0]
            anchor_scores = category_to_anchor_scores[category]
            top_ids = np.asarray(base_orders[text_id][:args.topn], dtype=np.int64)
            selected_pos = np.argpartition(-anchor_scores[top_ids], gate_m - 1)[:gate_m]
            selected_ids = set(top_ids[selected_pos].tolist())

            gt_in_gate = gt_id in selected_ids
            wrong_filtered = base_pred not in selected_ids

            target_gt_in_gate += int(gt_in_gate)
            target_wrong_top1_filtered += int(wrong_filtered)
            target_both += int(gt_in_gate and wrong_filtered)

        row = {
            "gate_m": gate_m,
            "threshold": threshold,
            "triggered_queries": len(triggered_queries),
            "trigger_rate_all": len(triggered_queries) / len(dataset.text),
            "trigger_rate_single_anchor": len(triggered_queries) / max(len(single_anchor_queries), 1),
            "triggered_exact_correct": triggered_exact_correct,
            "triggered_cross_category_wrong": triggered_cross_category_wrong,
            "triggered_same_category_wrong": triggered_same_category_wrong,
            "triggered_unknown_category": triggered_unknown_category,
            "t2i_r1": metrics["r1"],
            "t2i_r5": metrics["r5"],
            "t2i_r10": metrics["r10"],
            "t2i_medr": metrics["medr"],
            "t2i_meanr": metrics["meanr"],
            "exact_rescue": exact_rescue,
            "exact_corruption": exact_corruption,
            "exact_net": exact_rescue - exact_corruption,
            "category_rescue": category_rescue,
            "category_corruption": category_corruption,
            "category_net": category_rescue - category_corruption,
            "target_single_count": len(target_single_queries),
            "target_triggered": target_triggered,
            "target_trigger_rate": target_triggered / max(len(target_single_queries), 1),
            "target_exact_rescue": target_exact_rescue,
            "target_category_rescue": target_category_rescue,
            "target_gt_rank_improved": target_gt_rank_improved,
            "target_gt_in_gate": target_gt_in_gate,
            "target_wrong_top1_filtered": target_wrong_top1_filtered,
            "target_both": target_both,
        }
        sweep_rows.append(row)

        print(
            f"\nM={gate_m:2d}, T={threshold:2d} | "
            f"trigger={len(triggered_queries):4d} "
            f"({len(triggered_queries)/len(dataset.text):.2%}) | "
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
            f"  triggered baseline: correct={triggered_exact_correct:3d} "
            f"cross={triggered_cross_category_wrong:3d} "
            f"same={triggered_same_category_wrong:3d} "
            f"unknown={triggered_unknown_category:3d}"
        )
        print(
            f"  target({len(target_single_queries)}) triggered={target_triggered:3d} "
            f"exact_rescue={target_exact_rescue:3d} "
            f"category_rescue={target_category_rescue:3d} "
            f"gt_rank_improved={target_gt_rank_improved:3d}"
        )
        print(
            f"  target gate: gt_in={target_gt_in_gate:3d} "
            f"wrong_filtered={target_wrong_top1_filtered:3d} "
            f"both={target_both:3d}"
        )

        # 保存触发样本明细，便于后续看 false trigger。
        for text_id in triggered_queries:
            gt_id = int(dataset.txt2img[text_id])
            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])

            trigger_rows.append({
                "gate_m": gate_m,
                "threshold": threshold,
                "text_id": text_id,
                "query": dataset.text[text_id],
                "anchor_category": query_categories[text_id][0],
                "anchor_rank_of_baseline_top1": trigger_ranks[text_id],
                "gt_image_id": gt_id,
                "baseline_pred_image_id": base_pred,
                "new_pred_image_id": new_pred,
                "gt_label": image_labels[gt_id],
                "baseline_pred_label": image_labels[base_pred],
                "new_pred_label": image_labels[new_pred],
                "baseline_exact_correct": int(base_pred == gt_id),
                "new_exact_correct": int(new_pred == gt_id),
                "baseline_gt_rank": int(base_ranks[text_id]),
                "new_gt_rank": int(new_ranks[text_id]),
            })

    sweep_rows.sort(
        key=lambda x: (
            -x["t2i_r1"],
            -x["exact_net"],
            -x["category_net"],
        )
    )

    with (output_dir / "trigger_gate_sweep.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    if trigger_rows:
        with (output_dir / "triggered_query_diagnostic.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(trigger_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trigger_rows)

    best_by_r1 = max(sweep_rows, key=lambda x: x["t2i_r1"])
    best_by_exact_net = max(sweep_rows, key=lambda x: x["exact_net"])
    best_by_category_net = max(sweep_rows, key=lambda x: x["category_net"])

    summary = {
        "metadata": {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "topn": args.topn,
            "gate_ms": args.gate_ms,
            "thresholds": args.thresholds,
            "trigger": "baseline Top1 anchor-rank > threshold",
            "gate": "anchor Top-M inside baseline Top-N; preserve original caption order within groups",
            "test_label_used_by_trigger_or_reranker": False,
        },
        "coverage": {
            "num_queries": len(dataset.text),
            "single_anchor": len(single_anchor_queries),
            "no_anchor": len(no_anchor_queries),
            "multi_anchor": len(multi_anchor_queries),
            "gt_yes_pred_no": len(target_queries),
            "strict_single_gt_target": len(target_single_queries),
        },
        "baseline": base_metrics,
        "best_by_overall_r1": best_by_r1,
        "best_by_exact_net": best_by_exact_net,
        "best_by_category_net": best_by_category_net,
        "results": sweep_rows,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 122)
    print("SUMMARY")
    print("=" * 122)
    print(
        f"Best overall R@1 : M={best_by_r1['gate_m']}, T={best_by_r1['threshold']} "
        f"R1={best_by_r1['t2i_r1']:.2f} "
        f"exact_net={best_by_r1['exact_net']:+d} "
        f"category_net={best_by_r1['category_net']:+d}"
    )
    print(
        f"Best exact net   : M={best_by_exact_net['gate_m']}, T={best_by_exact_net['threshold']} "
        f"net={best_by_exact_net['exact_net']:+d} "
        f"R1={best_by_exact_net['t2i_r1']:.2f}"
    )
    print(
        f"Best category net: M={best_by_category_net['gate_m']}, T={best_by_category_net['threshold']} "
        f"net={best_by_category_net['category_net']:+d} "
        f"R1={best_by_category_net['t2i_r1']:.2f}"
    )
    print("-" * 122)
    print(f"Summary  : {output_dir / 'summary.json'}")
    print(f"Sweep    : {output_dir / 'trigger_gate_sweep.csv'}")
    print(f"Triggered: {output_dir / 'triggered_query_diagnostic.csv'}")
    print("=" * 122)


if __name__ == "__main__":
    main()
