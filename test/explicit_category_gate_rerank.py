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


# 只做高精度显式类别识别；rerank 推理过程中不读取 GT 类别。
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
        description="Category Gate + Original Caption Ranking diagnostic for RSICD T2I."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument(
        "--gate-ms",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30, 40, 50],
        help="在 baseline Top-N 中按类别 anchor 选出的候选数 M；M=TopN 应退化为 baseline。",
    )
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/explicit_category_gate_rerank",
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


def category_gate_order(base_order, anchor_scores, topn, gate_m):
    """
    只在 baseline Top-N 内做 category gate：
    1) anchor score 选 Top-M；
    2) 被 gate 选中的候选整体前移；
    3) gate 内部、gate 外部均严格保持原 Full Caption baseline 顺序；
    4) Top-N 外保持 baseline 顺序。

    M=TopN 时应严格退化为 baseline。
    """
    n = min(topn, len(base_order))
    m = min(max(int(gate_m), 1), n)

    top_ids = np.asarray(base_order[:n], dtype=np.int64)
    anchor_local = anchor_scores[top_ids]

    # 只利用 anchor 决定“进不进 gate”，不利用它决定 gate 内排序。
    selected_pos = np.argpartition(-anchor_local, m - 1)[:m]
    selected_ids = set(top_ids[selected_pos].tolist())

    selected_in_base_order = [x for x in top_ids if int(x) in selected_ids]
    remaining_in_base_order = [x for x in top_ids if int(x) not in selected_ids]

    reranked_top = np.asarray(
        selected_in_base_order + remaining_in_base_order,
        dtype=np.int64,
    )

    if n == len(base_order):
        return reranked_top

    return np.concatenate([
        reranked_top,
        np.asarray(base_order[n:], dtype=np.int64),
    ])


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


def main():
    args = parse_args()
    if any(m > args.topn for m in args.gate_ms):
        raise ValueError(f"所有 gate M 必须 <= topn={args.topn}。")

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

    print("=" * 116)
    print("CATEGORY GATE + ORIGINAL CAPTION RANKING")
    print("=" * 116)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Top-N      : {args.topn}")
    print(f"Gate M     : {args.gate_ms}")
    print("Rule       : only exactly-one-explicit-category queries are gated")
    print("Ranking    : anchor selects gate; full-caption baseline preserves order inside/outside gate")
    print("No test label is used by the reranker.")
    print("=" * 116)

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

    image_labels = [
        parse_image_category(ann)
        for ann in dataset.ann
    ]

    # 评估 cohort，只用于分析，不参与 rerank。
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
    print("-" * 116)
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
    print("-" * 116)
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
    per_target_rows = []

    for gate_m in args.gate_ms:
        new_orders = []

        for text_id, base_order in enumerate(base_orders):
            cats = query_categories[text_id]

            if len(cats) != 1:
                new_orders.append(base_order)
                continue

            category = cats[0]
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

        for text_id in single_anchor_queries:
            gt_id = int(dataset.txt2img[text_id])
            gt_label = image_labels[gt_id]
            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])

            if base_pred != gt_id and new_pred == gt_id:
                exact_rescue += 1
            if base_pred == gt_id and new_pred != gt_id:
                exact_corruption += 1

            base_label = image_labels[base_pred]
            new_label = image_labels[new_pred]

            if (
                gt_label is not None
                and base_label is not None
                and new_label is not None
            ):
                if base_label != gt_label and new_label == gt_label:
                    category_rescue += 1
                if base_label == gt_label and new_label != gt_label:
                    category_corruption += 1

        target_exact_rescue = 0
        target_category_rescue = 0
        target_top1_changed = 0
        target_gt_rank_improved = 0
        target_gt_in_gate = 0
        target_wrong_top1_filtered = 0
        target_gt_in_gate_wrong_filtered = 0

        for text_id in target_single_queries:
            gt_id = int(dataset.txt2img[text_id])
            gt_label = image_labels[gt_id]
            base_pred = int(base_orders[text_id][0])
            new_pred = int(new_orders[text_id][0])

            target_top1_changed += int(new_pred != base_pred)
            target_exact_rescue += int(new_pred == gt_id)
            target_category_rescue += int(
                image_labels[new_pred] is not None
                and image_labels[new_pred] == gt_label
            )
            target_gt_rank_improved += int(
                new_ranks[text_id] < base_ranks[text_id]
            )

            category = query_categories[text_id][0]
            anchor_scores = category_to_anchor_scores[category]
            top_ids = np.asarray(
                base_orders[text_id][:args.topn],
                dtype=np.int64,
            )
            m = min(gate_m, len(top_ids))
            selected_pos = np.argpartition(
                -anchor_scores[top_ids],
                m - 1,
            )[:m]
            selected_ids = set(top_ids[selected_pos].tolist())

            gt_in_gate = gt_id in selected_ids
            wrong_filtered = base_pred not in selected_ids

            target_gt_in_gate += int(gt_in_gate)
            target_wrong_top1_filtered += int(wrong_filtered)
            target_gt_in_gate_wrong_filtered += int(
                gt_in_gate and wrong_filtered
            )

            per_target_rows.append({
                "gate_m": gate_m,
                "text_id": text_id,
                "query": dataset.text[text_id],
                "anchor_category": category,
                "gt_image_id": gt_id,
                "baseline_pred_image_id": base_pred,
                "new_pred_image_id": new_pred,
                "gt_label": gt_label,
                "baseline_pred_label": image_labels[base_pred],
                "new_pred_label": image_labels[new_pred],
                "baseline_gt_rank": int(base_ranks[text_id]),
                "new_gt_rank": int(new_ranks[text_id]),
                "gt_in_gate": int(gt_in_gate),
                "wrong_top1_filtered": int(wrong_filtered),
                "exact_rescued": int(new_pred == gt_id),
                "category_rescued": int(
                    image_labels[new_pred] is not None
                    and image_labels[new_pred] == gt_label
                ),
            })

        row = {
            "gate_m": gate_m,
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
            "target_gt_in_gate": target_gt_in_gate,
            "target_wrong_top1_filtered": target_wrong_top1_filtered,
            "target_gt_in_gate_wrong_filtered": target_gt_in_gate_wrong_filtered,
        }
        sweep_rows.append(row)

        print(
            f"\nM={gate_m:2d} | "
            f"R1={metrics['r1']:.2f} "
            f"R5={metrics['r5']:.2f} "
            f"R10={metrics['r10']:.2f}"
        )
        print(
            f"  exact    rescue={exact_rescue:3d} "
            f"corruption={exact_corruption:3d} "
            f"net={exact_rescue-exact_corruption:+4d}"
        )
        print(
            f"  category rescue={category_rescue:3d} "
            f"corruption={category_corruption:3d} "
            f"net={category_rescue-category_corruption:+4d}"
        )
        print(
            f"  target({len(target_single_queries)}) "
            f"exact_rescue={target_exact_rescue:3d} "
            f"category_rescue={target_category_rescue:3d} "
            f"gt_rank_improved={target_gt_rank_improved:3d}"
        )
        print(
            f"  gate diag: gt_in_gate={target_gt_in_gate:3d} "
            f"wrong_top1_filtered={target_wrong_top1_filtered:3d} "
            f"both={target_gt_in_gate_wrong_filtered:3d}"
        )

    with (output_dir / "gate_sweep.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(sweep_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    if per_target_rows:
        with (output_dir / "strict_target_gate_diagnostic.csv").open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(per_target_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(per_target_rows)

    # 控制检查：若扫描包含 M=TopN，应严格与 baseline 一致。
    control = next(
        (x for x in sweep_rows if x["gate_m"] == args.topn),
        None,
    )
    if control is not None:
        control_ok = (
            abs(control["t2i_r1"] - base_metrics["r1"]) < 1e-9
            and abs(control["t2i_r5"] - base_metrics["r5"]) < 1e-9
            and abs(control["t2i_r10"] - base_metrics["r10"]) < 1e-9
        )
    else:
        control_ok = None

    best_by_r1 = max(
        sweep_rows,
        key=lambda x: x["t2i_r1"],
    )
    best_by_exact_net = max(
        sweep_rows,
        key=lambda x: x["exact_net_single_anchor"],
    )
    best_by_category_net = max(
        sweep_rows,
        key=lambda x: x["category_net_single_anchor"],
    )

    summary = {
        "metadata": {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "topn": args.topn,
            "gate_ms": args.gate_ms,
            "rerank_rule": (
                "anchor selects Top-M inside baseline Top-N; "
                "full-caption baseline order is preserved within selected/unselected groups"
            ),
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
        "control_m_equals_topn_matches_baseline": control_ok,
        "best_by_overall_r1": best_by_r1,
        "best_by_exact_net": best_by_exact_net,
        "best_by_category_net": best_by_category_net,
        "gate_results": sweep_rows,
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 116)
    print("SUMMARY")
    print("=" * 116)
    print(
        f"Best overall R@1 : M={best_by_r1['gate_m']} "
        f"R1={best_by_r1['t2i_r1']:.2f}"
    )
    print(
        f"Best exact net   : M={best_by_exact_net['gate_m']} "
        f"net={best_by_exact_net['exact_net_single_anchor']:+d}"
    )
    print(
        f"Best category net: M={best_by_category_net['gate_m']} "
        f"net={best_by_category_net['category_net_single_anchor']:+d}"
    )
    if control_ok is not None:
        print(
            f"Control M=TopN   : "
            f"{'PASS' if control_ok else 'FAIL'} "
            f"(should exactly match baseline)"
        )
    print("-" * 116)
    print(f"Summary : {output_dir / 'summary.json'}")
    print(f"Sweep   : {output_dir / 'gate_sweep.csv'}")
    print(f"Targets : {output_dir / 'strict_target_gate_diagnostic.csv'}")
    print("=" * 116)


if __name__ == "__main__":
    main()
