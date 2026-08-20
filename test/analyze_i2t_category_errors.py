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
from evaluation.retrieval import (
    extract_image_features,
    extract_text_features,
    compute_similarity_matrix,
)


RSICD_CLASSES = [
    "airport", "bareland", "baseballfield", "beach", "bridge", "center",
    "church", "commercial", "denseresidential", "desert", "farmland",
    "forest", "industrial", "meadow", "mediumresidential", "mountain",
    "park", "parking", "playground", "pond", "port", "railwaystation",
    "resort", "river", "school", "sparseresidential", "square", "stadium",
    "storagetanks", "viaduct",
]

# 仅做显式词面支持判断，不做语义推断。
CATEGORY_ALIASES = {
    "airport": ["airport", "airports"],
    "bareland": ["bare land", "bareland", "barren land"],
    "baseballfield": ["baseball field", "baseball fields", "baseballfield", "baseballfields"],
    "beach": ["beach", "beaches"],
    "bridge": ["bridge", "bridges"],
    "center": ["city center", "urban center", "town center", "city centre", "urban centre", "town centre"],
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
    "parking": ["parking lot", "parking lots", "parking area", "parking areas"],
    "playground": ["playground", "playgrounds"],
    "pond": ["pond", "ponds"],
    "port": ["port", "ports", "harbor", "harbour", "harbors", "harbours"],
    "railwaystation": ["railway station", "railway stations", "train station", "train stations"],
    "resort": ["resort", "resorts"],
    "river": ["river", "rivers"],
    "school": ["school", "schools", "campus", "campuses"],
    "sparseresidential": ["sparse residential", "sparse residential area", "sparse residential areas"],
    "square": ["public square", "city square", "urban square", "plaza", "plazas"],
    "stadium": ["stadium", "stadiums", "stadia"],
    "storagetanks": ["storage tank", "storage tanks", "storagetank", "storagetanks"],
    "viaduct": ["viaduct", "viaducts", "overpass", "overpasses"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSICD I2T Top-1 类别错误与显式类别支持诊断。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/i2t_category_error_analysis",
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


def text_supports_category(text, category):
    if category is None or category not in CATEGORY_ALIASES:
        return False
    return any(phrase_present(text, alias) for alias in CATEGORY_ALIASES[category])


def parse_image_category(image_ref):
    stem = Path(str(image_ref)).stem.lower()

    if stem.isdigit():
        return None

    prefix, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit() and prefix in RSICD_CLASSES:
        return prefix

    # 防止某些路径/文件名不完全遵循 “class_index”。
    for category in sorted(RSICD_CLASSES, key=len, reverse=True):
        if stem.startswith(category + "_"):
            return category

    return None


def safe_div(a, b):
    return float(a / b) if b else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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

    print("=" * 120)
    print("I2T CATEGORY ERROR ANALYSIS")
    print("=" * 120)
    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print("Unit       : one image query")
    print("Strict     : cross-category Top1 + any GT caption explicitly supports query category")
    print("             + wrong Top1 caption does NOT explicitly support query category")
    print("=" * 120)

    print("Extracting image features...")
    image_features = extract_image_features(model, loader, device)

    print("Extracting text features...")
    text_features = extract_text_features(
        model,
        dataset.text,
        device,
        batch_size=text_batch_size,
    )

    print("Computing similarity...")
    scores_i2t = compute_similarity_matrix(
        image_features,
        text_features,
        device,
    )

    image_categories = [
        parse_image_category(image_ref)
        for image_ref in dataset.image
    ]

    rows = []
    confusion = Counter()
    exact_correct = 0
    same_category_errors = 0
    cross_category_errors = 0
    unknown_category_errors = 0
    resolvable_errors = 0
    known_pair_queries = 0

    quadrant = Counter()

    for image_id in range(len(dataset.image)):
        scores = scores_i2t[image_id]
        order = np.argsort(scores)[::-1]
        pred_text_id = int(order[0])
        pred_source_image_id = int(dataset.txt2img[pred_text_id])

        gt_text_ids = [int(x) for x in dataset.img2txt[image_id]]
        exact_match = pred_text_id in set(gt_text_ids)

        query_category = image_categories[image_id]
        pred_category = image_categories[pred_source_image_id]

        if query_category is not None and pred_category is not None:
            known_pair_queries += 1

        if exact_match:
            exact_correct += 1
            error_type = "exact_correct"
        elif query_category is None or pred_category is None:
            unknown_category_errors += 1
            error_type = "unknown_category"
        elif query_category == pred_category:
            same_category_errors += 1
            resolvable_errors += 1
            error_type = "same_category_wrong"
        else:
            cross_category_errors += 1
            resolvable_errors += 1
            error_type = "cross_category_wrong"
            confusion[(query_category, pred_category)] += 1

        gt_captions = [dataset.text[text_id] for text_id in gt_text_ids]
        pred_caption = dataset.text[pred_text_id]

        gt_support_count = (
            sum(text_supports_category(text, query_category) for text in gt_captions)
            if query_category is not None
            else 0
        )
        gt_any_support = gt_support_count > 0
        pred_support_query = (
            text_supports_category(pred_caption, query_category)
            if query_category is not None
            else False
        )
        pred_support_own = (
            text_supports_category(pred_caption, pred_category)
            if pred_category is not None
            else False
        )

        strict_error = (
            error_type == "cross_category_wrong"
            and gt_any_support
            and not pred_support_query
        )

        if error_type == "cross_category_wrong":
            if gt_any_support and not pred_support_query:
                q = "GT_yes_Pred_no"
            elif gt_any_support and pred_support_query:
                q = "GT_yes_Pred_yes"
            elif not gt_any_support and pred_support_query:
                q = "GT_no_Pred_yes"
            else:
                q = "GT_no_Pred_no"
            quadrant[q] += 1
        else:
            q = ""

        # exact I2T rank：任意 GT caption 的最佳 rank。
        rank_lookup = np.empty(len(order), dtype=np.int32)
        rank_lookup[order] = np.arange(1, len(order) + 1)
        gt_rank = int(min(rank_lookup[text_id] for text_id in gt_text_ids))

        rows.append({
            "image_id": image_id,
            "image": dataset.image[image_id],
            "query_category": query_category or "unknown",
            "pred_text_id": pred_text_id,
            "pred_caption": pred_caption,
            "pred_source_image_id": pred_source_image_id,
            "pred_source_image": dataset.image[pred_source_image_id],
            "pred_category": pred_category or "unknown",
            "exact_match": int(exact_match),
            "gt_best_rank": gt_rank,
            "error_type": error_type,
            "category_resolvable": int(query_category is not None and pred_category is not None),
            "gt_support_count": gt_support_count,
            "gt_any_support": int(gt_any_support),
            "pred_support_query_category": int(pred_support_query),
            "pred_support_own_category": int(pred_support_own),
            "support_quadrant": q,
            "strict_category_error": int(strict_error),
            "top1_score": float(scores[pred_text_id]),
            "best_gt_score": float(max(scores[text_id] for text_id in gt_text_ids)),
            "top1_minus_best_gt": float(scores[pred_text_id] - max(scores[text_id] for text_id in gt_text_ids)),
            "gt_captions": " || ".join(gt_captions),
        })

    num_images = len(dataset.image)
    total_errors = num_images - exact_correct
    strict_count = sum(row["strict_category_error"] for row in rows)

    cross_rows = [r for r in rows if r["error_type"] == "cross_category_wrong"]
    strict_rows = [r for r in rows if r["strict_category_error"] == 1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "i2t_all_image_queries.csv", rows)
    write_csv(output_dir / "i2t_cross_category_errors.csv", cross_rows)
    write_csv(output_dir / "i2t_strict_category_errors.csv", strict_rows)

    confusion_rows = [
        {
            "gt_category": gt,
            "pred_category": pred,
            "count": count,
            "fraction_of_cross_errors": safe_div(count, cross_category_errors),
        }
        for (gt, pred), count in confusion.most_common()
    ]
    write_csv(output_dir / "i2t_cross_category_confusions.csv", confusion_rows)

    print("\nOVERALL I2T TOP-1")
    print("-" * 120)
    print(
        f"Exact I2T Top-1 accuracy               : {safe_div(exact_correct, num_images):.2%} "
        f"({exact_correct}/{num_images})"
    )
    print(f"Exact I2T Top-1 errors                 : {total_errors}")
    print(
        f"Known-category Top1 pair coverage      : {safe_div(known_pair_queries, num_images):.2%} "
        f"({known_pair_queries}/{num_images})"
    )
    print(
        f"Among all exact errors -> same class   : {safe_div(same_category_errors, total_errors):.2%} "
        f"({same_category_errors}/{total_errors})"
    )
    print(
        f"Among all exact errors -> cross class  : {safe_div(cross_category_errors, total_errors):.2%} "
        f"({cross_category_errors}/{total_errors})"
    )
    print(
        f"Among all exact errors -> unknown      : {safe_div(unknown_category_errors, total_errors):.2%} "
        f"({unknown_category_errors}/{total_errors})"
    )
    print(
        f"Among resolvable errors -> same class  : {safe_div(same_category_errors, resolvable_errors):.2%} "
        f"({same_category_errors}/{resolvable_errors})"
    )
    print(
        f"Among resolvable errors -> cross class : {safe_div(cross_category_errors, resolvable_errors):.2%} "
        f"({cross_category_errors}/{resolvable_errors})"
    )

    print("\nCROSS-CATEGORY EXPLICIT SUPPORT")
    print("-" * 120)
    for name in [
        "GT_yes_Pred_no",
        "GT_yes_Pred_yes",
        "GT_no_Pred_yes",
        "GT_no_Pred_no",
    ]:
        count = quadrant[name]
        print(
            f"{name:<22}: {count:4d} / {cross_category_errors} "
            f"= {safe_div(count, cross_category_errors):.2%}"
        )

    print("\nSTRICT I2T CATEGORY ERRORS")
    print("-" * 120)
    print(
        f"Strict count                            : {strict_count} / {num_images} "
        f"= {safe_div(strict_count, num_images):.2%} of all image queries"
    )
    print(
        f"Strict / all cross-category errors      : {strict_count} / {cross_category_errors} "
        f"= {safe_div(strict_count, cross_category_errors):.2%}"
    )

    print("\nTOP CROSS-CATEGORY CONFUSIONS")
    print("-" * 120)
    for (gt, pred), count in confusion.most_common(20):
        print(f"{gt:20s} -> {pred:20s} {count:4d}")

    summary = {
        "metadata": {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "num_images": num_images,
            "num_captions": len(dataset.text),
            "strict_definition": (
                "cross-category Top1; any GT caption explicitly supports query image category; "
                "wrong Top1 caption does not explicitly support query image category"
            ),
        },
        "overall": {
            "exact_top1_correct": exact_correct,
            "exact_top1_accuracy": safe_div(exact_correct, num_images),
            "exact_top1_errors": total_errors,
            "same_category_errors": same_category_errors,
            "cross_category_errors": cross_category_errors,
            "unknown_category_errors": unknown_category_errors,
            "resolvable_errors": resolvable_errors,
            "same_fraction_resolvable_errors": safe_div(same_category_errors, resolvable_errors),
            "cross_fraction_resolvable_errors": safe_div(cross_category_errors, resolvable_errors),
        },
        "cross_category_support_quadrant": {
            name: {
                "count": quadrant[name],
                "fraction": safe_div(quadrant[name], cross_category_errors),
            }
            for name in [
                "GT_yes_Pred_no",
                "GT_yes_Pred_yes",
                "GT_no_Pred_yes",
                "GT_no_Pred_no",
            ]
        },
        "strict": {
            "count": strict_count,
            "fraction_all_image_queries": safe_div(strict_count, num_images),
            "fraction_cross_category_errors": safe_div(strict_count, cross_category_errors),
        },
        "top_cross_category_confusions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in confusion.most_common(20)
        ],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nKEY OUTPUTS")
    print("-" * 120)
    print(f"Summary       : {output_dir / 'summary.json'}")
    print(f"All queries   : {output_dir / 'i2t_all_image_queries.csv'}")
    print(f"Cross errors  : {output_dir / 'i2t_cross_category_errors.csv'}")
    print(f"Strict errors : {output_dir / 'i2t_strict_category_errors.csv'}")
    print(f"Confusions    : {output_dir / 'i2t_cross_category_confusions.csv'}")
    print("=" * 120)


if __name__ == "__main__":
    main()
