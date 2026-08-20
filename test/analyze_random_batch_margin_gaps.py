import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from datasets.re_dataset import re_train_collate_fn
from losses.category_margin_loss import CrossCategoryMarginLoss
from models import CLIPRetrieval
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计真实随机 batch 下跨类别 hard-negative 的正负相似度 gap 分布。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="统计这些 margin 下的 active triplet 比例。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/category_margin_gap_analysis",
    )
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "negative_gap_ratio": None,
        }

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "negative_gap_ratio": float((values < 0).mean()),
    }


def margin_stats(values, margins):
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for margin in margins:
        key = f"{margin:.4f}"
        if values.size == 0:
            result[key] = {
                "active_count": 0,
                "active_ratio": None,
            }
        else:
            active = values < margin
            result[key] = {
                "active_count": int(active.sum()),
                "active_ratio": float(active.mean()),
            }
    return result


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        # 这里只研究 batch 组成与相似度几何，使用确定性的 eval preprocess。
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    max_batches = len(train_dataset) // args.batch_size
    num_batches = min(args.num_batches, max_batches)
    if num_batches <= 0:
        raise ValueError("Dataset is smaller than one full batch.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    permutation = torch.randperm(len(train_dataset), generator=generator).tolist()
    permutation = permutation[: num_batches * args.batch_size]

    criterion = CrossCategoryMarginLoss(margin=0.0)

    t2i_gaps = []
    i2t_gaps = []
    batch_rows = []

    print("=" * 112)
    print("RANDOM-BATCH CATEGORY MARGIN GAP ANALYSIS")
    print("=" * 112)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Batch size : {args.batch_size}")
    print(f"Batches    : {num_batches}")
    print(f"Seed       : {args.seed}")
    print(f"Margins    : {args.margins}")
    print("=" * 112)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            begin = batch_idx * args.batch_size
            end = begin + args.batch_size
            indices = permutation[begin:end]

            batch = re_train_collate_fn(
                [train_dataset[i] for i in indices]
            )
            (
                images,
                captions,
                image_ids,
                category_ids,
                entity_spans,
                entity_sample_ids,
                entity_counts,
            ) = batch

            images = images.to(device, non_blocking=True)
            category_ids = category_ids.to(device, non_blocking=True)

            outputs = model(images, captions)

            out = criterion(
                outputs["image_feat"],
                outputs["text_feat"],
                category_ids,
            )

            sim_i2t = out["similarity_i2t"]
            pos_sim = sim_i2t.diagonal()

            t2i_valid = out["t2i_valid_mask"]
            i2t_valid = out["i2t_valid_mask"]

            t2i_gap = (
                pos_sim[t2i_valid]
                - out["t2i_hard_neg_sim"][t2i_valid]
            )
            i2t_gap = (
                pos_sim[i2t_valid]
                - out["i2t_hard_neg_sim"][i2t_valid]
            )

            t2i_gap_np = t2i_gap.cpu().numpy()
            i2t_gap_np = i2t_gap.cpu().numpy()

            t2i_gaps.extend(t2i_gap_np.tolist())
            i2t_gaps.extend(i2t_gap_np.tolist())

            known_count = int((category_ids >= 0).sum().item())
            known_categories = category_ids[category_ids >= 0]
            unique_known_categories = int(
                torch.unique(known_categories).numel()
            ) if known_categories.numel() else 0

            row = {
                "batch": batch_idx,
                "known_anchors": known_count,
                "unique_known_categories": unique_known_categories,
                "t2i_valid": int(t2i_valid.sum().item()),
                "i2t_valid": int(i2t_valid.sum().item()),
                "t2i_gap_mean": float(t2i_gap.mean().item()) if t2i_gap.numel() else "",
                "i2t_gap_mean": float(i2t_gap.mean().item()) if i2t_gap.numel() else "",
                "t2i_gap_min": float(t2i_gap.min().item()) if t2i_gap.numel() else "",
                "i2t_gap_min": float(i2t_gap.min().item()) if i2t_gap.numel() else "",
            }

            for margin in args.margins:
                key = str(margin).replace(".", "p")
                row[f"t2i_active_m{key}"] = int(
                    (t2i_gap < margin).sum().item()
                )
                row[f"i2t_active_m{key}"] = int(
                    (i2t_gap < margin).sum().item()
                )

            batch_rows.append(row)

            if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == num_batches:
                print(f"Processed {batch_idx + 1:4d}/{num_batches} batches")

    t2i_desc = describe(t2i_gaps)
    i2t_desc = describe(i2t_gaps)
    t2i_margin = margin_stats(t2i_gaps, args.margins)
    i2t_margin = margin_stats(i2t_gaps, args.margins)

    known_per_batch = np.array(
        [r["known_anchors"] for r in batch_rows],
        dtype=np.float64,
    )
    categories_per_batch = np.array(
        [r["unique_known_categories"] for r in batch_rows],
        dtype=np.float64,
    )

    summary = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "batch_size": args.batch_size,
            "num_batches": num_batches,
            "seed": args.seed,
            "margins": args.margins,
            "transform": "model.backbone.preprocess_val",
        },
        "batch_composition": {
            "avg_known_anchors": float(known_per_batch.mean()),
            "min_known_anchors": int(known_per_batch.min()),
            "max_known_anchors": int(known_per_batch.max()),
            "avg_unique_known_categories": float(categories_per_batch.mean()),
            "min_unique_known_categories": int(categories_per_batch.min()),
            "max_unique_known_categories": int(categories_per_batch.max()),
        },
        "t2i_gap": t2i_desc,
        "i2t_gap": i2t_desc,
        "t2i_margin_activity": t2i_margin,
        "i2t_margin_activity": i2t_margin,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "batch_stats.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(batch_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(batch_rows)

    np.save(output_dir / "t2i_gaps.npy", np.asarray(t2i_gaps, dtype=np.float32))
    np.save(output_dir / "i2t_gaps.npy", np.asarray(i2t_gaps, dtype=np.float32))

    print("\nBATCH COMPOSITION")
    print("-" * 112)
    print(
        f"Known anchors / batch       : "
        f"{summary['batch_composition']['avg_known_anchors']:.2f} avg "
        f"[{summary['batch_composition']['min_known_anchors']}, "
        f"{summary['batch_composition']['max_known_anchors']}]"
    )
    print(
        f"Unique known classes / batch: "
        f"{summary['batch_composition']['avg_unique_known_categories']:.2f} avg "
        f"[{summary['batch_composition']['min_unique_known_categories']}, "
        f"{summary['batch_composition']['max_unique_known_categories']}]"
    )

    def print_direction(name, desc, activities):
        print(f"\n{name} GAP = positive_sim - hardest_cross_category_negative_sim")
        print("-" * 112)
        print(
            f"count={desc['count']} "
            f"mean={desc['mean']:.4f} std={desc['std']:.4f} "
            f"min={desc['min']:.4f}"
        )
        print(
            f"p10={desc['p10']:.4f} "
            f"p25={desc['p25']:.4f} "
            f"median={desc['median']:.4f} "
            f"p75={desc['p75']:.4f} "
            f"p90={desc['p90']:.4f} "
            f"p95={desc['p95']:.4f}"
        )
        print(
            f"hard negative outranks positive (gap<0): "
            f"{desc['negative_gap_ratio']:.2%}"
        )
        for margin in args.margins:
            key = f"{margin:.4f}"
            stat = activities[key]
            print(
                f"margin={margin:.4f} -> active "
                f"{stat['active_count']}/{desc['count']} "
                f"= {stat['active_ratio']:.2%}"
            )

    print_direction("T2I", t2i_desc, t2i_margin)
    print_direction("I2T", i2t_desc, i2t_margin)

    print("\nOUTPUTS")
    print("-" * 112)
    print(f"Summary    : {output_dir / 'summary.json'}")
    print(f"Batch stats: {output_dir / 'batch_stats.csv'}")
    print(f"T2I gaps   : {output_dir / 't2i_gaps.npy'}")
    print(f"I2T gaps   : {output_dir / 'i2t_gaps.npy'}")
    print("=" * 112)


if __name__ == "__main__":
    main()
