import argparse
import csv
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import CLIPRetrieval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raw GDINO vs object-filtered Top1/Mean prototype diagnostic"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--samples-csv", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gdino_object_purification",
    )
    parser.add_argument("--classes", type=str, nargs="+", default=["aircraft"])
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-area", type=float, default=0.35)
    parser.add_argument("--cluster-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=80)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_samples(path, classes):
    groups = {name: [] for name in classes}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["class"] not in groups:
                continue
            groups[row["class"]].append({
                "class": row["class"],
                "image_id": int(row["image_id"]),
                "dataset_index": int(row["dataset_index"]),
                "phrase": row["phrase"],
                "entity": row["entity"],
                "caption": row["caption"],
            })
    return groups


def cache_key(class_name, image_id):
    return f"{class_name}:{int(image_id)}"


@torch.no_grad()
def encode_phrases(model, groups, device, batch_size):
    phrases, seen = [], set()
    for samples in groups.values():
        for sample in samples:
            key = sample["phrase"].lower()
            if key not in seen:
                seen.add(key)
                phrases.append(sample["phrase"])

    features = []
    for start in range(0, len(phrases), batch_size):
        features.append(
            model.backbone.encode_text(
                phrases[start:start + batch_size],
                normalize=True,
            ).cpu()
        )
    features = torch.cat(features)
    return {phrase.lower(): features[i].float() for i, phrase in enumerate(phrases)}


def get_visual_variants(record, min_area, max_area):
    instance_features = record.get("instance_features")
    areas = record.get("area_ratios", [])

    if instance_features is None or len(areas) == 0:
        return None

    instance_features = F.normalize(instance_features.float(), dim=-1)

    raw_mean = F.normalize(instance_features.mean(dim=0), dim=0)
    valid = [
        i for i, area in enumerate(areas)
        if min_area <= float(area) <= max_area
    ]

    if not valid:
        return {
            "raw_mean": raw_mean,
            "filtered_top1": None,
            "filtered_mean": None,
            "valid_indices": [],
        }

    filtered = instance_features[valid]
    return {
        "raw_mean": raw_mean,
        "filtered_top1": F.normalize(filtered[0], dim=0),
        "filtered_mean": F.normalize(filtered.mean(dim=0), dim=0),
        "valid_indices": valid,
    }


def spherical_kmeans(x, k, seed, max_iters, restarts):
    x = F.normalize(x.float(), dim=-1)
    n = len(x)

    if k == 1:
        center = F.normalize(x.mean(dim=0, keepdim=True), dim=-1)
        labels = torch.zeros(n, dtype=torch.long)
        return labels, center

    best = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + 1009 * restart + k)
        centers = x[torch.randperm(n, generator=generator)[:k]].clone()
        labels = None

        for _ in range(max_iters):
            new_labels = torch.argmax(x @ centers.t(), dim=1)
            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels

            updated = []
            for cluster in range(k):
                mask = labels == cluster
                if mask.any():
                    center = x[mask].mean(dim=0)
                else:
                    center = x[
                        torch.randint(0, n, (1,), generator=generator).item()
                    ]
                updated.append(F.normalize(center, dim=0))
            centers = torch.stack(updated)

        objective = float((x * centers[labels]).sum().item())
        if best is None or objective > best[0]:
            best = (objective, labels.clone(), centers.clone())

    return best[1], best[2]


def cosine_silhouette(x, labels, k):
    if k <= 1 or len(x) <= k:
        return None

    x = F.normalize(x.float(), dim=-1)
    distance = 1.0 - x @ x.t()
    values = []

    for i in range(len(x)):
        own = int(labels[i])
        own_mask = labels == own
        own_mask = own_mask.clone()
        own_mask[i] = False

        if not own_mask.any():
            values.append(0.0)
            continue

        a = distance[i][own_mask].mean()
        b_values = [
            distance[i][labels == cluster].mean()
            for cluster in range(k)
            if cluster != own and (labels == cluster).any()
        ]
        if not b_values:
            values.append(0.0)
            continue

        b = torch.stack(b_values).min()
        values.append(float(((b - a) / torch.maximum(a, b).clamp_min(1e-8)).item()))

    return float(np.mean(values))


def compactness(x, labels, centers):
    x = F.normalize(x.float(), dim=-1)
    return float((x * centers[labels]).sum(dim=-1).mean().item())


def best_assignment(text_centers, visual_centers):
    matrix = text_centers @ visual_centers.t()
    k = matrix.shape[0]

    if k == 1:
        return [0], float(matrix[0, 0]), matrix

    best_perm, best_score = None, -math.inf
    for perm in permutations(range(k)):
        score = sum(float(matrix[i, perm[i]]) for i in range(k)) / k
        if score > best_score:
            best_perm, best_score = list(perm), score

    return best_perm, best_score, matrix


def pair_consistency(text_labels, visual_labels, permutation):
    mapped = torch.tensor(
        [permutation[int(label)] for label in text_labels],
        dtype=torch.long,
    )
    return float((mapped == visual_labels).float().mean().item())


def marginal_chance(text_labels, visual_labels, permutation, k):
    mapped = torch.tensor(
        [permutation[int(label)] for label in text_labels],
        dtype=torch.long,
    )
    p_text = torch.bincount(mapped, minlength=k).float() / len(mapped)
    p_visual = torch.bincount(visual_labels, minlength=k).float() / len(visual_labels)
    return float((p_text * p_visual).sum().item())


def matrix_selectivity(matrix):
    """
    两个简单指标：
    row_margin: 每个文本原型 best - second best 的均值；
    col_hubness: 有多少文本原型把同一视觉列选为 best，越高越 hub。
    """
    if matrix.shape[1] <= 1:
        return 0.0, 1.0

    top2 = torch.topk(matrix, k=2, dim=1).values
    row_margin = float((top2[:, 0] - top2[:, 1]).mean().item())

    best_cols = torch.argmax(matrix, dim=1)
    counts = torch.bincount(best_cols, minlength=matrix.shape[1]).float()
    col_hubness = float((counts.max() / matrix.shape[0]).item())
    return row_margin, col_hubness


def save_matrix(path, matrix, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Visual prototype")
    ax.set_ylabel("Text prototype")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    cache_blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache = cache_blob["records"]
    groups = load_samples(args.samples_csv, args.classes)
    phrase_features = encode_phrases(
        model, groups, device, args.text_batch_size
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("GDINO OBJECT PURIFICATION DIAGNOSTIC")
    print("=" * 112)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Area range : [{args.min_area}, {args.max_area}]")
    print("Modes      : raw_mean / filtered_top1 / filtered_mean")
    print("Comparison : all modes use the SAME common subset with >=1 valid object proposal")
    print("=" * 112)

    rows = []
    summary = {
        "metadata": {
            "cache": args.cache,
            "samples_csv": args.samples_csv,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "comparison_subset": "common samples with >=1 valid proposal",
            "modes": ["raw_mean", "filtered_top1", "filtered_mean"],
        },
        "classes": {},
    }

    for class_name in args.classes:
        samples = groups[class_name]
        prepared = []

        for sample in samples:
            record = cache.get(cache_key(class_name, sample["image_id"]))
            if record is None:
                continue

            variants = get_visual_variants(
                record, args.min_area, args.max_area
            )
            if variants is None:
                continue

            valid_indices = variants["valid_indices"]
            if not valid_indices:
                continue

            prepared.append({
                "sample": sample,
                "text": phrase_features[sample["phrase"].lower()],
                "raw_mean": variants["raw_mean"],
                "filtered_top1": variants["filtered_top1"],
                "filtered_mean": variants["filtered_mean"],
                "num_all": int(record["num_boxes"]),
                "num_valid": len(valid_indices),
                "valid_areas": [float(record["area_ratios"][i]) for i in valid_indices],
                "valid_scores": [float(record["scores"][i]) for i in valid_indices],
            })

        total = len(samples)
        common = len(prepared)
        coverage = common / max(total, 1)

        if common < 2:
            print(f"\n{class_name}: insufficient common samples ({common}/{total})")
            continue

        text_features = F.normalize(
            torch.stack([item["text"] for item in prepared]), dim=-1
        )

        class_summary = {
            "total_samples": total,
            "common_samples": common,
            "object_valid_coverage": coverage,
            "avg_all_boxes": float(np.mean([x["num_all"] for x in prepared])),
            "avg_valid_boxes": float(np.mean([x["num_valid"] for x in prepared])),
            "modes": {},
        }

        print(
            f"\n{class_name}: common={common}/{total} ({coverage:.1%}), "
            f"avg_all={class_summary['avg_all_boxes']:.2f}, "
            f"avg_valid={class_summary['avg_valid_boxes']:.2f}"
        )

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for mode_index, mode in enumerate(
            ["raw_mean", "filtered_top1", "filtered_mean"]
        ):
            visual_features = F.normalize(
                torch.stack([item[mode] for item in prepared]), dim=-1
            )
            paired_cos = float(
                (text_features * visual_features).sum(dim=-1).mean().item()
            )

            mode_summary = {
                "paired_text_visual_cosine": paired_cos,
                "k_results": {},
            }

            print(f"  [{mode}] paired cosine={paired_cos:.4f}")

            for k in args.cluster_ks:
                if k < 1 or k > common:
                    continue

                text_labels, text_centers = spherical_kmeans(
                    text_features, k, args.seed + 17,
                    args.kmeans_iters, args.kmeans_restarts,
                )
                visual_labels, visual_centers = spherical_kmeans(
                    visual_features, k, args.seed + 37 + 100 * mode_index,
                    args.kmeans_iters, args.kmeans_restarts,
                )

                text_sil = cosine_silhouette(
                    text_features, text_labels.clone(), k
                )
                visual_sil = cosine_silhouette(
                    visual_features, visual_labels.clone(), k
                )
                text_comp = compactness(
                    text_features, text_labels, text_centers
                )
                visual_comp = compactness(
                    visual_features, visual_labels, visual_centers
                )

                permutation, proto_cos, matrix = best_assignment(
                    text_centers, visual_centers
                )
                pair_cons = pair_consistency(
                    text_labels, visual_labels, permutation
                )
                chance = marginal_chance(
                    text_labels, visual_labels, permutation, k
                )
                excess = pair_cons - chance
                row_margin, hubness = matrix_selectivity(matrix)

                result = {
                    "k": k,
                    "text_silhouette": text_sil,
                    "visual_silhouette": visual_sil,
                    "text_compactness": text_comp,
                    "visual_compactness": visual_comp,
                    "prototype_alignment_cosine": proto_cos,
                    "paired_cluster_consistency": pair_cons,
                    "marginal_chance_consistency": chance,
                    "paircons_excess_over_chance": excess,
                    "prototype_row_margin": row_margin,
                    "prototype_column_hubness": hubness,
                    "prototype_similarity_matrix": matrix.tolist(),
                }
                mode_summary["k_results"][str(k)] = result

                rows.append({
                    "class": class_name,
                    "mode": mode,
                    "common_samples": common,
                    "object_valid_coverage": coverage,
                    "paired_text_visual_cosine": paired_cos,
                    **{key: value for key, value in result.items()
                       if key != "prototype_similarity_matrix"},
                })

                save_matrix(
                    class_dir / f"{mode}_k{k}_matrix.png",
                    matrix,
                    (
                        f"{class_name} | {mode} | K={k} | "
                        f"ProtoCos={proto_cos:.3f} | "
                        f"PairExcess={excess:+.3f} | "
                        f"RowMargin={row_margin:.3f} | Hub={hubness:.2f}"
                    ),
                )

                print(
                    f"    K={k:<2} | "
                    f"V-sil={visual_sil if visual_sil is not None else float('nan'):+.3f} | "
                    f"ProtoCos={proto_cos:.3f} | "
                    f"Pair={pair_cons:.3f} | Chance={chance:.3f} | "
                    f"Excess={excess:+.3f} | "
                    f"RowMargin={row_margin:.3f} | Hub={hubness:.2f}"
                )

            class_summary["modes"][mode] = mode_summary

        summary["classes"][class_name] = class_summary

    summary_path = output_dir / "gdino_object_purification_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "gdino_object_purification_metrics.csv"
    if rows:
        with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 112)
    print("READING ORDER")
    print("=" * 112)
    print("1. 先看 object_valid_coverage，过滤不能只剩少量漂亮样本。")
    print("2. 三种 mode 在同一 common subset 上比较，避免 sample selection 混淆。")
    print("3. PairCons 重点看 Excess=PairCons-MargChance，而不是只看 PairCons。")
    print("4. RowMargin 越大表示每个 Text prototype 对某个 Visual prototype 更有选择性。")
    print("5. Hub 越接近 1，表示大量 Text prototypes 都挤向同一个 Visual prototype，越不好。")
    print("6. 如果过滤后 V-sil 上升但 Excess/RowMargin 不升，说明主要问题已不是 localization。")
    print("-" * 112)
    print(f"Summary: {summary_path}")
    print(f"Metrics: {metrics_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
