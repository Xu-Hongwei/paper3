import argparse
import csv
import json
import math
import random
import re
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

from datasets import create_dataset
from datasets.structured_semantics import StructuredSemanticsReader
from models import CLIPRetrieval


ATTRIBUTE_ORDER = {"size": 0, "color": 1, "shape": 2, "state": 3}
DEFAULT_ATTRIBUTE_TYPES = ["color", "size", "shape", "state"]

# 第一轮只选语义边界比较清楚的粗类，后续可继续扩。
CLASS_PATTERNS = {
    "building": [r"\bbuilding\b", r"\bbuildings\b"],
    "aircraft": [
        r"\baircraft\b", r"\bairplane\b", r"\bairplanes\b",
        r"\bplane\b", r"\bplanes\b", r"\bjet\b", r"\bjets\b",
    ],
    "river": [r"\briver\b", r"\brivers\b"],
    "stadium": [r"\bstadium\b", r"\bstadiums\b"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype feasibility diagnostic: coarse class -> text/visual fine subclusters."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--ear-file", type=str,
        default="E:/paper3/data/structured_semantics/rsicd_train_qwen37_v30_open.json",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/prototype_feasibility")
    parser.add_argument(
        "--classes", type=str, nargs="+",
        default=["building", "aircraft", "river", "stadium"],
        choices=sorted(CLASS_PATTERNS),
    )
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument(
        "--require-attribute", action="store_true",
        help="只保留 Entity+Attribute 文本确实发生增强的样本。",
    )
    parser.add_argument(
        "--attribute-types", type=str, nargs="+", default=DEFAULT_ATTRIBUTE_TYPES,
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--local-topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--cluster-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--montage-k", type=int, default=4)
    parser.add_argument("--montage-per-cluster", type=int, default=6)
    parser.add_argument("--region-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=80)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-region-cache", action="store_true")
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def coarse_class(entity_text, classes):
    text = entity_text.lower()
    for name in classes:
        if any(re.search(pattern, text) for pattern in CLASS_PATTERNS[name]):
            return name
    return None


def build_phrase(entity_text, attributes, allowed_types):
    attrs = [attr for attr in attributes if attr["type"] in allowed_types]
    attrs.sort(key=lambda x: ATTRIBUTE_ORDER.get(x["type"], 99))

    values = []
    entity_lower = entity_text.lower()
    for attr in attrs:
        value = attr["value"].strip()
        value_lower = value.lower()
        if not value or value_lower in entity_lower:
            continue
        if value_lower not in {x.lower() for x in values}:
            values.append(value)

    return " ".join(values + [entity_text]).strip()


def is_absent(attributes):
    absent_values = {"absent", "no", "none", "without", "not present"}
    return any(
        attr["type"] == "presence"
        and attr["value"].strip().lower() in absent_values
        for attr in attributes
    )


def collect_occurrences(
    dataset, reader, classes, allowed_types, max_per_class,
    require_attribute, seed,
):
    groups = {name: [] for name in classes}
    seen = {name: set() for name in classes}

    candidates = list(range(len(dataset)))
    random.Random(seed).shuffle(candidates)

    for dataset_index in candidates:
        pair_index = int(dataset.ann[dataset_index]["_pair_index"])
        image_id = int(dataset.image_ids[dataset_index])

        try:
            semantics = reader.get_by_pair(pair_index)
        except KeyError:
            continue

        for entity in semantics["entities"]:
            entity_text = entity["text"].strip()
            cls = coarse_class(entity_text, classes)
            if cls is None or image_id in seen[cls] or len(groups[cls]) >= max_per_class:
                continue

            attributes = entity["attributes"]
            if is_absent(attributes):
                continue

            phrase = build_phrase(entity_text, attributes, allowed_types)
            changed = phrase.lower() != entity_text.lower()
            if require_attribute and not changed:
                continue

            groups[cls].append({
                "dataset_index": int(dataset_index),
                "pair_index": pair_index,
                "image_id": image_id,
                "image": dataset.ann[dataset_index]["image"],
                "caption": dataset.ann[dataset_index]["caption"],
                "entity": entity_text,
                "phrase": phrase,
                "attribute_changed": changed,
                "attributes": [
                    attr for attr in attributes if attr["type"] in allowed_types
                ],
            })
            seen[cls].add(image_id)

        if all(len(groups[name]) >= max_per_class for name in classes):
            break

    return groups


def sliding_positions(image_size, window, stride):
    positions = list(range(0, image_size - window + 1, stride))
    last = image_size - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_regions(image_size, windows):
    regions = []
    for window in windows:
        if window <= 0 or window > image_size:
            raise ValueError(f"Invalid window={window}")

        stride = max(window // 2, 1)
        xs = sliding_positions(image_size, window, stride)
        ys = sliding_positions(image_size, window, stride)

        for y1 in ys:
            for x1 in xs:
                regions.append({
                    "scale": int(window),
                    "box": [x1, y1, x1 + window, y1 + window],
                })
    return regions


def build_region_crops(image, regions):
    image_size = image.shape[-1]
    crops = []

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop, size=(image_size, image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        crops.append(crop.squeeze(0))

    return torch.stack(crops)


@torch.no_grad()
def build_region_cache(
    model, dataset, image_records, regions, device, batch_size,
    cache_path, rebuild,
):
    image_ids = sorted(image_records)

    if cache_path.exists() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            [int(x) for x in cache["image_ids"]] == image_ids
            and cache["features"].shape[1] == len(regions)
        ):
            print(f"Loaded Region cache: {cache_path}")
            return {
                int(image_id): cache["features"][row]
                for row, image_id in enumerate(cache["image_ids"])
            }
        print("Region cache 与当前样本集合不一致，重新提取。")

    features = []
    print(f"\nExtracting Regions for {len(image_ids)} unique images...")

    for pos, image_id in enumerate(image_ids):
        dataset_index = image_records[image_id]
        image = dataset[dataset_index][0]
        crops = build_region_crops(image, regions)

        batch_features = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size].to(device, non_blocking=True)
            batch_features.append(
                model.backbone.encode_image(batch, normalize=True).cpu()
            )

        features.append(torch.cat(batch_features).to(torch.float16))

        if pos == 0 or (pos + 1) % 20 == 0 or pos + 1 == len(image_ids):
            print(f"  [{pos + 1:>4}/{len(image_ids)}]")

    features = torch.stack(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "image_ids": image_ids,
        "features": features,
        "num_regions": len(regions),
    }, cache_path)
    print(f"Region cache saved: {cache_path}")

    return {image_id: features[row] for row, image_id in enumerate(image_ids)}


@torch.no_grad()
def encode_phrases(model, groups, device, batch_size):
    unique, seen = [], set()
    for samples in groups.values():
        for sample in samples:
            key = sample["phrase"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(sample["phrase"])

    features = []
    for start in range(0, len(unique), batch_size):
        features.append(
            model.backbone.encode_text(
                unique[start:start + batch_size], normalize=True
            ).cpu()
        )

    features = torch.cat(features)
    return {phrase.lower(): features[i] for i, phrase in enumerate(unique)}


def matched_visual_feature(
    text_feature, region_features, regions, topk, temperature,
):
    region_features = region_features.float()
    scores = region_features @ text_feature.float()

    k = min(topk, len(scores))
    values, indices = torch.topk(scores, k=k)
    weights = F.softmax(values / temperature, dim=0)

    pooled = (weights.unsqueeze(1) * region_features[indices]).sum(dim=0)
    pooled = F.normalize(pooled, dim=0)

    best = int(indices[0].item())
    return pooled, {
        "top1_region_index": best,
        "top1_score": float(values[0].item()),
        "top1_scale": regions[best]["scale"],
        "top1_box": regions[best]["box"],
    }


def spherical_kmeans(x, k, seed, max_iters, restarts):
    x = F.normalize(x.float(), dim=-1)
    n = len(x)

    if k == 1:
        center = F.normalize(x.mean(dim=0, keepdim=True), dim=-1)
        labels = torch.zeros(n, dtype=torch.long)
        objective = float((x @ center.t()).sum().item())
        return labels, center, objective

    best = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + 1009 * restart + k)
        init = torch.randperm(n, generator=generator)[:k]
        centers = x[init].clone()
        labels = None

        for _ in range(max_iters):
            new_labels = torch.argmax(x @ centers.t(), dim=1)
            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels

            new_centers = []
            for cluster in range(k):
                mask = labels == cluster
                if mask.any():
                    center = x[mask].mean(dim=0)
                else:
                    center = x[
                        torch.randint(0, n, (1,), generator=generator).item()
                    ]
                new_centers.append(F.normalize(center, dim=0))
            centers = torch.stack(new_centers)

        objective = float((x * centers[labels]).sum().item())
        if best is None or objective > best[2]:
            best = (labels.clone(), centers.clone(), objective)

    return best


def cosine_silhouette(x, labels, k):
    if k <= 1 or len(x) <= k:
        return None

    x = F.normalize(x.float(), dim=-1)
    distance = 1.0 - x @ x.t()
    values = []

    for i in range(len(x)):
        own = int(labels[i].item())
        own_mask = labels == own
        own_mask = own_mask.clone()
        own_mask[i] = False

        if not own_mask.any():
            values.append(0.0)
            continue

        a = distance[i][own_mask].mean()
        b_values = []

        for cluster in range(k):
            if cluster == own:
                continue
            mask = labels == cluster
            if mask.any():
                b_values.append(distance[i][mask].mean())

        if not b_values:
            values.append(0.0)
            continue

        b = torch.stack(b_values).min()
        denom = torch.maximum(a, b).clamp_min(1e-8)
        values.append(float(((b - a) / denom).item()))

    return float(np.mean(values))


def centroid_compactness(x, labels, centers):
    x = F.normalize(x.float(), dim=-1)
    return float((x * centers[labels]).sum(dim=-1).mean().item())


def best_prototype_assignment(text_centers, visual_centers):
    similarity = text_centers @ visual_centers.t()
    k = similarity.shape[0]

    if k == 1:
        return [0], float(similarity[0, 0]), similarity

    best_perm, best_score = None, -math.inf
    for perm in permutations(range(k)):
        score = sum(
            float(similarity[i, perm[i]].item()) for i in range(k)
        ) / k
        if score > best_score:
            best_perm = list(perm)
            best_score = score

    return best_perm, best_score, similarity


def assignment_consistency(text_labels, visual_labels, permutation):
    mapped = torch.tensor(
        [permutation[int(label)] for label in text_labels], dtype=torch.long
    )
    return float((mapped == visual_labels).float().mean().item())


def pca_2d(*arrays):
    sizes = [len(array) for array in arrays]
    x = torch.cat([array.float() for array in arrays], dim=0)
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].t()

    result = []
    start = 0
    for size in sizes:
        result.append(coords[start:start + size].numpy())
        start += size
    return result


def save_diagnostic_figure(
    path, text_features, visual_features, text_labels, visual_labels,
    text_centers, visual_centers, prototype_similarity, permutation, title,
):
    text_xy, visual_xy, text_center_xy, visual_center_xy = pca_2d(
        text_features, visual_features, text_centers, visual_centers
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].scatter(
        text_xy[:, 0], text_xy[:, 1], c=text_labels.numpy(), s=24
    )
    axes[0, 0].set_title("Text Entity+Attribute clusters")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")

    axes[0, 1].scatter(
        visual_xy[:, 0], visual_xy[:, 1], c=visual_labels.numpy(), s=24
    )
    axes[0, 1].set_title("Matched Region clusters")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    axes[1, 0].scatter(
        text_xy[:, 0], text_xy[:, 1], marker="o", s=16,
        alpha=0.35, label="Text samples",
    )
    axes[1, 0].scatter(
        visual_xy[:, 0], visual_xy[:, 1], marker="x", s=18,
        alpha=0.35, label="Visual samples",
    )
    axes[1, 0].scatter(
        text_center_xy[:, 0], text_center_xy[:, 1], marker="o",
        s=120, label="Text prototypes",
    )
    axes[1, 0].scatter(
        visual_center_xy[:, 0], visual_center_xy[:, 1], marker="X",
        s=120, label="Visual prototypes",
    )

    for text_cluster, visual_cluster in enumerate(permutation):
        axes[1, 0].plot(
            [text_center_xy[text_cluster, 0], visual_center_xy[visual_cluster, 0]],
            [text_center_xy[text_cluster, 1], visual_center_xy[visual_cluster, 1]],
            linewidth=1.5,
        )

    axes[1, 0].set_title("Joint CLIP space + best prototype alignment")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")

    image = axes[1, 1].imshow(prototype_similarity.numpy(), aspect="auto")
    axes[1, 1].set_title("Text ↔ Visual prototype cosine")
    axes[1, 1].set_xlabel("Visual prototype")
    axes[1, 1].set_ylabel("Text prototype")
    axes[1, 1].set_xticks(range(len(prototype_similarity)))
    axes[1, 1].set_yticks(range(len(prototype_similarity)))

    for i in range(prototype_similarity.shape[0]):
        for j in range(prototype_similarity.shape[1]):
            axes[1, 1].text(
                j, i, f"{prototype_similarity[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
            )

    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def recover_input_image(image_tensor, preprocess):
    mean = std = None
    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            mean = torch.as_tensor(transform.mean).view(3, 1, 1)
            std = torch.as_tensor(transform.std).view(3, 1, 1)
            break

    image = image_tensor.detach().cpu()
    if mean is not None and std is not None:
        image = image * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def save_visual_montage(
    path, dataset, preprocess, samples, visual_labels, k, per_cluster,
):
    fig, axes = plt.subplots(
        k, per_cluster, figsize=(2.7 * per_cluster, 2.7 * k), squeeze=False
    )

    for cluster in range(k):
        members = [
            i for i, label in enumerate(visual_labels.tolist()) if label == cluster
        ]

        for col in range(per_cluster):
            ax = axes[cluster, col]
            ax.axis("off")
            if col >= len(members):
                continue

            sample = samples[members[col]]
            image = dataset[sample["dataset_index"]][0]
            image = recover_input_image(image, preprocess)

            x1, y1, x2, y2 = sample["top1_box"]
            crop = image[y1:y2, x1:x2]
            ax.imshow(crop)
            ax.set_title(
                f"C{cluster} | {sample['phrase']}\n"
                f"s={sample['top1_score']:.3f}, {sample['top1_scale']}px",
                fontsize=8,
            )

    fig.suptitle("Top-1 Region examples by visual cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_cluster_texts(path, samples, labels, k):
    data = {}
    for cluster in range(k):
        members = [
            samples[i] for i, label in enumerate(labels.tolist()) if label == cluster
        ]
        data[str(cluster)] = [
            {
                "phrase": sample["phrase"],
                "entity": sample["entity"],
                "caption": sample["caption"],
                "image": sample["image"],
            }
            for sample in members
        ]

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_samples_csv(path, all_samples):
    fields = [
        "class", "dataset_index", "pair_index", "image_id", "image", "caption",
        "entity", "phrase", "attribute_changed", "attributes", "paired_cosine",
        "top1_region_index", "top1_score", "top1_scale", "top1_box",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)


def main():
    args = parse_args()

    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be > 0.")
    if args.local_topk <= 0:
        raise ValueError("--local-topk must be > 0.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"], evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    reader = StructuredSemanticsReader(args.ear_file)
    allowed_types = set(args.attribute_types)

    groups = collect_occurrences(
        train_dataset, reader, args.classes, allowed_types,
        args.max_per_class, args.require_attribute, args.seed,
    )

    print("=" * 108)
    print("PROTOTYPE FEASIBILITY DIAGNOSTIC")
    print("=" * 108)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Frozen EAR       : {args.ear_file}")
    print(f"Classes          : {args.classes}")
    print(f"Attributes       : {sorted(allowed_types)}")
    print(f"Require attr     : {args.require_attribute}")
    print(f"Cluster K        : {args.cluster_ks}")
    for name in args.classes:
        print(f"{name:<12}: {len(groups[name])} unique-image occurrences")
    print("=" * 108)

    image_records = {}
    for samples in groups.values():
        for sample in samples:
            image_records.setdefault(sample["image_id"], sample["dataset_index"])

    image_size = int(config["dataset"].get("image_res", 224))
    regions = generate_regions(image_size, args.windows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_tag = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    cache_path = output_dir / (
        f"region_cache_{checkpoint_tag}_{'-'.join(map(str, args.windows))}.pt"
    )

    region_cache = build_region_cache(
        model, train_dataset, image_records, regions, device,
        args.region_batch_size, cache_path, args.rebuild_region_cache,
    )

    print("\nEncoding Entity+Attribute phrases...")
    phrase_features = encode_phrases(
        model, groups, device, args.text_batch_size
    )

    summary = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "ear_file": args.ear_file,
            "classes": args.classes,
            "attribute_types": sorted(allowed_types),
            "require_attribute": args.require_attribute,
            "max_per_class": args.max_per_class,
            "windows": args.windows,
            "num_regions": len(regions),
            "local_topk": args.local_topk,
            "temperature": args.temperature,
            "cluster_ks": args.cluster_ks,
            "cluster_method": "spherical k-means in normalized CLIP joint space",
            "visual_sample": "Top-K soft pooled Full-CLIP Region feature",
        },
        "classes": {},
    }

    metric_rows = []
    all_sample_rows = []

    for class_index, class_name in enumerate(args.classes):
        samples = groups[class_name]
        if len(samples) < 2:
            print(f"\nSkip {class_name}: insufficient samples.")
            continue

        print(f"\n[{class_index + 1}/{len(args.classes)}] {class_name}: {len(samples)} samples")

        text_features = []
        visual_features = []

        for sample in samples:
            text_feature = phrase_features[sample["phrase"].lower()]
            visual_feature, match_info = matched_visual_feature(
                text_feature, region_cache[sample["image_id"]], regions,
                args.local_topk, args.temperature,
            )

            text_features.append(text_feature)
            visual_features.append(visual_feature)

            sample.update(match_info)
            sample["paired_cosine"] = float(
                (text_feature.float() @ visual_feature.float()).item()
            )
            all_sample_rows.append({"class": class_name, **sample})

        text_features = F.normalize(torch.stack(text_features).float(), dim=-1)
        visual_features = F.normalize(torch.stack(visual_features).float(), dim=-1)

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        paired_cosine = float(
            (text_features * visual_features).sum(dim=-1).mean().item()
        )

        class_summary = {
            "num_samples": len(samples),
            "attribute_changed_rate": float(
                np.mean([sample["attribute_changed"] for sample in samples])
            ),
            "paired_text_visual_cosine": paired_cosine,
            "k_results": {},
        }

        valid_ks = [k for k in args.cluster_ks if 1 <= k <= len(samples)]

        for k in valid_ks:
            text_labels, text_centers, _ = spherical_kmeans(
                text_features, k, args.seed + 17,
                args.kmeans_iters, args.kmeans_restarts,
            )
            visual_labels, visual_centers, _ = spherical_kmeans(
                visual_features, k, args.seed + 37,
                args.kmeans_iters, args.kmeans_restarts,
            )

            text_sil = cosine_silhouette(text_features, text_labels.clone(), k)
            visual_sil = cosine_silhouette(visual_features, visual_labels.clone(), k)
            text_compact = centroid_compactness(
                text_features, text_labels, text_centers
            )
            visual_compact = centroid_compactness(
                visual_features, visual_labels, visual_centers
            )

            permutation, proto_cos, proto_matrix = best_prototype_assignment(
                text_centers, visual_centers
            )
            consistency = assignment_consistency(
                text_labels, visual_labels, permutation
            )

            text_counts = torch.bincount(text_labels, minlength=k).tolist()
            visual_counts = torch.bincount(visual_labels, minlength=k).tolist()

            result = {
                "k": k,
                "text_silhouette": text_sil,
                "visual_silhouette": visual_sil,
                "text_compactness": text_compact,
                "visual_compactness": visual_compact,
                "prototype_alignment_cosine": proto_cos,
                "paired_cluster_consistency": consistency,
                "chance_consistency": 1.0 / k,
                "text_cluster_sizes": text_counts,
                "visual_cluster_sizes": visual_counts,
                "best_text_to_visual_permutation": permutation,
                "prototype_similarity_matrix": proto_matrix.tolist(),
            }
            class_summary["k_results"][str(k)] = result

            metric_rows.append({
                "class": class_name,
                "k": k,
                "text_silhouette": text_sil,
                "visual_silhouette": visual_sil,
                "text_compactness": text_compact,
                "visual_compactness": visual_compact,
                "prototype_alignment_cosine": proto_cos,
                "paired_cluster_consistency": consistency,
                "chance_consistency": 1.0 / k,
                "text_cluster_sizes": text_counts,
                "visual_cluster_sizes": visual_counts,
            })

            figure_path = class_dir / f"k{k}_diagnostic.png"
            save_diagnostic_figure(
                figure_path, text_features, visual_features,
                text_labels, visual_labels, text_centers, visual_centers,
                proto_matrix, permutation,
                (
                    f"{class_name} | K={k} | "
                    f"T-sil={text_sil if text_sil is not None else float('nan'):.3f} | "
                    f"V-sil={visual_sil if visual_sil is not None else float('nan'):.3f} | "
                    f"ProtoCos={proto_cos:.3f} | PairCons={consistency:.3f}"
                ),
            )

            save_cluster_texts(
                class_dir / f"k{k}_text_clusters.json", samples, text_labels, k
            )

            if k == args.montage_k:
                save_visual_montage(
                    class_dir / f"k{k}_visual_montage.png",
                    train_dataset, model.backbone.preprocess_val,
                    samples, visual_labels, k, args.montage_per_cluster,
                )

            print(
                f"  K={k:<2} | "
                f"T-sil={text_sil if text_sil is not None else float('nan'):+.3f} | "
                f"V-sil={visual_sil if visual_sil is not None else float('nan'):+.3f} | "
                f"T-comp={text_compact:.3f} | V-comp={visual_compact:.3f} | "
                f"ProtoCos={proto_cos:.3f} | PairCons={consistency:.3f} "
                f"(chance={1.0 / k:.3f})"
            )

        summary["classes"][class_name] = class_summary

    summary_path = output_dir / "prototype_feasibility_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "prototype_feasibility_metrics.csv"
    if metric_rows:
        with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    samples_path = output_dir / "prototype_samples.csv"
    if all_sample_rows:
        write_samples_csv(samples_path, all_sample_rows)

    print("\n" + "=" * 108)
    print("HOW TO READ")
    print("=" * 108)
    print("1. K>1 时 Text/Visual silhouette > 0：类内存在可分子结构。")
    print("2. K 增大后 compactness 提升：多原型比单原型更能解释类内变化。")
    print("3. Prototype alignment cosine 高：Text/Visual 子模式在 CLIP 空间可对应。")
    print("4. Paired cluster consistency 明显 > 1/K：配对样本在两模态形成相似子模式。")
    print("5. k4_visual_montage.png：人工检查视觉簇是否真有可解释语义，而非背景/尺度伪簇。")
    print("-" * 108)
    print(f"Summary : {summary_path}")
    print(f"Metrics : {metrics_path}")
    print(f"Samples : {samples_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
