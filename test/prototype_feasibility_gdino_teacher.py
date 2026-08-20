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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from datasets.structured_semantics import StructuredSemanticsReader
from models import CLIPRetrieval
from models.grounding_dino import GroundingDINODetector


ATTRIBUTE_ORDER = {"size": 0, "color": 1, "shape": 2, "state": 3}
DEFAULT_ATTRIBUTE_TYPES = ["color", "size", "shape", "state"]

CLASS_PATTERNS = {
    "building": [r"\bbuilding\b", r"\bbuildings\b"],
    "aircraft": [
        r"\baircraft\b", r"\bairplane\b", r"\bairplanes\b",
        r"\bplane\b", r"\bplanes\b", r"\bjet\b", r"\bjets\b",
    ],
    "river": [r"\briver\b", r"\brivers\b"],
    "stadium": [r"\bstadium\b", r"\bstadiums\b"],
}

# GDINO 只接收 coarse identity，禁止把 fine attribute 泄漏给视觉区域选择。
CLASS_QUERIES = {
    "building": "building",
    "aircraft": "airplane",
    "river": "river",
    "stadium": "stadium",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teacher/Oracle diagnostic: coarse Entity -> Grounding DINO boxes -> CLIP visual prototypes."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--ear-file", type=str,
        default="E:/paper3/data/structured_semantics/rsicd_train_qwen37_v30_open.json",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/prototype_feasibility_gdino_teacher",
    )
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
        "--attribute-types", type=str, nargs="+",
        default=DEFAULT_ATTRIBUTE_TYPES,
    )

    parser.add_argument(
        "--gdino-model-id", type=str,
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--max-boxes", type=int, default=5,
        help="每个 coarse Entity 最多保留多少个 GDINO proposal。",
    )
    parser.add_argument(
        "--min-box-area", type=float, default=0.001,
        help="按 224 view 面积比例过滤极小框。",
    )
    parser.add_argument(
        "--max-box-area", type=float, default=1.0,
        help="允许整图 river proposal；当前是 teacher/oracle diagnostic，不人为修框。",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rebuild-gdino-cache", action="store_true")

    parser.add_argument("--cluster-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--montage-k", type=int, default=4)
    parser.add_argument("--montage-per-cluster", type=int, default=6)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--visual-batch-size", type=int, default=64)
    parser.add_argument("--kmeans-iters", type=int, default=80)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
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


def recover_rgb_uint8(image_tensor, preprocess):
    mean = std = None
    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            mean = torch.as_tensor(transform.mean).view(3, 1, 1)
            std = torch.as_tensor(transform.std).view(3, 1, 1)
            break

    image = image_tensor.detach().cpu()
    if mean is not None and std is not None:
        image = image * std + mean

    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(x1 + 1.0, min(x2, width))
    y2 = max(y1 + 1.0, min(y2, height))
    return [x1, y1, x2, y2]


def box_area_ratio(box, width, height):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height)


def crop_rgb(image_rgb, box):
    x1, y1, x2, y2 = box
    x1, y1 = int(math.floor(x1)), int(math.floor(y1))
    x2, y2 = int(math.ceil(x2)), int(math.ceil(y2))
    x1 = max(0, min(x1, image_rgb.shape[1] - 1))
    y1 = max(0, min(y1, image_rgb.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image_rgb.shape[1]))
    y2 = max(y1 + 1, min(y2, image_rgb.shape[0]))
    return image_rgb[y1:y2, x1:x2]


@torch.no_grad()
def encode_pil_batch(model, images, device, batch_size):
    if not images:
        return torch.empty(0)

    tensors = torch.stack([
        model.backbone.preprocess_val(Image.fromarray(image))
        for image in images
    ])

    features = []
    for start in range(0, len(tensors), batch_size):
        batch = tensors[start:start + batch_size].to(device, non_blocking=True)
        features.append(model.backbone.encode_image(batch, normalize=True).cpu())

    return torch.cat(features)


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


def cache_key(class_name, image_id):
    return f"{class_name}:{int(image_id)}"


@torch.no_grad()
def build_gdino_teacher_cache(
    clip_model, detector, dataset, groups, device, cache_path, args,
):
    expected_keys = {
        cache_key(class_name, sample["image_id"])
        for class_name, samples in groups.items()
        for sample in samples
    }

    cache = {}
    if cache_path.exists() and not args.rebuild_gdino_cache:
        loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        if loaded.get("metadata", {}) == {
            "gdino_model_id": args.gdino_model_id,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "max_boxes": args.max_boxes,
            "min_box_area": args.min_box_area,
            "max_box_area": args.max_box_area,
        }:
            cache = loaded.get("records", {})
            print(f"Loaded GDINO teacher cache: {cache_path}")
        else:
            print("GDINO cache 配置不一致，将重新构建。")

    missing = sorted(expected_keys - set(cache))
    if not missing:
        return cache

    sample_lookup = {}
    for class_name, samples in groups.items():
        for sample in samples:
            sample_lookup[cache_key(class_name, sample["image_id"])] = (
                class_name, sample
            )

    print(f"\nBuilding GDINO teacher cache: missing {len(missing)} / {len(expected_keys)}")

    for pos, key in enumerate(missing, start=1):
        class_name, sample = sample_lookup[key]
        image_tensor = dataset[sample["dataset_index"]][0]
        image_rgb = recover_rgb_uint8(
            image_tensor, clip_model.backbone.preprocess_val
        )
        height, width = image_rgb.shape[:2]
        query = CLASS_QUERIES[class_name]

        result = detector.predict(
            image_rgb,
            text=query,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        order = np.argsort(result["scores"])[::-1]
        boxes, scores, areas, crops = [], [], [], []

        for idx in order:
            box = clip_box(result["boxes"][idx], width, height)
            area = box_area_ratio(box, width, height)

            if not args.min_box_area <= area <= args.max_box_area:
                continue

            boxes.append(box)
            scores.append(float(result["scores"][idx]))
            areas.append(float(area))
            crops.append(crop_rgb(image_rgb, box))

            if len(boxes) >= args.max_boxes:
                break

        if crops:
            instance_features = encode_pil_batch(
                clip_model, crops, device, args.visual_batch_size
            )
            # 第一版等权聚合，避免把 GDINO confidence 当成额外语义监督。
            visual_feature = F.normalize(
                instance_features.float().mean(dim=0), dim=0
            )
        else:
            instance_features = torch.empty(
                (0, clip_model.backbone.output_dim), dtype=torch.float32
            ) if hasattr(clip_model.backbone, "output_dim") else torch.empty((0, 0))
            visual_feature = None

        cache[key] = {
            "class": class_name,
            "image_id": int(sample["image_id"]),
            "dataset_index": int(sample["dataset_index"]),
            "query": query,
            "boxes": boxes,
            "scores": scores,
            "area_ratios": areas,
            "num_boxes": len(boxes),
            "visual_feature": (
                visual_feature.to(torch.float16).cpu()
                if visual_feature is not None else None
            ),
            "instance_features": (
                instance_features.to(torch.float16).cpu()
                if len(crops) else None
            ),
        }

        if pos == 1 or pos % 20 == 0 or pos == len(missing):
            print(f"  [{pos:>4}/{len(missing)}] {class_name:<9} detections={len(boxes)}")

        if pos % 20 == 0 or pos == len(missing):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "metadata": {
                    "gdino_model_id": args.gdino_model_id,
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "max_boxes": args.max_boxes,
                    "min_box_area": args.min_box_area,
                    "max_box_area": args.max_box_area,
                },
                "records": cache,
            }, cache_path)

    print(f"GDINO teacher cache saved: {cache_path}")
    return cache


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
        score = sum(float(similarity[i, perm[i]].item()) for i in range(k)) / k
        if score > best_score:
            best_perm = list(perm)
            best_score = score

    return best_perm, best_score, similarity


def assignment_consistency(text_labels, visual_labels, permutation):
    mapped = torch.tensor(
        [permutation[int(label)] for label in text_labels],
        dtype=torch.long,
    )
    return float((mapped == visual_labels).float().mean().item())


def marginal_chance_consistency(text_labels, visual_labels, permutation, k):
    mapped = torch.tensor(
        [permutation[int(label)] for label in text_labels],
        dtype=torch.long,
    )
    p_text = torch.bincount(mapped, minlength=k).float() / len(mapped)
    p_visual = torch.bincount(visual_labels, minlength=k).float() / len(visual_labels)
    return float((p_text * p_visual).sum().item())


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

    axes[0, 0].scatter(text_xy[:, 0], text_xy[:, 1], c=text_labels.numpy(), s=24)
    axes[0, 0].set_title("Text Entity+Attribute clusters")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")

    axes[0, 1].scatter(visual_xy[:, 0], visual_xy[:, 1], c=visual_labels.numpy(), s=24)
    axes[0, 1].set_title("GDINO-selected CLIP visual clusters")
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


def save_visual_montage(
    path, dataset, preprocess, samples, visual_labels, cache, class_name,
    k, per_cluster,
):
    fig, axes = plt.subplots(
        k, per_cluster, figsize=(2.8 * per_cluster, 2.8 * k), squeeze=False
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
            record = cache[cache_key(class_name, sample["image_id"])]
            image_tensor = dataset[sample["dataset_index"]][0]
            image_rgb = recover_rgb_uint8(image_tensor, preprocess)

            if record["boxes"]:
                crop = crop_rgb(image_rgb, record["boxes"][0])
                ax.imshow(crop)
                ax.set_title(
                    f"C{cluster} | {sample['phrase']}\n"
                    f"n={record['num_boxes']}, s={record['scores'][0]:.3f}, "
                    f"a={record['area_ratios'][0]:.2f}",
                    fontsize=8,
                )
            else:
                ax.imshow(image_rgb)
                ax.set_title(f"C{cluster} | NO DETECTION", fontsize=8)

    fig.suptitle("Top GDINO proposal examples by visual cluster")
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


def write_samples_csv(path, rows):
    fields = [
        "class", "dataset_index", "pair_index", "image_id", "image", "caption",
        "entity", "phrase", "attribute_changed", "attributes",
        "teacher_query", "num_boxes", "top1_gdino_score", "top1_box_area",
        "top1_box", "paired_cosine",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be > 0.")
    if args.max_boxes <= 0:
        raise ValueError("--max-boxes must be > 0.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    clip_model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(clip_model, args.checkpoint)
    clip_model = clip_model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"], evaluate=False,
        train_transform=clip_model.backbone.preprocess_val,
        eval_transform=clip_model.backbone.preprocess_val,
    )

    reader = StructuredSemanticsReader(args.ear_file)
    allowed_types = set(args.attribute_types)
    groups = collect_occurrences(
        train_dataset, reader, args.classes, allowed_types,
        args.max_per_class, args.require_attribute, args.seed,
    )

    print("=" * 112)
    print("GDINO TEACHER / ORACLE PROTOTYPE FEASIBILITY DIAGNOSTIC")
    print("=" * 112)
    print("Role             : diagnostic upper bound / training-time teacher candidate")
    print("Final inference  : Grounding DINO is NOT assumed to be used")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"GDINO            : {args.gdino_model_id}")
    print(f"Classes          : {args.classes}")
    print(f"Coarse queries   : {[CLASS_QUERIES[x] for x in args.classes]}")
    print(f"Attributes       : {sorted(allowed_types)}")
    print(f"Cluster K        : {args.cluster_ks}")
    for name in args.classes:
        print(f"{name:<12}: {len(groups[name])} candidate occurrences")
    print("=" * 112)

    detector = GroundingDINODetector(
        model_id=args.gdino_model_id,
        device=str(device),
        local_files_only=args.local_files_only,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_tag = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    gdino_tag = args.gdino_model_id.replace("/", "--")
    cache_path = output_dir / (
        f"gdino_teacher_cache_{gdino_tag}_{checkpoint_tag}_"
        f"b{args.box_threshold}_t{args.text_threshold}_k{args.max_boxes}.pt"
    )

    teacher_cache = build_gdino_teacher_cache(
        clip_model, detector, train_dataset, groups, device, cache_path, args
    )

    print("\nEncoding Entity+Attribute text features...")
    phrase_features = encode_phrases(
        clip_model, groups, device, args.text_batch_size
    )

    summary = {
        "metadata": {
            "role": "teacher_oracle_diagnostic_only",
            "grounding_dino_used_at_final_inference": False,
            "purpose": (
                "Test whether cleaner externally-grounded regions reveal stronger "
                "visual fine structure before designing Grounding-to-CLIP distillation."
            ),
            "config": args.config,
            "checkpoint": args.checkpoint,
            "ear_file": args.ear_file,
            "gdino_model_id": args.gdino_model_id,
            "classes": args.classes,
            "class_queries": {name: CLASS_QUERIES[name] for name in args.classes},
            "attribute_types": sorted(allowed_types),
            "max_per_class": args.max_per_class,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "max_boxes": args.max_boxes,
            "cluster_ks": args.cluster_ks,
            "cluster_method": "spherical k-means in normalized CLIP joint space",
            "visual_sample": (
                "Equal-mean pooled CLIP features of Top-K Grounding DINO boxes; "
                "Grounding DINO receives coarse Entity only."
            ),
        },
        "classes": {},
    }

    metric_rows = []
    sample_rows = []

    for class_index, class_name in enumerate(args.classes):
        candidates = groups[class_name]
        grounded_samples = []
        text_features = []
        visual_features = []
        detection_counts = []

        for sample in candidates:
            record = teacher_cache[cache_key(class_name, sample["image_id"])]
            detection_counts.append(record["num_boxes"])

            if record["visual_feature"] is None:
                continue

            text_feature = phrase_features[sample["phrase"].lower()].float()
            visual_feature = record["visual_feature"].float()

            grounded_sample = dict(sample)
            grounded_sample["teacher_query"] = record["query"]
            grounded_sample["num_boxes"] = record["num_boxes"]
            grounded_sample["top1_gdino_score"] = record["scores"][0]
            grounded_sample["top1_box_area"] = record["area_ratios"][0]
            grounded_sample["top1_box"] = record["boxes"][0]
            grounded_sample["paired_cosine"] = float(
                (text_feature @ visual_feature).item()
            )

            grounded_samples.append(grounded_sample)
            text_features.append(text_feature)
            visual_features.append(visual_feature)
            sample_rows.append({"class": class_name, **grounded_sample})

        coverage = len(grounded_samples) / max(len(candidates), 1)
        mean_boxes = float(np.mean(detection_counts)) if detection_counts else 0.0

        print(
            f"\n[{class_index + 1}/{len(args.classes)}] {class_name}: "
            f"grounded={len(grounded_samples)}/{len(candidates)} "
            f"({coverage:.1%}), avg_boxes={mean_boxes:.2f}"
        )

        if len(grounded_samples) < 2:
            print("  Skip clustering: insufficient grounded samples.")
            summary["classes"][class_name] = {
                "candidate_samples": len(candidates),
                "grounded_samples": len(grounded_samples),
                "grounding_coverage": coverage,
                "avg_detection_count": mean_boxes,
                "k_results": {},
            }
            continue

        text_features = F.normalize(torch.stack(text_features), dim=-1)
        visual_features = F.normalize(torch.stack(visual_features), dim=-1)

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        paired_cosine = float(
            (text_features * visual_features).sum(dim=-1).mean().item()
        )

        class_summary = {
            "candidate_samples": len(candidates),
            "grounded_samples": len(grounded_samples),
            "grounding_coverage": coverage,
            "avg_detection_count": mean_boxes,
            "attribute_changed_rate": float(
                np.mean([sample["attribute_changed"] for sample in grounded_samples])
            ),
            "paired_text_visual_cosine": paired_cosine,
            "k_results": {},
        }

        valid_ks = [k for k in args.cluster_ks if 1 <= k <= len(grounded_samples)]

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
            marginal_chance = marginal_chance_consistency(
                text_labels, visual_labels, permutation, k
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
                "uniform_chance_consistency": 1.0 / k,
                "marginal_chance_consistency": marginal_chance,
                "text_cluster_sizes": text_counts,
                "visual_cluster_sizes": visual_counts,
                "best_text_to_visual_permutation": permutation,
                "prototype_similarity_matrix": proto_matrix.tolist(),
            }
            class_summary["k_results"][str(k)] = result

            metric_rows.append({
                "class": class_name,
                "k": k,
                "grounding_coverage": coverage,
                "avg_detection_count": mean_boxes,
                "paired_text_visual_cosine": paired_cosine,
                "text_silhouette": text_sil,
                "visual_silhouette": visual_sil,
                "text_compactness": text_compact,
                "visual_compactness": visual_compact,
                "prototype_alignment_cosine": proto_cos,
                "paired_cluster_consistency": consistency,
                "uniform_chance_consistency": 1.0 / k,
                "marginal_chance_consistency": marginal_chance,
                "text_cluster_sizes": text_counts,
                "visual_cluster_sizes": visual_counts,
            })

            save_diagnostic_figure(
                class_dir / f"k{k}_diagnostic.png",
                text_features, visual_features, text_labels, visual_labels,
                text_centers, visual_centers, proto_matrix, permutation,
                (
                    f"{class_name} | GDINO teacher | K={k} | "
                    f"T-sil={text_sil if text_sil is not None else float('nan'):.3f} | "
                    f"V-sil={visual_sil if visual_sil is not None else float('nan'):.3f} | "
                    f"ProtoCos={proto_cos:.3f} | PairCons={consistency:.3f}"
                ),
            )

            save_cluster_texts(
                class_dir / f"k{k}_text_clusters.json",
                grounded_samples, text_labels, k,
            )

            if k == args.montage_k:
                save_visual_montage(
                    class_dir / f"k{k}_visual_montage.png",
                    train_dataset, clip_model.backbone.preprocess_val,
                    grounded_samples, visual_labels, teacher_cache,
                    class_name, k, args.montage_per_cluster,
                )

            print(
                f"  K={k:<2} | "
                f"T-sil={text_sil if text_sil is not None else float('nan'):+.3f} | "
                f"V-sil={visual_sil if visual_sil is not None else float('nan'):+.3f} | "
                f"T-comp={text_compact:.3f} | V-comp={visual_compact:.3f} | "
                f"ProtoCos={proto_cos:.3f} | PairCons={consistency:.3f} | "
                f"MargChance={marginal_chance:.3f}"
            )

        summary["classes"][class_name] = class_summary

    summary_path = output_dir / "prototype_feasibility_gdino_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_path = output_dir / "prototype_feasibility_gdino_metrics.csv"
    if metric_rows:
        with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    samples_path = output_dir / "prototype_samples_gdino.csv"
    if sample_rows:
        write_samples_csv(samples_path, sample_rows)

    print("\n" + "=" * 112)
    print("HOW TO READ THIS DIAGNOSTIC")
    print("=" * 112)
    print("1. 这是 Teacher/Oracle 上限诊断，不代表最终推理会使用 Grounding DINO。")
    print("2. 先比较旧 Rectangle diagnostic 与本脚本的 Visual silhouette 和 montage。")
    print("3. 若 Aircraft 等类别明显减少 scale/context 簇，说明主要瓶颈是 visual grounding quality。")
    print("4. Prototype cosine matrix 若从近似平坦变得更有选择性，才值得继续 Prototype/OT。")
    print("5. PairCons 要同时看 MargChance；不要再只用 1/K 作为真实随机基线。")
    print("6. Grounding coverage 必须报告；低 coverage 类不能只看成功样本的漂亮指标。")
    print("-" * 112)
    print(f"Summary : {summary_path}")
    print(f"Metrics : {metrics_path}")
    print(f"Samples : {samples_path}")
    print(f"Cache   : {cache_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
