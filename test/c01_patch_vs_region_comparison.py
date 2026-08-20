import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.1-P: Raw Patch vs Region Crop quantitative comparison."
    )
    parser.add_argument("--config", type=str, default="configs/baseline/rsicd.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/clip_rsicd_10ep/best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/c01_patch_vs_region",
    )
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


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
                regions.append(
                    {
                        "scale": int(window),
                        "box": [x1, y1, x1 + window, y1 + window],
                    }
                )
    return regions


def build_region_crops(image, regions):
    size = image.shape[-1]
    crops = []
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        crops.append(crop.squeeze(0))
    return torch.stack(crops)


@torch.no_grad()
def encode_regions(model, crops, device, batch_size):
    features = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size].to(device, non_blocking=True)
        features.append(model.backbone.encode_image(batch, normalize=True).cpu())
    return torch.cat(features)


@torch.no_grad()
def encode_visual_sample(dataset, index, model, regions, device, region_batch_size):
    image, caption, image_id, _ = dataset[index]
    image_gpu = image.unsqueeze(0).to(device)

    # Raw Patch：最终层 patch token 投影到 CLIP joint space 后 L2 normalize。
    global_feature, patch_features = model.backbone.encode_image_with_patches(
        image_gpu,
        normalize=True,
    )

    crops = build_region_crops(image, regions)
    region_features = encode_regions(
        model,
        crops,
        device,
        region_batch_size,
    )

    return {
        "dataset_index": int(index),
        "image_id": int(image_id),
        "image": dataset.ann[index]["image"],
        "caption": caption,
        "global_feature": global_feature.squeeze(0).cpu(),
        "patch_features": patch_features.squeeze(0).cpu(),
        "region_features": region_features,
    }


def get_visual(cache, dataset, index, model, regions, device, region_batch_size):
    key = dataset.ann[index]["image"]
    if key not in cache:
        cache[key] = encode_visual_sample(
            dataset,
            index,
            model,
            regions,
            device,
            region_batch_size,
        )
    return cache[key]


def select_unique_samples(dataset, num_samples, indices, seed):
    if indices:
        selected, used = [], set()
        for index in indices:
            if not 0 <= index < len(dataset):
                raise IndexError(index)
            image = dataset.ann[index]["image"]
            if image in used:
                continue
            if len(dataset.get_entity_texts(index)) < 2:
                continue
            selected.append(index)
            used.add(image)
        return selected

    rng = random.Random(seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    selected, used = [], set()
    for index in candidates:
        image = dataset.ann[index]["image"]
        if image in used or len(dataset.get_entity_texts(index)) < 2:
            continue
        selected.append(index)
        used.add(image)
        if len(selected) >= num_samples:
            break
    return selected


def choose_negatives(selected, current_pos, num_negatives, seed):
    candidates = [index for pos, index in enumerate(selected) if pos != current_pos]
    rng = random.Random(seed + current_pos)
    return rng.sample(candidates, min(num_negatives, len(candidates)))


def pearson(x, y):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if denom.item() > 1e-12 else 0.0


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(x.mean()),
        "median": float(statistics.median(values)),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def positive_rate(values):
    return sum(v > 0 for v in values) / len(values) if values else None


def metric_summary(values):
    return {**summarize(values), "positive_rate": positive_rate(values)}


@torch.no_grad()
def analyze_sample(
    dataset,
    index,
    current_pos,
    selected,
    model,
    regions,
    cache,
    device,
    region_batch_size,
    num_negatives,
    seed,
):
    matched = get_visual(
        cache,
        dataset,
        index,
        model,
        regions,
        device,
        region_batch_size,
    )

    entities = dataset.get_entity_texts(index)
    text = model.backbone.encode_text(entities, normalize=True).cpu()

    region_scores = text @ matched["region_features"].t()
    patch_scores = text @ matched["patch_features"].t()

    neg_indices = choose_negatives(
        selected,
        current_pos,
        num_negatives,
        seed,
    )
    neg_region_max, neg_patch_max = [], []

    for neg_index in neg_indices:
        negative = get_visual(
            cache,
            dataset,
            neg_index,
            model,
            regions,
            device,
            region_batch_size,
        )
        neg_region_max.append(
            (text @ negative["region_features"].t()).max(dim=1).values
        )
        neg_patch_max.append(
            (text @ negative["patch_features"].t()).max(dim=1).values
        )

    neg_region_max = torch.stack(neg_region_max, dim=1)
    neg_patch_max = torch.stack(neg_patch_max, dim=1)

    entity_records = []
    for entity_pos, entity in enumerate(entities):
        r = region_scores[entity_pos]
        p = patch_scores[entity_pos]

        r_match = r.max()
        p_match = p.max()
        r_mean_neg = neg_region_max[entity_pos].mean()
        p_mean_neg = neg_patch_max[entity_pos].mean()
        r_hard_neg = neg_region_max[entity_pos].max()
        p_hard_neg = neg_patch_max[entity_pos].max()

        r_gap_mean = r_match - r_mean_neg
        p_gap_mean = p_match - p_mean_neg
        r_gap_hard = r_match - r_hard_neg
        p_gap_hard = p_match - p_hard_neg

        entity_records.append(
            {
                "text": entity,
                "region_matched_max": float(r_match),
                "patch_matched_max": float(p_match),
                "region_gap_mean_negative": float(r_gap_mean),
                "patch_gap_mean_negative": float(p_gap_mean),
                "region_gap_hard_negative": float(r_gap_hard),
                "patch_gap_hard_negative": float(p_gap_hard),
                "region_minus_patch_mean_gap": float(r_gap_mean - p_gap_mean),
                "region_minus_patch_hard_gap": float(r_gap_hard - p_gap_hard),
            }
        )

    pair_records = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            pair_records.append(
                {
                    "entity_a": entities[i],
                    "entity_b": entities[j],
                    "region_correlation": pearson(region_scores[i], region_scores[j]),
                    "patch_correlation": pearson(patch_scores[i], patch_scores[j]),
                }
            )

    return {
        "dataset_index": int(index),
        "image_id": matched["image_id"],
        "image": matched["image"],
        "caption": matched["caption"],
        "num_entities": len(entities),
        "entities": entity_records,
        "entity_pairs": pair_records,
    }


def build_aggregate(samples):
    values = {
        "region_gap_mean": [],
        "patch_gap_mean": [],
        "region_gap_hard": [],
        "patch_gap_hard": [],
        "region_minus_patch_mean_gap": [],
        "region_minus_patch_hard_gap": [],
        "region_corr": [],
        "patch_corr": [],
    }

    for sample in samples:
        for entity in sample["entities"]:
            values["region_gap_mean"].append(entity["region_gap_mean_negative"])
            values["patch_gap_mean"].append(entity["patch_gap_mean_negative"])
            values["region_gap_hard"].append(entity["region_gap_hard_negative"])
            values["patch_gap_hard"].append(entity["patch_gap_hard_negative"])
            values["region_minus_patch_mean_gap"].append(
                entity["region_minus_patch_mean_gap"]
            )
            values["region_minus_patch_hard_gap"].append(
                entity["region_minus_patch_hard_gap"]
            )

        for pair in sample["entity_pairs"]:
            values["region_corr"].append(pair["region_correlation"])
            values["patch_corr"].append(pair["patch_correlation"])

    return {
        "num_samples": len(samples),
        "num_entities": sum(s["num_entities"] for s in samples),
        "region_gap_mean_negative": metric_summary(values["region_gap_mean"]),
        "patch_gap_mean_negative": metric_summary(values["patch_gap_mean"]),
        "region_gap_hard_negative": metric_summary(values["region_gap_hard"]),
        "patch_gap_hard_negative": metric_summary(values["patch_gap_hard"]),
        "region_minus_patch_mean_gap": metric_summary(
            values["region_minus_patch_mean_gap"]
        ),
        "region_minus_patch_hard_gap": metric_summary(
            values["region_minus_patch_hard_gap"]
        ),
        "region_entity_correlation": {
            **summarize(values["region_corr"]),
            "mean_absolute": (
                sum(abs(v) for v in values["region_corr"]) / len(values["region_corr"])
                if values["region_corr"]
                else None
            ),
        },
        "patch_entity_correlation": {
            **summarize(values["patch_corr"]),
            "mean_absolute": (
                sum(abs(v) for v in values["patch_corr"]) / len(values["patch_corr"])
                if values["patch_corr"]
                else None
            ),
        },
    }


def write_csv(output_dir, samples):
    entity_path = output_dir / "patch_vs_region_entities.csv"
    pair_path = output_dir / "patch_vs_region_correlations.csv"

    with entity_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "dataset_index",
            "image_id",
            "image",
            "caption",
            "entity",
            "region_matched_max",
            "patch_matched_max",
            "region_gap_mean_negative",
            "patch_gap_mean_negative",
            "region_gap_hard_negative",
            "patch_gap_hard_negative",
            "region_minus_patch_mean_gap",
            "region_minus_patch_hard_gap",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            for entity in sample["entities"]:
                writer.writerow(
                    {
                        "dataset_index": sample["dataset_index"],
                        "image_id": sample["image_id"],
                        "image": sample["image"],
                        "caption": sample["caption"],
                        "entity": entity["text"],
                        **{k: v for k, v in entity.items() if k != "text"},
                    }
                )

    with pair_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "dataset_index",
            "image_id",
            "entity_a",
            "entity_b",
            "region_correlation",
            "patch_correlation",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            for pair in sample["entity_pairs"]:
                writer.writerow(
                    {
                        "dataset_index": sample["dataset_index"],
                        "image_id": sample["image_id"],
                        **pair,
                    }
                )

    return entity_path, pair_path


def print_metric(name, metric):
    print(
        f"{name:<30}: "
        f"mean={metric['mean']:+.4f}, "
        f"median={metric['median']:+.4f}, "
        f"positive={metric['positive_rate']:.2%}"
    )


def print_aggregate(aggregate):
    print("\n" + "=" * 96)
    print("C0.1-P RAW PATCH vs REGION CROP")
    print("=" * 96)
    print(f"Samples  : {aggregate['num_samples']}")
    print(f"Entities : {aggregate['num_entities']}")
    print()

    print_metric(
        "Region - mean mismatch",
        aggregate["region_gap_mean_negative"],
    )
    print_metric(
        "Patch  - mean mismatch",
        aggregate["patch_gap_mean_negative"],
    )
    print_metric(
        "Region - hard mismatch",
        aggregate["region_gap_hard_negative"],
    )
    print_metric(
        "Patch  - hard mismatch",
        aggregate["patch_gap_hard_negative"],
    )
    print()

    print_metric(
        "Region minus Patch mean gap",
        aggregate["region_minus_patch_mean_gap"],
    )
    print_metric(
        "Region minus Patch hard gap",
        aggregate["region_minus_patch_hard_gap"],
    )
    print()

    r = aggregate["region_entity_correlation"]
    p = aggregate["patch_entity_correlation"]
    print(
        "Region entity correlation     : "
        f"mean={r['mean']:+.4f}, median={r['median']:+.4f}, "
        f"|mean|={r['mean_absolute']:.4f}"
    )
    print(
        "Patch entity correlation      : "
        f"mean={p['mean']:+.4f}, median={p['median']:+.4f}, "
        f"|mean|={p['mean_absolute']:.4f}"
    )
    print("=" * 96)


def main():
    args = parse_args()
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
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    image_size = int(config["dataset"].get("image_res", 224))
    regions = generate_regions(image_size, args.windows)
    selected = select_unique_samples(
        train_dataset,
        args.num_samples,
        args.indices,
        args.seed,
    )

    if len(selected) < 2:
        raise RuntimeError("Need at least two different images.")

    print("=" * 96)
    print("C0.1-P RAW PATCH vs REGION CROP")
    print("=" * 96)
    print(f"Config            : {args.config}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Checkpoint epoch  : {checkpoint.get('epoch', 'unknown')}")
    print(f"Device            : {device}")
    print(f"Selected images   : {len(selected)}")
    print(f"Regions/image     : {len(regions)}")
    print(f"Negatives/entity  : {args.num_negatives}")

    cache = {}
    samples = []

    for pos, index in enumerate(selected):
        result = analyze_sample(
            train_dataset,
            index,
            pos,
            selected,
            model,
            regions,
            cache,
            device,
            args.region_batch_size,
            args.num_negatives,
            args.seed,
        )
        samples.append(result)

        if pos == 0:
            visual = next(iter(cache.values()))
            print(
                f"Patch features     : {tuple(visual['patch_features'].shape)} "
                f"(expected ViT-B/32: 49×512)"
            )

        print(
            f"[{pos + 1:>3}/{len(selected)}] "
            f"index={index:<6} entities={result['num_entities']:<2} "
            f"cache={len(cache):<3}"
        )

    aggregate = build_aggregate(samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c01_patch_vs_region_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "seed": args.seed,
                    "windows": args.windows,
                    "num_regions": len(regions),
                    "num_negatives": args.num_negatives,
                    "region_representation": "full_CLIP_crop_global_feature",
                    "patch_representation": "final_projected_CLIP_patch_tokens",
                    "entity_representation": "independent_entity_phrase",
                },
                "aggregate": aggregate,
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    entity_csv, pair_csv = write_csv(output_dir, samples)

    print_aggregate(aggregate)
    print("\nSaved:")
    print(f"  JSON       : {summary_path}")
    print(f"  Entity CSV : {entity_csv}")
    print(f"  Pair CSV   : {pair_csv}")


if __name__ == "__main__":
    main()
