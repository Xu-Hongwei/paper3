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
        description="C0.1: Region-Entity quantitative diagnostic."
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
        default="outputs/c01_region_entity_analysis",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="指定训练集样本 index；默认随机选取不同图像且 Entity 数>=2 的样本。",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def sliding_positions(image_size, window, stride):
    if window <= 0 or window > image_size:
        raise ValueError(f"Invalid window={window} for image_size={image_size}")

    positions = list(range(0, image_size - window + 1, stride))
    last = image_size - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_region_boxes(image_size, windows):
    regions = []

    for window in windows:
        stride = max(window // 2, 1)
        xs = sliding_positions(image_size, window, stride)
        ys = sliding_positions(image_size, window, stride)

        for y1 in ys:
            for x1 in xs:
                regions.append(
                    {
                        "scale": int(window),
                        "stride": int(stride),
                        "box": [
                            int(x1),
                            int(y1),
                            int(x1 + window),
                            int(y1 + window),
                        ],
                    }
                )

    return regions


def build_scale_indices(regions):
    scale_indices = {}
    for index, region in enumerate(regions):
        scale_indices.setdefault(region["scale"], []).append(index)
    return scale_indices


def build_region_crops(image, regions, output_size):
    crops = []

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        crops.append(crop.squeeze(0))

    return torch.stack(crops, dim=0)


@torch.no_grad()
def encode_regions(model, crops, device, batch_size):
    features = []

    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size].to(
            device,
            non_blocking=True,
        )
        features.append(
            model.backbone.encode_image(
                batch,
                normalize=True,
            ).cpu()
        )

    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_visual_sample(
    dataset,
    index,
    model,
    regions,
    device,
    region_batch_size,
):
    image, caption, image_id, _ = dataset[index]
    image_size = image.shape[-1]

    global_feature = model.backbone.encode_image(
        image.unsqueeze(0).to(device),
        normalize=True,
    ).squeeze(0).cpu()

    crops = build_region_crops(
        image=image,
        regions=regions,
        output_size=image_size,
    )
    region_features = encode_regions(
        model=model,
        crops=crops,
        device=device,
        batch_size=region_batch_size,
    )

    return {
        "dataset_index": int(index),
        "image_id": int(image_id),
        "image": dataset.ann[index]["image"],
        "caption": caption,
        "global_feature": global_feature,
        "region_features": region_features,
    }


def get_visual_sample(
    cache,
    dataset,
    index,
    model,
    regions,
    device,
    region_batch_size,
):
    key = dataset.ann[index]["image"]

    if key not in cache:
        cache[key] = encode_visual_sample(
            dataset=dataset,
            index=index,
            model=model,
            regions=regions,
            device=device,
            region_batch_size=region_batch_size,
        )

    return cache[key]


def select_unique_samples(dataset, num_samples, indices, seed):
    if indices:
        selected = []
        used_images = set()

        for index in indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(f"Invalid dataset index: {index}")

            image_key = dataset.ann[index]["image"]
            entities = dataset.get_entity_texts(index)

            if len(entities) < 2:
                raise ValueError(
                    f"Dataset index {index} has fewer than 2 Entity texts."
                )
            if image_key in used_images:
                continue

            selected.append(index)
            used_images.add(image_key)

        return selected

    rng = random.Random(seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    selected = []
    used_images = set()

    for index in candidates:
        image_key = dataset.ann[index]["image"]
        if image_key in used_images:
            continue

        entities = dataset.get_entity_texts(index)
        if len(entities) < 2:
            continue

        selected.append(index)
        used_images.add(image_key)

        if len(selected) >= num_samples:
            break

    return selected


def pearson_correlation(x, y):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()

    denom = x.norm() * y.norm()
    if denom.item() <= 1e-12:
        return 0.0

    return float((x @ y / denom).item())


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    tensor = torch.tensor(values, dtype=torch.float32)

    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(statistics.median(values)),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def positive_rate(values):
    if not values:
        return None
    return sum(value > 0 for value in values) / len(values)


def choose_negative_indices(selected, current_pos, num_negatives, seed):
    candidates = [
        index
        for pos, index in enumerate(selected)
        if pos != current_pos
    ]

    if not candidates:
        raise RuntimeError("C0.1 needs at least two different images.")

    rng = random.Random(seed + current_pos)
    count = min(num_negatives, len(candidates))
    return rng.sample(candidates, count)


@torch.no_grad()
def analyze_sample(
    dataset,
    index,
    current_pos,
    selected,
    model,
    regions,
    scale_indices,
    visual_cache,
    device,
    region_batch_size,
    num_negatives,
    seed,
):
    matched = get_visual_sample(
        cache=visual_cache,
        dataset=dataset,
        index=index,
        model=model,
        regions=regions,
        device=device,
        region_batch_size=region_batch_size,
    )

    entities = dataset.get_entity_texts(index)
    entity_features = model.backbone.encode_text(
        entities,
        normalize=True,
    ).cpu()

    region_scores = entity_features @ matched["region_features"].t()
    global_scores = entity_features @ matched["global_feature"].unsqueeze(1)
    global_scores = global_scores.squeeze(1)

    negative_indices = choose_negative_indices(
        selected=selected,
        current_pos=current_pos,
        num_negatives=num_negatives,
        seed=seed,
    )

    negative_max_scores = []
    negative_records = []

    for negative_index in negative_indices:
        negative = get_visual_sample(
            cache=visual_cache,
            dataset=dataset,
            index=negative_index,
            model=model,
            regions=regions,
            device=device,
            region_batch_size=region_batch_size,
        )

        scores = entity_features @ negative["region_features"].t()
        max_scores = scores.max(dim=1).values
        negative_max_scores.append(max_scores)

        negative_records.append(
            {
                "dataset_index": negative["dataset_index"],
                "image_id": negative["image_id"],
                "image": negative["image"],
                "entity_max_scores": [
                    float(value)
                    for value in max_scores.tolist()
                ],
            }
        )

    negative_max_scores = torch.stack(
        negative_max_scores,
        dim=1,
    )

    entities_result = []

    for entity_pos, entity in enumerate(entities):
        scores = region_scores[entity_pos]
        matched_max, matched_argmax = scores.max(dim=0)

        mismatch_mean = negative_max_scores[entity_pos].mean()
        mismatch_hard = negative_max_scores[entity_pos].max()

        scale_stats = {}
        for scale, indices_for_scale in scale_indices.items():
            scale_scores = scores[indices_for_scale]
            scale_stats[str(scale)] = {
                "count": len(indices_for_scale),
                "max": float(scale_scores.max().item()),
                "mean": float(scale_scores.mean().item()),
                "std": float(
                    scale_scores.std(
                        unbiased=False,
                    ).item()
                ),
            }

        best_region = regions[int(matched_argmax.item())]

        entities_result.append(
            {
                "text": entity,
                "global_similarity": float(
                    global_scores[entity_pos].item()
                ),
                "matched_region_max": float(
                    matched_max.item()
                ),
                "region_gain": float(
                    matched_max.item()
                    - global_scores[entity_pos].item()
                ),
                "best_region_index": int(
                    matched_argmax.item()
                ),
                "best_scale": int(
                    best_region["scale"]
                ),
                "best_box": best_region["box"],
                "mismatch_mean_max": float(
                    mismatch_mean.item()
                ),
                "mismatch_hard_max": float(
                    mismatch_hard.item()
                ),
                "matched_gap_mean_negative": float(
                    matched_max.item()
                    - mismatch_mean.item()
                ),
                "matched_gap_hard_negative": float(
                    matched_max.item()
                    - mismatch_hard.item()
                ),
                "scale_stats": scale_stats,
                "region_scores": [
                    float(value)
                    for value in scores.tolist()
                ],
            }
        )

    correlations = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            correlations.append(
                {
                    "entity_a": entities[i],
                    "entity_b": entities[j],
                    "pearson": pearson_correlation(
                        region_scores[i],
                        region_scores[j],
                    ),
                }
            )

    return {
        "dataset_index": int(index),
        "pair_index": int(
            dataset.ann[index].get(
                "_pair_index",
                index,
            )
        ),
        "image_id": matched["image_id"],
        "image": matched["image"],
        "caption": matched["caption"],
        "entities": entities_result,
        "entity_pair_correlations": correlations,
        "negative_images": negative_records,
    }


def build_aggregate(samples, windows):
    region_gains = []
    matched_gap_mean = []
    matched_gap_hard = []
    correlations = []

    scale_values = {
        int(scale): {
            "max": [],
            "mean": [],
            "std": [],
            "wins": 0,
        }
        for scale in windows
    }

    entity_count = 0

    for sample in samples:
        for entity in sample["entities"]:
            entity_count += 1
            region_gains.append(entity["region_gain"])
            matched_gap_mean.append(
                entity["matched_gap_mean_negative"]
            )
            matched_gap_hard.append(
                entity["matched_gap_hard_negative"]
            )
            scale_values[entity["best_scale"]]["wins"] += 1

            for scale in windows:
                stats = entity["scale_stats"][str(scale)]
                scale_values[int(scale)]["max"].append(
                    stats["max"]
                )
                scale_values[int(scale)]["mean"].append(
                    stats["mean"]
                )
                scale_values[int(scale)]["std"].append(
                    stats["std"]
                )

        correlations.extend(
            pair["pearson"]
            for pair in sample["entity_pair_correlations"]
        )

    scale_summary = {}
    for scale, values in scale_values.items():
        scale_summary[str(scale)] = {
            "candidate_count_per_image": None,
            "top1_wins": values["wins"],
            "top1_win_rate": (
                values["wins"] / entity_count
                if entity_count
                else None
            ),
            "max_score": summarize(values["max"]),
            "mean_score": summarize(values["mean"]),
            "within_scale_std": summarize(values["std"]),
        }

    return {
        "num_samples": len(samples),
        "num_entities": entity_count,
        "region_gain": {
            **summarize(region_gains),
            "positive_rate": positive_rate(region_gains),
        },
        "matched_gap_mean_negative": {
            **summarize(matched_gap_mean),
            "positive_rate": positive_rate(
                matched_gap_mean
            ),
        },
        "matched_gap_hard_negative": {
            **summarize(matched_gap_hard),
            "positive_rate": positive_rate(
                matched_gap_hard
            ),
        },
        "entity_pair_correlation": {
            **summarize(correlations),
            "mean_absolute": (
                sum(abs(value) for value in correlations)
                / len(correlations)
                if correlations
                else None
            ),
        },
        "scale_summary": scale_summary,
    }


def write_csvs(output_dir, samples, windows):
    entity_path = output_dir / "entity_metrics.csv"
    correlation_path = output_dir / "entity_correlations.csv"
    scale_path = output_dir / "scale_metrics.csv"

    with entity_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "image_id",
                "image",
                "caption",
                "entity",
                "global_similarity",
                "matched_region_max",
                "region_gain",
                "best_scale",
                "best_box",
                "mismatch_mean_max",
                "mismatch_hard_max",
                "matched_gap_mean_negative",
                "matched_gap_hard_negative",
            ],
        )
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
                        "global_similarity": entity["global_similarity"],
                        "matched_region_max": entity["matched_region_max"],
                        "region_gain": entity["region_gain"],
                        "best_scale": entity["best_scale"],
                        "best_box": entity["best_box"],
                        "mismatch_mean_max": entity["mismatch_mean_max"],
                        "mismatch_hard_max": entity["mismatch_hard_max"],
                        "matched_gap_mean_negative": entity["matched_gap_mean_negative"],
                        "matched_gap_hard_negative": entity["matched_gap_hard_negative"],
                    }
                )

    with correlation_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "image_id",
                "entity_a",
                "entity_b",
                "pearson",
            ],
        )
        writer.writeheader()

        for sample in samples:
            for pair in sample["entity_pair_correlations"]:
                writer.writerow(
                    {
                        "dataset_index": sample["dataset_index"],
                        "image_id": sample["image_id"],
                        "entity_a": pair["entity_a"],
                        "entity_b": pair["entity_b"],
                        "pearson": pair["pearson"],
                    }
                )

    with scale_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_index",
                "image_id",
                "entity",
                "scale",
                "candidate_count",
                "max",
                "mean",
                "std",
            ],
        )
        writer.writeheader()

        for sample in samples:
            for entity in sample["entities"]:
                for scale in windows:
                    stats = entity["scale_stats"][str(scale)]
                    writer.writerow(
                        {
                            "dataset_index": sample["dataset_index"],
                            "image_id": sample["image_id"],
                            "entity": entity["text"],
                            "scale": scale,
                            "candidate_count": stats["count"],
                            "max": stats["max"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                        }
                    )

    return entity_path, correlation_path, scale_path


def print_aggregate(aggregate, scale_indices):
    print("\n" + "=" * 92)
    print("C0.1 AGGREGATE")
    print("=" * 92)
    print(f"Samples  : {aggregate['num_samples']}")
    print(f"Entities : {aggregate['num_entities']}")

    region_gain = aggregate["region_gain"]
    print(
        "Region gain               : "
        f"mean={region_gain['mean']:+.4f}, "
        f"median={region_gain['median']:+.4f}, "
        f"positive={region_gain['positive_rate']:.2%}"
    )

    gap_mean = aggregate["matched_gap_mean_negative"]
    print(
        "Matched - mean mismatch   : "
        f"mean={gap_mean['mean']:+.4f}, "
        f"median={gap_mean['median']:+.4f}, "
        f"positive={gap_mean['positive_rate']:.2%}"
    )

    gap_hard = aggregate["matched_gap_hard_negative"]
    print(
        "Matched - hard mismatch   : "
        f"mean={gap_hard['mean']:+.4f}, "
        f"median={gap_hard['median']:+.4f}, "
        f"positive={gap_hard['positive_rate']:.2%}"
    )

    corr = aggregate["entity_pair_correlation"]
    print(
        "Entity score correlation  : "
        f"mean={corr['mean']:+.4f}, "
        f"median={corr['median']:+.4f}, "
        f"|mean|={corr['mean_absolute']:.4f}"
    )

    print("\nPer-scale score statistics:")
    for scale, indices in scale_indices.items():
        stats = aggregate["scale_summary"][str(scale)]
        stats["candidate_count_per_image"] = len(indices)
        print(
            f"  {scale:>3}px | "
            f"regions={len(indices):>3} | "
            f"wins={stats['top1_wins']:>4} | "
            f"mean(max)={stats['max_score']['mean']:.4f} | "
            f"mean(mean)={stats['mean_score']['mean']:.4f} | "
            f"mean(std)={stats['within_scale_std']['mean']:.4f}"
        )

    print("=" * 92)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 92)
    print("C0.1 REGION-ENTITY QUANTITATIVE ANALYSIS")
    print("=" * 92)
    print(f"Config            : {args.config}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Device            : {device}")
    print(f"Samples           : {args.num_samples}")
    print(f"Negatives/sample  : {args.num_negatives}")
    print(f"Windows           : {args.windows}")
    print(f"Region batch size : {args.region_batch_size}")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
    )
    model = model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    image_size = int(
        config["dataset"].get(
            "image_res",
            224,
        )
    )
    regions = generate_region_boxes(
        image_size=image_size,
        windows=args.windows,
    )
    scale_indices = build_scale_indices(regions)

    selected = select_unique_samples(
        dataset=train_dataset,
        num_samples=args.num_samples,
        indices=args.indices,
        seed=args.seed,
    )

    if len(selected) < 2:
        raise RuntimeError(
            "C0.1 requires at least two different images."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Checkpoint epoch  : {checkpoint.get('epoch', 'unknown')}")
    print(f"Candidate regions : {len(regions)}")
    print(f"Selected images   : {len(selected)}")

    visual_cache = {}
    samples = []

    for pos, index in enumerate(selected):
        result = analyze_sample(
            dataset=train_dataset,
            index=index,
            current_pos=pos,
            selected=selected,
            model=model,
            regions=regions,
            scale_indices=scale_indices,
            visual_cache=visual_cache,
            device=device,
            region_batch_size=args.region_batch_size,
            num_negatives=args.num_negatives,
            seed=args.seed,
        )
        samples.append(result)

        print(
            f"[{pos + 1:>3}/{len(selected)}] "
            f"index={index:<6} "
            f"entities={len(result['entities']):<2} "
            f"cache={len(visual_cache):<3}"
        )

    aggregate = build_aggregate(
        samples=samples,
        windows=args.windows,
    )

    for scale, indices in scale_indices.items():
        aggregate["scale_summary"][str(scale)][
            "candidate_count_per_image"
        ] = len(indices)

    summary = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "image_size": image_size,
            "windows": args.windows,
            "num_candidate_regions": len(regions),
            "num_negatives": args.num_negatives,
            "seed": args.seed,
            "region_encoding": "full_clean_clip_vision_encoder",
            "entity_encoding": "independent_clean_clip_text_encoder",
            "negative_control": "random_different_image_from_selected_pool",
            "regions": regions,
        },
        "aggregate": aggregate,
        "samples": samples,
    }

    summary_path = output_dir / "c01_summary.json"
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    entity_csv, corr_csv, scale_csv = write_csvs(
        output_dir=output_dir,
        samples=samples,
        windows=args.windows,
    )

    print_aggregate(
        aggregate=aggregate,
        scale_indices=scale_indices,
    )

    print("\nSaved:")
    print(f"  JSON        : {summary_path}")
    print(f"  Entity CSV  : {entity_csv}")
    print(f"  Corr CSV    : {corr_csv}")
    print(f"  Scale CSV   : {scale_csv}")


if __name__ == "__main__":
    main()
