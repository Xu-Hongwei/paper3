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
        description="C0.2: Independent Entity vs Contextual Span Entity."
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
        default="outputs/c02_entity_text_ablation",
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


def load_entity_index(dataset):
    if not dataset.entity_index_file:
        raise ValueError("C0.2 requires Entity Index v2.")

    index = torch.load(
        dataset.entity_index_file,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "pair_to_semantic",
        "semantic_offsets",
        "span_start",
        "span_end",
        "span_entity_ids",
        "entity_vocab",
    }
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"Entity Index v2 missing keys: {missing}")

    return index


def get_valid_entity_records(dataset, entity_index, dataset_index):
    """严格按 span_entity_ids 对齐有效 span 与 Entity 文本。"""
    pair_index = dataset.ann[dataset_index]["_pair_index"]
    semantic_index = int(
        entity_index["pair_to_semantic"][pair_index].item()
    )

    begin = int(entity_index["semantic_offsets"][semantic_index].item())
    end = int(entity_index["semantic_offsets"][semantic_index + 1].item())

    records = []
    for pos in range(begin, end):
        entity_id = int(entity_index["span_entity_ids"][pos].item())
        records.append(
            {
                "text": entity_index["entity_vocab"][entity_id],
                "span": [
                    int(entity_index["span_start"][pos].item()),
                    int(entity_index["span_end"][pos].item()),
                ],
            }
        )
    return records


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
                        "box": [
                            int(x1),
                            int(y1),
                            int(x1 + window),
                            int(y1 + window),
                        ],
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
        features.append(
            model.backbone.encode_image(
                batch,
                normalize=True,
            ).cpu()
        )
    return torch.cat(features)


@torch.no_grad()
def encode_visual_sample(dataset, index, model, regions, device, batch_size):
    image, caption, image_id, _ = dataset[index]
    crops = build_region_crops(image, regions)

    return {
        "dataset_index": int(index),
        "image_id": int(image_id),
        "image": dataset.ann[index]["image"],
        "caption": caption,
        "region_features": encode_regions(
            model,
            crops,
            device,
            batch_size,
        ),
    }


def get_visual(cache, dataset, index, model, regions, device, batch_size):
    key = dataset.ann[index]["image"]
    if key not in cache:
        cache[key] = encode_visual_sample(
            dataset,
            index,
            model,
            regions,
            device,
            batch_size,
        )
    return cache[key]


def select_unique_samples(dataset, entity_index, num_samples, indices, seed):
    def valid(index):
        return len(get_valid_entity_records(dataset, entity_index, index)) >= 2

    if indices:
        selected, used = [], set()
        for index in indices:
            if not 0 <= index < len(dataset):
                raise IndexError(index)

            image = dataset.ann[index]["image"]
            if image in used or not valid(index):
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
        if image in used or not valid(index):
            continue

        selected.append(index)
        used.add(image)

        if len(selected) >= num_samples:
            break

    return selected


def choose_negatives(selected, current_pos, num_negatives, seed):
    candidates = [
        index
        for pos, index in enumerate(selected)
        if pos != current_pos
    ]
    rng = random.Random(seed + current_pos)
    return rng.sample(
        candidates,
        min(num_negatives, len(candidates)),
    )


def pool_contextual_spans(token_features, records):
    """
    token_features: [1, L, D]，已经投影到 CLIP joint space。
    先对 span 内未归一化 token 做 mean pooling，再整体 L2 normalize。
    """
    pooled = []

    for record in records:
        start, end = record["span"]
        if start < 0 or end > token_features.shape[1] or end <= start:
            raise ValueError(f"Invalid Entity span: {[start, end]}")

        pooled.append(
            token_features[0, start:end].mean(dim=0)
        )

    return F.normalize(
        torch.stack(pooled, dim=0),
        dim=-1,
    )


@torch.no_grad()
def encode_entity_features(model, caption, records, device):
    entity_texts = [record["text"] for record in records]

    independent = model.backbone.encode_text(
        entity_texts,
        normalize=True,
    )

    _, token_features = model.backbone.encode_text_with_tokens(
        [caption],
        normalize=False,
    )
    contextual = pool_contextual_spans(
        token_features,
        records,
    )

    return independent.cpu(), contextual.cpu()


def pearson(x, y):
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

    x = torch.tensor(values, dtype=torch.float32)

    return {
        "count": len(values),
        "mean": float(x.mean().item()),
        "median": float(statistics.median(values)),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def positive_rate(values):
    return (
        sum(value > 0 for value in values) / len(values)
        if values
        else None
    )


def metric_summary(values):
    return {
        **summarize(values),
        "positive_rate": positive_rate(values),
    }


@torch.no_grad()
def analyze_sample(
    dataset,
    entity_index,
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
    records = get_valid_entity_records(
        dataset,
        entity_index,
        index,
    )
    matched = get_visual(
        cache,
        dataset,
        index,
        model,
        regions,
        device,
        region_batch_size,
    )

    independent, contextual = encode_entity_features(
        model,
        matched["caption"],
        records,
        device,
    )

    ind_scores = independent @ matched["region_features"].t()
    ctx_scores = contextual @ matched["region_features"].t()

    neg_indices = choose_negatives(
        selected,
        current_pos,
        num_negatives,
        seed,
    )

    neg_ind_max = []
    neg_ctx_max = []

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
        neg_ind_max.append(
            (independent @ negative["region_features"].t())
            .max(dim=1)
            .values
        )
        neg_ctx_max.append(
            (contextual @ negative["region_features"].t())
            .max(dim=1)
            .values
        )

    neg_ind_max = torch.stack(neg_ind_max, dim=1)
    neg_ctx_max = torch.stack(neg_ctx_max, dim=1)

    entity_results = []

    for entity_pos, record in enumerate(records):
        ind = ind_scores[entity_pos]
        ctx = ctx_scores[entity_pos]

        ind_max, ind_argmax = ind.max(dim=0)
        ctx_max, ctx_argmax = ctx.max(dim=0)

        ind_mean_neg = neg_ind_max[entity_pos].mean()
        ctx_mean_neg = neg_ctx_max[entity_pos].mean()
        ind_hard_neg = neg_ind_max[entity_pos].max()
        ctx_hard_neg = neg_ctx_max[entity_pos].max()

        ind_gap_mean = ind_max - ind_mean_neg
        ctx_gap_mean = ctx_max - ctx_mean_neg
        ind_gap_hard = ind_max - ind_hard_neg
        ctx_gap_hard = ctx_max - ctx_hard_neg

        ind_best = regions[int(ind_argmax.item())]
        ctx_best = regions[int(ctx_argmax.item())]

        entity_results.append(
            {
                "text": record["text"],
                "span": record["span"],
                "independent_matched_max": float(ind_max.item()),
                "contextual_matched_max": float(ctx_max.item()),
                "independent_gap_mean_negative": float(ind_gap_mean.item()),
                "contextual_gap_mean_negative": float(ctx_gap_mean.item()),
                "independent_gap_hard_negative": float(ind_gap_hard.item()),
                "contextual_gap_hard_negative": float(ctx_gap_hard.item()),
                "context_minus_independent_mean_gap": float(
                    (ctx_gap_mean - ind_gap_mean).item()
                ),
                "context_minus_independent_hard_gap": float(
                    (ctx_gap_hard - ind_gap_hard).item()
                ),
                "text_feature_cosine": float(
                    (independent[entity_pos] @ contextual[entity_pos]).item()
                ),
                "region_score_correlation": pearson(ind, ctx),
                "independent_best_region_index": int(ind_argmax.item()),
                "contextual_best_region_index": int(ctx_argmax.item()),
                "same_best_region": bool(ind_argmax.item() == ctx_argmax.item()),
                "independent_best_scale": int(ind_best["scale"]),
                "contextual_best_scale": int(ctx_best["scale"]),
                "same_best_scale": bool(
                    ind_best["scale"] == ctx_best["scale"]
                ),
                "independent_best_box": ind_best["box"],
                "contextual_best_box": ctx_best["box"],
            }
        )

    entity_pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            entity_pairs.append(
                {
                    "entity_a": records[i]["text"],
                    "entity_b": records[j]["text"],
                    "independent_region_correlation": pearson(
                        ind_scores[i],
                        ind_scores[j],
                    ),
                    "contextual_region_correlation": pearson(
                        ctx_scores[i],
                        ctx_scores[j],
                    ),
                }
            )

    return {
        "dataset_index": int(index),
        "image_id": matched["image_id"],
        "image": matched["image"],
        "caption": matched["caption"],
        "num_entities": len(records),
        "entities": entity_results,
        "entity_pairs": entity_pairs,
    }


def build_aggregate(samples):
    values = {
        "ind_gap_mean": [],
        "ctx_gap_mean": [],
        "ind_gap_hard": [],
        "ctx_gap_hard": [],
        "ctx_minus_ind_mean": [],
        "ctx_minus_ind_hard": [],
        "text_cosine": [],
        "score_corr": [],
        "ind_pair_corr": [],
        "ctx_pair_corr": [],
    }

    same_region = 0
    same_scale = 0
    num_entities = 0

    for sample in samples:
        for entity in sample["entities"]:
            num_entities += 1
            values["ind_gap_mean"].append(
                entity["independent_gap_mean_negative"]
            )
            values["ctx_gap_mean"].append(
                entity["contextual_gap_mean_negative"]
            )
            values["ind_gap_hard"].append(
                entity["independent_gap_hard_negative"]
            )
            values["ctx_gap_hard"].append(
                entity["contextual_gap_hard_negative"]
            )
            values["ctx_minus_ind_mean"].append(
                entity["context_minus_independent_mean_gap"]
            )
            values["ctx_minus_ind_hard"].append(
                entity["context_minus_independent_hard_gap"]
            )
            values["text_cosine"].append(
                entity["text_feature_cosine"]
            )
            values["score_corr"].append(
                entity["region_score_correlation"]
            )
            same_region += int(entity["same_best_region"])
            same_scale += int(entity["same_best_scale"])

        for pair in sample["entity_pairs"]:
            values["ind_pair_corr"].append(
                pair["independent_region_correlation"]
            )
            values["ctx_pair_corr"].append(
                pair["contextual_region_correlation"]
            )

    def corr_summary(items):
        return {
            **summarize(items),
            "mean_absolute": (
                sum(abs(value) for value in items) / len(items)
                if items
                else None
            ),
        }

    return {
        "num_samples": len(samples),
        "num_entities": num_entities,
        "independent_gap_mean_negative": metric_summary(
            values["ind_gap_mean"]
        ),
        "contextual_gap_mean_negative": metric_summary(
            values["ctx_gap_mean"]
        ),
        "independent_gap_hard_negative": metric_summary(
            values["ind_gap_hard"]
        ),
        "contextual_gap_hard_negative": metric_summary(
            values["ctx_gap_hard"]
        ),
        "context_minus_independent_mean_gap": metric_summary(
            values["ctx_minus_ind_mean"]
        ),
        "context_minus_independent_hard_gap": metric_summary(
            values["ctx_minus_ind_hard"]
        ),
        "independent_contextual_text_cosine": summarize(
            values["text_cosine"]
        ),
        "independent_contextual_score_correlation": summarize(
            values["score_corr"]
        ),
        "same_best_region_rate": (
            same_region / num_entities
            if num_entities
            else None
        ),
        "same_best_scale_rate": (
            same_scale / num_entities
            if num_entities
            else None
        ),
        "independent_entity_pair_correlation": corr_summary(
            values["ind_pair_corr"]
        ),
        "contextual_entity_pair_correlation": corr_summary(
            values["ctx_pair_corr"]
        ),
    }


def write_csvs(output_dir, samples):
    entity_path = output_dir / "entity_text_ablation.csv"
    pair_path = output_dir / "entity_pair_correlations.csv"

    entity_fields = [
        "dataset_index",
        "image_id",
        "image",
        "caption",
        "entity",
        "span",
        "independent_matched_max",
        "contextual_matched_max",
        "independent_gap_mean_negative",
        "contextual_gap_mean_negative",
        "independent_gap_hard_negative",
        "contextual_gap_hard_negative",
        "context_minus_independent_mean_gap",
        "context_minus_independent_hard_gap",
        "text_feature_cosine",
        "region_score_correlation",
        "same_best_region",
        "independent_best_region_index",
        "contextual_best_region_index",
        "independent_best_scale",
        "contextual_best_scale",
        "same_best_scale",
        "independent_best_box",
        "contextual_best_box",
    ]

    with entity_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=entity_fields)
        writer.writeheader()

        for sample in samples:
            for entity in sample["entities"]:
                row = {
                    "dataset_index": sample["dataset_index"],
                    "image_id": sample["image_id"],
                    "image": sample["image"],
                    "caption": sample["caption"],
                    "entity": entity["text"],
                    **{
                        key: value
                        for key, value in entity.items()
                        if key != "text"
                    },
                }
                writer.writerow(row)

    pair_fields = [
        "dataset_index",
        "image_id",
        "entity_a",
        "entity_b",
        "independent_region_correlation",
        "contextual_region_correlation",
    ]

    with pair_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
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
        f"{name:<35}: "
        f"mean={metric['mean']:+.4f}, "
        f"median={metric['median']:+.4f}, "
        f"positive={metric['positive_rate']:.2%}"
    )


def print_aggregate(aggregate):
    print("\n" + "=" * 104)
    print("C0.2 INDEPENDENT ENTITY vs CONTEXTUAL SPAN ENTITY")
    print("=" * 104)
    print(f"Samples  : {aggregate['num_samples']}")
    print(f"Entities : {aggregate['num_entities']}")
    print()

    print_metric(
        "Independent - mean mismatch",
        aggregate["independent_gap_mean_negative"],
    )
    print_metric(
        "Contextual  - mean mismatch",
        aggregate["contextual_gap_mean_negative"],
    )
    print_metric(
        "Independent - hard mismatch",
        aggregate["independent_gap_hard_negative"],
    )
    print_metric(
        "Contextual  - hard mismatch",
        aggregate["contextual_gap_hard_negative"],
    )
    print()

    print_metric(
        "Context minus Independent mean",
        aggregate["context_minus_independent_mean_gap"],
    )
    print_metric(
        "Context minus Independent hard",
        aggregate["context_minus_independent_hard_gap"],
    )
    print()

    text = aggregate["independent_contextual_text_cosine"]
    score = aggregate["independent_contextual_score_correlation"]

    print(
        "Independent↔Context text cosine : "
        f"mean={text['mean']:.4f}, median={text['median']:.4f}"
    )
    print(
        "Independent↔Context score corr  : "
        f"mean={score['mean']:.4f}, median={score['median']:.4f}"
    )
    print(
        "Same Top-1 Region rate          : "
        f"{aggregate['same_best_region_rate']:.2%}"
    )
    print(
        "Same Top-1 Scale rate           : "
        f"{aggregate['same_best_scale_rate']:.2%}"
    )

    ind = aggregate["independent_entity_pair_correlation"]
    ctx = aggregate["contextual_entity_pair_correlation"]

    print()
    print(
        "Independent entity-pair corr    : "
        f"mean={ind['mean']:+.4f}, "
        f"median={ind['median']:+.4f}, "
        f"|mean|={ind['mean_absolute']:.4f}"
    )
    print(
        "Contextual entity-pair corr     : "
        f"mean={ctx['mean']:+.4f}, "
        f"median={ctx['median']:+.4f}, "
        f"|mean|={ctx['mean_absolute']:.4f}"
    )
    print("=" * 104)


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

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )
    entity_index = load_entity_index(train_dataset)

    image_size = int(
        config["dataset"].get(
            "image_res",
            224,
        )
    )
    regions = generate_regions(
        image_size,
        args.windows,
    )

    selected = select_unique_samples(
        train_dataset,
        entity_index,
        args.num_samples,
        args.indices,
        args.seed,
    )

    if len(selected) < 2:
        raise RuntimeError("Need at least two valid different images.")

    print("=" * 104)
    print("C0.2 INDEPENDENT ENTITY vs CONTEXTUAL SPAN ENTITY")
    print("=" * 104)
    print(f"Config            : {args.config}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Checkpoint epoch  : {checkpoint.get('epoch', 'unknown')}")
    print(f"Device            : {device}")
    print(f"Selected images   : {len(selected)}")
    print(f"Regions/image     : {len(regions)}")
    print(f"Negatives/entity  : {args.num_negatives}")
    print(
        "Contextual mode   : final CLIP text-token span mean pooling "
        "(causal left-context)"
    )

    cache = {}
    samples = []

    for pos, index in enumerate(selected):
        result = analyze_sample(
            train_dataset,
            entity_index,
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

        print(
            f"[{pos + 1:>3}/{len(selected)}] "
            f"index={index:<6} "
            f"entities={result['num_entities']:<2} "
            f"cache={len(cache):<3}"
        )

    aggregate = build_aggregate(samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_dir / "c02_entity_text_ablation_summary.json"
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
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
                    "visual_representation": "full_CLIP_multiscale_region_crop",
                    "independent_entity": "entity_phrase_encoded_independently",
                    "contextual_entity": (
                        "mean_pool_final_projected_CLIP_text_tokens_over_valid_span"
                    ),
                    "context_note": (
                        "CLIP text attention is causal, so span tokens encode "
                        "left/prefix context rather than bidirectional full-sentence context."
                    ),
                },
                "aggregate": aggregate,
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    entity_csv, pair_csv = write_csvs(
        output_dir,
        samples,
    )

    print_aggregate(aggregate)

    print("\nSaved:")
    print(f"  JSON       : {summary_path}")
    print(f"  Entity CSV : {entity_csv}")
    print(f"  Pair CSV   : {pair_csv}")


if __name__ == "__main__":
    main()
