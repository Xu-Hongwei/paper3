import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval


METRIC_KEYS = [
    "global_margin_12",
    "global_margin_15",
    "global_topn_std",
    "local_margin_12",
    "local_topn_std",
    "local_advantage_over_global_top1",
    "entity_region_margin_mean",
    "entity_region_margin_min",
    "entity_candidate_agreement",
    "global_local_spearman",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="C1-R1: Rescued / Corrupted reliability analysis."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--query-csv",
        type=str,
        default="outputs/c1r_train_structured_rerank/c1r_train_query_cases.csv",
    )
    parser.add_argument(
        "--region-cache",
        type=str,
        default=(
            "outputs/c1r_train_structured_rerank/"
            "regions_clip_rsicd_10ep_best_q200_seed42_top50_32-64-96-128.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/c1r_reliability_analysis",
    )
    parser.add_argument("--lambda-value", type=float, default=0.4)
    parser.add_argument("--global-topn", type=int, default=50)
    parser.add_argument("--local-topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_query_records(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phrases = ast.literal_eval(row["phrases"])
            rows.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "pair_index": int(row["pair_index"]),
                    "caption": row["caption"],
                    "gt_image_id": int(row["gt_image_id"]),
                    "image": row["image"],
                    "phrases": phrases,
                    "baseline_rank_saved": int(row["baseline_rank"]),
                    "rerankable_saved": row["rerankable"].strip().lower() == "true",
                }
            )

    if not rows:
        raise ValueError("query CSV 为空。")

    return rows


def build_unique_image_indices(dataset):
    first_index = [None] * dataset.num_images

    for index, image_id in enumerate(dataset.image_ids):
        if first_index[image_id] is None:
            first_index[image_id] = index

    if any(index is None for index in first_index):
        raise RuntimeError("训练 image_id 映射不完整。")

    return first_index


@torch.no_grad()
def extract_global_image_features(
    model,
    dataset,
    image_indices,
    device,
    batch_size,
):
    features = []

    for start in range(0, len(image_indices), batch_size):
        indices = image_indices[start:start + batch_size]
        images = torch.stack(
            [dataset[index][0] for index in indices]
        ).to(device, non_blocking=True)

        features.append(
            model.backbone.encode_image(
                images,
                normalize=True,
            ).cpu()
        )

        done = min(start + batch_size, len(image_indices))
        if start == 0 or done == len(image_indices) or (start // batch_size + 1) % 20 == 0:
            print(f"  Global images: {done}/{len(image_indices)}")

    return torch.cat(features)


@torch.no_grad()
def encode_captions(model, records, device, batch_size):
    captions = [record["caption"] for record in records]
    features = []

    for start in range(0, len(captions), batch_size):
        batch = captions[start:start + batch_size]
        features.append(
            model.backbone.encode_text(
                batch,
                normalize=True,
            ).cpu()
        )

    return torch.cat(features)


@torch.no_grad()
def encode_phrases(model, records, device, batch_size):
    unique = []
    seen = set()

    for record in records:
        for phrase in record["phrases"]:
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(phrase)

    features = []

    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        features.append(
            model.backbone.encode_text(
                batch,
                normalize=True,
            ).cpu()
        )

    features = torch.cat(features)

    return {
        phrase.lower(): features[i]
        for i, phrase in enumerate(unique)
    }


def zscore(x):
    std = x.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / std


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))

    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0

    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
        }

    x = np.asarray(values, dtype=np.float64)

    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def load_region_cache(path):
    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    image_ids = [int(x) for x in cache["image_ids"]]
    features = cache["features"]

    if len(image_ids) != len(features):
        raise ValueError("Region cache image_ids/features 数量不一致。")

    return image_ids, features


@torch.no_grad()
def analyze_query(
    record,
    global_scores,
    phrase_features,
    region_id_to_row,
    region_features,
    device,
    global_topn,
    local_topk,
    temperature,
    lambda_value,
):
    gt_image = record["gt_image_id"]
    global_order = torch.argsort(global_scores, descending=True)
    baseline_rank = int(
        (global_order == gt_image).nonzero(as_tuple=False)[0].item()
    )

    candidate_ids = global_order[:global_topn]

    if baseline_rank >= global_topn:
        return {
            "baseline_rank": baseline_rank + 1,
            "rerankable": False,
            "reranked_rank": baseline_rank + 1,
            "category": "unchanged_outside_topn",
        }

    missing = [
        int(image_id)
        for image_id in candidate_ids.tolist()
        if int(image_id) not in region_id_to_row
    ]
    if missing:
        raise KeyError(
            f"Region cache 缺少当前 query 的候选图像，例如 image_id={missing[0]}"
        )

    rows = [
        region_id_to_row[int(image_id)]
        for image_id in candidate_ids.tolist()
    ]
    candidates = region_features[rows].to(
        device,
        dtype=torch.float32,
        non_blocking=True,
    )

    query = torch.stack(
        [phrase_features[p.lower()] for p in record["phrases"]]
    ).to(
        device,
        non_blocking=True,
    )

    # [C, E, R]
    similarity = torch.einsum(
        "ed,crd->cer",
        query,
        candidates,
    )

    k = min(local_topk, similarity.shape[-1])
    top_values = torch.topk(
        similarity,
        k=k,
        dim=-1,
    ).values

    weights = F.softmax(
        top_values / temperature,
        dim=-1,
    )
    entity_scores = (weights * top_values).sum(dim=-1)  # [C, E]
    local_scores = entity_scores.mean(dim=-1)  # [C]

    global_candidate_scores = global_scores[candidate_ids].float().cpu()
    local_scores_cpu = local_scores.float().cpu()

    global_norm = zscore(global_candidate_scores)
    local_norm = zscore(local_scores_cpu)
    fused = (
        (1.0 - lambda_value) * global_norm
        + lambda_value * local_norm
    )

    fused_order_pos = torch.argsort(fused, descending=True)
    fused_order_ids = candidate_ids[fused_order_pos]
    reranked_rank = int(
        (fused_order_ids == gt_image).nonzero(as_tuple=False)[0].item()
    )

    global_top_values = global_candidate_scores[: min(5, len(global_candidate_scores))]
    global_margin_12 = float(
        global_candidate_scores[0] - global_candidate_scores[1]
    )
    global_margin_15 = float(
        global_candidate_scores[0] - global_top_values[-1]
    )
    global_topn_std = float(
        global_candidate_scores.std(unbiased=False)
    )

    local_order = torch.argsort(
        local_scores_cpu,
        descending=True,
    )
    local_best_pos = int(local_order[0].item())
    local_second_pos = int(local_order[1].item())

    local_margin_12 = float(
        local_scores_cpu[local_best_pos]
        - local_scores_cpu[local_second_pos]
    )
    local_topn_std = float(
        local_scores_cpu.std(unbiased=False)
    )

    global_top1_pos = 0
    local_advantage_over_global_top1 = float(
        local_scores_cpu[local_best_pos]
        - local_scores_cpu[global_top1_pos]
    )

    # local-best candidate 内：每个 Entity 自己的 Region Top1-Top2 margin。
    local_best_region_scores = similarity[local_best_pos]  # [E, R]
    top2_regions = torch.topk(
        local_best_region_scores,
        k=min(2, local_best_region_scores.shape[-1]),
        dim=-1,
    ).values

    if top2_regions.shape[-1] == 2:
        region_margins = (
            top2_regions[:, 0]
            - top2_regions[:, 1]
        ).float().cpu()
    else:
        region_margins = torch.zeros(
            top2_regions.shape[0],
            dtype=torch.float32,
        )

    entity_region_margin_mean = float(
        region_margins.mean()
    )
    entity_region_margin_min = float(
        region_margins.min()
    )

    # 每个 Entity 单独看 Top-N candidate，最喜欢哪个图；
    # 看它们有多少比例共同支持最终 local-best candidate。
    entity_best_candidate = torch.argmax(
        entity_scores,
        dim=0,
    )
    entity_candidate_agreement = float(
        (entity_best_candidate == local_best_pos)
        .float()
        .mean()
        .item()
    )

    global_local_spearman = spearman(
        global_candidate_scores.numpy(),
        local_scores_cpu.numpy(),
    )

    before = baseline_rank
    after = reranked_rank

    if before > 0 and after == 0:
        category = "rescued"
    elif before == 0 and after > 0:
        category = "corrupted"
    elif after < before:
        category = "improved"
    elif after > before:
        category = "worsened"
    else:
        category = "unchanged"

    gt_candidate_pos = int(
        (candidate_ids == gt_image).nonzero(as_tuple=False)[0].item()
    )

    return {
        "baseline_rank": before + 1,
        "rerankable": True,
        "reranked_rank": after + 1,
        "rank_delta": before - after,
        "category": category,
        "num_entities": len(record["phrases"]),
        "global_top1_image_id": int(candidate_ids[0].item()),
        "local_top1_image_id": int(candidate_ids[local_best_pos].item()),
        "gt_candidate_pos": gt_candidate_pos + 1,
        "global_margin_12": global_margin_12,
        "global_margin_15": global_margin_15,
        "global_topn_std": global_topn_std,
        "local_margin_12": local_margin_12,
        "local_topn_std": local_topn_std,
        "local_advantage_over_global_top1": local_advantage_over_global_top1,
        "entity_region_margin_mean": entity_region_margin_mean,
        "entity_region_margin_min": entity_region_margin_min,
        "entity_candidate_agreement": entity_candidate_agreement,
        "global_local_spearman": global_local_spearman,
    }


def group_summary(records):
    groups = {}

    for category in (
        "rescued",
        "corrupted",
        "improved",
        "worsened",
        "unchanged",
        "unchanged_outside_topn",
    ):
        items = [
            record
            for record in records
            if record["category"] == category
        ]

        groups[category] = {
            "count": len(items),
            "metrics": {
                key: summarize(
                    [item.get(key) for item in items]
                )
                for key in METRIC_KEYS
            },
        }

    return groups


def print_group(groups, category):
    group = groups[category]
    print(f"\n[{category}] n={group['count']}")

    if group["count"] == 0:
        return

    for key in METRIC_KEYS:
        item = group["metrics"][key]
        print(
            f"  {key:<34} "
            f"mean={item['mean']:+.5f} "
            f"median={item['median']:+.5f}"
        )


def write_csv(path, records):
    fields = []
    seen = set()

    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()

    if not 0.0 <= args.lambda_value <= 1.0:
        raise ValueError("--lambda-value 必须位于 [0, 1]。")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    records = load_query_records(args.query_csv)
    image_indices = build_unique_image_indices(train_dataset)

    print("=" * 104)
    print("C1-R1 RELIABILITY ANALYSIS")
    print("=" * 104)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Queries          : {len(records)}")
    print(f"Lambda           : {args.lambda_value}")
    print(f"Global Top-N     : {args.global_topn}")
    print(f"Local Top-K      : {args.local_topk}")
    print(f"Region cache     : {args.region_cache}")
    print("=" * 104)

    print("\nExtracting global image features...")
    image_features = extract_global_image_features(
        model,
        train_dataset,
        image_indices,
        device,
        args.image_batch_size,
    )

    print("\nEncoding captions...")
    text_features = encode_captions(
        model,
        records,
        device,
        args.text_batch_size,
    )

    print("Encoding structured phrases...")
    phrase_features = encode_phrases(
        model,
        records,
        device,
        args.text_batch_size,
    )
    print(f"Unique phrases: {len(phrase_features)}")

    image_features_gpu = image_features.to(
        device,
        non_blocking=True,
    )
    global_scores = (
        text_features.to(device)
        @ image_features_gpu.t()
    ).cpu()

    cached_ids, region_features = load_region_cache(
        args.region_cache
    )
    region_id_to_row = {
        image_id: row
        for row, image_id in enumerate(cached_ids)
    }

    analyzed = []

    print("\nAnalyzing queries...")
    for pos, record in enumerate(records):
        result = analyze_query(
            record,
            global_scores[pos],
            phrase_features,
            region_id_to_row,
            region_features,
            device,
            min(args.global_topn, train_dataset.num_images),
            args.local_topk,
            args.temperature,
            args.lambda_value,
        )

        analyzed.append(
            {
                "dataset_index": record["dataset_index"],
                "pair_index": record["pair_index"],
                "caption": record["caption"],
                "gt_image_id": record["gt_image_id"],
                "image": record["image"],
                "phrases": record["phrases"],
                **result,
            }
        )

        if (
            pos == 0
            or (pos + 1) % 50 == 0
            or pos + 1 == len(records)
        ):
            print(f"  {pos + 1}/{len(records)}")

    groups = group_summary(analyzed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / (
        f"c1r_reliability_lambda_{args.lambda_value:g}.json"
    )
    csv_path = output_dir / (
        f"c1r_reliability_lambda_{args.lambda_value:g}.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "metadata": {
                    "diagnostic_only": True,
                    "split": "train",
                    "checkpoint": args.checkpoint,
                    "query_csv": args.query_csv,
                    "region_cache": args.region_cache,
                    "lambda": args.lambda_value,
                    "global_topn": args.global_topn,
                    "local_topk": args.local_topk,
                    "temperature": args.temperature,
                },
                "groups": groups,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_csv(
        csv_path,
        analyzed,
    )

    print("\n" + "=" * 104)
    print("GROUP COMPARISON")
    print("=" * 104)

    for category in (
        "rescued",
        "corrupted",
        "improved",
        "worsened",
        "unchanged",
    ):
        print_group(groups, category)

    print("\n" + "-" * 104)
    print("优先观察：")
    print("1. rescued vs corrupted 的 global_margin_12")
    print("2. rescued vs corrupted 的 local_margin_12")
    print("3. entity_candidate_agreement 是否在 rescued 更高")
    print("4. entity_region_margin_mean 是否在 rescued 更高")
    print("5. local_advantage_over_global_top1 是否能区分有效纠错与错误破坏")
    print("-" * 104)
    print(f"JSON: {summary_path}")
    print(f"CSV : {csv_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
