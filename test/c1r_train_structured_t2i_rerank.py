import argparse
import csv
import json
import math
import random
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
from datasets.structured_semantics import StructuredSemanticsReader
from models import CLIPRetrieval


ATTRIBUTE_ORDER = {"size": 0, "color": 1, "shape": 2, "state": 3}
DEFAULT_ATTRIBUTE_TYPES = ["color", "size", "shape", "state"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="C1-R-dev: 复用冻结 Train EAR 的 training-free T2I Region reranking。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--ear-file",
        type=str,
        default="E:/paper3/data/structured_semantics/rsicd_train_qwen37_v30_open.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/c1r_train_structured_rerank",
    )
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-topn", type=int, default=50)
    parser.add_argument("--local-topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[32, 64, 96, 128],
    )
    parser.add_argument(
        "--attribute-types",
        type=str,
        nargs="+",
        default=DEFAULT_ATTRIBUTE_TYPES,
    )
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--region-batch-size", type=int, default=128)
    parser.add_argument("--rebuild-region-cache", action="store_true")
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def parse_query_phrases(reader, dataset, index, allowed_types):
    pair_index = dataset.ann[index]["_pair_index"]

    try:
        semantics = reader.get_by_pair(pair_index)
    except KeyError:
        return []

    phrases = []
    seen = set()

    for entity in semantics["entities"]:
        entity_text = entity["text"].strip()
        if not entity_text:
            continue

        # presence=absent 的实体不应该作为正向 Region query。
        absent = any(
            attr["type"] == "presence"
            and attr["value"].strip().lower()
            in {"absent", "no", "none", "without", "not present"}
            for attr in entity["attributes"]
        )
        if absent:
            continue

        attributes = [
            attr
            for attr in entity["attributes"]
            if attr["type"] in allowed_types
        ]
        attributes.sort(
            key=lambda x: ATTRIBUTE_ORDER.get(x["type"], 99)
        )

        values = []
        entity_lower = entity_text.lower()

        for attr in attributes:
            value = attr["value"].strip()
            value_lower = value.lower()

            if not value or value_lower in entity_lower:
                continue
            if value_lower not in {x.lower() for x in values}:
                values.append(value)

        # count 明确不进入 single-region description。
        phrase = " ".join(values + [entity_text]).strip()
        key = phrase.lower()

        if phrase and key not in seen:
            phrases.append(phrase)
            seen.add(key)

    return phrases


def select_queries(dataset, reader, allowed_types, num_queries, seed):
    rng = random.Random(seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    selected = []
    phrases = {}

    for index in candidates:
        query_phrases = parse_query_phrases(
            reader,
            dataset,
            index,
            allowed_types,
        )
        if not query_phrases:
            continue

        selected.append(index)
        phrases[index] = query_phrases

        if len(selected) >= num_queries:
            break

    if not selected:
        raise RuntimeError("冻结 EAR 中没有找到可用训练查询。")

    return selected, phrases


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
        ).to(
            device,
            non_blocking=True,
        )

        features.append(
            model.backbone.encode_image(
                images,
                normalize=True,
            ).cpu()
        )

        if (
            start == 0
            or start + batch_size >= len(image_indices)
            or ((start // batch_size) + 1) % 20 == 0
        ):
            done = min(start + batch_size, len(image_indices))
            print(f"  Global images: {done}/{len(image_indices)}")

    return torch.cat(features)


@torch.no_grad()
def extract_query_text_features(
    model,
    dataset,
    query_indices,
    device,
    batch_size,
):
    captions = [dataset.ann[index]["caption"] for index in query_indices]
    features = []

    for start in range(0, len(captions), batch_size):
        batch = captions[start:start + batch_size]
        features.append(
            model.backbone.encode_text(
                batch,
                normalize=True,
            ).cpu()
        )

    return captions, torch.cat(features)


@torch.no_grad()
def encode_unique_phrases(
    model,
    query_indices,
    query_phrases,
    device,
    batch_size,
):
    phrases = []
    seen = set()

    for index in query_indices:
        for phrase in query_phrases[index]:
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)

    features = []

    for start in range(0, len(phrases), batch_size):
        batch = phrases[start:start + batch_size]
        features.append(
            model.backbone.encode_text(
                batch,
                normalize=True,
            ).cpu()
        )

    features = torch.cat(features)

    return {
        phrase.lower(): features[i]
        for i, phrase in enumerate(phrases)
    }


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
                    (x1, y1, x1 + window, y1 + window, window)
                )

    return regions


def build_region_crops(image, regions):
    image_size = image.shape[-1]
    crops = []

    for x1, y1, x2, y2, _ in regions:
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        crops.append(crop.squeeze(0))

    return torch.stack(crops)


@torch.no_grad()
def extract_region_features(
    model,
    dataset,
    image_indices,
    candidate_image_ids,
    regions,
    device,
    batch_size,
    cache_path,
    rebuild,
):
    candidate_image_ids = sorted(candidate_image_ids)

    if cache_path.exists() and not rebuild:
        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        cached_ids = [int(x) for x in cache["image_ids"]]

        if (
            cached_ids == candidate_image_ids
            and cache["features"].shape[1] == len(regions)
        ):
            print(f"Loaded Region cache: {cache_path}")
            return cached_ids, cache["features"]

        print("Region cache 与当前候选集合不一致，重新提取。")

    features = []

    print(
        f"\nExtracting Region features for "
        f"{len(candidate_image_ids)} candidate images..."
    )

    for pos, image_id in enumerate(candidate_image_ids):
        dataset_index = image_indices[image_id]
        image = dataset[dataset_index][0]
        crops = build_region_crops(image, regions)

        region_features = []

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size].to(
                device,
                non_blocking=True,
            )
            region_features.append(
                model.backbone.encode_image(
                    batch,
                    normalize=True,
                ).cpu()
            )

        features.append(
            torch.cat(region_features).to(torch.float16)
        )

        if (
            pos == 0
            or (pos + 1) % 20 == 0
            or pos + 1 == len(candidate_image_ids)
        ):
            print(
                f"  Regions: {pos + 1}/{len(candidate_image_ids)}"
            )

    features = torch.stack(features)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": candidate_image_ids,
            "features": features,
            "num_regions": len(regions),
        },
        cache_path,
    )

    print(f"Region cache saved: {cache_path}")
    return candidate_image_ids, features


def zscore(values):
    std = values.std(unbiased=False)

    if std.item() < 1e-8:
        return torch.zeros_like(values)

    return (values - values.mean()) / std


@torch.no_grad()
def compute_local_score(
    phrases,
    phrase_features,
    candidate_ids,
    region_id_to_row,
    region_features,
    device,
    local_topk,
    temperature,
):
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
        [phrase_features[phrase.lower()] for phrase in phrases]
    ).to(
        device,
        non_blocking=True,
    )

    # [candidate, entity, region]
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

    entity_scores = (
        weights * top_values
    ).sum(dim=-1)

    # 第一版多个 Entity 等权平均。
    return entity_scores.mean(dim=-1).cpu()


def rank_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "r1": float(100.0 * np.mean(ranks < 1)),
        "r5": float(100.0 * np.mean(ranks < 5)),
        "r10": float(100.0 * np.mean(ranks < 10)),
        "mean": float(
            (
                100.0 * np.mean(ranks < 1)
                + 100.0 * np.mean(ranks < 5)
                + 100.0 * np.mean(ranks < 10)
            )
            / 3.0
        ),
        "medr": float(np.median(ranks) + 1),
        "meanr": float(np.mean(ranks) + 1),
    }


def print_metrics(label, item):
    print(
        f"{label:<10} "
        f"R@1={item['r1']:.2f} "
        f"R@5={item['r5']:.2f} "
        f"R@10={item['r10']:.2f} "
        f"Mean={item['mean']:.2f} "
        f"MedR={item['medr']:.1f}"
    )


def main():
    args = parse_args()

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0.")
    if args.global_topn <= 1:
        raise ValueError("--global-topn must be > 1.")
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

    reader = StructuredSemanticsReader(args.ear_file)
    allowed_types = set(args.attribute_types)

    query_indices, query_phrases = select_queries(
        train_dataset,
        reader,
        allowed_types,
        min(args.num_queries, len(train_dataset)),
        args.seed,
    )

    image_indices = build_unique_image_indices(train_dataset)

    print("=" * 104)
    print("C1-R-DEV: FROZEN TRAIN EAR STRUCTURED LOCAL T2I RERANKING")
    print("=" * 104)
    print("注意：这是 Train split 方法诊断，不是正式泛化指标。")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Frozen EAR       : {args.ear_file}")
    print(f"Train images     : {train_dataset.num_images}")
    print(f"Train pairs      : {len(train_dataset)}")
    print(f"Selected queries : {len(query_indices)}")
    print(f"Attributes       : {sorted(allowed_types)}")
    print(f"Global Top-N     : {args.global_topn}")
    print(f"Local Top-K      : {args.local_topk}")
    print(f"Temperature      : {args.temperature}")
    print(f"Lambdas          : {args.lambdas}")
    print("=" * 104)

    print("\nExtracting unique global image features...")
    image_features = extract_global_image_features(
        model,
        train_dataset,
        image_indices,
        device,
        args.image_batch_size,
    )

    print("\nEncoding query captions...")
    captions, text_features = extract_query_text_features(
        model,
        train_dataset,
        query_indices,
        device,
        args.text_batch_size,
    )

    print("Encoding unique Entity/Attribute phrases...")
    phrase_features = encode_unique_phrases(
        model,
        query_indices,
        query_phrases,
        device,
        args.text_batch_size,
    )
    print(f"Unique local phrases: {len(phrase_features)}")

    global_scores = (
        text_features.to(device)
        @ image_features.to(device).t()
    ).cpu()

    topn = min(args.global_topn, train_dataset.num_images)
    baseline_ranks = []
    top_candidates = []
    rerankable = []

    for query_pos, dataset_index in enumerate(query_indices):
        gt_image = int(train_dataset.image_ids[dataset_index])
        scores = global_scores[query_pos]
        order = torch.argsort(scores, descending=True)

        rank = int(
            (order == gt_image).nonzero(as_tuple=False)[0].item()
        )
        candidates = order[:topn]

        baseline_ranks.append(rank)
        top_candidates.append(candidates)
        rerankable.append(rank < topn)

    candidate_image_ids = set()

    for can_rerank, candidates in zip(
        rerankable,
        top_candidates,
    ):
        if can_rerank:
            candidate_image_ids.update(
                int(x) for x in candidates.tolist()
            )

    print(
        f"\nRerankable queries (GT in Top-{topn}): "
        f"{sum(rerankable)}/{len(rerankable)}"
    )
    print(
        f"Unique candidate images needing Regions: "
        f"{len(candidate_image_ids)}"
    )

    image_size = int(
        config["dataset"].get("image_res", 224)
    )
    regions = generate_regions(
        image_size,
        args.windows,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_tag = (
        f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / (
        f"regions_{checkpoint_tag}_"
        f"q{len(query_indices)}_seed{args.seed}_top{topn}_"
        f"{'-'.join(map(str, args.windows))}.pt"
    )

    cached_ids, region_features = extract_region_features(
        model,
        train_dataset,
        image_indices,
        candidate_image_ids,
        regions,
        device,
        args.region_batch_size,
        cache_path,
        args.rebuild_region_cache,
    )

    region_id_to_row = {
        image_id: row
        for row, image_id in enumerate(cached_ids)
    }

    lambdas = [float(value) for value in args.lambdas]
    ranks_by_lambda = {
        lam: []
        for lam in lambdas
    }
    records = []

    print("\nReranking...")
    for query_pos, dataset_index in enumerate(query_indices):
        gt_image = int(train_dataset.image_ids[dataset_index])
        before = baseline_ranks[query_pos]
        candidates = top_candidates[query_pos]

        if rerankable[query_pos]:
            local_scores = compute_local_score(
                query_phrases[dataset_index],
                phrase_features,
                candidates,
                region_id_to_row,
                region_features,
                device,
                args.local_topk,
                args.temperature,
            )

            global_candidate_scores = (
                global_scores[query_pos][candidates].float()
            )
            global_norm = zscore(global_candidate_scores)
            local_norm = zscore(local_scores)

            query_ranks = {}

            for lam in lambdas:
                fused = (
                    (1.0 - lam) * global_norm
                    + lam * local_norm
                )
                fused_order = candidates[
                    torch.argsort(
                        fused,
                        descending=True,
                    )
                ]
                rank = int(
                    (fused_order == gt_image)
                    .nonzero(as_tuple=False)[0]
                    .item()
                )
                query_ranks[lam] = rank
                ranks_by_lambda[lam].append(rank)
        else:
            query_ranks = {
                lam: before
                for lam in lambdas
            }
            for lam in lambdas:
                ranks_by_lambda[lam].append(before)

        record = {
            "dataset_index": int(dataset_index),
            "pair_index": int(
                train_dataset.ann[dataset_index]["_pair_index"]
            ),
            "caption": captions[query_pos],
            "gt_image_id": gt_image,
            "image": train_dataset.ann[dataset_index]["image"],
            "phrases": query_phrases[dataset_index],
            "baseline_rank": before + 1,
            "rerankable": bool(rerankable[query_pos]),
        }

        for lam in lambdas:
            record[f"rank_lambda_{lam:g}"] = (
                query_ranks[lam] + 1
            )

        records.append(record)

        if (
            query_pos == 0
            or (query_pos + 1) % 50 == 0
            or query_pos + 1 == len(query_indices)
        ):
            print(
                f"  {query_pos + 1}/{len(query_indices)}"
            )

    if 0.0 in ranks_by_lambda:
        if ranks_by_lambda[0.0] != baseline_ranks:
            raise RuntimeError(
                "λ=0 未严格退化为 Global CLIP 排序。"
            )

    baseline_metrics = rank_metrics(baseline_ranks)
    sweep = {}

    print("\n" + "=" * 104)
    print("C1-R-DEV RESULTS")
    print("=" * 104)
    print_metrics("Global", baseline_metrics)

    best_lambda = None
    best_mean = -math.inf

    for lam in lambdas:
        ranks = ranks_by_lambda[lam]
        metrics = rank_metrics(ranks)

        before = np.asarray(baseline_ranks)
        after = np.asarray(ranks)

        rescue = int(
            np.sum((before > 0) & (after == 0))
        )
        corruption = int(
            np.sum((before == 0) & (after > 0))
        )
        improved = int(
            np.sum(after < before)
        )
        worsened = int(
            np.sum(after > before)
        )
        unchanged = int(
            np.sum(after == before)
        )

        hard_mask = (
            (before > 0)
            & (before < topn)
        )
        hard_count = int(hard_mask.sum())
        hard_rescue = int(
            np.sum(
                hard_mask
                & (after == 0)
            )
        )
        hard_improved = int(
            np.sum(
                hard_mask
                & (after < before)
            )
        )

        sweep[str(lam)] = {
            "metrics": metrics,
            "rescued_to_r1": rescue,
            "corrupted_from_r1": corruption,
            "rank_improved": improved,
            "rank_worsened": worsened,
            "rank_unchanged": unchanged,
            "recoverable_hard_queries": hard_count,
            "hard_rescued_to_r1": hard_rescue,
            "hard_rank_improved": hard_improved,
        }

        print_metrics(f"λ={lam:g}", metrics)
        print(
            f"           rescue={rescue} "
            f"corruption={corruption} "
            f"improved={improved} "
            f"worsened={worsened} | "
            f"hard rescue={hard_rescue}/{hard_count}, "
            f"hard improved={hard_improved}/{hard_count}"
        )

        if metrics["mean"] > best_mean:
            best_mean = metrics["mean"]
            best_lambda = lam

    best_ranks = np.asarray(
        ranks_by_lambda[best_lambda]
    )
    before = np.asarray(baseline_ranks)

    for record, old_rank, new_rank in zip(
        records,
        before,
        best_ranks,
    ):
        record["best_lambda"] = best_lambda
        record["best_rank"] = int(new_rank + 1)
        record["rank_delta"] = int(old_rank - new_rank)
        record["rescued_to_r1"] = bool(
            old_rank > 0 and new_rank == 0
        )
        record["corrupted_from_r1"] = bool(
            old_rank == 0 and new_rank > 0
        )

    summary_path = output_dir / "c1r_train_summary.json"
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "metadata": {
                    "diagnostic_only": True,
                    "split": "train",
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "ear_file": args.ear_file,
                    "num_queries": len(query_indices),
                    "seed": args.seed,
                    "num_images": train_dataset.num_images,
                    "attribute_types": sorted(allowed_types),
                    "count_policy": (
                        "excluded from single-region description"
                    ),
                    "windows": args.windows,
                    "num_regions": len(regions),
                    "global_topn": topn,
                    "local_topk": args.local_topk,
                    "temperature": args.temperature,
                    "fusion": (
                        "candidate-wise zscore then "
                        "(1-lambda)*global + lambda*local"
                    ),
                },
                "baseline": baseline_metrics,
                "sweep": sweep,
                "best_lambda": best_lambda,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = output_dir / "c1r_train_query_cases.csv"
    fields = list(records[0].keys())

    with csv_path.open(
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

    rescued_path = output_dir / "rescued_cases.json"
    corrupted_path = output_dir / "corrupted_cases.json"

    with rescued_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [
                record
                for record in records
                if record["rescued_to_r1"]
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with corrupted_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [
                record
                for record in records
                if record["corrupted_from_r1"]
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("-" * 104)
    print(f"Best λ          : {best_lambda:g}")
    print(f"Summary         : {summary_path}")
    print(f"Query cases     : {csv_path}")
    print(f"Rescued cases   : {rescued_path}")
    print(f"Corrupted cases : {corrupted_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
