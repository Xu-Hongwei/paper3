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


def parse_args():
    parser = argparse.ArgumentParser(
        description="C1-R2: Coverage-aware multi-Entity local aggregation."
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
        default="outputs/c1r_coverage_ablation",
    )
    parser.add_argument("--global-topn", type=int, default=50)
    parser.add_argument("--local-topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument(
        "--coverage-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="0=原始 mean；1=只使用 lower-half Entity evidence。",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_query_records(path):
    records = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "pair_index": int(row["pair_index"]),
                    "caption": row["caption"],
                    "gt_image_id": int(row["gt_image_id"]),
                    "image": row["image"],
                    "phrases": ast.literal_eval(row["phrases"]),
                }
            )
    if not records:
        raise ValueError("query CSV 为空。")
    return records


def build_unique_image_indices(dataset):
    first_index = [None] * dataset.num_images
    for index, image_id in enumerate(dataset.image_ids):
        if first_index[image_id] is None:
            first_index[image_id] = index
    if any(index is None for index in first_index):
        raise RuntimeError("训练 image_id 映射不完整。")
    return first_index


@torch.no_grad()
def encode_images(model, dataset, image_indices, device, batch_size):
    features = []
    for start in range(0, len(image_indices), batch_size):
        indices = image_indices[start:start + batch_size]
        images = torch.stack([dataset[index][0] for index in indices]).to(
            device, non_blocking=True
        )
        features.append(model.backbone.encode_image(images, normalize=True).cpu())

        done = min(start + batch_size, len(image_indices))
        if start == 0 or done == len(image_indices) or (start // batch_size + 1) % 20 == 0:
            print(f"  Global images: {done}/{len(image_indices)}")
    return torch.cat(features)


@torch.no_grad()
def encode_captions(model, records, device, batch_size):
    captions = [record["caption"] for record in records]
    features = []
    for start in range(0, len(captions), batch_size):
        features.append(
            model.backbone.encode_text(
                captions[start:start + batch_size], normalize=True
            ).cpu()
        )
    return torch.cat(features)


@torch.no_grad()
def encode_phrases(model, records, device, batch_size):
    phrases, seen = [], set()
    for record in records:
        for phrase in record["phrases"]:
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)

    features = []
    for start in range(0, len(phrases), batch_size):
        features.append(
            model.backbone.encode_text(
                phrases[start:start + batch_size], normalize=True
            ).cpu()
        )
    features = torch.cat(features)
    return {phrase.lower(): features[i] for i, phrase in enumerate(phrases)}


def load_region_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    image_ids = [int(x) for x in cache["image_ids"]]
    features = cache["features"]
    if len(image_ids) != len(features):
        raise ValueError("Region cache image_ids/features 数量不一致。")
    return image_ids, features


def zscore(x):
    std = x.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / std


def coverage_local_scores(entity_scores, alpha):
    """
    entity_scores: [candidate, entity]
    mean 保留总体匹配；lower-half 强调较弱的 Entity，近似语义 coverage。
    """
    mean_score = entity_scores.mean(dim=-1)
    num_entities = entity_scores.shape[-1]
    lower_k = max(1, math.ceil(num_entities / 2))
    lower_half = torch.topk(
        entity_scores, k=lower_k, dim=-1, largest=False
    ).values.mean(dim=-1)
    return (1.0 - alpha) * mean_score + alpha * lower_half


@torch.no_grad()
def get_entity_scores(
    record,
    candidate_ids,
    phrase_features,
    region_id_to_row,
    region_features,
    device,
    local_topk,
    temperature,
):
    rows = [region_id_to_row[int(image_id)] for image_id in candidate_ids.tolist()]
    candidates = region_features[rows].to(
        device, dtype=torch.float32, non_blocking=True
    )
    query = torch.stack(
        [phrase_features[phrase.lower()] for phrase in record["phrases"]]
    ).to(device, non_blocking=True)

    similarity = torch.einsum("ed,crd->cer", query, candidates)
    k = min(local_topk, similarity.shape[-1])
    top_values = torch.topk(similarity, k=k, dim=-1).values
    weights = F.softmax(top_values / temperature, dim=-1)
    return (weights * top_values).sum(dim=-1).float().cpu()


def rank_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    return {
        "r1": float(r1),
        "r5": float(r5),
        "r10": float(r10),
        "mean": float((r1 + r5 + r10) / 3.0),
        "medr": float(np.median(ranks) + 1),
        "meanr": float(np.mean(ranks) + 1),
    }


def result_stats(before, after, topn):
    before = np.asarray(before)
    after = np.asarray(after)

    hard_mask = (before > 0) & (before < topn)
    return {
        "rescued_to_r1": int(np.sum((before > 0) & (after == 0))),
        "corrupted_from_r1": int(np.sum((before == 0) & (after > 0))),
        "rank_improved": int(np.sum(after < before)),
        "rank_worsened": int(np.sum(after > before)),
        "rank_unchanged": int(np.sum(after == before)),
        "recoverable_hard_queries": int(hard_mask.sum()),
        "hard_rescued_to_r1": int(np.sum(hard_mask & (after == 0))),
        "hard_rank_improved": int(np.sum(hard_mask & (after < before))),
    }


def print_metrics(label, metrics, stats=None):
    print(
        f"{label:<20} R@1={metrics['r1']:.2f} R@5={metrics['r5']:.2f} "
        f"R@10={metrics['r10']:.2f} Mean={metrics['mean']:.2f}"
    )
    if stats:
        print(
            f"{'':20} rescue={stats['rescued_to_r1']} "
            f"corruption={stats['corrupted_from_r1']} "
            f"improved={stats['rank_improved']} worsened={stats['rank_worsened']} | "
            f"hard rescue={stats['hard_rescued_to_r1']}/"
            f"{stats['recoverable_hard_queries']}"
        )


def main():
    args = parse_args()

    if any(not 0.0 <= alpha <= 1.0 for alpha in args.coverage_alphas):
        raise ValueError("coverage alpha 必须位于 [0, 1]。")
    if args.temperature <= 0:
        raise ValueError("temperature 必须 > 0。")

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

    records = load_query_records(args.query_csv)
    image_indices = build_unique_image_indices(train_dataset)

    print("=" * 108)
    print("C1-R2 COVERAGE-AWARE LOCAL AGGREGATION")
    print("=" * 108)
    print("注意：Train split 方法诊断，不作为最终泛化指标。")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Queries          : {len(records)}")
    print(f"Global Top-N     : {args.global_topn}")
    print(f"Local Top-K      : {args.local_topk}")
    print(f"Coverage alphas  : {args.coverage_alphas}")
    print(f"Lambdas          : {args.lambdas}")
    print("=" * 108)

    print("\nExtracting global image features...")
    image_features = encode_images(
        model, train_dataset, image_indices, device, args.image_batch_size
    )

    print("\nEncoding captions...")
    text_features = encode_captions(
        model, records, device, args.text_batch_size
    )

    print("Encoding structured phrases...")
    phrase_features = encode_phrases(
        model, records, device, args.text_batch_size
    )

    global_scores = (
        text_features.to(device) @ image_features.to(device).t()
    ).cpu()

    cached_ids, region_features = load_region_cache(args.region_cache)
    region_id_to_row = {
        image_id: row for row, image_id in enumerate(cached_ids)
    }

    topn = min(args.global_topn, train_dataset.num_images)
    baseline_ranks = []
    candidates_by_query = []
    entity_scores_by_query = []

    print("\nPreparing local Entity evidence...")
    for pos, record in enumerate(records):
        gt_image = record["gt_image_id"]
        scores = global_scores[pos]
        order = torch.argsort(scores, descending=True)
        baseline_rank = int(
            (order == gt_image).nonzero(as_tuple=False)[0].item()
        )
        candidates = order[:topn]

        baseline_ranks.append(baseline_rank)
        candidates_by_query.append(candidates)

        if baseline_rank < topn:
            missing = [
                int(image_id)
                for image_id in candidates.tolist()
                if int(image_id) not in region_id_to_row
            ]
            if missing:
                raise KeyError(
                    f"Region cache 缺少候选 image_id={missing[0]}，"
                    "请确认和原 C1-R 使用同一 query/topn/cache。"
                )

            entity_scores = get_entity_scores(
                record,
                candidates,
                phrase_features,
                region_id_to_row,
                region_features,
                device,
                args.local_topk,
                args.temperature,
            )
        else:
            entity_scores = None

        entity_scores_by_query.append(entity_scores)

        if pos == 0 or (pos + 1) % 50 == 0 or pos + 1 == len(records):
            print(f"  {pos + 1}/{len(records)}")

    baseline_metrics = rank_metrics(baseline_ranks)
    print("\n" + "=" * 108)
    print("RESULTS")
    print("=" * 108)
    print_metrics("Global", baseline_metrics)

    sweep = {}
    best = None
    best_mean = -math.inf

    for alpha in args.coverage_alphas:
        for lam in args.lambdas:
            ranks = []

            for pos, record in enumerate(records):
                before = baseline_ranks[pos]
                if before >= topn:
                    ranks.append(before)
                    continue

                candidates = candidates_by_query[pos]
                entity_scores = entity_scores_by_query[pos]

                local_scores = coverage_local_scores(
                    entity_scores, alpha
                )
                global_candidate_scores = global_scores[pos][candidates].float()

                fused = (
                    (1.0 - lam) * zscore(global_candidate_scores)
                    + lam * zscore(local_scores)
                )
                fused_order = candidates[
                    torch.argsort(fused, descending=True)
                ]
                gt_image = record["gt_image_id"]
                rank = int(
                    (fused_order == gt_image)
                    .nonzero(as_tuple=False)[0]
                    .item()
                )
                ranks.append(rank)

            metrics = rank_metrics(ranks)
            stats = result_stats(baseline_ranks, ranks, topn)

            key = f"alpha={alpha:g},lambda={lam:g}"
            sweep[key] = {
                "alpha": alpha,
                "lambda": lam,
                "metrics": metrics,
                **stats,
            }

            print_metrics(
                f"α={alpha:g}, λ={lam:g}",
                metrics,
                stats,
            )

            if metrics["mean"] > best_mean:
                best_mean = metrics["mean"]
                best = {
                    "alpha": alpha,
                    "lambda": lam,
                    "ranks": ranks,
                    "metrics": metrics,
                    **stats,
                }

    if 0.0 in args.coverage_alphas:
        # α=0 必须等价于原来的 Entity mean aggregation。
        print("\nSanity: α=0 is the original mean local aggregation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_cases = []
    for record, before, after in zip(
        records, baseline_ranks, best["ranks"]
    ):
        query_cases.append(
            {
                **record,
                "baseline_rank": before + 1,
                "best_rank": after + 1,
                "rank_delta": before - after,
                "best_alpha": best["alpha"],
                "best_lambda": best["lambda"],
                "rescued_to_r1": bool(before > 0 and after == 0),
                "corrupted_from_r1": bool(before == 0 and after > 0),
            }
        )

    summary_path = output_dir / "c1r_coverage_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "diagnostic_only": True,
                    "split": "train",
                    "checkpoint": args.checkpoint,
                    "query_csv": args.query_csv,
                    "region_cache": args.region_cache,
                    "global_topn": topn,
                    "local_topk": args.local_topk,
                    "temperature": args.temperature,
                    "coverage_definition": (
                        "L=(1-alpha)*mean(entity_scores) + "
                        "alpha*mean(lower-half entity_scores)"
                    ),
                    "note": (
                        "This is a diagnostic coverage surrogate, "
                        "not claimed as the exact formulation of a cited paper."
                    ),
                },
                "baseline": baseline_metrics,
                "sweep": sweep,
                "best": {
                    key: value
                    for key, value in best.items()
                    if key != "ranks"
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = output_dir / "c1r_coverage_query_cases.csv"
    fields = list(query_cases[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_cases)

    with (output_dir / "rescued_cases.json").open("w", encoding="utf-8") as f:
        json.dump(
            [x for x in query_cases if x["rescued_to_r1"]],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "corrupted_cases.json").open("w", encoding="utf-8") as f:
        json.dump(
            [x for x in query_cases if x["corrupted_from_r1"]],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 108)
    print(
        f"Best: α={best['alpha']:g}, λ={best['lambda']:g} | "
        f"R@1={best['metrics']['r1']:.2f}, "
        f"R@5={best['metrics']['r5']:.2f}, "
        f"R@10={best['metrics']['r10']:.2f}, "
        f"Mean={best['metrics']['mean']:.2f}"
    )
    print(
        f"rescue={best['rescued_to_r1']} "
        f"corruption={best['corrupted_from_r1']} "
        f"improved={best['rank_improved']} "
        f"worsened={best['rank_worsened']}"
    )
    print(f"JSON: {summary_path}")
    print(f"CSV : {csv_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
