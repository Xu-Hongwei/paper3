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
from datasets.structured_semantics import StructuredSemanticsReader
from models import CLIPRetrieval


DEFAULT_ATTRIBUTE_TYPES = ["count", "color", "size", "shape", "state"]
ATTRIBUTE_ORDER = {
    "count": 0,
    "size": 1,
    "color": 2,
    "shape": 3,
    "state": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.3: Entity vs Entity+Attribute vs Contextual Entity."
    )
    parser.add_argument("--config", type=str, default="configs/baseline/rsicd.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/clip_rsicd_10ep/best.pth",
    )
    parser.add_argument(
        "--ear-file",
        type=str,
        default="E:/paper3/data/structured_semantics/rsicd_train_qwen37_v30_open.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/c03_entity_attribute_ablation",
    )
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument(
        "--attribute-types",
        type=str,
        nargs="+",
        default=DEFAULT_ATTRIBUTE_TYPES,
        help="默认只测试更适合单 Region 的 count/color/size/shape/state。",
    )
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_entity_index(dataset):
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


def get_valid_span_records(dataset, entity_index, dataset_index):
    pair_index = dataset.ann[dataset_index]["_pair_index"]
    semantic_index = int(entity_index["pair_to_semantic"][pair_index].item())
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


def bind_structured_entities(
    dataset,
    entity_index,
    semantics_reader,
    dataset_index,
    allowed_types,
):
    """
    仅保留：
    1. 有有效 CLIP span；
    2. 在冻结 EAR 中能唯一按 text 对上的 Entity。
    不猜测、不做同义词映射。
    """
    pair_index = dataset.ann[dataset_index]["_pair_index"]
    spans = get_valid_span_records(dataset, entity_index, dataset_index)
    semantics = semantics_reader.get_by_pair(pair_index)

    by_text = {}
    for entity in semantics["entities"]:
        by_text.setdefault(entity["text"], []).append(entity)

    records = []
    for span_record in spans:
        matches = by_text.get(span_record["text"], [])
        if len(matches) != 1:
            continue

        entity = matches[0]
        attributes = [
            attr
            for attr in entity["attributes"]
            if attr["type"] in allowed_types
        ]
        records.append(
            {
                "text": span_record["text"],
                "span": span_record["span"],
                "attributes": attributes,
            }
        )

    return records


def build_attribute_phrase(entity, attributes):
    if not attributes:
        return entity

    attrs = sorted(
        attributes,
        key=lambda x: ATTRIBUTE_ORDER.get(x["type"], 99),
    )
    values = []
    entity_lower = entity.lower()
    for attr in attrs:
        value = attr["value"].strip()
        value_lower = value.lower()

        # EAR 有时已把属性写进 Entity 文本，如 "green trees"；
        # 这里避免生成 "green green trees"。
        if not value or value_lower in entity_lower:
            continue
        if value_lower not in {v.lower() for v in values}:
            values.append(value)

    return " ".join(values + [entity]).strip()


def sliding_positions(image_size, window, stride):
    positions = list(range(0, image_size - window + 1, stride))
    last = image_size - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_regions(image_size, windows):
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
def get_visual(cache, dataset, index, model, regions, device, batch_size):
    image_key = dataset.ann[index]["image"]
    if image_key in cache:
        return cache[image_key]

    image, caption, image_id, _ = dataset[index]
    crops = build_region_crops(image, regions)
    cache[image_key] = {
        "image": image_key,
        "caption": caption,
        "image_id": int(image_id),
        "region_features": encode_regions(
            model,
            crops,
            device,
            batch_size,
        ),
    }
    return cache[image_key]


def pool_contextual_spans(token_features, records):
    pooled = []
    for record in records:
        start, end = record["span"]
        pooled.append(token_features[0, start:end].mean(dim=0))
    return F.normalize(torch.stack(pooled), dim=-1)


@torch.no_grad()
def encode_text_variants(model, caption, records):
    entity_texts = [record["text"] for record in records]
    attr_phrases = [
        build_attribute_phrase(record["text"], record["attributes"])
        for record in records
    ]

    entity_features = model.backbone.encode_text(entity_texts, normalize=True).cpu()
    attribute_features = model.backbone.encode_text(attr_phrases, normalize=True).cpu()

    _, tokens = model.backbone.encode_text_with_tokens(
        [caption],
        normalize=False,
    )
    contextual_features = pool_contextual_spans(tokens, records).cpu()

    return entity_texts, attr_phrases, entity_features, attribute_features, contextual_features


def choose_negatives(selected, current_pos, num_negatives, seed):
    candidates = [x for pos, x in enumerate(selected) if pos != current_pos]
    rng = random.Random(seed + current_pos)
    return rng.sample(candidates, min(num_negatives, len(candidates)))


def pearson(x, y):
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if denom.item() > 1e-12 else 0.0


def select_samples(
    dataset,
    entity_index,
    semantics_reader,
    allowed_types,
    num_samples,
    indices,
    seed,
):
    def valid(index):
        records = bind_structured_entities(
            dataset,
            entity_index,
            semantics_reader,
            index,
            allowed_types,
        )
        return len(records) >= 2 and any(r["attributes"] for r in records)

    if indices:
        return [index for index in indices if valid(index)]

    rng = random.Random(seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    selected, used_images = [], set()
    for index in candidates:
        image = dataset.ann[index]["image"]
        if image in used_images or not valid(index):
            continue
        selected.append(index)
        used_images.add(image)
        if len(selected) >= num_samples:
            break
    return selected


@torch.no_grad()
def analyze_sample(
    dataset,
    entity_index,
    semantics_reader,
    allowed_types,
    index,
    pos,
    selected,
    model,
    regions,
    cache,
    device,
    region_batch_size,
    num_negatives,
    seed,
):
    records = bind_structured_entities(
        dataset,
        entity_index,
        semantics_reader,
        index,
        allowed_types,
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

    texts, phrases, e_feat, a_feat, c_feat = encode_text_variants(
        model,
        matched["caption"],
        records,
    )
    scores = {
        "entity": e_feat @ matched["region_features"].t(),
        "attribute": a_feat @ matched["region_features"].t(),
        "contextual": c_feat @ matched["region_features"].t(),
    }

    negative_indices = choose_negatives(
        selected,
        pos,
        num_negatives,
        seed,
    )
    negative_max = {
        "entity": [],
        "attribute": [],
        "contextual": [],
    }

    for negative_index in negative_indices:
        negative = get_visual(
            cache,
            dataset,
            negative_index,
            model,
            regions,
            device,
            region_batch_size,
        )
        for name, feature in (
            ("entity", e_feat),
            ("attribute", a_feat),
            ("contextual", c_feat),
        ):
            negative_max[name].append(
                (feature @ negative["region_features"].t()).max(dim=1).values
            )

    for name in negative_max:
        negative_max[name] = torch.stack(negative_max[name], dim=1)

    entities = []
    for i, record in enumerate(records):
        result = {
            "text": texts[i],
            "attribute_phrase": phrases[i],
            "attributes": record["attributes"],
            "has_attribute": bool(record["attributes"]),
            "span": record["span"],
        }

        best_indices = {}
        for name in ("entity", "attribute", "contextual"):
            matched_max, argmax = scores[name][i].max(dim=0)
            mean_neg = negative_max[name][i].mean()
            hard_neg = negative_max[name][i].max()

            result[f"{name}_matched_max"] = float(matched_max.item())
            result[f"{name}_gap_mean_negative"] = float(
                (matched_max - mean_neg).item()
            )
            result[f"{name}_gap_hard_negative"] = float(
                (matched_max - hard_neg).item()
            )
            result[f"{name}_best_region_index"] = int(argmax.item())
            result[f"{name}_best_scale"] = int(
                regions[int(argmax.item())]["scale"]
            )
            best_indices[name] = int(argmax.item())

        result["attribute_minus_entity_mean_gap"] = (
            result["attribute_gap_mean_negative"]
            - result["entity_gap_mean_negative"]
        )
        result["attribute_minus_entity_hard_gap"] = (
            result["attribute_gap_hard_negative"]
            - result["entity_gap_hard_negative"]
        )
        result["context_minus_entity_mean_gap"] = (
            result["contextual_gap_mean_negative"]
            - result["entity_gap_mean_negative"]
        )
        result["entity_attribute_text_cosine"] = float((e_feat[i] @ a_feat[i]).item())
        result["entity_attribute_score_corr"] = pearson(
            scores["entity"][i],
            scores["attribute"][i],
        )
        result["attribute_context_score_corr"] = pearson(
            scores["attribute"][i],
            scores["contextual"][i],
        )
        result["entity_attribute_same_top1"] = (
            best_indices["entity"] == best_indices["attribute"]
        )
        entities.append(result)

    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            pairs.append(
                {
                    "entity_a": texts[i],
                    "entity_b": texts[j],
                    "entity_corr": pearson(scores["entity"][i], scores["entity"][j]),
                    "attribute_corr": pearson(
                        scores["attribute"][i],
                        scores["attribute"][j],
                    ),
                    "contextual_corr": pearson(
                        scores["contextual"][i],
                        scores["contextual"][j],
                    ),
                }
            )

    return {
        "dataset_index": int(index),
        "pair_index": int(dataset.ann[index]["_pair_index"]),
        "image_id": matched["image_id"],
        "image": matched["image"],
        "caption": matched["caption"],
        "entities": entities,
        "entity_pairs": pairs,
    }


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


def metric(values):
    result = summarize(values)
    result["positive_rate"] = (
        sum(v > 0 for v in values) / len(values)
        if values
        else None
    )
    return result


def corr_metric(values):
    result = summarize(values)
    result["mean_absolute"] = (
        sum(abs(v) for v in values) / len(values)
        if values
        else None
    )
    return result


def build_aggregate(samples):
    all_entities = [e for s in samples for e in s["entities"]]
    attributed = [e for e in all_entities if e["has_attribute"]]

    def variant_metrics(items, name):
        return {
            "gap_mean_negative": metric(
                [e[f"{name}_gap_mean_negative"] for e in items]
            ),
            "gap_hard_negative": metric(
                [e[f"{name}_gap_hard_negative"] for e in items]
            ),
        }

    pair_values = {
        name: [
            p[f"{name}_corr"]
            for sample in samples
            for p in sample["entity_pairs"]
        ]
        for name in ("entity", "attribute", "contextual")
    }

    return {
        "num_samples": len(samples),
        "num_entities": len(all_entities),
        "num_attributed_entities": len(attributed),
        "attribute_coverage": len(attributed) / max(len(all_entities), 1),
        "all_entities": {
            "entity": variant_metrics(all_entities, "entity"),
            "attribute": variant_metrics(all_entities, "attribute"),
            "contextual": variant_metrics(all_entities, "contextual"),
        },
        "attributed_entities": {
            "entity": variant_metrics(attributed, "entity"),
            "attribute": variant_metrics(attributed, "attribute"),
            "contextual": variant_metrics(attributed, "contextual"),
            "attribute_minus_entity_mean_gap": metric(
                [e["attribute_minus_entity_mean_gap"] for e in attributed]
            ),
            "attribute_minus_entity_hard_gap": metric(
                [e["attribute_minus_entity_hard_gap"] for e in attributed]
            ),
            "context_minus_entity_mean_gap": metric(
                [e["context_minus_entity_mean_gap"] for e in attributed]
            ),
            "entity_attribute_text_cosine": summarize(
                [e["entity_attribute_text_cosine"] for e in attributed]
            ),
            "entity_attribute_score_correlation": summarize(
                [e["entity_attribute_score_corr"] for e in attributed]
            ),
            "same_top1_rate": (
                sum(e["entity_attribute_same_top1"] for e in attributed)
                / max(len(attributed), 1)
            ),
        },
        "entity_pair_correlation": {
            name: corr_metric(values)
            for name, values in pair_values.items()
        },
    }


def write_csv(output_dir, samples):
    path = output_dir / "entity_attribute_ablation.csv"
    rows = []

    for sample in samples:
        for entity in sample["entities"]:
            rows.append(
                {
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
            )

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return path


def print_variant(name, item):
    mean = item["gap_mean_negative"]
    hard = item["gap_hard_negative"]
    print(
        f"{name:<12} | mean-neg={mean['mean']:+.4f} "
        f"({mean['positive_rate']:.2%}) | "
        f"hard-neg={hard['mean']:+.4f} "
        f"({hard['positive_rate']:.2%})"
    )


def print_aggregate(aggregate):
    print("\n" + "=" * 108)
    print("C0.3 ENTITY vs ENTITY+ATTRIBUTE vs CONTEXTUAL")
    print("=" * 108)
    print(f"Samples             : {aggregate['num_samples']}")
    print(f"Valid entities      : {aggregate['num_entities']}")
    print(f"Attributed entities : {aggregate['num_attributed_entities']}")
    print(f"Attribute coverage  : {aggregate['attribute_coverage']:.2%}")

    print("\n[Primary] Attribute-bearing entities")
    primary = aggregate["attributed_entities"]
    print_variant("Entity", primary["entity"])
    print_variant("Entity+Attr", primary["attribute"])
    print_variant("Contextual", primary["contextual"])

    delta = primary["attribute_minus_entity_mean_gap"]
    hard = primary["attribute_minus_entity_hard_gap"]
    print(
        f"\nAttr - Entity mean gap : mean={delta['mean']:+.4f}, "
        f"median={delta['median']:+.4f}, positive={delta['positive_rate']:.2%}"
    )
    print(
        f"Attr - Entity hard gap : mean={hard['mean']:+.4f}, "
        f"median={hard['median']:+.4f}, positive={hard['positive_rate']:.2%}"
    )
    print(
        "Entity↔Attr text cosine  : "
        f"mean={primary['entity_attribute_text_cosine']['mean']:.4f}"
    )
    print(
        "Entity↔Attr score corr   : "
        f"mean={primary['entity_attribute_score_correlation']['mean']:.4f}"
    )
    print(
        f"Same Top-1 Region rate  : {primary['same_top1_rate']:.2%}"
    )

    print("\nEntity-pair Region score correlation")
    for name, label in (
        ("entity", "Entity"),
        ("attribute", "Entity+Attr"),
        ("contextual", "Contextual"),
    ):
        item = aggregate["entity_pair_correlation"][name]
        print(
            f"{label:<12}: mean={item['mean']:+.4f}, "
            f"median={item['median']:+.4f}, "
            f"|mean|={item['mean_absolute']:.4f}"
        )

    print("=" * 108)


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
    entity_index = load_entity_index(train_dataset)
    semantics_reader = StructuredSemanticsReader(args.ear_file)

    image_size = int(config["dataset"].get("image_res", 224))
    regions = generate_regions(image_size, args.windows)
    allowed_types = set(args.attribute_types)

    selected = select_samples(
        train_dataset,
        entity_index,
        semantics_reader,
        allowed_types,
        args.num_samples,
        args.indices,
        args.seed,
    )
    if len(selected) < 2:
        raise RuntimeError("没有找到足够的有效 Attribute 样本。")

    print("=" * 108)
    print("C0.3 ENTITY vs ENTITY+ATTRIBUTE vs CONTEXTUAL")
    print("=" * 108)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"EAR file         : {args.ear_file}")
    print(f"Attribute types  : {sorted(allowed_types)}")
    print(f"Selected images  : {len(selected)}")
    print(f"Regions/image    : {len(regions)}")

    cache, samples = {}, []
    for pos, index in enumerate(selected):
        result = analyze_sample(
            train_dataset,
            entity_index,
            semantics_reader,
            allowed_types,
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

        if pos < 5:
            phrases = [
                e["attribute_phrase"]
                for e in result["entities"]
                if e["has_attribute"]
            ]
            print(f"[{pos + 1:>3}] index={index:<6} attrs={phrases[:4]}")
        elif (pos + 1) % 10 == 0:
            print(f"[{pos + 1:>3}/{len(selected)}] cache={len(cache)}")

    aggregate = build_aggregate(samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c03_entity_attribute_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "ear_file": args.ear_file,
                    "attribute_types": sorted(allowed_types),
                    "windows": args.windows,
                    "num_regions": len(regions),
                    "num_negatives": args.num_negatives,
                    "entity_text": "independent entity phrase",
                    "attribute_text": "selected EAR attribute values + entity phrase",
                    "contextual_text": "causal CLIP token-span mean pooling",
                },
                "aggregate": aggregate,
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = write_csv(output_dir, samples)
    print_aggregate(aggregate)
    print(f"\nJSON: {summary_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
