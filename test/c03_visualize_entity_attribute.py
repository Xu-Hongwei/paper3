import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from datasets.structured_semantics import StructuredSemanticsReader
from models import CLIPRetrieval


DEFAULT_ATTRIBUTE_TYPES = ["count", "color", "size", "shape", "state"]
ATTRIBUTE_ORDER = {"count": 0, "size": 1, "color": 2, "shape": 3, "state": 4}


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.3 visualization: Entity vs Entity+Attribute vs Contextual."
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
        default="outputs/c03_visual_review",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="指定训练集样本 index；不指定则自动选有有效 Attribute 的不同图像。",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument(
        "--attribute-types",
        type=str,
        nargs="+",
        default=DEFAULT_ATTRIBUTE_TYPES,
    )
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="也显示 Entity+Attr 文本和原 Entity 完全相同的实体。",
    )
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def load_entity_index(dataset):
    if not dataset.entity_index_file:
        raise ValueError("需要 Entity Index v2。")

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

        if not value or value_lower in entity_lower:
            continue
        if value_lower not in {x.lower() for x in values}:
            values.append(value)

    return " ".join(values + [entity]).strip()


def bind_structured_entities(
    dataset,
    entity_index,
    semantics_reader,
    dataset_index,
    allowed_types,
):
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
        phrase = build_attribute_phrase(
            span_record["text"],
            attributes,
        )

        records.append(
            {
                "text": span_record["text"],
                "span": span_record["span"],
                "attributes": attributes,
                "attribute_phrase": phrase,
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
    image_size = image.shape[-1]
    crops = []

    for region in regions:
        x1, y1, x2, y2 = region["box"]
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
def encode_regions(model, image, regions, device, batch_size):
    crops = build_region_crops(image, regions)
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


def pool_contextual_spans(token_features, records):
    pooled = []

    for record in records:
        start, end = record["span"]
        pooled.append(
            token_features[0, start:end].mean(dim=0)
        )

    return F.normalize(
        torch.stack(pooled),
        dim=-1,
    )


@torch.no_grad()
def encode_text_variants(model, caption, records):
    entity_texts = [record["text"] for record in records]
    attribute_phrases = [record["attribute_phrase"] for record in records]

    entity_features = model.backbone.encode_text(
        entity_texts,
        normalize=True,
    ).cpu()

    attribute_features = model.backbone.encode_text(
        attribute_phrases,
        normalize=True,
    ).cpu()

    _, token_features = model.backbone.encode_text_with_tokens(
        [caption],
        normalize=False,
    )
    contextual_features = pool_contextual_spans(
        token_features,
        records,
    ).cpu()

    return {
        "Entity": entity_features,
        "Entity+Attr": attribute_features,
        "Contextual": contextual_features,
    }


def recover_input_image(image_tensor, preprocess):
    mean = None
    std = None

    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            mean = torch.as_tensor(transform.mean).view(3, 1, 1)
            std = torch.as_tensor(transform.std).view(3, 1, 1)
            break

    image = image_tensor.detach().cpu()

    if mean is not None and std is not None:
        image = image * std + mean

    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def select_samples(
    dataset,
    entity_index,
    semantics_reader,
    allowed_types,
    num_samples,
    indices,
    seed,
    include_unchanged,
):
    def visible_records(index):
        records = bind_structured_entities(
            dataset,
            entity_index,
            semantics_reader,
            index,
            allowed_types,
        )
        if include_unchanged:
            return [r for r in records if r["attributes"]]
        return [
            r
            for r in records
            if r["attributes"] and r["attribute_phrase"].lower() != r["text"].lower()
        ]

    if indices:
        selected = []
        for index in indices:
            if not 0 <= index < len(dataset):
                raise IndexError(index)
            if visible_records(index):
                selected.append(index)
        return selected

    rng = random.Random(seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    selected = []
    used_images = set()

    for index in candidates:
        image_key = dataset.ann[index]["image"]
        if image_key in used_images or not visible_records(index):
            continue

        selected.append(index)
        used_images.add(image_key)

        if len(selected) >= num_samples:
            break

    return selected


def get_top_regions(scores, regions, topk):
    k = min(topk, scores.shape[0])
    values, indices = torch.topk(scores, k=k)

    result = []
    for rank in range(k):
        region_index = int(indices[rank].item())
        result.append(
            {
                "rank": rank + 1,
                "region_index": region_index,
                "similarity": float(values[rank].item()),
                "scale": regions[region_index]["scale"],
                "box": regions[region_index]["box"],
            }
        )
    return result


def analyze_entity(entity_pos, record, text_features, region_features, regions, topk):
    variants = {}

    for name, features in text_features.items():
        scores = features[entity_pos] @ region_features.t()
        top_regions = get_top_regions(
            scores,
            regions,
            topk,
        )
        margin = (
            top_regions[0]["similarity"] - top_regions[1]["similarity"]
            if len(top_regions) >= 2
            else 0.0
        )

        variants[name] = {
            "top_regions": top_regions,
            "top1_similarity": top_regions[0]["similarity"],
            "top1_scale": top_regions[0]["scale"],
            "top1_box": top_regions[0]["box"],
            "top1_top2_margin": margin,
        }

    return {
        "entity": record["text"],
        "attribute_phrase": record["attribute_phrase"],
        "attributes": record["attributes"],
        "span": record["span"],
        "variants": variants,
    }


def draw_box(ax, image, region, title):
    ax.imshow(image)

    x1, y1, x2, y2 = region["box"]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
    )
    ax.text(
        x1,
        y1,
        "1",
        fontsize=10,
    )
    ax.set_title(title)
    ax.axis("off")


def save_overview(path, image, caption, entity_results):
    rows = len(entity_results)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(13, 3.4 * rows),
        squeeze=False,
    )

    for row, result in enumerate(entity_results):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Entity: {result['entity']}\n"
            f"Attr: {result['attribute_phrase']}"
        )
        axes[row, 0].axis("off")

        for column, name in enumerate(
            ("Entity", "Entity+Attr", "Contextual"),
            start=1,
        ):
            variant = result["variants"][name]
            region = variant["top_regions"][0]

            draw_box(
                axes[row, column],
                image,
                region,
                f"{name}\n"
                f"s={region['similarity']:.3f}, "
                f"{region['scale']}px",
            )

    fig.suptitle(
        textwrap.fill(caption, width=105),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_entity_detail(path, image, caption, result, topk):
    fig, axes = plt.subplots(
        3,
        topk + 1,
        figsize=(3.3 * (topk + 1), 10),
        squeeze=False,
    )

    for row, name in enumerate(
        ("Entity", "Entity+Attr", "Contextual")
    ):
        variant = result["variants"][name]
        top_regions = variant["top_regions"]

        axes[row, 0].imshow(image)
        for region in top_regions:
            x1, y1, x2, y2 = region["box"]
            axes[row, 0].add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=1.5,
                )
            )
            axes[row, 0].text(
                x1,
                y1,
                str(region["rank"]),
                fontsize=9,
            )

        axes[row, 0].set_title(
            f"{name}\n"
            f"Top1={variant['top1_similarity']:.3f}, "
            f"margin={variant['top1_top2_margin']:.3f}"
        )
        axes[row, 0].axis("off")

        for column, region in enumerate(top_regions, start=1):
            x1, y1, x2, y2 = region["box"]
            crop = image[y1:y2, x1:x2]

            axes[row, column].imshow(crop)
            axes[row, column].set_title(
                f"Top{region['rank']}\n"
                f"s={region['similarity']:.3f}, "
                f"{region['scale']}px\n"
                f"{region['box']}"
            )
            axes[row, column].axis("off")

    attr_text = ", ".join(
        f"{attr['type']}={attr['value']}"
        for attr in result["attributes"]
    )
    title = (
        f"Caption: {caption}\n"
        f"Entity: {result['entity']}    |    "
        f"Entity+Attr: {result['attribute_phrase']}    |    "
        f"Attributes: {attr_text}"
    )

    fig.suptitle(
        textwrap.fill(title, width=125),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def safe_name(text):
    keep = []
    for char in text.lower():
        keep.append(char if char.isalnum() else "_")
    return "".join(keep).strip("_")[:50] or "entity"


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
    regions = generate_regions(
        image_size,
        args.windows,
    )
    allowed_types = set(args.attribute_types)

    selected = select_samples(
        train_dataset,
        entity_index,
        semantics_reader,
        allowed_types,
        args.num_samples,
        args.indices,
        args.seed,
        args.include_unchanged,
    )
    if not selected:
        raise RuntimeError("没有找到可视化所需的有效 Attribute 样本。")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("C0.3 VISUAL REVIEW")
    print("=" * 96)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"EAR file         : {args.ear_file}")
    print(f"Samples          : {selected}")
    print(f"Top-K            : {args.topk}")
    print(f"Regions/image    : {len(regions)}")

    manifest = []

    for sample_pos, index in enumerate(selected, start=1):
        image, caption, image_id, _ = train_dataset[index]

        records = bind_structured_entities(
            train_dataset,
            entity_index,
            semantics_reader,
            index,
            allowed_types,
        )
        if not args.include_unchanged:
            records = [
                record
                for record in records
                if (
                    record["attributes"]
                    and record["attribute_phrase"].lower()
                    != record["text"].lower()
                )
            ]
        else:
            records = [
                record
                for record in records
                if record["attributes"]
            ]

        region_features = encode_regions(
            model,
            image,
            regions,
            device,
            args.region_batch_size,
        )
        text_features = encode_text_variants(
            model,
            caption,
            records,
        )

        entity_results = [
            analyze_entity(
                entity_pos,
                record,
                text_features,
                region_features,
                regions,
                args.topk,
            )
            for entity_pos, record in enumerate(records)
        ]

        display_image = recover_input_image(
            image,
            model.backbone.preprocess_val,
        )

        sample_dir = output_dir / f"sample_{index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        overview_path = sample_dir / "overview.png"
        save_overview(
            overview_path,
            display_image,
            caption,
            entity_results,
        )

        detail_paths = []
        for entity_pos, result in enumerate(entity_results):
            detail_path = (
                sample_dir
                / f"{entity_pos:02d}_{safe_name(result['entity'])}.png"
            )
            save_entity_detail(
                detail_path,
                display_image,
                caption,
                result,
                args.topk,
            )
            detail_paths.append(str(detail_path))

        record = {
            "dataset_index": int(index),
            "image_id": int(image_id),
            "image": train_dataset.ann[index]["image"],
            "caption": caption,
            "overview": str(overview_path),
            "details": detail_paths,
            "entities": entity_results,
        }
        manifest.append(record)

        print(
            f"[{sample_pos:>2}/{len(selected)}] "
            f"index={index:<6} "
            f"entities={len(entity_results):<2} "
            f"overview={overview_path}"
        )

    manifest_path = output_dir / "visual_review_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "ear_file": args.ear_file,
                    "attribute_types": sorted(allowed_types),
                    "windows": args.windows,
                    "topk": args.topk,
                    "num_regions": len(regions),
                },
                "samples": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinished.")
    print(f"Visuals  : {output_dir}")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
