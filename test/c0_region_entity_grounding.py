import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Rectangle

# 确保优先导入项目内 datasets / models，而不是 site-packages 同名包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval


DEFAULT_KEYWORDS = [
    "stadium",
    "football field",
    "airport",
    "runway",
    "airplane",
    "ship",
    "port",
    "bridge",
    "river",
    "pond",
    "storage tanks",
    "buildings",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0: Clean CLIP Region-Entity Grounding diagnostic."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline/rsicd.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/clip_rsicd_10ep/best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/c0_region_entity_grounding",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="指定训练集有效样本 index；不指定则自动选择典型多实体样本。",
    )
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--topk", type=int, default=5)
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
    """生成覆盖完整 224×224 eval 空间的滑窗起点，并补齐右/下边界。"""
    if window <= 0 or window > image_size:
        raise ValueError(f"Invalid window={window} for image_size={image_size}")

    positions = list(range(0, image_size - window + 1, stride))
    last = image_size - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_region_boxes(image_size, windows):
    """
    生成 multi-scale region boxes。

    box 格式统一为 [x1, y1, x2, y2)，坐标位于 CLIP 224×224 eval 空间。
    """
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


def build_region_crops(image, regions, output_size):
    """
    从已经进入 CLIP eval 空间的图像 Tensor 上裁剪 Region，
    每个 Region 再 resize 到完整 CLIP 输入尺寸。
    """
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
    """分 batch 使用完整 Clean CLIP Vision Encoder 编码所有 Region。"""
    all_features = []

    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size].to(
            device,
            non_blocking=True,
        )
        features = model.backbone.encode_image(
            batch,
            normalize=True,
        )
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def recover_input_image(image_tensor, preprocess):
    """反归一化当前 224×224 eval 输入图，仅用于可视化。"""
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


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter = inter_w * inter_h

    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def select_samples(dataset, num_samples, indices):
    """
    优先选择包含典型遥感实体的不同图像；
    fallback 优先选择 Entity 数量 >= 2 的样本。
    """
    if indices:
        selected = []
        for index in indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(f"Invalid dataset index: {index}")
            entities = dataset.get_entity_texts(index)
            if not entities:
                raise ValueError(f"Dataset index {index} has no Entity text.")
            selected.append(index)
        return selected

    selected = []
    used_images = set()

    for keyword in DEFAULT_KEYWORDS:
        for index, ann in enumerate(dataset.ann):
            if len(selected) >= num_samples:
                return selected

            if ann["image"] in used_images:
                continue

            entities = dataset.get_entity_texts(index)
            if len(entities) < 2:
                continue

            normalized = [entity.lower() for entity in entities]
            if not any(keyword in entity for entity in normalized):
                continue

            selected.append(index)
            used_images.add(ann["image"])
            break

    if len(selected) < num_samples:
        candidates = list(range(len(dataset)))
        random.shuffle(candidates)

        for index in candidates:
            if len(selected) >= num_samples:
                break

            ann = dataset.ann[index]
            if ann["image"] in used_images:
                continue

            entities = dataset.get_entity_texts(index)
            if len(entities) < 2:
                continue

            selected.append(index)
            used_images.add(ann["image"])

    return selected


@torch.no_grad()
def analyze_sample(
    model,
    image,
    caption,
    entities,
    regions,
    device,
    region_batch_size,
    topk,
):
    """完成一个样本的 Global / Region Entity 相似度诊断。"""
    image_size = image.shape[-1]

    global_feature = model.backbone.encode_image(
        image.unsqueeze(0).to(device),
        normalize=True,
    ).cpu()

    entity_features = model.backbone.encode_text(
        entities,
        normalize=True,
    ).cpu()

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

    global_scores = (entity_features @ global_feature.t()).squeeze(1)
    region_scores = entity_features @ region_features.t()

    k = min(topk, region_scores.shape[1])
    top_values, top_indices = torch.topk(
        region_scores,
        k=k,
        dim=1,
    )

    entity_results = []

    for entity_index, entity in enumerate(entities):
        best_region_score = float(top_values[entity_index, 0].item())
        global_score = float(global_scores[entity_index].item())

        if k >= 2:
            margin = float(
                top_values[entity_index, 0].item()
                - top_values[entity_index, 1].item()
            )
        else:
            margin = 0.0

        top_regions = []
        for rank in range(k):
            region_index = int(top_indices[entity_index, rank].item())
            metadata = regions[region_index]

            top_regions.append(
                {
                    "rank": rank + 1,
                    "region_index": region_index,
                    "scale": metadata["scale"],
                    "stride": metadata["stride"],
                    "box": metadata["box"],
                    "similarity": float(
                        top_values[entity_index, rank].item()
                    ),
                }
            )

        entity_results.append(
            {
                "text": entity,
                "global_similarity": global_score,
                "best_region_similarity": best_region_score,
                "region_gain": best_region_score - global_score,
                "top1_top2_margin": margin,
                "top_regions": top_regions,
            }
        )

    pairwise_top1_iou = []
    for i in range(len(entity_results)):
        for j in range(i + 1, len(entity_results)):
            box_i = entity_results[i]["top_regions"][0]["box"]
            box_j = entity_results[j]["top_regions"][0]["box"]

            pairwise_top1_iou.append(
                {
                    "entity_a": entity_results[i]["text"],
                    "entity_b": entity_results[j]["text"],
                    "iou": box_iou(box_i, box_j),
                }
            )

    return {
        "caption": caption,
        "num_regions": len(regions),
        "entities": entity_results,
        "pairwise_top1_iou": pairwise_top1_iou,
    }


def draw_entity_row(axes, image, entity_result, topk):
    """每个 Entity 一行：原图 Top-K boxes + Top-K Region crops。"""
    ax = axes[0]
    ax.imshow(image)

    for region in entity_result["top_regions"][:topk]:
        x1, y1, x2, y2 = region["box"]
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.5,
            )
        )
        ax.text(
            x1,
            y1,
            str(region["rank"]),
            fontsize=9,
        )

    ax.set_title(
        f"{entity_result['text']}\n"
        f"G={entity_result['global_similarity']:.3f}, "
        f"R={entity_result['best_region_similarity']:.3f}, "
        f"Δ={entity_result['region_gain']:+.3f}, "
        f"M={entity_result['top1_top2_margin']:.3f}"
    )
    ax.axis("off")

    for column, region in enumerate(
        entity_result["top_regions"][:topk],
        start=1,
    ):
        x1, y1, x2, y2 = region["box"]
        crop = image[y1:y2, x1:x2]

        axes[column].imshow(crop)
        axes[column].set_title(
            f"Top{region['rank']}  "
            f"s={region['similarity']:.3f}\n"
            f"{region['scale']}px {region['box']}"
        )
        axes[column].axis("off")


def visualize_sample(output_path, image, sample_result, topk):
    """保存一个样本的 Region–Entity Top-K 可视化。"""
    rows = len(sample_result["entities"])
    columns = topk + 1

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.4 * rows),
        squeeze=False,
    )

    for row, entity_result in enumerate(sample_result["entities"]):
        draw_entity_row(
            axes[row],
            image,
            entity_result,
            topk,
        )

    fig.suptitle(sample_result["caption"], fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_sample_result(sample_index, result):
    print("\n" + "-" * 88)
    print(f"Sample  : {sample_index}")
    print(f"Caption : {result['caption']}")
    print(f"Regions : {result['num_regions']}")

    for entity in result["entities"]:
        top1 = entity["top_regions"][0]
        print(
            f"{entity['text']:<24} "
            f"G={entity['global_similarity']:+.4f} | "
            f"R={entity['best_region_similarity']:+.4f} | "
            f"Gain={entity['region_gain']:+.4f} | "
            f"Margin={entity['top1_top2_margin']:.4f} | "
            f"Top1={top1['scale']:>3}px {top1['box']}"
        )


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("C0 CLEAN CLIP REGION-ENTITY GROUNDING")
    print("=" * 88)
    print(f"Config            : {args.config}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Device            : {device}")
    print(f"Windows           : {args.windows}")
    print(f"Top-K             : {args.topk}")
    print(f"Region batch size : {args.region_batch_size}")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    # C0 使用固定 eval view，避免训练增强影响 Region 坐标。
    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    image_size = int(config["dataset"].get("image_res", 224))
    regions = generate_region_boxes(
        image_size=image_size,
        windows=args.windows,
    )

    selected = select_samples(
        dataset=train_dataset,
        num_samples=args.num_samples,
        indices=args.indices,
    )

    if not selected:
        raise RuntimeError("No valid samples selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint epoch  : {checkpoint.get('epoch', 'unknown')}")
    print(f"Candidate regions : {len(regions)}")
    print(f"Selected samples  : {selected}")

    records = []

    for index in selected:
        image, caption, image_id, _ = train_dataset[index]
        entities = train_dataset.get_entity_texts(index)

        result = analyze_sample(
            model=model,
            image=image,
            caption=caption,
            entities=entities,
            regions=regions,
            device=device,
            region_batch_size=args.region_batch_size,
            topk=args.topk,
        )

        result["dataset_index"] = index
        result["pair_index"] = int(train_dataset.ann[index]["_pair_index"])
        result["image_id"] = int(image_id)
        result["image"] = train_dataset.ann[index]["image"]

        print_sample_result(index, result)

        display_image = recover_input_image(
            image,
            model.backbone.preprocess_val,
        )
        image_path = output_dir / f"sample_{index:05d}.png"
        visualize_sample(
            output_path=image_path,
            image=display_image,
            sample_result=result,
            topk=args.topk,
        )

        result["visualization"] = str(image_path)
        records.append(result)

    summary = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "windows": args.windows,
            "topk": args.topk,
            "image_size": image_size,
            "num_candidate_regions": len(regions),
            "region_encoding": "full_clean_clip_vision_encoder",
            "entity_encoding": "independent_clean_clip_text_encoder",
            "nms": False,
            "training": False,
        },
        "samples": records,
    }

    summary_path = output_dir / "c0_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("C0 FINISHED")
    print("=" * 88)
    print(f"Samples     : {len(records)}")
    print(f"Visuals     : {output_dir}")
    print(f"Summary     : {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
