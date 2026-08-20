import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval
from models.sam import SAM2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.4: Coarse CLIP box -> SAM2 mask -> CLIP coarse rerank"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--clip-checkpoint", type=str, required=True)
    parser.add_argument("--sam-checkpoint", type=str, required=True)
    parser.add_argument(
        "--sam-model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--coarse-query", type=str, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--top-boxes", type=int, default=3)
    parser.add_argument("--region-batch-size", type=int, default=128)
    parser.add_argument("--mask-batch-size", type=int, default=32)
    parser.add_argument("--min-mask-area", type=float, default=0.001)
    parser.add_argument("--max-mask-area", type=float, default=0.60)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/c04_clip_sam_region_refinement/sample.png",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_clip_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return checkpoint


def recover_rgb(image_tensor, preprocess):
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
            raise ValueError(f"非法 window={window}")

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


def build_region_crops(image_tensor, regions):
    image_size = image_tensor.shape[-1]
    crops = []

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        crop = image_tensor[:, y1:y2, x1:x2].unsqueeze(0)
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
def encode_region_features(model, crops, device, batch_size):
    features = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size].to(device, non_blocking=True)
        features.append(
            model.backbone.encode_image(batch, normalize=True).cpu()
        )
    return torch.cat(features)


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]


def build_sam_box_crop(image_rgb, mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    return crop, bbox


def build_sam_mask_crop(image_rgb, mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    if not crop_mask.any():
        return None, None

    # mask 外区域用目标像素均值填充，减少纯黑人工边界。
    fill = crop[crop_mask].mean(axis=0)
    crop[~crop_mask] = fill
    return crop, bbox


def preprocess_rgb_crops(crops, preprocess):
    return torch.stack(
        [preprocess(Image.fromarray(crop)) for crop in crops]
    )


@torch.no_grad()
def encode_rgb_crops(model, crops, device, batch_size):
    if not crops:
        return None

    features = []
    inputs = preprocess_rgb_crops(crops, model.backbone.preprocess_val)

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(device, non_blocking=True)
        features.append(
            model.backbone.encode_image(batch, normalize=True).cpu()
        )
    return torch.cat(features)


def overlay_mask(image, mask):
    canvas = image.astype(np.float32).copy()
    canvas[mask] = 0.55 * canvas[mask] + 0.45 * 255.0
    return canvas.clip(0, 255).astype(np.uint8)


def save_visualization(path, image_rgb, caption, query, boxes, candidates, selected):
    rows = max(len(boxes), 1)
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.3 * rows), squeeze=False)

    by_box = {}
    for candidate in candidates:
        by_box.setdefault(candidate["box_rank"], []).append(candidate)

    for row, box_info in enumerate(boxes):
        ax = axes[row, 0]
        ax.imshow(image_rgb)
        x1, y1, x2, y2 = box_info["box"]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        )
        ax.set_title(
            f"CLIP Box {row + 1}\n"
            f"score={box_info['clip_score']:.4f}, {box_info['scale']}px"
        )
        ax.axis("off")

        box_candidates = by_box.get(row + 1, [])
        for col in range(1, 4):
            ax = axes[row, col]
            if col - 1 >= len(box_candidates):
                ax.axis("off")
                continue

            candidate = box_candidates[col - 1]
            ax.imshow(overlay_mask(image_rgb, candidate["mask"]))
            ax.contour(candidate["mask"], levels=[0.5], linewidths=1)
            marker = "  SELECTED" if candidate["candidate_id"] == selected["candidate_id"] else ""
            ax.set_title(
                f"M{candidate['mask_rank']} | SAM={candidate['sam_score']:.3f}\n"
                f"BoxCLIP={candidate['sam_box_clip']:.3f} | "
                f"MaskCLIP={candidate['sam_mask_clip']:.3f}{marker}"
            )
            ax.axis("off")

        ax = axes[row, 4]
        if box_candidates:
            best = max(box_candidates, key=lambda x: x["sam_mask_clip"])
            ax.imshow(best["masked_crop"])
            ax.set_title(
                f"Best masked crop in Box {row + 1}\n"
                f"MaskCLIP={best['sam_mask_clip']:.4f}"
            )
        ax.axis("off")

    fig.suptitle(f'{caption}\nCoarse query: "{query}"')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    clip_model = CLIPRetrieval(config["model"])
    checkpoint = load_clip_checkpoint(clip_model, args.clip_checkpoint)
    clip_model = clip_model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=clip_model.backbone.preprocess_val,
        eval_transform=clip_model.backbone.preprocess_val,
    )

    if not 0 <= args.dataset_index < len(train_dataset):
        raise IndexError(
            f"dataset_index={args.dataset_index} 超出范围 [0, {len(train_dataset)-1}]"
        )

    image_tensor, caption, image_id, *_ = train_dataset[args.dataset_index]
    image_rgb = recover_rgb(image_tensor, clip_model.backbone.preprocess_val)
    image_size = image_tensor.shape[-1]

    regions = generate_regions(image_size, args.windows)
    region_crops = build_region_crops(image_tensor, regions)

    print("=" * 104)
    print("C0.4 CLIP -> SAM2 REGION REFINEMENT")
    print("=" * 104)
    print(f"Dataset index   : {args.dataset_index}")
    print(f"Image ID        : {image_id}")
    print(f"Caption         : {caption}")
    print(f"Coarse query    : {args.coarse_query}")
    print(f"CLIP checkpoint : {args.clip_checkpoint}")
    print(f"CLIP epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Regions         : {len(regions)}")
    print("=" * 104)

    print("\nEncoding coarse query and 230 Regions...")
    with torch.no_grad():
        text_feature = clip_model.backbone.encode_text(
            [args.coarse_query], normalize=True
        )[0].cpu()

    region_features = encode_region_features(
        clip_model,
        region_crops,
        device,
        args.region_batch_size,
    )
    region_scores = region_features @ text_feature

    top_box_values, top_box_indices = torch.topk(
        region_scores,
        k=min(args.top_boxes, len(regions)),
    )

    boxes = []
    for rank, (score, region_id) in enumerate(
        zip(top_box_values.tolist(), top_box_indices.tolist()), start=1
    ):
        region = regions[region_id]
        boxes.append(
            {
                "rank": rank,
                "region_id": int(region_id),
                "scale": region["scale"],
                "box": region["box"],
                "clip_score": float(score),
            }
        )
        print(
            f"Box {rank}: CLIP={score:.4f}, "
            f"scale={region['scale']}, box={region['box']}"
        )

    print("\nRunning SAM2...")
    sam = SAM2Segmenter(
        checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_model_cfg,
        device=str(device),
    )
    sam.set_image(image_rgb)

    raw_candidates = []
    candidate_id = 0

    for box_info in boxes:
        masks, sam_scores, _ = sam.predict_box(
            box_info["box"],
            multimask_output=True,
        )

        for mask_rank, (mask, sam_score) in enumerate(
            zip(masks, sam_scores), start=1
        ):
            mask = np.asarray(mask, dtype=bool)
            area_ratio = float(mask.mean())

            if not args.min_mask_area <= area_ratio <= args.max_mask_area:
                continue

            box_crop, tight_bbox = build_sam_box_crop(image_rgb, mask)
            masked_crop, _ = build_sam_mask_crop(image_rgb, mask)

            if box_crop is None or masked_crop is None:
                continue

            candidate_id += 1
            raw_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "box_rank": box_info["rank"],
                    "mask_rank": mask_rank,
                    "sam_score": float(sam_score),
                    "area_ratio": area_ratio,
                    "tight_bbox": tight_bbox,
                    "mask": mask,
                    "box_crop": box_crop,
                    "masked_crop": masked_crop,
                }
            )

    if not raw_candidates:
        raise RuntimeError("SAM 没有产生通过面积过滤的有效 mask。")

    sam_box_features = encode_rgb_crops(
        clip_model,
        [x["box_crop"] for x in raw_candidates],
        device,
        args.mask_batch_size,
    )
    sam_mask_features = encode_rgb_crops(
        clip_model,
        [x["masked_crop"] for x in raw_candidates],
        device,
        args.mask_batch_size,
    )

    sam_box_scores = sam_box_features @ text_feature
    sam_mask_scores = sam_mask_features @ text_feature

    for i, candidate in enumerate(raw_candidates):
        candidate["sam_box_clip"] = float(sam_box_scores[i].item())
        candidate["sam_mask_clip"] = float(sam_mask_scores[i].item())

    # 主选择只看 coarse text 与 masked crop 的 CLIP 相似度，不使用 fine phrase。
    selected = max(raw_candidates, key=lambda x: x["sam_mask_clip"])

    print("\nMask candidates:")
    for candidate in raw_candidates:
        flag = " <-- SELECTED" if candidate["candidate_id"] == selected["candidate_id"] else ""
        print(
            f"Box{candidate['box_rank']}/M{candidate['mask_rank']} | "
            f"SAM={candidate['sam_score']:.4f} | "
            f"area={candidate['area_ratio']:.4f} | "
            f"BoxCLIP={candidate['sam_box_clip']:.4f} | "
            f"MaskCLIP={candidate['sam_mask_clip']:.4f} | "
            f"bbox={candidate['tight_bbox']}{flag}"
        )

    output_path = Path(args.output)
    save_visualization(
        output_path,
        image_rgb,
        caption,
        args.coarse_query,
        boxes,
        raw_candidates,
        selected,
    )

    json_path = output_path.with_suffix(".json")
    serializable_candidates = []
    for candidate in raw_candidates:
        serializable_candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"mask", "box_crop", "masked_crop"}
            }
        )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_index": args.dataset_index,
                "image_id": int(image_id),
                "caption": caption,
                "coarse_query": args.coarse_query,
                "boxes": boxes,
                "candidates": serializable_candidates,
                "selected_candidate_id": selected["candidate_id"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 104)
    print(
        f"Selected: Box{selected['box_rank']}/M{selected['mask_rank']} | "
        f"SAM={selected['sam_score']:.4f} | "
        f"MaskCLIP={selected['sam_mask_clip']:.4f}"
    )
    print(f"Figure: {output_path}")
    print(f"JSON  : {json_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
