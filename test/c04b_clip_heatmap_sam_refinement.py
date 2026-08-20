import argparse
import json
import sys
from collections import deque
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
        description="C0.4B: CLIP multi-scale spatial heatmap -> aggregated box -> SAM2 -> CLIP rerank"
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
    parser.add_argument("--region-batch-size", type=int, default=128)
    parser.add_argument("--mask-batch-size", type=int, default=32)

    # Heatmap 只负责 object-level coarse localization。
    parser.add_argument("--heatmap-quantile", type=float, default=0.75)
    parser.add_argument("--dilate-kernel", type=int, default=13)
    parser.add_argument("--box-margin", type=float, default=0.10)
    parser.add_argument("--num-boxes", type=int, default=1)
    parser.add_argument("--min-component-area", type=float, default=0.01)

    parser.add_argument("--min-mask-area", type=float, default=0.001)
    parser.add_argument("--max-mask-area", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/c04b_clip_heatmap_sam/sample.png",
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


def zscore(x):
    std = x.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / std


def build_spatial_heatmap(regions, scores, image_size, windows):
    """
    每个尺度独立标准化并生成空间图，再等权融合。
    避免 32px 候选数量远多于 128px 时直接支配 heatmap。
    """
    scale_maps = {}

    for window in windows:
        indices = [
            i for i, region in enumerate(regions)
            if region["scale"] == window
        ]
        scale_scores = scores[indices].float()
        evidence = torch.relu(zscore(scale_scores))

        heat = np.zeros((image_size, image_size), dtype=np.float32)
        coverage = np.zeros_like(heat)

        for local_idx, region_idx in enumerate(indices):
            x1, y1, x2, y2 = regions[region_idx]["box"]
            weight = float(evidence[local_idx].item())
            heat[y1:y2, x1:x2] += weight
            coverage[y1:y2, x1:x2] += 1.0

        heat /= np.maximum(coverage, 1.0)

        max_value = float(heat.max())
        if max_value > 1e-8:
            heat /= max_value

        scale_maps[int(window)] = heat

    fused = np.mean(
        np.stack([scale_maps[int(window)] for window in windows], axis=0),
        axis=0,
    )
    return fused.astype(np.float32), scale_maps


def dilate_binary(mask, kernel_size):
    if kernel_size <= 1:
        return mask.astype(bool)

    if kernel_size % 2 == 0:
        kernel_size += 1

    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    dilated = F.max_pool2d(
        tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return dilated[0, 0].numpy() > 0.5


def connected_components(mask, heatmap):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            xs, ys = [], []

            while queue:
                cx, cy = queue.popleft()
                xs.append(cx)
                ys.append(cy)

                for nx, ny in (
                    (cx - 1, cy),
                    (cx + 1, cy),
                    (cx, cy - 1),
                    (cx, cy + 1),
                ):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            area = len(xs)
            mass = float(heatmap[ys, xs].sum())

            components.append(
                {
                    "box": [x1, y1, x2, y2],
                    "area": area,
                    "mass": mass,
                    "mean_heat": mass / max(area, 1),
                }
            )

    return components


def expand_box(box, image_size, margin_ratio):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_size, x2 + margin_x),
        min(image_size, y2 + margin_y),
    ]


def heatmap_to_boxes(
    heatmap,
    image_size,
    quantile,
    dilate_kernel,
    margin_ratio,
    num_boxes,
    min_component_area,
):
    if not 0.0 < quantile < 1.0:
        raise ValueError("--heatmap-quantile 必须位于 (0,1)。")

    threshold = float(np.quantile(heatmap, quantile))
    binary = heatmap >= threshold
    binary = dilate_binary(binary, dilate_kernel)

    components = connected_components(binary, heatmap)
    min_pixels = int(round(min_component_area * image_size * image_size))
    components = [
        item for item in components
        if item["area"] >= min_pixels
    ]
    components.sort(key=lambda x: x["mass"], reverse=True)

    boxes = []
    for rank, component in enumerate(components[:num_boxes], start=1):
        boxes.append(
            {
                "rank": rank,
                "box": expand_box(
                    component["box"],
                    image_size,
                    margin_ratio,
                ),
                "component_box": component["box"],
                "component_area": component["area"],
                "component_mass": component["mass"],
                "component_mean_heat": component["mean_heat"],
            }
        )

    return boxes, binary, threshold


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


def build_masked_crop(image_rgb, mask):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    if not crop_mask.any():
        return None, None

    fill = crop[crop_mask].mean(axis=0)
    crop[~crop_mask] = fill
    return crop, bbox


def preprocess_rgb_crops(crops, preprocess):
    return torch.stack(
        [preprocess(Image.fromarray(crop)) for crop in crops]
    )


@torch.no_grad()
def encode_rgb_crops(model, crops, device, batch_size):
    inputs = preprocess_rgb_crops(
        crops,
        model.backbone.preprocess_val,
    )
    features = []

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


def save_visualization(
    path,
    image_rgb,
    caption,
    query,
    raw_top_regions,
    heatmap,
    binary,
    boxes,
    candidates,
    selected,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(image_rgb)
    for rank, item in enumerate(raw_top_regions, start=1):
        x1, y1, x2, y2 = item["box"]
        axes[0, 0].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.5,
            )
        )
        axes[0, 0].text(x1, y1, str(rank), fontsize=10)
    axes[0, 0].set_title("Original Top-3 independent Regions")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_rgb)
    axes[0, 1].imshow(heatmap, alpha=0.55)
    axes[0, 1].set_title("Multi-scale CLIP spatial heatmap")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(binary)
    axes[0, 2].set_title("Threshold + dilation")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(image_rgb)
    for box_info in boxes:
        x1, y1, x2, y2 = box_info["box"]
        axes[0, 3].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
            )
        )
    axes[0, 3].set_title("Aggregated coarse box")
    axes[0, 3].axis("off")

    ordered = sorted(
        candidates,
        key=lambda x: x["mask_clip"],
        reverse=True,
    )[:3]

    for col in range(3):
        ax = axes[1, col]
        if col >= len(ordered):
            ax.axis("off")
            continue

        candidate = ordered[col]
        ax.imshow(overlay_mask(image_rgb, candidate["mask"]))
        ax.contour(candidate["mask"], levels=[0.5], linewidths=1)
        marker = " SELECTED" if candidate["candidate_id"] == selected["candidate_id"] else ""
        ax.set_title(
            f"Top{col + 1} mask{marker}\n"
            f"SAM={candidate['sam_score']:.3f}, "
            f"MaskCLIP={candidate['mask_clip']:.3f}"
        )
        ax.axis("off")

    axes[1, 3].imshow(selected["masked_crop"])
    axes[1, 3].set_title(
        f"Selected masked crop\nMaskCLIP={selected['mask_clip']:.4f}"
    )
    axes[1, 3].axis("off")

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
    checkpoint = load_clip_checkpoint(
        clip_model,
        args.clip_checkpoint,
    )
    clip_model = clip_model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=clip_model.backbone.preprocess_val,
        eval_transform=clip_model.backbone.preprocess_val,
    )

    image_tensor, caption, image_id, *_ = train_dataset[args.dataset_index]
    image_rgb = recover_rgb(
        image_tensor,
        clip_model.backbone.preprocess_val,
    )
    image_size = image_tensor.shape[-1]

    regions = generate_regions(
        image_size,
        args.windows,
    )
    region_crops = build_region_crops(
        image_tensor,
        regions,
    )

    with torch.no_grad():
        text_feature = clip_model.backbone.encode_text(
            [args.coarse_query],
            normalize=True,
        )[0].cpu()

    print("=" * 104)
    print("C0.4B CLIP SPATIAL HEATMAP -> SAM2")
    print("=" * 104)
    print(f"Dataset index   : {args.dataset_index}")
    print(f"Image ID        : {image_id}")
    print(f"Caption         : {caption}")
    print(f"Coarse query    : {args.coarse_query}")
    print(f"CLIP epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Regions         : {len(regions)}")
    print("=" * 104)

    print("\nEncoding 230 Regions...")
    region_features = encode_region_features(
        clip_model,
        region_crops,
        device,
        args.region_batch_size,
    )
    region_scores = region_features @ text_feature

    top_values, top_indices = torch.topk(
        region_scores,
        k=min(3, len(regions)),
    )
    raw_top_regions = []

    print("\nRaw Top-3 Regions:")
    for rank, (value, idx) in enumerate(
        zip(top_values.tolist(), top_indices.tolist()),
        start=1,
    ):
        item = {
            "rank": rank,
            "score": float(value),
            **regions[idx],
        }
        raw_top_regions.append(item)
        print(
            f"  Top{rank}: score={value:.4f}, "
            f"scale={item['scale']}, box={item['box']}"
        )

    heatmap, scale_maps = build_spatial_heatmap(
        regions,
        region_scores,
        image_size,
        args.windows,
    )
    boxes, binary, threshold = heatmap_to_boxes(
        heatmap,
        image_size,
        args.heatmap_quantile,
        args.dilate_kernel,
        args.box_margin,
        args.num_boxes,
        args.min_component_area,
    )

    if not boxes:
        raise RuntimeError(
            "Heatmap 未产生有效 connected component。"
            "可降低 --heatmap-quantile 或 --min-component-area。"
        )

    print(f"\nHeatmap threshold: {threshold:.4f}")
    print("Aggregated boxes:")
    for box in boxes:
        print(
            f"  Box{box['rank']}: box={box['box']}, "
            f"component={box['component_box']}, "
            f"area={box['component_area']}, "
            f"mass={box['component_mass']:.3f}"
        )

    print("\nRunning SAM2...")
    sam = SAM2Segmenter(
        checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_model_cfg,
        device=str(device),
    )
    sam.set_image(image_rgb)

    candidates = []
    candidate_id = 0

    for box_info in boxes:
        masks, sam_scores, _ = sam.predict_box(
            box_info["box"],
            multimask_output=True,
        )

        for mask_rank, (mask, sam_score) in enumerate(
            zip(masks, sam_scores),
            start=1,
        ):
            mask = np.asarray(mask, dtype=bool)
            area_ratio = float(mask.mean())

            if not args.min_mask_area <= area_ratio <= args.max_mask_area:
                continue

            masked_crop, tight_bbox = build_masked_crop(
                image_rgb,
                mask,
            )
            if masked_crop is None:
                continue

            candidate_id += 1
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "box_rank": box_info["rank"],
                    "mask_rank": mask_rank,
                    "sam_score": float(sam_score),
                    "area_ratio": area_ratio,
                    "tight_bbox": tight_bbox,
                    "mask": mask,
                    "masked_crop": masked_crop,
                }
            )

    if not candidates:
        raise RuntimeError("SAM 未产生有效 mask。")

    mask_features = encode_rgb_crops(
        clip_model,
        [item["masked_crop"] for item in candidates],
        device,
        args.mask_batch_size,
    )
    mask_scores = mask_features @ text_feature

    for i, candidate in enumerate(candidates):
        candidate["mask_clip"] = float(mask_scores[i].item())

    selected = max(
        candidates,
        key=lambda x: x["mask_clip"],
    )

    print("\nMask candidates:")
    for item in candidates:
        flag = " <-- SELECTED" if item["candidate_id"] == selected["candidate_id"] else ""
        print(
            f"  Box{item['box_rank']}/M{item['mask_rank']} | "
            f"SAM={item['sam_score']:.4f} | "
            f"area={item['area_ratio']:.4f} | "
            f"MaskCLIP={item['mask_clip']:.4f} | "
            f"bbox={item['tight_bbox']}{flag}"
        )

    output_path = Path(args.output)
    save_visualization(
        output_path,
        image_rgb,
        caption,
        args.coarse_query,
        raw_top_regions,
        heatmap,
        binary,
        boxes,
        candidates,
        selected,
    )

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_index": args.dataset_index,
                "image_id": int(image_id),
                "caption": caption,
                "coarse_query": args.coarse_query,
                "windows": args.windows,
                "heatmap_quantile": args.heatmap_quantile,
                "dilate_kernel": args.dilate_kernel,
                "box_margin": args.box_margin,
                "raw_top_regions": raw_top_regions,
                "heatmap_threshold": threshold,
                "aggregated_boxes": boxes,
                "scale_heatmap_max": {
                    str(scale): float(value.max())
                    for scale, value in scale_maps.items()
                },
                "candidates": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"mask", "masked_crop"}
                    }
                    for item in candidates
                ],
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
        f"MaskCLIP={selected['mask_clip']:.4f}"
    )
    print(f"Figure: {output_path}")
    print(f"JSON  : {json_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
