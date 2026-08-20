import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval
from models.sam import SAM2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.4C: CLIP spatial peaks -> SAM2 multi-point prompts"
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

    parser.add_argument("--point-windows", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--max-points", type=int, default=3)
    parser.add_argument("--min-point-distance", type=float, default=40.0)
    parser.add_argument("--region-batch-size", type=int, default=128)
    parser.add_argument("--mask-batch-size", type=int, default=32)

    parser.add_argument("--min-mask-area", type=float, default=0.001)
    parser.add_argument("--max-mask-area", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/c04c_clip_points_sam/sample.png",
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
                        "center": [
                            (x1 + x1 + window) / 2.0,
                            (y1 + y1 + window) / 2.0,
                        ],
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


def select_spatial_points(regions, scores, max_points, min_distance):
    order = torch.argsort(scores, descending=True).tolist()
    selected = []

    for region_idx in order:
        region = regions[region_idx]
        x, y = region["center"]

        if any(
            math.hypot(x - item["point"][0], y - item["point"][1]) < min_distance
            for item in selected
        ):
            continue

        selected.append(
            {
                "region_id": int(region_idx),
                "scale": region["scale"],
                "box": region["box"],
                "point": [float(x), float(y)],
                "clip_score": float(scores[region_idx].item()),
            }
        )

        if len(selected) >= max_points:
            break

    return selected


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


@torch.no_grad()
def encode_rgb_crops(model, crops, device, batch_size):
    inputs = torch.stack(
        [
            model.backbone.preprocess_val(Image.fromarray(crop))
            for crop in crops
        ]
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


def draw_points(ax, points):
    for i, item in enumerate(points, start=1):
        x, y = item["point"]
        ax.scatter([x], [y], s=80, marker="*", edgecolors="black")
        ax.text(x + 3, y + 3, str(i), fontsize=10)


def save_visualization(path, image_rgb, caption, query, points, results):
    rows = len(results)
    fig, axes = plt.subplots(
        rows,
        5,
        figsize=(19, 4.3 * rows),
        squeeze=False,
    )

    for row, result in enumerate(results):
        active_points = points[:result["num_points"]]

        axes[row, 0].imshow(image_rgb)
        draw_points(axes[row, 0], active_points)
        axes[row, 0].set_title(
            f"{result['num_points']} positive point(s)"
        )
        axes[row, 0].axis("off")

        for col in range(1, 4):
            ax = axes[row, col]
            candidate_idx = col - 1

            if candidate_idx >= len(result["candidates"]):
                ax.axis("off")
                continue

            candidate = result["candidates"][candidate_idx]
            ax.imshow(overlay_mask(image_rgb, candidate["mask"]))
            ax.contour(candidate["mask"], levels=[0.5], linewidths=1)
            marker = " SELECTED" if candidate["is_best"] else ""
            ax.set_title(
                f"M{candidate['mask_rank']}{marker}\n"
                f"SAM={candidate['sam_score']:.3f}, "
                f"CLIP={candidate['mask_clip']:.3f}"
            )
            ax.axis("off")

        axes[row, 4].imshow(result["best"]["masked_crop"])
        axes[row, 4].set_title(
            f"Best for {result['num_points']} point(s)\n"
            f"MaskCLIP={result['best']['mask_clip']:.4f}"
        )
        axes[row, 4].axis("off")

    fig.suptitle(
        f'{caption}\nCoarse query: "{query}" | CLIP spatial peaks -> SAM points'
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if args.max_points <= 0:
        raise ValueError("--max-points 必须 > 0。")

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
        args.point_windows,
    )
    crops = build_region_crops(
        image_tensor,
        regions,
    )

    with torch.no_grad():
        text_feature = clip_model.backbone.encode_text(
            [args.coarse_query],
            normalize=True,
        )[0].cpu()

    print("=" * 104)
    print("C0.4C CLIP SPATIAL PEAKS -> SAM2 MULTI-POINT")
    print("=" * 104)
    print(f"Dataset index      : {args.dataset_index}")
    print(f"Image ID           : {image_id}")
    print(f"Caption            : {caption}")
    print(f"Coarse query       : {args.coarse_query}")
    print(f"CLIP epoch         : {checkpoint.get('epoch', 'unknown')}")
    print(f"Point windows      : {args.point_windows}")
    print(f"Min point distance : {args.min_point_distance}")
    print("=" * 104)

    print("\nEncoding local Regions...")
    region_features = encode_region_features(
        clip_model,
        crops,
        device,
        args.region_batch_size,
    )
    scores = region_features @ text_feature

    points = select_spatial_points(
        regions,
        scores,
        args.max_points,
        args.min_point_distance,
    )

    if not points:
        raise RuntimeError("没有选出有效 CLIP point。")

    print("\nSelected spatial points:")
    for i, item in enumerate(points, start=1):
        print(
            f"  P{i}: point={item['point']}, "
            f"score={item['clip_score']:.4f}, "
            f"scale={item['scale']}, box={item['box']}"
        )

    sam = SAM2Segmenter(
        checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_model_cfg,
        device=str(device),
    )
    sam.set_image(image_rgb)

    results = []

    for num_points in range(1, len(points) + 1):
        active = points[:num_points]
        point_coords = np.asarray(
            [item["point"] for item in active],
            dtype=np.float32,
        )
        point_labels = np.ones(num_points, dtype=np.int32)

        masks, sam_scores, _ = sam.predict_points(
            point_coords,
            point_labels,
            multimask_output=True,
        )

        raw_candidates = []

        for mask_rank, (mask, sam_score) in enumerate(
            zip(masks, sam_scores),
            start=1,
        ):
            mask = np.asarray(mask, dtype=bool)
            area_ratio = float(mask.mean())

            if not args.min_mask_area <= area_ratio <= args.max_mask_area:
                continue

            masked_crop, bbox = build_masked_crop(
                image_rgb,
                mask,
            )
            if masked_crop is None:
                continue

            raw_candidates.append(
                {
                    "mask_rank": mask_rank,
                    "sam_score": float(sam_score),
                    "area_ratio": area_ratio,
                    "tight_bbox": bbox,
                    "mask": mask,
                    "masked_crop": masked_crop,
                }
            )

        if not raw_candidates:
            print(f"{num_points} point(s): 无有效 mask，跳过。")
            continue

        mask_features = encode_rgb_crops(
            clip_model,
            [item["masked_crop"] for item in raw_candidates],
            device,
            args.mask_batch_size,
        )
        mask_scores = mask_features @ text_feature

        for i, candidate in enumerate(raw_candidates):
            candidate["mask_clip"] = float(mask_scores[i].item())

        best = max(
            raw_candidates,
            key=lambda x: x["mask_clip"],
        )

        for candidate in raw_candidates:
            candidate["is_best"] = candidate is best

        results.append(
            {
                "num_points": num_points,
                "candidates": raw_candidates,
                "best": best,
            }
        )

        print(f"\n{num_points} positive point(s):")
        for candidate in raw_candidates:
            flag = " <-- BEST" if candidate["is_best"] else ""
            print(
                f"  M{candidate['mask_rank']} | "
                f"SAM={candidate['sam_score']:.4f} | "
                f"area={candidate['area_ratio']:.4f} | "
                f"MaskCLIP={candidate['mask_clip']:.4f} | "
                f"bbox={candidate['tight_bbox']}{flag}"
            )

    if not results:
        raise RuntimeError("所有 point 配置都没有产生有效 mask。")

    output_path = Path(args.output)
    save_visualization(
        output_path,
        image_rgb,
        caption,
        args.coarse_query,
        points,
        results,
    )

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_index": args.dataset_index,
                "image_id": int(image_id),
                "caption": caption,
                "coarse_query": args.coarse_query,
                "point_windows": args.point_windows,
                "min_point_distance": args.min_point_distance,
                "points": points,
                "results": [
                    {
                        "num_points": result["num_points"],
                        "candidates": [
                            {
                                key: value
                                for key, value in candidate.items()
                                if key not in {
                                    "mask",
                                    "masked_crop",
                                    "is_best",
                                }
                            }
                            for candidate in result["candidates"]
                        ],
                        "best_mask_rank": result["best"]["mask_rank"],
                    }
                    for result in results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 104)
    print("重点比较 1 / 2 / 3 positive points 的 mask 完整性，不要只看 MaskCLIP。")
    print(f"Figure: {output_path}")
    print(f"JSON  : {json_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
