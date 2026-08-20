import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from models import CLIPRetrieval
from models.grounding_dino import GroundingDINODetector
from models.sam import SAM2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.5B: Grounding DINO box -> SAM2 mask sanity check"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--gdino-model-id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-boxes", type=int, default=5)

    parser.add_argument("--sam-checkpoint", type=str, required=True)
    parser.add_argument(
        "--sam-model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    parser.add_argument("--min-mask-area", type=float, default=0.001)
    parser.add_argument("--max-mask-area", type=float, default=0.95)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/c05b_grounding_dino_sam/sample.png",
    )
    return parser.parse_args()


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


def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(x1 + 1.0, min(x2, width))
    y2 = max(y1 + 1.0, min(y2, height))
    return [x1, y1, x2, y2]


def overlay_mask(image, mask):
    canvas = image.astype(np.float32).copy()
    canvas[mask] = 0.55 * canvas[mask] + 0.45 * 255.0
    return canvas.clip(0, 255).astype(np.uint8)


def save_visualization(path, image_rgb, caption, text, detections):
    rows = max(len(detections), 1)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(16, 4.2 * rows),
        squeeze=False,
    )

    if not detections:
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("No Grounding DINO detection")
        axes[0, 0].axis("off")
        for col in range(1, 4):
            axes[0, col].axis("off")
    else:
        for row, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["box"]

            axes[row, 0].imshow(image_rgb)
            axes[row, 0].add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=2,
                )
            )
            axes[row, 0].set_title(
                f"GDINO Box {detection['rank']}\n"
                f"score={detection['score']:.3f}"
            )
            axes[row, 0].axis("off")

            for col in range(1, 4):
                ax = axes[row, col]
                mask_idx = col - 1

                if mask_idx >= len(detection["masks"]):
                    ax.axis("off")
                    continue

                mask_info = detection["masks"][mask_idx]
                ax.imshow(overlay_mask(image_rgb, mask_info["mask"]))
                ax.contour(mask_info["mask"], levels=[0.5], linewidths=1)
                ax.set_title(
                    f"M{mask_info['mask_rank']}\n"
                    f"SAM={mask_info['sam_score']:.3f}, "
                    f"area={mask_info['area_ratio']:.3f}"
                )
                ax.axis("off")

    fig.suptitle(
        f'{caption}\nGrounding DINO -> SAM2 | coarse query: "{text}"'
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 仅借用 CLIP eval transform，确保与前面实验使用同一个 224 view。
    clip_model = CLIPRetrieval(config["model"])
    preprocess = clip_model.backbone.preprocess_val
    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=preprocess,
        eval_transform=preprocess,
    )

    image_tensor, caption, image_id, *_ = train_dataset[args.dataset_index]
    image_rgb = recover_rgb(image_tensor, preprocess)
    height, width = image_rgb.shape[:2]

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detector = GroundingDINODetector(
        model_id=args.gdino_model_id,
        device=device,
        local_files_only=args.local_files_only,
    )

    gdino = detector.predict(
        image_rgb,
        text=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    order = np.argsort(gdino["scores"])[::-1]
    order = order[:args.max_boxes]

    sam = SAM2Segmenter(
        checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_model_cfg,
        device=device,
    )
    sam.set_image(image_rgb)

    detections = []

    for rank, idx in enumerate(order, start=1):
        box = clip_box(gdino["boxes"][idx], width, height)
        score = float(gdino["scores"][idx])
        label = (
            gdino["labels"][idx]
            if idx < len(gdino["labels"])
            else args.text
        )

        masks, sam_scores, _ = sam.predict_box(
            box,
            multimask_output=True,
        )

        mask_infos = []
        for mask_rank, (mask, sam_score) in enumerate(
            zip(masks, sam_scores),
            start=1,
        ):
            mask = np.asarray(mask, dtype=bool)
            area_ratio = float(mask.mean())

            if not args.min_mask_area <= area_ratio <= args.max_mask_area:
                continue

            mask_infos.append(
                {
                    "mask_rank": mask_rank,
                    "sam_score": float(sam_score),
                    "area_ratio": area_ratio,
                    "mask": mask,
                }
            )

        detections.append(
            {
                "rank": rank,
                "label": label,
                "score": score,
                "box": box,
                "masks": mask_infos,
            }
        )

    print("=" * 100)
    print("C0.5B GROUNDING DINO -> SAM2")
    print("=" * 100)
    print(f"Dataset index : {args.dataset_index}")
    print(f"Image ID      : {image_id}")
    print(f"Caption       : {caption}")
    print(f"Text query    : {args.text}")
    print(f"GDINO model   : {args.gdino_model_id}")
    print(f"Image size    : {width} x {height}")
    print(f"Detections    : {len(detections)}")
    print("=" * 100)

    for detection in detections:
        print(
            f"\nBox {detection['rank']} | "
            f"label={detection['label']} | "
            f"GDINO={detection['score']:.4f} | "
            f"box={[round(x, 2) for x in detection['box']]}"
        )
        for mask in detection["masks"]:
            print(
                f"  M{mask['mask_rank']} | "
                f"SAM={mask['sam_score']:.4f} | "
                f"area={mask['area_ratio']:.4f}"
            )

    output_path = Path(args.output)
    save_visualization(
        output_path,
        image_rgb,
        caption,
        args.text,
        detections,
    )

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_index": args.dataset_index,
                "image_id": int(image_id),
                "caption": caption,
                "text": args.text,
                "gdino_model_id": args.gdino_model_id,
                "image_size": [width, height],
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "detections": [
                    {
                        "rank": detection["rank"],
                        "label": detection["label"],
                        "score": detection["score"],
                        "box": detection["box"],
                        "masks": [
                            {
                                "mask_rank": mask["mask_rank"],
                                "sam_score": mask["sam_score"],
                                "area_ratio": mask["area_ratio"],
                            }
                            for mask in detection["masks"]
                        ],
                    }
                    for detection in detections
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 100)
    print("当前阶段不做 CLIP rerank，只人工检查 GDINO box 是否能驱动 SAM 得到完整且干净的 object mask。")
    print(f"Figure: {output_path}")
    print(f"JSON  : {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
