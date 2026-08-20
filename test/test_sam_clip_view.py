import argparse
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
from models.sam import SAM2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="在 CLIP 224×224 eval view 上测试 SAM2 box prompt"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sam_demo/sam_clip_view.png",
    )
    parser.add_argument("--device", type=str, default=None)
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

    image = (
        image.clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    return (image * 255.0).round().astype(np.uint8)


def overlay_mask(image, mask):
    canvas = image.astype(np.float32).copy()
    canvas[mask] = (
        0.55 * canvas[mask]
        + 0.45 * np.array([255, 255, 255], dtype=np.float32)
    )
    return canvas.clip(0, 255).astype(np.uint8)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 这里只借用 CLIP 的 eval transform，保证和 C0 的 224×224 view 完全一致。
    clip_model = CLIPRetrieval(config["model"]).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=clip_model.backbone.preprocess_val,
        eval_transform=clip_model.backbone.preprocess_val,
    )

    if not 0 <= args.dataset_index < len(train_dataset):
        raise IndexError(
            f"dataset_index={args.dataset_index} 超出范围 [0, {len(train_dataset) - 1}]"
        )

    image_tensor, caption, image_id, *_ = train_dataset[args.dataset_index]
    image_rgb = recover_rgb(
        image_tensor,
        clip_model.backbone.preprocess_val,
    )

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = args.box

    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            f"box={args.box} 超出当前 CLIP view 尺寸 {w}×{h}"
        )

    segmenter = SAM2Segmenter(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )
    segmenter.set_image(image_rgb)

    masks, scores, _ = segmenter.predict_box(
        args.box,
        multimask_output=True,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        1 + len(masks),
        figsize=(5 * (1 + len(masks)), 5),
    )

    axes[0].imshow(image_rgb)
    axes[0].add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
    )
    axes[0].set_title(
        f"CLIP eval view\nindex={args.dataset_index}, image_id={image_id}"
    )
    axes[0].axis("off")

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        axes[i].imshow(overlay_mask(image_rgb, mask))
        axes[i].contour(mask, levels=[0.5], linewidths=1)
        axes[i].set_title(
            f"Mask {i}\nscore={score:.4f}, area={mask.mean():.3f}"
        )
        axes[i].axis("off")

    fig.suptitle(caption)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 88)
    print("SAM2 ON CLIP EVAL VIEW")
    print("=" * 88)
    print(f"Dataset index : {args.dataset_index}")
    print(f"Image ID      : {image_id}")
    print(f"Caption       : {caption}")
    print(f"View size     : {w} x {h}")
    print(f"Box           : {args.box}")

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            bbox = None
        else:
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ]

        print(
            f"Mask {i}: score={score:.4f}, "
            f"area_ratio={mask.mean():.4f}, bbox={bbox}"
        )

    print(f"Saved         : {output}")
    print("=" * 88)


if __name__ == "__main__":
    main()
