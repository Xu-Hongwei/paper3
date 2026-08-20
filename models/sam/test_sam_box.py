import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sam import SAM2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 box-prompt 可视化测试")
    parser.add_argument("--image", type=str, required=True)
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
        default="outputs/sam_demo/sam_box_demo.png",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def overlay_mask(image, mask):
    canvas = image.astype(np.float32).copy()
    # 仅用于可视化，不修改原始分割结果。
    canvas[mask] = 0.55 * canvas[mask] + 0.45 * np.array([255, 255, 255])
    return canvas.clip(0, 255).astype(np.uint8)


def main():
    args = parse_args()

    image = np.asarray(
        Image.open(args.image).convert("RGB")
    )

    segmenter = SAM2Segmenter(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )
    segmenter.set_image(image)

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

    axes[0].imshow(image)
    x1, y1, x2, y2 = args.box
    axes[0].add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
    )
    axes[0].set_title("Input + Box")
    axes[0].axis("off")

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        axes[i].imshow(overlay_mask(image, mask))
        axes[i].contour(mask, levels=[0.5], linewidths=1)
        axes[i].set_title(
            f"Mask {i}\\nSAM score={score:.4f}\\narea={mask.mean():.3f}"
        )
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("SAM2 BOX TEST")
    print("=" * 72)
    print(f"Image : {args.image}")
    print(f"Size  : {image.shape[1]} x {image.shape[0]}")
    print(f"Box   : {args.box}")

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        print(
            f"Mask {i}: score={score:.4f}, "
            f"area_ratio={mask.mean():.4f}"
        )

    print(f"Saved : {output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
