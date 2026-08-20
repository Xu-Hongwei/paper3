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


def parse_args():
    parser = argparse.ArgumentParser(
        description="C0.5A: Grounding DINO 在与 CLIP 一致的 224 eval view 上做 coarse entity 定位"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--model-id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/c05_grounding_dino/sample.png",
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


def save_visualization(path, image_rgb, caption, text, result):
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_rgb)

    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = box.tolist()
        label = labels[i - 1] if i - 1 < len(labels) else text

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
            max(0, y1 - 3),
            f"{i}: {label} {score:.3f}",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )

    ax.set_title(
        f'{caption}\nGrounding DINO coarse query: "{text}" | detections={len(boxes)}'
    )
    ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 只借用原 CLIP eval transform，确保和前面 C0.4 都处于同一个 224 坐标系。
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

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detector = GroundingDINODetector(
        model_id=args.model_id,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    print("=" * 96)
    print("C0.5A GROUNDING DINO COARSE LOCALIZATION")
    print("=" * 96)
    print(f"Dataset index : {args.dataset_index}")
    print(f"Image ID      : {image_id}")
    print(f"Caption       : {caption}")
    print(f"Text query    : {args.text}")
    print(f"Model         : {args.model_id}")
    print(f"Image size    : {image_rgb.shape[1]} x {image_rgb.shape[0]}")
    print(f"Box threshold : {args.box_threshold}")
    print(f"Text threshold: {args.text_threshold}")
    print("=" * 96)

    result = detector.predict(
        image_rgb,
        text=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    print(f"\nDetections: {len(result['boxes'])}")
    for i, (box, score) in enumerate(
        zip(result["boxes"], result["scores"]),
        start=1,
    ):
        label = (
            result["labels"][i - 1]
            if i - 1 < len(result["labels"])
            else args.text
        )
        print(
            f"  #{i}: label={label}, score={score:.4f}, "
            f"box={[round(float(x), 2) for x in box]}"
        )

    output_path = Path(args.output)
    save_visualization(
        output_path,
        image_rgb,
        caption,
        args.text,
        result,
    )

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_index": args.dataset_index,
                "image_id": int(image_id),
                "caption": caption,
                "text": args.text,
                "model_id": args.model_id,
                "image_size": [
                    int(image_rgb.shape[1]),
                    int(image_rgb.shape[0]),
                ],
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "detections": [
                    {
                        "rank": i + 1,
                        "label": (
                            result["labels"][i]
                            if i < len(result["labels"])
                            else args.text
                        ),
                        "score": float(result["scores"][i]),
                        "box": [
                            float(x)
                            for x in result["boxes"][i].tolist()
                        ],
                    }
                    for i in range(len(result["boxes"]))
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 96)
    print(f"Figure: {output_path}")
    print(f"JSON  : {json_path}")
    print("当前阶段只判断 Grounding DINO box 是否具有 object-level 完整性，不接 SAM、不调最优阈值。")
    print("=" * 96)


if __name__ == "__main__":
    main()
