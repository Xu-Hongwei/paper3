import argparse
from pathlib import Path
import sys

import torch

# Allow running:
# python tools/check_dataset.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config
from datasets import create_dataset, create_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check RSITR dataset pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def inspect_train_dataset(config, batch_size):
    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
    )

    print("=" * 70)
    print("TRAIN DATASET CHECK")
    print("=" * 70)

    print(f"Number of image-caption pairs: {len(train_dataset)}")

    image, caption = train_dataset[0]

    print(f"Single image shape: {tuple(image.shape)}")
    print(f"Single caption: {caption}")

    loader = create_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        is_train=False,
        pin_memory=False,
    )

    images, captions = next(iter(loader))

    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch size: {len(captions)}")
    print("First batch captions:")

    for idx, text in enumerate(captions):
        print(f"  [{idx}] {text}")

    print("=" * 70)


def inspect_eval_dataset(config, split, batch_size):
    eval_dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=split,
    )

    print("=" * 70)
    print(f"{split.upper()} DATASET CHECK")
    print("=" * 70)

    print(f"Number of unique images : {len(eval_dataset.image)}")
    print(f"Number of captions      : {len(eval_dataset.text)}")
    print(f"txt2img entries         : {len(eval_dataset.txt2img)}")
    print(f"img2txt entries         : {len(eval_dataset.img2txt)}")

    if len(eval_dataset.image) > 0:
        first_img_caps = eval_dataset.img2txt[0]

        print(f"Image 0 caption ids     : {first_img_caps}")

        mapped_back = [
            eval_dataset.txt2img[txt_id]
            for txt_id in first_img_caps
        ]

        print(f"Caption -> image mapping: {mapped_back}")

        print("Image 0 captions:")

        for txt_id in first_img_caps:
            print(
                f"  [{txt_id}] {eval_dataset.text[txt_id]}"
            )

    # Mapping consistency check.
    for img_id, txt_ids in eval_dataset.img2txt.items():
        for txt_id in txt_ids:
            assert eval_dataset.txt2img[txt_id] == img_id, (
                f"Mapping mismatch: img={img_id}, txt={txt_id}"
            )

    print("Mapping consistency     : PASS")

    loader = create_loader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=0,
        is_train=False,
        pin_memory=False,
    )

    images, image_ids = next(iter(loader))

    print(f"Batch image shape       : {tuple(images.shape)}")
    print(f"Batch image ids         : {image_ids.tolist()}")

    print("=" * 70)


def main():
    args = parse_args()

    config = load_config(args.config)

    if args.split == "train":
        inspect_train_dataset(
            config,
            args.batch_size,
        )
    else:
        inspect_eval_dataset(
            config,
            args.split,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
