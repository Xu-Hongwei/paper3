import argparse
from pathlib import Path
import sys

import numpy as np
import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# Project imports
# ============================================================

from utils import (
    load_config,
    set_seed,
)

from datasets import (
    create_dataset,
    create_loader,
)

from models import (
    CLIPRetrieval,
)

from evaluation import (
    evaluate_retrieval,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CLIP image-text retrieval "
            "on RSITR datasets."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "val",
            "test",
        ],
        help="Evaluation split.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional baseline or Adapter checkpoint path. "
            "If omitted, evaluate zero-init Adapter on pretrained CLIP."
        ),
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=128,
        help="Image inference batch size.",
    )

    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Text inference batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )

    return parser.parse_args()


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(model, checkpoint_path):
    """加载旧 baseline 或包含 Adapter 的新训练 checkpoint。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    print("\n" + "=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)
    print(f"Path         : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    # 旧 CLIP baseline 没有 Adapter 参数，这是唯一允许的缺失。
    invalid_missing = [
        name for name in missing
        if not name.startswith(("visual_adapter.", "text_adapter."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model architecture mismatch, "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )

    if missing:
        print("Adapter weights : not found, using zero initialization")
    else:
        print("Adapter weights : loaded")

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(f"Epoch        : {checkpoint['epoch']}")

        metrics = checkpoint.get("metrics")
        if isinstance(metrics, dict) and "mR" in metrics:
            print(f"Stored Val mR: {metrics['mR']:.2f}")

    print("Checkpoint loaded successfully.")
    return checkpoint


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Config
    # ========================================================

    config = load_config(
        args.config
    )

    seed = config.get(
        "seed",
        42,
    )

    set_seed(
        seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Evaluation mode label
    # ========================================================

    if args.checkpoint is None:

        eval_mode = (
            "ZERO-SHOT"
        )

    else:

        eval_mode = (
            "CHECKPOINT"
        )

    print("=" * 70)

    print(
        f"{eval_mode} RETRIEVAL CHECK"
    )

    print("=" * 70)

    print(
        f"Device       : "
        f"{device}"
    )

    print(
        f"Split        : "
        f"{args.split}"
    )

    print(
        f"Backbone     : "
        f"{config['model']['backbone']}"
    )

    print(
        f"Pretrained   : "
        f"{config['model']['pretrained']}"
    )

    print(
        f"Seed         : "
        f"{seed}"
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Building CLIP model..."
    )

    model = CLIPRetrieval(
        config["model"]
    )

    # ========================================================
    # Optional checkpoint loading
    # ========================================================

    if args.checkpoint is not None:

        load_checkpoint(
            model=model,
            checkpoint_path=(
                args.checkpoint
            ),
        )

    # --------------------------------------------------------
    # Move model to GPU after loading.
    #
    # Loading on CPU avoids unnecessary GPU memory usage.
    # --------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()

    # ========================================================
    # Evaluation dataset
    # ========================================================

    print()
    print(
        "Building evaluation dataset..."
    )

    eval_dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=(
            model.backbone.preprocess_val
        ),
    )

    # ========================================================
    # Evaluation DataLoader
    # ========================================================

    eval_loader = create_loader(
        eval_dataset,
        batch_size=(
            args.image_batch_size
        ),
        num_workers=(
            args.num_workers
        ),
        is_train=False,
        pin_memory=True,
    )

    # ========================================================
    # Dataset information
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Number images  : "
        f"{len(eval_dataset)}"
    )

    print(
        f"Number captions: "
        f"{len(eval_dataset.text)}"
    )

    print(
        f"Image batch    : "
        f"{args.image_batch_size}"
    )

    print(
        f"Text batch     : "
        f"{args.text_batch_size}"
    )

    # ========================================================
    # Retrieval evaluation
    # ========================================================

    print()
    print("=" * 70)
    print("START RETRIEVAL")
    print("=" * 70)

    metrics, scores = (
        evaluate_retrieval(
            model=model,
            data_loader=eval_loader,
            dataset=eval_dataset,
            device=device,
            text_batch_size=(
                args.text_batch_size
            ),
        )
    )

    # ========================================================
    # Sanity check 1:
    # similarity matrix shape
    # ========================================================

    expected_shape = (
        len(eval_dataset),
        len(eval_dataset.text),
    )

    if (
        scores.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected similarity matrix shape.\n"
            f"Actual   : {scores.shape}\n"
            f"Expected : {expected_shape}"
        )

    # ========================================================
    # Sanity check 2:
    # scores must be finite
    # ========================================================

    if not np.isfinite(
        scores
    ).all():

        raise RuntimeError(
            "Similarity matrix contains "
            "NaN or Inf values."
        )

    # ========================================================
    # Sanity check 3:
    # metrics must be finite
    # ========================================================

    for (
        metric_name,
        value,
    ) in metrics.items():

        if not np.isfinite(
            value
        ):

            raise RuntimeError(
                f"Metric '{metric_name}' "
                f"is not finite: {value}"
            )

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"{args.split.upper()} "
        f"RETRIEVAL RESULT"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Image -> Text
    # --------------------------------------------------------

    print()
    print(
        "Image -> Text"
    )

    print(
        f"R@1  : "
        f"{metrics['i2t_r1']:.2f}"
    )

    print(
        f"R@5  : "
        f"{metrics['i2t_r5']:.2f}"
    )

    print(
        f"R@10 : "
        f"{metrics['i2t_r10']:.2f}"
    )

    # --------------------------------------------------------
    # Text -> Image
    # --------------------------------------------------------

    print()
    print(
        "Text -> Image"
    )

    print(
        f"R@1  : "
        f"{metrics['t2i_r1']:.2f}"
    )

    print(
        f"R@5  : "
        f"{metrics['t2i_r5']:.2f}"
    )

    print(
        f"R@10 : "
        f"{metrics['t2i_r10']:.2f}"
    )

    # --------------------------------------------------------
    # Mean Recall
    # --------------------------------------------------------

    print()

    print(
        f"I2T mean : "
        f"{metrics['i2t_mean']:.2f}"
    )

    print(
        f"T2I mean : "
        f"{metrics['t2i_mean']:.2f}"
    )

    print()

    print(
        f"mR       : "
        f"{metrics['mR']:.2f}"
    )

    # --------------------------------------------------------
    # Rank statistics
    # --------------------------------------------------------

    print()

    print(
        f"I2T MedR : "
        f"{metrics['i2t_medr']:.2f}"
    )

    print(
        f"T2I MedR : "
        f"{metrics['t2i_medr']:.2f}"
    )

    if (
        "i2t_meanr"
        in metrics
    ):

        print(
            f"I2T MeanR: "
            f"{metrics['i2t_meanr']:.2f}"
        )

    if (
        "t2i_meanr"
        in metrics
    ):

        print(
            f"T2I MeanR: "
            f"{metrics['t2i_meanr']:.2f}"
        )

    # ========================================================
    # Final status
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"{eval_mode} "
        f"RETRIEVAL CHECK: PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()