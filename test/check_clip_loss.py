import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from utils import (
    load_config,
    set_seed,
)

from datasets import create_dataset

from models import CLIPRetrieval

from losses import CLIPLoss


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Check CLIP contrastive loss "
            "and single-batch overfitting."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
    )

    return parser.parse_args()


def select_unique_image_indices(
    dataset,
    num_samples,
):
    """
    Select training pairs belonging to different images.

    This avoids putting multiple captions of the same
    image into the sanity-check batch.
    """

    selected_indices = []

    seen_images = set()

    for index, ann in enumerate(
        dataset.ann
    ):

        image_name = ann["image"]

        if image_name in seen_images:
            continue

        seen_images.add(
            image_name
        )

        selected_indices.append(
            index
        )

        if len(selected_indices) >= num_samples:
            break

    if len(selected_indices) < num_samples:
        raise RuntimeError(
            "Not enough unique images "
            "to construct test batch."
        )

    return selected_indices


def print_similarity_matrix(
    image_features,
    text_features,
):
    similarity = (
        image_features
        @ text_features.t()
    )

    print(
        similarity.detach().cpu()
    )


def main():

    args = parse_args()

    # ==================================================
    # 1. Config / Seed / Device
    # ==================================================

    config = load_config(
        args.config
    )

    seed = config.get(
        "seed",
        42,
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CLIP LOSS CHECK")
    print("=" * 70)

    print(
        f"Device     : {device}"
    )

    print(
        f"Backbone   : "
        f"{config['model']['backbone']}"
    )

    print(
        f"Pretrained : "
        f"{config['model']['pretrained']}"
    )

    print(
        f"Steps      : {args.steps}"
    )

    print(
        f"LR         : {args.lr}"
    )

    # ==================================================
    # 2. Model
    # ==================================================

    model = CLIPRetrieval(
        config["model"]
    ).to(device)

    # ==================================================
    # 3. Dataset
    #
    # Important:
    # use CLIP official training preprocessing.
    # ==================================================

    train_dataset, _ = create_dataset(
        config["dataset"],
        train_transform=(
            model.backbone.preprocess_train
        ),
        eval_transform=(
            model.backbone.preprocess_val
        ),
    )

    # ==================================================
    # 4. Construct a clean batch:
    #    one pair per unique image
    # ==================================================

    selected_indices = (
        select_unique_image_indices(
            train_dataset,
            args.batch_size,
        )
    )

    print()
    print(
        "Selected dataset indices:",
        selected_indices
    )

    subset = Subset(
        train_dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, captions, image_ids = next(
        iter(loader)
    )

    images = images.to(
        device
    )

    image_ids = image_ids.to(
        device
    )

    print()
    print(
        "Image batch shape:",
        tuple(images.shape)
    )

    print(
        "Image ids:",
        image_ids.detach().cpu().tolist()
    )

    print("Captions:")

    for index, caption in enumerate(
        captions
    ):
        print(
            f"  [{index}] {caption}"
        )

    # ==================================================
    # 5. Loss
    # ==================================================

    criterion = CLIPLoss()

    # ==================================================
    # 6. Forward BEFORE training
    # ==================================================

    model.eval()

    with torch.no_grad():

        outputs = model(
            images,
            captions,
        )

        initial_losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )

    initial_loss = float(
        initial_losses["loss"]
    )

    print()
    print("=" * 70)
    print("BEFORE OVERFIT")
    print("=" * 70)

    print(
        f"Loss I2T : "
        f"{float(initial_losses['loss_i2t']):.6f}"
    )

    print(
        f"Loss T2I : "
        f"{float(initial_losses['loss_t2i']):.6f}"
    )

    print(
        f"Total Loss: "
        f"{initial_loss:.6f}"
    )

    print()
    print("Cosine similarity:")

    print_similarity_matrix(
        outputs["image_feat"],
        outputs["text_feat"],
    )

    # ==================================================
    # 7. Optimizer
    #
    # Only for sanity-check overfitting.
    # ==================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    # ==================================================
    # 8. Single-batch overfitting
    # ==================================================

    model.train()

    print()
    print("=" * 70)
    print("OVERFITTING ONE FIXED BATCH")
    print("=" * 70)

    for step in range(
        1,
        args.steps + 1,
    ):

        outputs = model(
            images,
            captions,
        )

        losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )

        loss = losses["loss"]

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (
            step == 1
            or step % 10 == 0
            or step == args.steps
        ):

            print(
                f"Step "
                f"{step:03d}/{args.steps} | "
                f"Loss: {float(loss):.6f} | "
                f"I2T: "
                f"{float(losses['loss_i2t']):.6f} | "
                f"T2I: "
                f"{float(losses['loss_t2i']):.6f}"
            )

    # ==================================================
    # 9. Forward AFTER training
    # ==================================================

    model.eval()

    with torch.no_grad():

        outputs = model(
            images,
            captions,
        )

        final_losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )

    final_loss = float(
        final_losses["loss"]
    )

    print()
    print("=" * 70)
    print("AFTER OVERFIT")
    print("=" * 70)

    print(
        f"Initial Loss : "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final Loss   : "
        f"{final_loss:.6f}"
    )

    print(
        f"Loss decrease: "
        f"{initial_loss - final_loss:.6f}"
    )

    print()
    print("Cosine similarity:")

    print_similarity_matrix(
        outputs["image_feat"],
        outputs["text_feat"],
    )

    # ==================================================
    # 10. Basic checks
    # ==================================================

    assert torch.isfinite(
        final_losses["loss"]
    )

    assert (
        final_loss
        <
        initial_loss
    ), (
        "Single-batch loss did not decrease."
    )

    print()
    print("=" * 70)
    print(
        "CLIP LOSS + BACKWARD CHECK: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
