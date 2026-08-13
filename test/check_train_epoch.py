import argparse
from pathlib import Path
import sys

import torch


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

from datasets import (
    create_dataset,
    create_loader,
)

from models import CLIPRetrieval

from losses import CLIPLoss

from engine import (
    build_optimizer,
    train_one_epoch,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for CLIP "
            "training on real RSICD batches."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==========================================
    # 1. Config
    # ==========================================

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
    print("CLIP TRAIN EPOCH CHECK")
    print("=" * 70)

    print(
        f"Device       : {device}"
    )

    print(
        f"Backbone     : "
        f"{config['model']['backbone']}"
    )

    print(
        f"Pretrained   : "
        f"{config['model']['pretrained']}"
    )

    # ==========================================
    # 2. Model
    # ==========================================

    model = CLIPRetrieval(
        config["model"]
    ).to(device)

    # ==========================================
    # 3. Dataset
    # ==========================================

    train_dataset, _ = create_dataset(
        config["dataset"],
        train_transform=(
            model.backbone.preprocess_train
        ),
        eval_transform=(
            model.backbone.preprocess_val
        ),
    )

    batch_size = (
        config["training"]["batch_size"]
    )

    train_loader = create_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        is_train=True,
        pin_memory=True,
    )

    print()
    print(
        f"Train pairs  : "
        f"{len(train_dataset)}"
    )

    print(
        f"Batch size   : "
        f"{batch_size}"
    )

    print(
        f"Loader steps : "
        f"{len(train_loader)}"
    )

    print(
        f"Test steps   : "
        f"{args.max_steps}"
    )

    # ==========================================
    # 4. Loss
    # ==========================================

    criterion = CLIPLoss()

    # ==========================================
    # 5. Optimizer
    # ==========================================

    optimizer = build_optimizer(
        model,
        lr=config["optimizer"]["lr"],
        weight_decay=(
            config["optimizer"][
                "weight_decay"
            ]
        ),
    )

    print()
    print(
        f"Learning rate: "
        f"{config['optimizer']['lr']}"
    )

    print(
        f"Weight decay : "
        f"{config['optimizer']['weight_decay']}"
    )

    # ==========================================
    # 6. Train
    # ==========================================

    max_steps = (
        None
        if args.max_steps <= 0
        else args.max_steps
    )

    print()
    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    stats = train_one_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=0,
        max_steps=max_steps,
        log_interval=10,
    )

    # ==========================================
    # 7. Result
    # ==========================================

    print()
    print("=" * 70)
    print("TRAINING STATS")
    print("=" * 70)

    print(
        f"Steps       : "
        f"{stats['num_steps']}"
    )

    print(
        f"Average loss: "
        f"{stats['loss']:.6f}"
    )

    print(
        f"Average I2T : "
        f"{stats['loss_i2t']:.6f}"
    )

    print(
        f"Average T2I : "
        f"{stats['loss_t2i']:.6f}"
    )

    print(
        f"Logit scale : "
        f"{stats['logit_scale']:.6f}"
    )

    # ==========================================
    # 8. Assertions
    # ==========================================

    assert (
        stats["num_steps"] > 0
    )

    assert torch.isfinite(
        torch.tensor(
            stats["loss"]
        )
    )

    assert (
        1.0
        <= stats["logit_scale"]
        <= 100.0001
    )

    print()
    print("=" * 70)
    print(
        "CLIP TRAIN EPOCH CHECK: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()