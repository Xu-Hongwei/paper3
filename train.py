import argparse
from datetime import datetime
from pathlib import Path

import torch

from utils import (
    load_config,
    set_seed,
    save_checkpoint,
    setup_logger,
    append_jsonl,
)

from datasets import (
    create_dataset,
    create_loader,
)

from models import (
    CLIPRetrieval,
)

from losses import (
    CLIPLoss,
)

from engine import (
    build_optimizer,
    build_scheduler,
    train_one_epoch,
)

from evaluation import (
    evaluate_retrieval,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Full Fine-Tuning for RSITR"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    return parser.parse_args()


def main():

    # ==================================================
    # 1. Load config
    # ==================================================

    args = parse_args()

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

    # ==================================================
    # 2. Output directory
    #
    # Must be created BEFORE logger initialization,
    # because the log file is stored here.
    # ==================================================

    output_root = Path(
        config["output"]["root"]
    )

    experiment_name = (
        config["experiment"]["name"]
    )

    output_dir = (
        output_root
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # 3. Logger
    # ==================================================

    log_path, log_file = setup_logger(
        output_dir=output_dir,
        prefix="train",
    )

    metrics_path = (
        output_dir
        / "metrics.jsonl"
    )

    # ==================================================
    # 4. Device
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ==================================================
    # 5. Experiment header
    # ==================================================

    print("=" * 70)
    print("CLIP FULL FINE-TUNING")
    print("=" * 70)

    print(
        f"Start time   : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"Config file  : "
        f"{args.config}"
    )

    print(
        f"Experiment   : "
        f"{config['experiment']['name']}"
    )

    print(
        f"Dataset      : "
        f"{config['dataset']['name']}"
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
        f"Device       : "
        f"{device}"
    )

    print(
        f"Seed         : "
        f"{seed}"
    )

    print(
        f"Output dir   : "
        f"{output_dir}"
    )

    print(
        f"Log file     : "
        f"{log_path}"
    )

    print(
        f"Metrics file : "
        f"{metrics_path}"
    )

    # ==================================================
    # 6. Build model
    # ==================================================

    print()
    print(
        "Building CLIP model..."
    )

    model = CLIPRetrieval(
        config["model"]
    ).to(
        device
    )

    # ==================================================
    # 7. Build datasets
    # ==================================================

    print(
        "Building datasets..."
    )

    datasets = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=(
            model.backbone.preprocess_train
        ),
        eval_transform=(
            model.backbone.preprocess_val
        ),
    )

    if (
        not isinstance(
            datasets,
            (tuple, list),
        )
        or len(datasets) != 2
    ):
        raise RuntimeError(
            "create_dataset(..., evaluate=False) "
            "must return "
            "(train_dataset, val_dataset)."
        )

    train_dataset, val_dataset = (
        datasets
    )

    # ==================================================
    # 8. Read training config
    # ==================================================

    train_batch_size = (
        config["training"][
            "batch_size"
        ]
    )

    eval_batch_size = (
        config["training"].get(
            "eval_batch_size",
            128,
        )
    )

    text_batch_size = (
        config["training"].get(
            "text_batch_size",
            256,
        )
    )

    num_workers = (
        config["training"].get(
            "num_workers",
            8,
        )
    )

    epochs = (
        config["training"][
            "epochs"
        ]
    )

    eval_every = (
        config["training"].get(
            "eval_every",
            1,
        )
    )

    # ==================================================
    # 9. Build DataLoaders
    # ==================================================

    train_loader = create_loader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        is_train=True,
        pin_memory=True,
    )

    val_loader = create_loader(
        val_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        is_train=False,
        pin_memory=True,
    )

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Train pairs      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val images       : "
        f"{len(val_dataset)}"
    )

    print(
        f"Val captions     : "
        f"{len(val_dataset.text)}"
    )

    print(
        f"Train batches    : "
        f"{len(train_loader)}"
    )

    print(
        f"Train batch size : "
        f"{train_batch_size}"
    )

    print(
        f"Eval batch size  : "
        f"{eval_batch_size}"
    )

    print(
        f"Text batch size  : "
        f"{text_batch_size}"
    )

    # ==================================================
    # 10. Criterion
    # ==================================================

    criterion = CLIPLoss()

    # ==================================================
    # 11. Optimizer
    # ==================================================

    optimizer = build_optimizer(
        model=model,
        lr=config["optimizer"]["lr"],
        weight_decay=(
            config["optimizer"][
                "weight_decay"
            ]
        ),
    )

    # ==================================================
    # 12. Scheduler
    # ==================================================

    total_steps = (
        len(train_loader)
        * epochs
    )

    warmup_ratio = (
        config["scheduler"].get(
            "warmup_ratio",
            0.05,
        )
    )

    warmup_steps = int(
        total_steps
        * warmup_ratio
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )

    print()
    print("=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)

    print(
        f"Epochs         : "
        f"{epochs}"
    )

    print(
        f"Learning rate  : "
        f"{config['optimizer']['lr']}"
    )

    print(
        f"Weight decay   : "
        f"{config['optimizer']['weight_decay']}"
    )

    print(
        f"Total steps    : "
        f"{total_steps}"
    )

    print(
        f"Warmup ratio   : "
        f"{warmup_ratio}"
    )

    print(
        f"Warmup steps   : "
        f"{warmup_steps}"
    )

    print(
        f"Eval every     : "
        f"{eval_every} epoch(s)"
    )

    # ==================================================
    # 13. Training state
    # ==================================================

    best_mr = float(
        "-inf"
    )

    best_epoch = -1

    early_stop_patience = config["training"].get(
        "early_stop_patience",
        None,
    )

    early_stop_min_delta = config["training"].get(
        "early_stop_min_delta",
        0.0,
    )

    epochs_without_improvement = 0

    # ==================================================
    # 14. Main training loop
    # ==================================================

    for epoch_idx in range(
        epochs
    ):

        epoch = (
            epoch_idx + 1
        )

        print()
        print("=" * 70)

        print(
            f"EPOCH {epoch}/{epochs}"
        )

        print("=" * 70)

        epoch_start_time = (
            datetime.now()
        )

        # ==============================================
        # Train one epoch
        # ==============================================

        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            max_steps=None,
            log_interval=50,
        )

        train_end_time = (
            datetime.now()
        )

        train_seconds = (
            train_end_time
            -
            epoch_start_time
        ).total_seconds()

        print()
        print("-" * 70)

        print(
            f"EPOCH {epoch} TRAIN SUMMARY"
        )

        print("-" * 70)

        print(
            f"Average loss : "
            f"{train_stats['loss']:.6f}"
        )

        print(
            f"Average I2T  : "
            f"{train_stats['loss_i2t']:.6f}"
        )

        print(
            f"Average T2I  : "
            f"{train_stats['loss_t2i']:.6f}"
        )

        print(
            f"Current LR   : "
            f"{train_stats['lr']:.8f}"
        )

        print(
            f"Logit scale  : "
            f"{train_stats['logit_scale']:.4f}"
        )

        print(
            f"Train time   : "
            f"{train_seconds:.2f} s"
        )

        # ==============================================
        # If this epoch does not perform validation,
        # still save training statistics to JSONL.
        # ==============================================

        if (
            epoch % eval_every
            != 0
        ):

            epoch_record = {
                "epoch": int(
                    epoch
                ),

                "train_loss": float(
                    train_stats["loss"]
                ),

                "train_loss_i2t": float(
                    train_stats["loss_i2t"]
                ),

                "train_loss_t2i": float(
                    train_stats["loss_t2i"]
                ),

                "lr": float(
                    train_stats["lr"]
                ),

                "logit_scale": float(
                    train_stats["logit_scale"]
                ),

                "train_seconds": float(
                    train_seconds
                ),

                "validated": False,
            }

            append_jsonl(
                metrics_path,
                epoch_record,
            )

            continue

        # ==============================================
        # Validation
        # ==============================================

        print()
        print("=" * 70)

        print(
            f"VALIDATION - EPOCH {epoch}"
        )

        print("=" * 70)

        validation_start_time = (
            datetime.now()
        )

        metrics, _ = evaluate_retrieval(
            model=model,
            data_loader=val_loader,
            dataset=val_dataset,
            device=device,
            text_batch_size=text_batch_size,
        )

        validation_end_time = (
            datetime.now()
        )

        validation_seconds = (
            validation_end_time
            -
            validation_start_time
        ).total_seconds()

        current_mr = (
            metrics["mR"]
        )

        # ==============================================
        # Print validation metrics
        # ==============================================

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
            f"VAL mR   : "
            f"{current_mr:.2f}"
        )

        print(
            f"Val time : "
            f"{validation_seconds:.2f} s"
        )

        # ==============================================
        # Determine whether this is a new best model
        #
        # Do this BEFORE writing JSONL so the log
        # can record whether this epoch is best.
        # ==============================================

        is_best = current_mr > (best_mr + early_stop_min_delta)

        if is_best:
            best_mr = current_mr
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ==============================================
        # Save structured epoch log
        # ==============================================

        epoch_record = {
            "epoch": int(
                epoch
            ),

            # ------------------------------------------
            # Training
            # ------------------------------------------

            "train_loss": float(
                train_stats["loss"]
            ),

            "train_loss_i2t": float(
                train_stats["loss_i2t"]
            ),

            "train_loss_t2i": float(
                train_stats["loss_t2i"]
            ),

            "lr": float(
                train_stats["lr"]
            ),

            "logit_scale": float(
                train_stats["logit_scale"]
            ),

            "train_seconds": float(
                train_seconds
            ),

            # ------------------------------------------
            # Validation
            # ------------------------------------------

            "validated": True,

            "val_i2t_r1": float(
                metrics["i2t_r1"]
            ),

            "val_i2t_r5": float(
                metrics["i2t_r5"]
            ),

            "val_i2t_r10": float(
                metrics["i2t_r10"]
            ),

            "val_t2i_r1": float(
                metrics["t2i_r1"]
            ),

            "val_t2i_r5": float(
                metrics["t2i_r5"]
            ),

            "val_t2i_r10": float(
                metrics["t2i_r10"]
            ),

            "val_i2t_mean": float(
                metrics["i2t_mean"]
            ),

            "val_t2i_mean": float(
                metrics["t2i_mean"]
            ),

            "val_mR": float(
                current_mr
            ),

            "val_i2t_medr": float(
                metrics["i2t_medr"]
            ),

            "val_t2i_medr": float(
                metrics["t2i_medr"]
            ),

            "val_i2t_meanr": float(
                metrics.get(
                    "i2t_meanr",
                    0.0,
                )
            ),

            "val_t2i_meanr": float(
                metrics.get(
                    "t2i_meanr",
                    0.0,
                )
            ),

            "validation_seconds": float(
                validation_seconds
            ),

            # ------------------------------------------
            # Best model information
            # ------------------------------------------

            "is_best": bool(
                is_best
            ),

            "best_epoch": int(
                best_epoch
            ),

            "best_val_mR": float(
                best_mr
            ),
        }

        append_jsonl(
            metrics_path,
            epoch_record,
        )

        # ==============================================
        # Save last checkpoint
        # ==============================================

        last_path = (
            output_dir
            / "last.pth"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )

        print()
        print(
            f"Saved last checkpoint: "
            f"{last_path}"
        )

        # ==============================================
        # Save best checkpoint
        # ==============================================

        if is_best:

            best_path = (
                output_dir
                / "best.pth"
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )

            print()
            print(
                ">>> NEW BEST MODEL"
            )

            print(
                f"Epoch : "
                f"{best_epoch}"
            )

            print(
                f"mR    : "
                f"{best_mr:.2f}"
            )

            print(
                f"Saved : "
                f"{best_path}"
            )


        if (
            early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print()
            print("=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(f"No improvement for {epochs_without_improvement} validation rounds.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best Val mR: {best_mr:.2f}")
            break

        # ==============================================
        # Best so far
        # ==============================================

        print()

        print(
            f"Best epoch so far : "
            f"{best_epoch}"
        )

        print(
            f"Best Val mR       : "
            f"{best_mr:.2f}"
        )

        epoch_end_time = (
            datetime.now()
        )

        epoch_seconds = (
            epoch_end_time
            -
            epoch_start_time
        ).total_seconds()

        print(
            f"Epoch total time  : "
            f"{epoch_seconds:.2f} s"
        )

        # Force log content to disk immediately.
        log_file.flush()

    # ==================================================
    # 15. Finished
    # ==================================================

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Finish time  : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"Best epoch   : "
        f"{best_epoch}"
    )

    print(
        f"Best Val mR  : "
        f"{best_mr:.2f}"
    )

    print(
        f"Best model   : "
        f"{output_dir / 'best.pth'}"
    )

    print(
        f"Log file     : "
        f"{log_path}"
    )

    print(
        f"Metrics file : "
        f"{metrics_path}"
    )

    log_file.flush()


if __name__ == "__main__":
    main()
