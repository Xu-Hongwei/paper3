import argparse
from datetime import datetime
from pathlib import Path

import torch

from utils import load_config, set_seed, save_checkpoint, setup_logger, append_jsonl
from datasets import create_dataset, create_loader
from models import CLIPRetrieval
from losses import CLIPLoss
from engine import build_optimizer, build_scheduler, train_one_epoch
from evaluation import evaluate_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Adapter Training for RSITR")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_initial_checkpoint(model, checkpoint_path):
    """加载旧 baseline 权重，允许新 Adapter 参数缺失。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    invalid_missing = [
        name for name in missing
        if not name.startswith(("visual_adapter.", "text_adapter."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Baseline checkpoint 不兼容，missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )

    print(f"Loaded baseline : {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Baseline epoch  : {checkpoint['epoch']}")
    metrics = checkpoint.get("metrics", {})
    if isinstance(metrics, dict) and "mR" in metrics:
        print(f"Baseline Val mR : {metrics['mR']:.2f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # 输出与日志
    output_dir = Path(config["output"]["root"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path, log_file = setup_logger(output_dir=output_dir, prefix="train")
    metrics_path = output_dir / "metrics.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CLIP ADAPTER TRAINING")
    print("=" * 70)
    print(f"Start time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file  : {args.config}")
    print(f"Experiment   : {config['experiment']['name']}")
    print(f"Dataset      : {config['dataset']['name']}")
    print(f"Backbone     : {config['model']['backbone']}")
    print(f"Pretrained   : {config['model']['pretrained']}")
    print(f"Device       : {device}")
    print(f"Seed         : {seed}")
    print(f"Output dir   : {output_dir}")
    print(f"Log file     : {log_path}")
    print(f"Metrics file : {metrics_path}")

    # 模型与数据
    print("\nBuilding CLIP model...")
    model = CLIPRetrieval(config["model"])

    init_checkpoint = config["model"].get("init_checkpoint")
    if model.freeze_backbone and not init_checkpoint:
        raise ValueError(
            "freeze_backbone=true 时必须提供 model.init_checkpoint，"
            "避免直接冻结原始 OpenAI CLIP。"
        )
    if init_checkpoint:
        load_initial_checkpoint(model, init_checkpoint)

    model = model.to(device)

    print("Building datasets...")
    train_dataset, val_dataset = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_train,
        eval_transform=model.backbone.preprocess_val,
    )

    train_cfg = config["training"]
    train_batch_size = train_cfg["batch_size"]
    eval_batch_size = train_cfg.get("eval_batch_size", 128)
    text_batch_size = train_cfg.get("text_batch_size", 256)
    num_workers = train_cfg.get("num_workers", 8)
    epochs = train_cfg["epochs"]
    eval_every = train_cfg.get("eval_every", 1)
    max_steps = train_cfg.get("max_steps")
    log_interval = int(train_cfg.get("log_interval", 50))
    entity_loss_weight = float(train_cfg.get("entity_loss_weight", 0.1))

    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("training.max_steps 必须为正整数或 null")
    if log_interval <= 0:
        raise ValueError("training.log_interval 必须为正整数")
    if entity_loss_weight < 0:
        raise ValueError("training.entity_loss_weight 必须 >= 0")

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

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Train pairs      : {len(train_dataset)}")
    print(f"Val images       : {len(val_dataset)}")
    print(f"Val captions     : {len(val_dataset.text)}")
    print(f"Train batches    : {len(train_loader)}")
    print(f"Train batch size : {train_batch_size}")
    print(f"Eval batch size  : {eval_batch_size}")
    print(f"Text batch size  : {text_batch_size}")

    # Loss、优化器、学习率
    criterion = CLIPLoss()
    optimizer = build_optimizer(
        model=model,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    steps_per_epoch = len(train_loader)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)

    total_steps = steps_per_epoch * epochs
    warmup_ratio = config["scheduler"].get("warmup_ratio", 0.05)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )

    print("\n" + "=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    print(f"Epochs         : {epochs}")
    print(f"Learning rate  : {config['optimizer']['lr']}")
    print(f"Weight decay   : {config['optimizer']['weight_decay']}")
    print(f"Steps / epoch  : {steps_per_epoch}")
    print(f"Total steps    : {total_steps}")
    print(f"Warmup ratio   : {warmup_ratio}")
    print(f"Warmup steps   : {warmup_steps}")
    print(f"Eval every     : {eval_every} epoch(s)")
    print(f"Max train steps: {max_steps if max_steps is not None else 'full epoch'}")
    print(f"Log interval   : {log_interval}")
    print(f"Entity loss wt : {entity_loss_weight}")
    print(f"Init checkpoint: {init_checkpoint if init_checkpoint else 'none'}")
    print(f"Freeze backbone: {model.freeze_backbone}")
    print(f"Visual adapter : {config['model'].get('visual_adapter_dim', 128)}")
    print(f"Text adapter   : {config['model'].get('text_adapter_dim', 64)}")

    best_mr = float("-inf")
    best_epoch = -1
    early_stop_patience = train_cfg.get("early_stop_patience")
    early_stop_min_delta = train_cfg.get("early_stop_min_delta", 0.0)
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 70)

        epoch_start = datetime.now()
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            max_steps=max_steps,
            log_interval=log_interval,
            entity_loss_weight=entity_loss_weight,
        )
        train_seconds = (datetime.now() - epoch_start).total_seconds()

        print("\n" + "-" * 70)
        print(f"EPOCH {epoch} TRAIN SUMMARY")
        print("-" * 70)
        print(f"Average loss   : {train_stats['loss']:.6f}")
        print(f"Average global : {train_stats['loss_global']:.6f}")
        print(f"Average entity : {train_stats['loss_entity']:.6f}")
        print(f"Average I2T    : {train_stats['loss_i2t']:.6f}")
        print(f"Average T2I    : {train_stats['loss_t2i']:.6f}")
        print(f"Current LR     : {train_stats['lr']:.8f}")
        print(f"Logit scale    : {train_stats['logit_scale']:.4f}")
        print(f"Train time     : {train_seconds:.2f} s")

        # 所有 epoch 都记录训练信息
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_loss_global": float(train_stats["loss_global"]),
            "train_loss_entity": float(train_stats["loss_entity"]),
            "entity_loss_weight": entity_loss_weight,
            "train_loss_i2t": float(train_stats["loss_i2t"]),
            "train_loss_t2i": float(train_stats["loss_t2i"]),
            "lr": float(train_stats["lr"]),
            "logit_scale": float(train_stats["logit_scale"]),
            "train_seconds": train_seconds,
            "validated": False,
        }

        if epoch % eval_every != 0:
            append_jsonl(metrics_path, epoch_record)
            log_file.flush()
            continue

        # 验证
        print("\n" + "=" * 70)
        print(f"VALIDATION - EPOCH {epoch}")
        print("=" * 70)

        val_start = datetime.now()
        metrics, _ = evaluate_retrieval(
            model=model,
            data_loader=val_loader,
            dataset=val_dataset,
            device=device,
            text_batch_size=text_batch_size,
        )
        validation_seconds = (datetime.now() - val_start).total_seconds()
        current_mr = metrics["mR"]

        print("\nImage -> Text")
        print(f"R@1  : {metrics['i2t_r1']:.2f}")
        print(f"R@5  : {metrics['i2t_r5']:.2f}")
        print(f"R@10 : {metrics['i2t_r10']:.2f}")
        print("\nText -> Image")
        print(f"R@1  : {metrics['t2i_r1']:.2f}")
        print(f"R@5  : {metrics['t2i_r5']:.2f}")
        print(f"R@10 : {metrics['t2i_r10']:.2f}")
        print(f"\nI2T mean : {metrics['i2t_mean']:.2f}")
        print(f"T2I mean : {metrics['t2i_mean']:.2f}")
        print(f"\nVAL mR   : {current_mr:.2f}")
        print(f"Val time : {validation_seconds:.2f} s")

        is_best = current_mr > best_mr + early_stop_min_delta
        if is_best:
            best_mr = current_mr
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record.update({
            "validated": True,
            "val_i2t_r1": float(metrics["i2t_r1"]),
            "val_i2t_r5": float(metrics["i2t_r5"]),
            "val_i2t_r10": float(metrics["i2t_r10"]),
            "val_t2i_r1": float(metrics["t2i_r1"]),
            "val_t2i_r5": float(metrics["t2i_r5"]),
            "val_t2i_r10": float(metrics["t2i_r10"]),
            "val_i2t_mean": float(metrics["i2t_mean"]),
            "val_t2i_mean": float(metrics["t2i_mean"]),
            "val_mR": float(current_mr),
            "val_i2t_medr": float(metrics["i2t_medr"]),
            "val_t2i_medr": float(metrics["t2i_medr"]),
            "val_i2t_meanr": float(metrics.get("i2t_meanr", 0.0)),
            "val_t2i_meanr": float(metrics.get("t2i_meanr", 0.0)),
            "validation_seconds": validation_seconds,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_val_mR": float(best_mr),
        })
        append_jsonl(metrics_path, epoch_record)

        # Checkpoint
        last_path = output_dir / "last.pth"
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
        print(f"\nSaved last checkpoint: {last_path}")

        if is_best:
            best_path = output_dir / "best.pth"
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            print("\n>>> NEW BEST MODEL")
            print(f"Epoch : {best_epoch}")
            print(f"mR    : {best_mr:.2f}")
            print(f"Saved : {best_path}")

        if (
            early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print("\n" + "=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(f"No improvement for {epochs_without_improvement} validation rounds.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best Val mR: {best_mr:.2f}")
            break

        print(f"\nBest epoch so far : {best_epoch}")
        print(f"Best Val mR       : {best_mr:.2f}")
        print(f"Epoch total time  : {(datetime.now() - epoch_start).total_seconds():.2f} s")
        log_file.flush()

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Finish time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best Val mR  : {best_mr:.2f}")
    print(f"Best model   : {output_dir / 'best.pth'}")
    print(f"Log file     : {log_path}")
    print(f"Metrics file : {metrics_path}")
    log_file.flush()


if __name__ == "__main__":
    main()
