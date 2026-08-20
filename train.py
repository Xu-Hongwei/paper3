import argparse
from datetime import datetime
from pathlib import Path

import torch

from datasets import create_dataset, create_loader
from engine import build_optimizer, build_scheduler, train_one_epoch
from evaluation import evaluate_retrieval
from losses import CLIPLoss, CrossCategoryMarginLoss
from models import CLIPRetrieval
from utils import (
    append_jsonl,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean CLIP Training for RSITR"
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def print_retrieval_metrics(metrics):
    """打印标准图文检索指标。"""
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
    print(f"VAL mR   : {metrics['mR']:.2f}")


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_cfg = config["training"]
    output_dir = Path(config["output"]["root"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path, log_file = setup_logger(output_dir=output_dir, prefix="train")
    metrics_path = output_dir / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CLEAN CLIP TRAINING")
    print("=" * 70)
    print(f"Start time   : {datetime.now():%Y-%m-%d %H:%M:%S}")
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

    # --------------------------------------------------
    # Model / Dataset
    # --------------------------------------------------
    print("\nBuilding Clean CLIP model...")
    model = CLIPRetrieval(config["model"]).to(device)

    print("Building datasets...")
    train_dataset, val_dataset = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_train,
        eval_transform=model.backbone.preprocess_val,
    )

    batch_size = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg.get("eval_batch_size", 128))
    text_batch_size = int(train_cfg.get("text_batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 8))
    epochs = int(train_cfg["epochs"])
    eval_every = int(train_cfg.get("eval_every", 1))
    log_interval = int(train_cfg.get("log_interval", 50))
    max_steps = train_cfg.get("max_steps")

    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("training.max_steps 必须为正整数或 null。")

    if eval_every <= 0:
        raise ValueError("training.eval_every 必须 > 0。")
    if log_interval <= 0:
        raise ValueError("training.log_interval 必须 > 0。")

    train_loader = create_loader(
        train_dataset,
        batch_size=batch_size,
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
    print(f"Train batch size : {batch_size}")
    print(f"Eval batch size  : {eval_batch_size}")
    print(f"Text batch size  : {text_batch_size}")

    # --------------------------------------------------
    # Loss / Optimizer / Scheduler
    # --------------------------------------------------
    criterion = CLIPLoss()

    category_cfg = config.get("category_margin", {})
    category_enabled = bool(category_cfg.get("enabled", False))
    category_margin = float(category_cfg.get("margin", 0.10))
    category_t2i_weight = float(category_cfg.get("t2i_weight", 0.0))
    category_i2t_weight = float(category_cfg.get("i2t_weight", 0.0))

    legacy_reliable_t2i = bool(
        category_cfg.get("reliable_t2i", False)
    )
    reliability_mode = category_cfg.get("reliability_mode")
    if reliability_mode is None:
        reliability_mode = (
            "post_gate"
            if legacy_reliable_t2i
            else "none"
        )
    reliability_mode = str(reliability_mode).lower()

    valid_reliability_modes = {
        "none",
        "post_gate",
        "reliable_mining",
    }
    if reliability_mode not in valid_reliability_modes:
        raise ValueError(
            "category_margin.reliability_mode 必须为 "
            "none / post_gate / reliable_mining，"
            f"当前为 {reliability_mode!r}。"
        )

    if (
        legacy_reliable_t2i
        and reliability_mode == "none"
    ):
        raise ValueError(
            "reliable_t2i=True 与 reliability_mode=none 冲突。"
        )

    reliable_t2i = reliability_mode != "none"

    support_threshold = float(
        category_cfg.get("support_threshold", 0.0)
    )
    category_threshold = float(
        category_cfg.get("category_threshold", 0.0)
    )
    support_cache_path = category_cfg.get("support_cache")

    if category_margin < 0:
        raise ValueError("category_margin.margin 必须 >= 0。")
    if category_t2i_weight < 0 or category_i2t_weight < 0:
        raise ValueError("category margin 权重必须 >= 0。")
    if reliable_t2i and not category_enabled:
        raise ValueError(
            "reliability_mode!=none 时必须启用 category_margin.enabled。"
        )
    if reliable_t2i and category_t2i_weight <= 0:
        raise ValueError(
            "reliability_mode!=none 时 category_margin.t2i_weight 必须 > 0。"
        )
    if reliable_t2i and not support_cache_path:
        raise ValueError(
            "reliability_mode!=none 时必须提供 category_margin.support_cache。"
        )

    category_criterion = (
        CrossCategoryMarginLoss(
            margin=category_margin,
            reliable_t2i=reliable_t2i,
            support_threshold=support_threshold,
            category_threshold=category_threshold,
            reliability_mode=reliability_mode,
        )
        if category_enabled
        else None
    )

    # Frozen teacher support cache 只加载一次，整个训练过程保持不变。
    category_support_cache = None
    if reliable_t2i:
        support_cache_path = Path(support_cache_path)
        if not support_cache_path.is_file():
            raise FileNotFoundError(
                f"Category support cache not found: {support_cache_path}"
            )

        category_support_cache = torch.load(
            support_cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(category_support_cache, dict):
            raise TypeError("category support cache 必须是 dict。")

        required_cache_keys = {
            "caption_support",
            "image_support",
        }
        missing_cache_keys = sorted(
            required_cache_keys - set(category_support_cache)
        )
        if missing_cache_keys:
            raise ValueError(
                f"Category support cache missing keys: "
                f"{missing_cache_keys}"
            )

        caption_support = category_support_cache["caption_support"]
        image_support = category_support_cache["image_support"]

        if caption_support.ndim != 2:
            raise ValueError(
                "caption_support must have shape [N_pair, C]."
            )
        if image_support.ndim != 2:
            raise ValueError(
                "image_support must have shape [N_image, C]."
            )
        if caption_support.shape[0] != len(train_dataset):
            raise ValueError(
                "caption support / train dataset length mismatch: "
                f"{caption_support.shape[0]} vs {len(train_dataset)}"
            )
        if image_support.shape[0] != train_dataset.num_images:
            raise ValueError(
                "image support / train image count mismatch: "
                f"{image_support.shape[0]} vs {train_dataset.num_images}"
            )
        if caption_support.shape[1] != image_support.shape[1]:
            raise ValueError(
                "caption/image support category dimension mismatch."
            )

        # cache 中若保存了索引，训练前先做一次全量对齐检查。
        if "sample_image_ids" in category_support_cache:
            cached = category_support_cache[
                "sample_image_ids"
            ].long().cpu()
            current = torch.tensor(
                train_dataset.image_ids,
                dtype=torch.long,
            )
            if not torch.equal(cached, current):
                raise RuntimeError(
                    "Support cache sample_image_ids 与当前训练集不一致。"
                )

        if "sample_category_ids" in category_support_cache:
            cached = category_support_cache[
                "sample_category_ids"
            ].long().cpu()
            current = torch.tensor(
                train_dataset.category_ids,
                dtype=torch.long,
            )
            if not torch.equal(cached, current):
                raise RuntimeError(
                    "Support cache sample_category_ids 与当前训练集不一致。"
                )

        if "image_category_ids" in category_support_cache:
            cached = category_support_cache[
                "image_category_ids"
            ].long().cpu()
            current = torch.tensor(
                train_dataset.image_category_ids,
                dtype=torch.long,
            )
            if not torch.equal(cached, current):
                raise RuntimeError(
                    "Support cache image_category_ids 与当前训练集不一致。"
                )

    optimizer = build_optimizer(
        model,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    steps_per_epoch = len(train_loader)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)

    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(config["scheduler"].get("warmup_ratio", 0.05))
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
    )

    print("\n" + "=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    print(f"Epochs          : {epochs}")
    print(f"Learning rate   : {config['optimizer']['lr']}")
    print(f"Weight decay    : {config['optimizer']['weight_decay']}")
    print(f"Steps / epoch   : {steps_per_epoch}")
    print(f"Total steps     : {total_steps}")
    print(f"Warmup ratio    : {warmup_ratio}")
    print(f"Warmup steps    : {warmup_steps}")
    print(f"Eval every      : {eval_every} epoch(s)")
    print(f"Max train steps : {max_steps if max_steps is not None else 'full epoch'}")
    print(f"Log interval    : {log_interval}")
    if reliability_mode == "reliable_mining":
        print(
            "Objective       : Multi-positive CLIP + "
            "Reliability-Guided T2I Hard-Negative Mining"
        )
    elif reliability_mode == "post_gate":
        print(
            "Objective       : Multi-positive CLIP + "
            "Post-Gate Reliable T2I Cross-Category Margin"
        )
    elif category_enabled:
        print(
            "Objective       : Multi-positive CLIP + "
            "Fixed Cross-Category Margin"
        )
    else:
        print("Objective       : Multi-positive CLIP loss")

    print(
        "Category margin : "
        f"{'enabled' if category_enabled else 'disabled'}"
    )
    if category_enabled:
        print(f"Category m      : {category_margin:.4f}")
        print(f"Category T2I λ  : {category_t2i_weight:.4f}")
        print(f"Category I2T λ  : {category_i2t_weight:.4f}")
        print(f"Reliability mode: {reliability_mode}")

        if reliable_t2i:
            print(f"Support thresh : {support_threshold:+.4f}")
            print(f"Category thresh: {category_threshold:+.4f}")
            print(f"Support cache  : {support_cache_path}")
            print(
                "Support shape  : "
                f"{tuple(category_support_cache['caption_support'].shape)} / "
                f"{tuple(category_support_cache['image_support'].shape)}"
            )

    print("Best criterion  : maximum Validation mR")

    # --------------------------------------------------
    # Best model / Early stopping
    # --------------------------------------------------
    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    early_stop_patience = train_cfg.get("early_stop_patience")
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
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
            category_criterion=category_criterion,
            category_t2i_weight=category_t2i_weight,
            category_i2t_weight=category_i2t_weight,
            category_support_cache=category_support_cache,
        )

        train_seconds = (datetime.now() - epoch_start).total_seconds()

        print("\n" + "-" * 70)
        print(f"EPOCH {epoch} TRAIN SUMMARY")
        print("-" * 70)
        print(f"Average loss : {train_stats['loss']:.6f}")
        print(f"Average CLIP : {train_stats['clip_loss']:.6f}")
        print(f"Average I2T  : {train_stats['loss_i2t']:.6f}")
        print(f"Average T2I  : {train_stats['loss_t2i']:.6f}")

        if category_enabled:
            print(f"Cat T2I loss : {train_stats['cat_loss_t2i']:.6f}")
            print(f"Cat I2T loss : {train_stats['cat_loss_i2t']:.6f}")

            if reliable_t2i:
                if reliability_mode == "reliable_mining":
                    print(
                        f"Cat T2I mine : "
                        f"{train_stats['cat_t2i_reliable']}/"
                        f"{train_stats['cat_t2i_valid']} "
                        f"({train_stats['cat_t2i_reliable_ratio']:.2%})"
                    )
                    print(
                        f"Cat T2I relN : "
                        f"avg={train_stats['cat_t2i_avg_reliable_negatives']:.2f} "
                        f"| none={train_stats['cat_t2i_no_reliable']}/"
                        f"{train_stats['cat_t2i_valid']} "
                        f"({train_stats['cat_t2i_no_reliable_ratio']:.2%})"
                    )
                    print(
                        f"Cat T2I repl : "
                        f"{train_stats['cat_t2i_replacement']}/"
                        f"{train_stats['cat_t2i_valid']} "
                        f"({train_stats['cat_t2i_replacement_ratio']:.2%})"
                    )
                    print(
                        f"Cat T2I neg  : "
                        f"fixed={train_stats['cat_t2i_fixed_hard_neg_sim']:.4f}, "
                        f"reliable={train_stats['cat_t2i_hard_neg_sim']:.4f}"
                    )
                else:
                    print(
                        f"Cat T2I rel. : "
                        f"{train_stats['cat_t2i_reliable']}/"
                        f"{train_stats['cat_t2i_valid']} "
                        f"({train_stats['cat_t2i_reliable_ratio']:.2%})"
                    )

                print(
                    f"Cat T2I pre  : "
                    f"{train_stats['cat_t2i_active_before_gate']}/"
                    f"{train_stats['cat_t2i_valid']} "
                    f"({train_stats['cat_t2i_active_before_gate_ratio']:.2%})"
                )
                print(
                    f"Cat T2I post : "
                    f"{train_stats['cat_t2i_active']}/"
                    f"{train_stats['cat_t2i_reliable']} "
                    f"({train_stats['cat_t2i_active_reliable_ratio']:.2%})"
                )
                print(
                    "Cat T2I fixed A/B/C/D: "
                    f"{train_stats['cat_t2i_region_a']}/"
                    f"{train_stats['cat_t2i_region_b']}/"
                    f"{train_stats['cat_t2i_region_c']}/"
                    f"{train_stats['cat_t2i_region_d']} | "
                    f"{train_stats['cat_t2i_region_a_ratio']:.2%}/"
                    f"{train_stats['cat_t2i_region_b_ratio']:.2%}/"
                    f"{train_stats['cat_t2i_region_c_ratio']:.2%}/"
                    f"{train_stats['cat_t2i_region_d_ratio']:.2%}"
                )
                print(
                    f"Cat T2I G    : "
                    f"Gsup={train_stats['cat_t2i_g_sup']:+.4f}, "
                    f"Gcat={train_stats['cat_t2i_g_cat']:+.4f}"
                )
            else:
                print(
                    f"Cat T2I act. : "
                    f"{train_stats['cat_t2i_active']}/"
                    f"{train_stats['cat_t2i_valid']} "
                    f"({train_stats['cat_t2i_active_ratio']:.2%})"
                )

            print(
                f"Cat I2T act. : "
                f"{train_stats['cat_i2t_active']}/"
                f"{train_stats['cat_i2t_valid']} "
                f"({train_stats['cat_i2t_active_ratio']:.2%})"
            )
            print(
                f"Cat T2I sim  : "
                f"pos={train_stats['cat_t2i_pos_sim']:.4f}, "
                f"neg={train_stats['cat_t2i_hard_neg_sim']:.4f}"
            )
            print(
                f"Cat I2T sim  : "
                f"pos={train_stats['cat_i2t_pos_sim']:.4f}, "
                f"neg={train_stats['cat_i2t_hard_neg_sim']:.4f}"
            )

        print(f"Current LR   : {train_stats['lr']:.8f}")
        print(f"Logit scale  : {train_stats['logit_scale']:.4f}")
        print(f"Train time   : {train_seconds:.2f} s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_clip_loss": float(train_stats["clip_loss"]),
            "train_loss_i2t": float(train_stats["loss_i2t"]),
            "train_loss_t2i": float(train_stats["loss_t2i"]),
            "train_cat_loss_t2i": float(train_stats["cat_loss_t2i"]),
            "train_cat_loss_i2t": float(train_stats["cat_loss_i2t"]),
            "train_cat_t2i_valid": int(train_stats["cat_t2i_valid"]),
            "train_cat_i2t_valid": int(train_stats["cat_i2t_valid"]),
            "train_cat_t2i_active": int(train_stats["cat_t2i_active"]),
            "train_cat_i2t_active": int(train_stats["cat_i2t_active"]),
            "train_cat_t2i_active_ratio": float(
                train_stats["cat_t2i_active_ratio"]
            ),
            "train_cat_i2t_active_ratio": float(
                train_stats["cat_i2t_active_ratio"]
            ),
            "train_cat_t2i_reliable": int(
                train_stats["cat_t2i_reliable"]
            ),
            "train_cat_t2i_reliable_ratio": float(
                train_stats["cat_t2i_reliable_ratio"]
            ),
            "train_cat_t2i_active_before_gate": int(
                train_stats["cat_t2i_active_before_gate"]
            ),
            "train_cat_t2i_active_before_gate_ratio": float(
                train_stats["cat_t2i_active_before_gate_ratio"]
            ),
            "train_cat_t2i_active_reliable_ratio": float(
                train_stats["cat_t2i_active_reliable_ratio"]
            ),
            "train_cat_t2i_region_a": int(
                train_stats["cat_t2i_region_a"]
            ),
            "train_cat_t2i_region_b": int(
                train_stats["cat_t2i_region_b"]
            ),
            "train_cat_t2i_region_c": int(
                train_stats["cat_t2i_region_c"]
            ),
            "train_cat_t2i_region_d": int(
                train_stats["cat_t2i_region_d"]
            ),
            "train_cat_t2i_region_a_ratio": float(
                train_stats["cat_t2i_region_a_ratio"]
            ),
            "train_cat_t2i_region_b_ratio": float(
                train_stats["cat_t2i_region_b_ratio"]
            ),
            "train_cat_t2i_region_c_ratio": float(
                train_stats["cat_t2i_region_c_ratio"]
            ),
            "train_cat_t2i_region_d_ratio": float(
                train_stats["cat_t2i_region_d_ratio"]
            ),
            "train_cat_t2i_g_sup": float(
                train_stats["cat_t2i_g_sup"]
            ),
            "train_cat_t2i_g_cat": float(
                train_stats["cat_t2i_g_cat"]
            ),
            "train_cat_t2i_reliability_mode": train_stats[
                "cat_t2i_reliability_mode"
            ],
            "train_cat_t2i_reliable_candidate_total": int(
                train_stats["cat_t2i_reliable_candidate_total"]
            ),
            "train_cat_t2i_avg_reliable_negatives": float(
                train_stats["cat_t2i_avg_reliable_negatives"]
            ),
            "train_cat_t2i_no_reliable": int(
                train_stats["cat_t2i_no_reliable"]
            ),
            "train_cat_t2i_no_reliable_ratio": float(
                train_stats["cat_t2i_no_reliable_ratio"]
            ),
            "train_cat_t2i_replacement": int(
                train_stats["cat_t2i_replacement"]
            ),
            "train_cat_t2i_replacement_ratio": float(
                train_stats["cat_t2i_replacement_ratio"]
            ),
            "train_cat_t2i_fixed_hard_neg_sim": float(
                train_stats["cat_t2i_fixed_hard_neg_sim"]
            ),
            "train_cat_t2i_pos_sim": float(train_stats["cat_t2i_pos_sim"]),
            "train_cat_i2t_pos_sim": float(train_stats["cat_i2t_pos_sim"]),
            "train_cat_t2i_hard_neg_sim": float(
                train_stats["cat_t2i_hard_neg_sim"]
            ),
            "train_cat_i2t_hard_neg_sim": float(
                train_stats["cat_i2t_hard_neg_sim"]
            ),
            "lr": float(train_stats["lr"]),
            "logit_scale": float(train_stats["logit_scale"]),
            "train_seconds": train_seconds,
            "validated": False,
        }

        metrics = None
        is_best = False

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        if epoch % eval_every == 0:
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

            print_retrieval_metrics(metrics)
            print(f"Val time : {validation_seconds:.2f} s")

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
                "val_mR": float(metrics["mR"]),
                "val_i2t_medr": float(metrics["i2t_medr"]),
                "val_t2i_medr": float(metrics["t2i_medr"]),
                "val_i2t_meanr": float(metrics.get("i2t_meanr", 0.0)),
                "val_t2i_meanr": float(metrics.get("t2i_meanr", 0.0)),
                "validation_seconds": validation_seconds,
            })

            current_score = float(metrics["mR"])
            is_best = current_score > best_score + early_stop_min_delta

            if is_best:
                best_score = current_score
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        epoch_record.update({
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_val_mR": float(best_score) if best_epoch >= 0 else None,
        })
        append_jsonl(metrics_path, epoch_record)

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------
        checkpoint_metrics = {
            "train_loss": float(train_stats["loss"]),
            "train_clip_loss": float(train_stats["clip_loss"]),
            "train_loss_i2t": float(train_stats["loss_i2t"]),
            "train_loss_t2i": float(train_stats["loss_t2i"]),
            "train_cat_loss_t2i": float(train_stats["cat_loss_t2i"]),
            "train_cat_loss_i2t": float(train_stats["cat_loss_i2t"]),
            "train_cat_t2i_active_ratio": float(
                train_stats["cat_t2i_active_ratio"]
            ),
            "train_cat_i2t_active_ratio": float(
                train_stats["cat_i2t_active_ratio"]
            ),
            "train_cat_t2i_reliable_ratio": float(
                train_stats["cat_t2i_reliable_ratio"]
            ),
            "train_cat_t2i_active_before_gate_ratio": float(
                train_stats["cat_t2i_active_before_gate_ratio"]
            ),
            "train_cat_t2i_active_reliable_ratio": float(
                train_stats["cat_t2i_active_reliable_ratio"]
            ),
            "train_cat_t2i_region_a_ratio": float(
                train_stats["cat_t2i_region_a_ratio"]
            ),
            "train_cat_t2i_region_b_ratio": float(
                train_stats["cat_t2i_region_b_ratio"]
            ),
            "train_cat_t2i_region_c_ratio": float(
                train_stats["cat_t2i_region_c_ratio"]
            ),
            "train_cat_t2i_region_d_ratio": float(
                train_stats["cat_t2i_region_d_ratio"]
            ),
            "train_cat_t2i_reliability_mode": train_stats[
                "cat_t2i_reliability_mode"
            ],
            "train_cat_t2i_avg_reliable_negatives": float(
                train_stats["cat_t2i_avg_reliable_negatives"]
            ),
            "train_cat_t2i_no_reliable_ratio": float(
                train_stats["cat_t2i_no_reliable_ratio"]
            ),
            "train_cat_t2i_replacement_ratio": float(
                train_stats["cat_t2i_replacement_ratio"]
            ),
            "train_cat_t2i_fixed_hard_neg_sim": float(
                train_stats["cat_t2i_fixed_hard_neg_sim"]
            ),
            "train_cat_t2i_reliable_hard_neg_sim": float(
                train_stats["cat_t2i_hard_neg_sim"]
            ),
        }
        if metrics is not None:
            checkpoint_metrics.update(metrics)

        last_path = output_dir / "last.pth"
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=checkpoint_metrics,
            config=config,
        )
        print(f"\nSaved last checkpoint: {last_path}")

        if is_best:
            best_path = output_dir / "best.pth"
            best_metrics = dict(checkpoint_metrics)
            best_metrics["selection_metric"] = "val_mR"

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=best_metrics,
                config=config,
            )

            print("\n>>> NEW BEST MODEL")
            print(f"Epoch  : {best_epoch}")
            print(f"Val mR : {best_score:.2f}")
            print(f"Saved  : {best_path}")

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------
        if (
            early_stop_patience is not None
            and metrics is not None
            and epochs_without_improvement >= int(early_stop_patience)
        ):
            print("\n" + "=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(
                f"No improvement for {epochs_without_improvement} "
                "validation rounds."
            )
            print(f"Best Val epoch: {best_epoch}")
            print(f"Best Val mR   : {best_score:.2f}")
            break

        print(f"\nBest Val epoch : {best_epoch}")
        print(
            f"Best Val mR    : {best_score:.2f}"
            if best_epoch >= 0
            else "Best Val mR    : not evaluated yet"
        )
        print(
            f"Epoch total time: "
            f"{(datetime.now() - epoch_start).total_seconds():.2f} s"
        )
        log_file.flush()

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Finish time : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Best Val epoch : {best_epoch}")

    if best_epoch >= 0:
        print(f"Best Val mR    : {best_score:.2f}")
        print(f"Best model     : {output_dir / 'best.pth'}")
    else:
        print("Best Val mR    : not evaluated")

    print(f"Last model     : {output_dir / 'last.pth'}")
    print(f"Log file       : {log_path}")
    print(f"Metrics file   : {metrics_path}")

    log_file.flush()


if __name__ == "__main__":
    main()
