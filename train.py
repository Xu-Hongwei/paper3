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
from datasets import create_dataset, create_loader
from models import CLIPRetrieval
from losses import CLIPLoss
from engine import build_optimizer, build_scheduler, train_one_epoch
from evaluation import evaluate_retrieval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean CLIP / B1b Local Self-Distillation Training for RSITR"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_initial_checkpoint(
    model,
    checkpoint_path,
):
    """
    加载 Clean CLIP checkpoint。

    支持：
        1. 纯 RSICD-finetuned CLIP baseline；
        2. 当前 Clean B1b checkpoint。

    纯 CLIP baseline 不包含 local_teacher_visual，
    因此加载完成后需要调用 model.sync_local_teacher()。

    Returns:
        checkpoint:
            原始 checkpoint。

        teacher_loaded:
            checkpoint 是否已经包含 Frozen Teacher。
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initial checkpoint 不存在: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    teacher_loaded = any(
        name.startswith(
            "local_teacher_visual."
        )
        for name in state_dict
    )

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    # 纯 CLIP baseline 唯一允许缺失的，
    # 是 B1b 新增的 Frozen Teacher。
    invalid_missing = [
        name
        for name in missing
        if not name.startswith(
            "local_teacher_visual."
        )
    ]

    if invalid_missing or unexpected:
        raise RuntimeError(
            "Initial checkpoint 与 Clean CLIP 模型不兼容。\n"
            f"missing={invalid_missing}\n"
            f"unexpected={unexpected}\n"
            "请确认 init_checkpoint 指向纯 CLIP baseline "
            "或当前 Clean B1b checkpoint，"
            "不要使用旧 Adapter / Local Head checkpoint。"
        )

    print(
        f"Loaded checkpoint : "
        f"{checkpoint_path}"
    )

    teacher_missing = sum(
        name.startswith(
            "local_teacher_visual."
        )
        for name in missing
    )

    if teacher_missing:
        print(
            f"New teacher params : "
            f"{teacher_missing} "
            "(will sync from initial Student)"
        )

    if isinstance(
        checkpoint,
        dict,
    ):
        if "epoch" in checkpoint:
            print(
                f"Checkpoint epoch   : "
                f"{checkpoint['epoch']}"
            )

        metrics = checkpoint.get(
            "metrics",
            {},
        )

        if (
            isinstance(metrics, dict)
            and "mR" in metrics
        ):
            try:
                print(
                    f"Checkpoint Val mR  : "
                    f"{float(metrics['mR']):.2f}"
                )
            except (TypeError, ValueError):
                pass

    return checkpoint, teacher_loaded


def main():
    args = parse_args()
    config = load_config(
        args.config
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )
    set_seed(
        seed
    )

    train_cfg = config[
        "training"
    ]

    local_distill_only = bool(
        train_cfg.get(
            "local_distill_only",
            False,
        )
    )
    local_distill_weight = float(
        train_cfg.get(
            "local_distill_weight",
            0.0,
        )
    )
    global_preserve_weight = float(
        train_cfg.get(
            "global_preserve_weight",
            0.0,
        )
    )

    # Clean CLIP 当前阶段不再训练旧 Entity Grounding。
    entity_loss_weight = float(
        train_cfg.get(
            "entity_loss_weight",
            0.0,
        )
    )

    local_trainable_blocks = int(
        config["model"].get(
            "local_trainable_blocks",
            0,
        )
    )

    # ==================================================
    # 配置合法性检查
    # ==================================================
    if entity_loss_weight != 0:
        raise ValueError(
            "Clean CLIP B1b 已移除旧 Entity Grounding，"
            "training.entity_loss_weight 必须为 0。"
        )

    if local_distill_weight < 0:
        raise ValueError(
            "training.local_distill_weight "
            "必须 >= 0"
        )

    if global_preserve_weight < 0:
        raise ValueError(
            "training.global_preserve_weight "
            "必须 >= 0"
        )

    if local_trainable_blocks < 0:
        raise ValueError(
            "model.local_trainable_blocks "
            "必须 >= 0"
        )

    if local_distill_only:
        if local_distill_weight <= 0:
            raise ValueError(
                "local_distill_only=true 时 "
                "training.local_distill_weight 必须 > 0"
            )

        if local_trainable_blocks <= 0:
            raise ValueError(
                "Clean B1b 必须设置 "
                "model.local_trainable_blocks > 0"
            )

        if global_preserve_weight <= 0:
            raise ValueError(
                "Clean B1b 直接训练 Vision Block 时必须设置 "
                "training.global_preserve_weight > 0"
            )

    if (
        global_preserve_weight > 0
        and local_distill_weight <= 0
    ):
        raise ValueError(
            "global_preserve_weight > 0 时 "
            "必须同时启用 Local Self-Distillation"
        )

    # ==================================================
    # 输出与日志
    # ==================================================
    output_dir = (
        Path(
            config["output"]["root"]
        )
        / config["experiment"]["name"]
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path, log_file = setup_logger(
        output_dir=output_dir,
        prefix="train",
    )
    metrics_path = (
        output_dir
        / "metrics.jsonl"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    mode_name = (
        "CLEAN CLIP B1b LOCAL SELF-DISTILLATION"
        if local_distill_only
        else "CLEAN CLIP TRAINING"
    )

    print(
        "=" * 70
    )
    print(
        mode_name
    )
    print(
        "=" * 70
    )
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
    # 模型
    # ==================================================
    print(
        "\nBuilding Clean CLIP model..."
    )

    model = CLIPRetrieval(
        config["model"]
    )

    init_checkpoint = (
        config["model"].get(
            "init_checkpoint"
        )
    )

    if (
        model.freeze_backbone
        and not init_checkpoint
    ):
        raise ValueError(
            "freeze_backbone=true 时必须提供 "
            "model.init_checkpoint。"
            "Clean B1b 应从 RSICD-finetuned CLIP "
            "baseline 初始化，而不是直接冻结原始 OpenAI CLIP。"
        )

    if (
        local_distill_only
        and not model.freeze_backbone
    ):
        raise ValueError(
            "Clean B1b local_distill_only=true 时必须设置 "
            "model.freeze_backbone=true"
        )

    teacher_loaded = False

    if init_checkpoint:
        _, teacher_loaded = (
            load_initial_checkpoint(
                model,
                init_checkpoint,
            )
        )

    # --------------------------------------------------
    # Frozen Teacher
    #
    # 纯 CLIP baseline 没有 Teacher：
    #     baseline Student -> sync -> Frozen Teacher
    #
    # B1b checkpoint 已含 Teacher：
    #     直接保留原 Teacher，不重新覆盖。
    # --------------------------------------------------
    if (
        local_distill_weight > 0
        and hasattr(
            model,
            "sync_local_teacher",
        )
    ):
        if teacher_loaded:
            print(
                "Local teacher     : "
                "loaded from checkpoint | frozen"
            )
        else:
            model.sync_local_teacher()
            print(
                "Local teacher     : "
                "synced from initial Student | frozen"
            )

        teacher = getattr(
            model,
            "local_teacher_visual",
            None,
        )

        if teacher is None:
            raise RuntimeError(
                "Clean B1b 缺少 local_teacher_visual。"
            )

        if any(
            param.requires_grad
            for param in teacher.parameters()
        ):
            raise RuntimeError(
                "Frozen Teacher must be fully frozen."
            )

    model = model.to(
        device
    )

    # ==================================================
    # Dataset
    # ==================================================
    print(
        "Building datasets..."
    )

    train_dataset, val_dataset = (
        create_dataset(
            config["dataset"],
            evaluate=False,
            train_transform=(
                model.backbone.preprocess_train
            ),
            eval_transform=(
                model.backbone.preprocess_val
            ),
        )
    )

    train_batch_size = int(
        train_cfg["batch_size"]
    )
    eval_batch_size = int(
        train_cfg.get(
            "eval_batch_size",
            128,
        )
    )
    text_batch_size = int(
        train_cfg.get(
            "text_batch_size",
            256,
        )
    )
    num_workers = int(
        train_cfg.get(
            "num_workers",
            8,
        )
    )
    epochs = int(
        train_cfg["epochs"]
    )
    eval_every = int(
        train_cfg.get(
            "eval_every",
            1,
        )
    )
    max_steps = train_cfg.get(
        "max_steps"
    )
    log_interval = int(
        train_cfg.get(
            "log_interval",
            50,
        )
    )

    if max_steps is not None:
        max_steps = int(
            max_steps
        )
        if max_steps <= 0:
            raise ValueError(
                "training.max_steps "
                "必须为正整数或 null"
            )

    if log_interval <= 0:
        raise ValueError(
            "training.log_interval "
            "必须为正整数"
        )

    if eval_every <= 0:
        raise ValueError(
            "training.eval_every "
            "必须为正整数"
        )

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

    print(
        "\n" + "=" * 70
    )
    print(
        "DATASET"
    )
    print(
        "=" * 70
    )
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
    # Loss / Optimizer / Scheduler
    # ==================================================
    criterion = CLIPLoss()

    optimizer = build_optimizer(
        model=model,
        lr=config["optimizer"]["lr"],
        weight_decay=(
            config["optimizer"]["weight_decay"]
        ),
        local_distill_only=(
            local_distill_only
        ),
    )

    steps_per_epoch = len(
        train_loader
    )

    if max_steps is not None:
        steps_per_epoch = min(
            steps_per_epoch,
            max_steps,
        )

    total_steps = (
        steps_per_epoch
        * epochs
    )

    warmup_ratio = float(
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

    print(
        "\n" + "=" * 70
    )
    print(
        "TRAINING CONFIG"
    )
    print(
        "=" * 70
    )
    print(
        f"Epochs          : "
        f"{epochs}"
    )
    print(
        f"Learning rate   : "
        f"{config['optimizer']['lr']}"
    )
    print(
        f"Weight decay    : "
        f"{config['optimizer']['weight_decay']}"
    )
    print(
        f"Steps / epoch   : "
        f"{steps_per_epoch}"
    )
    print(
        f"Total steps     : "
        f"{total_steps}"
    )
    print(
        f"Warmup ratio    : "
        f"{warmup_ratio}"
    )
    print(
        f"Warmup steps    : "
        f"{warmup_steps}"
    )
    print(
        f"Eval every      : "
        f"{eval_every} epoch(s)"
    )
    print(
        f"Max train steps : "
        f"{max_steps if max_steps is not None else 'full epoch'}"
    )
    print(
        f"Log interval    : "
        f"{log_interval}"
    )
    print(
        f"Local loss wt   : "
        f"{local_distill_weight}"
    )
    print(
        f"Preserve loss wt: "
        f"{global_preserve_weight}"
    )
    print(
        f"Local-only mode : "
        f"{local_distill_only}"
    )
    print(
        f"Init checkpoint : "
        f"{init_checkpoint if init_checkpoint else 'none'}"
    )
    print(
        f"Freeze backbone : "
        f"{model.freeze_backbone}"
    )

    if local_distill_only:
        print(
            f"Regions / image : "
            f"{config['model'].get('local_num_regions', 2)}"
        )
        print(
            f"Region scale    : "
            f"{config['model'].get('local_min_scale', 0.20)}"
            f" ~ "
            f"{config['model'].get('local_max_scale', 0.60)}"
        )
        print(
            f"Train ViT blocks: "
            f"{local_trainable_blocks}"
        )

        total_blocks = int(
            getattr(
                model,
                "num_visual_blocks",
                0,
            )
        )

        first_block = (
            total_blocks
            - local_trainable_blocks
            + 1
        )

        print(
            f"Local student   : "
            f"Vision Blocks "
            f"{first_block}~{total_blocks}"
        )
        print(
            "Global retrieval: raw Clean CLIP features"
        )
        print(
            "Val mR role      : monitor Clean CLIP semantic drift"
        )
        print(
            "Best criterion  : minimum train B1 objective "
            "(weighted Local + Preserve)"
        )
    else:
        print(
            "Global retrieval: raw Clean CLIP features"
        )
        print(
            "Best criterion  : maximum Val mR"
        )

    # ==================================================
    # Best model selection
    #
    # Clean B1b 维护两个独立 checkpoint：
    #   best_b1.pth  -> 最小训练 B1 objective
    #   best_val.pth -> 最大 Validation mR
    # ==================================================
    best_b1_score = float("inf")
    best_b1_epoch = -1

    best_val_score = float("-inf")
    best_val_epoch = -1

    early_stop_patience = (
        train_cfg.get(
            "early_stop_patience"
        )
    )

    early_stop_min_delta = float(
        train_cfg.get(
            "early_stop_min_delta",
            0.0,
        )
    )

    epochs_without_improvement = 0
    last_val_metrics = None

    # ==================================================
    # Training
    # ==================================================
    for epoch in range(
        1,
        epochs + 1,
    ):
        print(
            "\n" + "=" * 70
        )
        print(
            f"EPOCH {epoch}/{epochs}"
        )
        print(
            "=" * 70
        )

        epoch_start = (
            datetime.now()
        )

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
            entity_loss_weight=0.0,
            local_distill_weight=(
                local_distill_weight
            ),
            global_preserve_weight=(
                global_preserve_weight
            ),
            local_distill_only=(
                local_distill_only
            ),
        )

        train_seconds = (
            datetime.now()
            - epoch_start
        ).total_seconds()

        print(
            "\n" + "-" * 70
        )
        print(
            f"EPOCH {epoch} TRAIN SUMMARY"
        )
        print(
            "-" * 70
        )
        print(
            f"Average loss      : "
            f"{train_stats['loss']:.6f}"
        )
        print(
            f"Average global    : "
            f"{train_stats['loss_global']:.6f}"
        )
        print(
            f"Average local     : "
            f"{train_stats['loss_local']:.6f}"
        )
        print(
            f"Average localCos  : "
            f"{train_stats['local_cosine']:.6f}"
        )
        print(
            f"Average preserve  : "
            f"{train_stats['loss_preserve']:.6f}"
        )
        print(
            f"Average globalCos : "
            f"{train_stats['global_cosine']:.6f}"
        )
        print(
            f"Average I2T       : "
            f"{train_stats['loss_i2t']:.6f}"
        )
        print(
            f"Average T2I       : "
            f"{train_stats['loss_t2i']:.6f}"
        )
        print(
            f"Current LR        : "
            f"{train_stats['lr']:.8f}"
        )
        print(
            f"Logit scale       : "
            f"{train_stats['logit_scale']:.4f}"
        )
        print(
            f"Train time        : "
            f"{train_seconds:.2f} s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                train_stats["loss"]
            ),
            "train_loss_global": float(
                train_stats["loss_global"]
            ),
            "train_loss_local": float(
                train_stats["loss_local"]
            ),
            "train_local_cosine": float(
                train_stats["local_cosine"]
            ),
            "train_loss_preserve": float(
                train_stats["loss_preserve"]
            ),
            "train_global_cosine": float(
                train_stats["global_cosine"]
            ),
            "local_distill_weight": (
                local_distill_weight
            ),
            "global_preserve_weight": (
                global_preserve_weight
            ),
            "local_distill_only": (
                local_distill_only
            ),
            "local_trainable_blocks": (
                local_trainable_blocks
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
            "train_seconds": (
                train_seconds
            ),
            "validated": False,
        }

        # Clean B1b：
        # best_b1 只根据实际 backward objective 判断。
        current_b1_score = float(
            train_stats["loss"]
        )

        is_best_b1 = (
            local_distill_only
            and current_b1_score
            < best_b1_score
            - early_stop_min_delta
        )

        is_best_val = False

        # ==================================================
        # Validation
        # ==================================================
        metrics = None
        validation_seconds = 0.0

        if (
            epoch
            % eval_every
            == 0
        ):
            print(
                "\n" + "=" * 70
            )
            print(
                f"VALIDATION - EPOCH {epoch}"
            )
            print(
                "=" * 70
            )

            val_start = (
                datetime.now()
            )

            metrics, _ = (
                evaluate_retrieval(
                    model=model,
                    data_loader=val_loader,
                    dataset=val_dataset,
                    device=device,
                    text_batch_size=(
                        text_batch_size
                    ),
                )
            )

            validation_seconds = (
                datetime.now()
                - val_start
            ).total_seconds()

            last_val_metrics = (
                metrics
            )
            current_mr = (
                metrics["mR"]
            )

            print(
                "\nImage -> Text"
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

            print(
                "\nText -> Image"
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

            print(
                f"\nI2T mean : "
                f"{metrics['i2t_mean']:.2f}"
            )
            print(
                f"T2I mean : "
                f"{metrics['t2i_mean']:.2f}"
            )
            print(
                f"\nVAL mR   : "
                f"{current_mr:.2f}"
            )
            print(
                f"Val time : "
                f"{validation_seconds:.2f} s"
            )

            epoch_record.update({
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
                "validation_seconds": (
                    validation_seconds
                ),
            })

            current_val_score = float(
                current_mr
            )

            is_best_val = (
                current_val_score
                > best_val_score
                + early_stop_min_delta
            )

        # 分别更新 best_b1 / best_val。
        if is_best_b1:
            best_b1_score = current_b1_score
            best_b1_epoch = epoch

        if (
            metrics is not None
            and is_best_val
        ):
            best_val_score = float(
                metrics["mR"]
            )
            best_val_epoch = epoch
            epochs_without_improvement = 0
        elif metrics is not None:
            epochs_without_improvement += 1

        epoch_record.update({
            "is_best_b1": bool(
                is_best_b1
            ),
            "is_best_val": bool(
                metrics is not None
                and is_best_val
            ),
            "best_b1_epoch": (
                best_b1_epoch
            ),
            "best_b1_objective": (
                float(best_b1_score)
                if best_b1_epoch >= 0
                else None
            ),
            "best_val_epoch": (
                best_val_epoch
            ),
            "best_val_mR": (
                float(best_val_score)
                if best_val_epoch >= 0
                else None
            ),
        })

        append_jsonl(
            metrics_path,
            epoch_record,
        )

        # ==================================================
        # Checkpoint
        # ==================================================
        checkpoint_metrics = {
            "mR": (
                float(metrics["mR"])
                if metrics is not None
                else (
                    float(
                        last_val_metrics["mR"]
                    )
                    if last_val_metrics is not None
                    else float("nan")
                )
            ),
            "train_b1_objective": float(
                train_stats["loss"]
            ),
            "train_local_loss": float(
                train_stats["loss_local"]
            ),
            "train_local_cosine": float(
                train_stats["local_cosine"]
            ),
            "train_preserve_loss": float(
                train_stats["loss_preserve"]
            ),
            "train_global_cosine": float(
                train_stats["global_cosine"]
            ),
        }

        if metrics is not None:
            checkpoint_metrics.update(
                metrics
            )

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
            metrics=checkpoint_metrics,
            config=config,
        )

        print(
            f"\nSaved last checkpoint: "
            f"{last_path}"
        )

        # best_b1.pth：最小 B1 objective。
        if is_best_b1:
            best_b1_path = (
                output_dir
                / "best_b1.pth"
            )

            b1_metrics = dict(
                checkpoint_metrics
            )
            b1_metrics[
                "selection_metric"
            ] = "train_b1_objective"

            save_checkpoint(
                path=best_b1_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=b1_metrics,
                config=config,
            )

            print(
                "\n>>> NEW BEST B1 MODEL"
            )
            print(
                f"Epoch        : "
                f"{best_b1_epoch}"
            )
            print(
                f"B1 objective : "
                f"{best_b1_score:.6f}"
            )
            print(
                f"Local loss   : "
                f"{train_stats['loss_local']:.6f}"
            )
            print(
                f"Preserve loss: "
                f"{train_stats['loss_preserve']:.6f}"
            )
            print(
                f"Global cosine: "
                f"{train_stats['global_cosine']:.6f}"
            )
            if metrics is not None:
                print(
                    f"Val mR       : "
                    f"{metrics['mR']:.2f}"
                )
            print(
                f"Saved : "
                f"{best_b1_path}"
            )

        # best_val.pth：最大 Validation mR。
        if (
            metrics is not None
            and is_best_val
        ):
            best_val_path = (
                output_dir
                / "best_val.pth"
            )

            val_metrics = dict(
                checkpoint_metrics
            )
            val_metrics[
                "selection_metric"
            ] = "val_mR"

            save_checkpoint(
                path=best_val_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )

            print(
                "\n>>> NEW BEST VAL MODEL"
            )
            print(
                f"Epoch        : "
                f"{best_val_epoch}"
            )
            print(
                f"Val mR       : "
                f"{best_val_score:.2f}"
            )
            print(
                f"B1 objective : "
                f"{train_stats['loss']:.6f}"
            )
            print(
                f"Local cosine : "
                f"{train_stats['local_cosine']:.6f}"
            )
            print(
                f"Global cosine: "
                f"{train_stats['global_cosine']:.6f}"
            )
            print(
                f"Saved : "
                f"{best_val_path}"
            )

            # 普通 Clean CLIP 模式继续维护历史 best.pth。
            if not local_distill_only:
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
                    metrics=val_metrics,
                    config=config,
                )

        # ==================================================
        # Early stopping
        # ==================================================
        if (
            early_stop_patience
            is not None
            and metrics is not None
            and epochs_without_improvement
            >= int(
                early_stop_patience
            )
        ):
            print(
                "\n" + "=" * 70
            )
            print(
                "EARLY STOPPING"
            )
            print(
                "=" * 70
            )
            print(
                "No improvement for "
                f"{epochs_without_improvement} rounds."
            )
            print(
                f"Best Val epoch: "
                f"{best_val_epoch}"
            )
            print(
                f"Best Val mR   : "
                f"{best_val_score:.2f}"
            )
            break

        if local_distill_only:
            print(
                f"\nBest B1 epoch      : "
                f"{best_b1_epoch}"
            )
            print(
                f"Best B1 objective  : "
                f"{best_b1_score:.6f}"
            )
            print(
                f"Best Val epoch     : "
                f"{best_val_epoch}"
            )
            if best_val_epoch >= 0:
                print(
                    f"Best Val mR        : "
                    f"{best_val_score:.2f}"
                )
            if metrics is not None:
                print(
                    f"Current Val mR     : "
                    f"{metrics['mR']:.2f} "
                    "(Clean CLIP semantic drift monitor)"
                )
        else:
            print(
                f"\nBest Val epoch     : "
                f"{best_val_epoch}"
            )
            print(
                f"Best Val mR        : "
                f"{best_val_score:.2f}"
            )

        print(
            f"Epoch total time   : "
            f"{(datetime.now() - epoch_start).total_seconds():.2f} s"
        )

        log_file.flush()

    # ==================================================
    # Finish
    # ==================================================
    print(
        "\n" + "=" * 70
    )
    print(
        "TRAINING FINISHED"
    )
    print(
        "=" * 70
    )
    print(
        f"Finish time : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if local_distill_only:
        print(
            f"Best B1 epoch     : "
            f"{best_b1_epoch}"
        )
        print(
            f"Best B1 objective : "
            f"{best_b1_score:.6f}"
        )
        print(
            f"Best Val epoch    : "
            f"{best_val_epoch}"
        )
        if best_val_epoch >= 0:
            print(
                f"Best Val mR       : "
                f"{best_val_score:.2f}"
            )
        print(
            f"Best B1 model     : "
            f"{output_dir / 'best_b1.pth'}"
        )
        print(
            f"Best Val model    : "
            f"{output_dir / 'best_val.pth'}"
        )
    else:
        print(
            f"Best Val epoch : "
            f"{best_val_epoch}"
        )
        print(
            f"Best Val mR    : "
            f"{best_val_score:.2f}"
        )
        print(
            f"Best model     : "
            f"{output_dir / 'best.pth'}"
        )
    print(
        f"Log file    : "
        f"{log_path}"
    )
    print(
        f"Metrics file: "
        f"{metrics_path}"
    )

    log_file.flush()


if __name__ == "__main__":
    main()
