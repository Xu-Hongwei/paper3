import math

import torch


def _get_local_student_names(model):
    """获取 Clean B1b 唯一允许训练的 Student 参数名。"""
    if not hasattr(model, "get_local_student_parameter_names"):
        raise RuntimeError(
            "Model does not provide get_local_student_parameter_names()."
        )

    names = set(
        model.get_local_student_parameter_names()
    )

    if not names:
        raise RuntimeError(
            "没有找到可训练的 Local Student 参数。"
        )

    return names


def build_optimizer(
    model,
    lr=1e-5,
    weight_decay=0.01,
    local_distill_only=False,
):
    """
    构建 AdamW。

    local_distill_only=True:
        仅训练 Clean B1b Student Vision 最后 N 个 Block。

    local_distill_only=False:
        使用模型当前 requires_grad=True 的参数，
        可用于普通 CLIP 训练。
    """
    local_student_names = None

    if local_distill_only:
        local_student_names = (
            _get_local_student_names(model)
        )

        for name, param in model.named_parameters():
            param.requires_grad = (
                name in local_student_names
            )

    decay_params = []
    no_decay_params = []

    no_weight_decay_names = set()
    core_model = model.backbone.model

    if hasattr(core_model, "no_weight_decay"):
        no_weight_decay_names = {
            f"backbone.model.{name}"
            for name in core_model.no_weight_decay()
        }

    trainable_names = []
    trainable_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        trainable_names.append(name)
        trainable_params += param.numel()

        if (
            param.ndim <= 1
            or name in no_weight_decay_names
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if trainable_params == 0:
        raise RuntimeError(
            "模型没有可训练参数。"
        )

    if local_distill_only:
        invalid_names = [
            name
            for name in trainable_names
            if name not in local_student_names
        ]
        missing_names = [
            name
            for name in local_student_names
            if name not in trainable_names
        ]

        if invalid_names or missing_names:
            raise RuntimeError(
                "Clean B1b 参数配置异常，"
                f"unexpected={invalid_names}, "
                f"missing={missing_names}"
            )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    if local_distill_only:
        trainable_blocks = int(
            getattr(
                model,
                "local_trainable_blocks",
                0,
            )
        )
        total_blocks = int(
            getattr(
                model,
                "num_visual_blocks",
                0,
            )
        )

        if trainable_blocks <= 0:
            raise RuntimeError(
                "Clean B1b 要求 local_trainable_blocks > 0。"
            )

        first_block = (
            total_blocks
            - trainable_blocks
            + 1
        )

        print(
            "Trainable module    : "
            f"Vision Blocks "
            f"{first_block}~{total_blocks}"
        )

    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
    )


def compute_local_distill_loss(
    student_feat,
    teacher_feat,
):
    """
    Local Self-Distillation：

        L_local = mean(1 - cosine(r_R^S, t_R^T))
    """
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            "Local feature shape mismatch: "
            f"{tuple(student_feat.shape)} vs "
            f"{tuple(teacher_feat.shape)}"
        )

    if student_feat.ndim != 2:
        raise ValueError(
            "Local features must be [R,D], got "
            f"{tuple(student_feat.shape)}"
        )

    if student_feat.shape[0] == 0:
        raise ValueError(
            "Local Self-Distillation 没有有效 Region。"
        )

    cosine = (
        student_feat
        * teacher_feat.detach()
    ).sum(dim=-1)

    loss = (
        1.0 - cosine
    ).mean()

    return loss, cosine


def compute_global_preserve_loss(
    student_feat,
    teacher_feat,
):
    """
    Global Semantic Preservation：

        L_preserve
        = mean(1 - cosine(z_I^S, z_I^T))
    """
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            "Global preserve feature shape mismatch: "
            f"{tuple(student_feat.shape)} vs "
            f"{tuple(teacher_feat.shape)}"
        )

    if student_feat.ndim != 2:
        raise ValueError(
            "Global preserve features must be [B,D], got "
            f"{tuple(student_feat.shape)}"
        )

    if student_feat.shape[0] == 0:
        raise ValueError(
            "Global Preservation 没有有效样本。"
        )

    cosine = (
        student_feat
        * teacher_feat.detach()
    ).sum(dim=-1)

    loss = (
        1.0 - cosine
    ).mean()

    return loss, cosine


def _print_cuda_memory(device):
    """打印首个 step 的 CUDA 显存统计。"""
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize(device)

    gb = 1024 ** 3
    allocated = (
        torch.cuda.memory_allocated(device)
        / gb
    )
    reserved = (
        torch.cuda.memory_reserved(device)
        / gb
    )
    peak = (
        torch.cuda.max_memory_allocated(device)
        / gb
    )

    print(
        f"CUDA memory | allocated "
        f"{allocated:.3f} GB | "
        f"reserved {reserved:.3f} GB | "
        f"peak {peak:.3f} GB"
    )


def _check_local_gradients(model):
    """
    Clean B1b 首步梯度检查。

    只允许 Student Vision 最后 N 个 Block 有梯度；
    Frozen Teacher 和其余 CLIP 参数不得有梯度。
    """
    allowed_names = (
        _get_local_student_names(model)
    )

    unexpected_grad_names = []
    active_grad_names = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if name not in allowed_names:
            unexpected_grad_names.append(name)
            continue

        if (
            torch.isfinite(param.grad).all()
            and param.grad.abs().sum().item() > 0
        ):
            active_grad_names.append(name)

    if unexpected_grad_names:
        raise RuntimeError(
            "Clean B1b 出现非法梯度参数: "
            f"{unexpected_grad_names[:20]}"
        )

    if not active_grad_names:
        raise RuntimeError(
            "Student Vision 没有收到有效梯度。"
        )

    teacher = getattr(
        model,
        "local_teacher_visual",
        None,
    )

    if teacher is None:
        raise RuntimeError(
            "Clean B1b 缺少 Frozen Teacher。"
        )

    if any(
        param.grad is not None
        for param in teacher.parameters()
    ):
        raise RuntimeError(
            "Frozen Teacher unexpectedly received gradients."
        )

    trainable_blocks = int(
        getattr(
            model,
            "local_trainable_blocks",
            0,
        )
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
        - trainable_blocks
        + 1
    )

    print(
        "Gradient check       : "
        f"Teacher frozen | "
        f"Vision Blocks "
        f"{first_block}~{total_blocks} active | "
        "Other CLIP parameters frozen"
    )


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    epoch=0,
    max_steps=None,
    log_interval=20,
    scheduler=None,
    entity_loss_weight=0.0,
    local_distill_weight=0.0,
    global_preserve_weight=0.0,
    local_distill_only=False,
):
    """
    支持两种 Clean CLIP 模式：

    1. 普通 Global CLIP：
        L = L_global

    2. Clean B1b：
        local_distill_only=True

        L_backward
        = lambda_l * L_local
        + lambda_p * L_preserve

    当前不再支持旧 Naive Entity Grounding。
    Entity / Attribute / Relation 将在后续 Structured Grounding
    阶段重新设计。
    """
    if entity_loss_weight != 0:
        raise ValueError(
            "Clean CLIP 主线已移除旧 Entity Grounding，"
            "training.entity_loss_weight 必须为 0。"
        )

    if (
        local_distill_weight < 0
        or global_preserve_weight < 0
    ):
        raise ValueError(
            "Loss weight 不能为负数。"
        )

    use_local = (
        local_distill_weight > 0
    )

    if (
        local_distill_only
        and not use_local
    ):
        raise ValueError(
            "local_distill_only=True 时 "
            "local_distill_weight 必须 > 0。"
        )

    if (
        global_preserve_weight > 0
        and not use_local
    ):
        raise ValueError(
            "global_preserve_weight > 0 时 "
            "必须启用 Local Self-Distillation。"
        )

    model.train()

    total_loss = 0.0
    total_global = 0.0
    total_local = 0.0
    total_local_cos = 0.0
    total_preserve = 0.0
    total_global_cos = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    num_steps = 0

    for step, batch in enumerate(
        data_loader
    ):
        if (
            max_steps is not None
            and step >= max_steps
        ):
            break

        (
            images,
            captions,
            image_ids,
            entity_spans,
            entity_sample_ids,
            entity_counts,
        ) = batch

        batch_size = images.size(0)

        if len(captions) != batch_size:
            raise RuntimeError(
                "Caption batch size mismatch."
            )

        if image_ids.size(0) != batch_size:
            raise RuntimeError(
                "image_ids batch size mismatch."
            )

        images = images.to(
            device,
            non_blocking=True,
        )
        image_ids = image_ids.to(
            device,
            non_blocking=True,
        )

        if (
            epoch == 1
            and step == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.reset_peak_memory_stats(
                device
            )

        # --------------------------------------------------
        # Clean B1b 完全忽略 Dataset 里的 Entity spans。
        # --------------------------------------------------
        if use_local:
            outputs = model(
                images,
                captions,
                local_distill=True,
            )
        else:
            outputs = model(
                images,
                captions,
            )

        # Global retrieval loss：
        # B1b 中仅监控，不参与 backward。
        losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )

        global_loss = losses["loss"]

        local_loss = (
            global_loss.new_zeros(())
        )
        preserve_loss = (
            global_loss.new_zeros(())
        )
        local_cosine = None
        global_cosine = None

        if use_local:
            (
                local_loss,
                local_cosine,
            ) = compute_local_distill_loss(
                outputs["local_student_feat"],
                outputs["local_teacher_feat"],
            )

            (
                preserve_loss,
                global_cosine,
            ) = compute_global_preserve_loss(
                outputs["local_student_global_feat"],
                outputs["local_teacher_global_feat"],
            )

        if local_distill_only:
            loss = (
                local_distill_weight
                * local_loss
                + global_preserve_weight
                * preserve_loss
            )
        else:
            loss = (
                global_loss
                + local_distill_weight
                * local_loss
                + global_preserve_weight
                * preserve_loss
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"step {step + 1}: "
                f"{loss.item()}"
            )

        # --------------------------------------------------
        # 首步 Smoke Report
        # --------------------------------------------------
        if (
            epoch == 1
            and step == 0
        ):
            print()
            print("=" * 72)

            if use_local:
                print(
                    "Clean CLIP B1b Self-Distillation Smoke Test"
                )
                print("=" * 72)
                print(
                    f"batch size          : "
                    f"{batch_size}"
                )
                print(
                    f"regions             : "
                    f"{outputs['local_student_feat'].shape[0]}"
                )
                print(
                    f"global loss(monitor): "
                    f"{global_loss.item():.6f}"
                )
                print(
                    f"local loss          : "
                    f"{local_loss.item():.6f}"
                )
                print(
                    f"local loss weight   : "
                    f"{local_distill_weight:.4f}"
                )
                print(
                    f"weighted local loss : "
                    f"{(local_distill_weight * local_loss).item():.6f}"
                )
                print(
                    f"mean local cosine   : "
                    f"{local_cosine.mean().item():.6f}"
                )
                print(
                    f"preserve loss       : "
                    f"{preserve_loss.item():.6f}"
                )
                print(
                    f"preserve loss weight: "
                    f"{global_preserve_weight:.4f}"
                )
                print(
                    f"weighted preserve   : "
                    f"{(global_preserve_weight * preserve_loss).item():.6f}"
                )
                print(
                    f"mean global cosine  : "
                    f"{global_cosine.mean().item():.6f}"
                )
                print(
                    f"min global cosine   : "
                    f"{global_cosine.min().item():.6f}"
                )
                print(
                    f"max global cosine   : "
                    f"{global_cosine.max().item():.6f}"
                )
                print(
                    f"min local cosine    : "
                    f"{local_cosine.min().item():.6f}"
                )
                print(
                    f"max local cosine    : "
                    f"{local_cosine.max().item():.6f}"
                )
                print(
                    f"backward loss       : "
                    f"{loss.item():.6f}"
                )
                print(
                    f"trainable ViT blocks: "
                    f"{getattr(model, 'local_trainable_blocks', 0)}"
                )
            else:
                print(
                    "Clean CLIP Global Training Smoke Test"
                )
                print("=" * 72)
                print(
                    f"batch size          : "
                    f"{batch_size}"
                )
                print(
                    f"global loss         : "
                    f"{global_loss.item():.6f}"
                )
                print(
                    f"total loss          : "
                    f"{loss.item():.6f}"
                )

            print("=" * 72)

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        if (
            epoch == 1
            and step == 0
            and local_distill_only
        ):
            _check_local_gradients(
                model
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (
            model.backbone.model
            .logit_scale
            .requires_grad
        ):
            with torch.no_grad():
                model.backbone.model.logit_scale.clamp_(
                    0.0,
                    math.log(100.0),
                )

        total_loss += loss.item()
        total_global += global_loss.item()
        total_local += local_loss.item()
        total_preserve += (
            preserve_loss.item()
        )

        if local_cosine is not None:
            total_local_cos += (
                local_cosine.mean().item()
            )

        if global_cosine is not None:
            total_global_cos += (
                global_cosine.mean().item()
            )

        total_i2t += (
            losses["loss_i2t"].item()
        )
        total_t2i += (
            losses["loss_t2i"].item()
        )
        num_steps += 1

        if (
            step == 0
            or (step + 1) % log_interval == 0
        ):
            current_scale = (
                model.backbone.model
                .logit_scale
                .exp()
                .item()
            )
            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            message = (
                f"Epoch {epoch:03d} | "
                f"Step {step + 1:04d} | "
                f"Loss {loss.item():.6f} | "
                f"Global {global_loss.item():.6f}"
            )

            if use_local:
                message += (
                    f" | Local "
                    f"{local_loss.item():.6f}"
                    f" | LocalCos "
                    f"{local_cosine.mean().item():.6f}"
                    f" | Preserve "
                    f"{preserve_loss.item():.6f}"
                    f" | GlobalCos "
                    f"{global_cosine.mean().item():.6f}"
                )

            message += (
                f" | I2T "
                f"{losses['loss_i2t'].item():.6f}"
                f" | T2I "
                f"{losses['loss_t2i'].item():.6f}"
                f" | LR "
                f"{current_lr:.8f}"
                f" | Scale "
                f"{current_scale:.4f}"
            )

            print(message)

            if (
                epoch == 1
                and step == 0
            ):
                _print_cuda_memory(
                    device
                )

    if num_steps == 0:
        raise RuntimeError(
            "No training steps were executed."
        )

    return {
        "loss": (
            total_loss
            / num_steps
        ),
        "loss_global": (
            total_global
            / num_steps
        ),
        # 保留该字段仅兼容当前 train.py，
        # Clean CLIP 主线中始终为 0。
        "loss_entity": 0.0,
        "loss_local": (
            total_local
            / num_steps
        ),
        "local_cosine": (
            total_local_cos
            / num_steps
            if use_local
            else 0.0
        ),
        "loss_preserve": (
            total_preserve
            / num_steps
        ),
        "global_cosine": (
            total_global_cos
            / num_steps
            if use_local
            else 0.0
        ),
        "loss_i2t": (
            total_i2t
            / num_steps
        ),
        "loss_t2i": (
            total_t2i
            / num_steps
        ),
        "num_steps": num_steps,
        "lr": (
            optimizer.param_groups[0]["lr"]
        ),
        "logit_scale": (
            model.backbone.model
            .logit_scale
            .exp()
            .item()
        ),
    }
