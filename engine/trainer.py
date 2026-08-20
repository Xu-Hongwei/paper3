import math

import torch


def build_optimizer(model, lr=1e-5, weight_decay=0.01):
    """构建 AdamW，并对 1D 参数和 OpenCLIP 指定参数关闭 weight decay。"""
    decay_params, no_decay_params = [], []

    no_weight_decay = set()
    core_model = model.backbone.model
    if hasattr(core_model, "no_weight_decay"):
        no_weight_decay = {
            f"backbone.model.{name}"
            for name in core_model.no_weight_decay()
        }

    trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        trainable_params += param.numel()
        if param.ndim <= 1 or name in no_weight_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if trainable_params == 0:
        raise RuntimeError("模型没有可训练参数。")

    print(f"Trainable parameters: {trainable_params:,}")

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _print_cuda_memory(device):
    """打印首个 step 的 CUDA 显存占用。"""
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize(device)
    gb = 1024 ** 3

    allocated = torch.cuda.memory_allocated(device) / gb
    reserved = torch.cuda.memory_reserved(device) / gb
    peak = torch.cuda.max_memory_allocated(device) / gb

    print(
        f"CUDA memory | allocated {allocated:.3f} GB | "
        f"reserved {reserved:.3f} GB | peak {peak:.3f} GB"
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
    category_criterion=None,
    category_t2i_weight=0.0,
    category_i2t_weight=0.0,
):
    """
    单轮训练。

    总目标：
        L = L_clip
          + lambda_t2i * L_cat_t2i
          + lambda_i2t * L_cat_i2t

    category_criterion=None 或两个权重都为 0 时，退化为原始 Clean CLIP。
    """
    if log_interval <= 0:
        raise ValueError("log_interval 必须 > 0。")
    if category_t2i_weight < 0 or category_i2t_weight < 0:
        raise ValueError("Category margin loss 权重必须 >= 0。")

    category_enabled = (
        category_criterion is not None
        and (
            category_t2i_weight > 0
            or category_i2t_weight > 0
        )
    )

    model.train()

    total_loss = 0.0
    total_clip_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_cat_t2i = 0.0
    total_cat_i2t = 0.0

    total_t2i_valid = 0
    total_i2t_valid = 0
    total_t2i_active = 0
    total_i2t_active = 0

    total_t2i_pos_sum = 0.0
    total_i2t_pos_sum = 0.0
    total_t2i_neg_sum = 0.0
    total_i2t_neg_sum = 0.0

    num_steps = 0

    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break

        (
            images,
            captions,
            image_ids,
            category_ids,
            entity_spans,
            entity_sample_ids,
            entity_counts,
        ) = batch

        # Entity 信息保留在数据链中，当前阶段暂不参与 loss。
        _ = (entity_spans, entity_sample_ids, entity_counts)

        batch_size = images.size(0)
        if len(captions) != batch_size:
            raise RuntimeError("Caption batch size mismatch.")
        if image_ids.size(0) != batch_size:
            raise RuntimeError("image_ids batch size mismatch.")
        if category_ids.size(0) != batch_size:
            raise RuntimeError("category_ids batch size mismatch.")

        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)
        category_ids = category_ids.to(device, non_blocking=True)

        if epoch == 1 and step == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        outputs = model(images, captions)

        clip_losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )
        clip_loss = clip_losses["loss"]

        if category_enabled:
            cat_losses = category_criterion(
                outputs["image_feat"],
                outputs["text_feat"],
                category_ids,
            )
            cat_t2i_loss = cat_losses["loss_t2i"]
            cat_i2t_loss = cat_losses["loss_i2t"]
        else:
            zero = clip_loss.new_zeros(())
            cat_t2i_loss = zero
            cat_i2t_loss = zero
            cat_losses = None

        loss = (
            clip_loss
            + category_t2i_weight * cat_t2i_loss
            + category_i2t_weight * cat_i2t_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, step {step + 1}: "
                f"{loss.item()}"
            )

        if epoch == 1 and step == 0:
            print()
            print("=" * 80)
            print("CLIP + Cross-Category Margin Training Smoke Test")
            print("=" * 80)
            print(f"batch size       : {batch_size}")
            print(f"known categories : {int((category_ids >= 0).sum().item())}/{batch_size}")
            print(f"total loss       : {loss.item():.6f}")
            print(f"CLIP loss        : {clip_loss.item():.6f}")
            print(f"CLIP I2T         : {clip_losses['loss_i2t'].item():.6f}")
            print(f"CLIP T2I         : {clip_losses['loss_t2i'].item():.6f}")
            print(
                f"Cat T2I          : {cat_t2i_loss.item():.6f} "
                f"(weight={category_t2i_weight:.3f})"
            )
            print(
                f"Cat I2T          : {cat_i2t_loss.item():.6f} "
                f"(weight={category_i2t_weight:.3f})"
            )

            if cat_losses is not None:
                t2i_valid = cat_losses["t2i_valid_count"]
                i2t_valid = cat_losses["i2t_valid_count"]
                t2i_active = cat_losses["t2i_active_count"]
                i2t_active = cat_losses["i2t_active_count"]

                print(
                    f"T2I valid/active  : {t2i_valid}/{t2i_active} "
                    f"({t2i_active / max(t2i_valid, 1):.2%})"
                )
                print(
                    f"I2T valid/active  : {i2t_valid}/{i2t_active} "
                    f"({i2t_active / max(i2t_valid, 1):.2%})"
                )
                print(
                    f"T2I pos/neg sim   : "
                    f"{cat_losses['t2i_mean_pos_sim'].item():.4f} / "
                    f"{cat_losses['t2i_mean_hard_neg_sim'].item():.4f}"
                )
                print(
                    f"I2T pos/neg sim   : "
                    f"{cat_losses['i2t_mean_pos_sim'].item():.4f} / "
                    f"{cat_losses['i2t_mean_hard_neg_sim'].item():.4f}"
                )

            print(f"entities         : {int(entity_counts.sum().item())}")
            print("=" * 80)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if model.backbone.model.logit_scale.requires_grad:
            with torch.no_grad():
                model.backbone.model.logit_scale.clamp_(
                    0.0,
                    math.log(100.0),
                )

        total_loss += loss.item()
        total_clip_loss += clip_loss.item()
        total_i2t += clip_losses["loss_i2t"].item()
        total_t2i += clip_losses["loss_t2i"].item()
        total_cat_t2i += cat_t2i_loss.item()
        total_cat_i2t += cat_i2t_loss.item()
        num_steps += 1

        if cat_losses is not None:
            t2i_valid = cat_losses["t2i_valid_count"]
            i2t_valid = cat_losses["i2t_valid_count"]
            t2i_active = cat_losses["t2i_active_count"]
            i2t_active = cat_losses["i2t_active_count"]

            total_t2i_valid += t2i_valid
            total_i2t_valid += i2t_valid
            total_t2i_active += t2i_active
            total_i2t_active += i2t_active

            total_t2i_pos_sum += (
                cat_losses["t2i_mean_pos_sim"].item()
                * t2i_valid
            )
            total_i2t_pos_sum += (
                cat_losses["i2t_mean_pos_sim"].item()
                * i2t_valid
            )
            total_t2i_neg_sum += (
                cat_losses["t2i_mean_hard_neg_sim"].item()
                * t2i_valid
            )
            total_i2t_neg_sum += (
                cat_losses["i2t_mean_hard_neg_sim"].item()
                * i2t_valid
            )

        if step == 0 or (step + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            current_scale = (
                model.backbone.model.logit_scale.exp().item()
            )

            message = (
                f"Epoch {epoch:03d} | Step {step + 1:04d} | "
                f"Loss {loss.item():.6f} | "
                f"CLIP {clip_loss.item():.6f} | "
                f"I2T {clip_losses['loss_i2t'].item():.6f} | "
                f"T2I {clip_losses['loss_t2i'].item():.6f}"
            )

            if cat_losses is not None:
                t2i_valid = cat_losses["t2i_valid_count"]
                i2t_valid = cat_losses["i2t_valid_count"]
                t2i_active = cat_losses["t2i_active_count"]
                i2t_active = cat_losses["i2t_active_count"]

                message += (
                    f" | CatT2I {cat_t2i_loss.item():.6f} "
                    f"({t2i_active}/{t2i_valid})"
                    f" | CatI2T {cat_i2t_loss.item():.6f} "
                    f"({i2t_active}/{i2t_valid})"
                )

            message += (
                f" | LR {current_lr:.8f} | "
                f"Scale {current_scale:.4f}"
            )
            print(message)

            if epoch == 1 and step == 0:
                _print_cuda_memory(device)

    if num_steps == 0:
        raise RuntimeError("No training steps were executed.")

    stats = {
        "loss": total_loss / num_steps,
        "clip_loss": total_clip_loss / num_steps,
        "loss_i2t": total_i2t / num_steps,
        "loss_t2i": total_t2i / num_steps,
        "cat_loss_t2i": total_cat_t2i / num_steps,
        "cat_loss_i2t": total_cat_i2t / num_steps,
        "cat_t2i_valid": total_t2i_valid,
        "cat_i2t_valid": total_i2t_valid,
        "cat_t2i_active": total_t2i_active,
        "cat_i2t_active": total_i2t_active,
        "cat_t2i_active_ratio": (
            total_t2i_active / max(total_t2i_valid, 1)
        ),
        "cat_i2t_active_ratio": (
            total_i2t_active / max(total_i2t_valid, 1)
        ),
        "cat_t2i_pos_sim": (
            total_t2i_pos_sum / max(total_t2i_valid, 1)
        ),
        "cat_i2t_pos_sim": (
            total_i2t_pos_sum / max(total_i2t_valid, 1)
        ),
        "cat_t2i_hard_neg_sim": (
            total_t2i_neg_sum / max(total_t2i_valid, 1)
        ),
        "cat_i2t_hard_neg_sim": (
            total_i2t_neg_sum / max(total_i2t_valid, 1)
        ),
        "num_steps": num_steps,
        "lr": optimizer.param_groups[0]["lr"],
        "logit_scale": (
            model.backbone.model.logit_scale.exp().item()
        ),
    }

    return stats
