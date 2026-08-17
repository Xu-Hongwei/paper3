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
):
    """
    Clean CLIP 单轮训练。

    当前目标：
        L = Multi-positive CLIP loss

    Dataset 仍保留 Entity spans，供后续细粒度模块使用；
    当前 Global CLIP 阶段不使用实体监督。
    """
    if log_interval <= 0:
        raise ValueError("log_interval 必须 > 0。")

    model.train()

    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    num_steps = 0

    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break

        (
            images,
            captions,
            image_ids,
            entity_spans,
            entity_sample_ids,
            entity_counts,
        ) = batch

        # Entity 信息保留在数据链中，当前 Global CLIP 暂不参与 loss。
        _ = (entity_spans, entity_sample_ids, entity_counts)

        batch_size = images.size(0)
        if len(captions) != batch_size:
            raise RuntimeError("Caption batch size mismatch.")
        if image_ids.size(0) != batch_size:
            raise RuntimeError("image_ids batch size mismatch.")

        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)

        if epoch == 1 and step == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        outputs = model(images, captions)

        losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )
        loss = losses["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, step {step + 1}: "
                f"{loss.item()}"
            )

        if epoch == 1 and step == 0:
            print()
            print("=" * 72)
            print("Clean CLIP Global Training Smoke Test")
            print("=" * 72)
            print(f"batch size : {batch_size}")
            print(f"loss       : {loss.item():.6f}")
            print(f"I2T loss   : {losses['loss_i2t'].item():.6f}")
            print(f"T2I loss   : {losses['loss_t2i'].item():.6f}")
            print(f"entities   : {int(entity_counts.sum().item())}")
            print("=" * 72)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if model.backbone.model.logit_scale.requires_grad:
            with torch.no_grad():
                model.backbone.model.logit_scale.clamp_(0.0, math.log(100.0))

        total_loss += loss.item()
        total_i2t += losses["loss_i2t"].item()
        total_t2i += losses["loss_t2i"].item()
        num_steps += 1

        if step == 0 or (step + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            current_scale = model.backbone.model.logit_scale.exp().item()

            print(
                f"Epoch {epoch:03d} | Step {step + 1:04d} | "
                f"Loss {loss.item():.6f} | "
                f"I2T {losses['loss_i2t'].item():.6f} | "
                f"T2I {losses['loss_t2i'].item():.6f} | "
                f"LR {current_lr:.8f} | Scale {current_scale:.4f}"
            )

            if epoch == 1 and step == 0:
                _print_cuda_memory(device)

    if num_steps == 0:
        raise RuntimeError("No training steps were executed.")

    return {
        "loss": total_loss / num_steps,
        "loss_i2t": total_i2t / num_steps,
        "loss_t2i": total_t2i / num_steps,
        "num_steps": num_steps,
        "lr": optimizer.param_groups[0]["lr"],
        "logit_scale": model.backbone.model.logit_scale.exp().item(),
    }
