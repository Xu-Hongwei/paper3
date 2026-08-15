import math

import torch


def build_optimizer(model, lr=1e-5, weight_decay=0.01):
    """只优化 requires_grad=True 的参数，兼容冻结 Backbone 与新增 Adapter。"""
    decay_params = []
    no_decay_params = []

    # 保留 OpenCLIP 自带的不衰减参数规则。
    no_weight_decay_names = set()
    core_model = model.backbone.model
    if hasattr(core_model, "no_weight_decay"):
        no_weight_decay_names = {
            f"backbone.model.{name}"
            for name in core_model.no_weight_decay()
        }

    trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        trainable_params += param.numel()
        if param.ndim <= 1 or name in no_weight_decay_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if trainable_params == 0:
        raise RuntimeError("模型没有可训练参数")

    print(f"Trainable parameters: {trainable_params:,}")

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def compute_entity_loss(entity_feat, visual_entity_feat, entity_counts):
    """
    Caption-balanced Entity Grounding loss。

    先对每个 caption 内的 Entity 求平均，再对有有效 Entity 的 caption 求平均，
    避免 Entity 数量多的 caption 在训练中天然占更大权重。
    """
    if entity_feat.shape != visual_entity_feat.shape:
        raise ValueError(
            "Entity feature shape mismatch: "
            f"{tuple(entity_feat.shape)} vs {tuple(visual_entity_feat.shape)}"
        )

    if entity_feat.shape[0] == 0:
        return entity_feat.sum() * 0.0

    # 两侧都已 L2 normalize，点积就是 cosine similarity。
    per_entity_loss = 1.0 - (entity_feat * visual_entity_feat).sum(dim=-1)

    sample_losses = []
    offset = 0
    for count in entity_counts.tolist():
        count = int(count)
        if count <= 0:
            continue

        sample_losses.append(
            per_entity_loss[offset:offset + count].mean()
        )
        offset += count

    if offset != per_entity_loss.shape[0]:
        raise RuntimeError(
            "entity_counts does not match the number of Entity features."
        )

    if not sample_losses:
        return per_entity_loss.sum() * 0.0

    return torch.stack(sample_losses).mean()


def _print_cuda_memory(device):
    """打印首个 smoke step 的 PyTorch CUDA 显存统计。"""
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


def _check_adapter_gradients(model):
    """首步检查：冻结 Backbone 无梯度，Adapter 有有效梯度。"""
    if not getattr(model, "freeze_backbone", False):
        return

    backbone_has_grad = any(
        param.grad is not None
        for param in model.backbone.parameters()
    )
    if backbone_has_grad:
        raise RuntimeError("Frozen Backbone unexpectedly received gradients.")

    adapter_params = [
        param
        for name, param in model.named_parameters()
        if "adapter" in name and param.requires_grad
    ]
    if not adapter_params:
        raise RuntimeError("No trainable Adapter parameters found.")

    adapter_has_grad = any(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum().item() > 0
        for param in adapter_params
    )
    if not adapter_has_grad:
        raise RuntimeError("Adapter did not receive valid gradients.")

    print("Gradient check       : Backbone frozen | Adapter active")


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
    entity_loss_weight=0.1,
):
    """
    总损失：
        L = L_global + lambda_e * L_entity

    L_global 保持当前 CLIP loss 不变；
    L_entity 使用 caption-balanced cosine alignment。
    """
    model.train()

    # Backbone 冻结时保持 eval，避免训练态行为扰动预训练特征。
    if getattr(model, "freeze_backbone", False):
        model.backbone.eval()

    total_loss = 0.0
    total_global = 0.0
    total_entity = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    num_steps = 0

    for step, (
        images,
        captions,
        image_ids,
        entity_spans,
        entity_sample_ids,
        entity_counts,
    ) in enumerate(data_loader):

        if max_steps is not None and step >= max_steps:
            break

        batch_size = images.size(0)
        num_entities = entity_spans.size(0)

        if len(captions) != batch_size:
            raise RuntimeError("Caption batch size mismatch.")
        if image_ids.size(0) != batch_size:
            raise RuntimeError("image_ids batch size mismatch.")
        if entity_counts.size(0) != batch_size:
            raise RuntimeError("entity_counts batch size mismatch.")
        if entity_spans.ndim != 2 or entity_spans.size(1) != 2:
            raise RuntimeError(
                f"entity_spans must be [N, 2], got {tuple(entity_spans.shape)}"
            )
        if entity_sample_ids.shape != (num_entities,):
            raise RuntimeError("entity_sample_ids shape mismatch.")
        if int(entity_counts.sum().item()) != num_entities:
            raise RuntimeError("entity_counts sum does not match entity_spans.")

        images = images.to(device, non_blocking=True)
        image_ids = image_ids.to(device, non_blocking=True)

        if epoch == 1 and step == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        outputs = model(
            images,
            captions,
            entity_spans=entity_spans,
            entity_sample_ids=entity_sample_ids,
            entity_counts=entity_counts,
        )

        # 原有 Global CLIP loss 保持不变。
        losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )
        global_loss = losses["loss"]

        # 新增局部 Entity Grounding loss。
        entity_loss = compute_entity_loss(
            outputs["entity_feat"],
            outputs["visual_entity_feat"],
            entity_counts,
        )

        loss = global_loss + entity_loss_weight * entity_loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, step {step + 1}: "
                f"{loss.item()}"
            )

        if epoch == 1 and step == 0:
            valid_samples = int((entity_counts > 0).sum().item())

            with torch.no_grad():
                if num_entities > 0:
                    entity_cos = (
                        outputs["entity_feat"]
                        * outputs["visual_entity_feat"]
                    ).sum(dim=-1)
                    mean_entity_cos = entity_cos.mean().item()
                else:
                    mean_entity_cos = float("nan")

            print()
            print("=" * 72)
            print("Entity Grounding Loss Smoke Test")
            print("=" * 72)
            print(f"batch size          : {batch_size}")
            print(f"valid Entity spans  : {num_entities}")
            print(f"valid captions      : {valid_samples}/{batch_size}")
            print(f"global loss         : {global_loss.item():.6f}")
            print(f"entity loss         : {entity_loss.item():.6f}")
            print(f"entity loss weight  : {entity_loss_weight:.4f}")
            print(f"weighted entity loss: {(entity_loss_weight * entity_loss).item():.6f}")
            print(f"total loss          : {loss.item():.6f}")
            print(f"mean Entity cosine  : {mean_entity_cos:.6f}")
            print("=" * 72)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if epoch == 1 and step == 0:
            _check_adapter_gradients(model)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # 仅在 logit_scale 可训练时执行 CLIP 原有约束。
        if model.backbone.model.logit_scale.requires_grad:
            with torch.no_grad():
                model.backbone.model.logit_scale.clamp_(0.0, math.log(100.0))

        total_loss += loss.item()
        total_global += global_loss.item()
        total_entity += entity_loss.item()
        total_i2t += losses["loss_i2t"].item()
        total_t2i += losses["loss_t2i"].item()
        num_steps += 1

        if step == 0 or (step + 1) % log_interval == 0:
            current_scale = model.backbone.model.logit_scale.exp().item()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | Step {step + 1:04d} | "
                f"Loss {loss.item():.6f} | "
                f"Global {global_loss.item():.6f} | "
                f"Entity {entity_loss.item():.6f} | "
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
        "loss_global": total_global / num_steps,
        "loss_entity": total_entity / num_steps,
        "loss_i2t": total_i2t / num_steps,
        "loss_t2i": total_t2i / num_steps,
        "num_steps": num_steps,
        "lr": optimizer.param_groups[0]["lr"],
        "logit_scale": model.backbone.model.logit_scale.exp().item(),
    }
