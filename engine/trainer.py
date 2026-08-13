import math

import torch


def build_optimizer(
    model,
    lr=1e-5,
    weight_decay=0.01,
):
    """
    Build AdamW optimizer for CLIP fine-tuning.

    Parameters such as:
        - bias
        - LayerNorm scale
        - logit_scale

    should not receive weight decay.

    Model-declared no_weight_decay parameters are also excluded.
    """

    decay_params = []
    no_decay_params = []

    # OpenCLIP core model
    core_model = model.backbone.model

    # Some OpenCLIP models explicitly declare parameters
    # which should not receive weight decay.
    no_weight_decay_names = set()

    if hasattr(core_model, "no_weight_decay"):
        no_weight_decay_names = set(
            core_model.no_weight_decay()
        )

    for name, param in core_model.named_parameters():

        if not param.requires_grad:
            continue

        if (
            param.ndim <= 1
            or name in no_weight_decay_names
        ):
            no_decay_params.append(param)

        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
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

    return optimizer


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
    Train CLIP for one epoch or a limited number of steps.

    The training dataset is expected to return:

        image, caption, image_id

    image_id is used by the multi-positive CLIP loss
    to identify captions/images belonging to the same
    underlying image.
    """

    model.train()

    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0

    num_steps = 0

    # ==================================================
    # Training loop
    # ==================================================

    for step, (
        images,
        captions,
        image_ids,
    ) in enumerate(data_loader):

        # ------------------------------------------
        # Optional early stop for smoke test
        # ------------------------------------------

        if (
            max_steps is not None
            and step >= max_steps
        ):
            break

        # ------------------------------------------
        # Move tensors to device
        #
        # captions remain list[str].
        # Tokenization is handled inside CLIPBackbone.
        # ------------------------------------------

        images = images.to(
            device,
            non_blocking=True,
        )

        image_ids = image_ids.to(
            device,
            non_blocking=True,
        )

        # ==========================================
        # Forward
        # ==========================================

        outputs = model(
            images,
            captions,
        )

        # ==========================================
        # Multi-positive CLIP loss
        #
        # image_ids are used to determine which
        # image-text pairs in the batch are positives.
        # ==========================================

        losses = criterion(
            outputs["image_feat"],
            outputs["text_feat"],
            outputs["logit_scale"],
            image_ids,
        )

        loss = losses["loss"]

        # ==========================================
        # Safety check
        # ==========================================

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected "
                f"at epoch {epoch}, "
                f"step {step + 1}: "
                f"{loss.item()}"
            )

        # ==========================================
        # Backward
        # ==========================================

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ==========================================
        # Learning-rate scheduler
        # ==========================================

        if scheduler is not None:
            scheduler.step()

        # ==========================================
        # CLIP logit-scale constraint
        # ==========================================

        with torch.no_grad():

            model.backbone.model.logit_scale.clamp_(
                0.0,
                math.log(100.0),
            )

        # ==========================================
        # Statistics
        # ==========================================

        total_loss += loss.item()

        total_i2t += (
            losses["loss_i2t"].item()
        )

        total_t2i += (
            losses["loss_t2i"].item()
        )

        num_steps += 1

        # ==========================================
        # Logging
        # ==========================================

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
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Step {step + 1:04d} | "
                f"Loss {loss.item():.6f} | "
                f"I2T "
                f"{losses['loss_i2t'].item():.6f} | "
                f"T2I "
                f"{losses['loss_t2i'].item():.6f} | "
                f"LR {current_lr:.8f} | "
                f"Scale {current_scale:.4f}"
            )

    # ==================================================
    # Check that at least one batch was trained
    # ==================================================

    if num_steps == 0:
        raise RuntimeError(
            "No training steps were executed."
        )

    # ==================================================
    # Epoch statistics
    # ==================================================

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    current_scale = (
        model.backbone.model
        .logit_scale
        .exp()
        .item()
    )

    stats = {
        "loss": (
            total_loss
            / num_steps
        ),

        "loss_i2t": (
            total_i2t
            / num_steps
        ),

        "loss_t2i": (
            total_t2i
            / num_steps
        ),

        "num_steps": (
            num_steps
        ),

        "lr": (
            current_lr
        ),

        "logit_scale": (
            current_scale
        ),
    }

    return stats