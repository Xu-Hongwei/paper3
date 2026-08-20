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


def _validate_category_support_cache(cache):
    """检查 Reliable T2I 所需 frozen support cache。"""
    if not isinstance(cache, dict):
        raise TypeError("category_support_cache 必须是 dict。")

    required = {"caption_support", "image_support"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(
            f"Category support cache missing keys: {missing}"
        )

    caption_support = cache["caption_support"]
    image_support = cache["image_support"]

    if not torch.is_tensor(caption_support) or caption_support.ndim != 2:
        raise ValueError(
            "caption_support must be a tensor with shape [N_pair, C]."
        )

    if not torch.is_tensor(image_support) or image_support.ndim != 2:
        raise ValueError(
            "image_support must be a tensor with shape [N_image, C]."
        )

    if caption_support.shape[1] != image_support.shape[1]:
        raise ValueError(
            "caption_support / image_support category dim mismatch: "
            f"{caption_support.shape[1]} vs {image_support.shape[1]}."
        )

    if caption_support.device.type != "cpu":
        raise ValueError(
            "caption_support 建议保存在 CPU；trainer 每 batch 只搬运 [B, C]。"
        )

    if image_support.device.type != "cpu":
        raise ValueError(
            "image_support 建议保存在 CPU；trainer 每 batch 只搬运 [B, C]。"
        )


def _get_batch_category_support(
    cache,
    sample_indices,
    image_ids,
    category_ids,
):
    """
    从 frozen cache 取当前 batch 的：
        caption_support: [B, C]
        image_support  : [B, C]

    同时利用 cache 中可选的索引字段检查 pair/image/category 是否严格对齐。
    """
    sample_indices = sample_indices.long().cpu()
    image_ids = image_ids.long().cpu()
    category_ids = category_ids.long().cpu()

    caption_support = cache["caption_support"].index_select(
        0,
        sample_indices,
    )
    image_support = cache["image_support"].index_select(
        0,
        image_ids,
    )

    if "sample_image_ids" in cache:
        cached = cache["sample_image_ids"].long().cpu().index_select(
            0,
            sample_indices,
        )
        if not torch.equal(cached, image_ids):
            raise RuntimeError(
                "Support cache sample_image_ids 与当前 batch image_ids 不一致。"
            )

    if "sample_category_ids" in cache:
        cached = cache["sample_category_ids"].long().cpu().index_select(
            0,
            sample_indices,
        )
        if not torch.equal(cached, category_ids):
            raise RuntimeError(
                "Support cache sample_category_ids 与当前 batch category_ids 不一致。"
            )

    if "image_category_ids" in cache:
        cached = cache["image_category_ids"].long().cpu().index_select(
            0,
            image_ids,
        )
        if not torch.equal(cached, category_ids):
            raise RuntimeError(
                "Support cache image_category_ids 与当前 batch category_ids 不一致。"
            )

    return caption_support, image_support


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
    category_support_cache=None,
):
    """
    单轮训练。

    总目标：
        L = L_clip
          + lambda_t2i * L_cat_t2i
          + lambda_i2t * L_cat_i2t

    category_criterion=None 或两个权重都为 0 时，退化为原始 Clean CLIP。

    Reliable T2I 开启时，category_support_cache 由 train.py 一次加载并传入；
    trainer 只按 sample_indices / image_ids 取当前 batch 的 support。

    reliability_mode:
        none            -> Fixed Mining
        post_gate       -> 先选 fixed hard negative，再做 A-zone gate
        reliable_mining -> 先过滤可靠候选，再在可靠集合中选 hardest
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
    reliability_mode = (
        str(
            getattr(
                category_criterion,
                "reliability_mode",
                "post_gate"
                if getattr(
                    category_criterion,
                    "reliable_t2i",
                    False,
                )
                else "none",
            )
        ).lower()
        if category_criterion is not None
        else "none"
    )
    reliable_t2i_enabled = (
        category_enabled
        and category_t2i_weight > 0
        and reliability_mode != "none"
    )
    reliable_mining_enabled = (
        reliable_t2i_enabled
        and reliability_mode == "reliable_mining"
    )

    if reliable_t2i_enabled:
        if category_support_cache is None:
            raise ValueError(
                "Reliable T2I 已启用，但 category_support_cache=None。"
            )
        _validate_category_support_cache(
            category_support_cache
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
    total_t2i_reliable = 0
    total_t2i_active_before_gate = 0
    total_t2i_active = 0
    total_i2t_active = 0

    total_t2i_region_a = 0
    total_t2i_region_b = 0
    total_t2i_region_c = 0
    total_t2i_region_d = 0

    total_t2i_g_sup_sum = 0.0
    total_t2i_g_cat_sum = 0.0

    # Reliable Mining 专用诊断。
    total_t2i_reliable_candidate = 0
    total_t2i_no_reliable = 0
    total_t2i_replacement = 0
    total_t2i_fixed_neg_sum = 0.0

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
            sample_indices,
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
        if sample_indices.size(0) != batch_size:
            raise RuntimeError("sample_indices batch size mismatch.")

        batch_caption_support = None
        batch_image_support = None

        if reliable_t2i_enabled:
            (
                batch_caption_support,
                batch_image_support,
            ) = _get_batch_category_support(
                category_support_cache,
                sample_indices,
                image_ids,
                category_ids,
            )

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
                caption_support=batch_caption_support,
                image_support=batch_image_support,
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

                if reliable_t2i_enabled:
                    t2i_reliable = cat_losses[
                        "t2i_reliable_count"
                    ]
                    active_before = cat_losses[
                        "t2i_active_before_gate_count"
                    ]

                    if reliable_mining_enabled:
                        print(
                            f"T2I valid/minable : "
                            f"{t2i_valid}/{t2i_reliable} "
                            f"({t2i_reliable / max(t2i_valid, 1):.2%})"
                        )
                        print(
                            f"T2I reliable negs : "
                            f"avg={cat_losses['t2i_avg_reliable_negatives']:.2f} "
                            f"| none={cat_losses['t2i_no_reliable_count']}"
                        )
                        print(
                            f"T2I replacement   : "
                            f"{cat_losses['t2i_replacement_count']}/{t2i_valid} "
                            f"({cat_losses['t2i_replacement_ratio']:.2%})"
                        )
                        print(
                            f"T2I fixed/rel neg : "
                            f"{cat_losses['t2i_mean_fixed_hard_neg_sim'].item():.4f} / "
                            f"{cat_losses['t2i_mean_hard_neg_sim'].item():.4f}"
                        )
                    else:
                        print(
                            f"T2I valid/reliable: "
                            f"{t2i_valid}/{t2i_reliable} "
                            f"({t2i_reliable / max(t2i_valid, 1):.2%})"
                        )

                    print(
                        f"T2I active pre/post: "
                        f"{active_before}/{t2i_active} | "
                        f"pre={active_before / max(t2i_valid, 1):.2%}, "
                        f"post/selected="
                        f"{t2i_active / max(t2i_reliable, 1):.2%}"
                    )
                    print(
                        "T2I fixed A/B/C/D : "
                        f"{cat_losses['t2i_region_a_count']}/"
                        f"{cat_losses['t2i_region_b_count']}/"
                        f"{cat_losses['t2i_region_c_count']}/"
                        f"{cat_losses['t2i_region_d_count']}"
                    )
                    print(
                        f"T2I Gsup/Gcat     : "
                        f"{cat_losses['t2i_mean_g_sup'].item():+.4f} / "
                        f"{cat_losses['t2i_mean_g_cat'].item():+.4f}"
                    )
                else:
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

            t2i_reliable = cat_losses.get(
                "t2i_reliable_count",
                t2i_valid,
            )
            t2i_active_before = cat_losses.get(
                "t2i_active_before_gate_count",
                t2i_active,
            )

            total_t2i_valid += t2i_valid
            total_i2t_valid += i2t_valid
            total_t2i_reliable += t2i_reliable
            total_t2i_active_before_gate += t2i_active_before
            total_t2i_active += t2i_active
            total_i2t_active += i2t_active

            total_t2i_region_a += cat_losses.get(
                "t2i_region_a_count",
                0,
            )
            total_t2i_region_b += cat_losses.get(
                "t2i_region_b_count",
                0,
            )
            total_t2i_region_c += cat_losses.get(
                "t2i_region_c_count",
                0,
            )
            total_t2i_region_d += cat_losses.get(
                "t2i_region_d_count",
                0,
            )

            if reliable_t2i_enabled:
                total_t2i_g_sup_sum += (
                    cat_losses["t2i_mean_g_sup"].item()
                    * t2i_reliable
                )
                total_t2i_g_cat_sum += (
                    cat_losses["t2i_mean_g_cat"].item()
                    * t2i_reliable
                )

            if reliable_mining_enabled:
                total_t2i_reliable_candidate += cat_losses[
                    "t2i_reliable_candidate_total"
                ]
                total_t2i_no_reliable += cat_losses[
                    "t2i_no_reliable_count"
                ]
                total_t2i_replacement += cat_losses[
                    "t2i_replacement_count"
                ]
                total_t2i_fixed_neg_sum += (
                    cat_losses[
                        "t2i_mean_fixed_hard_neg_sim"
                    ].item()
                    * t2i_valid
                )

            # Reliable T2I 的 pos/neg mean 是对 selected anchors 求均值；
            # Fixed T2I 则仍然对全部 valid anchors 求均值。
            t2i_stat_count = (
                t2i_reliable
                if reliable_t2i_enabled
                else t2i_valid
            )

            total_t2i_pos_sum += (
                cat_losses["t2i_mean_pos_sim"].item()
                * t2i_stat_count
            )
            total_i2t_pos_sum += (
                cat_losses["i2t_mean_pos_sim"].item()
                * i2t_valid
            )
            total_t2i_neg_sum += (
                cat_losses["t2i_mean_hard_neg_sim"].item()
                * t2i_stat_count
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

                if reliable_t2i_enabled:
                    t2i_reliable = cat_losses[
                        "t2i_reliable_count"
                    ]
                    if reliable_mining_enabled:
                        message += (
                            f" | CatT2I {cat_t2i_loss.item():.6f} "
                            f"(act/sel/valid "
                            f"{t2i_active}/{t2i_reliable}/{t2i_valid}, "
                            f"repl={cat_losses['t2i_replacement_ratio']:.1%})"
                        )
                    else:
                        message += (
                            f" | CatT2I {cat_t2i_loss.item():.6f} "
                            f"(A {t2i_active}/{t2i_reliable}/{t2i_valid})"
                        )
                else:
                    message += (
                        f" | CatT2I {cat_t2i_loss.item():.6f} "
                        f"({t2i_active}/{t2i_valid})"
                    )

                message += (
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
        "cat_t2i_reliable": total_t2i_reliable,
        "cat_t2i_active_before_gate": (
            total_t2i_active_before_gate
        ),
        "cat_t2i_active": total_t2i_active,
        "cat_i2t_active": total_i2t_active,
        "cat_t2i_reliable_ratio": (
            total_t2i_reliable / max(total_t2i_valid, 1)
        ),
        "cat_t2i_active_before_gate_ratio": (
            total_t2i_active_before_gate
            / max(total_t2i_valid, 1)
        ),
        # 保留旧字段语义：active / all valid。
        "cat_t2i_active_ratio": (
            total_t2i_active / max(total_t2i_valid, 1)
        ),
        # 新增：真正进入 A 区监督后，其中多少仍违反 margin。
        "cat_t2i_active_reliable_ratio": (
            total_t2i_active / max(total_t2i_reliable, 1)
        ),
        "cat_i2t_active_ratio": (
            total_i2t_active / max(total_i2t_valid, 1)
        ),
        "cat_t2i_region_a": total_t2i_region_a,
        "cat_t2i_region_b": total_t2i_region_b,
        "cat_t2i_region_c": total_t2i_region_c,
        "cat_t2i_region_d": total_t2i_region_d,
        "cat_t2i_region_a_ratio": (
            total_t2i_region_a / max(total_t2i_valid, 1)
        ),
        "cat_t2i_region_b_ratio": (
            total_t2i_region_b / max(total_t2i_valid, 1)
        ),
        "cat_t2i_region_c_ratio": (
            total_t2i_region_c / max(total_t2i_valid, 1)
        ),
        "cat_t2i_region_d_ratio": (
            total_t2i_region_d / max(total_t2i_valid, 1)
        ),
        "cat_t2i_g_sup": (
            total_t2i_g_sup_sum / max(total_t2i_reliable, 1)
        ),
        "cat_t2i_g_cat": (
            total_t2i_g_cat_sum / max(total_t2i_reliable, 1)
        ),
        "cat_t2i_reliability_mode": reliability_mode,
        "cat_t2i_reliable_candidate_total": (
            total_t2i_reliable_candidate
        ),
        "cat_t2i_avg_reliable_negatives": (
            total_t2i_reliable_candidate
            / max(total_t2i_valid, 1)
        ),
        "cat_t2i_no_reliable": total_t2i_no_reliable,
        "cat_t2i_no_reliable_ratio": (
            total_t2i_no_reliable
            / max(total_t2i_valid, 1)
        ),
        "cat_t2i_replacement": total_t2i_replacement,
        "cat_t2i_replacement_ratio": (
            total_t2i_replacement
            / max(total_t2i_valid, 1)
        ),
        "cat_t2i_fixed_hard_neg_sim": (
            total_t2i_fixed_neg_sum
            / max(total_t2i_valid, 1)
            if reliable_mining_enabled
            else 0.0
        ),
        "cat_t2i_pos_sim": (
            total_t2i_pos_sum
            / max(
                total_t2i_reliable
                if reliable_t2i_enabled
                else total_t2i_valid,
                1,
            )
        ),
        "cat_i2t_pos_sim": (
            total_i2t_pos_sum / max(total_i2t_valid, 1)
        ),
        "cat_t2i_hard_neg_sim": (
            total_t2i_neg_sum
            / max(
                total_t2i_reliable
                if reliable_t2i_enabled
                else total_t2i_valid,
                1,
            )
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
