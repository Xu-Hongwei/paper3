import math

import torch
import torch.nn.functional as F


def sample_region_boxes(
    batch_size,
    image_size=224,
    num_regions=2,
    min_scale=0.20,
    max_scale=0.60,
    min_aspect_ratio=0.75,
    max_aspect_ratio=1.33,
    device=None,
):
    """
    为 batch 中每张图随机采样若干矩形区域。

    Returns:
        boxes: [R, 5]
            每行为:
            [sample_id, x1, y1, x2, y2]

            坐标基于模型实际输入图，例如 224×224。
            x2 / y2 为右下边界，不包含该像素。

    Notes:
        scale 表示区域面积占整图面积的比例。
        第一版故意限制在 20%~60%，避免区域太小或接近整图。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if num_regions <= 0:
        raise ValueError("num_regions must be > 0.")
    if not (0 < min_scale <= max_scale <= 1):
        raise ValueError("Require 0 < min_scale <= max_scale <= 1.")
    if not (0 < min_aspect_ratio <= max_aspect_ratio):
        raise ValueError(
            "Require 0 < min_aspect_ratio <= max_aspect_ratio."
        )

    image_area = float(image_size * image_size)
    boxes = []

    for sample_id in range(batch_size):
        for _ in range(num_regions):
            scale = torch.empty(1).uniform_(
                min_scale,
                max_scale,
            ).item()

            log_ratio = torch.empty(1).uniform_(
                math.log(min_aspect_ratio),
                math.log(max_aspect_ratio),
            ).item()
            aspect_ratio = math.exp(log_ratio)

            target_area = image_area * scale
            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            width = min(max(width, 1), image_size)
            height = min(max(height, 1), image_size)

            max_x1 = image_size - width
            max_y1 = image_size - height

            x1 = (
                torch.randint(0, max_x1 + 1, (1,)).item()
                if max_x1 > 0
                else 0
            )
            y1 = (
                torch.randint(0, max_y1 + 1, (1,)).item()
                if max_y1 > 0
                else 0
            )

            x2 = x1 + width
            y2 = y1 + height

            boxes.append(
                [sample_id, x1, y1, x2, y2]
            )

    return torch.tensor(
        boxes,
        dtype=torch.float32,
        device=device,
    )


def crop_regions(
    images,
    boxes,
    output_size=224,
):
    """
    从模型实际输入 Tensor 中裁出区域并 resize。

    Args:
        images:
            [B, C, H, W]
            已经过 CLIP normalization 也可以直接使用。

        boxes:
            [R, 5]
            [sample_id, x1, y1, x2, y2]

    Returns:
        crops:
            [R, C, output_size, output_size]

    说明:
        CLIP normalization 是逐通道仿射变换，
        因此第一版直接在 normalized tensor 上 crop + resize，
        不需要反归一化后再重新 Normalize。
    """
    if images.ndim != 4:
        raise ValueError(
            f"images must be [B,C,H,W], got {tuple(images.shape)}"
        )
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(
            f"boxes must be [R,5], got {tuple(boxes.shape)}"
        )

    _, _, height, width = images.shape
    crops = []

    for box in boxes:
        sample_id = int(box[0].item())
        x1 = int(round(box[1].item()))
        y1 = int(round(box[2].item()))
        x2 = int(round(box[3].item()))
        y2 = int(round(box[4].item()))

        if sample_id < 0 or sample_id >= images.shape[0]:
            raise IndexError(f"Invalid sample_id: {sample_id}")

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        crop = images[
            sample_id:sample_id + 1,
            :,
            y1:y2,
            x1:x2,
        ]

        crop = F.interpolate(
            crop,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )

        crops.append(crop)

    if not crops:
        return images.new_empty(
            (0, images.shape[1], output_size, output_size)
        )

    return torch.cat(crops, dim=0)


def boxes_to_patch_weights(
    boxes,
    image_size=224,
    grid_size=7,
):
    """
    将像素区域映射到 Patch Grid，使用面积交叠比例作为软权重。

    Args:
        boxes:
            [R, 5]
            [sample_id, x1, y1, x2, y2]

    Returns:
        weights:
            [R, grid_size * grid_size]

            每一行归一化后和为 1。

    为什么使用软权重:
        随机 crop 通常不会刚好落在 32×32 Patch 边界上。
        使用 crop 与各 Patch 的交叠面积，比简单的
        “Patch 中心是否落入 crop”更准确、也更稳定。
    """
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(
            f"boxes must be [R,5], got {tuple(boxes.shape)}"
        )

    device = boxes.device
    dtype = boxes.dtype
    patch_size = float(image_size) / grid_size

    rows = torch.arange(
        grid_size,
        device=device,
        dtype=dtype,
    )
    cols = torch.arange(
        grid_size,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        rows,
        cols,
        indexing="ij",
    )

    patch_x1 = (xx * patch_size).reshape(-1)
    patch_y1 = (yy * patch_size).reshape(-1)
    patch_x2 = patch_x1 + patch_size
    patch_y2 = patch_y1 + patch_size

    region_x1 = boxes[:, 1:2]
    region_y1 = boxes[:, 2:3]
    region_x2 = boxes[:, 3:4]
    region_y2 = boxes[:, 4:5]

    inter_w = (
        torch.minimum(region_x2, patch_x2.unsqueeze(0))
        - torch.maximum(region_x1, patch_x1.unsqueeze(0))
    ).clamp_min(0)

    inter_h = (
        torch.minimum(region_y2, patch_y2.unsqueeze(0))
        - torch.maximum(region_y1, patch_y1.unsqueeze(0))
    ).clamp_min(0)

    overlap = inter_w * inter_h
    weight_sum = overlap.sum(dim=1, keepdim=True)

    if torch.any(weight_sum <= 0):
        raise RuntimeError(
            "At least one crop does not overlap any Patch."
        )

    return overlap / weight_sum


def pool_region_patches(
    patch_features,
    boxes,
    image_size=224,
    grid_size=7,
):
    """
    从完整图 Patch Features 中池化出每个 crop 对应的 Region Feature。

    Args:
        patch_features:
            [B, N_patch, D]

        boxes:
            [R, 5]
            [sample_id, x1, y1, x2, y2]

    Returns:
        region_features:
            [R, D]

        patch_weights:
            [R, N_patch]
    """
    if patch_features.ndim != 3:
        raise ValueError(
            "patch_features must be [B,N_patch,D], got "
            f"{tuple(patch_features.shape)}"
        )

    expected_patches = grid_size * grid_size
    if patch_features.shape[1] != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} patches for "
            f"{grid_size}x{grid_size} grid, got "
            f"{patch_features.shape[1]}."
        )

    boxes = boxes.to(
        device=patch_features.device,
        dtype=patch_features.dtype,
    )

    sample_ids = boxes[:, 0].long()
    if torch.any(sample_ids < 0) or torch.any(
        sample_ids >= patch_features.shape[0]
    ):
        raise IndexError("boxes contain invalid sample_id.")

    patch_weights = boxes_to_patch_weights(
        boxes,
        image_size=image_size,
        grid_size=grid_size,
    )

    selected_patches = patch_features[
        sample_ids
    ]  # [R, N_patch, D]

    region_features = torch.sum(
        selected_patches
        * patch_weights.unsqueeze(-1),
        dim=1,
    )

    return region_features, patch_weights
