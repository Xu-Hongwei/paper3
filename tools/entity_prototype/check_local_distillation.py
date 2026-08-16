import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import load_config, set_seed
from datasets import create_dataset
from models import CLIPRetrieval
from models import LocalPatchHead
from models import (
    sample_region_boxes,
    crop_regions,
    pool_region_patches,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Local Self-Distillation smoke test."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-regions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_baseline(model, checkpoint_path):
    """加载旧 RSICD baseline，允许 Adapter 参数缺失。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    invalid_missing = [
        name
        for name in missing
        if not name.startswith(
            ("visual_adapter.", "text_adapter.")
        )
    ]

    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch: "
            f"missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )

    return checkpoint


def build_image_batch(dataset, batch_size):
    """取前若干训练图，仅使用图像，不依赖 Entity。"""
    images = []

    for index in range(batch_size):
        image, _, _, _ = dataset[index]
        images.append(image)

    return torch.stack(images, dim=0)


def check_gradients(model, local_head):
    """确保 Backbone 无梯度，Local Head 有有效梯度。"""
    backbone_grads = [
        param.grad
        for param in model.backbone.parameters()
        if param.grad is not None
    ]

    if backbone_grads:
        raise RuntimeError(
            "Frozen CLIP backbone unexpectedly received gradients."
        )

    local_grads = [
        param.grad
        for param in local_head.parameters()
        if param.grad is not None
    ]

    if not local_grads:
        raise RuntimeError(
            "Local Head did not receive gradients."
        )

    finite_nonzero = any(
        torch.isfinite(grad).all()
        and grad.abs().max().item() > 0
        for grad in local_grads
    )

    if not finite_nonzero:
        raise RuntimeError(
            "Local Head gradients are zero or non-finite."
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("LOCAL SELF-DISTILLATION SMOKE TEST")
    print("=" * 72)
    print(f"Device        : {device}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Regions/image : {args.num_regions}")

    # ------------------------------------------------------------------
    # Frozen RSICD CLIP
    # ------------------------------------------------------------------
    model = CLIPRetrieval(
        config["model"]
    )

    checkpoint = load_baseline(
        model,
        args.checkpoint,
    )

    model = model.to(device)
    model.eval()

    for param in model.backbone.parameters():
        param.requires_grad = False

    # 这一阶段只训练 Local Head。
    local_head = LocalPatchHead(
        dim=512
    ).to(device)

    local_head.train()

    # ------------------------------------------------------------------
    # Dataset：使用 deterministic eval transform。
    # ------------------------------------------------------------------
    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    images = build_image_batch(
        train_dataset,
        args.batch_size,
    ).to(device)

    if images.shape[-2:] != (224, 224):
        raise RuntimeError(
            f"Expected 224x224 model input, got "
            f"{tuple(images.shape[-2:])}"
        )

    # ------------------------------------------------------------------
    # 随机区域
    # ------------------------------------------------------------------
    boxes = sample_region_boxes(
        batch_size=args.batch_size,
        image_size=224,
        num_regions=args.num_regions,
        min_scale=0.20,
        max_scale=0.60,
        device=device,
    )

    expected_regions = (
        args.batch_size
        * args.num_regions
    )

    if boxes.shape != (
        expected_regions,
        5,
    ):
        raise RuntimeError(
            f"Unexpected boxes shape: {tuple(boxes.shape)}"
        )

    # ------------------------------------------------------------------
    # Teacher crops
    # ------------------------------------------------------------------
    crops = crop_regions(
        images,
        boxes,
        output_size=224,
    )

    if crops.shape != (
        expected_regions,
        3,
        224,
        224,
    ):
        raise RuntimeError(
            f"Unexpected crop shape: {tuple(crops.shape)}"
        )

    # ------------------------------------------------------------------
    # Frozen CLIP forward
    # ------------------------------------------------------------------
    with torch.no_grad():
        _, patch_features = (
            model.backbone.encode_image_with_patches(
                images,
                normalize=False,
            )
        )

        teacher_features = (
            model.backbone.encode_image(
                crops,
                normalize=True,
            )
        )

    if patch_features.shape != (
        args.batch_size,
        49,
        512,
    ):
        raise RuntimeError(
            "Unexpected Patch shape: "
            f"{tuple(patch_features.shape)}"
        )

    if teacher_features.shape != (
        expected_regions,
        512,
    ):
        raise RuntimeError(
            "Unexpected Teacher shape: "
            f"{tuple(teacher_features.shape)}"
        )

    # ------------------------------------------------------------------
    # Zero-init 等价性
    # ------------------------------------------------------------------
    enhanced_patches = local_head(
        patch_features
    )

    zero_init_error = (
        enhanced_patches
        - patch_features
    ).abs().max().item()

    if zero_init_error > 1e-7:
        raise RuntimeError(
            "Local Head zero-init equivalence failed: "
            f"max abs error={zero_init_error}"
        )

    # ------------------------------------------------------------------
    # Region Student
    # ------------------------------------------------------------------
    student_regions, patch_weights = (
        pool_region_patches(
            enhanced_patches,
            boxes,
            image_size=224,
            grid_size=7,
        )
    )

    if student_regions.shape != (
        expected_regions,
        512,
    ):
        raise RuntimeError(
            "Unexpected Student shape: "
            f"{tuple(student_regions.shape)}"
        )

    weight_sums = patch_weights.sum(
        dim=1
    )

    max_weight_sum_error = (
        weight_sums - 1.0
    ).abs().max().item()

    if max_weight_sum_error > 1e-6:
        raise RuntimeError(
            "Patch weights are not normalized: "
            f"max error={max_weight_sum_error}"
        )

    student_regions = F.normalize(
        student_regions,
        dim=-1,
    )

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------
    cosine = (
        student_regions
        * teacher_features.detach()
    ).sum(dim=-1)

    loss = (
        1.0 - cosine
    ).mean()

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite distillation loss: {loss.item()}"
        )

    # ------------------------------------------------------------------
    # Backward：只允许 Local Head 收梯度
    # ------------------------------------------------------------------
    local_head.zero_grad(
        set_to_none=True
    )

    loss.backward()

    check_gradients(
        model,
        local_head,
    )

    grad_max = max(
        param.grad.abs().max().item()
        for param in local_head.parameters()
        if param.grad is not None
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("Shapes")
    print(f"Images           : {tuple(images.shape)}")
    print(f"Boxes            : {tuple(boxes.shape)}")
    print(f"Teacher crops    : {tuple(crops.shape)}")
    print(f"L12 patches      : {tuple(patch_features.shape)}")
    print(f"Teacher features : {tuple(teacher_features.shape)}")
    print(f"Student regions  : {tuple(student_regions.shape)}")

    print()
    print("Checks")
    print(
        f"Zero-init max abs error : "
        f"{zero_init_error:.8e}"
    )
    print(
        f"Patch weight sum error  : "
        f"{max_weight_sum_error:.8e}"
    )
    print(
        f"Mean teacher/student cos: "
        f"{cosine.mean().item():.6f}"
    )
    print(
        f"Min teacher/student cos : "
        f"{cosine.min().item():.6f}"
    )
    print(
        f"Max teacher/student cos : "
        f"{cosine.max().item():.6f}"
    )
    print(
        f"Distillation loss       : "
        f"{loss.item():.6f}"
    )
    print(
        f"Local Head max grad     : "
        f"{grad_max:.8e}"
    )

    print()
    print("Gradient check : Backbone frozen | Local Head active")

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(
                f"Baseline epoch : "
                f"{checkpoint['epoch']}"
            )

        metrics = checkpoint.get(
            "metrics"
        )
        if (
            isinstance(metrics, dict)
            and "mR" in metrics
        ):
            print(
                f"Baseline Val mR: "
                f"{metrics['mR']:.4f}"
            )

    print()
    print("=" * 72)
    print("LOCAL SELF-DISTILLATION SMOKE TEST: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
