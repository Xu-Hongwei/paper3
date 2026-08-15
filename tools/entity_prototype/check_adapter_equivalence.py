import argparse

import torch
import torch.nn.functional as F

from utils import load_config
from datasets import create_dataset, create_loader
from models import CLIPRetrieval
from evaluation import evaluate_retrieval


def load_baseline(model, checkpoint_path):
    """加载旧 baseline，允许新增 Adapter 参数缺失。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    invalid_missing = [
        name for name in missing
        if not name.startswith(("visual_adapter.", "text_adapter."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint 不兼容，missing={invalid_missing}, unexpected={unexpected}"
        )

    print(f"Checkpoint : {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Epoch      : {checkpoint['epoch']}")

    metrics = checkpoint.get("metrics", {})
    if isinstance(metrics, dict) and "mR" in metrics:
        print(f"Saved mR   : {metrics['mR']:.4f}")

    return checkpoint


@torch.no_grad()
def check_feature_equivalence(model, loader, texts, text_batch_size, device):
    """检查原 CLIP 与 zero-init Adapter 的全局特征是否一致。"""
    image_error = 0.0
    text_error = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        raw = model.backbone.encode_image(images, normalize=False)
        baseline = F.normalize(raw, dim=-1)
        adapted = F.normalize(
            raw + model.visual_adapter(raw),
            dim=-1,
        )

        image_error = max(
            image_error,
            (baseline - adapted).abs().max().item(),
        )

    for start in range(0, len(texts), text_batch_size):
        batch_texts = texts[start:start + text_batch_size]

        raw = model.backbone.encode_text(batch_texts, normalize=False)
        baseline = F.normalize(raw, dim=-1)
        adapted = F.normalize(
            raw + model.text_adapter(raw),
            dim=-1,
        )

        text_error = max(
            text_error,
            (baseline - adapted).abs().max().item(),
        )

    return image_error, text_error


def main():
    parser = argparse.ArgumentParser(
        description="验证 zero-init Adapter 是否保持 baseline 表示与正式检索结果。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPRetrieval(config["model"])
    checkpoint = load_baseline(model, args.checkpoint)
    model = model.to(device).eval()

    val_dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split="val",
        eval_transform=model.backbone.preprocess_val,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=config["training"].get("eval_batch_size", 128),
        num_workers=config["training"].get("num_workers", 8),
        is_train=False,
        pin_memory=True,
    )

    text_batch_size = config["training"].get("text_batch_size", 256)

    print("\nChecking feature equivalence...")
    image_error, text_error = check_feature_equivalence(
        model=model,
        loader=val_loader,
        texts=val_dataset.text,
        text_batch_size=text_batch_size,
        device=device,
    )

    print("Running official retrieval evaluation...")
    metrics, _ = evaluate_retrieval(
        model=model,
        data_loader=val_loader,
        dataset=val_dataset,
        device=device,
        text_batch_size=text_batch_size,
    )

    saved_metrics = checkpoint.get("metrics", {})

    print("\n" + "=" * 64)
    print("Zero-init Adapter Equivalence Test")
    print("=" * 64)
    print(f"Val images          : {len(val_dataset)}")
    print(f"Val captions        : {len(val_dataset.text)}")
    print(f"Image max abs error : {image_error:.8e}")
    print(f"Text max abs error  : {text_error:.8e}")

    keys = (
        "i2t_r1", "i2t_r5", "i2t_r10",
        "t2i_r1", "t2i_r5", "t2i_r10", "mR",
    )

    print("\nOfficial retrieval metrics")
    print("-" * 64)

    metric_diffs = []
    for key in keys:
        current = float(metrics[key])
        saved = saved_metrics.get(key)

        if saved is None:
            print(f"{key:<10}: current={current:.6f} | saved=N/A")
            continue

        saved = float(saved)
        diff = current - saved
        metric_diffs.append(abs(diff))

        print(
            f"{key:<10}: current={current:.6f} | "
            f"saved={saved:.6f} | diff={diff:+.8f}"
        )

    max_metric_diff = max(metric_diffs) if metric_diffs else float("nan")

    print("-" * 64)
    print(f"Max metric diff     : {max_metric_diff:.8e}")

    feature_pass = image_error <= 1e-7 and text_error <= 1e-7
    metric_pass = not metric_diffs or max_metric_diff <= 1e-8

    if feature_pass and metric_pass:
        print("PASS: zero-init Adapter 与 baseline 等价。")
    else:
        print("CHECK: 存在差异，请检查上方 feature / metric diff。")


if __name__ == "__main__":
    main()
