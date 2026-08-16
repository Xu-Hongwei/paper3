import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from utils import load_config, set_seed
from datasets import create_dataset
from models import CLIPRetrieval


DEFAULT_KEYWORDS = [
    "airplane", "aircraft", "plane", "runway",
    "ship", "port", "bridge", "river",
    "stadium", "road", "building",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较 baseline 与 B2 的 Entity-Patch Grounding。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/grounding_vis")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    """兼容旧 baseline 和包含 Adapter 的新 checkpoint。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    invalid_missing = [
        name for name in missing
        if not name.startswith(("visual_adapter.", "text_adapter."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint 不兼容，missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )

    return checkpoint


def decode_entity(model, caption, span):
    """根据 CLIP token span 恢复 Entity 文本。"""
    tokens = model.backbone.tokenize([caption])[0]
    start, end = map(int, span)

    tokenizer = model.backbone.tokenizer
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(tokens[start:end].tolist())
        text = (
            text.replace("<|startoftext|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        if text:
            return text

    return f"tokens[{start}:{end}]"


def recover_input_image(image_tensor, preprocess):
    """反归一化模型实际输入图，保证 Patch 坐标与热图一致。"""
    mean = None
    std = None

    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            mean = torch.as_tensor(transform.mean).view(3, 1, 1)
            std = torch.as_tensor(transform.std).view(3, 1, 1)
            break

    image = image_tensor.detach().cpu()
    if mean is not None and std is not None:
        image = image * std + mean

    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def select_samples(dataset, num_samples, indices):
    """优先选择包含典型遥感实体的不同图像。"""
    if indices:
        selected = []
        for index in indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(f"Invalid dataset index: {index}")
            if dataset[index][3].shape[0] == 0:
                raise ValueError(f"Dataset index {index} has no valid Entity span.")
            selected.append(index)
        return selected

    selected = []
    used_images = set()

    for keyword in DEFAULT_KEYWORDS:
        for index, ann in enumerate(dataset.ann):
            if len(selected) >= num_samples:
                break
            if keyword not in ann["caption"].lower():
                continue
            if ann["image"] in used_images:
                continue

            pair_index = ann["_pair_index"]
            if dataset._get_entity_spans(pair_index).shape[0] == 0:
                continue

            selected.append(index)
            used_images.add(ann["image"])
            break

    if len(selected) < num_samples:
        for index, ann in enumerate(dataset.ann):
            if len(selected) >= num_samples:
                break
            if ann["image"] in used_images:
                continue

            pair_index = ann["_pair_index"]
            if dataset._get_entity_spans(pair_index).shape[0] == 0:
                continue

            selected.append(index)
            used_images.add(ann["image"])

    return selected


@torch.no_grad()
def compute_grounding(model, image, caption, spans, device):
    """计算 Entity-Patch attention，并返回可视化统计。"""
    image = image.unsqueeze(0).to(device)

    _, patch_features = model.backbone.encode_image_with_patches(
        image,
        normalize=False,
    )
    _, token_features = model.backbone.encode_text_with_tokens(
        [caption],
        normalize=False,
    )

    spans_device = spans.to(device)
    sample_ids = torch.zeros(
        spans.shape[0],
        dtype=torch.long,
        device=device,
    )

    entity_features = model._pool_entity_spans(
        token_features,
        spans_device,
        sample_ids,
    )

    patch_features = F.normalize(
        patch_features + model.visual_adapter(patch_features),
        dim=-1,
    )
    entity_features = F.normalize(
        entity_features + model.text_adapter(entity_features),
        dim=-1,
    )

    similarity = entity_features @ patch_features[0].transpose(0, 1)
    weights = F.softmax(
        similarity / model.grounding_temperature,
        dim=-1,
    )

    num_patches = weights.shape[1]
    grid_size = int(round(num_patches ** 0.5))
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"Patch 数量 {num_patches} 不能恢复为方形网格。")

    results = []
    for entity_index, span in enumerate(spans.tolist()):
        weight = weights[entity_index]
        entropy = -(
            weight * weight.clamp_min(1e-12).log()
        ).sum().item()
        normalized_entropy = entropy / np.log(num_patches)

        top_values, top_indices = torch.topk(
            weight,
            k=min(num_patches, 5),
        )

        results.append({
            "entity": decode_entity(model, caption, span),
            "span": span,
            "heatmap": weight.view(grid_size, grid_size).cpu().numpy(),
            "entropy": normalized_entropy,
            "top5_mass": top_values.sum().item(),
            "top_indices": top_indices.cpu().tolist(),
        })

    return results


def draw_topk_boxes(ax, top_indices, grid_size, image_size, topk):
    """在模型输入图上标出 Top-K Patch。"""
    patch_size = image_size / grid_size

    for patch_index in top_indices[:topk]:
        row = patch_index // grid_size
        col = patch_index % grid_size
        ax.add_patch(
            Rectangle(
                (col * patch_size, row * patch_size),
                patch_size,
                patch_size,
                fill=False,
                linewidth=2,
            )
        )


def draw_heatmap(ax, image, result, topk, prefix):
    """叠加 Grounding heatmap 与 Top-K Patch。"""
    height, width = image.shape[:2]
    heatmap = result["heatmap"]

    ax.imshow(image)
    ax.imshow(
        heatmap,
        extent=(0, width, height, 0),
        alpha=0.45,
        interpolation="nearest",
    )
    draw_topk_boxes(
        ax,
        result["top_indices"],
        heatmap.shape[0],
        width,
        topk,
    )
    ax.set_title(
        f"{prefix}: {result['entity']}\n"
        f"H={result['entropy']:.3f}, "
        f"Top5={result['top5_mass']:.3f}"
    )
    ax.axis("off")


def visualize_sample(
    output_path,
    image,
    caption,
    baseline_results,
    trained_results,
    topk,
):
    """每行一个 Entity：原图、Baseline、B2 并排比较。"""
    if len(baseline_results) != len(trained_results):
        raise RuntimeError("Baseline/B2 Entity 数量不一致。")

    rows = len(baseline_results)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(12, 4 * rows),
        squeeze=False,
    )

    for row, (baseline, trained) in enumerate(
        zip(baseline_results, trained_results)
    ):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Input\nEntity: {baseline['entity']}")
        axes[row, 0].axis("off")

        draw_heatmap(
            axes[row, 1],
            image,
            baseline,
            topk,
            "Baseline",
        )
        draw_heatmap(
            axes[row, 2],
            image,
            trained,
            topk,
            "B2 Entity",
        )

    fig.suptitle(caption, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "grounding_summary.jsonl"

    baseline_model = CLIPRetrieval(config["model"])
    baseline_checkpoint = load_checkpoint(
        baseline_model,
        args.baseline,
    )
    baseline_model = baseline_model.to(device).eval()

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=baseline_model.backbone.preprocess_val,
        eval_transform=baseline_model.backbone.preprocess_val,
    )

    selected = select_samples(
        train_dataset,
        args.num_samples,
        args.indices,
    )

    print("=" * 72)
    print("ENTITY GROUNDING VISUALIZATION")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Baseline        : {args.baseline}")
    print(f"B2 checkpoint   : {args.checkpoint}")
    print(f"Selected samples: {selected}")
    print(f"Output dir      : {output_dir}")

    samples = []
    baseline_all = {}

    for index in selected:
        image, caption, _, spans = train_dataset[index]
        baseline_results = compute_grounding(
            baseline_model,
            image,
            caption,
            spans,
            device,
        )

        samples.append({
            "index": index,
            "image": image.cpu(),
            "caption": caption,
            "spans": spans.cpu(),
        })
        baseline_all[index] = baseline_results
        print(f"[Baseline] index={index}, caption={caption}")

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model = CLIPRetrieval(config["model"])
    trained_checkpoint = load_checkpoint(
        trained_model,
        args.checkpoint,
    )
    trained_model = trained_model.to(device).eval()

    records = []

    for sample in samples:
        index = sample["index"]
        image = sample["image"]
        caption = sample["caption"]
        spans = sample["spans"]

        trained_results = compute_grounding(
            trained_model,
            image,
            caption,
            spans,
            device,
        )

        display_image = recover_input_image(
            image,
            trained_model.backbone.preprocess_val,
        )

        output_path = output_dir / f"sample_{index:05d}.png"
        visualize_sample(
            output_path,
            display_image,
            caption,
            baseline_all[index],
            trained_results,
            args.topk,
        )

        for baseline, trained in zip(
            baseline_all[index],
            trained_results,
        ):
            records.append({
                "sample_index": index,
                "caption": caption,
                "entity": baseline["entity"],
                "span": baseline["span"],
                "baseline_entropy": baseline["entropy"],
                "trained_entropy": trained["entropy"],
                "entropy_delta": trained["entropy"] - baseline["entropy"],
                "baseline_top5_mass": baseline["top5_mass"],
                "trained_top5_mass": trained["top5_mass"],
                "top5_mass_delta": (
                    trained["top5_mass"] - baseline["top5_mass"]
                ),
            })

        print(f"[Saved] {output_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 72)
    print(f"Images saved  : {len(samples)}")
    print(f"Summary       : {summary_path}")
    print(f"Baseline epoch: {baseline_checkpoint.get('epoch')}")
    print(f"B2 epoch      : {trained_checkpoint.get('epoch')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
