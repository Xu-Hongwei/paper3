import argparse
import csv
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
        description="诊断 CLIP 第 6/8/10/12 层 Entity-Patch Grounding。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/intermediate_grounding")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_baseline(model, checkpoint_path):
    """加载旧 RSICD baseline，允许 Adapter 参数缺失，但诊断时完全不使用 Adapter。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

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
        name for name in missing
        if not name.startswith(("visual_adapter.", "text_adapter."))
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model architecture mismatch, "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )

    print(f"Checkpoint       : {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch : {checkpoint['epoch']}")

    metrics = checkpoint.get("metrics", {})
    if isinstance(metrics, dict) and "mR" in metrics:
        print(f"Stored Val mR    : {metrics['mR']:.4f}")

    return checkpoint


def decode_entity(model, caption, span):
    """根据当前 caption 的 CLIP token span 恢复 Entity 文本。"""
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
    """反归一化模型实际输入图，保证 7×7 Patch 与热图坐标一致。"""
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

            pair_index = dataset.ann[index]["_pair_index"]
            spans = dataset._get_entity_spans(pair_index)
            if spans.shape[0] == 0:
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
            spans = dataset._get_entity_spans(pair_index)
            if spans.shape[0] == 0:
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
            spans = dataset._get_entity_spans(pair_index)
            if spans.shape[0] == 0:
                continue

            selected.append(index)
            used_images.add(ann["image"])

    return selected


@torch.no_grad()
def extract_entity_features(model, caption, spans, device):
    """从完整 caption 的 contextualized token features 中提取 Entity。"""
    _, token_features = model.backbone.encode_text_with_tokens(
        [caption],
        normalize=False,
    )

    spans = spans.to(device)
    sample_ids = torch.zeros(
        spans.shape[0],
        dtype=torch.long,
        device=device,
    )

    entity_features = model._pool_entity_spans(
        token_features,
        spans,
        sample_ids,
    )
    return F.normalize(entity_features, dim=-1)


@torch.no_grad()
def compute_layer_grounding(
    model,
    image,
    caption,
    spans,
    layers,
    temperature,
    topk,
    device,
):
    """同一次 CLIP Vision forward 中比较多个 Transformer 层。"""
    image_batch = image.unsqueeze(0).to(device)

    _, layer_patches = model.backbone.encode_image_intermediate_patches(
        image_batch,
        layers=tuple(layers),
        normalize=False,
    )
    entity_features = extract_entity_features(
        model,
        caption,
        spans,
        device,
    )

    results = {}

    for layer in layers:
        patch_features = F.normalize(
            layer_patches[layer][0],
            dim=-1,
        )

        similarity = entity_features @ patch_features.transpose(0, 1)
        weights = F.softmax(
            similarity / temperature,
            dim=-1,
        )

        num_patches = weights.shape[1]
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise RuntimeError(
                f"Layer {layer}: Patch 数量 {num_patches} 不能恢复成方形网格。"
            )

        layer_results = []

        for entity_index, span in enumerate(spans.tolist()):
            weight = weights[entity_index]
            sim = similarity[entity_index]

            entropy = -(
                weight * weight.clamp_min(1e-12).log()
            ).sum().item()
            normalized_entropy = entropy / np.log(num_patches)

            k = min(topk, num_patches)
            top_values, top_indices = torch.topk(weight, k=k)

            layer_results.append({
                "entity": decode_entity(model, caption, span),
                "span": span,
                "heatmap": weight.view(grid_size, grid_size).cpu().numpy(),
                "entropy": normalized_entropy,
                "topk_mass": top_values.sum().item(),
                "top_indices": top_indices.cpu().tolist(),
                "sim_mean": sim.mean().item(),
                "sim_std": sim.std(unbiased=False).item(),
                "sim_max": sim.max().item(),
                "sim_min": sim.min().item(),
                "sim_max_minus_mean": (
                    sim.max() - sim.mean()
                ).item(),
                "sim_max_minus_min": (
                    sim.max() - sim.min()
                ).item(),
            })

        results[layer] = layer_results

    return results


@torch.no_grad()
def check_final_layer_equivalence(model, image, final_layer, device):
    """
    验证新增多层接口中的最后层 Patch 与原 encode_image_with_patches()
    是否一致，避免诊断接口本身改变特征。
    """
    image = image.unsqueeze(0).to(device)

    _, original_patch = model.backbone.encode_image_with_patches(
        image,
        normalize=False,
    )
    _, intermediate = model.backbone.encode_image_intermediate_patches(
        image,
        layers=(final_layer,),
        normalize=False,
    )

    candidate = intermediate[final_layer]
    if original_patch.shape != candidate.shape:
        raise RuntimeError(
            f"Final layer shape mismatch: "
            f"{tuple(original_patch.shape)} vs {tuple(candidate.shape)}"
        )

    return (original_patch - candidate).abs().max().item()


def draw_topk_boxes(ax, top_indices, grid_size, width, height, topk):
    """在模型输入图上标出 Top-K Patch。"""
    patch_w = width / grid_size
    patch_h = height / grid_size

    for patch_index in top_indices[:topk]:
        row = patch_index // grid_size
        col = patch_index % grid_size

        ax.add_patch(
            Rectangle(
                (col * patch_w, row * patch_h),
                patch_w,
                patch_h,
                fill=False,
                linewidth=2,
            )
        )


def draw_grounding(ax, image, result, layer, topk):
    """绘制单层 Entity-Patch heatmap。"""
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
        height,
        topk,
    )

    ax.set_title(
        f"L{layer} | H={result['entropy']:.3f}\n"
        f"Top{topk}={result['topk_mass']:.3f}, "
        f"σ={result['sim_std']:.3f}"
    )
    ax.axis("off")


def visualize_sample(
    output_path,
    image,
    caption,
    layer_results,
    layers,
    topk,
):
    """每行一个 Entity：原图 + 多个 Transformer 层并排比较。"""
    num_entities = len(layer_results[layers[0]])
    cols = len(layers) + 1

    fig, axes = plt.subplots(
        num_entities,
        cols,
        figsize=(4 * cols, 4 * num_entities),
        squeeze=False,
    )

    for entity_index in range(num_entities):
        entity_name = layer_results[layers[0]][entity_index]["entity"]

        axes[entity_index, 0].imshow(image)
        axes[entity_index, 0].set_title(
            f"Input\nEntity: {entity_name}"
        )
        axes[entity_index, 0].axis("off")

        for col, layer in enumerate(layers, start=1):
            draw_grounding(
                axes[entity_index, col],
                image,
                layer_results[layer][entity_index],
                layer,
                topk,
            )

    fig.suptitle(caption, fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_aggregate_csv(records, layers, output_path):
    """按层汇总所有选中 Entity 的平均统计。"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "num_entities",
                "mean_entropy",
                "mean_topk_mass",
                "mean_sim_std",
                "mean_max_minus_mean",
                "mean_max_minus_min",
            ],
        )
        writer.writeheader()

        for layer in layers:
            rows = [
                record
                for record in records
                if record["layer"] == layer
            ]

            writer.writerow({
                "layer": layer,
                "num_entities": len(rows),
                "mean_entropy": np.mean(
                    [row["entropy"] for row in rows]
                ),
                "mean_topk_mass": np.mean(
                    [row["topk_mass"] for row in rows]
                ),
                "mean_sim_std": np.mean(
                    [row["sim_std"] for row in rows]
                ),
                "mean_max_minus_mean": np.mean(
                    [row["sim_max_minus_mean"] for row in rows]
                ),
                "mean_max_minus_min": np.mean(
                    [row["sim_max_minus_min"] for row in rows]
                ),
            })


def main():
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")

    set_seed(args.seed)

    config = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "intermediate_grounding.jsonl"
    aggregate_path = output_dir / "layer_summary.csv"

    model = CLIPRetrieval(config["model"])
    load_baseline(
        model,
        args.checkpoint,
    )
    model = model.to(device).eval()

    # 诊断必须使用确定性的 eval transform，不能使用随机训练增强。
    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    selected = select_samples(
        train_dataset,
        args.num_samples,
        args.indices,
    )

    print()
    print("=" * 72)
    print("INTERMEDIATE CLIP PATCH GROUNDING DIAGNOSTIC")
    print("=" * 72)
    print(f"Device           : {device}")
    print(f"Layers           : {args.layers}")
    print(f"Temperature      : {args.temperature}")
    print(f"Top-K            : {args.topk}")
    print(f"Selected samples : {selected}")
    print(f"Output dir       : {output_dir}")

    # 首张样本先验证 L12 与旧接口的一致性。
    first_image, _, _, _ = train_dataset[selected[0]]
    final_layer = max(args.layers)
    if final_layer == 12:
        max_error = check_final_layer_equivalence(
            model,
            first_image,
            final_layer=12,
            device=device,
        )
        print(f"L12 old/new max abs error: {max_error:.8e}")
        if max_error > 1e-5:
            raise RuntimeError(
                "Layer-12 diagnostic feature does not match "
                "encode_image_with_patches()."
            )

    records = []

    for index in selected:
        image, caption, _, spans = train_dataset[index]

        layer_results = compute_layer_grounding(
            model=model,
            image=image,
            caption=caption,
            spans=spans,
            layers=args.layers,
            temperature=args.temperature,
            topk=args.topk,
            device=device,
        )

        display_image = recover_input_image(
            image,
            model.backbone.preprocess_val,
        )

        output_path = output_dir / f"sample_{index:05d}.png"
        visualize_sample(
            output_path=output_path,
            image=display_image,
            caption=caption,
            layer_results=layer_results,
            layers=args.layers,
            topk=args.topk,
        )

        for layer in args.layers:
            for result in layer_results[layer]:
                records.append({
                    "sample_index": index,
                    "caption": caption,
                    "entity": result["entity"],
                    "span": result["span"],
                    "layer": layer,
                    "entropy": result["entropy"],
                    "topk_mass": result["topk_mass"],
                    "sim_mean": result["sim_mean"],
                    "sim_std": result["sim_std"],
                    "sim_max": result["sim_max"],
                    "sim_min": result["sim_min"],
                    "sim_max_minus_mean": result["sim_max_minus_mean"],
                    "sim_max_minus_min": result["sim_max_minus_min"],
                })

        print(f"[Saved] {output_path}")

    with open(detail_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    write_aggregate_csv(
        records,
        args.layers,
        aggregate_path,
    )

    print()
    print("=" * 72)
    print(f"Images saved : {len(selected)}")
    print(f"Entity rows  : {len(records)}")
    print(f"Details      : {detail_path}")
    print(f"Layer summary: {aggregate_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
