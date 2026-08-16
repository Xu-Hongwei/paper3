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
        description="比较 Original L12 与 Self-Distilled L12 Entity Grounding。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/local_distill_grounding",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    """
    兼容：
        1. Global-only Adapter checkpoint：缺 local_head；
        2. B1 checkpoint：包含 local_head。
    """
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

    allowed_missing = (
        "visual_adapter.",
        "text_adapter.",
        "local_head.",
    )
    invalid_missing = [
        name for name in missing
        if not name.startswith(allowed_missing)
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
        text = tokenizer.decode(
            tokens[start:end].tolist()
        )
        text = (
            text.replace("<|startoftext|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        if text:
            return text

    return f"tokens[{start}:{end}]"


def recover_input_image(image_tensor, preprocess):
    """反归一化模型实际输入图，保证热图和 7×7 Patch 坐标一致。"""
    mean = None
    std = None

    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            mean = torch.as_tensor(
                transform.mean
            ).view(3, 1, 1)
            std = torch.as_tensor(
                transform.std
            ).view(3, 1, 1)
            break

    image = image_tensor.detach().cpu()

    if mean is not None and std is not None:
        image = image * std + mean

    return (
        image.clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def select_samples(dataset, num_samples, indices):
    """优先选择包含典型遥感实体的不同图像。"""
    if indices:
        selected = []

        for index in indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(
                    f"Invalid dataset index: {index}"
                )

            pair_index = dataset.ann[index]["_pair_index"]
            spans = dataset._get_entity_spans(
                pair_index
            )

            if spans.shape[0] == 0:
                raise ValueError(
                    f"Dataset index {index} has no valid Entity span."
                )

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
            spans = dataset._get_entity_spans(
                pair_index
            )
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
            spans = dataset._get_entity_spans(
                pair_index
            )
            if spans.shape[0] == 0:
                continue

            selected.append(index)
            used_images.add(ann["image"])

    return selected


@torch.no_grad()
def extract_entity_features(model, caption, spans, device):
    """
    使用完整 caption 的 contextualized Entity feature。

    注意：
        B1 诊断中不使用 text_adapter，
        保持与 raw CLIP L12 joint space 一致。
    """
    _, token_features = (
        model.backbone.encode_text_with_tokens(
            [caption],
            normalize=False,
        )
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

    return F.normalize(
        entity_features,
        dim=-1,
    )


def summarize_grounding(
    model,
    caption,
    spans,
    patch_features,
    temperature,
    topk,
    device,
):
    """计算 Entity-Patch similarity、softmax heatmap 与统计量。"""
    entity_features = extract_entity_features(
        model,
        caption,
        spans,
        device,
    )

    patch_features = F.normalize(
        patch_features,
        dim=-1,
    )

    similarity = (
        entity_features
        @ patch_features.transpose(0, 1)
    )

    weights = F.softmax(
        similarity / temperature,
        dim=-1,
    )

    num_patches = weights.shape[1]
    grid_size = int(round(num_patches ** 0.5))

    if grid_size * grid_size != num_patches:
        raise RuntimeError(
            f"Patch 数量 {num_patches} 不能恢复为方形网格。"
        )

    results = []

    for entity_index, span in enumerate(
        spans.tolist()
    ):
        weight = weights[entity_index]
        sim = similarity[entity_index]

        entropy = -(
            weight
            * weight.clamp_min(1e-12).log()
        ).sum().item()

        normalized_entropy = (
            entropy
            / np.log(num_patches)
        )

        k = min(
            topk,
            num_patches,
        )
        top_values, top_indices = torch.topk(
            weight,
            k=k,
        )

        results.append({
            "entity": decode_entity(
                model,
                caption,
                span,
            ),
            "span": span,
            "heatmap": (
                weight.view(
                    grid_size,
                    grid_size,
                )
                .cpu()
                .numpy()
            ),
            "entropy": normalized_entropy,
            "topk_mass": top_values.sum().item(),
            "top_indices": top_indices.cpu().tolist(),
            "sim_mean": sim.mean().item(),
            "sim_std": sim.std(
                unbiased=False
            ).item(),
            "sim_max": sim.max().item(),
            "sim_min": sim.min().item(),
            "sim_max_minus_mean": (
                sim.max()
                - sim.mean()
            ).item(),
            "sim_max_minus_min": (
                sim.max()
                - sim.min()
            ).item(),
        })

    return results


@torch.no_grad()
def compute_original_grounding(
    model,
    image,
    caption,
    spans,
    temperature,
    topk,
    device,
):
    """Original：直接使用 raw CLIP L12 Patch。"""
    image = image.unsqueeze(0).to(
        device
    )

    _, patch_features = (
        model.backbone.encode_image_with_patches(
            image,
            normalize=False,
        )
    )

    return summarize_grounding(
        model=model,
        caption=caption,
        spans=spans,
        patch_features=patch_features[0],
        temperature=temperature,
        topk=topk,
        device=device,
    )


@torch.no_grad()
def compute_distilled_grounding(
    model,
    image,
    caption,
    spans,
    temperature,
    topk,
    device,
):
    """B1：raw L12 Patch 经训练后的 Local Head。"""
    image = image.unsqueeze(0).to(
        device
    )

    patch_features = model.encode_local_patches(
        image,
        normalize=False,
    )

    return summarize_grounding(
        model=model,
        caption=caption,
        spans=spans,
        patch_features=patch_features[0],
        temperature=temperature,
        topk=topk,
        device=device,
    )


def draw_topk_boxes(
    ax,
    top_indices,
    grid_size,
    width,
    height,
    topk,
):
    """在模型输入图上标出 Top-K Patch。"""
    patch_w = width / grid_size
    patch_h = height / grid_size

    for patch_index in top_indices[:topk]:
        row = patch_index // grid_size
        col = patch_index % grid_size

        ax.add_patch(
            Rectangle(
                (
                    col * patch_w,
                    row * patch_h,
                ),
                patch_w,
                patch_h,
                fill=False,
                linewidth=2,
            )
        )


def draw_heatmap(
    ax,
    image,
    result,
    topk,
    prefix,
):
    """叠加 Grounding heatmap 与 Top-K Patch。"""
    height, width = image.shape[:2]
    heatmap = result["heatmap"]

    ax.imshow(image)
    ax.imshow(
        heatmap,
        extent=(
            0,
            width,
            height,
            0,
        ),
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
        f"{prefix}: {result['entity']}\n"
        f"H={result['entropy']:.3f}, "
        f"Top{topk}={result['topk_mass']:.3f}, "
        f"σ={result['sim_std']:.3f}"
    )
    ax.axis("off")


def visualize_sample(
    output_path,
    image,
    caption,
    original_results,
    distilled_results,
    topk,
):
    """
    每行一个 Entity：
        Input | Original L12 | Self-Distilled L12
    """
    if len(
        original_results
    ) != len(
        distilled_results
    ):
        raise RuntimeError(
            "Original/Distilled Entity 数量不一致。"
        )

    rows = len(
        original_results
    )

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(
            12,
            4 * rows,
        ),
        squeeze=False,
    )

    for row, (
        original,
        distilled,
    ) in enumerate(
        zip(
            original_results,
            distilled_results,
        )
    ):
        axes[row, 0].imshow(
            image
        )
        axes[row, 0].set_title(
            f"Input\nEntity: "
            f"{original['entity']}"
        )
        axes[row, 0].axis(
            "off"
        )

        draw_heatmap(
            axes[row, 1],
            image,
            original,
            topk,
            "Original L12",
        )

        draw_heatmap(
            axes[row, 2],
            image,
            distilled,
            topk,
            "Self-Distilled",
        )

    fig.suptitle(
        caption,
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be > 0"
        )
    if args.topk <= 0:
        raise ValueError(
            "--topk must be > 0"
        )

    set_seed(
        args.seed
    )

    config = load_config(
        args.config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "local_distill_grounding_summary.jsonl"
    )

    # ==================================================
    # Original L12 baseline
    # ==================================================
    baseline_model = CLIPRetrieval(
        config["model"]
    )

    baseline_checkpoint = load_checkpoint(
        baseline_model,
        args.baseline,
    )

    baseline_model = (
        baseline_model
        .to(device)
        .eval()
    )

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
    print("LOCAL SELF-DISTILLATION GROUNDING DIAGNOSTIC")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Original        : {args.baseline}")
    print(f"Distilled       : {args.checkpoint}")
    print(f"Temperature     : {args.temperature}")
    print(f"Top-K           : {args.topk}")
    print(f"Selected samples: {selected}")
    print(f"Output dir      : {output_dir}")

    samples = []
    original_all = {}

    for index in selected:
        image, caption, _, spans = (
            train_dataset[index]
        )

        results = compute_original_grounding(
            model=baseline_model,
            image=image,
            caption=caption,
            spans=spans,
            temperature=args.temperature,
            topk=args.topk,
            device=device,
        )

        samples.append({
            "index": index,
            "image": image.cpu(),
            "caption": caption,
            "spans": spans.cpu(),
        })

        original_all[index] = results

        print(
            f"[Original] index={index}, "
            f"caption={caption}"
        )

    del baseline_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================================================
    # Self-Distilled L12
    # ==================================================
    distilled_model = CLIPRetrieval(
        config["model"]
    )

    distilled_checkpoint = load_checkpoint(
        distilled_model,
        args.checkpoint,
    )

    distilled_model = (
        distilled_model
        .to(device)
        .eval()
    )

    records = []

    for sample in samples:
        index = sample["index"]
        image = sample["image"]
        caption = sample["caption"]
        spans = sample["spans"]

        distilled_results = compute_distilled_grounding(
            model=distilled_model,
            image=image,
            caption=caption,
            spans=spans,
            temperature=args.temperature,
            topk=args.topk,
            device=device,
        )

        display_image = recover_input_image(
            image,
            distilled_model.backbone.preprocess_val,
        )

        output_path = (
            output_dir
            / f"sample_{index:05d}.png"
        )

        visualize_sample(
            output_path=output_path,
            image=display_image,
            caption=caption,
            original_results=original_all[index],
            distilled_results=distilled_results,
            topk=args.topk,
        )

        for original, distilled in zip(
            original_all[index],
            distilled_results,
        ):
            records.append({
                "sample_index": index,
                "caption": caption,
                "entity": original["entity"],
                "span": original["span"],

                "original_entropy": original["entropy"],
                "distilled_entropy": distilled["entropy"],
                "entropy_delta": (
                    distilled["entropy"]
                    - original["entropy"]
                ),

                "original_topk_mass": original["topk_mass"],
                "distilled_topk_mass": distilled["topk_mass"],
                "topk_mass_delta": (
                    distilled["topk_mass"]
                    - original["topk_mass"]
                ),

                "original_sim_std": original["sim_std"],
                "distilled_sim_std": distilled["sim_std"],
                "sim_std_delta": (
                    distilled["sim_std"]
                    - original["sim_std"]
                ),

                "original_max_minus_mean": (
                    original["sim_max_minus_mean"]
                ),
                "distilled_max_minus_mean": (
                    distilled["sim_max_minus_mean"]
                ),
                "max_minus_mean_delta": (
                    distilled["sim_max_minus_mean"]
                    - original["sim_max_minus_mean"]
                ),

                "original_max_minus_min": (
                    original["sim_max_minus_min"]
                ),
                "distilled_max_minus_min": (
                    distilled["sim_max_minus_min"]
                ),
                "max_minus_min_delta": (
                    distilled["sim_max_minus_min"]
                    - original["sim_max_minus_min"]
                ),
            })

        print(
            f"[Saved] {output_path}"
        )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    mean_entropy_delta = np.mean(
        [
            record["entropy_delta"]
            for record in records
        ]
    )
    mean_topk_delta = np.mean(
        [
            record["topk_mass_delta"]
            for record in records
        ]
    )
    mean_std_delta = np.mean(
        [
            record["sim_std_delta"]
            for record in records
        ]
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Images saved        : {len(samples)}")
    print(f"Entity records      : {len(records)}")
    print(
        f"Mean entropy delta  : "
        f"{mean_entropy_delta:+.6f} "
        f"(negative is more concentrated)"
    )
    print(
        f"Mean Top-K delta    : "
        f"{mean_topk_delta:+.6f} "
        f"(positive is more concentrated)"
    )
    print(
        f"Mean sim std delta  : "
        f"{mean_std_delta:+.6f}"
    )
    print(f"Summary             : {summary_path}")
    print(
        f"Original epoch      : "
        f"{baseline_checkpoint.get('epoch')}"
    )
    print(
        f"Distilled epoch     : "
        f"{distilled_checkpoint.get('epoch')}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
