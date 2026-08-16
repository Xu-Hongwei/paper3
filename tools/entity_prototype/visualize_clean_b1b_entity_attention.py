import argparse
import copy
import csv
import json
import sys
from pathlib import Path

# 允许从 tools/entity_prototype/ 直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        description="比较 Clean CLIP Baseline 与 B1b 的 Entity-Patch Attention。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--checkpoint-b1",
        type=str,
        default=None,
        help="可选：同时比较 best_b1.pth。",
    )
    parser.add_argument(
        "--entity-index",
        type=str,
        default="E:/paper3/data/rsicd/entity_prototype/rsicd_entity_span_index.pt",
        help="Entity span index。训练 B1b 时可禁用，但可视化时需要重新加载。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/clean_b1b_attention",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.02,
        help="Entity-Patch similarity softmax temperature。",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_clean_checkpoint(model, checkpoint_path):
    """
    兼容：
        1. 纯 RSICD CLIP baseline；
        2. Clean B1b checkpoint。

    baseline 缺少 local_teacher_visual.* 属于正常情况。
    """
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
        if not name.startswith("local_teacher_visual.")
    ]

    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint 与当前 Clean CLIP 模型不兼容。\n"
            f"missing={invalid_missing}\n"
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
    """反归一化模型实际输入图，保证 Patch 网格和显示图严格对应。"""
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
    """优先选择具有典型遥感实体且带有效 Entity span 的不同图像。"""
    if indices:
        selected = []

        for index in indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(
                    f"Invalid dataset index: {index}"
                )

            sample = dataset[index]
            spans = sample[3]

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
def compute_entity_attention(
    model,
    image,
    caption,
    spans,
    device,
    temperature,
):
    """
    计算 Clean CLIP Entity-Patch Attention。

    文本：
        contextualized Entity span feature

    图像：
        当前 Vision Encoder 最后一层 Patch feature

    不经过：
        Adapter
        Local Head
        旧 Entity Grounding module
    """
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")

    image = image.unsqueeze(0).to(device)

    _, patch_features = (
        model.backbone.encode_image_with_patches(
            image,
            normalize=False,
        )
    )

    _, token_features = (
        model.backbone.encode_text_with_tokens(
            [caption],
            normalize=False,
        )
    )

    spans_device = spans.to(
        device=device,
        non_blocking=True,
    )

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
        patch_features,
        dim=-1,
    )
    entity_features = F.normalize(
        entity_features,
        dim=-1,
    )

    similarity = (
        entity_features
        @ patch_features[0].transpose(0, 1)
    )

    attention = F.softmax(
        similarity / temperature,
        dim=-1,
    )

    num_patches = attention.shape[1]
    grid_size = int(
        round(num_patches ** 0.5)
    )

    if grid_size * grid_size != num_patches:
        raise RuntimeError(
            f"Patch 数量 {num_patches} 不能恢复为方形网格。"
        )

    results = []

    for entity_index, span in enumerate(
        spans.tolist()
    ):
        sim = similarity[entity_index]
        weight = attention[entity_index]

        entropy = -(
            weight
            * weight.clamp_min(1e-12).log()
        ).sum().item()

        normalized_entropy = (
            entropy / np.log(num_patches)
        )

        k = min(
            num_patches,
            5,
        )

        top_values, top_indices = torch.topk(
            weight,
            k=k,
        )

        sim_mean = sim.mean().item()
        sim_std = sim.std(
            unbiased=False
        ).item()
        sim_max = sim.max().item()
        sim_min = sim.min().item()

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
            "similarity_map": (
                sim.view(
                    grid_size,
                    grid_size,
                )
                .cpu()
                .numpy()
            ),
            "entropy": normalized_entropy,
            "top5_mass": top_values.sum().item(),
            "sim_std": sim_std,
            "max_mean": sim_max - sim_mean,
            "max_min": sim_max - sim_min,
            "sim_mean": sim_mean,
            "sim_max": sim_max,
            "sim_min": sim_min,
            "top_indices": (
                top_indices.cpu().tolist()
            ),
        })

    return results


def draw_topk_boxes(
    ax,
    top_indices,
    grid_size,
    image_size,
    topk,
):
    """在输入图上标出 Top-K Patch。"""
    patch_size = image_size / grid_size

    for patch_index in top_indices[:topk]:
        row = patch_index // grid_size
        col = patch_index % grid_size

        ax.add_patch(
            Rectangle(
                (
                    col * patch_size,
                    row * patch_size,
                ),
                patch_size,
                patch_size,
                fill=False,
                linewidth=2,
            )
        )


def draw_attention(
    ax,
    image,
    result,
    topk,
    prefix,
    vmax,
):
    """叠加 Attention heatmap 与 Top-K Patch。"""
    height, width = image.shape[:2]
    heatmap = result["heatmap"]

    ax.imshow(image)
    ax.imshow(
        heatmap,
        extent=(0, width, height, 0),
        alpha=0.50,
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
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
        f"H={result['entropy']:.3f} | "
        f"Top5={result['top5_mass']:.3f}\n"
        f"Std={result['sim_std']:.3f} | "
        f"Max-Mean={result['max_mean']:.3f}"
    )
    ax.axis("off")


def visualize_sample(
    output_path,
    image,
    caption,
    result_groups,
    topk,
):
    """
    每行一个 Entity：
        Input | Baseline | B1b-best-val | 可选 B1b-best-b1
    """
    labels = list(result_groups.keys())
    result_lists = list(result_groups.values())

    entity_count = len(
        result_lists[0]
    )

    for results in result_lists[1:]:
        if len(results) != entity_count:
            raise RuntimeError(
                "不同模型的 Entity 数量不一致。"
            )

    cols = 1 + len(labels)

    fig, axes = plt.subplots(
        entity_count,
        cols,
        figsize=(
            4 * cols,
            4 * entity_count,
        ),
        squeeze=False,
    )

    for row in range(entity_count):
        baseline_result = result_lists[0][row]

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Input\nEntity: {baseline_result['entity']}"
        )
        axes[row, 0].axis("off")

        # 同一 Entity 的不同模型共用 heatmap 色阶，避免视觉误导。
        vmax = max(
            results[row]["heatmap"].max()
            for results in result_lists
        )
        vmax = max(
            float(vmax),
            1e-8,
        )

        for col, (
            label,
            results,
        ) in enumerate(
            zip(
                labels,
                result_lists,
            ),
            start=1,
        ):
            draw_attention(
                axes[row, col],
                image,
                results[row],
                topk,
                label,
                vmax,
            )

    fig.suptitle(
        caption,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def aggregate_records(records, model_name):
    """汇总某个模型相对 Baseline 的平均统计。"""
    fields = [
        "entropy",
        "top5_mass",
        "sim_std",
        "max_mean",
        "max_min",
    ]

    subset = [
        record
        for record in records
        if record["model"] == model_name
    ]

    if not subset:
        return None

    summary = {
        "model": model_name,
        "num_entities": len(subset),
    }

    for field in fields:
        base_key = f"baseline_{field}"
        model_key = f"model_{field}"
        delta_key = f"{field}_delta"

        summary[f"baseline_{field}"] = float(
            np.mean([
                record[base_key]
                for record in subset
            ])
        )
        summary[f"{model_name}_{field}"] = float(
            np.mean([
                record[model_key]
                for record in subset
            ])
        )
        summary[delta_key] = float(
            np.mean([
                record[delta_key]
                for record in subset
            ])
        )

    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

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

    jsonl_path = (
        output_dir
        / "attention_summary.jsonl"
    )
    csv_path = (
        output_dir
        / "attention_summary.csv"
    )
    aggregate_path = (
        output_dir
        / "attention_aggregate.json"
    )

    # B1b 训练配置为了干净关闭了 Entity span；
    # 可视化阶段重新加载离线 span index。
    dataset_config = copy.deepcopy(
        config["dataset"]
    )
    dataset_config[
        "entity_index_file"
    ] = args.entity_index

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------
    baseline_model = CLIPRetrieval(
        config["model"]
    )
    baseline_checkpoint = load_clean_checkpoint(
        baseline_model,
        args.baseline,
    )
    baseline_model = (
        baseline_model.to(device).eval()
    )

    train_dataset, _ = create_dataset(
        dataset_config,
        evaluate=False,
        train_transform=(
            baseline_model.backbone.preprocess_val
        ),
        eval_transform=(
            baseline_model.backbone.preprocess_val
        ),
    )

    selected = select_samples(
        train_dataset,
        args.num_samples,
        args.indices,
    )

    print("=" * 78)
    print("CLEAN B1b ENTITY-PATCH ATTENTION VISUALIZATION")
    print("=" * 78)
    print(f"Device          : {device}")
    print(f"Baseline        : {args.baseline}")
    print(f"B1b best-val    : {args.checkpoint}")
    if args.checkpoint_b1:
        print(f"B1b best-b1     : {args.checkpoint_b1}")
    print(f"Temperature     : {args.temperature}")
    print(f"Selected samples: {selected}")
    print(f"Output dir      : {output_dir}")

    samples = []
    baseline_all = {}

    for index in selected:
        image, caption, _, spans = (
            train_dataset[index]
        )

        baseline_results = (
            compute_entity_attention(
                baseline_model,
                image,
                caption,
                spans,
                device,
                args.temperature,
            )
        )

        samples.append({
            "index": index,
            "image": image.cpu(),
            "caption": caption,
            "spans": spans.cpu(),
        })

        baseline_all[index] = (
            baseline_results
        )

        print(
            f"[Baseline] index={index}, "
            f"entities={len(baseline_results)}, "
            f"caption={caption}"
        )

    del baseline_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # B1b best-val
    # --------------------------------------------------
    val_model = CLIPRetrieval(
        config["model"]
    )
    val_checkpoint = load_clean_checkpoint(
        val_model,
        args.checkpoint,
    )
    val_model = (
        val_model.to(device).eval()
    )

    val_all = {}

    for sample in samples:
        val_all[sample["index"]] = (
            compute_entity_attention(
                val_model,
                sample["image"],
                sample["caption"],
                sample["spans"],
                device,
                args.temperature,
            )
        )

    del val_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # 可选 B1b best-b1
    # --------------------------------------------------
    b1_all = None
    b1_checkpoint = None

    if args.checkpoint_b1:
        b1_model = CLIPRetrieval(
            config["model"]
        )
        b1_checkpoint = load_clean_checkpoint(
            b1_model,
            args.checkpoint_b1,
        )
        b1_model = (
            b1_model.to(device).eval()
        )

        b1_all = {}

        for sample in samples:
            b1_all[sample["index"]] = (
                compute_entity_attention(
                    b1_model,
                    sample["image"],
                    sample["caption"],
                    sample["spans"],
                    device,
                    args.temperature,
                )
            )

        del b1_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 可视化 + 统计
    # --------------------------------------------------
    records = []

    for sample in samples:
        index = sample["index"]
        image = sample["image"]
        caption = sample["caption"]

        display_image = recover_input_image(
            image,
            train_dataset.transform,
        )

        groups = {
            "Baseline": baseline_all[index],
            "B1b-Val": val_all[index],
        }

        if b1_all is not None:
            groups[
                "B1b-B1"
            ] = b1_all[index]

        output_path = (
            output_dir
            / f"sample_{index:05d}.png"
        )

        visualize_sample(
            output_path,
            display_image,
            caption,
            groups,
            args.topk,
        )

        comparisons = [
            (
                "B1b-Val",
                val_all[index],
            )
        ]

        if b1_all is not None:
            comparisons.append(
                (
                    "B1b-B1",
                    b1_all[index],
                )
            )

        for model_name, trained_results in comparisons:
            for baseline, trained in zip(
                baseline_all[index],
                trained_results,
            ):
                record = {
                    "sample_index": index,
                    "caption": caption,
                    "entity": baseline["entity"],
                    "span": baseline["span"],
                    "model": model_name,
                }

                for field in [
                    "entropy",
                    "top5_mass",
                    "sim_std",
                    "max_mean",
                    "max_min",
                ]:
                    record[
                        f"baseline_{field}"
                    ] = baseline[field]
                    record[
                        f"model_{field}"
                    ] = trained[field]
                    record[
                        f"{field}_delta"
                    ] = (
                        trained[field]
                        - baseline[field]
                    )

                records.append(record)

        print(f"[Saved] {output_path}")

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    csv_fields = list(
        records[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )
        writer.writeheader()
        writer.writerows(records)

    aggregate = {
        "baseline_checkpoint": args.baseline,
        "best_val_checkpoint": args.checkpoint,
        "best_b1_checkpoint": args.checkpoint_b1,
        "temperature": args.temperature,
        "selected_samples": selected,
        "comparisons": [],
    }

    model_names = ["B1b-Val"]

    if b1_all is not None:
        model_names.append(
            "B1b-B1"
        )

    for model_name in model_names:
        summary = aggregate_records(
            records,
            model_name,
        )
        aggregate[
            "comparisons"
        ].append(summary)

    with open(
        aggregate_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aggregate,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("AGGREGATE RESULT")
    print("=" * 78)

    for summary in aggregate["comparisons"]:
        print(
            f"\n[{summary['model']}] "
            f"entities={summary['num_entities']}"
        )
        print(
            f"Entropy : "
            f"{summary['baseline_entropy']:.4f}"
            f" -> "
            f"{summary[summary['model'] + '_entropy']:.4f}"
            f" | Δ={summary['entropy_delta']:+.4f}"
        )
        print(
            f"Top5    : "
            f"{summary['baseline_top5_mass']:.4f}"
            f" -> "
            f"{summary[summary['model'] + '_top5_mass']:.4f}"
            f" | Δ={summary['top5_mass_delta']:+.4f}"
        )
        print(
            f"SimStd  : "
            f"{summary['baseline_sim_std']:.4f}"
            f" -> "
            f"{summary[summary['model'] + '_sim_std']:.4f}"
            f" | Δ={summary['sim_std_delta']:+.4f}"
        )
        print(
            f"MaxMean : "
            f"{summary['baseline_max_mean']:.4f}"
            f" -> "
            f"{summary[summary['model'] + '_max_mean']:.4f}"
            f" | Δ={summary['max_mean_delta']:+.4f}"
        )
        print(
            f"MaxMin  : "
            f"{summary['baseline_max_min']:.4f}"
            f" -> "
            f"{summary[summary['model'] + '_max_min']:.4f}"
            f" | Δ={summary['max_min_delta']:+.4f}"
        )

    print("\n" + "=" * 78)
    print(f"Images saved     : {len(samples)}")
    print(f"JSONL summary    : {jsonl_path}")
    print(f"CSV summary      : {csv_path}")
    print(f"Aggregate summary: {aggregate_path}")
    print(
        f"Baseline epoch   : "
        f"{baseline_checkpoint.get('epoch')}"
    )
    print(
        f"Best-Val epoch   : "
        f"{val_checkpoint.get('epoch')}"
    )
    if b1_checkpoint is not None:
        print(
            f"Best-B1 epoch    : "
            f"{b1_checkpoint.get('epoch')}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
