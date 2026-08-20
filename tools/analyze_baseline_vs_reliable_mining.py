import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset, create_loader
from models import CLIPRetrieval


CLASS_PHRASES = {
    "airport": "airport",
    "bareland": "bare land",
    "baseballfield": "baseball field",
    "beach": "beach",
    "bridge": "bridge",
    "center": "center",
    "church": "church",
    "commercial": "commercial area",
    "denseresidential": "dense residential area",
    "desert": "desert",
    "farmland": "farmland",
    "forest": "forest",
    "industrial": "industrial area",
    "meadow": "meadow",
    "mediumresidential": "medium residential area",
    "mountain": "mountain",
    "park": "park",
    "parking": "parking lot",
    "playground": "playground",
    "playfields": "play fields",
    "pond": "pond",
    "port": "port",
    "railwaystation": "railway station",
    "resort": "resort",
    "river": "river",
    "school": "school",
    "sparseresidential": "sparse residential area",
    "square": "square",
    "stadium": "stadium",
    "storagetanks": "storage tanks",
    "viaduct": "viaduct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline vs Reliable Mining 样本级对比："
            "检查提升是否主要来自目标跨类别错误，尤其 A 区。"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--reliable-checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--class-dir", required=True)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--name-template", default="{}")
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/baseline_vs_reliable_mining",
    )
    parser.add_argument("--html-limit", type=int, default=200)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def normalize_filename(name):
    return Path(str(name).replace("\\", "/")).name.lower()


def load_category_map(class_dir):
    class_dir = Path(class_dir)
    if not class_dir.is_dir():
        raise FileNotFoundError(f"class-dir 不存在: {class_dir}")

    mapping = {}
    txt_files = sorted(class_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"目录中没有 txt 文件: {class_dir}")

    for txt_file in txt_files:
        category = txt_file.stem.strip().lower()
        with txt_file.open("r", encoding="utf-8-sig") as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue

                key = normalize_filename(filename)
                old = mapping.get(key)
                if old is not None and old != category:
                    raise ValueError(
                        f"重复类别映射: {key}: {old} vs {category}"
                    )
                mapping[key] = category

    categories = sorted(set(mapping.values()))
    if len(categories) != 31:
        raise RuntimeError(
            f"期望 released 31 groups，当前为 {len(categories)}。"
        )

    return mapping, categories


def get_category(image_ref, mapping):
    key = normalize_filename(image_ref)
    if key not in mapping:
        raise KeyError(f"类别映射缺失: {image_ref} -> {key}")
    return mapping[key]


def phrase_for_category(category):
    return CLASS_PHRASES.get(category, category)


@torch.no_grad()
def encode_images(model, loader, device):
    features = [None] * len(loader.dataset)

    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        batch_features = model.backbone.encode_image(
            images,
            normalize=True,
        ).cpu()

        for local_idx, dataset_idx in enumerate(indices.tolist()):
            features[dataset_idx] = batch_features[local_idx]

    if any(item is None for item in features):
        raise RuntimeError("存在未提取的 image feature。")

    return torch.stack(features, dim=0)


@torch.no_grad()
def encode_texts(model, texts, device, batch_size):
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        outputs.append(
            model.backbone.encode_text(
                batch,
                normalize=True,
            ).cpu()
        )

    return torch.cat(outputs, dim=0)


def exact_rank(scores, gt_index):
    """1-based exact GT rank。"""
    gt_score = scores[gt_index]
    return int((scores > gt_score).sum().item()) + 1


def assign_region(g_cat, g_sup):
    if g_cat > 0 and g_sup > 0:
        return "A_clean_candidate"
    if g_cat > 0 and g_sup <= 0:
        return "B_caption_label_misalignment"
    if g_cat <= 0 and g_sup > 0:
        return "C_category_neighbor_or_unstable"
    return "D_ambiguous"


def error_type(pred_image, gt_image, pred_class, gt_class):
    if pred_image == gt_image:
        return "exact_correct"
    if pred_class == gt_class:
        return "same_category_error"
    return "cross_category_error"


def transition_name(base_type, rel_type):
    if base_type == "exact_correct" and rel_type == "exact_correct":
        return "both_correct"
    if base_type != "exact_correct" and rel_type == "exact_correct":
        return "corrected_to_exact"
    if base_type == "exact_correct" and rel_type != "exact_correct":
        return "regressed_from_exact"
    if base_type == "cross_category_error" and rel_type == "same_category_error":
        return "cross_to_same_category"
    if base_type == "same_category_error" and rel_type == "cross_category_error":
        return "same_to_cross_category"
    if base_type == "cross_category_error" and rel_type == "cross_category_error":
        return "cross_still_cross"
    if base_type == "same_category_error" and rel_type == "same_category_error":
        return "same_still_same"
    return f"{base_type}_to_{rel_type}"


def build_rows(
    dataset,
    baseline_image_features,
    baseline_text_features,
    reliable_image_features,
    reliable_text_features,
    image_categories,
    categories,
    baseline_class_features,
):
    category_to_idx = {
        category: index
        for index, category in enumerate(categories)
    }

    base_scores = (
        baseline_text_features
        @ baseline_image_features.t()
    )
    rel_scores = (
        reliable_text_features
        @ reliable_image_features.t()
    )

    # A/B/C/D 固定使用 baseline teacher，避免用训练后的 student 重新定义 reliability。
    caption_class_scores = (
        baseline_text_features
        @ baseline_class_features.t()
    )
    class_image_scores = (
        baseline_class_features
        @ baseline_image_features.t()
    )

    rows = []

    for text_id in range(len(dataset.text)):
        gt_image = int(dataset.txt2img[text_id])
        gt_class = image_categories[gt_image]

        base_row = base_scores[text_id]
        rel_row = rel_scores[text_id]

        base_pred = int(torch.argmax(base_row).item())
        rel_pred = int(torch.argmax(rel_row).item())

        base_pred_class = image_categories[base_pred]
        rel_pred_class = image_categories[rel_pred]

        base_type = error_type(
            base_pred,
            gt_image,
            base_pred_class,
            gt_class,
        )
        rel_type = error_type(
            rel_pred,
            gt_image,
            rel_pred_class,
            gt_class,
        )

        base_rank = exact_rank(base_row, gt_image)
        rel_rank = exact_rank(rel_row, gt_image)

        gt_class_idx = category_to_idx[gt_class]

        # 只对 baseline cross-category Top1 error 定义原始 A/B/C/D。
        region = ""
        g_full = float("nan")
        g_cat = float("nan")
        g_sup = float("nan")

        if base_type == "cross_category_error":
            pred_class_idx = category_to_idx[base_pred_class]

            base_gt_score = float(base_row[gt_image].item())
            base_wrong_score = float(base_row[base_pred].item())
            g_full = base_gt_score - base_wrong_score

            gt_class_image_scores = class_image_scores[
                gt_class_idx
            ]
            g_cat = float(
                gt_class_image_scores[gt_image].item()
                - gt_class_image_scores[base_pred].item()
            )

            support_row = caption_class_scores[text_id]
            g_sup = float(
                support_row[gt_class_idx].item()
                - support_row[pred_class_idx].item()
            )

            region = assign_region(g_cat, g_sup)

        rows.append({
            "text_id": text_id,
            "caption": dataset.text[text_id],
            "gt_image_id": gt_image,
            "gt_image": dataset.image[gt_image],
            "gt_class": gt_class,

            "baseline_pred_image_id": base_pred,
            "baseline_pred_image": dataset.image[base_pred],
            "baseline_pred_class": base_pred_class,
            "baseline_type": base_type,
            "baseline_gt_rank": base_rank,
            "baseline_gt_score": float(base_row[gt_image].item()),
            "baseline_top1_score": float(base_row[base_pred].item()),

            "reliable_pred_image_id": rel_pred,
            "reliable_pred_image": dataset.image[rel_pred],
            "reliable_pred_class": rel_pred_class,
            "reliable_type": rel_type,
            "reliable_gt_rank": rel_rank,
            "reliable_gt_score": float(rel_row[gt_image].item()),
            "reliable_top1_score": float(rel_row[rel_pred].item()),

            # 正数 = rank 变好。
            "rank_improvement": base_rank - rel_rank,
            "transition": transition_name(base_type, rel_type),

            "baseline_cross_region": region,
            "G_full": g_full,
            "G_cat": g_cat,
            "G_sup": g_sup,
        })

    return rows


def safe_mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return float(np.mean(values))


def safe_median(values):
    values = list(values)
    if not values:
        return float("nan")
    return float(np.median(values))


def summarize(rows):
    n = len(rows)

    base_types = Counter(row["baseline_type"] for row in rows)
    rel_types = Counter(row["reliable_type"] for row in rows)
    transitions = Counter(row["transition"] for row in rows)

    rank_better = sum(row["rank_improvement"] > 0 for row in rows)
    rank_same = sum(row["rank_improvement"] == 0 for row in rows)
    rank_worse = sum(row["rank_improvement"] < 0 for row in rows)

    base_cross_rows = [
        row
        for row in rows
        if row["baseline_type"] == "cross_category_error"
    ]

    region_summary = {}
    for region in [
        "A_clean_candidate",
        "B_caption_label_misalignment",
        "C_category_neighbor_or_unstable",
        "D_ambiguous",
    ]:
        group = [
            row
            for row in base_cross_rows
            if row["baseline_cross_region"] == region
        ]

        corrected_exact = sum(
            row["reliable_type"] == "exact_correct"
            for row in group
        )
        moved_same = sum(
            row["reliable_type"] == "same_category_error"
            for row in group
        )
        still_cross = sum(
            row["reliable_type"] == "cross_category_error"
            for row in group
        )
        rank_better_region = sum(
            row["rank_improvement"] > 0
            for row in group
        )

        region_summary[region] = {
            "baseline_cross_errors": len(group),
            "corrected_to_exact": corrected_exact,
            "corrected_to_exact_rate": (
                corrected_exact / max(len(group), 1)
            ),
            "moved_to_same_category": moved_same,
            "moved_to_same_category_rate": (
                moved_same / max(len(group), 1)
            ),
            "still_cross_category": still_cross,
            "still_cross_category_rate": (
                still_cross / max(len(group), 1)
            ),
            "rank_better": rank_better_region,
            "rank_better_rate": (
                rank_better_region / max(len(group), 1)
            ),
            "mean_rank_improvement": safe_mean(
                row["rank_improvement"]
                for row in group
            ),
            "median_rank_improvement": safe_median(
                row["rank_improvement"]
                for row in group
            ),
        }

    base_cross = base_types["cross_category_error"]
    rel_cross = rel_types["cross_category_error"]

    new_cross = sum(
        row["baseline_type"] != "cross_category_error"
        and row["reliable_type"] == "cross_category_error"
        for row in rows
    )

    resolved_cross_to_exact = transitions["corrected_to_exact"]
    # corrected_to_exact 也可能来自 baseline same-category error，
    # 因此另算 baseline cross -> exact。
    cross_to_exact = sum(
        row["baseline_type"] == "cross_category_error"
        and row["reliable_type"] == "exact_correct"
        for row in rows
    )
    cross_to_same = transitions["cross_to_same_category"]

    return {
        "num_captions": n,
        "baseline": {
            "exact_correct": base_types["exact_correct"],
            "exact_top1_accuracy": (
                base_types["exact_correct"] / max(n, 1)
            ),
            "same_category_errors": base_types[
                "same_category_error"
            ],
            "cross_category_errors": base_cross,
        },
        "reliable": {
            "exact_correct": rel_types["exact_correct"],
            "exact_top1_accuracy": (
                rel_types["exact_correct"] / max(n, 1)
            ),
            "same_category_errors": rel_types[
                "same_category_error"
            ],
            "cross_category_errors": rel_cross,
        },
        "delta": {
            "exact_correct": (
                rel_types["exact_correct"]
                - base_types["exact_correct"]
            ),
            "cross_category_errors": rel_cross - base_cross,
            "cross_category_error_reduction": (
                base_cross - rel_cross
            ),
            "cross_category_error_reduction_rate": (
                (base_cross - rel_cross)
                / max(base_cross, 1)
            ),
        },
        "transitions": dict(transitions),
        "target_cross_error_flow": {
            "baseline_cross_errors": base_cross,
            "cross_to_exact": cross_to_exact,
            "cross_to_same_category": cross_to_same,
            "cross_still_cross": transitions[
                "cross_still_cross"
            ],
            "new_cross_errors": new_cross,
        },
        "rank_change_all": {
            "better": rank_better,
            "same": rank_same,
            "worse": rank_worse,
            "better_rate": rank_better / max(n, 1),
            "worse_rate": rank_worse / max(n, 1),
            "mean_improvement": safe_mean(
                row["rank_improvement"]
                for row in rows
            ),
            "median_improvement": safe_median(
                row["rank_improvement"]
                for row in rows
            ),
        },
        "baseline_cross_region_analysis": region_summary,
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(image_root, image_ref):
    direct = Path(image_root) / image_ref
    if direct.is_file():
        return direct

    flat = (
        Path(image_root)
        / Path(str(image_ref).replace("\\", "/")).name
    )
    if flat.is_file():
        return flat

    raise FileNotFoundError(
        f"找不到图像: {image_ref}; tried={direct}, {flat}"
    )


def file_uri(image_root, image_ref):
    try:
        return resolve_image_path(
            image_root,
            image_ref,
        ).resolve().as_uri()
    except Exception:
        return ""


def build_html(
    rows,
    output_path,
    image_root,
    title,
    selector,
    limit,
):
    selected = [
        row for row in rows
        if selector(row)
    ]

    selected.sort(
        key=lambda row: (
            row["rank_improvement"],
            (
                row["G_sup"]
                if np.isfinite(row["G_sup"])
                else -999.0
            ),
        ),
        reverse=True,
    )
    selected = selected[:limit]

    cards = []
    for row in selected:
        gt_uri = file_uri(
            image_root,
            row["gt_image"],
        )
        base_uri = file_uri(
            image_root,
            row["baseline_pred_image"],
        )
        rel_uri = file_uri(
            image_root,
            row["reliable_pred_image"],
        )

        cards.append(
            f"""
            <div class="card">
              <div class="caption">
                <b>Caption:</b> {html.escape(str(row['caption']))}
              </div>

              <div class="imgs">
                <div>
                  <img src="{gt_uri}">
                  <p><b>GT</b><br>
                  {html.escape(row['gt_class'])}<br>
                  {html.escape(str(row['gt_image']))}</p>
                </div>

                <div>
                  <img src="{base_uri}">
                  <p><b>Baseline Top1</b><br>
                  {html.escape(row['baseline_pred_class'])}<br>
                  {html.escape(str(row['baseline_pred_image']))}</p>
                </div>

                <div>
                  <img src="{rel_uri}">
                  <p><b>Reliable Top1</b><br>
                  {html.escape(row['reliable_pred_class'])}<br>
                  {html.escape(str(row['reliable_pred_image']))}</p>
                </div>
              </div>

              <table>
                <tr>
                  <th></th>
                  <th>Baseline</th>
                  <th>Reliable</th>
                </tr>
                <tr>
                  <td>GT rank</td>
                  <td>{row['baseline_gt_rank']}</td>
                  <td>{row['reliable_gt_rank']}</td>
                </tr>
                <tr>
                  <td>Type</td>
                  <td>{html.escape(row['baseline_type'])}</td>
                  <td>{html.escape(row['reliable_type'])}</td>
                </tr>
              </table>

              <p>
                <b>Rank improvement:</b>
                {row['rank_improvement']:+d}
                &nbsp; | &nbsp;
                <b>Transition:</b>
                {html.escape(row['transition'])}
              </p>

              <p>
                <b>Baseline region:</b>
                {html.escape(row['baseline_cross_region'])}
                &nbsp; | &nbsp;
                G_full={row['G_full']:+.4f},
                G_cat={row['G_cat']:+.4f},
                G_sup={row['G_sup']:+.4f}
              </p>
            </div>
            """
        )

    output_path.write_text(
        f"""<!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>{html.escape(title)}</title>
          <style>
            body {{
              font-family: Arial, sans-serif;
              margin: 24px;
              background: #f5f5f5;
            }}
            .card {{
              background: white;
              padding: 16px;
              margin-bottom: 20px;
              border-radius: 8px;
            }}
            .caption {{
              font-size: 16px;
              margin-bottom: 12px;
            }}
            .imgs {{
              display: flex;
              gap: 22px;
            }}
            .imgs div {{
              width: 260px;
            }}
            img {{
              width: 245px;
              height: 245px;
              object-fit: cover;
              border: 1px solid #ccc;
            }}
            table {{
              border-collapse: collapse;
              margin-top: 12px;
            }}
            th, td {{
              border: 1px solid #ccc;
              padding: 7px 10px;
            }}
          </style>
        </head>
        <body>
          <h1>{html.escape(title)}</h1>
          <p>Samples shown: {len(selected)}</p>
          {''.join(cards)}
        </body>
        </html>
        """,
        encoding="utf-8",
    )


def print_summary(summary):
    base = summary["baseline"]
    rel = summary["reliable"]
    delta = summary["delta"]
    flow = summary["target_cross_error_flow"]

    print("\n" + "=" * 96)
    print("BASELINE vs RELIABLE MINING — SAMPLE-LEVEL T2I ANALYSIS")
    print("=" * 96)

    print("\n[1] Exact Top-1")
    print(
        f"Baseline : {base['exact_correct']}/{summary['num_captions']} "
        f"({base['exact_top1_accuracy']:.2%})"
    )
    print(
        f"Reliable : {rel['exact_correct']}/{summary['num_captions']} "
        f"({rel['exact_top1_accuracy']:.2%})"
    )
    print(
        f"Delta    : {delta['exact_correct']:+d} exact-correct captions"
    )

    print("\n[2] Cross-category Top-1 errors — 核心目标")
    print(
        f"Baseline : {base['cross_category_errors']}"
    )
    print(
        f"Reliable : {rel['cross_category_errors']}"
    )
    print(
        f"Reduction: {delta['cross_category_error_reduction']:+d} "
        f"({delta['cross_category_error_reduction_rate']:+.2%})"
    )
    print(
        f"Cross -> Exact       : {flow['cross_to_exact']}"
    )
    print(
        f"Cross -> Same class  : {flow['cross_to_same_category']}"
    )
    print(
        f"Cross -> Cross       : {flow['cross_still_cross']}"
    )
    print(
        f"New cross errors     : {flow['new_cross_errors']}"
    )

    print("\n[3] Baseline cross-error A/B/C/D")
    for region, values in summary[
        "baseline_cross_region_analysis"
    ].items():
        print(
            f"{region:>34} | "
            f"N={values['baseline_cross_errors']:4d} | "
            f"ExactFix={values['corrected_to_exact']:4d} "
            f"({values['corrected_to_exact_rate']:6.2%}) | "
            f"ToSame={values['moved_to_same_category']:4d} "
            f"({values['moved_to_same_category_rate']:6.2%}) | "
            f"RankBetter={values['rank_better']:4d} "
            f"({values['rank_better_rate']:6.2%}) | "
            f"MeanΔRank={values['mean_rank_improvement']:+.2f}"
        )

    rank = summary["rank_change_all"]
    print("\n[4] All-caption GT rank change")
    print(
        f"Better/Same/Worse : "
        f"{rank['better']}/{rank['same']}/{rank['worse']}"
    )
    print(
        f"Better rate       : {rank['better_rate']:.2%}"
    )
    print(
        f"Worse rate        : {rank['worse_rate']:.2%}"
    )
    print(
        f"Mean rank improve : {rank['mean_improvement']:+.3f}"
    )

    print("\n优先判断：")
    print("1) Cross-category Top1 error 总数是否下降；")
    print("2) A 区 ExactFix / RankBetter 是否高于 B/C/D；")
    print("3) Cross->Same 是否明显增加：即先回到正确类别邻域；")
    print("4) New cross errors 是否明显小于被修正的 cross errors。")
    print("=" * 96)


def main():
    args = parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_map, categories = load_category_map(
        args.class_dir
    )

    print("=" * 96)
    print("BASELINE vs RELIABLE MINING COMPARISON")
    print("=" * 96)
    print(f"Config              : {args.config}")
    print(f"Baseline checkpoint : {args.baseline_checkpoint}")
    print(f"Reliable checkpoint : {args.reliable_checkpoint}")
    print(f"Split               : {args.split}")
    print(f"Device              : {device}")
    print(f"Category groups     : {len(categories)}")

    # --------------------------------------------------
    # Baseline teacher / retrieval model
    # --------------------------------------------------
    print("\nBuilding baseline model...")
    baseline_model = CLIPRetrieval(
        config["model"]
    )
    baseline_checkpoint = load_checkpoint(
        baseline_model,
        args.baseline_checkpoint,
    )
    baseline_model = baseline_model.to(device).eval()

    print(
        f"Baseline epoch: "
        f"{baseline_checkpoint.get('epoch', 'unknown')}"
    )

    print(f"\nBuilding {args.split} dataset...")
    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=baseline_model.backbone.preprocess_val,
    )
    loader = create_loader(
        dataset,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    image_categories = [
        get_category(ref, category_map)
        for ref in dataset.image
    ]

    print("Encoding baseline images...")
    baseline_image_features = encode_images(
        baseline_model,
        loader,
        device,
    )
    print("Encoding baseline captions...")
    baseline_text_features = encode_texts(
        baseline_model,
        dataset.text,
        device,
        args.text_batch_size,
    )

    class_texts = [
        args.name_template.format(
            phrase_for_category(category)
        )
        for category in categories
    ]
    print("Encoding baseline class names...")
    baseline_class_features = encode_texts(
        baseline_model,
        class_texts,
        device,
        args.text_batch_size,
    )

    # --------------------------------------------------
    # Reliable Mining model
    # --------------------------------------------------
    print("\nBuilding Reliable Mining model...")
    reliable_model = CLIPRetrieval(
        config["model"]
    )
    reliable_checkpoint = load_checkpoint(
        reliable_model,
        args.reliable_checkpoint,
    )
    reliable_model = reliable_model.to(device).eval()

    print(
        f"Reliable epoch: "
        f"{reliable_checkpoint.get('epoch', 'unknown')}"
    )

    # 使用同一 dataset 顺序，只替换 transform 对应的模型 preprocess。
    reliable_dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=reliable_model.backbone.preprocess_val,
    )

    if list(reliable_dataset.image) != list(dataset.image):
        raise RuntimeError(
            "Baseline / Reliable dataset.image 顺序不一致。"
        )
    if list(reliable_dataset.text) != list(dataset.text):
        raise RuntimeError(
            "Baseline / Reliable dataset.text 顺序不一致。"
        )
    if list(reliable_dataset.txt2img) != list(dataset.txt2img):
        raise RuntimeError(
            "Baseline / Reliable txt2img 顺序不一致。"
        )

    reliable_loader = create_loader(
        reliable_dataset,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    print("Encoding Reliable images...")
    reliable_image_features = encode_images(
        reliable_model,
        reliable_loader,
        device,
    )
    print("Encoding Reliable captions...")
    reliable_text_features = encode_texts(
        reliable_model,
        reliable_dataset.text,
        device,
        args.text_batch_size,
    )

    # 显存清理，后续分析全部 CPU。
    del baseline_model
    del reliable_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Sample-level comparison
    # --------------------------------------------------
    print("\nComparing every T2I query...")
    rows = build_rows(
        dataset=dataset,
        baseline_image_features=baseline_image_features,
        baseline_text_features=baseline_text_features,
        reliable_image_features=reliable_image_features,
        reliable_text_features=reliable_text_features,
        image_categories=image_categories,
        categories=categories,
        baseline_class_features=baseline_class_features,
    )
    summary = summarize(rows)
    print_summary(summary)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    csv_path = output_dir / (
        f"{args.split}_baseline_vs_reliable_all.csv"
    )
    json_path = output_dir / (
        f"{args.split}_baseline_vs_reliable_summary.json"
    )

    save_csv(csv_path, rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": args.config,
                "baseline_checkpoint": args.baseline_checkpoint,
                "reliable_checkpoint": args.reliable_checkpoint,
                "split": args.split,
                "class_dir": args.class_dir,
                "name_template": args.name_template,
                "categories": categories,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 三组人工核查页面。
    html_specs = [
        (
            "A_corrected_to_exact",
            "A-zone: Baseline Cross Error -> Reliable Exact Correct",
            lambda row: (
                row["baseline_cross_region"]
                == "A_clean_candidate"
                and row["reliable_type"]
                == "exact_correct"
            ),
        ),
        (
            "A_remaining_or_partial",
            "A-zone: Remaining / Partial Improvement",
            lambda row: (
                row["baseline_cross_region"]
                == "A_clean_candidate"
                and row["reliable_type"]
                != "exact_correct"
            ),
        ),
        (
            "new_cross_errors",
            "New Cross-Category Errors Introduced by Reliable Mining",
            lambda row: (
                row["baseline_type"]
                != "cross_category_error"
                and row["reliable_type"]
                == "cross_category_error"
            ),
        ),
        (
            "regressed_from_exact",
            "Baseline Exact Correct -> Reliable Wrong",
            lambda row: (
                row["baseline_type"]
                == "exact_correct"
                and row["reliable_type"]
                != "exact_correct"
            ),
        ),
    ]

    html_paths = []
    for suffix, title, selector in html_specs:
        path = output_dir / (
            f"{args.split}_{suffix}.html"
        )
        build_html(
            rows=rows,
            output_path=path,
            image_root=config["dataset"]["image_root"],
            title=title,
            selector=selector,
            limit=args.html_limit,
        )
        html_paths.append(path)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    for path in html_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
