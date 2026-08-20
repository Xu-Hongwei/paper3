import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
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
        description="RSICD 31类 Caption-Support + Rank-Discrepancy Probe（纯离线，不训练）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--category-map", default=None)
    parser.add_argument("--class-dir", default=None)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--name-template",
        default="{}",
        help='类名模板，默认只编码类名；例如 "a remote sensing image of {}"',
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/probes/caption_category_support",
    )
    parser.add_argument("--html-limit", type=int, default=200)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def normalize_filename(name):
    return Path(str(name).replace("\\", "/")).name.lower()


def load_category_map(category_map=None, class_dir=None):
    if bool(category_map) == bool(class_dir):
        raise ValueError("请且仅请提供 --category-map 或 --class-dir 其中一个。")

    if category_map:
        with open(category_map, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "mapping" in raw and isinstance(raw["mapping"], dict):
            raw = raw["mapping"]
        if not isinstance(raw, dict):
            raise ValueError("category map 必须是 filename -> category 的 JSON 字典。")
        mapping = {
            normalize_filename(filename): str(category).strip().lower()
            for filename, category in raw.items()
        }
    else:
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
                        raise ValueError(f"重复类别映射: {key}: {old} vs {category}")
                    mapping[key] = category

    categories = sorted(set(mapping.values()))
    print(f"Category images : {len(mapping)}")
    print(f"Category groups : {len(categories)}")
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
        batch_features = model.backbone.encode_image(images, normalize=True).cpu()
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
        outputs.append(model.backbone.encode_text(batch, normalize=True).cpu())
    return torch.cat(outputs, dim=0)


def rank_1based(scores, target_index):
    target = scores[target_index]
    return int((scores > target).sum().item()) + 1


def percentile_from_rank(rank, total):
    if total <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (total - 1)


def summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def analyze(
    dataset,
    image_features,
    text_features,
    image_categories,
    categories,
    class_features,
):
    category_to_idx = {c: i for i, c in enumerate(categories)}
    num_images = len(image_features)
    full_scores = text_features @ image_features.t()

    # Caption -> 31个类别名称，只用于判断 caption 更支持 GT 还是 Pred
    caption_class_scores = text_features @ class_features.t()

    # 每个目标类别名称 -> 全部 test images，用于已有 Rank Discrepancy
    class_image_scores = class_features @ image_features.t()

    rows = []
    for text_id in range(len(dataset.text)):
        gt_image = int(dataset.txt2img[text_id])
        gt_class = image_categories[gt_image]

        full_row = full_scores[text_id]
        pred_image = int(torch.argmax(full_row).item())
        pred_class = image_categories[pred_image]

        # 只研究 Baseline Top1 的跨类别错误
        if pred_image == gt_image or pred_class == gt_class:
            continue

        gt_class_idx = category_to_idx[gt_class]
        pred_class_idx = category_to_idx[pred_class]

        # A. Caption 对 31 类名称的支持
        support_scores = caption_class_scores[text_id]
        support_gt_score = float(support_scores[gt_class_idx].item())
        support_pred_score = float(support_scores[pred_class_idx].item())
        support_gap = support_gt_score - support_pred_score
        support_gt_rank = rank_1based(support_scores, gt_class_idx)
        support_pred_rank = rank_1based(support_scores, pred_class_idx)
        support_rank_gap = support_pred_rank - support_gt_rank  # 正：GT更靠前

        # B. Full-caption -> image 的真实错误强度
        full_gt_score = float(full_row[gt_image].item())
        full_pred_score = float(full_row[pred_image].item())
        full_gt_rank = rank_1based(full_row, gt_image)
        full_pred_rank = rank_1based(full_row, pred_image)  # Top1 error下通常为1

        # C. GT 类名 -> image 的类别身份判断
        gt_class_image_row = class_image_scores[gt_class_idx]
        cat_gt_rank = rank_1based(gt_class_image_row, gt_image)
        cat_wrong_rank = rank_1based(gt_class_image_row, pred_image)
        cat_gt_pct = percentile_from_rank(cat_gt_rank, num_images)
        cat_wrong_pct = percentile_from_rank(cat_wrong_rank, num_images)
        cat_sep = cat_gt_pct - cat_wrong_pct

        # 与之前 Stage 2 定义保持一致
        rank_discrepancy = math.log(
            (1.0 + cat_wrong_rank) / (1.0 + full_pred_rank)
        )

        # 仅作为人工排序分数，不直接作为未来loss公式
        clean_probe_score = max(support_gap, 0.0) * max(rank_discrepancy, 0.0)
        misalign_probe_score = max(-support_gap, 0.0) * max(rank_discrepancy, 0.0)

        rows.append({
            "text_id": text_id,
            "caption": dataset.text[text_id],
            "gt_image_id": gt_image,
            "gt_image": dataset.image[gt_image],
            "gt_class": gt_class,
            "pred_image_id": pred_image,
            "pred_image": dataset.image[pred_image],
            "pred_class": pred_class,

            "full_gt_score": full_gt_score,
            "full_pred_score": full_pred_score,
            "full_wrong_minus_gt_gap": full_pred_score - full_gt_score,
            "full_gt_rank": full_gt_rank,
            "full_pred_rank": full_pred_rank,

            "support_gt_score": support_gt_score,
            "support_pred_score": support_pred_score,
            "support_gap_gt_minus_pred": support_gap,
            "support_gt_rank_31": support_gt_rank,
            "support_pred_rank_31": support_pred_rank,
            "support_rank_gap_pred_minus_gt": support_rank_gap,
            "caption_support_relation": (
                "GT>Pred" if support_gap > 0 else
                "Pred>GT" if support_gap < 0 else
                "Tie"
            ),

            "cat_gt_rank_image": cat_gt_rank,
            "cat_wrong_rank_image": cat_wrong_rank,
            "cat_gt_percentile": cat_gt_pct,
            "cat_wrong_percentile": cat_wrong_pct,
            "cat_percentile_separation": cat_sep,
            "rank_discrepancy": rank_discrepancy,

            "clean_probe_score": clean_probe_score,
            "misalign_probe_score": misalign_probe_score,
        })

    return rows


def build_summary(rows):
    gt_better = [r for r in rows if r["support_gap_gt_minus_pred"] > 0]
    pred_better = [r for r in rows if r["support_gap_gt_minus_pred"] < 0]
    tied = [r for r in rows if r["support_gap_gt_minus_pred"] == 0]

    def group_stats(group):
        return {
            "count": len(group),
            "ratio": 100.0 * len(group) / max(len(rows), 1),
            "support_gap": summary_stats(
                [r["support_gap_gt_minus_pred"] for r in group]
            ),
            "rank_discrepancy": summary_stats(
                [r["rank_discrepancy"] for r in group]
            ),
            "cat_percentile_separation": summary_stats(
                [r["cat_percentile_separation"] for r in group]
            ),
            "full_wrong_minus_gt_gap": summary_stats(
                [r["full_wrong_minus_gt_gap"] for r in group]
            ),
        }

    return {
        "cross_category_top1_errors": len(rows),
        "all": group_stats(rows),
        "caption_support_GT_gt_Pred": group_stats(gt_better),
        "caption_support_Pred_gt_GT": group_stats(pred_better),
        "caption_support_tie": group_stats(tied),
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(image_root, image_ref):
    direct = Path(image_root) / image_ref
    if direct.is_file():
        return direct
    flat = Path(image_root) / Path(str(image_ref).replace("\\", "/")).name
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"找不到图像: {image_ref}; tried={direct}, {flat}")


def file_uri(image_root, image_ref):
    try:
        return resolve_image_path(image_root, image_ref).resolve().as_uri()
    except Exception:
        return ""


def build_html(rows, output_path, image_root, title, sort_key, reverse=True, limit=200):
    chosen = sorted(rows, key=lambda r: r[sort_key], reverse=reverse)[:limit]
    cards = []

    for r in chosen:
        gt_uri = file_uri(image_root, r["gt_image"])
        pred_uri = file_uri(image_root, r["pred_image"])
        cards.append(f"""
        <div class="card">
          <div class="caption"><b>Caption:</b> {html.escape(str(r["caption"]))}</div>
          <div class="imgs">
            <div><img src="{gt_uri}"><p><b>GT</b><br>{html.escape(r["gt_class"])}<br>{html.escape(str(r["gt_image"]))}</p></div>
            <div><img src="{pred_uri}"><p><b>Wrong Top1</b><br>{html.escape(r["pred_class"])}<br>{html.escape(str(r["pred_image"]))}</p></div>
          </div>
          <table>
            <tr><th>Signal</th><th>GT</th><th>Wrong/Pred</th><th>Gap / Sep</th></tr>
            <tr><td>Full image retrieval</td><td>rank {r["full_gt_rank"]}</td><td>rank {r["full_pred_rank"]}</td><td>wrong-GT {r["full_wrong_minus_gt_gap"]:+.4f}</td></tr>
            <tr><td>Caption -> 31 class names</td><td>rank {r["support_gt_rank_31"]}<br>score {r["support_gt_score"]:.4f}</td><td>rank {r["support_pred_rank_31"]}<br>score {r["support_pred_score"]:.4f}</td><td>GT-Pred {r["support_gap_gt_minus_pred"]:+.4f}</td></tr>
            <tr><td>GT class name -> images</td><td>rank {r["cat_gt_rank_image"]}<br>pct {r["cat_gt_percentile"]:.3f}</td><td>rank {r["cat_wrong_rank_image"]}<br>pct {r["cat_wrong_percentile"]:.3f}</td><td>Sep {r["cat_percentile_separation"]:+.3f}</td></tr>
          </table>
          <p><b>Rank discrepancy:</b> {r["rank_discrepancy"]:+.3f}
             &nbsp; | &nbsp; <b>Caption support:</b> {r["caption_support_relation"]}
             &nbsp; | &nbsp; <b>{html.escape(sort_key)}:</b> {r[sort_key]:+.4f}</p>
        </div>
        """)

    output_path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
        <style>
        body{{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5}}
        .card{{background:white;padding:16px;margin:0 0 20px;border-radius:8px}}
        .caption{{font-size:16px;margin-bottom:12px}}
        .imgs{{display:flex;gap:24px}} .imgs div{{width:280px}}
        img{{width:260px;height:260px;object-fit:cover;border:1px solid #ccc}}
        table{{border-collapse:collapse;margin-top:12px}}
        th,td{{border:1px solid #ccc;padding:7px 10px;text-align:right}}
        th:first-child,td:first-child{{text-align:left}}
        </style></head><body>
        <h1>{html.escape(title)}</h1>
        """ + "\n".join(cards) + "</body></html>",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_map, categories = load_category_map(args.category_map, args.class_dir)

    print("\nBuilding Clean CLIP...")
    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device          : {device}")

    print(f"\nBuilding {args.split} dataset...")
    dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )
    loader = create_loader(
        dataset,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    print(f"Extracting {args.split.upper()} image features...")
    image_features = encode_images(model, loader, device)

    print(f"Extracting {args.split.upper()} caption features...")
    text_features = encode_texts(
        model,
        dataset.text,
        device,
        args.text_batch_size,
    )

    image_categories = [get_category(ref, category_map) for ref in dataset.image]

    class_texts = [
        args.name_template.format(phrase_for_category(category))
        for category in categories
    ]
    print("Encoding 31 category names...")
    class_features = encode_texts(
        model,
        class_texts,
        device,
        args.text_batch_size,
    )

    rows = analyze(
        dataset=dataset,
        image_features=image_features,
        text_features=text_features,
        image_categories=image_categories,
        categories=categories,
        class_features=class_features,
    )
    summary = build_summary(rows)

    print("\n" + "=" * 100)
    print("CAPTION CATEGORY SUPPORT + RANK DISCREPANCY PROBE")
    print("=" * 100)
    print(f"Cross-category Top1 errors: {summary['cross_category_top1_errors']}")

    for key in [
        "all",
        "caption_support_GT_gt_Pred",
        "caption_support_Pred_gt_GT",
        "caption_support_tie",
    ]:
        s = summary[key]
        print(
            f"{key:>30} | "
            f"N={s['count']:4d} ({s['ratio']:5.2f}%) | "
            f"SupportGap med={s['support_gap']['median']:+.4f} | "
            f"RankDisc med={s['rank_discrepancy']['median']:+.3f} | "
            f"CatSep med={s['cat_percentile_separation']['median']:+.3f}"
        )

    csv_path = output_dir / f"{args.split}_caption_category_support.csv"
    json_path = output_dir / f"{args.split}_caption_category_support_summary.json"
    clean_html = output_dir / f"{args.split}_clean_candidates.html"
    misalign_html = output_dir / f"{args.split}_misalignment_candidates.html"

    save_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "name_template": args.name_template,
                "categories": categories,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    clean_rows = [r for r in rows if r["support_gap_gt_minus_pred"] > 0]
    misalign_rows = [r for r in rows if r["support_gap_gt_minus_pred"] <= 0]

    build_html(
        clean_rows,
        clean_html,
        config["dataset"]["image_root"],
        title="Clean Cross-Category Candidates: Caption supports GT more than Pred",
        sort_key="clean_probe_score",
        reverse=True,
        limit=args.html_limit,
    )
    build_html(
        misalign_rows,
        misalign_html,
        config["dataset"]["image_root"],
        title="Potential Caption-Label Misalignment: Caption supports Pred at least as much as GT",
        sort_key="rank_discrepancy",
        reverse=True,
        limit=args.html_limit,
    )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {clean_html}")
    print(f"  {misalign_html}")
    print("\n优先判断：")
    print("1) Caption 中 GT class 是否通常比 Pred class 更受支持；")
    print("2) Pred>=GT 支持组里是否大量出现 school->playfields、park->pond 这类粒度错位；")
    print("3) GT>Pred 且 RankDiscrepancy 大的样本是否更接近 clean category mistakes；")
    print("4) 暂时不要把 clean_probe_score 直接用于训练，它只用于人工排序探针。")


if __name__ == "__main__":
    main()
