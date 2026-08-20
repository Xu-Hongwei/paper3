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
    "airport": "airport", "bareland": "bare land", "baseballfield": "baseball field",
    "beach": "beach", "bridge": "bridge", "center": "center", "church": "church",
    "commercial": "commercial area", "denseresidential": "dense residential area",
    "desert": "desert", "farmland": "farmland", "forest": "forest",
    "industrial": "industrial area", "meadow": "meadow",
    "mediumresidential": "medium residential area", "mountain": "mountain",
    "park": "park", "parking": "parking lot", "playground": "playground",
    "playfields": "play fields", "pond": "pond", "port": "port",
    "railwaystation": "railway station", "resort": "resort", "river": "river",
    "school": "school", "sparseresidential": "sparse residential area",
    "square": "square", "stadium": "stadium", "storagetanks": "storage tanks",
    "viaduct": "viaduct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSICD 类间三信号 Pair Probe：G_full / G_cat / G_sup（纯离线，不训练）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--category-map", default=None)
    parser.add_argument("--class-dir", default=None)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--name-template", default="{}")
    parser.add_argument("--output-dir", default="outputs/probes/pair_cross_granularity")
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
        mapping = {normalize_filename(k): str(v).strip().lower() for k, v in raw.items()}
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


def safe_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
    }


def assign_region(g_cat, g_sup):
    # 这里只做诊断分区，不代表最终训练规则
    if g_cat > 0 and g_sup > 0:
        return "A_clean_candidate"
    if g_cat > 0 and g_sup <= 0:
        return "B_caption_label_misalignment"
    if g_cat <= 0 and g_sup > 0:
        return "C_category_neighbor_or_unstable"
    return "D_ambiguous"


def analyze(dataset, image_features, text_features, image_categories, categories, class_features):
    category_to_idx = {c: i for i, c in enumerate(categories)}
    full_scores = text_features @ image_features.t()
    caption_class_scores = text_features @ class_features.t()
    class_image_scores = class_features @ image_features.t()

    rows = []
    for text_id in range(len(dataset.text)):
        gt_image = int(dataset.txt2img[text_id])
        gt_class = image_categories[gt_image]
        full_row = full_scores[text_id]
        pred_image = int(torch.argmax(full_row).item())
        pred_class = image_categories[pred_image]

        if pred_image == gt_image or pred_class == gt_class:
            continue

        gt_class_idx = category_to_idx[gt_class]
        pred_class_idx = category_to_idx[pred_class]

        full_gt = float(full_row[gt_image].item())
        full_wrong = float(full_row[pred_image].item())
        g_full = full_gt - full_wrong

        gt_class_image_row = class_image_scores[gt_class_idx]
        cat_gt = float(gt_class_image_row[gt_image].item())
        cat_wrong = float(gt_class_image_row[pred_image].item())
        g_cat = cat_gt - cat_wrong

        support_row = caption_class_scores[text_id]
        sup_gt = float(support_row[gt_class_idx].item())
        sup_pred = float(support_row[pred_class_idx].item())
        g_sup = sup_gt - sup_pred

        d_cg = g_cat - g_full
        region = assign_region(g_cat, g_sup)

        rows.append({
            "text_id": text_id, "caption": dataset.text[text_id],
            "gt_image_id": gt_image, "gt_image": dataset.image[gt_image], "gt_class": gt_class,
            "pred_image_id": pred_image, "pred_image": dataset.image[pred_image], "pred_class": pred_class,
            "full_gt_score": full_gt, "full_wrong_score": full_wrong, "G_full": g_full,
            "cat_gt_score": cat_gt, "cat_wrong_score": cat_wrong, "G_cat": g_cat,
            "support_gt_score": sup_gt, "support_pred_score": sup_pred, "G_sup": g_sup,
            "D_cross_granularity": d_cg, "region": region,
        })
    return rows


def build_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["region"]].append(row)

    summary = {}
    region_names = [
        "all", "A_clean_candidate", "B_caption_label_misalignment",
        "C_category_neighbor_or_unstable", "D_ambiguous",
    ]
    for name in region_names:
        group = groups[name]
        summary[name] = {
            "count": len(group), "ratio": 100.0 * len(group) / max(len(rows), 1),
            "G_full": safe_stats([r["G_full"] for r in group]),
            "G_cat": safe_stats([r["G_cat"] for r in group]),
            "G_sup": safe_stats([r["G_sup"] for r in group]),
            "D_cross_granularity": safe_stats([r["D_cross_granularity"] for r in group]),
        }

    pair_counter = Counter((r["gt_class"], r["pred_class"], r["region"]) for r in rows)
    summary["top_confusion_region_pairs"] = [
        {"gt_class": gt, "pred_class": pred, "region": region, "count": count}
        for (gt, pred, region), count in pair_counter.most_common(50)
    ]
    return summary


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


def build_region_html(rows, output_path, image_root, region, limit):
    selected = [r for r in rows if r["region"] == region]
    if region == "A_clean_candidate":
        selected.sort(key=lambda r: (r["G_sup"], r["G_cat"]), reverse=True)
    elif region == "B_caption_label_misalignment":
        selected.sort(key=lambda r: (-r["G_sup"], r["G_cat"]), reverse=True)
    elif region == "C_category_neighbor_or_unstable":
        selected.sort(key=lambda r: (r["G_sup"], -r["G_cat"]), reverse=True)
    else:
        selected.sort(key=lambda r: (-r["G_sup"], -r["G_cat"]), reverse=True)
    selected = selected[:limit]

    cards = []
    for r in selected:
        gt_uri = file_uri(image_root, r["gt_image"])
        pred_uri = file_uri(image_root, r["pred_image"])
        cards.append(f"""
        <div class="card">
          <div class="caption"><b>Caption:</b> {html.escape(str(r['caption']))}</div>
          <div class="imgs">
            <div><img src="{gt_uri}"><p><b>GT</b><br>{html.escape(r['gt_class'])}<br>{html.escape(str(r['gt_image']))}</p></div>
            <div><img src="{pred_uri}"><p><b>Wrong Top1</b><br>{html.escape(r['pred_class'])}<br>{html.escape(str(r['pred_image']))}</p></div>
          </div>
          <table>
            <tr><th>Signal</th><th>GT</th><th>Wrong/Pred</th><th>Gap</th></tr>
            <tr><td>Full caption → image</td><td>{r['full_gt_score']:.4f}</td><td>{r['full_wrong_score']:.4f}</td><td>G_full={r['G_full']:+.4f}</td></tr>
            <tr><td>GT class name → image</td><td>{r['cat_gt_score']:.4f}</td><td>{r['cat_wrong_score']:.4f}</td><td>G_cat={r['G_cat']:+.4f}</td></tr>
            <tr><td>Caption → class name</td><td>{r['support_gt_score']:.4f}</td><td>{r['support_pred_score']:.4f}</td><td>G_sup={r['G_sup']:+.4f}</td></tr>
          </table>
          <p><b>D_cross_granularity = G_cat - G_full:</b> {r['D_cross_granularity']:+.4f}</p>
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
        <h1>{html.escape(region)}</h1>
        <p>这里只做人工诊断分区，不代表最终训练规则。</p>
        """ + "\n".join(cards) + "</body></html>", encoding="utf-8"
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
        config["dataset"], evaluate=True, eval_split=args.split,
        eval_transform=model.backbone.preprocess_val,
    )
    loader = create_loader(
        dataset, batch_size=args.image_batch_size,
        num_workers=args.num_workers, is_train=False,
    )

    print(f"Extracting {args.split.upper()} image features...")
    image_features = encode_images(model, loader, device)
    print(f"Extracting {args.split.upper()} caption features...")
    text_features = encode_texts(model, dataset.text, device, args.text_batch_size)

    image_categories = [get_category(ref, category_map) for ref in dataset.image]
    class_texts = [args.name_template.format(phrase_for_category(c)) for c in categories]
    print("Encoding 31 category names...")
    class_features = encode_texts(model, class_texts, device, args.text_batch_size)

    rows = analyze(dataset, image_features, text_features, image_categories, categories, class_features)
    summary = build_summary(rows)

    print("\n" + "=" * 104)
    print("PAIR-LEVEL CROSS-GRANULARITY PROBE")
    print("=" * 104)
    for name in ["all", "A_clean_candidate", "B_caption_label_misalignment", "C_category_neighbor_or_unstable", "D_ambiguous"]:
        s = summary[name]
        print(
            f"{name:>34} | N={s['count']:4d} ({s['ratio']:5.2f}%) | "
            f"Gfull med={s['G_full']['median']:+.4f} | "
            f"Gcat med={s['G_cat']['median']:+.4f} | "
            f"Gsup med={s['G_sup']['median']:+.4f} | "
            f"Dcg med={s['D_cross_granularity']['median']:+.4f}"
        )

    csv_path = output_dir / f"{args.split}_pair_cross_granularity.csv"
    json_path = output_dir / f"{args.split}_pair_cross_granularity_summary.json"
    save_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "config": args.config, "checkpoint": args.checkpoint,
            "split": args.split, "name_template": args.name_template,
            "categories": categories, "summary": summary,
        }, f, ensure_ascii=False, indent=2)

    for region in ["A_clean_candidate", "B_caption_label_misalignment", "C_category_neighbor_or_unstable", "D_ambiguous"]:
        build_region_html(rows, output_dir / f"{args.split}_{region}.html", config["dataset"]["image_root"], region, args.html_limit)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    for region in ["A_clean_candidate", "B_caption_label_misalignment", "C_category_neighbor_or_unstable", "D_ambiguous"]:
        print(f"  {output_dir / f'{args.split}_{region}.html'}")

    print("\n优先判断：")
    print("1) A区是否确实主要是 clean cross-category errors；")
    print("2) B区是否大量出现 park->pond、school->playfields 这类 caption-label 粒度错位；")
    print("3) C区是否主要是 bridge/river、playground/stadium 等类别近邻或类别锚点不稳定样本；")
    print("4) 如果人工分区成立，再把 A/B/C/D 逻辑接入训练；当前脚本不定义最终 reliability 公式。")


if __name__ == "__main__":
    main()
