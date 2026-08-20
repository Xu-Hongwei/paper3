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
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
        description="RSICD 类间可靠性探针：类名 vs 单原型 vs 多原型（纯离线，不训练）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--category-map", default=None)
    parser.add_argument("--class-dir", default=None)
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
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
        default="outputs/probes/cross_category_reliability",
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


def resolve_image_path(image_root, image_ref):
    direct = Path(image_root) / image_ref
    if direct.is_file():
        return direct
    flat = Path(image_root) / Path(str(image_ref).replace("\\", "/")).name
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"找不到图像: {image_ref}; tried={direct}, {flat}")


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"annotation 必须是 list: {path}")
    return data


def flatten_train_annotations(train_file):
    files = train_file if isinstance(train_file, list) else [train_file]
    rows = []
    for path in files:
        for ann in load_json_list(path):
            if "image" not in ann or "caption" not in ann:
                raise KeyError(f"训练 annotation 缺少 image/caption: {path}")
            captions = ann["caption"]
            if isinstance(captions, str):
                rows.append({"image": ann["image"], "caption": captions})
            elif isinstance(captions, list):
                rows.extend({"image": ann["image"], "caption": c} for c in captions)
            else:
                raise ValueError(f"非法 caption 类型: {type(captions)}")
    return rows


class UniqueImageDataset(Dataset):
    def __init__(self, image_refs, image_root, transform):
        self.image_refs = image_refs
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.image_refs)

    def __getitem__(self, index):
        path = resolve_image_path(self.image_root, self.image_refs[index])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, index


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
        batch = texts[start : start + batch_size]
        outputs.append(model.backbone.encode_text(batch, normalize=True).cpu())
    return torch.cat(outputs, dim=0)


def phrase_for_category(category):
    return CLASS_PHRASES.get(category, category)


def spherical_kmeans(features, k, iters=30, seed=42):
    """轻量 cosine KMeans；features 已为 L2-normalized。"""
    x = F.normalize(features.float(), dim=1)
    if len(x) <= k:
        return x.clone()

    generator = torch.Generator().manual_seed(seed)
    init_ids = torch.randperm(len(x), generator=generator)[:k]
    centers = x[init_ids].clone()

    for _ in range(iters):
        labels = torch.argmax(x @ centers.t(), dim=1)
        new_centers = []
        for cid in range(k):
            members = x[labels == cid]
            if len(members) == 0:
                # 空簇直接保留旧中心，避免引入额外随机性
                new_centers.append(centers[cid])
            else:
                new_centers.append(F.normalize(members.mean(dim=0), dim=0))
        new_centers = torch.stack(new_centers)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=0):
            centers = new_centers
            break
        centers = new_centers

    return F.normalize(centers, dim=1)


def build_category_representations(
    model,
    config,
    category_map,
    categories,
    device,
    text_batch_size,
    num_prototypes,
    kmeans_iters,
    seed,
    name_template,
):
    rows = flatten_train_annotations(config["dataset"]["train_file"])
    train_texts = [row["caption"] for row in rows]
    train_categories = [get_category(row["image"], category_map) for row in rows]

    print("\nExtracting TRAIN caption features...")
    train_text_features = encode_texts(model, train_texts, device, text_batch_size)

    # 1) 单独类别名称
    name_texts = [
        name_template.format(phrase_for_category(category))
        for category in categories
    ]
    name_features = encode_texts(model, name_texts, device, text_batch_size)
    name_bank = {category: name_features[i] for i, category in enumerate(categories)}

    # 2) 真实 train captions 的单原型 + 3) 多原型
    by_class = defaultdict(list)
    for feature, category in zip(train_text_features, train_categories):
        by_class[category].append(feature)

    single_bank = {}
    multi_bank = {}
    for class_idx, category in enumerate(categories):
        if not by_class[category]:
            raise RuntimeError(f"训练集缺少类别: {category}")
        feats = torch.stack(by_class[category], dim=0)
        feats = F.normalize(feats, dim=1)
        single_bank[category] = F.normalize(feats.mean(dim=0), dim=0)
        multi_bank[category] = spherical_kmeans(
            feats,
            k=num_prototypes,
            iters=kmeans_iters,
            seed=seed + class_idx,
        )

    return name_bank, single_bank, multi_bank


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
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def method_scores(method, category, image_features, name_bank, single_bank, multi_bank):
    if method == "name":
        return image_features @ name_bank[category]
    if method == "single_proto":
        return image_features @ single_bank[category]
    if method == "multi_proto":
        # 每个图像与目标类别多个文本原型匹配，取最相似子原型
        return (image_features @ multi_bank[category].t()).max(dim=1).values
    raise ValueError(method)


def analyze_cross_category_errors(
    dataset,
    image_features,
    text_features,
    image_categories,
    name_bank,
    single_bank,
    multi_bank,
):
    full_scores = text_features @ image_features.t()
    num_images = len(image_features)
    methods = ["name", "single_proto", "multi_proto"]

    rows = []
    method_cache = {}

    for text_id in range(len(dataset.text)):
        gt_image = int(dataset.txt2img[text_id])
        gt_class = image_categories[gt_image]
        full_row = full_scores[text_id]
        pred_image = int(torch.argmax(full_row).item())
        pred_class = image_categories[pred_image]

        # 当前 probe 只研究 Baseline 的 Top1 跨类别误判
        if pred_image == gt_image or pred_class == gt_class:
            continue

        full_pred_rank = rank_1based(full_row, pred_image)
        full_gt_rank = rank_1based(full_row, gt_image)

        row = {
            "text_id": text_id,
            "caption": dataset.text[text_id],
            "gt_image_id": gt_image,
            "gt_image": dataset.image[gt_image],
            "gt_class": gt_class,
            "pred_image_id": pred_image,
            "pred_image": dataset.image[pred_image],
            "pred_class": pred_class,
            "full_gt_score": float(full_row[gt_image].item()),
            "full_pred_score": float(full_row[pred_image].item()),
            "full_gap_pred_minus_gt": float((full_row[pred_image] - full_row[gt_image]).item()),
            "full_gt_rank": full_gt_rank,
            "full_pred_rank": full_pred_rank,
        }

        for method in methods:
            key = (method, gt_class)
            if key not in method_cache:
                method_cache[key] = method_scores(
                    method,
                    gt_class,
                    image_features,
                    name_bank,
                    single_bank,
                    multi_bank,
                )
            scores = method_cache[key]
            gt_score = float(scores[gt_image].item())
            pred_score = float(scores[pred_image].item())
            gt_rank = rank_1based(scores, gt_image)
            pred_rank = rank_1based(scores, pred_image)
            gt_pct = percentile_from_rank(gt_rank, num_images)
            pred_pct = percentile_from_rank(pred_rank, num_images)

            # 正值越大：目标类别表示越能“保住GT并压低错误异类候选”
            separation = gt_pct - pred_pct
            # 正值越大：错误候选在类别表示下比在完整caption下掉得越明显
            discrepancy = math.log((1.0 + pred_rank) / (1.0 + full_pred_rank))

            row.update({
                f"{method}_gt_score": gt_score,
                f"{method}_pred_score": pred_score,
                f"{method}_score_gap_gt_minus_pred": gt_score - pred_score,
                f"{method}_gt_rank": gt_rank,
                f"{method}_pred_rank": pred_rank,
                f"{method}_gt_percentile": gt_pct,
                f"{method}_pred_percentile": pred_pct,
                f"{method}_percentile_separation": separation,
                f"{method}_rank_discrepancy": discrepancy,
                f"{method}_reject_correct": int(gt_score > pred_score),
            })

        # 重点看：类名与多原型对同一错误候选判断差多少
        row["multi_vs_name_pred_percentile"] = (
            row["multi_proto_pred_percentile"] - row["name_pred_percentile"]
        )
        row["multi_vs_name_separation_gain"] = (
            row["multi_proto_percentile_separation"] - row["name_percentile_separation"]
        )
        rows.append(row)

    return rows


def build_summary(rows):
    methods = ["name", "single_proto", "multi_proto"]
    summary = {
        "cross_category_top1_errors": len(rows),
        "methods": {},
    }

    for method in methods:
        reject = [r[f"{method}_reject_correct"] for r in rows]
        pred_pct = [r[f"{method}_pred_percentile"] for r in rows]
        gt_pct = [r[f"{method}_gt_percentile"] for r in rows]
        sep = [r[f"{method}_percentile_separation"] for r in rows]
        disc = [r[f"{method}_rank_discrepancy"] for r in rows]
        score_gap = [r[f"{method}_score_gap_gt_minus_pred"] for r in rows]

        summary["methods"][method] = {
            "gt_beats_wrong_rate": float(np.mean(reject) * 100.0) if reject else float("nan"),
            "wrong_candidate_percentile": summary_stats(pred_pct),
            "gt_percentile": summary_stats(gt_pct),
            "gt_minus_wrong_percentile": summary_stats(sep),
            "rank_discrepancy": summary_stats(disc),
            "score_gap_gt_minus_wrong": summary_stats(score_gap),
        }

    return summary


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_uri(image_root, image_ref):
    try:
        return resolve_image_path(image_root, image_ref).resolve().as_uri()
    except Exception:
        return ""


def build_html(rows, output_path, image_root, limit):
    # 优先展示“类名 vs 多原型”判断差异最大的样本
    chosen = sorted(
        rows,
        key=lambda r: abs(r["multi_vs_name_separation_gain"]),
        reverse=True,
    )[:limit]

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
            <tr><th>Method</th><th>GT rank</th><th>Wrong rank</th><th>GT pct</th><th>Wrong pct</th><th>Sep</th><th>Rank discrepancy</th></tr>
            <tr><td>Class name</td><td>{r["name_gt_rank"]}</td><td>{r["name_pred_rank"]}</td><td>{r["name_gt_percentile"]:.3f}</td><td>{r["name_pred_percentile"]:.3f}</td><td>{r["name_percentile_separation"]:+.3f}</td><td>{r["name_rank_discrepancy"]:+.3f}</td></tr>
            <tr><td>Single proto</td><td>{r["single_proto_gt_rank"]}</td><td>{r["single_proto_pred_rank"]}</td><td>{r["single_proto_gt_percentile"]:.3f}</td><td>{r["single_proto_pred_percentile"]:.3f}</td><td>{r["single_proto_percentile_separation"]:+.3f}</td><td>{r["single_proto_rank_discrepancy"]:+.3f}</td></tr>
            <tr><td>Multi proto</td><td>{r["multi_proto_gt_rank"]}</td><td>{r["multi_proto_pred_rank"]}</td><td>{r["multi_proto_gt_percentile"]:.3f}</td><td>{r["multi_proto_pred_percentile"]:.3f}</td><td>{r["multi_proto_percentile_separation"]:+.3f}</td><td>{r["multi_proto_rank_discrepancy"]:+.3f}</td></tr>
          </table>
          <p><b>Full:</b> GT rank={r["full_gt_rank"]}, wrong rank={r["full_pred_rank"]}, wrong-GT score gap={r["full_gap_pred_minus_gt"]:+.4f}</p>
        </div>
        """)

    output_path.write_text(
        """<!doctype html><html><head><meta charset="utf-8">
        <style>
        body{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5}
        .card{background:white;padding:16px;margin:0 0 20px;border-radius:8px}
        .caption{font-size:16px;margin-bottom:12px}
        .imgs{display:flex;gap:24px}.imgs div{width:260px}
        img{width:240px;height:240px;object-fit:cover;border:1px solid #ccc}
        table{border-collapse:collapse;margin-top:12px}
        th,td{border:1px solid #ccc;padding:6px 9px;text-align:right}
        th:first-child,td:first-child{text-align:left}
        </style></head><body>
        <h1>Cross-Category Reliability Probe</h1>
        <p>按“Multi-Prototype 相对 Class-Name 的 separation 变化绝对值”排序。</p>
        """ + "\n".join(cards) + "</body></html>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    name_bank, single_bank, multi_bank = build_category_representations(
        model=model,
        config=config,
        category_map=category_map,
        categories=categories,
        device=device,
        text_batch_size=args.text_batch_size,
        num_prototypes=args.num_prototypes,
        kmeans_iters=args.kmeans_iters,
        seed=args.seed,
        name_template=args.name_template,
    )

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

    print(f"Extracting {args.split.upper()} text features...")
    text_features = encode_texts(model, dataset.text, device, args.text_batch_size)

    image_categories = [get_category(ref, category_map) for ref in dataset.image]

    print("\nAnalyzing baseline cross-category Top1 errors...")
    rows = analyze_cross_category_errors(
        dataset=dataset,
        image_features=image_features,
        text_features=text_features,
        image_categories=image_categories,
        name_bank=name_bank,
        single_bank=single_bank,
        multi_bank=multi_bank,
    )

    summary = build_summary(rows)

    print("\n" + "=" * 96)
    print("CROSS-CATEGORY RELIABILITY PROBE")
    print("=" * 96)
    print(f"Cross-category Top1 errors: {summary['cross_category_top1_errors']}")
    print()
    for method, stats in summary["methods"].items():
        print(
            f"{method:>12} | "
            f"GT>Wrong={stats['gt_beats_wrong_rate']:.2f}% | "
            f"WrongPct med={stats['wrong_candidate_percentile']['median']:.3f} | "
            f"GTPct med={stats['gt_percentile']['median']:.3f} | "
            f"Sep med={stats['gt_minus_wrong_percentile']['median']:+.3f} | "
            f"RankDisc med={stats['rank_discrepancy']['median']:+.3f}"
        )

    csv_path = output_dir / f"{args.split}_cross_category_reliability.csv"
    json_path = output_dir / f"{args.split}_cross_category_reliability_summary.json"
    html_path = output_dir / f"{args.split}_method_disagreement.html"

    save_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "num_prototypes": args.num_prototypes,
                "name_template": args.name_template,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    build_html(
        rows=rows,
        output_path=html_path,
        image_root=config["dataset"]["image_root"],
        limit=args.html_limit,
    )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {html_path}")
    print("\n优先判断：")
    print("1) Multi-Prototype 的 GT>Wrong 是否高于 Class-Name；")
    print("2) Multi-Prototype 的 GT-minus-Wrong percentile separation 是否更大；")
    print("3) Wrong candidate 在 Multi-Prototype 下是否被更合理地下调，同时 GT percentile 不明显下降；")
    print("4) 打开 HTML 人工检查 Class-Name 与 Multi-Prototype 分歧最大的样本。")


if __name__ == "__main__":
    main()
