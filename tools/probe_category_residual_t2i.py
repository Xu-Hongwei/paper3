import argparse
import csv
import json
import os
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="RSICD T2I 类内 Category-Residual Probe（纯离线，不训练）"
    )
    parser.add_argument("--config", required=True, help="Clean CLIP YAML")
    parser.add_argument("--checkpoint", required=True, help="Clean CLIP checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--category-map",
        default=None,
        help="filename_to_category.json；与 --class-dir 二选一",
    )
    parser.add_argument(
        "--class-dir",
        default=None,
        help="RSICD txtclasses_rsicd 目录；自动读取全部 txt，包括 playfields",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="outputs/probes/category_residual_t2i",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
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

        # 支持 {"xxx.jpg": "airport"} 以及 {"mapping": {...}} 两种格式
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
                        raise ValueError(
                            f"重复类别映射: {key}: {old} vs {category}"
                        )
                    mapping[key] = category

    categories = sorted(set(mapping.values()))
    print(f"Category images : {len(mapping)}")
    print(f"Category groups : {len(categories)}")
    print(f"Categories      : {categories}")
    return mapping


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

    raise FileNotFoundError(
        f"找不到图像: {image_ref}; tried={direct}, {flat}"
    )


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
                for caption in captions:
                    rows.append({"image": ann["image"], "caption": caption})
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
        features = model.backbone.encode_text(batch, normalize=True)
        outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def build_train_prototypes(
    model,
    config,
    category_map,
    device,
    image_batch_size,
    text_batch_size,
    num_workers,
):
    rows = flatten_train_annotations(config["dataset"]["train_file"])

    # 保持首次出现顺序，去重训练图像
    image_refs = list(dict.fromkeys(row["image"] for row in rows))
    image_categories = [get_category(ref, category_map) for ref in image_refs]

    image_dataset = UniqueImageDataset(
        image_refs=image_refs,
        image_root=config["dataset"]["image_root"],
        transform=model.backbone.preprocess_val,
    )
    image_loader = DataLoader(
        image_dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\nExtracting TRAIN image features...")
    image_features = encode_images(model, image_loader, device)

    print("Extracting TRAIN text features...")
    train_texts = [row["caption"] for row in rows]
    text_features = encode_texts(
        model,
        train_texts,
        device,
        text_batch_size,
    )
    text_categories = [get_category(row["image"], category_map) for row in rows]

    image_sum = defaultdict(lambda: torch.zeros(image_features.shape[1]))
    image_count = defaultdict(int)
    for feature, category in zip(image_features, image_categories):
        image_sum[category] += feature
        image_count[category] += 1

    text_sum = defaultdict(lambda: torch.zeros(text_features.shape[1]))
    text_count = defaultdict(int)
    for feature, category in zip(text_features, text_categories):
        text_sum[category] += feature
        text_count[category] += 1

    categories = sorted(set(image_categories) | set(text_categories))
    prototypes = {}

    for category in categories:
        if image_count[category] == 0 or text_count[category] == 0:
            raise RuntimeError(f"类别缺少图像或文本样本: {category}")

        mu_v = F.normalize(
            image_sum[category] / image_count[category],
            dim=0,
        )
        mu_t = F.normalize(
            text_sum[category] / text_count[category],
            dim=0,
        )

        # 共享跨模态类别方向：train-only，避免 test leakage
        prototype = F.normalize(mu_v + mu_t, dim=0)
        prototypes[category] = prototype

    print(f"Built prototypes: {len(prototypes)}")
    return prototypes


def residualize(features, categories, prototypes, alpha):
    if alpha == 0:
        return features.clone()

    proto = torch.stack([prototypes[c] for c in categories], dim=0)
    coeff = (features * proto).sum(dim=1, keepdim=True)
    residual = features - float(alpha) * coeff * proto
    return F.normalize(residual, dim=1)


def ranks_to_recall(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "r1": float(np.mean(ranks <= 1) * 100.0),
        "r5": float(np.mean(ranks <= 5) * 100.0),
        "r10": float(np.mean(ranks <= 10) * 100.0),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def evaluate_t2i(
    image_features,
    text_features,
    image_categories,
    text_gt_images,
):
    scores = text_features @ image_features.t()

    intra_ranks = []
    global_ranks = []
    intra_gaps = []
    inter_gaps = []
    top1_images = []

    per_class = defaultdict(
        lambda: {
            "ranks": [],
            "intra_gaps": [],
            "inter_gaps": [],
        }
    )

    category_to_images = defaultdict(list)
    for image_id, category in enumerate(image_categories):
        category_to_images[category].append(image_id)

    all_image_ids = torch.arange(len(image_categories))

    for text_id, gt_image in enumerate(text_gt_images):
        gt_image = int(gt_image)
        category = image_categories[gt_image]
        row = scores[text_id]
        pos = float(row[gt_image].item())

        # 全局 rank
        global_rank = int((row > row[gt_image]).sum().item()) + 1
        global_ranks.append(global_rank)

        # 类内 rank
        same_ids = torch.tensor(category_to_images[category], dtype=torch.long)
        same_scores = row[same_ids]
        gt_pos = (same_ids == gt_image).nonzero(as_tuple=False).item()
        intra_rank = int((same_scores > same_scores[gt_pos]).sum().item()) + 1
        intra_ranks.append(intra_rank)

        local_top = int(torch.argmax(same_scores).item())
        top1_images.append(int(same_ids[local_top].item()))

        wrong_same = same_ids[same_ids != gt_image]
        if len(wrong_same) > 0:
            hard_same = float(row[wrong_same].max().item())
            intra_gap = pos - hard_same
        else:
            intra_gap = float("nan")

        diff_mask = torch.tensor(
            [c != category for c in image_categories],
            dtype=torch.bool,
        )
        diff_ids = all_image_ids[diff_mask]
        if len(diff_ids) > 0:
            hard_inter = float(row[diff_ids].max().item())
            inter_gap = pos - hard_inter
        else:
            inter_gap = float("nan")

        intra_gaps.append(intra_gap)
        inter_gaps.append(inter_gap)

        cls = per_class[category]
        cls["ranks"].append(intra_rank)
        cls["intra_gaps"].append(intra_gap)
        cls["inter_gaps"].append(inter_gap)

    metrics = {}
    metrics.update({f"intra_{k}": v for k, v in ranks_to_recall(intra_ranks).items()})
    metrics.update({f"global_{k}": v for k, v in ranks_to_recall(global_ranks).items()})

    intra_arr = np.asarray(intra_gaps, dtype=np.float64)
    inter_arr = np.asarray(inter_gaps, dtype=np.float64)
    metrics["intra_gap_mean"] = float(np.nanmean(intra_arr))
    metrics["intra_gap_median"] = float(np.nanmedian(intra_arr))
    metrics["inter_gap_mean"] = float(np.nanmean(inter_arr))
    metrics["inter_gap_median"] = float(np.nanmedian(inter_arr))

    class_metrics = {}
    for category, values in sorted(per_class.items()):
        rank_metrics = ranks_to_recall(values["ranks"])
        class_metrics[category] = {
            **rank_metrics,
            "intra_gap_mean": float(np.nanmean(values["intra_gaps"])),
            "inter_gap_mean": float(np.nanmean(values["inter_gaps"])),
            "num_queries": len(values["ranks"]),
        }

    return metrics, np.asarray(intra_ranks), np.asarray(top1_images), class_metrics


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_map = load_category_map(
        category_map=args.category_map,
        class_dir=args.class_dir,
    )

    print("\nBuilding Clean CLIP...")
    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device          : {device}")

    # 1) 只用 train split 建类别方向
    prototypes = build_train_prototypes(
        model=model,
        config=config,
        category_map=category_map,
        device=device,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
    )

    # 2) 提取 val/test 全局 CLIP 特征
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
    text_features = encode_texts(
        model,
        dataset.text,
        device,
        args.text_batch_size,
    )

    image_categories = [
        get_category(image_ref, category_map)
        for image_ref in dataset.image
    ]
    text_gt_images = np.asarray(
        [int(dataset.txt2img[i]) for i in range(len(dataset.text))],
        dtype=np.int64,
    )
    text_categories = [
        image_categories[gt_image]
        for gt_image in text_gt_images
    ]

    missing_proto = sorted(
        (set(image_categories) | set(text_categories)) - set(prototypes)
    )
    if missing_proto:
        raise RuntimeError(
            f"{args.split} 中出现 train prototype 不存在的类别: {missing_proto}"
        )

    print("\n" + "=" * 88)
    print("CATEGORY-RESIDUAL T2I PROBE")
    print("=" * 88)

    summary_rows = []
    per_class_rows = []
    baseline_top1 = None

    for alpha in args.alphas:
        image_res = residualize(
            image_features,
            image_categories,
            prototypes,
            alpha,
        )
        text_res = residualize(
            text_features,
            text_categories,
            prototypes,
            alpha,
        )

        metrics, intra_ranks, top1_images, class_metrics = evaluate_t2i(
            image_features=image_res,
            text_features=text_res,
            image_categories=image_categories,
            text_gt_images=text_gt_images,
        )

        if baseline_top1 is None:
            baseline_top1 = top1_images.copy()
            repair = 0
            damage = 0
            net_repair = 0
        else:
            gt = text_gt_images
            baseline_correct = baseline_top1 == gt
            current_correct = top1_images == gt
            repair = int(np.sum((~baseline_correct) & current_correct))
            damage = int(np.sum(baseline_correct & (~current_correct)))
            net_repair = repair - damage

        row = {
            "alpha": float(alpha),
            **metrics,
            "repair": repair,
            "damage": damage,
            "net_repair": net_repair,
        }
        summary_rows.append(row)

        print(
            f"alpha={alpha:>4.2f} | "
            f"Intra R@1/5/10={metrics['intra_r1']:.2f}/"
            f"{metrics['intra_r5']:.2f}/{metrics['intra_r10']:.2f} | "
            f"IntraGap mean/med={metrics['intra_gap_mean']:+.4f}/"
            f"{metrics['intra_gap_median']:+.4f} | "
            f"InterGap mean/med={metrics['inter_gap_mean']:+.4f}/"
            f"{metrics['inter_gap_median']:+.4f} | "
            f"Repair/Damage/Net={repair}/{damage}/{net_repair}"
        )

        for category, cls_metrics in class_metrics.items():
            per_class_rows.append(
                {
                    "alpha": float(alpha),
                    "category": category,
                    **cls_metrics,
                }
            )

    summary_path = output_dir / f"{args.split}_category_residual_summary.csv"
    class_path = output_dir / f"{args.split}_category_residual_per_class.csv"
    json_path = output_dir / f"{args.split}_category_residual_summary.json"

    save_csv(
        summary_path,
        summary_rows,
        fieldnames=list(summary_rows[0].keys()),
    )
    save_csv(
        class_path,
        per_class_rows,
        fieldnames=list(per_class_rows[0].keys()),
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "alphas": args.alphas,
                "category_groups": sorted(set(category_map.values())),
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {class_path}")
    print(f"  {json_path}")
    print("\n判断重点：")
    print("1) alpha>0 后 Intra R@1 是否提高；")
    print("2) intra_gap_mean/median 是否增大；")
    print("3) net_repair 是否持续为正；")
    print("4) inter_gap 是否出现明显恶化；")
    print("5) 哪些类别受益、哪些类别受损。")


if __name__ == "__main__":
    main()
