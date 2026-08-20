import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from datasets.re_dataset import (
    get_rsicd_category_name,
    re_train_collate_fn,
)
from losses.category_margin_loss import CrossCategoryMarginLoss
from models import CLIPRetrieval
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="用真实 RSICD batch 检查 CrossCategoryMarginLoss 的 hard negative。"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--preview", type=int, default=12)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CLIPRetrieval(config["model"])
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    datasets = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )
    train_dataset = datasets[0]

    begin = args.start_index
    end = min(begin + args.batch_size, len(train_dataset))
    if begin < 0 or begin >= len(train_dataset):
        raise ValueError(
            f"start-index must be in [0, {len(train_dataset) - 1}]"
        )
    if end - begin < 2:
        raise ValueError("Debug batch must contain at least 2 samples.")

    dataset_indices = list(range(begin, end))
    raw_samples = [train_dataset[i] for i in dataset_indices]
    batch = re_train_collate_fn(raw_samples)

    (
        images,
        captions,
        image_ids,
        category_ids,
        entity_spans,
        entity_sample_ids,
        entity_counts,
    ) = batch

    images = images.to(device, non_blocking=True)
    category_ids = category_ids.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images, captions)

    criterion = CrossCategoryMarginLoss(
        margin=args.margin,
    )

    margin_out = criterion(
        outputs["image_feat"],
        outputs["text_feat"],
        category_ids,
    )

    print("=" * 120)
    print("REAL RSICD BATCH CROSS-CATEGORY MARGIN CHECK")
    print("=" * 120)
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Epoch             : {checkpoint.get('epoch', 'unknown')}")
    print(f"Dataset indices   : {begin} ~ {end - 1}")
    print(f"Batch size        : {end - begin}")
    print(f"Margin            : {args.margin:.4f}")
    print(
        f"Known categories  : "
        f"{int((category_ids >= 0).sum().item())}/{len(category_ids)}"
    )
    print(
        f"T2I valid/active  : "
        f"{margin_out['t2i_valid_count']}/"
        f"{margin_out['t2i_active_count']}"
    )
    print(
        f"I2T valid/active  : "
        f"{margin_out['i2t_valid_count']}/"
        f"{margin_out['i2t_active_count']}"
    )
    print(
        f"T2I pos/hard-neg  : "
        f"{margin_out['t2i_mean_pos_sim'].item():.4f} / "
        f"{margin_out['t2i_mean_hard_neg_sim'].item():.4f}"
    )
    print(
        f"I2T pos/hard-neg  : "
        f"{margin_out['i2t_mean_pos_sim'].item():.4f} / "
        f"{margin_out['i2t_mean_hard_neg_sim'].item():.4f}"
    )
    print(
        f"T2I loss          : "
        f"{margin_out['loss_t2i'].item():.6f}"
    )
    print(
        f"I2T loss          : "
        f"{margin_out['loss_i2t'].item():.6f}"
    )

    sim_i2t = margin_out["similarity_i2t"]
    t2i_hard = margin_out["t2i_hard_neg_index"].cpu()
    i2t_hard = margin_out["i2t_hard_neg_index"].cpu()
    category_ids_cpu = category_ids.cpu()

    print("\nT2I HARD NEGATIVE PREVIEW")
    print("-" * 120)
    shown = 0
    for local_idx in range(len(captions)):
        neg_local_idx = int(t2i_hard[local_idx].item())
        if neg_local_idx < 0:
            continue

        anchor_cat = int(category_ids_cpu[local_idx].item())
        neg_cat = int(category_ids_cpu[neg_local_idx].item())

        assert anchor_cat >= 0
        assert neg_cat >= 0
        assert anchor_cat != neg_cat

        global_idx = dataset_indices[local_idx]
        neg_global_idx = dataset_indices[neg_local_idx]

        pos_sim = float(
            sim_i2t[local_idx, local_idx].item()
        )
        neg_sim = float(
            sim_i2t[neg_local_idx, local_idx].item()
        )

        print(
            f"[{shown + 1:02d}] pair={global_idx} "
            f"{get_rsicd_category_name(anchor_cat)} -> "
            f"{get_rsicd_category_name(neg_cat)}"
        )
        print(f"     caption : {captions[local_idx]}")
        print(
            f"     positive: "
            f"{train_dataset.ann[global_idx]['image']} "
            f"sim={pos_sim:.4f}"
        )
        print(
            f"     negative: "
            f"{train_dataset.ann[neg_global_idx]['image']} "
            f"sim={neg_sim:.4f}"
        )

        shown += 1
        if shown >= args.preview:
            break

    print("\nI2T HARD NEGATIVE PREVIEW")
    print("-" * 120)
    shown = 0
    for local_idx in range(len(captions)):
        neg_local_idx = int(i2t_hard[local_idx].item())
        if neg_local_idx < 0:
            continue

        anchor_cat = int(category_ids_cpu[local_idx].item())
        neg_cat = int(category_ids_cpu[neg_local_idx].item())

        assert anchor_cat >= 0
        assert neg_cat >= 0
        assert anchor_cat != neg_cat

        global_idx = dataset_indices[local_idx]
        neg_global_idx = dataset_indices[neg_local_idx]

        pos_sim = float(
            sim_i2t[local_idx, local_idx].item()
        )
        neg_sim = float(
            sim_i2t[local_idx, neg_local_idx].item()
        )

        print(
            f"[{shown + 1:02d}] pair={global_idx} "
            f"{get_rsicd_category_name(anchor_cat)} -> "
            f"{get_rsicd_category_name(neg_cat)}"
        )
        print(
            f"     query image : "
            f"{train_dataset.ann[global_idx]['image']}"
        )
        print(
            f"     positive txt: {captions[local_idx]} "
            f"sim={pos_sim:.4f}"
        )
        print(
            f"     negative txt: {captions[neg_local_idx]} "
            f"sim={neg_sim:.4f}"
        )
        print(
            f"     negative src: "
            f"{train_dataset.ann[neg_global_idx]['image']}"
        )

        shown += 1
        if shown >= args.preview:
            break

    print("=" * 120)
    print("CHECK COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
