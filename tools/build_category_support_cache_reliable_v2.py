import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset
from datasets.re_dataset import RSICD_CLASSES, resolve_image_path
from models import CLIPRetrieval
from utils import load_config


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
            "Precompute frozen baseline support cache for "
            "Reliable T2I Cross-Category Margin"
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--class-dir",
        type=str,
        default=None,
        help=(
            "可选：直接指定 released txtclasses_rsicd 目录。"
            "若提供，会覆盖 config.dataset.category_class_dir。"
        ),
    )
    parser.add_argument(
        "--name-template",
        type=str,
        default="{}",
        help='类别文本模板。默认 "{}" 与前面的 pair probe 保持一致。',
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


class UniqueTrainImageDataset(Dataset):
    """严格按照 train_dataset.image_id 顺序读取唯一训练图像。"""

    def __init__(self, image_references, image_root, transform):
        self.image_references = list(image_references)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.image_references)

    def __getitem__(self, index):
        image_reference = self.image_references[index]
        image_path = resolve_image_path(
            self.image_root,
            image_reference,
        )
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, index


@torch.no_grad()
def encode_texts(model, texts, device, batch_size):
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        features = model.backbone.encode_text(
            batch_texts,
            normalize=True,
        )
        outputs.append(features.cpu())

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_images(model, data_loader, device):
    outputs = [None] * len(data_loader.dataset)

    for images, indices in data_loader:
        images = images.to(device, non_blocking=True)

        features = model.backbone.encode_image(
            images,
            normalize=True,
        ).cpu()

        for local_index, image_id in enumerate(indices.tolist()):
            outputs[image_id] = features[local_index]

    if any(feature is None for feature in outputs):
        raise RuntimeError("存在未成功提取的训练图像特征。")

    return torch.stack(outputs, dim=0)


def check_dataset(train_dataset):
    """确保 support cache 与 Reliable 实验的数据索引严格一致。"""
    required_attrs = {
        "ann",
        "image_ids",
        "category_ids",
        "image_references",
        "image_category_ids",
        "num_images",
    }
    missing = [
        name
        for name in required_attrs
        if not hasattr(train_dataset, name)
    ]
    if missing:
        raise AttributeError(
            "当前 re_dataset.py 缺少 Reliable cache 所需字段: "
            f"{missing}"
        )

    if len(train_dataset.image_ids) != len(train_dataset):
        raise RuntimeError(
            "train_dataset.image_ids 与训练 pair 数量不一致。"
        )

    if len(train_dataset.category_ids) != len(train_dataset):
        raise RuntimeError(
            "train_dataset.category_ids 与训练 pair 数量不一致。"
        )

    if len(train_dataset.image_references) != train_dataset.num_images:
        raise RuntimeError(
            "image_references 与 num_images 不一致。"
        )

    if len(train_dataset.image_category_ids) != train_dataset.num_images:
        raise RuntimeError(
            "image_category_ids 与 num_images 不一致。"
        )

    unknown_pairs = sum(
        int(category_id) < 0
        for category_id in train_dataset.category_ids
    )
    unknown_images = sum(
        int(category_id) < 0
        for category_id in train_dataset.image_category_ids
    )

    if unknown_pairs > 0 or unknown_images > 0:
        raise RuntimeError(
            "Reliable cache 要求 official 31-group mapping 全覆盖；"
            f"当前 unknown pairs={unknown_pairs}, "
            f"unknown images={unknown_images}。"
        )

    if len(RSICD_CLASSES) != 31:
        raise RuntimeError(
            f"期望 31 个 released groups，当前为 {len(RSICD_CLASSES)}。"
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    # Baseline YAML 通常没有 category_class_dir。
    # Reliable cache 构建必须显式使用 released 31-group mapping，
    # 因此允许命令行直接覆盖，避免回退到旧文件名前缀解析。
    if args.class_dir is not None:
        config["dataset"]["category_class_dir"] = args.class_dir
        config["dataset"].pop("category_map_file", None)

    if not config["dataset"].get("category_class_dir") and not config[
        "dataset"
    ].get("category_map_file"):
        raise ValueError(
            "Reliable cache 构建需要 official category mapping。"
            "请在 YAML 的 dataset 中设置 category_class_dir，"
            "或通过 --class-dir 指定 txtclasses_rsicd。"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 88)
    print("BUILD FROZEN CATEGORY SUPPORT CACHE")
    print("=" * 88)
    print(f"Config          : {args.config}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Output          : {args.output}")
    print(f"Device          : {device}")
    print(f"Name template   : {args.name_template}")
    print(
        "Category source : "
        f"{config['dataset'].get('category_class_dir') or config['dataset'].get('category_map_file')}"
    )

    # --------------------------------------------------
    # 1. Frozen baseline teacher
    # --------------------------------------------------
    print("\nBuilding frozen baseline teacher...")
    model = CLIPRetrieval(config["model"]).to(device)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
    )
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # --------------------------------------------------
    # 2. Build training dataset with EVAL transform
    # --------------------------------------------------
    print("\nBuilding training dataset with eval transform...")

    train_dataset, _ = create_dataset(
        config["dataset"],
        evaluate=False,
        train_transform=model.backbone.preprocess_val,
        eval_transform=model.backbone.preprocess_val,
    )

    check_dataset(train_dataset)

    print(f"Train pairs      : {len(train_dataset)}")
    print(f"Train images     : {train_dataset.num_images}")
    print(f"Category groups  : {len(RSICD_CLASSES)}")

    # --------------------------------------------------
    # 3. Encode 31 category names
    # --------------------------------------------------
    class_texts = [
        args.name_template.format(
            CLASS_PHRASES.get(category, category)
        )
        for category in RSICD_CLASSES
    ]

    print("\nEncoding 31 category texts...")
    class_features = encode_texts(
        model=model,
        texts=class_texts,
        device=device,
        batch_size=args.text_batch_size,
    )

    expected_class_shape = (
        len(RSICD_CLASSES),
        class_features.shape[1],
    )
    if tuple(class_features.shape) != expected_class_shape:
        raise RuntimeError(
            f"Unexpected class feature shape: "
            f"{tuple(class_features.shape)}"
        )

    # --------------------------------------------------
    # 4. Caption -> 31 classes
    # sample_index == train_dataset.ann index
    # --------------------------------------------------
    captions = [
        ann["caption"]
        for ann in train_dataset.ann
    ]

    print("Encoding all train captions...")
    caption_features = encode_texts(
        model=model,
        texts=captions,
        device=device,
        batch_size=args.text_batch_size,
    )

    caption_support = (
        caption_features
        @ class_features.t()
    ).contiguous().float()

    # --------------------------------------------------
    # 5. Unique image -> 31 classes
    # image_support row index == train_dataset image_id
    # --------------------------------------------------
    image_dataset = UniqueTrainImageDataset(
        image_references=train_dataset.image_references,
        image_root=config["dataset"]["image_root"],
        transform=model.backbone.preprocess_val,
    )

    image_loader = DataLoader(
        image_dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("Encoding unique train images...")
    image_features = encode_images(
        model=model,
        data_loader=image_loader,
        device=device,
    )

    image_support = (
        image_features
        @ class_features.t()
    ).contiguous().float()

    # --------------------------------------------------
    # 6. Build strict index-alignment metadata
    # --------------------------------------------------
    sample_image_ids = torch.tensor(
        train_dataset.image_ids,
        dtype=torch.long,
    )
    sample_category_ids = torch.tensor(
        train_dataset.category_ids,
        dtype=torch.long,
    )
    image_category_ids = torch.tensor(
        train_dataset.image_category_ids,
        dtype=torch.long,
    )
    pair_indices = torch.tensor(
        [
            int(ann["_pair_index"])
            for ann in train_dataset.ann
        ],
        dtype=torch.long,
    )

    expected_caption_shape = (
        len(train_dataset),
        len(RSICD_CLASSES),
    )
    expected_image_shape = (
        train_dataset.num_images,
        len(RSICD_CLASSES),
    )

    if tuple(caption_support.shape) != expected_caption_shape:
        raise RuntimeError(
            "caption_support shape mismatch: "
            f"{tuple(caption_support.shape)} vs "
            f"{expected_caption_shape}"
        )

    if tuple(image_support.shape) != expected_image_shape:
        raise RuntimeError(
            "image_support shape mismatch: "
            f"{tuple(image_support.shape)} vs "
            f"{expected_image_shape}"
        )

    reconstructed_category_ids = (
        image_category_ids[sample_image_ids]
    )
    if not torch.equal(
        reconstructed_category_ids,
        sample_category_ids,
    ):
        raise RuntimeError(
            "sample image/category index alignment failed."
        )

    # --------------------------------------------------
    # 7. Save cache
    # --------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {
        "caption_support": caption_support,
        "image_support": image_support,
        "sample_image_ids": sample_image_ids,
        "sample_category_ids": sample_category_ids,
        "image_category_ids": image_category_ids,
        "pair_indices": pair_indices,
        "category_names": list(RSICD_CLASSES),
        "class_texts": class_texts,
        "image_references": list(
            train_dataset.image_references
        ),
        "metadata": {
            "format": "rsicd_reliable_category_support_v1",
            "teacher_checkpoint": str(args.checkpoint),
            "teacher_epoch": checkpoint.get("epoch"),
            "num_pairs": len(train_dataset),
            "num_images": train_dataset.num_images,
            "num_categories": len(RSICD_CLASSES),
            "name_template": args.name_template,
        },
    }

    torch.save(cache, output_path)

    # 额外保存轻量 JSON，方便人工检查，不参与训练。
    meta_path = output_path.with_suffix(".json")
    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                **cache["metadata"],
                "category_names": cache["category_names"],
                "class_texts": cache["class_texts"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("CACHE BUILT SUCCESSFULLY")
    print("=" * 88)
    print(
        f"caption_support : {tuple(caption_support.shape)}"
    )
    print(
        f"image_support   : {tuple(image_support.shape)}"
    )
    print(
        f"sample_image_ids: {tuple(sample_image_ids.shape)}"
    )
    print(
        f"sample_cat_ids  : {tuple(sample_category_ids.shape)}"
    )
    print(
        f"image_cat_ids   : {tuple(image_category_ids.shape)}"
    )
    print(f"saved           : {output_path}")
    print(f"metadata        : {meta_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
