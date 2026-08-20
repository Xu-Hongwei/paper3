from torch.utils.data import DataLoader

from .re_dataset import (
    re_eval_dataset,
    re_train_collate_fn,
    re_train_dataset,
)
from .transforms import build_eval_transform, build_train_transform


def create_dataset(
    config,
    evaluate=False,
    eval_split="val",
    train_transform=None,
    eval_transform=None,
):
    """创建 RSITR 训练 / 验证 / 测试数据集。"""
    image_res = config.get("image_res", 224)
    max_words = config.get("max_words", 30)

    if train_transform is None:
        train_transform = build_train_transform(image_res)
    if eval_transform is None:
        eval_transform = build_eval_transform(image_res)

    if evaluate:
        if eval_split not in {"val", "test"}:
            raise ValueError(
                f"Unknown eval split: {eval_split}"
            )

        ann_file = (
            config["val_file"]
            if eval_split == "val"
            else config["test_file"]
        )

        return re_eval_dataset(
            ann_file=ann_file,
            transform=eval_transform,
            image_root=config["image_root"],
            max_words=max_words,
        )

    train_dataset = re_train_dataset(
        ann_file=config["train_file"],
        transform=train_transform,
        image_root=config["image_root"],
        max_words=max_words,
        entity_index_file=config.get("entity_index_file"),
        category_class_dir=config.get("category_class_dir"),
        category_map_file=config.get("category_map_file"),
    )

    val_dataset = re_eval_dataset(
        ann_file=config["val_file"],
        transform=eval_transform,
        image_root=config["image_root"],
        max_words=max_words,
    )

    return train_dataset, val_dataset


def create_loader(
    dataset,
    batch_size,
    num_workers=4,
    is_train=False,
    pin_memory=True,
):
    """
    训练 batch 保留 category_id、sample_index 与 Entity spans；
    验证 / 测试使用默认 PyTorch collation。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=re_train_collate_fn if is_train else None,
    )
