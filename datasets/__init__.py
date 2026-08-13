from torch.utils.data import DataLoader

from .re_dataset import (
    re_train_dataset,
    re_eval_dataset,
)
from .transforms import (
    build_train_transform,
    build_eval_transform,
)


def create_dataset(
    config,
    evaluate=False,
    eval_split="val",
    train_transform=None,
    eval_transform=None,
):
    """
    Build RSITR retrieval datasets.

    Args:
        config:
            Dataset configuration dictionary.

        evaluate:
            False -> return train_dataset, val_dataset
            True  -> return eval_dataset only

        eval_split:
            "val" or "test"

        train_transform:
            Optional external training transform.
            If None, use the default transform defined in dataset/transforms.py.

        eval_transform:
            Optional external evaluation transform.
            If None, use the default transform defined in dataset/transforms.py.

    Expected config fields:
        image_root
        train_file
        val_file
        test_file
        image_res
        max_words
    """

    image_res = config.get("image_res", 224)
    max_words = config.get("max_words", 30)

    # --------------------------------------------------
    # Build default transforms only when external
    # transforms are not provided.
    # --------------------------------------------------
    if train_transform is None:
        train_transform = build_train_transform(
            image_res
        )

    if eval_transform is None:
        eval_transform = build_eval_transform(
            image_res
        )

    # --------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------
    if evaluate:
        if eval_split not in ("val", "test"):
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

    # --------------------------------------------------
    # Training dataset
    # --------------------------------------------------
    train_dataset = re_train_dataset(
        ann_file=config["train_file"],
        transform=train_transform,
        image_root=config["image_root"],
        max_words=max_words,
    )

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------
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
    Minimal DataLoader builder for Stage 2.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
