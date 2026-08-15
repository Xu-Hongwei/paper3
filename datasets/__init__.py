from torch.utils.data import DataLoader

from .re_dataset import (
    re_train_dataset,
    re_eval_dataset,
    re_train_collate_fn,
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
            If None, use the default transform defined
            in datasets/transforms.py.

        eval_transform:
            Optional external evaluation transform.
            If None, use the default transform defined
            in datasets/transforms.py.

    Expected config fields:
        image_root
        train_file
        val_file
        test_file
        image_res
        max_words

    Optional training field:
        entity_index_file

    entity_index_file 指向紧凑的 Entity token-span 索引。
    训练集返回每个样本对应的 Entity spans，不再传递 Entity 字符串。
    """

    image_res = config.get(
        "image_res",
        224,
    )

    max_words = config.get(
        "max_words",
        30,
    )

    # --------------------------------------------------
    # Build transforms.
    # --------------------------------------------------

    if train_transform is None:

        train_transform = (
            build_train_transform(
                image_res
            )
        )

    if eval_transform is None:

        eval_transform = (
            build_eval_transform(
                image_res
            )
        )

    # --------------------------------------------------
    # Evaluation mode.
    #
    # Validation/test currently do NOT use EAR entities.
    # --------------------------------------------------

    if evaluate:

        if eval_split not in (
            "val",
            "test",
        ):

            raise ValueError(
                f"Unknown eval split: "
                f"{eval_split}"
            )

        ann_file = (
            config["val_file"]
            if eval_split == "val"
            else config["test_file"]
        )

        return re_eval_dataset(
            ann_file=ann_file,
            transform=eval_transform,
            image_root=(
                config["image_root"]
            ),
            max_words=max_words,
        )

    # --------------------------------------------------
    # Training dataset.
    #
    # EAR entity index is training-only for now.
    # --------------------------------------------------

    train_dataset = re_train_dataset(
        ann_file=config["train_file"],
        transform=train_transform,
        image_root=config["image_root"],
        max_words=max_words,
        entity_index_file=config.get(
            "entity_index_file"
        ),
    )

    # --------------------------------------------------
    # Validation dataset.
    # --------------------------------------------------

    val_dataset = re_eval_dataset(
        ann_file=config["val_file"],
        transform=eval_transform,
        image_root=config["image_root"],
        max_words=max_words,
    )

    return (
        train_dataset,
        val_dataset,
    )


def create_loader(
    dataset,
    batch_size,
    num_workers=4,
    is_train=False,
    pin_memory=True,
):
    """
    Build retrieval DataLoader.

    训练 batch:
        images             Tensor [B, 3, H, W]
        captions           List[str]
        image_ids          LongTensor [B]
        entity_spans       LongTensor [N_entity, 2]
        entity_sample_ids  LongTensor [N_entity]
        entity_counts      LongTensor [B]

    验证/测试保持默认 PyTorch collation。
    """

    # 训练样本的 Entity span 数量可变，因此使用自定义 collate。

    collate_fn = (
        re_train_collate_fn
        if is_train
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        collate_fn=collate_fn,
    )