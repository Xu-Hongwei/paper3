import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.category_margin_loss import CrossCategoryMarginLoss


def unit(x, y):
    return F.normalize(
        torch.tensor([x, y], dtype=torch.float32),
        dim=0,
    )


def check_t2i_same_category_excluded():
    """
    text_0 查询：
        image_1 同类且更相似 -> 必须排除；
        image_2 跨类 -> 应成为 hard negative。
    """
    image_features = torch.stack([
        unit(0.80, 0.60),   # 0: class 0, paired positive
        unit(0.99, 0.10),   # 1: class 0, 更相似但同类
        unit(0.90, 0.44),   # 2: class 1, hardest cross-category
        unit(0.10, 0.99),   # 3: class 2
    ])
    text_features = torch.stack([
        unit(1.00, 0.00),
        unit(0.95, 0.31),
        unit(0.90, 0.44),
        unit(0.10, 0.99),
    ])
    category_ids = torch.tensor(
        [0, 0, 1, 2],
        dtype=torch.long,
    )

    criterion = CrossCategoryMarginLoss(
        margin=0.05,
    )
    out = criterion(
        image_features,
        text_features,
        category_ids,
    )

    selected = int(
        out["t2i_hard_neg_index"][0].item()
    )

    assert selected == 2, (
        "T2I failed: same-category image must be excluded. "
        f"Expected hard negative index 2, got {selected}."
    )

    expected = max(
        0.0,
        0.05
        - float(out["similarity_i2t"][0, 0])
        + float(out["similarity_i2t"][2, 0]),
    )
    actual = float(
        (
            0.05
            - out["similarity_i2t"][0, 0]
            + out["similarity_i2t"][2, 0]
        ).clamp_min(0)
    )

    assert abs(actual - expected) < 1e-6

    print(
        "[PASS] T2I 同类别候选被正确排除，"
        "hard negative = index 2 (cross-category)."
    )


def check_i2t_same_category_excluded():
    """
    image_0 查询：
        text_1 同类且更相似 -> 必须排除；
        text_2 跨类 -> 应成为 hard negative。
    """
    image_features = torch.stack([
        unit(1.00, 0.00),
        unit(0.95, 0.31),
        unit(0.20, 0.98),
        unit(-0.20, 0.98),
    ])
    text_features = torch.stack([
        unit(0.80, 0.60),   # 0: class 0, paired positive
        unit(0.99, 0.10),   # 1: class 0, 更相似但同类
        unit(0.90, 0.44),   # 2: class 1, hardest cross-category
        unit(0.10, 0.99),   # 3: class 2
    ])
    category_ids = torch.tensor(
        [0, 0, 1, 2],
        dtype=torch.long,
    )

    criterion = CrossCategoryMarginLoss(
        margin=0.05,
    )
    out = criterion(
        image_features,
        text_features,
        category_ids,
    )

    selected = int(
        out["i2t_hard_neg_index"][0].item()
    )

    assert selected == 2, (
        "I2T failed: same-category text must be excluded. "
        f"Expected hard negative index 2, got {selected}."
    )

    print(
        "[PASS] I2T 同类别候选被正确排除，"
        "hard negative = index 2 (cross-category)."
    )


def check_unknown_category_ignored():
    image_features = torch.stack([
        unit(1.0, 0.0),
        unit(0.9, 0.4),
        unit(0.8, 0.6),
    ])
    text_features = image_features.clone()
    category_ids = torch.tensor(
        [-1, 0, 1],
        dtype=torch.long,
    )

    out = CrossCategoryMarginLoss(0.05)(
        image_features,
        text_features,
        category_ids,
    )

    assert not bool(
        out["t2i_valid_mask"][0].item()
    )
    assert not bool(
        out["i2t_valid_mask"][0].item()
    )
    assert int(
        out["t2i_hard_neg_index"][0].item()
    ) == -1
    assert int(
        out["i2t_hard_neg_index"][0].item()
    ) == -1

    print(
        "[PASS] category_id=-1 的 anchor 被正确忽略。"
    )


def check_no_cross_category_negative():
    image_features = torch.randn(
        4,
        8,
        requires_grad=True,
    )
    text_features = torch.randn(
        4,
        8,
        requires_grad=True,
    )
    category_ids = torch.tensor(
        [3, 3, 3, 3],
        dtype=torch.long,
    )

    out = CrossCategoryMarginLoss(0.05)(
        image_features,
        text_features,
        category_ids,
    )

    assert out["t2i_valid_count"] == 0
    assert out["i2t_valid_count"] == 0
    assert torch.isfinite(out["loss"])
    assert float(out["loss"].detach()) == 0.0

    out["loss"].backward()

    assert image_features.grad is not None
    assert text_features.grad is not None
    assert torch.isfinite(image_features.grad).all()
    assert torch.isfinite(text_features.grad).all()

    print(
        "[PASS] batch 内没有跨类别 negative 时安全返回 0，"
        "backward 正常。"
    )


def check_active_margin_and_backward():
    # 两类样本故意设计为 hard negative 比 positive 更相似。
    image_features = torch.stack([
        unit(0.80, 0.60),
        unit(0.90, 0.44),
        unit(-0.80, 0.60),
        unit(-0.90, 0.44),
    ]).requires_grad_()

    text_features = torch.stack([
        unit(1.00, 0.00),
        unit(1.00, 0.00),
        unit(-1.00, 0.00),
        unit(-1.00, 0.00),
    ]).requires_grad_()

    category_ids = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    out = CrossCategoryMarginLoss(0.05)(
        image_features,
        text_features,
        category_ids,
    )

    assert out["t2i_valid_count"] == 4
    assert out["i2t_valid_count"] == 4
    assert out["t2i_active_count"] > 0
    assert out["i2t_active_count"] > 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()

    assert image_features.grad is not None
    assert text_features.grad is not None
    assert torch.isfinite(image_features.grad).all()
    assert torch.isfinite(text_features.grad).all()

    print(
        "[PASS] active triplet 能产生有限 loss 和梯度。"
    )


def main():
    print("=" * 88)
    print("CrossCategoryMarginLoss Sanity Check")
    print("=" * 88)

    check_t2i_same_category_excluded()
    check_i2t_same_category_excluded()
    check_unknown_category_ignored()
    check_no_cross_category_negative()
    check_active_margin_and_backward()

    print("-" * 88)
    print("ALL CHECKS PASSED")
    print("=" * 88)


if __name__ == "__main__":
    main()
