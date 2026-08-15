import argparse

import torch
import torch.nn.functional as F
import yaml

from models import CLIPRetrieval


def compare_features(name, old_feat, new_feat, atol=1e-5, rtol=1e-5):
    """比较两条前向路径是否产生等价的全局特征。"""
    if old_feat.shape != new_feat.shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"{tuple(old_feat.shape)} vs {tuple(new_feat.shape)}"
        )

    diff = (old_feat - new_feat).abs()
    cosine = F.cosine_similarity(old_feat, new_feat, dim=-1)
    allclose = torch.allclose(old_feat, new_feat, atol=atol, rtol=rtol)

    print(f"\n{name}")
    print("-" * 64)
    print(f"shape          : {tuple(old_feat.shape)}")
    print(f"max abs error  : {diff.max().item():.8e}")
    print(f"mean abs error : {diff.mean().item():.8e}")
    print(f"cosine min     : {cosine.min().item():.8f}")
    print(f"cosine mean    : {cosine.mean().item():.8f}")
    print(f"allclose       : {allclose}")

    return allclose


def main():
    parser = argparse.ArgumentParser(
        description="验证优化前后 CLIP global feature 是否等价。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline/rsicd_entity_smoke.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_res = int(config["dataset"].get("image_res", 224))

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 使用和 train.py 完全一致的模型入口，避免直接依赖内部模块路径。
    model = CLIPRetrieval(config["model"]).to(device)
    model.eval()
    backbone = model.backbone

    base_captions = [
        "some buildings and meadows are near a viaduct",
        "several airplanes are parked at an airport",
        "a bridge crosses a river surrounded by trees",
        "many houses are located beside a road",
        "a large ship is sailing on the sea",
        "green fields are next to several buildings",
        "a stadium is surrounded by residential buildings",
        "two roads intersect near a group of trees",
    ]
    repeats = (args.batch_size + len(base_captions) - 1) // len(base_captions)
    captions = (base_captions * repeats)[:args.batch_size]

    images = torch.randn(
        args.batch_size,
        3,
        image_res,
        image_res,
        device=device,
    )

    with torch.no_grad():
        old_text = backbone.encode_text(captions, normalize=True)

        new_text_raw, token_features = backbone.encode_text_with_tokens(
            captions,
            normalize=False,
        )
        new_text = F.normalize(new_text_raw, dim=-1)

        old_image = backbone.encode_image(images, normalize=True)
        new_image, patch_features = backbone.encode_image_with_patches(
            images,
            normalize=True,
        )

    print()
    print("=" * 64)
    print("Global Feature Equivalence Test")
    print("=" * 64)
    print(f"device         : {device}")
    print(f"batch size     : {args.batch_size}")
    print(f"token features : {tuple(token_features.shape)}")
    print(f"patch features : {tuple(patch_features.shape)}")

    text_ok = compare_features(
        "Text global feature",
        old_text,
        new_text,
    )
    image_ok = compare_features(
        "Image global feature",
        old_image,
        new_image,
    )

    raw_scale = backbone.model.logit_scale.detach().item()
    exp_scale = backbone.model.logit_scale.detach().exp().item()

    print("\nLogit scale")
    print("-" * 64)
    print(f"raw logit_scale  : {raw_scale:.8f}")
    print(f"exp(logit_scale) : {exp_scale:.8f}")

    print()
    print("=" * 64)
    if text_ok and image_ok:
        print("PASS: optimized global features are equivalent.")
    else:
        print("FAIL: global feature mismatch detected.")
        raise SystemExit(1)
    print("=" * 64)


if __name__ == "__main__":
    main()
