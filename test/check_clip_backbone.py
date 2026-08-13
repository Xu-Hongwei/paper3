import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from utils import load_config

from datasets import (
    create_dataset,
    create_loader,
)

from models import CLIPRetrieval


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Check CLIP backbone"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==============================================
    # 1. Config
    # ==============================================

    config = load_config(
        args.config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CLIP BACKBONE CHECK")
    print("=" * 70)

    print(
        f"Device       : {device}"
    )

    print(
        f"Backbone     : "
        f"{config['model']['backbone']}"
    )

    print(
        f"Pretrained   : "
        f"{config['model']['pretrained']}"
    )

    # ==============================================
    # 2. Build CLIP
    # ==============================================

    model = CLIPRetrieval(
        config["model"]
    )

    model = model.to(device)

    # IMPORTANT:
    # Stage 3 only performs feature extraction.
    model.eval()

    # ==============================================
    # 3. Build validation dataset using
    #    CLIP official preprocessing
    # ==============================================

    val_dataset = create_dataset(
        config["dataset"],
        evaluate=True,
        eval_split="val",
        eval_transform=(
            model.backbone.preprocess_val
        ),
    )

    val_loader = create_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        is_train=False,
        pin_memory=False,
    )

    # ==============================================
    # 4. Read a batch of unique images
    # ==============================================

    images, image_ids = next(
        iter(val_loader)
    )

    images = images.to(device)

    # Select the first GT caption
    # associated with each image.
    captions = []

    for image_id in image_ids.tolist():

        text_ids = (
            val_dataset.img2txt[
                image_id
            ]
        )

        first_text_id = (
            text_ids[0]
        )

        captions.append(
            val_dataset.text[
                first_text_id
            ]
        )

    print()
    print(
        f"Image batch shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Number captions   : "
        f"{len(captions)}"
    )

    print()

    for i, caption in enumerate(
        captions
    ):
        print(
            f"[{i}] {caption}"
        )

    # ==============================================
    # 5. CLIP forward
    # ==============================================

    with torch.no_grad():

        outputs = model(
            images,
            captions,
        )

    image_feat = outputs[
        "image_feat"
    ]

    text_feat = outputs[
        "text_feat"
    ]

    logit_scale = outputs[
        "logit_scale"
    ]

    # ==============================================
    # 6. Check shape
    # ==============================================

    print()
    print("=" * 70)
    print("FEATURE CHECK")
    print("=" * 70)

    print(
        "Image feature shape:",
        tuple(image_feat.shape),
    )

    print(
        "Text feature shape :",
        tuple(text_feat.shape),
    )

    print(
        "Logit scale        :",
        float(logit_scale),
    )

    # ==============================================
    # 7. Check L2 normalization
    # ==============================================

    image_norm = image_feat.norm(
        dim=-1
    )

    text_norm = text_feat.norm(
        dim=-1
    )

    print()
    print(
        "Image feature norms:",
        image_norm.cpu().tolist(),
    )

    print(
        "Text feature norms :",
        text_norm.cpu().tolist(),
    )

    # ==============================================
    # 8. Check finite values
    # ==============================================

    image_finite = (
        torch.isfinite(
            image_feat
        ).all()
    )

    text_finite = (
        torch.isfinite(
            text_feat
        ).all()
    )

    print()
    print(
        "Image finite:",
        bool(image_finite),
    )

    print(
        "Text finite :",
        bool(text_finite),
    )

    # ==============================================
    # 9. Simple cosine similarity matrix
    # ==============================================

    similarity = (
        image_feat
        @ text_feat.T
    )

    print()
    print(
        "Cosine similarity matrix:"
    )

    print(
        similarity.cpu()
    )

    # ==============================================
    # 10. Assertions
    # ==============================================

    assert (
        image_feat.shape[0]
        ==
        text_feat.shape[0]
    )

    assert (
        image_feat.shape[-1]
        ==
        text_feat.shape[-1]
    )

    assert torch.allclose(
        image_norm,
        torch.ones_like(
            image_norm
        ),
        atol=1e-4,
    )

    assert torch.allclose(
        text_norm,
        torch.ones_like(
            text_norm
        ),
        atol=1e-4,
    )

    assert image_finite
    assert text_finite

    print()
    print("=" * 70)
    print(
        "CLIP BACKBONE CHECK: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()