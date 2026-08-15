import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_retrieval_metrics


@torch.no_grad()
def extract_image_features(model, data_loader, device):
    """提取经过 Visual Adapter 的归一化图像全局特征。"""
    model.eval()

    all_features = []
    all_ids = []

    for images, image_ids in data_loader:
        images = images.to(device, non_blocking=True)

        image_features = model.backbone.encode_image(
            images,
            normalize=False,
        )
        image_features = F.normalize(
            image_features + model.visual_adapter(image_features),
            dim=-1,
        )

        all_features.append(image_features.cpu())
        all_ids.append(image_ids.cpu())

    image_features = torch.cat(all_features, dim=0)
    image_ids = torch.cat(all_ids, dim=0)

    # 保证行顺序严格对应 dataset image_id。
    order = torch.argsort(image_ids)
    return image_features[order]


@torch.no_grad()
def extract_text_features(model, texts, device, batch_size=256):
    """提取经过 Text Adapter 的归一化文本全局特征。"""
    model.eval()

    all_features = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        text_features = model.backbone.encode_text(
            batch_texts,
            normalize=False,
        )
        text_features = F.normalize(
            text_features + model.text_adapter(text_features),
            dim=-1,
        )

        all_features.append(text_features.cpu())

    return torch.cat(all_features, dim=0)


@torch.no_grad()
def compute_similarity_matrix(image_features, text_features, device):
    """计算归一化图文特征的 cosine similarity matrix。"""
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    scores = image_features @ text_features.t()

    return scores.cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def evaluate_retrieval(
    model,
    data_loader,
    dataset,
    device,
    text_batch_size=256,
):
    """完整图文检索评测。"""
    print("Extracting image features...")
    image_features = extract_image_features(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    print("Extracting text features...")
    text_features = extract_text_features(
        model=model,
        texts=dataset.text,
        device=device,
        batch_size=text_batch_size,
    )

    print(f"Image features: {tuple(image_features.shape)}")
    print(f"Text features : {tuple(text_features.shape)}")

    print("Computing similarity matrix...")
    scores_i2t = compute_similarity_matrix(
        image_features,
        text_features,
        device,
    )
    print(f"Similarity matrix: {scores_i2t.shape}")

    print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(
        scores_i2t=scores_i2t,
        img2txt=dataset.img2txt,
        txt2img=dataset.txt2img,
    )

    return metrics, scores_i2t
