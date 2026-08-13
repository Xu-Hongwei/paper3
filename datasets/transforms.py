from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (
    0.48145466,
    0.4578275,
    0.40821073,
)

CLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711,
)


def build_train_transform(image_res=224):
    """
    Basic training transform for the Stage-2 dataset pipeline.

    Note:
        In Stage 3, after OpenCLIP is connected, this preprocessing
        should be checked against the selected CLIP/OpenCLIP recipe.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_res,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CLIP_MEAN,
            std=CLIP_STD,
        ),
    ])


def build_eval_transform(image_res=224):
    """
    Deterministic validation/test transform.
    """
    return transforms.Compose([
        transforms.Resize(
            (image_res, image_res),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CLIP_MEAN,
            std=CLIP_STD,
        ),
    ])
