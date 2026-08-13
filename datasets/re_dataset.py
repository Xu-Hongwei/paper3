import json
import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .utils import pre_caption


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def resolve_image_path(image_root: str, image_reference: str) -> str:
    """
    Resolve an image path.

    Supports both:
        image_root/split/xxx.jpg
    and:
        image_root/xxx.jpg

    This is useful for RSICD/RSITMD variants with different folder layouts.
    """
    image_path = os.path.join(image_root, image_reference)

    if os.path.isfile(image_path):
        return image_path

    flat_path = os.path.join(
        image_root,
        os.path.basename(image_reference)
    )

    if os.path.isfile(flat_path):
        return flat_path

    raise FileNotFoundError(
        f"Image not found for reference {image_reference!r}; "
        f"tried {image_path!r} and {flat_path!r}"
    )


class re_train_dataset(Dataset):
    """
    Retrieval training dataset.

    Expected training annotation format:
    [
        {
            "image": "xxx.jpg",
            "caption": "a remote sensing caption"
        },
        ...
    ]

    `ann_file` should be a list of JSON files.
    """

    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
    ):
        super().__init__()

        if isinstance(ann_file, str):
            ann_file = [ann_file]

        self.ann = []

        for file_path in ann_file:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"Training annotation must be a list: {file_path}"
                )

            self.ann.extend(data)

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if "image" not in ann:
            raise KeyError(
                f"Missing key 'image' in training annotation index {index}"
            )

        if "caption" not in ann:
            raise KeyError(
                f"Missing key 'caption' in training annotation index {index}"
            )

        image_path = resolve_image_path(
            self.image_root,
            ann["image"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption = pre_caption(
            ann["caption"],
            self.max_words
        )

        return image, caption


class re_eval_dataset(Dataset):
    """
    Retrieval validation/test dataset.

    Expected evaluation annotation format:
    [
        {
            "image": "xxx.jpg",
            "caption": [
                "caption 1",
                "caption 2",
                "caption 3",
                "caption 4",
                "caption 5"
            ]
        },
        ...
    ]

    Builds:
        self.text    : all captions
        self.image   : all image references
        self.img2txt : image index -> caption indices
        self.txt2img : caption index -> image index
    """

    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
    ):
        super().__init__()

        with open(ann_file, "r", encoding="utf-8") as f:
            self.ann = json.load(f)

        if not isinstance(self.ann, list):
            raise ValueError(
                f"Evaluation annotation must be a list: {ann_file}"
            )

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0

        for img_id, ann in enumerate(self.ann):
            if "image" not in ann:
                raise KeyError(
                    f"Missing key 'image' in evaluation annotation index {img_id}"
                )

            if "caption" not in ann:
                raise KeyError(
                    f"Missing key 'caption' in evaluation annotation index {img_id}"
                )

            captions = ann["caption"]

            if not isinstance(captions, list):
                raise ValueError(
                    f"'caption' must be a list for evaluation sample {img_id}"
                )

            self.image.append(ann["image"])
            self.img2txt[img_id] = []

            for caption in captions:
                clean_caption = pre_caption(
                    caption,
                    self.max_words
                )

                self.text.append(clean_caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id

                txt_id += 1

    def __len__(self):
        # Evaluation loader iterates unique images.
        return len(self.image)

    def __getitem__(self, index):
        image_path = resolve_image_path(
            self.image_root,
            self.ann[index]["image"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, index
