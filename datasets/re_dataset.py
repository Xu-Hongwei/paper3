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
    image_path = os.path.join(
        image_root,
        image_reference,
    )

    if os.path.isfile(image_path):
        return image_path

    flat_path = os.path.join(
        image_root,
        os.path.basename(image_reference),
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

    Different captions belonging to the same image share
    the same image_id.

    Example:
        [
            {"image": "a.jpg", "caption": "caption 1"},
            {"image": "a.jpg", "caption": "caption 2"},
            {"image": "b.jpg", "caption": "caption 1"}
        ]

    will produce:

        image_id:
            a.jpg -> 0
            a.jpg -> 0
            b.jpg -> 1

    `ann_file` can be either a JSON path or a list
    of JSON paths.
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

        # --------------------------------------------------
        # Load annotations
        # --------------------------------------------------

        for file_path in ann_file:
            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"Training annotation must be a list: "
                    f"{file_path}"
                )

            self.ann.extend(data)

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        valid_ann = []

        for ann_index, ann in enumerate(self.ann):

            if "caption" not in ann:
                raise KeyError(
                    f"Missing key 'caption' in training "
                    f"annotation index {ann_index}"
                )

            try:
                clean_caption = pre_caption(
                    ann["caption"],
                    self.max_words,
                )
            except ValueError:
                continue

            valid_item = dict(ann)
            valid_item["caption"] = clean_caption
            valid_ann.append(valid_item)

        if not valid_ann:
            raise ValueError(
                "Training annotation has no valid captions"
            )

        self.ann = valid_ann

        # --------------------------------------------------
        # Build image identity mapping.
        #
        # All captions referring to the same image receive
        # the same image_id.
        #
        # This is used by the multi-positive CLIP loss to
        # prevent valid positive captions from being treated
        # as negatives.
        # --------------------------------------------------

        self.image_ids = []

        image_to_id = {}

        for ann_index, ann in enumerate(self.ann):

            if "image" not in ann:
                raise KeyError(
                    f"Missing key 'image' in training "
                    f"annotation index {ann_index}"
                )

            image_key = ann["image"]

            if image_key not in image_to_id:
                image_to_id[image_key] = len(image_to_id)

            self.image_ids.append(
                image_to_id[image_key]
            )

        # Number of unique training images.
        self.num_images = len(image_to_id)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if "image" not in ann:
            raise KeyError(
                f"Missing key 'image' in training "
                f"annotation index {index}"
            )

        if "caption" not in ann:
            raise KeyError(
                f"Missing key 'caption' in training "
                f"annotation index {index}"
            )

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        image_path = resolve_image_path(
            self.image_root,
            ann["image"],
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # --------------------------------------------------
        # Caption
        # --------------------------------------------------

        caption = ann["caption"]

        # --------------------------------------------------
        # Image identity
        #
        # Different captions of the same image share
        # the same image_id.
        # --------------------------------------------------

        image_id = self.image_ids[index]

        return image, caption, image_id


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

        with open(
            ann_file,
            "r",
            encoding="utf-8",
        ) as f:
            self.ann = json.load(f)

        if not isinstance(self.ann, list):
            raise ValueError(
                f"Evaluation annotation must be a list: "
                f"{ann_file}"
            )

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0

        # --------------------------------------------------
        # Build retrieval mappings
        # --------------------------------------------------

        for img_id, ann in enumerate(self.ann):

            if "image" not in ann:
                raise KeyError(
                    f"Missing key 'image' in evaluation "
                    f"annotation index {img_id}"
                )

            if "caption" not in ann:
                raise KeyError(
                    f"Missing key 'caption' in evaluation "
                    f"annotation index {img_id}"
                )

            captions = ann["caption"]

            if not isinstance(captions, list):
                raise ValueError(
                    f"'caption' must be a list for "
                    f"evaluation sample {img_id}"
                )

            self.image.append(
                ann["image"]
            )

            self.img2txt[img_id] = []

            valid_caption_count = 0

            for caption in captions:

                try:
                    clean_caption = pre_caption(
                        caption,
                        self.max_words,
                    )
                except ValueError:
                    continue

                self.text.append(
                    clean_caption
                )

                self.img2txt[
                    img_id
                ].append(
                    txt_id
                )

                self.txt2img[
                    txt_id
                ] = img_id

                txt_id += 1
                valid_caption_count += 1

            if valid_caption_count == 0:
                raise ValueError(
                    f"Evaluation sample {img_id} has no "
                    "valid captions after preprocessing"
                )

    def __len__(self):
        # Evaluation loader iterates unique images.
        return len(self.image)

    def __getitem__(self, index):

        image_path = resolve_image_path(
            self.image_root,
            self.ann[index]["image"],
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, index
