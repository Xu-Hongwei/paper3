import json
import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .utils import pre_caption


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def resolve_image_path(image_root: str, image_reference: str) -> str:
    """兼容 image_root/split/xxx.jpg 和 image_root/xxx.jpg 两种布局。"""
    image_path = os.path.join(image_root, image_reference)
    if os.path.isfile(image_path):
        return image_path

    flat_path = os.path.join(image_root, os.path.basename(image_reference))
    if os.path.isfile(flat_path):
        return flat_path

    raise FileNotFoundError(
        f"Image not found: {image_reference!r}; "
        f"tried {image_path!r} and {flat_path!r}"
    )


def re_train_collate_fn(batch):
    """
    将每个样本的变长 Entity span 拼成紧凑 batch。

    返回：
        images             [B, C, H, W]
        captions           List[str]
        image_ids          [B]
        entity_spans       [N_entity, 2]
        entity_sample_ids  [N_entity]
        entity_counts      [B]
    """
    images, captions, image_ids, spans = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = list(captions)
    image_ids = torch.tensor(image_ids, dtype=torch.long)

    entity_counts = torch.tensor(
        [item.shape[0] for item in spans],
        dtype=torch.long,
    )

    total_entities = int(entity_counts.sum().item())
    if total_entities > 0:
        entity_spans = torch.cat(spans, dim=0)
        entity_sample_ids = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.long),
            entity_counts,
        )
    else:
        entity_spans = torch.empty((0, 2), dtype=torch.long)
        entity_sample_ids = torch.empty((0,), dtype=torch.long)

    return (
        images,
        captions,
        image_ids,
        entity_spans,
        entity_sample_ids,
        entity_counts,
    )


class re_train_dataset(Dataset):
    """RSITR 训练集：使用紧凑 Entity token-span 索引，不再传递 Entity 字符串。"""

    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
        entity_index_file=None,
    ):
        super().__init__()

        if isinstance(ann_file, str):
            ann_file = [ann_file]

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        raw_ann = []
        for file_path in ann_file:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Training annotation must be a list: {file_path}")
            raw_ann.extend(data)

        if not raw_ann:
            raise ValueError("Training annotation is empty.")

        self.num_raw_pairs = len(raw_ann)
        self._load_entity_span_index(entity_index_file)

        # 保留原始 pair_index，因为 compact index 对应原始标注顺序。
        valid_ann = []
        for pair_index, ann in enumerate(raw_ann):
            if "caption" not in ann:
                raise KeyError(f"Missing key 'caption' at training index {pair_index}")

            try:
                caption = pre_caption(ann["caption"], self.max_words)
            except ValueError:
                continue

            item = dict(ann)
            item["caption"] = caption
            item["_pair_index"] = pair_index
            valid_ann.append(item)

        if not valid_ann:
            raise ValueError("Training annotation has no valid captions.")

        self.ann = valid_ann
        self.num_filtered_pairs = self.num_raw_pairs - len(self.ann)

        self.image_ids = []
        image_to_id = {}
        for ann_index, ann in enumerate(self.ann):
            if "image" not in ann:
                raise KeyError(f"Missing key 'image' at training index {ann_index}")

            image_key = ann["image"]
            if image_key not in image_to_id:
                image_to_id[image_key] = len(image_to_id)
            self.image_ids.append(image_to_id[image_key])

        self.num_images = len(image_to_id)
        self._print_report()

    def _load_entity_span_index(self, entity_index_file):
        self.entity_index_file = entity_index_file
        self.pair_to_semantic = None
        self.semantic_offsets = None
        self.span_start = None
        self.span_end = None
        self.entity_index_stats = None

        if entity_index_file is None:
            return

        index = torch.load(
            entity_index_file,
            map_location="cpu",
            weights_only=True,
        )

        required = (
            "pair_to_semantic",
            "semantic_offsets",
            "span_start",
            "span_end",
        )
        missing = [key for key in required if key not in index]
        if missing:
            raise ValueError(f"Entity span index missing keys: {missing}")

        self.pair_to_semantic = index["pair_to_semantic"]
        self.semantic_offsets = index["semantic_offsets"]
        self.span_start = index["span_start"]
        self.span_end = index["span_end"]
        self.entity_index_stats = index.get("statistics", {})

        if self.pair_to_semantic.ndim != 1:
            raise ValueError("pair_to_semantic must be 1D.")
        if len(self.pair_to_semantic) != self.num_raw_pairs:
            raise ValueError(
                "Entity span index / annotation length mismatch: "
                f"{len(self.pair_to_semantic)} vs {self.num_raw_pairs}"
            )
        if self.semantic_offsets.ndim != 1:
            raise ValueError("semantic_offsets must be 1D.")
        if self.span_start.ndim != 1 or self.span_end.ndim != 1:
            raise ValueError("span_start/span_end must be 1D.")
        if len(self.span_start) != len(self.span_end):
            raise ValueError("span_start/span_end length mismatch.")

        metadata = index.get("metadata", {})
        index_max_words = metadata.get("max_words")
        if index_max_words is not None and int(index_max_words) != self.max_words:
            raise ValueError(
                f"max_words mismatch: dataset={self.max_words}, "
                f"span_index={index_max_words}"
            )

    def _get_entity_spans(self, pair_index: int) -> torch.Tensor:
        """返回当前 pair 的有效 Entity spans，格式为 [E, 2] 的 [start, end)。"""
        if self.pair_to_semantic is None:
            return torch.empty((0, 2), dtype=torch.long)

        semantic_index = int(self.pair_to_semantic[pair_index].item())
        if semantic_index < 0 or semantic_index + 1 >= len(self.semantic_offsets):
            raise IndexError(
                f"Invalid semantic index: pair={pair_index}, semantic={semantic_index}"
            )

        begin = int(self.semantic_offsets[semantic_index].item())
        end = int(self.semantic_offsets[semantic_index + 1].item())

        if begin == end:
            return torch.empty((0, 2), dtype=torch.long)

        starts = self.span_start[begin:end].to(dtype=torch.long)
        ends = self.span_end[begin:end].to(dtype=torch.long)
        return torch.stack((starts, ends), dim=1)

    def _print_report(self):
        print()
        print("=" * 72)
        print("Retrieval Training Dataset")
        print("=" * 72)
        print(f"Raw training pairs    : {self.num_raw_pairs}")
        print(f"Valid training pairs  : {len(self.ann)}")
        print(f"Filtered captions     : {self.num_filtered_pairs}")
        print(f"Unique training images: {self.num_images}")

        if self.pair_to_semantic is not None:
            valid_entities = self.entity_index_stats.get(
                "valid_unique_entities",
                len(self.span_start),
            )
            print(f"Entity span index     : enabled")
            print(f"Valid entity spans    : {valid_entities}")
        else:
            print("Entity span index     : disabled")

        print("=" * 72)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        pair_index = ann["_pair_index"]

        image_path = resolve_image_path(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            ann["caption"],
            self.image_ids[index],
            self._get_entity_spans(pair_index),
        )


class re_eval_dataset(Dataset):
    """验证/测试集保持原有检索评测逻辑，不使用 Entity span。"""

    def __init__(self, ann_file, transform, image_root, max_words=30):
        super().__init__()

        with open(ann_file, "r", encoding="utf-8") as f:
            self.ann = json.load(f)

        if not isinstance(self.ann, list):
            raise ValueError(f"Evaluation annotation must be a list: {ann_file}")

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            if "image" not in ann or "caption" not in ann:
                raise KeyError(f"Invalid evaluation annotation at index {img_id}")

            captions = ann["caption"]
            if not isinstance(captions, list):
                raise ValueError(f"'caption' must be a list for sample {img_id}")

            self.image.append(ann["image"])
            self.img2txt[img_id] = []

            for caption in captions:
                try:
                    clean_caption = pre_caption(caption, self.max_words)
                except ValueError:
                    continue

                self.text.append(clean_caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

            if not self.img2txt[img_id]:
                raise ValueError(
                    f"Evaluation sample {img_id} has no valid captions."
                )

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = resolve_image_path(
            self.image_root,
            self.ann[index]["image"],
        )
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, index
