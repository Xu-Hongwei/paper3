import json
import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .utils import pre_caption


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


RSICD_CLASSES = [
    "airport",
    "bareland",
    "baseballfield",
    "beach",
    "bridge",
    "center",
    "church",
    "commercial",
    "denseresidential",
    "desert",
    "farmland",
    "forest",
    "industrial",
    "meadow",
    "mediumresidential",
    "mountain",
    "park",
    "parking",
    "playground",
    "playfields",
    "pond",
    "port",
    "railwaystation",
    "resort",
    "river",
    "school",
    "sparseresidential",
    "square",
    "stadium",
    "storagetanks",
    "viaduct",
]
RSICD_CLASS_TO_ID = {name: idx for idx, name in enumerate(RSICD_CLASSES)}


def resolve_image_path(image_root, image_reference):
    """兼容 image_root/split/xxx.jpg 与 image_root/xxx.jpg。"""
    nested_path = os.path.join(image_root, image_reference)
    if os.path.isfile(nested_path):
        return nested_path

    flat_path = os.path.join(image_root, os.path.basename(image_reference))
    if os.path.isfile(flat_path):
        return flat_path

    raise FileNotFoundError(
        f"Image not found: {image_reference!r}; "
        f"tried {nested_path!r} and {flat_path!r}"
    )


def _normalize_filename(image_reference):
    """统一为小写 basename，兼容 split/xxx.jpg 与 xxx.jpg。"""
    return os.path.basename(
        str(image_reference).replace("\\", "/")
    ).lower()


def load_rsicd_category_mapping(
    category_class_dir=None,
    category_map_file=None,
):
    """
    读取 filename -> category_name 映射。

    推荐 Reliable Category 实验使用 released txtclasses_rsicd：
        category_class_dir=.../txtclasses_rsicd

    category_map_file 支持：
        1. {"airport_1.jpg": "airport", ...}
        2. {"mapping": {...}}

    两者都未提供时返回 None，并回退到旧文件名解析，
    以保证旧 baseline 配置仍可加载。
    """
    if category_class_dir and category_map_file:
        raise ValueError(
            "category_class_dir 和 category_map_file 只能提供一个。"
        )

    if category_class_dir is None and category_map_file is None:
        return None

    mapping = {}

    if category_class_dir is not None:
        if not os.path.isdir(category_class_dir):
            raise FileNotFoundError(
                f"category_class_dir not found: {category_class_dir}"
            )

        txt_files = sorted(
            file_name
            for file_name in os.listdir(category_class_dir)
            if file_name.lower().endswith(".txt")
        )
        if not txt_files:
            raise FileNotFoundError(
                f"No class txt files found in: {category_class_dir}"
            )

        for file_name in txt_files:
            category_name = os.path.splitext(file_name)[0].strip().lower()
            if category_name not in RSICD_CLASS_TO_ID:
                raise ValueError(
                    f"Unexpected RSICD class file: {file_name}; "
                    f"known classes={RSICD_CLASSES}"
                )

            file_path = os.path.join(category_class_dir, file_name)
            with open(file_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    image_name = line.strip()
                    if not image_name:
                        continue

                    key = _normalize_filename(image_name)
                    old = mapping.get(key)
                    if old is not None and old != category_name:
                        raise ValueError(
                            f"Duplicate class mapping: "
                            f"{key}: {old} vs {category_name}"
                        )
                    mapping[key] = category_name

    else:
        with open(category_map_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "mapping" in raw:
            raw = raw["mapping"]

        if not isinstance(raw, dict):
            raise ValueError(
                "category_map_file 必须是 filename -> category_name 字典。"
            )

        for image_name, category_name in raw.items():
            category_name = str(category_name).strip().lower()
            if category_name not in RSICD_CLASS_TO_ID:
                raise ValueError(
                    f"Unknown RSICD category: {category_name}"
                )
            mapping[_normalize_filename(image_name)] = category_name

    if not mapping:
        raise ValueError("RSICD category mapping is empty.")

    return mapping


def get_rsicd_category_id(image_reference, category_mapping=None):
    """
    优先使用 official filename->class mapping。

    若未提供 mapping，则保留旧文件名前缀解析作为兼容模式；
    此时纯数字文件名仍会得到 -1。
    """
    if category_mapping is not None:
        category_name = category_mapping.get(
            _normalize_filename(image_reference)
        )
        if category_name is None:
            return -1
        return RSICD_CLASS_TO_ID[category_name]

    stem = os.path.splitext(
        os.path.basename(str(image_reference))
    )[0].lower()

    if stem.isdigit():
        return -1

    prefix, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit() and prefix in RSICD_CLASS_TO_ID:
        return RSICD_CLASS_TO_ID[prefix]

    return -1


def get_rsicd_category_name(category_id):
    """category_id=-1 返回 unknown。"""
    if 0 <= int(category_id) < len(RSICD_CLASSES):
        return RSICD_CLASSES[int(category_id)]
    return "unknown"


def re_train_collate_fn(batch):
    """合并训练 batch，并压紧变长 Entity spans。"""
    (
        images,
        captions,
        image_ids,
        category_ids,
        sample_indices,
        spans,
    ) = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = list(captions)
    image_ids = torch.tensor(image_ids, dtype=torch.long)
    category_ids = torch.tensor(category_ids, dtype=torch.long)
    sample_indices = torch.tensor(sample_indices, dtype=torch.long)

    entity_counts = torch.tensor(
        [item.shape[0] for item in spans],
        dtype=torch.long,
    )

    if entity_counts.sum().item() > 0:
        entity_spans = torch.cat(spans, dim=0)
        entity_sample_ids = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.long),
            entity_counts,
        )
    else:
        entity_spans = torch.empty((0, 2), dtype=torch.long)
        entity_sample_ids = torch.empty(0, dtype=torch.long)

    return (
        images,
        captions,
        image_ids,
        category_ids,
        sample_indices,
        entity_spans,
        entity_sample_ids,
        entity_counts,
    )


class re_train_dataset(Dataset):
    """
    RSITR 训练集。

    训练样本保留：
        1. image_id：同一图像的多 caption 共享；
        2. category_id：Reliable Category 实验优先使用 released 31 类映射；
        3. sample_index：用于查询离线 frozen support cache；
        4. Entity spans：现有 Entity / contextual 分支继续使用。

    未配置 official mapping 时保留旧文件名解析，便于旧 baseline 兼容。
    """

    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
        entity_index_file=None,
        category_class_dir=None,
        category_map_file=None,
    ):
        super().__init__()

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.category_mapping = load_rsicd_category_mapping(
            category_class_dir=category_class_dir,
            category_map_file=category_map_file,
        )
        self.category_source = (
            "official_mapping"
            if self.category_mapping is not None
            else "legacy_filename"
        )

        ann_files = [ann_file] if isinstance(ann_file, str) else ann_file
        raw_ann = []

        for file_path in ann_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"Training annotation must be a list: {file_path}"
                )

            raw_ann.extend(data)

        if not raw_ann:
            raise ValueError("Training annotation is empty.")

        self.num_raw_pairs = len(raw_ann)
        self._load_entity_index(entity_index_file)

        # compact Entity index 依赖原始 pair 顺序，因此保留 _pair_index。
        self.ann = []

        for pair_index, ann in enumerate(raw_ann):
            if "caption" not in ann:
                raise KeyError(
                    f"Missing key 'caption' at training index {pair_index}"
                )
            if "image" not in ann:
                raise KeyError(
                    f"Missing key 'image' at training index {pair_index}"
                )

            try:
                caption = pre_caption(ann["caption"], self.max_words)
            except ValueError:
                continue

            item = dict(ann)
            item["caption"] = caption
            item["_pair_index"] = pair_index
            item["_category_id"] = get_rsicd_category_id(
                ann["image"],
                self.category_mapping,
            )

            if (
                self.category_mapping is not None
                and item["_category_id"] < 0
            ):
                raise KeyError(
                    f"Official category mapping missing image: "
                    f"{ann['image']}"
                )

            self.ann.append(item)

        if not self.ann:
            raise ValueError("Training annotation has no valid captions.")

        self.num_filtered_pairs = self.num_raw_pairs - len(self.ann)
        self._build_image_ids()
        self._build_category_stats()
        self._print_report()

    def _build_image_ids(self):
        """为同一图像的多条 caption 分配相同 image_id。"""
        image_to_id = {}
        self.image_ids = []
        self.image_references = []

        for ann in self.ann:
            image_key = ann["image"]
            if image_key not in image_to_id:
                image_to_id[image_key] = len(image_to_id)
                self.image_references.append(image_key)

            self.image_ids.append(image_to_id[image_key])

        self.num_images = len(image_to_id)

    def _build_category_stats(self):
        """统计类别覆盖，并构造与 image_id 对齐的 image_category_ids。"""
        self.category_ids = [
            int(ann["_category_id"])
            for ann in self.ann
        ]

        self.num_known_category_pairs = sum(
            category_id >= 0
            for category_id in self.category_ids
        )
        self.num_unknown_category_pairs = (
            len(self.category_ids) - self.num_known_category_pairs
        )

        self.image_category_ids = [-1] * self.num_images

        for sample_index, image_id in enumerate(self.image_ids):
            category_id = self.category_ids[sample_index]
            old = self.image_category_ids[image_id]

            if old >= 0 and category_id >= 0 and old != category_id:
                raise RuntimeError(
                    f"Same image has inconsistent categories: "
                    f"image_id={image_id}, {old} vs {category_id}"
                )

            if old < 0:
                self.image_category_ids[image_id] = category_id

        self.num_known_category_images = sum(
            category_id >= 0
            for category_id in self.image_category_ids
        )
        self.num_unknown_category_images = (
            self.num_images - self.num_known_category_images
        )

        if (
            self.category_mapping is not None
            and (
                self.num_unknown_category_pairs > 0
                or self.num_unknown_category_images > 0
            )
        ):
            raise RuntimeError(
                "Official category mapping enabled, but unresolved "
                "training categories still exist."
            )

    def _load_entity_index(self, entity_index_file):
        """读取 Entity span/text 索引；兼容旧 v1 span-only 格式。"""
        self.entity_index_file = entity_index_file

        self.pair_to_semantic = None
        self.semantic_offsets = None
        self.span_start = None
        self.span_end = None

        self.entity_vocab = None
        self.semantic_entity_offsets = None
        self.semantic_entity_ids = None

        self.entity_index_stats = {}
        self.entity_index_format = None

        if entity_index_file is None:
            return

        index = torch.load(
            entity_index_file,
            map_location="cpu",
            weights_only=True,
        )

        required = {
            "pair_to_semantic",
            "semantic_offsets",
            "span_start",
            "span_end",
        }
        missing = sorted(required - set(index))
        if missing:
            raise ValueError(f"Entity index missing keys: {missing}")

        self.pair_to_semantic = index["pair_to_semantic"]
        self.semantic_offsets = index["semantic_offsets"]
        self.span_start = index["span_start"]
        self.span_end = index["span_end"]

        self.entity_index_stats = index.get("statistics", {})
        metadata = index.get("metadata", {})
        self.entity_index_format = metadata.get(
            "format",
            "entity_span_index_v1",
        )

        if {
            "entity_vocab",
            "semantic_entity_offsets",
            "semantic_entity_ids",
        }.issubset(index):
            self.entity_vocab = index["entity_vocab"]
            self.semantic_entity_offsets = index[
                "semantic_entity_offsets"
            ]
            self.semantic_entity_ids = index["semantic_entity_ids"]

        if self.pair_to_semantic.ndim != 1:
            raise ValueError("pair_to_semantic must be 1D.")
        if len(self.pair_to_semantic) != self.num_raw_pairs:
            raise ValueError(
                "Entity index / annotation length mismatch: "
                f"{len(self.pair_to_semantic)} vs {self.num_raw_pairs}"
            )
        if self.semantic_offsets.ndim != 1:
            raise ValueError("semantic_offsets must be 1D.")
        if self.span_start.ndim != 1 or self.span_end.ndim != 1:
            raise ValueError("span_start/span_end must be 1D.")
        if len(self.span_start) != len(self.span_end):
            raise ValueError("span_start/span_end length mismatch.")

        index_max_words = metadata.get("max_words")
        if (
            index_max_words is not None
            and int(index_max_words) != self.max_words
        ):
            raise ValueError(
                f"max_words mismatch: dataset={self.max_words}, "
                f"entity_index={index_max_words}"
            )

    def _semantic_index(self, pair_index):
        if self.pair_to_semantic is None:
            return None

        semantic_index = int(
            self.pair_to_semantic[pair_index].item()
        )
        if semantic_index < 0:
            raise IndexError(
                f"Invalid semantic index: pair={pair_index}, "
                f"semantic={semantic_index}"
            )

        return semantic_index

    def _get_entity_spans_by_pair(self, pair_index):
        """返回当前 caption 的有效 Entity spans：[E, 2] 的 [start, end)。"""
        semantic_index = self._semantic_index(pair_index)
        if semantic_index is None:
            return torch.empty((0, 2), dtype=torch.long)

        if semantic_index + 1 >= len(self.semantic_offsets):
            raise IndexError(
                f"Invalid semantic offset index: {semantic_index}"
            )

        begin = int(
            self.semantic_offsets[semantic_index].item()
        )
        end = int(
            self.semantic_offsets[semantic_index + 1].item()
        )

        if begin == end:
            return torch.empty((0, 2), dtype=torch.long)

        starts = self.span_start[begin:end].long()
        ends = self.span_end[begin:end].long()
        return torch.stack((starts, ends), dim=1)

    def get_entity_texts(self, index):
        """
        返回第 index 条有效训练 caption 的完整 EAR Entity 文本。

        v2 index 示例：
            ["storage tanks", "pond", "buildings"]

        旧 v1 span-only index 无 Entity 文本，因此返回空列表。
        """
        if self.entity_vocab is None:
            return []

        pair_index = self.ann[index]["_pair_index"]
        semantic_index = self._semantic_index(pair_index)

        begin = int(
            self.semantic_entity_offsets[semantic_index].item()
        )
        end = int(
            self.semantic_entity_offsets[semantic_index + 1].item()
        )

        entity_ids = self.semantic_entity_ids[begin:end].tolist()
        return [
            self.entity_vocab[entity_id]
            for entity_id in entity_ids
        ]

    def get_category_id(self, index):
        """返回第 index 条有效训练 pair 的 RSICD category_id。"""
        return self.category_ids[index]

    def get_category_name(self, index):
        """返回第 index 条有效训练 pair 的 RSICD 类别名。"""
        return get_rsicd_category_name(
            self.category_ids[index]
        )

    def _print_report(self):
        print()
        print("=" * 72)
        print("Retrieval Training Dataset")
        print("=" * 72)
        print(f"Raw training pairs    : {self.num_raw_pairs}")
        print(f"Valid training pairs  : {len(self.ann)}")
        print(f"Filtered captions     : {self.num_filtered_pairs}")
        print(f"Unique training images: {self.num_images}")
        print(f"Category source       : {self.category_source}")
        print(f"Category groups       : {len(RSICD_CLASSES)}")
        print(
            f"Known category pairs  : "
            f"{self.num_known_category_pairs}"
        )
        print(
            f"Unknown category pairs: "
            f"{self.num_unknown_category_pairs}"
        )
        print(
            f"Known category images : "
            f"{self.num_known_category_images}"
        )
        print(
            f"Unknown category imgs : "
            f"{self.num_unknown_category_images}"
        )

        if self.pair_to_semantic is None:
            print("Entity index          : disabled")
        else:
            valid_entities = self.entity_index_stats.get(
                "valid_unique_entities",
                len(self.span_start),
            )
            print(
                f"Entity index format   : "
                f"{self.entity_index_format}"
            )
            print(
                f"Valid entity spans    : "
                f"{valid_entities}"
            )
            print(
                "Entity text lookup     : "
                f"{'enabled' if self.entity_vocab is not None else 'disabled'}"
            )

        print("=" * 72)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = resolve_image_path(
            self.image_root,
            ann["image"],
        )
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            ann["caption"],
            self.image_ids[index],
            self.category_ids[index],
            index,
            self._get_entity_spans_by_pair(
                ann["_pair_index"]
            ),
        )


class re_eval_dataset(Dataset):
    """验证 / 测试集使用标准检索评测，不引入 Entity 或类别监督。"""

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
            if "image" not in ann or "caption" not in ann:
                raise KeyError(
                    f"Invalid evaluation annotation at index {img_id}"
                )

            captions = ann["caption"]
            if not isinstance(captions, list):
                raise ValueError(
                    f"'caption' must be a list for sample {img_id}"
                )

            self.image.append(ann["image"])
            self.img2txt[img_id] = []

            for caption in captions:
                try:
                    caption = pre_caption(
                        caption,
                        self.max_words,
                    )
                except ValueError:
                    continue

                self.text.append(caption)
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
