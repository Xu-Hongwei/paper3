import argparse
import json
import random
from pathlib import Path

import open_clip
import torch

from datasets.utils import pre_caption


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def content_token_ids(tokenizer, text):
    """去掉 SOT/EOT/PAD，只保留文本内容 token。"""
    tokens = tokenizer([text])[0].tolist()
    nonzero = [token for token in tokens if token != 0]

    if len(nonzero) < 2:
        return []

    return nonzero[1:-1]


def validate_index(train_ann, index):
    required = {
        "pair_to_semantic",
        "semantic_offsets",
        "span_start",
        "span_end",
        "span_entity_ids",
        "entity_vocab",
        "semantic_entity_offsets",
        "semantic_entity_ids",
    }
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"Entity index missing keys: {missing}")

    pair_to_semantic = index["pair_to_semantic"]
    semantic_offsets = index["semantic_offsets"]
    semantic_entity_offsets = index["semantic_entity_offsets"]

    if len(pair_to_semantic) != len(train_ann):
        raise ValueError(
            "pair_to_semantic / train annotation length mismatch: "
            f"{len(pair_to_semantic)} vs {len(train_ann)}"
        )

    num_semantic = int(pair_to_semantic.max().item()) + 1

    if len(semantic_offsets) != num_semantic + 1:
        raise ValueError("semantic_offsets length mismatch.")
    if len(semantic_entity_offsets) != num_semantic + 1:
        raise ValueError("semantic_entity_offsets length mismatch.")

    if len(index["span_start"]) != len(index["span_end"]):
        raise ValueError("span_start/span_end length mismatch.")
    if len(index["span_start"]) != len(index["span_entity_ids"]):
        raise ValueError("span/span_entity_ids length mismatch.")

    vocab_size = len(index["entity_vocab"])
    if len(index["semantic_entity_ids"]) > 0:
        max_entity_id = int(index["semantic_entity_ids"].max().item())
        if max_entity_id >= vocab_size:
            raise ValueError("semantic_entity_ids contains invalid id.")

    if len(index["span_entity_ids"]) > 0:
        max_span_entity_id = int(index["span_entity_ids"].max().item())
        if max_span_entity_id >= vocab_size:
            raise ValueError("span_entity_ids contains invalid id.")


def inspect_pair(pair_index, train_ann, index, tokenizer, max_words):
    semantic_index = int(index["pair_to_semantic"][pair_index].item())
    entity_vocab = index["entity_vocab"]

    raw_caption = train_ann[pair_index]["caption"]
    caption = pre_caption(raw_caption, max_words)
    caption_tokens = tokenizer([caption])[0].tolist()

    entity_begin = int(index["semantic_entity_offsets"][semantic_index].item())
    entity_end = int(index["semantic_entity_offsets"][semantic_index + 1].item())
    entity_ids = index["semantic_entity_ids"][entity_begin:entity_end].tolist()
    entity_texts = [entity_vocab[entity_id] for entity_id in entity_ids]

    span_begin = int(index["semantic_offsets"][semantic_index].item())
    span_end = int(index["semantic_offsets"][semantic_index + 1].item())

    span_rows = []
    for pos in range(span_begin, span_end):
        start = int(index["span_start"][pos].item())
        end = int(index["span_end"][pos].item())
        entity_id = int(index["span_entity_ids"][pos].item())
        entity_text = entity_vocab[entity_id]

        expected = content_token_ids(
            tokenizer,
            pre_caption(entity_text, 100000),
        )
        actual = caption_tokens[start:end]
        matched = actual == expected

        span_rows.append(
            (entity_text, start, end, matched)
        )

    print("\n" + "=" * 72)
    print(f"pair_index     : {pair_index}")
    print(f"semantic_index : {semantic_index}")
    print(f"caption        : {caption}")
    print(f"entities       : {entity_texts}")

    if not span_rows:
        print("valid spans    : []")
        return

    print("valid spans:")
    for entity_text, start, end, matched in span_rows:
        flag = "OK" if matched else "MISMATCH"
        print(
            f"  - {entity_text!r:<30} "
            f"[{start:02d}, {end:02d})  {flag}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="检查 Entity index v2 的文本、span 和 caption 对应关系。"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-indices",
        type=int,
        nargs="*",
        default=None,
        help="指定原始 training pair index；不提供时随机抽样。",
    )
    args = parser.parse_args()

    train_ann = load_json(args.train_file)
    if not isinstance(train_ann, list):
        raise ValueError("Training annotation must be a list.")

    index = torch.load(
        args.index,
        map_location="cpu",
        weights_only=True,
    )
    validate_index(train_ann, index)

    metadata = index.get("metadata", {})
    model_name = metadata.get(
        "model_name",
        "ViT-B-32-quickgelu",
    )
    max_words = int(metadata.get("max_words", 30))
    tokenizer = open_clip.get_tokenizer(model_name)

    stats = index.get("statistics", {})

    print("=" * 72)
    print("Entity Index v2 Check")
    print("=" * 72)
    print(f"format          : {metadata.get('format', 'unknown')}")
    print(f"model           : {model_name}")
    print(f"max_words       : {max_words}")
    print(f"training pairs  : {len(train_ann)}")
    print(f"entity vocab    : {len(index['entity_vocab'])}")
    print(f"total entities  : {stats.get('total_entities', 'unknown')}")
    print(f"valid spans     : {len(index['span_start'])}")
    print("=" * 72)

    if args.pair_indices:
        pair_indices = args.pair_indices
    else:
        rng = random.Random(args.seed)
        count = min(args.num_samples, len(train_ann))
        pair_indices = rng.sample(range(len(train_ann)), count)

    for pair_index in pair_indices:
        if pair_index < 0 or pair_index >= len(train_ann):
            raise IndexError(f"Invalid pair index: {pair_index}")

        inspect_pair(
            pair_index,
            train_ann,
            index,
            tokenizer,
            max_words,
        )

    print("\nCheck finished.")


if __name__ == "__main__":
    main()
