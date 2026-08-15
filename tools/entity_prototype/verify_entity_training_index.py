import json
import random
from pathlib import Path


TRAIN_FILE = Path(
    "E:/paper3/data/rsicd/rsicd_train.json"
)

INDEX_FILE = Path(
    "E:/paper3/data/rsicd/entity_prototype/"
    "rsicd_entity_training_index.json"
)


def normalize(text):
    return " ".join(
        text.strip().lower().split()
    )


def load(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def get_train_records(data):

    if isinstance(data, list):
        return data

    for key in [
        "annotations",
        "samples",
        "data",
    ]:
        if (
            isinstance(data, dict)
            and isinstance(data.get(key), list)
        ):
            return data[key]

    raise ValueError(
        "Cannot locate train records."
    )


def get_caption(record):

    for key in [
        "caption",
        "text",
        "sentence",
    ]:

        value = record.get(key)

        if isinstance(value, str):
            return value

    raise ValueError(
        f"Cannot find caption in:\n{record}"
    )


train_data = load(TRAIN_FILE)
index_data = load(INDEX_FILE)

train_records = get_train_records(
    train_data
)

pair_to_semantic = index_data[
    "pair_to_semantic"
]

semantic_records = index_data[
    "semantic_records"
]


print("=" * 80)
print("Entity Training Index Verification")
print("=" * 80)

print(
    "Train pairs          :",
    len(train_records),
)

print(
    "pair_to_semantic     :",
    len(pair_to_semantic),
)

print(
    "Semantic records     :",
    len(semantic_records),
)


# ============================================================
# Basic length check
# ============================================================

assert (
    len(train_records)
    == len(pair_to_semantic)
), (
    "ERROR: train pair count does not "
    "match pair_to_semantic length."
)


# ============================================================
# Index range check
# ============================================================

invalid_indices = []

for pair_index, semantic_index in enumerate(
    pair_to_semantic
):

    if not (
        0
        <= semantic_index
        < len(semantic_records)
    ):
        invalid_indices.append(
            (
                pair_index,
                semantic_index,
            )
        )


print(
    "Invalid semantic idx :",
    len(invalid_indices),
)


# ============================================================
# FULL caption alignment check
# ============================================================

mismatches = []

for pair_index, train_record in enumerate(
    train_records
):

    train_caption = normalize(
        get_caption(train_record)
    )

    semantic_index = pair_to_semantic[
        pair_index
    ]

    semantic_caption = normalize(
        semantic_records[
            semantic_index
        ][
            "caption"
        ]
    )

    if (
        train_caption
        != semantic_caption
    ):

        mismatches.append(
            {
                "pair_index": pair_index,
                "semantic_index": (
                    semantic_index
                ),
                "train_caption": (
                    train_caption
                ),
                "semantic_caption": (
                    semantic_caption
                ),
            }
        )


print(
    "Caption mismatches   :",
    len(mismatches),
)

if mismatches:

    print()
    print("First mismatches:")

    for item in mismatches[:10]:
        print(item)


# ============================================================
# Check entity availability
# ============================================================

empty_entity_records = 0
total_entities = 0

for record in semantic_records:

    semantics = record[
        "semantics"
    ]

    entities = semantics.get(
        "entities",
        [],
    )

    total_entities += len(
        entities
    )

    if len(entities) == 0:
        empty_entity_records += 1


print(
    "Total entities       :",
    total_entities,
)

print(
    "Zero-entity captions :",
    empty_entity_records,
)


# ============================================================
# Random visual inspection
# ============================================================

print()
print("=" * 80)
print("Random Examples")
print("=" * 80)

random.seed(42)

sample_indices = random.sample(
    range(len(train_records)),
    k=min(
        10,
        len(train_records),
    ),
)

for pair_index in sample_indices:

    semantic_index = (
        pair_to_semantic[
            pair_index
        ]
    )

    record = semantic_records[
        semantic_index
    ]

    entities = record[
        "semantics"
    ].get(
        "entities",
        [],
    )

    entity_texts = [
        entity.get(
            "text",
            "",
        )
        for entity in entities
    ]

    print()
    print(
        f"pair_index     : "
        f"{pair_index}"
    )

    print(
        f"semantic_index : "
        f"{semantic_index}"
    )

    print(
        f"caption        : "
        f"{get_caption(train_records[pair_index])}"
    )

    print(
        f"entities       : "
        f"{entity_texts}"
    )


print()
print("=" * 80)

if (
    len(invalid_indices) == 0
    and len(mismatches) == 0
):

    print(
        "PASS: training index mapping is correct."
    )

else:

    print(
        "FAIL: training index mapping has errors."
    )

print("=" * 80)