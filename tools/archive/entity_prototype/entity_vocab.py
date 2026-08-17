import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Minimal lexical sanity filter
# ============================================================

# IMPORTANT:
# This is NOT an entity ontology.
# This is NOT semantic normalization.
#
# Only isolated words that are clearly impossible to serve as
# visual entity phrases are filtered.
#
# Do NOT add:
#   part
#   area
#   center
#   ground
#   land
#   water
#   sun
#   rock
#
# simply because they are semantically broad.
#
# They may still be valid visual entities / regions.
INVALID_ENTITY_TEXTS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "a",
    "an",
    "the",
}


# ============================================================
# Entity text normalization
# ============================================================

def normalize_entity_text(
    text: str,
) -> str:
    """
    Lightweight normalization for entity phrase identity.

    Operations:
    - strip
    - lowercase
    - collapse whitespace

    IMPORTANT:
    We intentionally DO NOT perform:
    - singular/plural merging
    - stemming
    - lemmatization
    - synonym replacement
    - semantic normalization

    Therefore:

        building != buildings
        road != roads
        house != houses

    Their semantic relationship will later be handled by
    CLIP embeddings + prototype learning.
    """

    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# Minimal lexical validity check
# ============================================================

def is_valid_entity_text(
    text: str,
) -> bool:
    """
    Conservative lexical sanity check.

    This function only removes entity strings that are
    obviously invalid at the lexical level.

    Examples removed:
        "."
        ","
        "..."
        "-"
        "is"
        "are"
        "the"

    Examples deliberately retained:
        "part"
        "area"
        "center"
        "ground"
        "sun"
        "rock"
        "green trees"
        "piece of river"

    This is NOT a semantic entity classifier.
    """

    if not isinstance(
        text,
        str,
    ):
        return False

    text = text.strip().lower()

    if not text:
        return False

    # --------------------------------------------------------
    # Pure punctuation / symbol strings
    #
    # Examples:
    #   "."
    #   ","
    #   "..."
    #   "-"
    #   "_"
    # --------------------------------------------------------

    if re.fullmatch(
        r"[\W_]+",
        text,
    ):
        return False

    # --------------------------------------------------------
    # Clearly invalid isolated function / copular words
    # --------------------------------------------------------

    if text in INVALID_ENTITY_TEXTS:
        return False

    # --------------------------------------------------------
    # Require at least one alphabetic character.
    #
    # Remote-sensing entity phrases in RSICD are English.
    # --------------------------------------------------------

    if not re.search(
        r"[a-zA-Z]",
        text,
    ):
        return False

    return True


# ============================================================
# JSON loading
# ============================================================

def load_json(
    input_path: str,
) -> Dict[str, Any]:

    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Input structured-semantics file "
            "must contain a top-level JSON object."
        )

    return data


# ============================================================
# JSON saving
# ============================================================

def save_json(
    data: Dict[str, Any],
    output_path: str,
) -> None:

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Presence helpers
# ============================================================

def is_explicitly_absent(
    entity: Dict[str, Any],
) -> bool:
    """
    Check whether an entity is explicitly marked as absent.

    Example:

        {
            "text": "trees",
            "attributes": [
                {
                    "type": "presence",
                    "value": "absent"
                }
            ]
        }

    Such entity phrases are still retained in the vocabulary.

    However, later Entity-Patch Grounding should normally
    ignore explicitly absent entity occurrences.
    """

    attributes = entity.get(
        "attributes",
        [],
    )

    if not isinstance(
        attributes,
        list,
    ):
        return False

    for attribute in attributes:

        if not isinstance(
            attribute,
            dict,
        ):
            continue

        attr_type = attribute.get(
            "type",
            "",
        )

        attr_value = attribute.get(
            "value",
            "",
        )

        if not isinstance(
            attr_type,
            str,
        ):
            continue

        if not isinstance(
            attr_value,
            str,
        ):
            continue

        attr_type = (
            attr_type
            .strip()
            .lower()
        )

        attr_value = (
            attr_value
            .strip()
            .lower()
        )

        if (
            attr_type == "presence"
            and attr_value == "absent"
        ):
            return True

    return False


# ============================================================
# Original pair occurrence count
# ============================================================

def get_pair_occurrence_count(
    sample: Dict[str, Any],
) -> int:
    """
    Get how many original image-caption pairs are represented
    by one unique-caption extraction sample.

    Preferred source:
        source_indices

    Fallback:
        num_occurrences

    Final fallback:
        1
    """

    source_indices = sample.get(
        "source_indices"
    )

    if isinstance(
        source_indices,
        list,
    ):

        if len(
            source_indices
        ) > 0:

            return len(
                source_indices
            )

    num_occurrences = sample.get(
        "num_occurrences"
    )

    if isinstance(
        num_occurrences,
        int,
    ):

        if num_occurrences > 0:

            return (
                num_occurrences
            )

    return 1


# ============================================================
# Semantic source
# ============================================================

def get_semantics(
    sample: Dict[str, Any],
    semantic_source: str,
) -> Dict[str, Any]:
    """
    Read structured semantics from one sample.

    Recommended:
        semantic_source = "sanitized"
    """

    if semantic_source == "sanitized":

        semantics = sample.get(
            "sanitized_structured_semantics"
        )

    elif semantic_source == "raw":

        semantics = sample.get(
            "raw_structured_semantics"
        )

    else:

        raise ValueError(
            f"Unknown semantic source: "
            f"{semantic_source}"
        )

    if not isinstance(
        semantics,
        dict,
    ):
        return {}

    return semantics


# ============================================================
# Build entity vocabulary
# ============================================================

def build_entity_vocab(
    data: Dict[str, Any],
    semantic_source: str = "sanitized",
    max_examples: int = 3,
) -> Dict[str, Any]:
    """
    Build open-vocabulary entity dictionary.

    Three frequencies are distinguished:

    ----------------------------------------------------------
    1. entity_instance_count
    ----------------------------------------------------------

    Number of entity instances appearing in unique-caption
    EAR annotations.

    ----------------------------------------------------------
    2. unique_caption_count
    ----------------------------------------------------------

    Number of UNIQUE captions containing the phrase.

    Repeated identical entity text inside one caption counts
    only once.

    ----------------------------------------------------------
    3. pair_count
    ----------------------------------------------------------

    Number of ORIGINAL image-caption training pairs whose
    captions contain the entity phrase.

    Example:

        one unique caption occurs in 156 original pairs

    then:

        unique_caption_count += 1
        pair_count += 156

    ----------------------------------------------------------

    Prototype initialization will later encode each unique
    entity phrase exactly once.
    """

    samples = data.get(
        "samples",
        [],
    )

    if not isinstance(
        samples,
        list,
    ):
        raise ValueError(
            "Input file does not contain "
            "a valid 'samples' list."
        )

    # ========================================================
    # Vocabulary
    # ========================================================

    vocab: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # ========================================================
    # Global statistics
    # ========================================================

    # Raw valid entity instances before lexical filtering.
    raw_entity_instances = 0

    # Entity instances retained after lexical filtering.
    retained_entity_instances = 0

    valid_samples = 0

    captions_with_entities = 0

    # Captions with no retained entity after filtering.
    zero_entity_captions = 0

    malformed_entity_count = 0
    empty_entity_text_count = 0

    duplicate_entity_text_caption_count = 0

    explicit_absent_entity_instances = 0

    # ========================================================
    # Lexical filtering statistics
    # ========================================================

    invalid_lexical_entity_instances = 0

    invalid_lexical_counter = Counter()

    invalid_lexical_examples: Dict[
        str,
        List[str],
    ] = {}

    # ========================================================
    # Iterate unique captions
    # ========================================================

    for fallback_index, sample in enumerate(
        samples
    ):

        if not isinstance(
            sample,
            dict,
        ):
            continue

        semantics = get_semantics(
            sample=sample,
            semantic_source=semantic_source,
        )

        if not semantics:
            continue

        valid_samples += 1

        entities = semantics.get(
            "entities",
            [],
        )

        if not isinstance(
            entities,
            list,
        ):
            entities = []

        caption = sample.get(
            "caption",
            "",
        )

        if not isinstance(
            caption,
            str,
        ):
            caption = ""

        sample_index = sample.get(
            "sample_index"
        )

        if not isinstance(
            sample_index,
            int,
        ):

            sample_index = (
                fallback_index
            )

        pair_occurrences = (
            get_pair_occurrence_count(
                sample
            )
        )

        # ----------------------------------------------------
        # Phrases already counted in this unique caption.
        # ----------------------------------------------------

        seen_in_caption = set()

        caption_has_entity = False
        caption_has_duplicate = False

        # ====================================================
        # Entity instances
        # ====================================================

        for entity in entities:

            if not isinstance(
                entity,
                dict,
            ):

                malformed_entity_count += 1

                continue

            text = entity.get(
                "text"
            )

            if not isinstance(
                text,
                str,
            ):

                malformed_entity_count += 1

                continue

            normalized_text = (
                normalize_entity_text(
                    text
                )
            )

            if not normalized_text:

                empty_entity_text_count += 1

                continue

            # ------------------------------------------------
            # This entity existed in the sanitized EAR.
            # ------------------------------------------------

            raw_entity_instances += 1

            # =================================================
            # Minimal lexical sanity check
            # =================================================

            if not is_valid_entity_text(
                normalized_text
            ):

                invalid_lexical_entity_instances += 1

                invalid_lexical_counter[
                    normalized_text
                ] += 1

                if (
                    normalized_text
                    not in invalid_lexical_examples
                ):

                    invalid_lexical_examples[
                        normalized_text
                    ] = []

                examples = (
                    invalid_lexical_examples[
                        normalized_text
                    ]
                )

                if (
                    caption
                    and len(examples)
                    < max_examples
                ):

                    examples.append(
                        caption
                    )

                continue

            # ------------------------------------------------
            # Retained valid entity.
            # ------------------------------------------------

            caption_has_entity = True

            retained_entity_instances += 1

            absent = (
                is_explicitly_absent(
                    entity
                )
            )

            if absent:

                explicit_absent_entity_instances += 1

            # =================================================
            # Create vocabulary record
            # =================================================

            if normalized_text not in vocab:

                vocab[
                    normalized_text
                ] = {

                    "text": (
                        normalized_text
                    ),

                    "entity_instance_count": 0,

                    "unique_caption_count": 0,

                    "pair_count": 0,

                    "explicit_absent_instance_count": 0,

                    "caption_indices": [],

                    "example_captions": [],
                }

            record = vocab[
                normalized_text
            ]

            # =================================================
            # Instance-level count
            # =================================================

            record[
                "entity_instance_count"
            ] += 1

            if absent:

                record[
                    "explicit_absent_instance_count"
                ] += 1

            # =================================================
            # Caption-level count
            # =================================================

            if (
                normalized_text
                in seen_in_caption
            ):

                caption_has_duplicate = True

                # Do not increase:
                # unique_caption_count
                # pair_count
                # caption_indices

                continue

            seen_in_caption.add(
                normalized_text
            )

            record[
                "unique_caption_count"
            ] += 1

            record[
                "pair_count"
            ] += pair_occurrences

            record[
                "caption_indices"
            ].append(
                sample_index
            )

            # =================================================
            # Example captions
            # =================================================

            if (
                len(
                    record[
                        "example_captions"
                    ]
                )
                < max_examples
            ):

                if caption:

                    record[
                        "example_captions"
                    ].append(
                        caption
                    )

        # ====================================================
        # Caption statistics after lexical filtering
        # ====================================================

        if caption_has_entity:

            captions_with_entities += 1

        else:

            zero_entity_captions += 1

        if caption_has_duplicate:

            duplicate_entity_text_caption_count += 1

    # ========================================================
    # Sort vocabulary
    # ========================================================

    sorted_records = sorted(
        vocab.values(),

        key=lambda item: (
            -item[
                "unique_caption_count"
            ],
            -item[
                "entity_instance_count"
            ],
            item[
                "text"
            ],
        ),
    )

    # ========================================================
    # Stable entity indices
    # ========================================================

    for entity_index, record in enumerate(
        sorted_records
    ):

        record[
            "entity_index"
        ] = entity_index

    # ========================================================
    # Reorder output fields
    # ========================================================

    final_records = []

    for record in sorted_records:

        final_record = {

            "entity_index": (
                record[
                    "entity_index"
                ]
            ),

            "text": (
                record[
                    "text"
                ]
            ),

            "entity_instance_count": (
                record[
                    "entity_instance_count"
                ]
            ),

            "unique_caption_count": (
                record[
                    "unique_caption_count"
                ]
            ),

            "pair_count": (
                record[
                    "pair_count"
                ]
            ),

            "explicit_absent_instance_count": (
                record[
                    "explicit_absent_instance_count"
                ]
            ),

            "caption_indices": (
                record[
                    "caption_indices"
                ]
            ),

            "example_captions": (
                record[
                    "example_captions"
                ]
            ),
        }

        final_records.append(
            final_record
        )

    # ========================================================
    # Singleton statistics
    # ========================================================

    singleton_by_caption_count = sum(
        1
        for record in final_records
        if record[
            "unique_caption_count"
        ] == 1
    )

    singleton_by_instance_count = sum(
        1
        for record in final_records
        if record[
            "entity_instance_count"
        ] == 1
    )

    if len(
        final_records
    ) > 0:

        singleton_caption_ratio = (
            singleton_by_caption_count
            / len(final_records)
        )

        singleton_instance_ratio = (
            singleton_by_instance_count
            / len(final_records)
        )

    else:

        singleton_caption_ratio = 0.0
        singleton_instance_ratio = 0.0

    # ========================================================
    # Frequency distributions
    # ========================================================

    unique_caption_frequency_histogram = (
        Counter(
            record[
                "unique_caption_count"
            ]
            for record in final_records
        )
    )

    instance_frequency_histogram = (
        Counter(
            record[
                "entity_instance_count"
            ]
            for record in final_records
        )
    )

    # ========================================================
    # Filter report
    # ========================================================

    filtered_invalid_entities = []

    for text, count in sorted(
        invalid_lexical_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):

        filtered_invalid_entities.append(
            {
                "text": text,

                "instance_count": (
                    count
                ),

                "example_captions": (
                    invalid_lexical_examples.get(
                        text,
                        [],
                    )
                ),
            }
        )

    # ========================================================
    # Source metadata
    # ========================================================

    source_metadata = data.get(
        "metadata",
        {},
    )

    if not isinstance(
        source_metadata,
        dict,
    ):

        source_metadata = {}

    # ========================================================
    # Output
    # ========================================================

    output = {

        "metadata": {

            "dataset": (
                source_metadata.get(
                    "dataset"
                )
            ),

            "source_file": (
                source_metadata.get(
                    "input_file"
                )
            ),

            "semantic_source": (
                semantic_source
            ),

            "total_original_pairs": (
                source_metadata.get(
                    "total_pairs"
                )
            ),

            "total_unique_captions": (
                source_metadata.get(
                    "unique_captions",
                    len(samples),
                )
            ),

            "entity_normalization": {

                "lowercase": True,

                "strip": True,

                "collapse_whitespace": True,

                "lemmatization": False,

                "stemming": False,

                "singular_plural_merge": False,

                "synonym_merge": False,

                "semantic_normalization": False,
            },

            "lexical_filter": {

                "enabled": True,

                "strategy": (
                    "minimal_conservative_sanity_filter"
                ),

                "pure_punctuation_removed": True,

                "invalid_exact_strings": sorted(
                    INVALID_ENTITY_TEXTS
                ),

                "semantic_entity_classification": False,

                "ontology_mapping": False,
            },

            "frequency_definition": {

                "entity_instance_count": (
                    "Number of retained EAR entity instances "
                    "in unique captions."
                ),

                "unique_caption_count": (
                    "Number of unique captions containing "
                    "this retained entity phrase."
                ),

                "pair_count": (
                    "Number of original image-caption pairs "
                    "represented by those unique captions."
                ),
            },
        },

        "statistics": {

            "processed_semantic_samples": (
                valid_samples
            ),

            "captions_with_entities": (
                captions_with_entities
            ),

            "zero_entity_captions": (
                zero_entity_captions
            ),

            # ---------------------------------------------
            # Before / after lexical filtering
            # ---------------------------------------------

            "raw_entity_instances": (
                raw_entity_instances
            ),

            "retained_entity_instances": (
                retained_entity_instances
            ),

            # Keep old-compatible name.
            "total_entity_instances": (
                retained_entity_instances
            ),

            "invalid_lexical_entity_instances": (
                invalid_lexical_entity_instances
            ),

            "invalid_lexical_unique_texts": (
                len(
                    invalid_lexical_counter
                )
            ),

            "unique_entity_phrases": (
                len(final_records)
            ),

            # ---------------------------------------------
            # Two singleton definitions
            # ---------------------------------------------

            "singleton_by_caption_count": (
                singleton_by_caption_count
            ),

            "singleton_by_caption_ratio_percent": round(
                singleton_caption_ratio
                * 100.0,
                4,
            ),

            "singleton_by_instance_count": (
                singleton_by_instance_count
            ),

            "singleton_by_instance_ratio_percent": round(
                singleton_instance_ratio
                * 100.0,
                4,
            ),

            # Backwards-compatible interpretation:
            # singleton = unique_caption_count == 1
            "singleton_phrases": (
                singleton_by_caption_count
            ),

            "singleton_ratio_percent": round(
                singleton_caption_ratio
                * 100.0,
                4,
            ),

            "duplicate_entity_text_captions": (
                duplicate_entity_text_caption_count
            ),

            "explicit_absent_entity_instances": (
                explicit_absent_entity_instances
            ),

            "malformed_entity_count": (
                malformed_entity_count
            ),

            "empty_entity_text_count": (
                empty_entity_text_count
            ),

            "unique_caption_frequency_histogram": {
                str(key): value
                for key, value in sorted(
                    unique_caption_frequency_histogram.items()
                )
            },

            "instance_frequency_histogram": {
                str(key): value
                for key, value in sorted(
                    instance_frequency_histogram.items()
                )
            },
        },

        # ====================================================
        # Explicit report of everything removed
        # ====================================================

        "filtered_invalid_entities": (
            filtered_invalid_entities
        ),

        "entities": (
            final_records
        ),
    }

    return output


# ============================================================
# Console preview
# ============================================================

def print_summary(
    vocab_data: Dict[str, Any],
    top_k: int,
) -> None:

    stats = vocab_data[
        "statistics"
    ]

    entities = vocab_data[
        "entities"
    ]

    invalid_entities = vocab_data.get(
        "filtered_invalid_entities",
        [],
    )

    print()
    print("=" * 72)
    print("Entity Vocabulary Summary")
    print("=" * 72)

    print(
        f"Processed captions        : "
        f"{stats['processed_semantic_samples']}"
    )

    print(
        f"Captions with entities    : "
        f"{stats['captions_with_entities']}"
    )

    print(
        f"Zero-entity captions      : "
        f"{stats['zero_entity_captions']}"
    )

    print()

    print(
        f"Raw entity instances      : "
        f"{stats['raw_entity_instances']}"
    )

    print(
        f"Filtered instances        : "
        f"{stats['invalid_lexical_entity_instances']}"
    )

    print(
        f"Retained entity instances : "
        f"{stats['retained_entity_instances']}"
    )

    print(
        f"Unique entity phrases     : "
        f"{stats['unique_entity_phrases']}"
    )

    print()

    print(
        f"Singleton by caption      : "
        f"{stats['singleton_by_caption_count']}"
    )

    print(
        f"Caption singleton ratio   : "
        f"{stats['singleton_by_caption_ratio_percent']:.4f}%"
    )

    print(
        f"Singleton by instance     : "
        f"{stats['singleton_by_instance_count']}"
    )

    print(
        f"Instance singleton ratio  : "
        f"{stats['singleton_by_instance_ratio_percent']:.4f}%"
    )

    print()

    print(
        f"Duplicate-text captions   : "
        f"{stats['duplicate_entity_text_captions']}"
    )

    print(
        f"Explicit absent instances : "
        f"{stats['explicit_absent_entity_instances']}"
    )

    print(
        f"Malformed entities        : "
        f"{stats['malformed_entity_count']}"
    )

    print(
        f"Empty entity texts        : "
        f"{stats['empty_entity_text_count']}"
    )

    # ========================================================
    # Filter report
    # ========================================================

    print()
    print("-" * 72)
    print("Filtered Lexically Invalid Entities")
    print("-" * 72)

    if not invalid_entities:

        print(
            "None"
        )

    else:

        for item in invalid_entities:

            print(
                f"{item['text']!r:<20} "
                f"instances="
                f"{item['instance_count']}"
            )

            for example in item.get(
                "example_captions",
                [],
            ):

                print(
                    f"    example: "
                    f"{example}"
                )

    # ========================================================
    # Top entities
    # ========================================================

    print()
    print("-" * 72)

    print(
        f"Top {min(top_k, len(entities))} "
        f"entity phrases"
    )

    print("-" * 72)

    for record in entities[
        :top_k
    ]:

        print(
            f"[{record['entity_index']:04d}] "
            f"{record['text']:<30} "
            f"instance="
            f"{record['entity_instance_count']:<6} "
            f"caption="
            f"{record['unique_caption_count']:<6} "
            f"pair="
            f"{record['pair_count']:<6} "
            f"absent="
            f"{record['explicit_absent_instance_count']}"
        )

    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Build an open-vocabulary entity dictionary "
            "from sanitized structured semantics."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Full structured-semantics JSON file."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output entity vocabulary JSON."
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default="sanitized",
        choices=[
            "sanitized",
            "raw",
        ],
        help=(
            "Structured semantic source. "
            "Use sanitized for training."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help=(
            "Maximum example captions saved "
            "for each entity phrase."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help=(
            "Number of entity phrases printed "
            "to console."
        ),
    )

    return parser


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    data = load_json(
        args.input
    )

    vocab_data = build_entity_vocab(
        data=data,
        semantic_source=args.source,
        max_examples=args.max_examples,
    )

    save_json(
        data=vocab_data,
        output_path=args.output,
    )

    print_summary(
        vocab_data=vocab_data,
        top_k=args.top_k,
    )

    print()

    print(
        f"Saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()