import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Basic I/O
# ============================================================

def load_json(path: str) -> Any:

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"File not found: {path_obj}"
        )

    with path_obj.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: str,
) -> None:

    path_obj = Path(path)

    path_obj.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path_obj.open(
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
# Text helpers
# ============================================================

def normalize_caption(text: str) -> str:

    return " ".join(
        text.strip().lower().split()
    )


# ============================================================
# EAR sample readers
# ============================================================

def get_samples(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    samples = data.get(
        "samples"
    )

    if not isinstance(
        samples,
        list,
    ):
        raise ValueError(
            "EAR file does not contain a valid "
            "'samples' list."
        )

    return samples


def get_caption(
    sample: Dict[str, Any],
) -> str:

    possible_keys = [
        "caption",
        "text",
        "normalized_caption",
        "raw_caption",
        "sentence",
    ]

    for key in possible_keys:

        value = sample.get(key)

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if value:
                return value

    raise ValueError(
        "Unable to find caption text in EAR sample."
    )


def get_source_indices(
    sample: Dict[str, Any],
) -> List[int]:
    """
    source_indices should contain the indices of the original
    training pairs represented by this deduplicated caption.
    """

    value = sample.get(
        "source_indices"
    )

    if isinstance(
        value,
        list,
    ):

        result = []

        for item in value:

            if not isinstance(
                item,
                int,
            ):
                raise ValueError(
                    "source_indices contains "
                    "a non-integer value."
                )

            if item < 0:
                raise ValueError(
                    "source_indices contains "
                    "a negative index."
                )

            result.append(item)

        if result:
            return result

    # --------------------------------------------------------
    # Defensive fallback for a single source index.
    # --------------------------------------------------------

    value = sample.get(
        "source_index"
    )

    if isinstance(
        value,
        int,
    ):

        if value < 0:
            raise ValueError(
                "source_index must be >= 0."
            )

        return [value]

    raise ValueError(
        "EAR sample does not contain "
        "source_indices/source_index."
    )


def get_final_semantics(
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prefer the sanitized/final EAR result.

    This function does NOT modify the semantics.
    """

    possible_keys = [
        "sanitized_structured_semantics",
        "final_structured_semantics",
        "structured_semantics",
        "sanitized",
        "semantics",
    ]

    for key in possible_keys:

        value = sample.get(key)

        if isinstance(
            value,
            dict,
        ):

            if (
                "entities" in value
                or "relations" in value
                or "attributes" in value
            ):
                return value

    # --------------------------------------------------------
    # Some formats may store entities/relations directly
    # inside the sample.
    # --------------------------------------------------------

    if (
        "entities" in sample
        or "relations" in sample
        or "attributes" in sample
    ):

        return {
            key: sample[key]
            for key in (
                "entities",
                "attributes",
                "relations",
            )
            if key in sample
        }

    raise ValueError(
        "Unable to locate final structured semantics "
        "inside EAR sample."
    )


# ============================================================
# Semantic validation
# ============================================================

def validate_semantics(
    semantics: Dict[str, Any],
    semantic_index: int,
) -> Tuple[int, int, int]:

    entities = semantics.get(
        "entities",
        [],
    )

    relations = semantics.get(
        "relations",
        [],
    )

    top_level_attributes = semantics.get(
        "attributes",
        [],
    )

    if not isinstance(
        entities,
        list,
    ):
        raise ValueError(
            f"semantic_index={semantic_index}: "
            "'entities' must be a list."
        )

    if not isinstance(
        relations,
        list,
    ):
        raise ValueError(
            f"semantic_index={semantic_index}: "
            "'relations' must be a list."
        )

    if not isinstance(
        top_level_attributes,
        list,
    ):
        raise ValueError(
            f"semantic_index={semantic_index}: "
            "'attributes' must be a list when present."
        )

    nested_attribute_count = 0

    for entity_index, entity in enumerate(
        entities
    ):

        if not isinstance(
            entity,
            dict,
        ):
            raise ValueError(
                f"semantic_index={semantic_index}, "
                f"entity={entity_index}: "
                "entity must be a dictionary."
            )

        text = entity.get(
            "text"
        )

        if not isinstance(
            text,
            str,
        ) or not text.strip():

            raise ValueError(
                f"semantic_index={semantic_index}, "
                f"entity={entity_index}: "
                "entity text is missing."
            )

        attributes = entity.get(
            "attributes",
            [],
        )

        if attributes is None:
            attributes = []

        if not isinstance(
            attributes,
            list,
        ):
            raise ValueError(
                f"semantic_index={semantic_index}, "
                f"entity={entity_index}: "
                "attributes must be a list."
            )

        nested_attribute_count += len(
            attributes
        )

    attribute_count = max(
        nested_attribute_count,
        len(top_level_attributes),
    )

    return (
        len(entities),
        attribute_count,
        len(relations),
    )


# ============================================================
# Main index builder
# ============================================================

def build_training_index(
    ear_data: Dict[str, Any],
    expected_pairs: Optional[int],
) -> Dict[str, Any]:

    samples = get_samples(
        ear_data
    )

    semantic_records = []

    pair_assignment: Dict[
        int,
        int,
    ] = {}

    total_entities = 0
    total_attributes = 0
    total_relations = 0

    duplicate_caption_count = 0

    seen_normalized_captions = set()

    max_pair_index = -1

    # ========================================================
    # Iterate unique EAR samples
    # ========================================================

    for fallback_index, sample in enumerate(
        samples
    ):

        if not isinstance(
            sample,
            dict,
        ):
            raise ValueError(
                f"EAR sample {fallback_index} "
                "is not a dictionary."
            )

        caption = get_caption(
            sample
        )

        normalized_caption = normalize_caption(
            caption
        )

        if normalized_caption in (
            seen_normalized_captions
        ):
            duplicate_caption_count += 1

        seen_normalized_captions.add(
            normalized_caption
        )

        source_indices = sorted(
            set(
                get_source_indices(
                    sample
                )
            )
        )

        semantics = get_final_semantics(
            sample
        )

        semantic_index = len(
            semantic_records
        )

        (
            num_entities,
            num_attributes,
            num_relations,
        ) = validate_semantics(
            semantics=semantics,
            semantic_index=semantic_index,
        )

        total_entities += num_entities
        total_attributes += num_attributes
        total_relations += num_relations

        # ----------------------------------------------------
        # Preserve final EAR semantics exactly.
        # ----------------------------------------------------

        semantic_record = {
            "semantic_index": (
                semantic_index
            ),

            "caption": (
                caption
            ),

            "normalized_caption": (
                normalized_caption
            ),

            "source_indices": (
                source_indices
            ),

            "semantics": (
                semantics
            ),
        }

        semantic_records.append(
            semantic_record
        )

        # ----------------------------------------------------
        # Build pair -> semantic mapping
        # ----------------------------------------------------

        for pair_index in source_indices:

            if pair_index in pair_assignment:

                previous = pair_assignment[
                    pair_index
                ]

                raise ValueError(
                    "A training pair was assigned to "
                    "multiple semantic samples.\n"
                    f"pair_index={pair_index}\n"
                    f"previous_semantic={previous}\n"
                    f"current_semantic={semantic_index}"
                )

            pair_assignment[
                pair_index
            ] = semantic_index

            max_pair_index = max(
                max_pair_index,
                pair_index,
            )

    # ========================================================
    # Determine number of original training pairs
    # ========================================================

    if expected_pairs is not None:

        num_pairs = expected_pairs

        if max_pair_index >= num_pairs:

            raise ValueError(
                "source_indices exceed expected pair count.\n"
                f"max source index = {max_pair_index}\n"
                f"expected pairs   = {num_pairs}"
            )

    else:

        num_pairs = (
            max_pair_index + 1
        )

    # ========================================================
    # Dense pair -> semantic lookup
    # ========================================================

    pair_to_semantic = [
        -1
    ] * num_pairs

    for pair_index, semantic_index in (
        pair_assignment.items()
    ):

        pair_to_semantic[
            pair_index
        ] = semantic_index

    missing_pairs = [
        index
        for index, semantic_index in enumerate(
            pair_to_semantic
        )
        if semantic_index < 0
    ]

    # ========================================================
    # Strict coverage check
    # ========================================================

    if missing_pairs:

        preview = missing_pairs[
            :20
        ]

        raise ValueError(
            "EAR index does not cover every training pair.\n"
            f"Missing pairs: {len(missing_pairs)}\n"
            f"First missing indices: {preview}"
        )

    # ========================================================
    # Pair-count consistency
    # ========================================================

    mapped_pair_count = sum(
        len(
            record[
                "source_indices"
            ]
        )
        for record in semantic_records
    )

    if mapped_pair_count != num_pairs:

        raise ValueError(
            "Pair mapping count mismatch.\n"
            f"Mapped source indices = {mapped_pair_count}\n"
            f"Expected pair count   = {num_pairs}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    source_counts = [
        len(
            record[
                "source_indices"
            ]
        )
        for record in semantic_records
    ]

    max_occurrences = (
        max(source_counts)
        if source_counts
        else 0
    )

    min_occurrences = (
        min(source_counts)
        if source_counts
        else 0
    )

    avg_occurrences = (
        sum(source_counts)
        / max(
            len(source_counts),
            1,
        )
    )

    statistics = {
        "num_unique_semantic_samples": (
            len(
                semantic_records
            )
        ),

        "num_original_pairs": (
            num_pairs
        ),

        "num_mapped_pairs": (
            len(
                pair_assignment
            )
        ),

        "pair_coverage": (
            len(
                pair_assignment
            )
            / max(
                num_pairs,
                1,
            )
        ),

        "duplicate_normalized_caption_records": (
            duplicate_caption_count
        ),

        "total_entities": (
            total_entities
        ),

        "total_attributes": (
            total_attributes
        ),

        "total_relations": (
            total_relations
        ),

        "avg_entities_per_unique_caption": (
            total_entities
            / max(
                len(
                    semantic_records
                ),
                1,
            )
        ),

        "avg_attributes_per_unique_caption": (
            total_attributes
            / max(
                len(
                    semantic_records
                ),
                1,
            )
        ),

        "avg_relations_per_unique_caption": (
            total_relations
            / max(
                len(
                    semantic_records
                ),
                1,
            )
        ),

        "avg_original_pairs_per_unique_caption": (
            avg_occurrences
        ),

        "min_original_pairs_per_unique_caption": (
            min_occurrences
        ),

        "max_original_pairs_per_unique_caption": (
            max_occurrences
        ),
    }

    return {
        "metadata": {
            "format": (
                "ear_training_index_v1"
            ),

            "description": (
                "Direct training index from frozen "
                "LLM EAR structured semantics."
            ),

            "semantic_modification": (
                False
            ),

            "entity_normalization": (
                False
            ),

            "entity_filtering": (
                False
            ),

            "secondary_llm_processing": (
                False
            ),

            "pair_lookup_rule": (
                "pair_index -> semantic_index"
            ),
        },

        "statistics": (
            statistics
        ),

        "pair_to_semantic": (
            pair_to_semantic
        ),

        "semantic_records": (
            semantic_records
        ),
    }


# ============================================================
# Console report
# ============================================================

def print_report(
    result: Dict[str, Any],
) -> None:

    stats = result[
        "statistics"
    ]

    print()
    print("=" * 88)
    print("EAR Training Index Summary")
    print("=" * 88)

    print(
        f"Unique semantic captions : "
        f"{stats['num_unique_semantic_samples']}"
    )

    print(
        f"Original training pairs  : "
        f"{stats['num_original_pairs']}"
    )

    print(
        f"Mapped training pairs    : "
        f"{stats['num_mapped_pairs']}"
    )

    print(
        f"Pair coverage            : "
        f"{stats['pair_coverage'] * 100:.4f}%"
    )

    print()
    print(
        f"Total entities           : "
        f"{stats['total_entities']}"
    )

    print(
        f"Total attributes         : "
        f"{stats['total_attributes']}"
    )

    print(
        f"Total relations          : "
        f"{stats['total_relations']}"
    )

    print()
    print(
        f"Avg entities / caption   : "
        f"{stats['avg_entities_per_unique_caption']:.4f}"
    )

    print(
        f"Avg attrs / caption      : "
        f"{stats['avg_attributes_per_unique_caption']:.4f}"
    )

    print(
        f"Avg relations / caption  : "
        f"{stats['avg_relations_per_unique_caption']:.4f}"
    )

    print()
    print(
        f"Avg pair multiplicity    : "
        f"{stats['avg_original_pairs_per_unique_caption']:.4f}"
    )

    print(
        f"Min pair multiplicity    : "
        f"{stats['min_original_pairs_per_unique_caption']}"
    )

    print(
        f"Max pair multiplicity    : "
        f"{stats['max_original_pairs_per_unique_caption']}"
    )

    print("=" * 88)


# ============================================================
# Run
# ============================================================

def run(
    input_path: str,
    output_path: str,
    expected_pairs: Optional[int],
) -> None:

    print()
    print("=" * 88)
    print("Build EAR Training Index")
    print("=" * 88)

    print(
        f"EAR input      : "
        f"{input_path}"
    )

    print(
        f"Output         : "
        f"{output_path}"
    )

    if expected_pairs is not None:

        print(
            f"Expected pairs : "
            f"{expected_pairs}"
        )

    else:

        print(
            "Expected pairs : inferred"
        )

    print("=" * 88)

    ear_data = load_json(
        input_path
    )

    if not isinstance(
        ear_data,
        dict,
    ):

        raise ValueError(
            "Top-level EAR file "
            "must be a JSON object."
        )

    result = build_training_index(
        ear_data=ear_data,
        expected_pairs=expected_pairs,
    )

    save_json(
        result,
        output_path,
    )

    print_report(
        result
    )

    print()

    print(
        "Saved EAR training index:"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# CLI
# ============================================================

def build_arg_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Build a direct training-time lookup index "
            "from frozen EAR structured semantics."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=0,
        help=(
            "Expected number of original training pairs. "
            "Use 0 to infer from source_indices."
        ),
    )

    return parser


def main():

    parser = build_arg_parser()

    args = parser.parse_args()

    expected_pairs = (
        args.expected_pairs
        if args.expected_pairs > 0
        else None
    )

    run(
        input_path=args.input,
        output_path=args.output,
        expected_pairs=expected_pairs,
    )


if __name__ == "__main__":
    main()