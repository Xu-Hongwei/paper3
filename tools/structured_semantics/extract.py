import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from .llm_client import LLMClient
from .sanitizer import sanitize_structured_semantics
from .validator import validate_structured_semantics


# ============================================================
# Version information
# ============================================================

PROMPT_VERSION = "v3.0-open"
SCHEMA_VERSION = "v1"
SANITIZER_VERSION = "v1"


# ============================================================
# Text normalization
# ============================================================

def normalize_caption(text: str) -> str:
    """
    Normalize caption only for duplicate detection.

    IMPORTANT:
    The original caption is still sent to the LLM.

    We intentionally keep this normalization lightweight:
    - lowercase
    - strip
    - collapse whitespace

    No semantic rewriting is performed.
    """

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(
    input_path: str,
) -> List[Dict[str, Any]]:
    """
    Load dataset annotation JSON.

    Expected top-level format:
        [
            {...},
            {...}
        ]

    Supported caption fields:
        caption
        text
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must contain a top-level list."
        )

    return data


# ============================================================
# Caption extraction
# ============================================================

def get_caption(
    sample: Dict[str, Any],
) -> str:
    """
    Extract caption text from one dataset sample.
    """

    caption = sample.get("caption")

    if (
        not isinstance(caption, str)
        or not caption.strip()
    ):
        caption = sample.get("text")

    if (
        not isinstance(caption, str)
        or not caption.strip()
    ):
        return ""

    return caption.strip()


# ============================================================
# Unique caption pool
# ============================================================

def build_unique_caption_pool(
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate captions before API calls.

    Each unique normalized caption keeps the metadata from the
    first occurrence.

    This is ONLY API-call deduplication.

    It does NOT modify the original training dataset.
    """

    seen = set()

    unique_samples: List[
        Dict[str, Any]
    ] = []

    for source_index, sample in enumerate(
        dataset
    ):

        if not isinstance(sample, dict):
            continue

        caption = get_caption(sample)

        if not caption:
            continue

        normalized = normalize_caption(
            caption
        )

        if normalized in seen:
            continue

        seen.add(normalized)

        unique_samples.append(
            {
                "_source_index": source_index,
                "_normalized_caption": (
                    normalized
                ),

                "caption": caption,

                # Preserve useful metadata.
                "image_id": sample.get(
                    "image_id"
                ),
                "image": sample.get(
                    "image"
                ),
                "label_name": sample.get(
                    "label_name"
                ),
                "label": sample.get(
                    "label"
                ),
            }
        )

    return unique_samples


# ============================================================
# Sampling
# ============================================================

def select_samples(
    unique_samples: List[Dict[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Select a reproducible random subset of UNIQUE captions.

    limit <= 0:
        use all unique captions.
    """

    if (
        limit <= 0
        or limit >= len(unique_samples)
    ):
        return list(unique_samples)

    rng = random.Random(seed)

    selected = rng.sample(
        unique_samples,
        limit,
    )

    return selected


# ============================================================
# Output helpers
# ============================================================

def save_json(
    data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save JSON output safely.
    """

    path = Path(output_path)

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
# Main extraction
# ============================================================

def run_extraction(
    input_path: str,
    output_path: str,
    limit: int,
    seed: int,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> None:

    # ========================================================
    # 1. Load dataset
    # ========================================================

    dataset = load_dataset(
        input_path
    )

    unique_samples = (
        build_unique_caption_pool(
            dataset
        )
    )

    selected_samples = select_samples(
        unique_samples=unique_samples,
        limit=limit,
        seed=seed,
    )

    print()
    print("=" * 70)
    print("Structured Semantics Extraction")
    print("=" * 70)

    print(
        f"Input pairs         : "
        f"{len(dataset)}"
    )

    print(
        f"Unique captions     : "
        f"{len(unique_samples)}"
    )

    print(
        f"Selected captions   : "
        f"{len(selected_samples)}"
    )

    print(
        f"Prompt version      : "
        f"{PROMPT_VERSION}"
    )

    print(
        f"Schema version      : "
        f"{SCHEMA_VERSION}"
    )

    print(
        f"Sanitizer version   : "
        f"{SANITIZER_VERSION}"
    )

    print(
        f"Model               : "
        f"{model}"
    )

    print("=" * 70)
    print()

    # ========================================================
    # 2. Initialize LLM
    # ========================================================

    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )

    # ========================================================
    # 3. Global counters
    # ========================================================

    api_success_count = 0
    api_failure_count = 0

    raw_validation_valid_count = 0
    raw_validation_invalid_count = 0

    sanitized_validation_valid_count = 0
    sanitized_validation_invalid_count = 0

    warning_sample_count = 0
    error_sample_count = 0

    total_warnings = 0
    total_validation_errors = 0

    sanitizer_changed_sample_count = 0
    sanitizer_unchanged_sample_count = 0

    total_sanitizer_actions = 0

    total_dropped_entities = 0
    total_dropped_attributes = 0
    total_dropped_relations = 0
    total_reassigned_entity_ids = 0

    # Final sanitized semantic statistics.
    total_entities = 0
    total_attributes = 0
    total_relations = 0

    results: List[
        Dict[str, Any]
    ] = []

    # ========================================================
    # 4. Extraction loop
    # ========================================================

    for sample_index, sample in enumerate(
        selected_samples
    ):

        caption = sample["caption"]

        print(
            f"[{sample_index + 1:04d}/"
            f"{len(selected_samples):04d}] "
            f"{caption}"
        )

        result_item: Dict[str, Any] = {
            "sample_index": sample_index,

            "source_index": sample.get(
                "_source_index"
            ),

            "image_id": sample.get(
                "image_id"
            ),

            "image": sample.get(
                "image"
            ),

            "label_name": sample.get(
                "label_name"
            ),

            "label": sample.get(
                "label"
            ),

            "caption": caption,

            "normalized_caption": (
                sample.get(
                    "_normalized_caption"
                )
            ),
        }

        # ====================================================
        # 4.1 LLM extraction
        # ====================================================

        try:
            raw_semantics = client.extract(
                caption
            )

            api_success_count += 1

            result_item[
                "api_success"
            ] = True

        except Exception as exc:

            api_failure_count += 1

            result_item[
                "api_success"
            ] = False

            result_item[
                "api_error"
            ] = str(exc)

            print(
                f"    API ERROR: {exc}"
            )

            results.append(
                result_item
            )

            continue

        # ====================================================
        # 4.2 Preserve RAW EAR
        # ====================================================

        result_item[
            "raw_structured_semantics"
        ] = raw_semantics

        # ----------------------------------------------------
        # Optional RAW structural validation.
        #
        # This allows us to distinguish:
        #
        # LLM raw quality
        # vs
        # sanitizer-repaired quality
        # ----------------------------------------------------

        raw_validation = (
            validate_structured_semantics(
                raw_semantics
            )
        )

        result_item[
            "raw_validation"
        ] = raw_validation

        if raw_validation["valid"]:
            raw_validation_valid_count += 1
        else:
            raw_validation_invalid_count += 1

        # ====================================================
        # 4.3 Structural sanitization
        # ====================================================

        sanitization_result = (
            sanitize_structured_semantics(
                raw_semantics
            )
        )

        sanitized_semantics = (
            sanitization_result[
                "sanitized"
            ]
        )

        sanitization_report = (
            sanitization_result[
                "report"
            ]
        )

        result_item[
            "sanitized_structured_semantics"
        ] = sanitized_semantics

        result_item[
            "sanitization"
        ] = sanitization_report

        # ----------------------------------------------------
        # Sanitizer statistics
        # ----------------------------------------------------

        if sanitization_report[
            "changed"
        ]:
            sanitizer_changed_sample_count += 1
        else:
            sanitizer_unchanged_sample_count += 1

        sanitizer_actions = (
            sanitization_report.get(
                "actions",
                [],
            )
        )

        total_sanitizer_actions += len(
            sanitizer_actions
        )

        sanitizer_stats = (
            sanitization_report.get(
                "stats",
                {},
            )
        )

        total_dropped_entities += (
            sanitizer_stats.get(
                "dropped_entities",
                0,
            )
        )

        total_dropped_attributes += (
            sanitizer_stats.get(
                "dropped_attributes",
                0,
            )
        )

        total_dropped_relations += (
            sanitizer_stats.get(
                "dropped_relations",
                0,
            )
        )

        total_reassigned_entity_ids += (
            sanitizer_stats.get(
                "reassigned_entity_ids",
                0,
            )
        )

        # ====================================================
        # 4.4 Validate sanitized EAR
        # ====================================================

        validation = (
            validate_structured_semantics(
                sanitized_semantics
            )
        )

        result_item[
            "validation"
        ] = validation

        if validation["valid"]:
            (
                sanitized_validation_valid_count
            ) += 1
        else:
            (
                sanitized_validation_invalid_count
            ) += 1

        if validation["warnings"]:
            warning_sample_count += 1

        if validation["errors"]:
            error_sample_count += 1

        total_warnings += len(
            validation["warnings"]
        )

        total_validation_errors += len(
            validation["errors"]
        )

        # ====================================================
        # 4.5 Semantic statistics
        # ====================================================

        stats = validation[
            "stats"
        ]

        total_entities += (
            stats[
                "num_entities"
            ]
        )

        total_attributes += (
            stats[
                "num_attributes"
            ]
        )

        total_relations += (
            stats[
                "num_relations"
            ]
        )

        # ====================================================
        # 4.6 Console diagnostics
        # ====================================================

        raw_status = (
            "VALID"
            if raw_validation[
                "valid"
            ]
            else "INVALID"
        )

        final_status = (
            "VALID"
            if validation[
                "valid"
            ]
            else "INVALID"
        )

        sanitize_status = (
            "changed"
            if sanitization_report[
                "changed"
            ]
            else "unchanged"
        )

        print(
            f"    Raw validation : "
            f"{raw_status}"
        )

        print(
            f"    Sanitizer      : "
            f"{sanitize_status} "
            f"({len(sanitizer_actions)} actions)"
        )

        print(
            f"    Final validation: "
            f"{final_status}"
        )

        if validation[
            "errors"
        ]:
            for error in validation[
                "errors"
            ]:
                print(
                    f"      ERROR: "
                    f"{error}"
                )

        if validation[
            "warnings"
        ]:
            for warning in validation[
                "warnings"
            ]:
                print(
                    f"      WARNING: "
                    f"{warning}"
                )

        results.append(
            result_item
        )

    # ========================================================
    # 5. Summary
    # ========================================================

    successful_samples = (
        api_success_count
    )

    if successful_samples > 0:

        avg_entities = (
            total_entities
            / successful_samples
        )

        avg_attributes = (
            total_attributes
            / successful_samples
        )

        avg_relations = (
            total_relations
            / successful_samples
        )

        sanitizer_change_rate = (
            sanitizer_changed_sample_count
            / successful_samples
        )

    else:

        avg_entities = 0.0
        avg_attributes = 0.0
        avg_relations = 0.0
        sanitizer_change_rate = 0.0

    summary = {
        # ----------------------------------------------------
        # API
        # ----------------------------------------------------

        "api_success_count": (
            api_success_count
        ),

        "api_failure_count": (
            api_failure_count
        ),

        # ----------------------------------------------------
        # Raw structural validation
        # ----------------------------------------------------

        "raw_validation_valid_count": (
            raw_validation_valid_count
        ),

        "raw_validation_invalid_count": (
            raw_validation_invalid_count
        ),

        # ----------------------------------------------------
        # Sanitized validation
        # ----------------------------------------------------

        "validation_valid_count": (
            sanitized_validation_valid_count
        ),

        "validation_invalid_count": (
            sanitized_validation_invalid_count
        ),

        "warning_sample_count": (
            warning_sample_count
        ),

        "error_sample_count": (
            error_sample_count
        ),

        "total_warnings": (
            total_warnings
        ),

        "total_validation_errors": (
            total_validation_errors
        ),

        # ----------------------------------------------------
        # Sanitizer
        # ----------------------------------------------------

        "sanitizer_changed_sample_count": (
            sanitizer_changed_sample_count
        ),

        "sanitizer_unchanged_sample_count": (
            sanitizer_unchanged_sample_count
        ),

        "sanitizer_change_rate": round(
            sanitizer_change_rate,
            6,
        ),

        "total_sanitizer_actions": (
            total_sanitizer_actions
        ),

        "total_dropped_entities": (
            total_dropped_entities
        ),

        "total_dropped_attributes": (
            total_dropped_attributes
        ),

        "total_dropped_relations": (
            total_dropped_relations
        ),

        "total_reassigned_entity_ids": (
            total_reassigned_entity_ids
        ),

        # ----------------------------------------------------
        # Final sanitized EAR semantic statistics
        # ----------------------------------------------------

        "total_entities": (
            total_entities
        ),

        "total_attributes": (
            total_attributes
        ),

        "total_relations": (
            total_relations
        ),

        "avg_entities_per_caption": round(
            avg_entities,
            4,
        ),

        "avg_attributes_per_caption": round(
            avg_attributes,
            4,
        ),

        "avg_relations_per_caption": round(
            avg_relations,
            4,
        ),
    }

    # ========================================================
    # 6. Metadata
    # ========================================================

    metadata = {
        "dataset": "RSICD",

        "input_file": input_path,

        "total_pairs": len(
            dataset
        ),

        "unique_captions": len(
            unique_samples
        ),

        "selected_samples": len(
            selected_samples
        ),

        "random_seed": seed,

        "provider": (
            "Alibaba Cloud Bailian"
        ),

        "model": model,

        "base_url": base_url,

        "temperature": temperature,

        "prompt_version": (
            PROMPT_VERSION
        ),

        "schema_version": (
            SCHEMA_VERSION
        ),

        "sanitizer_version": (
            SANITIZER_VERSION
        ),

        "pipeline": [
            "llm_raw_ear",
            "structural_sanitizer",
            "generic_schema_validator",
        ],
    }

    # ========================================================
    # 7. Final output
    # ========================================================

    output_data = {
        "metadata": metadata,
        "summary": summary,
        "samples": results,
    }

    save_json(
        data=output_data,
        output_path=output_path,
    )

    # ========================================================
    # 8. Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("Extraction Summary")
    print("=" * 70)

    print()
    print("API:")

    print(
        f"  Successful        : "
        f"{api_success_count}"
    )

    print(
        f"  Failed            : "
        f"{api_failure_count}"
    )

    print()
    print("Raw validation:")

    print(
        f"  Valid             : "
        f"{raw_validation_valid_count}"
    )

    print(
        f"  Invalid           : "
        f"{raw_validation_invalid_count}"
    )

    print()
    print("Sanitizer:")

    print(
        f"  Changed samples   : "
        f"{sanitizer_changed_sample_count}"
    )

    print(
        f"  Unchanged samples : "
        f"{sanitizer_unchanged_sample_count}"
    )

    print(
        f"  Total actions     : "
        f"{total_sanitizer_actions}"
    )

    print(
        f"  Dropped entities  : "
        f"{total_dropped_entities}"
    )

    print(
        f"  Dropped attributes: "
        f"{total_dropped_attributes}"
    )

    print(
        f"  Dropped relations : "
        f"{total_dropped_relations}"
    )

    print(
        f"  Reassigned IDs    : "
        f"{total_reassigned_entity_ids}"
    )

    print()
    print("Final validation:")

    print(
        f"  Valid             : "
        f"{sanitized_validation_valid_count}"
    )

    print(
        f"  Invalid           : "
        f"{sanitized_validation_invalid_count}"
    )

    print(
        f"  Samples w/warning : "
        f"{warning_sample_count}"
    )

    print(
        f"  Samples w/error   : "
        f"{error_sample_count}"
    )

    print(
        f"  Total warnings    : "
        f"{total_warnings}"
    )

    print(
        f"  Total errors      : "
        f"{total_validation_errors}"
    )

    print()
    print("Semantic statistics:")

    print(
        f"  Total entities    : "
        f"{total_entities}"
    )

    print(
        f"  Total attributes  : "
        f"{total_attributes}"
    )

    print(
        f"  Total relations   : "
        f"{total_relations}"
    )

    print(
        f"  Avg entities      : "
        f"{avg_entities:.4f}"
    )

    print(
        f"  Avg attributes    : "
        f"{avg_attributes:.4f}"
    )

    print(
        f"  Avg relations     : "
        f"{avg_relations:.4f}"
    )

    print()
    print(
        f"Saved to: "
        f"{output_path}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Extract open-vocabulary EAR structured "
            "semantics from image captions."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset JSON file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Number of UNIQUE captions to process. "
            "Use <=0 for all captions."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random sampling seed.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=(
            "https://dashscope.aliyuncs.com/"
            "compatible-mode/v1"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "qwen3.7-flash-2026-07-15"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
    )

    return parser


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    api_key = os.getenv(
        "DASHSCOPE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "Environment variable "
            "DASHSCOPE_API_KEY is not set."
        )

    run_extraction(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        seed=args.seed,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()