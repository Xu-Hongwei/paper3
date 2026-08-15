import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import StructuredSemanticsCache
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
    Lightweight normalization used for exact caption
    deduplication and cache lookup.

    Operations:
    - strip
    - lowercase
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
    Read caption field from one sample.
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
# Exact caption deduplication
# ============================================================

def build_unique_caption_pool(
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a pool of normalized exact-unique captions.

    Every unique caption stores ALL original source indices.

    Example:

        index 10 -> caption A
        index 20 -> caption A
        index 30 -> caption A

    becomes:

        {
            "caption": caption A,
            "_source_indices": [10, 20, 30]
        }

    This allows one LLM request to later map back to all
    original image-caption pairs.

    No semantic deduplication is performed.
    """

    unique_map: Dict[
        str,
        Dict[str, Any],
    ] = {}

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

        # ====================================================
        # First occurrence
        # ====================================================

        if normalized not in unique_map:

            unique_map[normalized] = {
                "_source_indices": [
                    source_index
                ],

                "_normalized_caption": (
                    normalized
                ),

                # Preserve original form from first occurrence.
                "caption": caption,

                # Metadata from first occurrence only.
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

        # ====================================================
        # Duplicate occurrence
        # ====================================================

        else:

            unique_map[
                normalized
            ][
                "_source_indices"
            ].append(
                source_index
            )

    return list(
        unique_map.values()
    )


# ============================================================
# Mapping statistics
# ============================================================

def get_mapping_statistics(
    dataset: List[Dict[str, Any]],
    unique_samples: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Verify that all valid original caption pairs remain mapped
    after deduplication.
    """

    valid_caption_pairs = 0

    for sample in dataset:

        if not isinstance(sample, dict):
            continue

        if get_caption(sample):
            valid_caption_pairs += 1

    mapped_pairs = sum(
        len(
            sample.get(
                "_source_indices",
                [],
            )
        )
        for sample in unique_samples
    )

    max_occurrences = max(
        (
            len(
                sample.get(
                    "_source_indices",
                    [],
                )
            )
            for sample in unique_samples
        ),
        default=0,
    )

    return {
        "total_pairs": len(dataset),

        "valid_caption_pairs": (
            valid_caption_pairs
        ),

        "unique_captions": len(
            unique_samples
        ),

        "mapped_pairs": mapped_pairs,

        "duplicate_pairs": (
            valid_caption_pairs
            - len(unique_samples)
        ),

        "max_occurrences": (
            max_occurrences
        ),
    }


# ============================================================
# Sampling
# ============================================================

def select_samples(
    unique_samples: List[Dict[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Select reproducible UNIQUE captions.

    limit <= 0:
        use all unique captions.
    """

    if (
        limit <= 0
        or limit >= len(unique_samples)
    ):
        return list(unique_samples)

    rng = random.Random(seed)

    return rng.sample(
        unique_samples,
        limit,
    )


# ============================================================
# Save final JSON
# ============================================================

def save_json(
    data: Dict[str, Any],
    output_path: str,
) -> None:

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
    cache_path: str,
    limit: int,
    seed: int,
    api_key: Optional[str],
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> None:

    # ========================================================
    # 1. Dataset
    # ========================================================

    dataset = load_dataset(
        input_path
    )

    unique_samples = (
        build_unique_caption_pool(
            dataset
        )
    )

    mapping_stats = (
        get_mapping_statistics(
            dataset=dataset,
            unique_samples=unique_samples,
        )
    )

    # ========================================================
    # 1.1 Mapping safety check
    # ========================================================

    if (
        mapping_stats[
            "mapped_pairs"
        ]
        !=
        mapping_stats[
            "valid_caption_pairs"
        ]
    ):
        raise RuntimeError(
            "Caption mapping failed: "
            f"mapped_pairs="
            f"{mapping_stats['mapped_pairs']}, "
            f"valid_caption_pairs="
            f"{mapping_stats['valid_caption_pairs']}"
        )

    # ========================================================
    # 1.2 Select unique captions
    # ========================================================

    selected_samples = select_samples(
        unique_samples=unique_samples,
        limit=limit,
        seed=seed,
    )

    # ========================================================
    # 2. Initialize cache
    # ========================================================

    cache = StructuredSemanticsCache(
        cache_path=cache_path,
        model=model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )

    # ========================================================
    # 2.1 Pre-check cache coverage
    # ========================================================

    preexisting_cache_hits = 0

    for sample in selected_samples:

        normalized_caption = sample.get(
            "_normalized_caption",
            "",
        )

        if (
            normalized_caption
            and cache.has(
                normalized_caption
            )
        ):
            preexisting_cache_hits += 1

    expected_cache_misses = (
        len(selected_samples)
        - preexisting_cache_hits
    )

    # ========================================================
    # Console header
    # ========================================================

    print()
    print("=" * 72)
    print("Structured Semantics Extraction")
    print("=" * 72)

    print(
        f"Input pairs           : "
        f"{len(dataset)}"
    )

    print(
        f"Valid caption pairs   : "
        f"{mapping_stats['valid_caption_pairs']}"
    )

    print(
        f"Unique captions       : "
        f"{len(unique_samples)}"
    )

    print(
        f"Duplicate pairs       : "
        f"{mapping_stats['duplicate_pairs']}"
    )

    print(
        f"Mapped pairs          : "
        f"{mapping_stats['mapped_pairs']}"
    )

    print(
        f"Max occurrences       : "
        f"{mapping_stats['max_occurrences']}"
    )

    print(
        f"Selected captions     : "
        f"{len(selected_samples)}"
    )

    print()

    print(
        f"Cache path            : "
        f"{cache_path}"
    )

    print(
        f"Cache records         : "
        f"{len(cache)}"
    )

    print(
        f"Selected cache hits   : "
        f"{preexisting_cache_hits}"
    )

    print(
        f"Expected API requests : "
        f"{expected_cache_misses}"
    )

    print()

    print(
        f"Prompt version        : "
        f"{PROMPT_VERSION}"
    )

    print(
        f"Schema version        : "
        f"{SCHEMA_VERSION}"
    )

    print(
        f"Sanitizer version     : "
        f"{SANITIZER_VERSION}"
    )

    print(
        f"Model                 : "
        f"{model}"
    )

    print("=" * 72)
    print()

    # ========================================================
    # 3. API key validation
    # ========================================================

    # If every selected caption already exists in cache,
    # no API key is required.

    if (
        expected_cache_misses > 0
        and not api_key
    ):
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set, "
            f"but {expected_cache_misses} selected captions "
            "are missing from cache."
        )

    # ========================================================
    # 4. LLM client
    # ========================================================

    client: Optional[LLMClient] = None

    if expected_cache_misses > 0:

        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ========================================================
    # 5. Counters
    # ========================================================

    extraction_success_count = 0
    extraction_failure_count = 0

    cache_hit_count = 0
    cache_miss_count = 0

    api_call_count = 0
    api_call_success_count = 0
    api_call_failure_count = 0

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

    total_entities = 0
    total_attributes = 0
    total_relations = 0

    selected_original_pair_count = sum(
        len(
            sample.get(
                "_source_indices",
                [],
            )
        )
        for sample in selected_samples
    )

    results: List[
        Dict[str, Any]
    ] = []

    # ========================================================
    # 6. Extraction loop
    # ========================================================

    for sample_index, sample in enumerate(
        selected_samples
    ):

        caption = sample["caption"]

        normalized_caption = sample[
            "_normalized_caption"
        ]

        source_indices = sample.get(
            "_source_indices",
            [],
        )

        print(
            f"[{sample_index + 1:05d}/"
            f"{len(selected_samples):05d}] "
            f"{caption}"
        )

        # ====================================================
        # Result metadata
        # ====================================================

        result_item: Dict[str, Any] = {
            "sample_index": sample_index,

            "source_indices": (
                source_indices
            ),

            "num_occurrences": len(
                source_indices
            ),

            # Metadata from first occurrence only.
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
                normalized_caption
            ),
        }

        # ====================================================
        # 6.1 Cache lookup
        # ====================================================

        cache_record = cache.get(
            normalized_caption
        )

        raw_semantics = None

        if isinstance(
            cache_record,
            dict,
        ):

            cached_raw = cache_record.get(
                "raw_structured_semantics"
            )

            if isinstance(
                cached_raw,
                dict,
            ):

                raw_semantics = cached_raw

                cache_hit_count += 1

                result_item[
                    "cache_hit"
                ] = True

                result_item[
                    "api_called"
                ] = False

                result_item[
                    "extraction_source"
                ] = "cache"

                print(
                    "    Source          : CACHE HIT"
                )

        # ====================================================
        # 6.2 Cache miss -> API
        # ====================================================

        if raw_semantics is None:

            cache_miss_count += 1

            result_item[
                "cache_hit"
            ] = False

            result_item[
                "api_called"
            ] = True

            result_item[
                "extraction_source"
            ] = "api"

            api_call_count += 1

            print(
                "    Source          : API"
            )

            try:

                if client is None:
                    raise RuntimeError(
                        "LLM client was not initialized."
                    )

                raw_semantics = client.extract(
                    caption
                )

                api_call_success_count += 1

                # ============================================
                # IMPORTANT:
                # Immediately save RAW EAR to JSONL cache.
                #
                # If the program crashes after this point,
                # this API result is still preserved.
                # ============================================

                cache.put(
                    normalized_caption=(
                        normalized_caption
                    ),

                    caption=caption,

                    raw_structured_semantics=(
                        raw_semantics
                    ),
                )

            except Exception as exc:

                api_call_failure_count += 1
                extraction_failure_count += 1

                result_item[
                    "api_success"
                ] = False

                result_item[
                    "extraction_success"
                ] = False

                result_item[
                    "api_error"
                ] = str(exc)

                print(
                    f"    API ERROR       : "
                    f"{exc}"
                )

                results.append(
                    result_item
                )

                continue

        # ====================================================
        # 6.3 Extraction available
        # ====================================================

        extraction_success_count += 1

        # Keep for backwards compatibility with statistics.py.
        # Here True means usable raw extraction exists.
        result_item[
            "api_success"
        ] = True

        result_item[
            "extraction_success"
        ] = True

        # ====================================================
        # 6.4 Preserve RAW EAR
        # ====================================================

        result_item[
            "raw_structured_semantics"
        ] = raw_semantics

        # ====================================================
        # 6.5 Validate RAW EAR
        # ====================================================

        raw_validation = (
            validate_structured_semantics(
                raw_semantics
            )
        )

        result_item[
            "raw_validation"
        ] = raw_validation

        if raw_validation[
            "valid"
        ]:
            raw_validation_valid_count += 1
        else:
            raw_validation_invalid_count += 1

        # ====================================================
        # 6.6 Sanitize
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

        # ====================================================
        # Sanitizer statistics
        # ====================================================

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
        # 6.7 Validate sanitized EAR
        # ====================================================

        validation = (
            validate_structured_semantics(
                sanitized_semantics
            )
        )

        result_item[
            "validation"
        ] = validation

        if validation[
            "valid"
        ]:
            sanitized_validation_valid_count += 1
        else:
            sanitized_validation_invalid_count += 1

        if validation[
            "warnings"
        ]:
            warning_sample_count += 1

        if validation[
            "errors"
        ]:
            error_sample_count += 1

        total_warnings += len(
            validation[
                "warnings"
            ]
        )

        total_validation_errors += len(
            validation[
                "errors"
            ]
        )

        # ====================================================
        # 6.8 Semantic statistics
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
        # 6.9 Console diagnostics
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
            f"    Raw validation  : "
            f"{raw_status}"
        )

        print(
            f"    Sanitizer       : "
            f"{sanitize_status} "
            f"({len(sanitizer_actions)} actions)"
        )

        print(
            f"    Final validation: "
            f"{final_status}"
        )

        results.append(
            result_item
        )

    # ========================================================
    # 7. Summary
    # ========================================================

    successful_samples = (
        extraction_success_count
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

        cache_hit_rate = (
            cache_hit_count
            / successful_samples
        )

    else:

        avg_entities = 0.0
        avg_attributes = 0.0
        avg_relations = 0.0

        sanitizer_change_rate = 0.0
        cache_hit_rate = 0.0

    summary = {
        # ----------------------------------------------------
        # Mapping
        # ----------------------------------------------------

        "selected_unique_captions": (
            len(selected_samples)
        ),

        "selected_original_pairs": (
            selected_original_pair_count
        ),

        # ----------------------------------------------------
        # Extraction
        # ----------------------------------------------------

        "extraction_success_count": (
            extraction_success_count
        ),

        "extraction_failure_count": (
            extraction_failure_count
        ),

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        "cache_hit_count": (
            cache_hit_count
        ),

        "cache_miss_count": (
            cache_miss_count
        ),

        "cache_hit_rate": round(
            cache_hit_rate,
            6,
        ),

        "final_cache_records": len(
            cache
        ),

        # ----------------------------------------------------
        # Real API calls
        # ----------------------------------------------------

        "api_call_count": (
            api_call_count
        ),

        "api_call_success_count": (
            api_call_success_count
        ),

        "api_call_failure_count": (
            api_call_failure_count
        ),

        # ----------------------------------------------------
        # Raw validation
        # ----------------------------------------------------

        "raw_validation_valid_count": (
            raw_validation_valid_count
        ),

        "raw_validation_invalid_count": (
            raw_validation_invalid_count
        ),

        # ----------------------------------------------------
        # Final validation
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
        # Semantics
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
    # 8. Metadata
    # ========================================================

    metadata = {
        "dataset": "RSICD",

        "input_file": input_path,

        "total_pairs": len(
            dataset
        ),

        "valid_caption_pairs": (
            mapping_stats[
                "valid_caption_pairs"
            ]
        ),

        "unique_captions": len(
            unique_samples
        ),

        "duplicate_pairs": (
            mapping_stats[
                "duplicate_pairs"
            ]
        ),

        "mapped_pairs": (
            mapping_stats[
                "mapped_pairs"
            ]
        ),

        "max_caption_occurrences": (
            mapping_stats[
                "max_occurrences"
            ]
        ),

        "selected_samples": len(
            selected_samples
        ),

        "selected_original_pairs": (
            selected_original_pair_count
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

        "cache": {
            "enabled": True,
            "path": cache_path,
            "stores": "raw_structured_semantics",
        },

        "deduplication": {
            "enabled": True,

            "method": (
                "normalized_exact_match"
            ),

            "normalization": [
                "strip",
                "lowercase",
                "collapse_whitespace",
            ],

            "semantic_deduplication": (
                False
            ),

            "preserve_all_source_indices": (
                True
            ),
        },

        "pipeline": [
            "caption_normalization",
            "exact_caption_deduplication",
            "cache_lookup",
            "llm_raw_ear_if_cache_miss",
            "raw_ear_cache_write",
            "structural_sanitizer",
            "generic_schema_validator",
        ],
    }

    # ========================================================
    # 9. Save final JSON
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
    # 10. Console summary
    # ========================================================

    print()
    print("=" * 72)
    print("Extraction Summary")
    print("=" * 72)

    print()
    print("Caption mapping:")

    print(
        f"  Original pairs      : "
        f"{len(dataset)}"
    )

    print(
        f"  Unique captions     : "
        f"{len(unique_samples)}"
    )

    print(
        f"  Mapped pairs        : "
        f"{mapping_stats['mapped_pairs']}"
    )

    print(
        f"  Selected unique     : "
        f"{len(selected_samples)}"
    )

    print(
        f"  Selected pairs      : "
        f"{selected_original_pair_count}"
    )

    print()
    print("Cache:")

    print(
        f"  Hits                : "
        f"{cache_hit_count}"
    )

    print(
        f"  Misses              : "
        f"{cache_miss_count}"
    )

    print(
        f"  Final records       : "
        f"{len(cache)}"
    )

    print()
    print("API:")

    print(
        f"  Actual API calls    : "
        f"{api_call_count}"
    )

    print(
        f"  Successful calls    : "
        f"{api_call_success_count}"
    )

    print(
        f"  Failed calls        : "
        f"{api_call_failure_count}"
    )

    print()
    print("Extraction:")

    print(
        f"  Successful samples  : "
        f"{extraction_success_count}"
    )

    print(
        f"  Failed samples      : "
        f"{extraction_failure_count}"
    )

    print()
    print("Raw validation:")

    print(
        f"  Valid               : "
        f"{raw_validation_valid_count}"
    )

    print(
        f"  Invalid             : "
        f"{raw_validation_invalid_count}"
    )

    print()
    print("Sanitizer:")

    print(
        f"  Changed samples     : "
        f"{sanitizer_changed_sample_count}"
    )

    print(
        f"  Unchanged samples   : "
        f"{sanitizer_unchanged_sample_count}"
    )

    print(
        f"  Total actions       : "
        f"{total_sanitizer_actions}"
    )

    print(
        f"  Dropped entities    : "
        f"{total_dropped_entities}"
    )

    print(
        f"  Dropped attributes  : "
        f"{total_dropped_attributes}"
    )

    print(
        f"  Dropped relations   : "
        f"{total_dropped_relations}"
    )

    print()
    print("Final validation:")

    print(
        f"  Valid               : "
        f"{sanitized_validation_valid_count}"
    )

    print(
        f"  Invalid             : "
        f"{sanitized_validation_invalid_count}"
    )

    print(
        f"  Samples w/warning   : "
        f"{warning_sample_count}"
    )

    print(
        f"  Samples w/error     : "
        f"{error_sample_count}"
    )

    print()
    print("Semantic statistics:")

    print(
        f"  Total entities      : "
        f"{total_entities}"
    )

    print(
        f"  Total attributes    : "
        f"{total_attributes}"
    )

    print(
        f"  Total relations     : "
        f"{total_relations}"
    )

    print(
        f"  Avg entities        : "
        f"{avg_entities:.4f}"
    )

    print(
        f"  Avg attributes      : "
        f"{avg_attributes:.4f}"
    )

    print(
        f"  Avg relations       : "
        f"{avg_relations:.4f}"
    )

    print()
    print(
        f"Saved to: "
        f"{output_path}"
    )

    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Extract open-vocabulary EAR structured semantics "
            "with exact caption deduplication and persistent "
            "cache/resume."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset JSON.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Final extraction JSON.",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=(
            "cache/structured_semantics/"
            "qwen37_v30_open_schema_v1.jsonl"
        ),
        help="Persistent RAW EAR JSONL cache.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Number of UNIQUE captions. "
            "Use <=0 for all captions."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    run_extraction(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache_path,
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