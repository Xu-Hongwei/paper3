import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Basic helpers
# ============================================================

def _normalize_text(
    value: str,
) -> str:
    """
    Lightweight normalization used ONLY for statistics.

    This does not modify the original annotation.

    Operations:
    - strip
    - lowercase
    - collapse whitespace
    """

    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def _safe_div(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe division.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _percentage(
    numerator: float,
    denominator: float,
) -> float:
    """
    Return percentage in [0, 100].
    """

    return (
        _safe_div(
            numerator,
            denominator,
        )
        * 100.0
    )


def _distribution_summary(
    values: List[int],
) -> Dict[str, Any]:
    """
    Summarize a list of integer counts.

    Example:
        number of entities per caption.
    """

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "histogram": {},
        }

    histogram = Counter(values)

    return {
        "count": len(values),

        "mean": round(
            sum(values) / len(values),
            6,
        ),

        "min": min(values),

        "max": max(values),

        "histogram": {
            str(key): value
            for key, value
            in sorted(
                histogram.items()
            )
        },
    }


# ============================================================
# Loading
# ============================================================

def load_extraction_file(
    input_path: str,
) -> Dict[str, Any]:
    """
    Load one extraction-result JSON.
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

    if not isinstance(data, dict):
        raise ValueError(
            "Extraction file must contain "
            "a top-level JSON object."
        )

    samples = data.get(
        "samples"
    )

    if not isinstance(samples, list):
        raise ValueError(
            "Extraction file must contain "
            "a top-level 'samples' list."
        )

    return data


# ============================================================
# EAR selection
# ============================================================

def get_semantics(
    sample: Dict[str, Any],
    source: str,
) -> Optional[Dict[str, Any]]:
    """
    Select which EAR representation to analyze.

    source:
        sanitized
        raw

    Sanitized mode prefers:
        sanitized_structured_semantics

    Raw mode prefers:
        raw_structured_semantics

    Legacy fallback:
        structured_semantics
    """

    if source == "sanitized":

        semantics = sample.get(
            "sanitized_structured_semantics"
        )

        if isinstance(semantics, dict):
            return semantics

        semantics = sample.get(
            "structured_semantics"
        )

        if isinstance(semantics, dict):
            return semantics

        semantics = sample.get(
            "raw_structured_semantics"
        )

        if isinstance(semantics, dict):
            return semantics

        return None

    if source == "raw":

        semantics = sample.get(
            "raw_structured_semantics"
        )

        if isinstance(semantics, dict):
            return semantics

        semantics = sample.get(
            "structured_semantics"
        )

        if isinstance(semantics, dict):
            return semantics

        return None

    raise ValueError(
        f"Unsupported source: {source}"
    )


# ============================================================
# Sanitizer action grouping
# ============================================================

def classify_sanitizer_action(
    action: str,
) -> str:
    """
    Convert verbose sanitizer messages into coarse categories.

    IMPORTANT:
    This is only for reporting.
    """

    text = action.lower()

    if (
        "missing 'attributes'"
        in text
        or
        "attributes was not a list"
        in text
    ):
        return "repair_attributes_field"

    if "reassigned entity id" in text:
        return "reassign_entity_id"

    if "dropped duplicate relation" in text:
        return "drop_duplicate_relation"

    if "dropped duplicate attribute" in text:
        return "drop_duplicate_attribute"

    if (
        "does not reference a unique valid entity"
        in text
    ):
        return "drop_invalid_relation_endpoint"

    if "self relation" in text:
        return "drop_self_relation"

    if (
        "missing or empty predicate"
        in text
    ):
        return "drop_empty_predicate"

    if (
        "missing or empty subject"
        in text
        or
        "missing or empty object"
        in text
    ):
        return "drop_empty_relation_endpoint"

    if (
        "entity was not an object"
        in text
        or
        "missing or empty entity text"
        in text
    ):
        return "drop_invalid_entity"

    if (
        "attribute was not an object"
        in text
        or
        "missing or empty type"
        in text
        or
        "missing or empty value"
        in text
    ):
        return "drop_invalid_attribute"

    if "relation was not an object" in text:
        return "drop_invalid_relation"

    if (
        "missing top-level 'entities'"
        in text
        or
        "top-level 'entities' was not a list"
        in text
    ):
        return "repair_entities_field"

    if (
        "missing top-level 'relations'"
        in text
        or
        "top-level 'relations' was not a list"
        in text
    ):
        return "repair_relations_field"

    if "duplicated" in text:
        return "ambiguous_duplicate_entity_id"

    return "other"


# ============================================================
# Main analyzer
# ============================================================

def analyze_extraction(
    data: Dict[str, Any],
    source: str = "sanitized",
    top_k: int = 30,
) -> Dict[str, Any]:
    """
    Analyze extraction results.

    No dataset-specific ontology is assumed.
    """

    metadata = data.get(
        "metadata",
        {},
    )

    extraction_summary = data.get(
        "summary",
        {},
    )

    samples = data.get(
        "samples",
        [],
    )

    # ========================================================
    # General counters
    # ========================================================

    total_samples = len(samples)

    api_success = 0
    api_failure = 0

    raw_valid = 0
    raw_invalid = 0

    final_valid = 0
    final_invalid = 0

    warning_samples = 0
    error_samples = 0

    total_warnings = 0
    total_errors = 0

    # ========================================================
    # Sanitizer counters
    # ========================================================

    sanitizer_changed = 0
    sanitizer_unchanged = 0

    sanitizer_action_counter = Counter()

    total_sanitizer_actions = 0

    total_dropped_entities = 0
    total_dropped_attributes = 0
    total_dropped_relations = 0

    total_reassigned_entity_ids = 0

    # ========================================================
    # Semantic counters
    # ========================================================

    entity_counter = Counter()

    attribute_type_counter = Counter()

    attribute_value_counter = Counter()

    attribute_type_value_counter = Counter()

    relation_predicate_counter = Counter()

    # --------------------------------------------------------
    # Per-caption distributions
    # --------------------------------------------------------

    entities_per_caption: List[int] = []
    attributes_per_caption: List[int] = []
    relations_per_caption: List[int] = []

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    zero_entity_captions = 0
    zero_attribute_captions = 0
    zero_relation_captions = 0

    captions_with_attributes = 0
    captions_with_relations = 0

    # --------------------------------------------------------
    # Entity-level distributions
    # --------------------------------------------------------

    attributes_per_entity: List[int] = []

    # --------------------------------------------------------
    # Open-vocabulary diagnostics
    # --------------------------------------------------------

    duplicate_entity_text_samples = 0

    samples_with_repeated_predicate = 0

    analyzed_semantics_count = 0

    # ========================================================
    # Iterate samples
    # ========================================================

    for sample in samples:

        if not isinstance(sample, dict):
            continue

        # ====================================================
        # API
        # ====================================================

        api_success_flag = sample.get(
            "api_success"
        )

        if api_success_flag is False:
            api_failure += 1
            continue

        # Legacy files may not contain api_success.
        if api_success_flag is True:
            api_success += 1
        else:
            semantics_probe = get_semantics(
                sample,
                source,
            )

            if semantics_probe is not None:
                api_success += 1
            else:
                api_failure += 1
                continue

        # ====================================================
        # Raw validation
        # ====================================================

        raw_validation = sample.get(
            "raw_validation"
        )

        if isinstance(
            raw_validation,
            dict,
        ):

            if raw_validation.get(
                "valid"
            ):
                raw_valid += 1
            else:
                raw_invalid += 1

        # ====================================================
        # Final validation
        # ====================================================

        validation = sample.get(
            "validation"
        )

        if isinstance(
            validation,
            dict,
        ):

            if validation.get(
                "valid"
            ):
                final_valid += 1
            else:
                final_invalid += 1

            warnings = validation.get(
                "warnings",
                [],
            )

            errors = validation.get(
                "errors",
                [],
            )

            if isinstance(
                warnings,
                list,
            ):
                total_warnings += len(
                    warnings
                )

                if warnings:
                    warning_samples += 1

            if isinstance(
                errors,
                list,
            ):
                total_errors += len(
                    errors
                )

                if errors:
                    error_samples += 1

        # ====================================================
        # Sanitization
        # ====================================================

        sanitization = sample.get(
            "sanitization"
        )

        if isinstance(
            sanitization,
            dict,
        ):

            if sanitization.get(
                "changed"
            ):
                sanitizer_changed += 1
            else:
                sanitizer_unchanged += 1

            actions = sanitization.get(
                "actions",
                [],
            )

            if isinstance(actions, list):

                total_sanitizer_actions += len(
                    actions
                )

                for action in actions:

                    if not isinstance(
                        action,
                        str,
                    ):
                        continue

                    category = (
                        classify_sanitizer_action(
                            action
                        )
                    )

                    sanitizer_action_counter[
                        category
                    ] += 1

            stats = sanitization.get(
                "stats",
                {},
            )

            if isinstance(stats, dict):

                total_dropped_entities += int(
                    stats.get(
                        "dropped_entities",
                        0,
                    )
                    or 0
                )

                total_dropped_attributes += int(
                    stats.get(
                        "dropped_attributes",
                        0,
                    )
                    or 0
                )

                total_dropped_relations += int(
                    stats.get(
                        "dropped_relations",
                        0,
                    )
                    or 0
                )

                (
                    total_reassigned_entity_ids
                ) += int(
                    stats.get(
                        "reassigned_entity_ids",
                        0,
                    )
                    or 0
                )

        # ====================================================
        # EAR
        # ====================================================

        semantics = get_semantics(
            sample=sample,
            source=source,
        )

        if not isinstance(
            semantics,
            dict,
        ):
            continue

        analyzed_semantics_count += 1

        entities = semantics.get(
            "entities",
            [],
        )

        relations = semantics.get(
            "relations",
            [],
        )

        if not isinstance(
            entities,
            list,
        ):
            entities = []

        if not isinstance(
            relations,
            list,
        ):
            relations = []

        # ====================================================
        # Entity / attribute statistics
        # ====================================================

        caption_entity_count = 0
        caption_attribute_count = 0

        local_entity_counter = Counter()

        for entity in entities:

            if not isinstance(
                entity,
                dict,
            ):
                continue

            text = entity.get(
                "text"
            )

            if (
                isinstance(text, str)
                and text.strip()
            ):
                normalized_text = (
                    _normalize_text(
                        text
                    )
                )

                entity_counter[
                    normalized_text
                ] += 1

                local_entity_counter[
                    normalized_text
                ] += 1

                caption_entity_count += 1

            attributes = entity.get(
                "attributes",
                [],
            )

            if not isinstance(
                attributes,
                list,
            ):
                attributes = []

            valid_attribute_count = 0

            for attr in attributes:

                if not isinstance(
                    attr,
                    dict,
                ):
                    continue

                attr_type = attr.get(
                    "type"
                )

                attr_value = attr.get(
                    "value"
                )

                if (
                    not isinstance(
                        attr_type,
                        str,
                    )
                    or not attr_type.strip()
                ):
                    continue

                if (
                    not isinstance(
                        attr_value,
                        str,
                    )
                    or not attr_value.strip()
                ):
                    continue

                normalized_type = (
                    _normalize_text(
                        attr_type
                    )
                )

                normalized_value = (
                    _normalize_text(
                        attr_value
                    )
                )

                attribute_type_counter[
                    normalized_type
                ] += 1

                attribute_value_counter[
                    normalized_value
                ] += 1

                attribute_type_value_counter[
                    (
                        normalized_type,
                        normalized_value,
                    )
                ] += 1

                valid_attribute_count += 1

                caption_attribute_count += 1

            attributes_per_entity.append(
                valid_attribute_count
            )

        if any(
            count > 1
            for count
            in local_entity_counter.values()
        ):
            duplicate_entity_text_samples += 1

        # ====================================================
        # Relation statistics
        # ====================================================

        caption_relation_count = 0

        local_predicate_counter = Counter()

        for relation in relations:

            if not isinstance(
                relation,
                dict,
            ):
                continue

            predicate = relation.get(
                "predicate"
            )

            if (
                not isinstance(
                    predicate,
                    str,
                )
                or not predicate.strip()
            ):
                continue

            normalized_predicate = (
                _normalize_text(
                    predicate
                )
            )

            relation_predicate_counter[
                normalized_predicate
            ] += 1

            local_predicate_counter[
                normalized_predicate
            ] += 1

            caption_relation_count += 1

        if any(
            count > 1
            for count
            in local_predicate_counter.values()
        ):
            samples_with_repeated_predicate += 1

        # ====================================================
        # Per-caption counts
        # ====================================================

        entities_per_caption.append(
            caption_entity_count
        )

        attributes_per_caption.append(
            caption_attribute_count
        )

        relations_per_caption.append(
            caption_relation_count
        )

        # ====================================================
        # Coverage
        # ====================================================

        if caption_entity_count == 0:
            zero_entity_captions += 1

        if caption_attribute_count == 0:
            zero_attribute_captions += 1
        else:
            captions_with_attributes += 1

        if caption_relation_count == 0:
            zero_relation_captions += 1
        else:
            captions_with_relations += 1

    # ========================================================
    # Derived vocabulary statistics
    # ========================================================

    unique_entities = len(
        entity_counter
    )

    unique_attribute_types = len(
        attribute_type_counter
    )

    unique_attribute_values = len(
        attribute_value_counter
    )

    unique_relation_predicates = len(
        relation_predicate_counter
    )

    entity_singletons = sum(
        1
        for count
        in entity_counter.values()
        if count == 1
    )

    relation_singletons = sum(
        1
        for count
        in relation_predicate_counter.values()
        if count == 1
    )

    # ========================================================
    # Total semantic counts
    # ========================================================

    total_entities = sum(
        entity_counter.values()
    )

    total_attributes = sum(
        attribute_type_counter.values()
    )

    total_relations = sum(
        relation_predicate_counter.values()
    )

    # ========================================================
    # Top-K helper
    # ========================================================

    top_entities = [
        {
            "text": key,
            "count": count,
        }
        for key, count
        in entity_counter.most_common(
            top_k
        )
    ]

    top_attribute_types = [
        {
            "type": key,
            "count": count,
        }
        for key, count
        in attribute_type_counter.most_common(
            top_k
        )
    ]

    top_attribute_values = [
        {
            "value": key,
            "count": count,
        }
        for key, count
        in attribute_value_counter.most_common(
            top_k
        )
    ]

    top_attribute_pairs = [
        {
            "type": key[0],
            "value": key[1],
            "count": count,
        }
        for key, count
        in attribute_type_value_counter.most_common(
            top_k
        )
    ]

    relation_frequency = [
        {
            "predicate": key,
            "count": count,
        }
        for key, count
        in relation_predicate_counter.most_common()
    ]

    top_relations = (
        relation_frequency[
            :top_k
        ]
    )

    singleton_relations = sorted(
        [
            predicate
            for predicate, count
            in relation_predicate_counter.items()
            if count == 1
        ]
    )

    # ========================================================
    # Final report
    # ========================================================

    report = {
        "analysis": {
            "semantic_source": source,
            "top_k": top_k,
            "analyzed_semantics_count": (
                analyzed_semantics_count
            ),
        },

        "metadata": metadata,

        "original_extraction_summary": (
            extraction_summary
        ),

        # ====================================================
        # Pipeline quality
        # ====================================================

        "pipeline_quality": {
            "total_samples": total_samples,

            "api_success": api_success,
            "api_failure": api_failure,

            "api_success_rate_percent": round(
                _percentage(
                    api_success,
                    total_samples,
                ),
                4,
            ),

            "raw_valid": raw_valid,
            "raw_invalid": raw_invalid,

            "raw_valid_rate_percent": round(
                _percentage(
                    raw_valid,
                    raw_valid + raw_invalid,
                ),
                4,
            ),

            "final_valid": final_valid,
            "final_invalid": final_invalid,

            "final_valid_rate_percent": round(
                _percentage(
                    final_valid,
                    final_valid + final_invalid,
                ),
                4,
            ),

            "warning_samples": (
                warning_samples
            ),

            "error_samples": (
                error_samples
            ),

            "total_warnings": (
                total_warnings
            ),

            "total_errors": (
                total_errors
            ),
        },

        # ====================================================
        # Sanitizer
        # ====================================================

        "sanitizer": {
            "changed_samples": (
                sanitizer_changed
            ),

            "unchanged_samples": (
                sanitizer_unchanged
            ),

            "change_rate_percent": round(
                _percentage(
                    sanitizer_changed,
                    sanitizer_changed
                    + sanitizer_unchanged,
                ),
                4,
            ),

            "total_actions": (
                total_sanitizer_actions
            ),

            "action_frequency": {
                key: value
                for key, value
                in sanitizer_action_counter.most_common()
            },

            "dropped_entities": (
                total_dropped_entities
            ),

            "dropped_attributes": (
                total_dropped_attributes
            ),

            "dropped_relations": (
                total_dropped_relations
            ),

            "reassigned_entity_ids": (
                total_reassigned_entity_ids
            ),
        },

        # ====================================================
        # Entity
        # ====================================================

        "entities": {
            "total": total_entities,

            "unique_phrases": (
                unique_entities
            ),

            "singleton_phrases": (
                entity_singletons
            ),

            "singleton_ratio_percent": round(
                _percentage(
                    entity_singletons,
                    unique_entities,
                ),
                4,
            ),

            "avg_per_caption": round(
                _safe_div(
                    total_entities,
                    analyzed_semantics_count,
                ),
                6,
            ),

            "per_caption_distribution": (
                _distribution_summary(
                    entities_per_caption
                )
            ),

            "duplicate_entity_text_samples": (
                duplicate_entity_text_samples
            ),

            "top": top_entities,
        },

        # ====================================================
        # Attribute
        # ====================================================

        "attributes": {
            "total": total_attributes,

            "unique_types": (
                unique_attribute_types
            ),

            "unique_values": (
                unique_attribute_values
            ),

            "avg_per_caption": round(
                _safe_div(
                    total_attributes,
                    analyzed_semantics_count,
                ),
                6,
            ),

            "avg_per_entity": round(
                _safe_div(
                    total_attributes,
                    total_entities,
                ),
                6,
            ),

            "captions_with_attributes": (
                captions_with_attributes
            ),

            "zero_attribute_captions": (
                zero_attribute_captions
            ),

            "coverage_percent": round(
                _percentage(
                    captions_with_attributes,
                    analyzed_semantics_count,
                ),
                4,
            ),

            "per_caption_distribution": (
                _distribution_summary(
                    attributes_per_caption
                )
            ),

            "per_entity_distribution": (
                _distribution_summary(
                    attributes_per_entity
                )
            ),

            "type_frequency": {
                key: value
                for key, value
                in attribute_type_counter.most_common()
            },

            "top_types": (
                top_attribute_types
            ),

            "top_values": (
                top_attribute_values
            ),

            "top_type_value_pairs": (
                top_attribute_pairs
            ),
        },

        # ====================================================
        # Relation
        # ====================================================

        "relations": {
            "total": total_relations,

            "unique_predicates": (
                unique_relation_predicates
            ),

            "singleton_predicates": (
                relation_singletons
            ),

            "singleton_ratio_percent": round(
                _percentage(
                    relation_singletons,
                    unique_relation_predicates,
                ),
                4,
            ),

            "avg_per_caption": round(
                _safe_div(
                    total_relations,
                    analyzed_semantics_count,
                ),
                6,
            ),

            "captions_with_relations": (
                captions_with_relations
            ),

            "zero_relation_captions": (
                zero_relation_captions
            ),

            "coverage_percent": round(
                _percentage(
                    captions_with_relations,
                    analyzed_semantics_count,
                ),
                4,
            ),

            "per_caption_distribution": (
                _distribution_summary(
                    relations_per_caption
                )
            ),

            "samples_with_repeated_predicate": (
                samples_with_repeated_predicate
            ),

            # Full vocabulary, sorted by frequency.
            "predicate_frequency": (
                relation_frequency
            ),

            "top": top_relations,

            "singleton_predicate_list": (
                singleton_relations
            ),
        },

        # ====================================================
        # Coverage
        # ====================================================

        "coverage": {
            "zero_entity_captions": (
                zero_entity_captions
            ),

            "zero_attribute_captions": (
                zero_attribute_captions
            ),

            "zero_relation_captions": (
                zero_relation_captions
            ),
        },
    }

    return report


# ============================================================
# Console output
# ============================================================

def print_report(
    report: Dict[str, Any],
    top_k: int,
) -> None:
    """
    Pretty-print important statistics.
    """

    quality = report[
        "pipeline_quality"
    ]

    sanitizer = report[
        "sanitizer"
    ]

    entities = report[
        "entities"
    ]

    attributes = report[
        "attributes"
    ]

    relations = report[
        "relations"
    ]

    print()
    print("=" * 72)
    print("Structured Semantics Statistics")
    print("=" * 72)

    print(
        f"Semantic source       : "
        f"{report['analysis']['semantic_source']}"
    )

    print(
        f"Analyzed samples      : "
        f"{report['analysis']['analyzed_semantics_count']}"
    )

    # ========================================================
    # Pipeline
    # ========================================================

    print()
    print("Pipeline quality:")

    print(
        f"  API success         : "
        f"{quality['api_success']} / "
        f"{quality['total_samples']} "
        f"({quality['api_success_rate_percent']:.2f}%)"
    )

    print(
        f"  Raw valid           : "
        f"{quality['raw_valid']} / "
        f"{quality['raw_valid'] + quality['raw_invalid']} "
        f"({quality['raw_valid_rate_percent']:.2f}%)"
    )

    print(
        f"  Final valid         : "
        f"{quality['final_valid']} / "
        f"{quality['final_valid'] + quality['final_invalid']} "
        f"({quality['final_valid_rate_percent']:.2f}%)"
    )

    print(
        f"  Warning samples     : "
        f"{quality['warning_samples']}"
    )

    print(
        f"  Error samples       : "
        f"{quality['error_samples']}"
    )

    # ========================================================
    # Sanitizer
    # ========================================================

    print()
    print("Sanitizer:")

    print(
        f"  Changed samples     : "
        f"{sanitizer['changed_samples']} "
        f"({sanitizer['change_rate_percent']:.2f}%)"
    )

    print(
        f"  Total actions       : "
        f"{sanitizer['total_actions']}"
    )

    print(
        f"  Dropped entities    : "
        f"{sanitizer['dropped_entities']}"
    )

    print(
        f"  Dropped attributes  : "
        f"{sanitizer['dropped_attributes']}"
    )

    print(
        f"  Dropped relations   : "
        f"{sanitizer['dropped_relations']}"
    )

    print(
        f"  Reassigned IDs      : "
        f"{sanitizer['reassigned_entity_ids']}"
    )

    if sanitizer[
        "action_frequency"
    ]:

        print(
            "  Action frequency:"
        )

        for name, count in sanitizer[
            "action_frequency"
        ].items():

            print(
                f"    {name:<32} "
                f"{count}"
            )

    # ========================================================
    # Entity
    # ========================================================

    print()
    print("Entities:")

    print(
        f"  Total               : "
        f"{entities['total']}"
    )

    print(
        f"  Unique phrases      : "
        f"{entities['unique_phrases']}"
    )

    print(
        f"  Singleton phrases   : "
        f"{entities['singleton_phrases']} "
        f"({entities['singleton_ratio_percent']:.2f}%)"
    )

    print(
        f"  Avg / caption       : "
        f"{entities['avg_per_caption']:.4f}"
    )

    print(
        f"  Duplicate-text caps : "
        f"{entities['duplicate_entity_text_samples']}"
    )

    print()
    print(
        f"Top-{top_k} entities:"
    )

    for item in entities["top"]:

        print(
            f"  {item['text']:<35} "
            f"{item['count']}"
        )

    # ========================================================
    # Attribute
    # ========================================================

    print()
    print("Attributes:")

    print(
        f"  Total               : "
        f"{attributes['total']}"
    )

    print(
        f"  Unique types        : "
        f"{attributes['unique_types']}"
    )

    print(
        f"  Unique values       : "
        f"{attributes['unique_values']}"
    )

    print(
        f"  Avg / caption       : "
        f"{attributes['avg_per_caption']:.4f}"
    )

    print(
        f"  Avg / entity        : "
        f"{attributes['avg_per_entity']:.4f}"
    )

    print(
        f"  Attribute coverage  : "
        f"{attributes['coverage_percent']:.2f}%"
    )

    print()
    print("Attribute type frequency:")

    for attr_type, count in (
        attributes[
            "type_frequency"
        ].items()
    ):

        print(
            f"  {attr_type:<25} "
            f"{count}"
        )

    # ========================================================
    # Relation
    # ========================================================

    print()
    print("Relations:")

    print(
        f"  Total               : "
        f"{relations['total']}"
    )

    print(
        f"  Unique predicates   : "
        f"{relations['unique_predicates']}"
    )

    print(
        f"  Singleton predicates: "
        f"{relations['singleton_predicates']} "
        f"({relations['singleton_ratio_percent']:.2f}%)"
    )

    print(
        f"  Avg / caption       : "
        f"{relations['avg_per_caption']:.4f}"
    )

    print(
        f"  Relation coverage   : "
        f"{relations['coverage_percent']:.2f}%"
    )

    print(
        f"  Zero-relation caps  : "
        f"{relations['zero_relation_captions']}"
    )

    print()
    print(
        f"Top-{top_k} raw relation predicates:"
    )

    for item in relations[
        "top"
    ]:

        print(
            f"  {item['predicate']:<35} "
            f"{item['count']}"
        )

    if relations[
        "singleton_predicate_list"
    ]:

        print()
        print(
            "Singleton raw predicates:"
        )

        for predicate in relations[
            "singleton_predicate_list"
        ]:

            print(
                f"  {predicate}"
            )

    print()
    print("=" * 72)


# ============================================================
# Saving report
# ============================================================

def save_report(
    report: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save statistics as JSON.
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
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze open-vocabulary "
            "Entity-Attribute-Relation "
            "extraction results."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Extraction JSON generated "
            "by extract.py."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional statistics JSON output."
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=[
            "raw",
            "sanitized",
        ],
        default="sanitized",
        help=(
            "Analyze raw or sanitized EAR."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help=(
            "Number of top vocabulary items "
            "to print."
        ),
    )

    return parser


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    data = load_extraction_file(
        args.input
    )

    report = analyze_extraction(
        data=data,
        source=args.source,
        top_k=args.top_k,
    )

    print_report(
        report=report,
        top_k=args.top_k,
    )

    if args.output:

        save_report(
            report=report,
            output_path=args.output,
        )

        print(
            f"Statistics saved to: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()