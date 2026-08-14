from typing import Any, Dict, List, Set, Tuple


# ============================================================
# Generic EAR schema configuration
# ============================================================

# These are preferred high-level visual attribute categories.
#
# IMPORTANT:
# They are NOT a hard whitelist.
#
# Unknown attribute types will only produce a warning,
# not invalidate the sample.
#
# This keeps the validator dataset-agnostic while preserving
# our EAR representation design.
PREFERRED_ATTRIBUTE_TYPES = {
    "count",
    "size",
    "shape",
    "color",
    "density",
    "orientation",
    "position",
    "presence",
    "state",
    "other",
}


# Canonical values for attributes whose value space is naturally
# constrained.
#
# Again, these are soft constraints only.
PREFERRED_PRESENCE_VALUES = {
    "present",
    "absent",
}


# ============================================================
# Basic helpers
# ============================================================

def _is_nonempty_string(
    value: Any,
) -> bool:
    """
    Return True when value is a non-empty string.
    """

    return (
        isinstance(value, str)
        and len(value.strip()) > 0
    )


def _normalize_string(
    value: str,
) -> str:
    """
    Lightweight normalization used only during validation.
    """

    return (
        value
        .strip()
        .lower()
    )


def _expected_entity_ids(
    num_entities: int,
) -> List[str]:
    """
    For N entities, expected IDs are:

        e1, e2, ..., eN
    """

    return [
        f"e{i}"
        for i in range(
            1,
            num_entities + 1,
        )
    ]


# ============================================================
# Main validator
# ============================================================

def validate_structured_semantics(
    semantics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic dataset-agnostic validator for EAR structured semantics.

    This validator checks STRUCTURE, not semantic correctness.

    It does NOT assume:
    - a fixed relation vocabulary;
    - a fixed entity vocabulary;
    - a specific remote sensing dataset;
    - a manually defined ontology.

    ------------------------------------------------------------
    Expected format
    ------------------------------------------------------------

    {
        "entities": [
            {
                "id": "e1",
                "text": "trees",
                "attributes": [
                    {
                        "type": "color",
                        "value": "green"
                    }
                ]
            }
        ],

        "relations": [
            {
                "subject": "e1",
                "predicate": "near",
                "object": "e2"
            }
        ]
    }

    ------------------------------------------------------------
    Returns
    ------------------------------------------------------------

    {
        "valid": bool,

        "errors": [...],

        "warnings": [...],

        "stats": {
            "num_entities": int,
            "num_attributes": int,
            "num_relations": int
        }
    }

    ------------------------------------------------------------
    Philosophy
    ------------------------------------------------------------

    errors:
        Structural problems that make the annotation unsafe
        for downstream use.

    warnings:
        Structurally valid but potentially unusual cases.

    A warning does NOT make valid=False.
    """

    errors: List[str] = []
    warnings: List[str] = []

    # ========================================================
    # 1. Top-level validation
    # ========================================================

    if not isinstance(
        semantics,
        dict,
    ):
        return {
            "valid": False,

            "errors": [
                (
                    "structured_semantics must "
                    "be a JSON object"
                )
            ],

            "warnings": [],

            "stats": {
                "num_entities": 0,
                "num_attributes": 0,
                "num_relations": 0,
            },
        }

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    if "entities" not in semantics:
        errors.append(
            "missing top-level field: entities"
        )

    if "relations" not in semantics:
        errors.append(
            "missing top-level field: relations"
        )

    entities = semantics.get(
        "entities",
        [],
    )

    relations = semantics.get(
        "relations",
        [],
    )

    # --------------------------------------------------------
    # Field types
    # --------------------------------------------------------

    if not isinstance(
        entities,
        list,
    ):
        errors.append(
            "entities must be a list"
        )

        entities = []

    if not isinstance(
        relations,
        list,
    ):
        errors.append(
            "relations must be a list"
        )

        relations = []

    # ========================================================
    # 2. Entity validation
    # ========================================================

    entity_ids: List[str] = []

    entity_id_set: Set[str] = set()

    total_attributes = 0

    # Used only for duplicate entity warnings.
    normalized_entity_texts: List[str] = []

    for entity_index, entity in enumerate(
        entities
    ):

        prefix = (
            f"entities[{entity_index}]"
        )

        # ----------------------------------------------------
        # Entity must be an object
        # ----------------------------------------------------

        if not isinstance(
            entity,
            dict,
        ):
            errors.append(
                f"{prefix} must be an object"
            )

            continue

        # ====================================================
        # Entity ID
        # ====================================================

        entity_id = entity.get(
            "id"
        )

        if not _is_nonempty_string(
            entity_id
        ):
            errors.append(
                f"{prefix}.id must be "
                f"a non-empty string"
            )

        else:
            entity_id = (
                entity_id.strip()
            )

            entity_ids.append(
                entity_id
            )

            if (
                entity_id
                in entity_id_set
            ):
                errors.append(
                    f"duplicate entity id: "
                    f"{entity_id}"
                )

            entity_id_set.add(
                entity_id
            )

        # ====================================================
        # Entity text
        # ====================================================

        entity_text = entity.get(
            "text"
        )

        if not _is_nonempty_string(
            entity_text
        ):
            errors.append(
                f"{prefix}.text must be "
                f"a non-empty string"
            )

        else:
            normalized_entity_texts.append(
                _normalize_string(
                    entity_text
                )
            )

        # ====================================================
        # Attributes
        # ====================================================

        if "attributes" not in entity:
            errors.append(
                f"{prefix}.attributes is missing"
            )

            attributes = []

        else:
            attributes = entity.get(
                "attributes"
            )

            if not isinstance(
                attributes,
                list,
            ):
                errors.append(
                    f"{prefix}.attributes "
                    f"must be a list"
                )

                attributes = []

        total_attributes += len(
            attributes
        )

        seen_attributes: Set[
            Tuple[str, str]
        ] = set()

        for attr_index, attr in enumerate(
            attributes
        ):

            attr_prefix = (
                f"{prefix}."
                f"attributes[{attr_index}]"
            )

            # ------------------------------------------------
            # Attribute must be an object
            # ------------------------------------------------

            if not isinstance(
                attr,
                dict,
            ):
                errors.append(
                    f"{attr_prefix} "
                    f"must be an object"
                )

                continue

            # ------------------------------------------------
            # Attribute type
            # ------------------------------------------------

            attr_type = attr.get(
                "type"
            )

            if not _is_nonempty_string(
                attr_type
            ):
                errors.append(
                    f"{attr_prefix}.type must "
                    f"be a non-empty string"
                )

                normalized_type = None

            else:
                normalized_type = (
                    _normalize_string(
                        attr_type
                    )
                )

                # --------------------------------------------
                # Soft schema warning only
                # --------------------------------------------

                if (
                    normalized_type
                    not in
                    PREFERRED_ATTRIBUTE_TYPES
                ):
                    warnings.append(
                        f"non-preferred attribute type "
                        f"'{normalized_type}' "
                        f"in {attr_prefix}"
                    )

            # ------------------------------------------------
            # Attribute value
            # ------------------------------------------------

            attr_value = attr.get(
                "value"
            )

            if not _is_nonempty_string(
                attr_value
            ):
                errors.append(
                    f"{attr_prefix}.value must "
                    f"be a non-empty string"
                )

                normalized_value = None

            else:
                normalized_value = (
                    _normalize_string(
                        attr_value
                    )
                )

            # ------------------------------------------------
            # Duplicate attribute
            # ------------------------------------------------

            if (
                normalized_type is not None
                and
                normalized_value is not None
            ):
                attr_key = (
                    normalized_type,
                    normalized_value,
                )

                if (
                    attr_key
                    in seen_attributes
                ):
                    warnings.append(
                        f"duplicate attribute "
                        f"in {prefix}: "
                        f"{normalized_type}="
                        f"{normalized_value}"
                    )

                seen_attributes.add(
                    attr_key
                )

            # ------------------------------------------------
            # Presence soft sanity check
            # ------------------------------------------------

            if (
                normalized_type
                == "presence"
                and
                normalized_value is not None
                and
                normalized_value
                not in
                PREFERRED_PRESENCE_VALUES
            ):
                warnings.append(
                    f"non-standard presence value "
                    f"'{normalized_value}' "
                    f"in {attr_prefix}"
                )

    # ========================================================
    # 3. Entity ID sequence
    # ========================================================

    if len(
        entities
    ) == 0:

        warnings.append(
            "no entities extracted"
        )

    else:
        expected_ids = (
            _expected_entity_ids(
                len(entities)
            )
        )

        if (
            entity_ids
            != expected_ids
        ):
            warnings.append(
                "entity IDs are not sequential: "
                f"expected {expected_ids}, "
                f"got {entity_ids}"
            )

    # ========================================================
    # 4. Duplicate entity-text warning
    # ========================================================

    _check_duplicate_entity_texts(
        entity_texts=normalized_entity_texts,
        warnings=warnings,
    )

    # ========================================================
    # 5. Relation validation
    # ========================================================

    seen_relations: Set[
        Tuple[str, str, str]
    ] = set()

    for relation_index, relation in enumerate(
        relations
    ):

        prefix = (
            f"relations[{relation_index}]"
        )

        # ----------------------------------------------------
        # Relation must be an object
        # ----------------------------------------------------

        if not isinstance(
            relation,
            dict,
        ):
            errors.append(
                f"{prefix} must be an object"
            )

            continue

        # ====================================================
        # Subject
        # ====================================================

        subject = relation.get(
            "subject"
        )

        if not _is_nonempty_string(
            subject
        ):
            errors.append(
                f"{prefix}.subject must "
                f"be a non-empty string"
            )

            normalized_subject = None

        else:
            normalized_subject = (
                subject.strip()
            )

            if (
                normalized_subject
                not in entity_id_set
            ):
                errors.append(
                    f"{prefix}.subject "
                    f"references unknown "
                    f"entity id "
                    f"'{normalized_subject}'"
                )

        # ====================================================
        # Object
        # ====================================================

        obj = relation.get(
            "object"
        )

        if not _is_nonempty_string(
            obj
        ):
            errors.append(
                f"{prefix}.object must "
                f"be a non-empty string"
            )

            normalized_object = None

        else:
            normalized_object = (
                obj.strip()
            )

            if (
                normalized_object
                not in entity_id_set
            ):
                errors.append(
                    f"{prefix}.object "
                    f"references unknown "
                    f"entity id "
                    f"'{normalized_object}'"
                )

        # ====================================================
        # Predicate
        # ====================================================

        predicate = relation.get(
            "predicate"
        )

        if not _is_nonempty_string(
            predicate
        ):
            errors.append(
                f"{prefix}.predicate must "
                f"be a non-empty string"
            )

            normalized_predicate = None

        else:
            # IMPORTANT:
            #
            # No relation whitelist here.
            #
            # Any non-empty relation phrase is structurally valid.
            normalized_predicate = (
                _normalize_string(
                    predicate
                )
            )

        # ====================================================
        # Self relation
        # ====================================================

        if (
            normalized_subject is not None
            and
            normalized_object is not None
            and
            normalized_subject
            == normalized_object
        ):
            errors.append(
                f"{prefix} contains "
                f"self relation: "
                f"{normalized_subject} "
                f"-> "
                f"{normalized_object}"
            )

        # ====================================================
        # Duplicate relation
        # ====================================================

        if (
            normalized_subject is not None
            and
            normalized_predicate is not None
            and
            normalized_object is not None
        ):
            relation_key = (
                normalized_subject,
                normalized_predicate,
                normalized_object,
            )

            if (
                relation_key
                in seen_relations
            ):
                warnings.append(
                    "duplicate relation: "
                    f"{normalized_subject} "
                    f"--{normalized_predicate}--> "
                    f"{normalized_object}"
                )

            seen_relations.add(
                relation_key
            )

    # ========================================================
    # 6. Generic cross-field sanity checks
    # ========================================================

    _check_relations_without_entities(
        entities=entities,
        relations=relations,
        warnings=warnings,
    )

    _check_attribute_heavy_entities(
        entities=entities,
        warnings=warnings,
    )

    _check_relation_heavy_sample(
        entities=entities,
        relations=relations,
        warnings=warnings,
    )

    # ========================================================
    # 7. Final result
    # ========================================================

    return {
        "valid": (
            len(errors) == 0
        ),

        "errors": errors,

        "warnings": warnings,

        "stats": {
            "num_entities": len(
                entities
            ),

            "num_attributes": (
                total_attributes
            ),

            "num_relations": len(
                relations
            ),
        },
    }


# ============================================================
# Generic heuristic checks
# ============================================================

def _check_duplicate_entity_texts(
    entity_texts: List[str],
    warnings: List[str],
) -> None:
    """
    Detect repeated entity text.

    This is only a warning.

    Repeated semantic entity names may be legitimate, for example:

        red buildings
        white buildings

    after visual attributes are separated from entity text.

    Therefore duplicate text must NOT invalidate the sample.
    """

    counts: Dict[str, int] = {}

    for text in entity_texts:

        counts[text] = (
            counts.get(
                text,
                0,
            )
            + 1
        )

    for text, count in counts.items():

        if count > 1:
            warnings.append(
                f"duplicate entity text "
                f"'{text}' appears "
                f"{count} times"
            )


def _check_relations_without_entities(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    """
    Generic sanity check.

    Relations should normally not exist when fewer than
    two entities are extracted.
    """

    if (
        len(relations) > 0
        and
        len(entities) < 2
    ):
        warnings.append(
            "relations exist although fewer "
            "than two entities were extracted"
        )


def _check_attribute_heavy_entities(
    entities: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    """
    Detect unusually attribute-heavy entities.

    This is NOT dataset-specific.

    It only catches extreme outputs that may indicate
    parser instability.

    Threshold is intentionally loose.
    """

    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            continue

        entity_id = entity.get(
            "id",
            "?",
        )

        attributes = entity.get(
            "attributes",
            [],
        )

        if not isinstance(
            attributes,
            list,
        ):
            continue

        # Intentionally high threshold.
        #
        # This should only flag obviously unusual outputs,
        # not normal fine-grained descriptions.
        if len(attributes) > 10:
            warnings.append(
                f"entity {entity_id} has "
                f"an unusually large number "
                f"of attributes: "
                f"{len(attributes)}"
            )


def _check_relation_heavy_sample(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    """
    Detect extremely relation-heavy outputs.

    Maximum possible directed non-self relations for N entities:

        N * (N - 1)

    We do not enforce graph sparsity.

    This check only catches outputs whose relation count
    is structurally suspicious.
    """

    num_entities = len(
        entities
    )

    num_relations = len(
        relations
    )

    if num_entities < 2:
        return

    max_directed_relations = (
        num_entities
        * (
            num_entities - 1
        )
    )

    if (
        num_relations
        >
        max_directed_relations
    ):
        warnings.append(
            "relation count exceeds the "
            "number of possible unique "
            "directed non-self entity pairs: "
            f"{num_relations} relations for "
            f"{num_entities} entities"
        )


# ============================================================
# Convenience functions
# ============================================================

def is_valid_structured_semantics(
    semantics: Dict[str, Any],
) -> bool:
    """
    Convenience wrapper.

    Return only True / False.
    """

    result = (
        validate_structured_semantics(
            semantics
        )
    )

    return result["valid"]


def get_validation_errors(
    semantics: Dict[str, Any],
) -> List[str]:
    """
    Return structural errors only.
    """

    result = (
        validate_structured_semantics(
            semantics
        )
    )

    return result["errors"]


def get_validation_warnings(
    semantics: Dict[str, Any],
) -> List[str]:
    """
    Return warnings only.
    """

    result = (
        validate_structured_semantics(
            semantics
        )
    )

    return result["warnings"]