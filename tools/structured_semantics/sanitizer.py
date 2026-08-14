from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# Basic helpers
# ============================================================

def _is_nonempty_string(value: Any) -> bool:
    """
    Return True if value is a non-empty string.
    """
    return (
        isinstance(value, str)
        and len(value.strip()) > 0
    )


def _clean_string(value: str) -> str:
    """
    Strip leading/trailing whitespace only.

    IMPORTANT:
    This is NOT semantic normalization.
    """
    return value.strip()


# ============================================================
# Main sanitizer
# ============================================================

def sanitize_structured_semantics(
    semantics: Any,
) -> Dict[str, Any]:
    """
    Deterministically sanitize one EAR structured semantic object.

    This function performs STRUCTURAL repair only.

    It does NOT:
    - normalize relation semantics;
    - merge semantically similar predicates;
    - infer missing entities;
    - infer missing relations;
    - use dataset-specific vocabulary;
    - use world knowledge.

    ------------------------------------------------------------
    Input
    ------------------------------------------------------------

    Expected input:

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
    Output
    ------------------------------------------------------------

    {
        "sanitized": {
            "entities": [...],
            "relations": [...]
        },

        "report": {
            "changed": True / False,

            "actions": [...],

            "stats": {
                ...
            }
        }
    }
    """

    actions: List[str] = []

    # ========================================================
    # 1. Ensure top-level object
    # ========================================================

    if not isinstance(semantics, dict):
        actions.append(
            "input was not a JSON object; "
            "replaced with empty EAR structure"
        )

        return {
            "sanitized": {
                "entities": [],
                "relations": [],
            },

            "report": {
                "changed": True,

                "actions": actions,

                "stats": {
                    "input_entities": 0,
                    "output_entities": 0,

                    "input_attributes": 0,
                    "output_attributes": 0,

                    "input_relations": 0,
                    "output_relations": 0,

                    "dropped_entities": 0,
                    "dropped_attributes": 0,
                    "dropped_relations": 0,

                    "reassigned_entity_ids": 0,
                },
            },
        }

    semantics = deepcopy(semantics)

    # ========================================================
    # 2. Read entities / relations
    # ========================================================

    raw_entities = semantics.get(
        "entities",
        [],
    )

    raw_relations = semantics.get(
        "relations",
        [],
    )

    # --------------------------------------------------------
    # Missing / malformed entities
    # --------------------------------------------------------

    if "entities" not in semantics:
        actions.append(
            "missing top-level 'entities'; "
            "inserted empty list"
        )

    if not isinstance(raw_entities, list):
        actions.append(
            "top-level 'entities' was not a list; "
            "replaced with empty list"
        )

        raw_entities = []

    # --------------------------------------------------------
    # Missing / malformed relations
    # --------------------------------------------------------

    if "relations" not in semantics:
        actions.append(
            "missing top-level 'relations'; "
            "inserted empty list"
        )

    if not isinstance(raw_relations, list):
        actions.append(
            "top-level 'relations' was not a list; "
            "replaced with empty list"
        )

        raw_relations = []

    # ========================================================
    # 3. Input statistics
    # ========================================================

    input_entity_count = len(raw_entities)
    input_relation_count = len(raw_relations)

    input_attribute_count = 0

    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        attrs = entity.get(
            "attributes",
            [],
        )

        if isinstance(attrs, list):
            input_attribute_count += len(attrs)

    # ========================================================
    # 4. First pass: sanitize entities
    # ========================================================

    provisional_entities: List[
        Dict[str, Any]
    ] = []

    # Keeps original ID for later relation remapping.
    provisional_old_ids: List[
        Optional[str]
    ] = []

    dropped_entities = 0
    dropped_attributes = 0

    for entity_index, entity in enumerate(
        raw_entities
    ):

        # ----------------------------------------------------
        # Entity must be dict
        # ----------------------------------------------------

        if not isinstance(entity, dict):
            dropped_entities += 1

            actions.append(
                f"dropped entities[{entity_index}]: "
                f"entity was not an object"
            )

            continue

        # ----------------------------------------------------
        # Entity text
        # ----------------------------------------------------

        text = entity.get(
            "text"
        )

        if not _is_nonempty_string(text):
            dropped_entities += 1

            actions.append(
                f"dropped entities[{entity_index}]: "
                f"missing or empty entity text"
            )

            continue

        text = _clean_string(text)

        # ----------------------------------------------------
        # Original entity ID
        # ----------------------------------------------------

        raw_id = entity.get(
            "id"
        )

        if _is_nonempty_string(raw_id):
            old_id: Optional[str] = (
                _clean_string(raw_id)
            )
        else:
            old_id = None

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        if "attributes" not in entity:
            attributes = []

            actions.append(
                f"entities[{entity_index}] missing "
                f"'attributes'; inserted []"
            )

        else:
            attributes = entity.get(
                "attributes"
            )

            if not isinstance(
                attributes,
                list,
            ):
                actions.append(
                    f"entities[{entity_index}].attributes "
                    f"was not a list; replaced with []"
                )

                attributes = []

        clean_attributes: List[
            Dict[str, str]
        ] = []

        seen_attributes: Set[
            Tuple[str, str]
        ] = set()

        for attr_index, attr in enumerate(
            attributes
        ):

            if not isinstance(attr, dict):
                dropped_attributes += 1

                actions.append(
                    f"dropped entities[{entity_index}]"
                    f".attributes[{attr_index}]: "
                    f"attribute was not an object"
                )

                continue

            attr_type = attr.get(
                "type"
            )

            attr_value = attr.get(
                "value"
            )

            if not _is_nonempty_string(
                attr_type
            ):
                dropped_attributes += 1

                actions.append(
                    f"dropped entities[{entity_index}]"
                    f".attributes[{attr_index}]: "
                    f"missing or empty type"
                )

                continue

            if not _is_nonempty_string(
                attr_value
            ):
                dropped_attributes += 1

                actions.append(
                    f"dropped entities[{entity_index}]"
                    f".attributes[{attr_index}]: "
                    f"missing or empty value"
                )

                continue

            clean_type = _clean_string(
                attr_type
            )

            clean_value = _clean_string(
                attr_value
            )

            duplicate_key = (
                clean_type.lower(),
                clean_value.lower(),
            )

            # ----------------------------------------------
            # Exact duplicate attribute
            # ----------------------------------------------

            if duplicate_key in seen_attributes:
                dropped_attributes += 1

                actions.append(
                    f"dropped duplicate attribute "
                    f"in entities[{entity_index}]: "
                    f"{clean_type}={clean_value}"
                )

                continue

            seen_attributes.add(
                duplicate_key
            )

            clean_attributes.append(
                {
                    "type": clean_type,
                    "value": clean_value,
                }
            )

        provisional_entities.append(
            {
                "id": old_id,
                "text": text,
                "attributes": clean_attributes,
            }
        )

        provisional_old_ids.append(
            old_id
        )

    # ========================================================
    # 5. Detect ambiguous original IDs
    # ========================================================

    old_id_counts: Dict[str, int] = {}

    for old_id in provisional_old_ids:

        if old_id is None:
            continue

        old_id_counts[old_id] = (
            old_id_counts.get(
                old_id,
                0,
            )
            + 1
        )

    ambiguous_old_ids: Set[str] = {
        old_id
        for old_id, count
        in old_id_counts.items()
        if count > 1
    }

    for old_id in sorted(
        ambiguous_old_ids
    ):
        actions.append(
            f"original entity id '{old_id}' "
            f"was duplicated; relations referring "
            f"to this id will be treated as ambiguous"
        )

    # ========================================================
    # 6. Reassign sequential entity IDs
    # ========================================================

    clean_entities: List[
        Dict[str, Any]
    ] = []

    # Only unique old IDs can safely map to new IDs.
    old_to_new_id: Dict[
        str,
        str,
    ] = {}

    reassigned_entity_ids = 0

    for new_index, (
        entity,
        old_id,
    ) in enumerate(
        zip(
            provisional_entities,
            provisional_old_ids,
        ),
        start=1,
    ):

        new_id = f"e{new_index}"

        if old_id != new_id:
            reassigned_entity_ids += 1

            actions.append(
                f"reassigned entity id "
                f"'{old_id}' -> '{new_id}'"
            )

        clean_entity = {
            "id": new_id,
            "text": entity["text"],
            "attributes": entity[
                "attributes"
            ],
        }

        clean_entities.append(
            clean_entity
        )

        # ----------------------------------------------------
        # Only safe mappings are stored
        # ----------------------------------------------------

        if (
            old_id is not None
            and
            old_id
            not in ambiguous_old_ids
        ):
            old_to_new_id[
                old_id
            ] = new_id

    valid_new_ids: Set[str] = {
        entity["id"]
        for entity
        in clean_entities
    }

    # ========================================================
    # 7. Sanitize relations
    # ========================================================

    clean_relations: List[
        Dict[str, str]
    ] = []

    seen_relations: Set[
        Tuple[str, str, str]
    ] = set()

    dropped_relations = 0

    for relation_index, relation in enumerate(
        raw_relations
    ):

        # ----------------------------------------------------
        # Relation must be dict
        # ----------------------------------------------------

        if not isinstance(
            relation,
            dict,
        ):
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"relation was not an object"
            )

            continue

        subject = relation.get(
            "subject"
        )

        predicate = relation.get(
            "predicate"
        )

        obj = relation.get(
            "object"
        )

        # ----------------------------------------------------
        # Required relation fields
        # ----------------------------------------------------

        if not _is_nonempty_string(
            subject
        ):
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"missing or empty subject"
            )

            continue

        if not _is_nonempty_string(
            predicate
        ):
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"missing or empty predicate"
            )

            continue

        if not _is_nonempty_string(
            obj
        ):
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"missing or empty object"
            )

            continue

        subject = _clean_string(
            subject
        )

        predicate = _clean_string(
            predicate
        )

        obj = _clean_string(
            obj
        )

        # ----------------------------------------------------
        # Map original IDs to new sequential IDs
        # ----------------------------------------------------

        mapped_subject = _map_relation_endpoint(
            endpoint=subject,
            old_to_new_id=old_to_new_id,
            valid_new_ids=valid_new_ids,
            ambiguous_old_ids=(
                ambiguous_old_ids
            ),
        )

        mapped_object = _map_relation_endpoint(
            endpoint=obj,
            old_to_new_id=old_to_new_id,
            valid_new_ids=valid_new_ids,
            ambiguous_old_ids=(
                ambiguous_old_ids
            ),
        )

        # ----------------------------------------------------
        # Invalid / unknown endpoint
        # ----------------------------------------------------

        if mapped_subject is None:
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"subject '{subject}' does not "
                f"reference a unique valid entity"
            )

            continue

        if mapped_object is None:
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"object '{obj}' does not "
                f"reference a unique valid entity"
            )

            continue

        # ----------------------------------------------------
        # Self relation
        # ----------------------------------------------------

        if mapped_subject == mapped_object:
            dropped_relations += 1

            actions.append(
                f"dropped relations[{relation_index}]: "
                f"self relation "
                f"{mapped_subject} "
                f"--{predicate}--> "
                f"{mapped_object}"
            )

            continue

        # ----------------------------------------------------
        # Exact duplicate relation
        # ----------------------------------------------------

        duplicate_key = (
            mapped_subject,
            predicate.lower(),
            mapped_object,
        )

        if duplicate_key in seen_relations:
            dropped_relations += 1

            actions.append(
                f"dropped duplicate relation: "
                f"{mapped_subject} "
                f"--{predicate}--> "
                f"{mapped_object}"
            )

            continue

        seen_relations.add(
            duplicate_key
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Predicate itself is preserved.
        #
        # No:
        # nearby -> near
        # in -> inside
        # has -> contain
        # etc.
        # ----------------------------------------------------

        clean_relations.append(
            {
                "subject": (
                    mapped_subject
                ),

                "predicate": (
                    predicate
                ),

                "object": (
                    mapped_object
                ),
            }
        )

    # ========================================================
    # 8. Final structure
    # ========================================================

    sanitized = {
        "entities": clean_entities,
        "relations": clean_relations,
    }

    output_attribute_count = sum(
        len(
            entity.get(
                "attributes",
                [],
            )
        )
        for entity
        in clean_entities
    )

    # ========================================================
    # 9. Report
    # ========================================================

    report = {
        "changed": (
            len(actions) > 0
        ),

        "actions": actions,

        "stats": {
            "input_entities": (
                input_entity_count
            ),

            "output_entities": len(
                clean_entities
            ),

            "input_attributes": (
                input_attribute_count
            ),

            "output_attributes": (
                output_attribute_count
            ),

            "input_relations": (
                input_relation_count
            ),

            "output_relations": len(
                clean_relations
            ),

            "dropped_entities": (
                dropped_entities
            ),

            "dropped_attributes": (
                dropped_attributes
            ),

            "dropped_relations": (
                dropped_relations
            ),

            "reassigned_entity_ids": (
                reassigned_entity_ids
            ),
        },
    }

    return {
        "sanitized": sanitized,
        "report": report,
    }


# ============================================================
# Relation endpoint mapping
# ============================================================

def _map_relation_endpoint(
    endpoint: str,
    old_to_new_id: Dict[str, str],
    valid_new_ids: Set[str],
    ambiguous_old_ids: Set[str],
) -> Optional[str]:
    """
    Safely map one relation endpoint.

    Rules:

    1. Ambiguous duplicate original IDs are rejected.

    2. If endpoint exists in old_to_new_id,
       map it to its new sequential ID.

    3. Otherwise, if endpoint already happens to be
       a valid new sequential ID, preserve it.

    4. Otherwise return None.

    No semantic guessing is allowed.
    """

    if endpoint in ambiguous_old_ids:
        return None

    if endpoint in old_to_new_id:
        return old_to_new_id[
            endpoint
        ]

    if endpoint in valid_new_ids:
        return endpoint

    return None


# ============================================================
# Convenience wrapper
# ============================================================

def get_sanitized_semantics(
    semantics: Any,
) -> Dict[str, Any]:
    """
    Return sanitized EAR only,
    without the sanitization report.
    """

    result = (
        sanitize_structured_semantics(
            semantics
        )
    )

    return result["sanitized"]


# ============================================================
# Simple smoke test
# ============================================================

if __name__ == "__main__":

    example = {
        "entities": [
            {
                "id": "entity_a",
                "text": "trees",
                "attributes": [
                    {
                        "type": "color",
                        "value": "green",
                    }
                ],
            },

            {
                "id": "entity_b",
                "text": "river",
                # intentionally missing attributes
            },
        ],

        "relations": [
            {
                "subject": "entity_a",
                "predicate": "nearby",
                "object": "entity_b",
            },

            {
                "subject": "entity_a",
                "predicate": "nearby",
                "object": "entity_b",
            },

            {
                "subject": "entity_a",
                "predicate": "provide",
                "object": "convenience",
            },
        ],
    }

    result = (
        sanitize_structured_semantics(
            example
        )
    )

    from pprint import pprint

    pprint(result)