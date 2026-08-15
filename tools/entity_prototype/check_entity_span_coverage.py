import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import open_clip

from datasets.utils import pre_caption


# ============================================================
# Data loading
# ============================================================


def load_json(path: str) -> Any:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def get_records(data: Any) -> List[Dict[str, Any]]:
    """
    Support both:

    1. Frozen EAR source:
       {
           "samples": [...]
       }

    2. Entity training index:
       {
           "semantic_records": [...]
       }

    3. Plain list of records.
    """

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise TypeError(
            "Input JSON must be a list or dict."
        )

    if isinstance(
        data.get("samples"),
        list,
    ):
        return data["samples"]

    if isinstance(
        data.get("semantic_records"),
        list,
    ):
        return data["semantic_records"]

    raise KeyError(
        "Could not find record list. Expected one of: "
        "'samples', 'semantic_records', or a top-level list."
    )


def get_semantics(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prefer the final sanitized EAR semantics.

    Fallback keys are included so this diagnostic can also
    inspect older/sample files or the training index.
    """

    candidate_keys = (
        "sanitized_structured_semantics",
        "structured_semantics",
        "semantics",
    )

    for key in candidate_keys:
        value = record.get(key)

        if (
            isinstance(value, dict)
            and isinstance(
                value.get("entities"),
                list,
            )
        ):
            return value

    return {
        "entities": [],
        "relations": [],
    }


def get_entity_texts(
    record: Dict[str, Any],
) -> List[str]:
    semantics = get_semantics(
        record
    )

    result = []

    for entity in semantics.get(
        "entities",
        [],
    ):
        if isinstance(entity, str):
            text = entity

        elif isinstance(entity, dict):
            text = entity.get(
                "text",
                "",
            )

        else:
            continue

        if not isinstance(
            text,
            str,
        ):
            continue

        text = text.strip()

        if text:
            result.append(
                text
            )

    return result


# ============================================================
# CLIP token utilities
# ============================================================


def get_special_token_ids(
    tokenizer,
) -> Tuple[int, int]:
    """
    Resolve CLIP SOT/EOT token IDs without hard-coding them.
    """

    sot_id = getattr(
        tokenizer,
        "sot_token_id",
        None,
    )

    eot_id = getattr(
        tokenizer,
        "eot_token_id",
        None,
    )

    encoder = getattr(
        tokenizer,
        "encoder",
        None,
    )

    if (
        sot_id is None
        and isinstance(
            encoder,
            dict,
        )
    ):
        sot_id = encoder.get(
            "<start_of_text>"
        )

        if sot_id is None:
            sot_id = encoder.get(
                "<|startoftext|>"
            )

    if (
        eot_id is None
        and isinstance(
            encoder,
            dict,
        )
    ):
        eot_id = encoder.get(
            "<end_of_text>"
        )

        if eot_id is None:
            eot_id = encoder.get(
                "<|endoftext|>"
            )

    # Final robust fallback for OpenAI CLIP-style tokenizer:
    # infer SOT/EOT from tokenizing an empty string.
    if (
        sot_id is None
        or eot_id is None
    ):
        empty_tokens = tokenizer(
            [""]
        )[0].tolist()

        nonzero = [
            token
            for token in empty_tokens
            if token != 0
        ]

        if len(nonzero) >= 2:
            inferred_sot = nonzero[0]
            inferred_eot = nonzero[1]

            if sot_id is None:
                sot_id = inferred_sot

            if eot_id is None:
                eot_id = inferred_eot

    if (
        sot_id is None
        or eot_id is None
    ):
        raise RuntimeError(
            "Could not resolve CLIP SOT/EOT token IDs."
        )

    return (
        int(sot_id),
        int(eot_id),
    )


def tokenize_content(
    tokenizer,
    text: str,
    sot_id: int,
    eot_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize text and return:

        content_token_ids
        content_token_positions

    Positions are positions in the original CLIP token sequence.

    Example:
        [SOT, tok1, tok2, tok3, EOT, PAD, ...]

    returns:
        ids       = [tok1, tok2, tok3]
        positions = [1, 2, 3]
    """

    token_tensor = tokenizer(
        [text]
    )[0]

    if not torch.is_tensor(
        token_tensor
    ):
        token_tensor = torch.tensor(
            token_tensor,
            dtype=torch.long,
        )

    tokens = token_tensor.tolist()

    try:
        sot_pos = tokens.index(
            sot_id
        )
    except ValueError as exc:
        raise RuntimeError(
            "SOT token not found in tokenized text."
        ) from exc

    try:
        eot_pos = tokens.index(
            eot_id,
            sot_pos + 1,
        )
    except ValueError as exc:
        raise RuntimeError(
            "EOT token not found in tokenized text."
        ) from exc

    content_positions = list(
        range(
            sot_pos + 1,
            eot_pos,
        )
    )

    content_ids = [
        tokens[pos]
        for pos in content_positions
    ]

    return (
        content_ids,
        content_positions,
    )


def find_subsequence_starts(
    sequence: Sequence[int],
    subsequence: Sequence[int],
) -> List[int]:
    """
    Return every start index where `subsequence`
    occurs contiguously inside `sequence`.
    """

    sequence = list(
        sequence
    )

    subsequence = list(
        subsequence
    )

    if not subsequence:
        return []

    if len(subsequence) > len(
        sequence
    ):
        return []

    result = []

    last_start = (
        len(sequence)
        - len(subsequence)
    )

    for start in range(
        last_start + 1
    ):
        end = (
            start
            + len(subsequence)
        )

        if (
            sequence[start:end]
            == subsequence
        ):
            result.append(
                start
            )

    return result


# ============================================================
# Span analysis
# ============================================================


def normalize_for_dataset(
    text: str,
    max_words: int,
) -> Optional[str]:
    """
    Use exactly the same caption preprocessing used by
    the retrieval Dataset.

    Return None only when preprocessing yields invalid text.
    """

    try:
        return pre_caption(
            text,
            max_words,
        )
    except ValueError:
        return None


def analyze_entity_span(
    tokenizer,
    caption_raw: str,
    entity_raw: str,
    max_words: int,
    sot_id: int,
    eot_id: int,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # The caption actually seen by the current training model.
    # --------------------------------------------------------

    caption_train = (
        normalize_for_dataset(
            caption_raw,
            max_words,
        )
    )

    if caption_train is None:
        return {
            "status": "invalid_caption",
        }

    # --------------------------------------------------------
    # A non-truncated normalized caption is used only to
    # diagnose whether max_words truncation caused a miss.
    # --------------------------------------------------------

    caption_full = (
        normalize_for_dataset(
            caption_raw,
            100000,
        )
    )

    entity_clean = (
        normalize_for_dataset(
            entity_raw,
            100000,
        )
    )

    if entity_clean is None:
        return {
            "status": "invalid_entity",
        }

    # --------------------------------------------------------
    # Tokenize the actual training caption and the entity.
    # --------------------------------------------------------

    (
        caption_ids,
        caption_positions,
    ) = tokenize_content(
        tokenizer,
        caption_train,
        sot_id,
        eot_id,
    )

    (
        entity_ids,
        _,
    ) = tokenize_content(
        tokenizer,
        entity_clean,
        sot_id,
        eot_id,
    )

    if len(entity_ids) == 0:
        return {
            "status": "empty_entity_tokens",
            "caption_train": caption_train,
            "entity_clean": entity_clean,
        }

    starts = find_subsequence_starts(
        caption_ids,
        entity_ids,
    )

    # --------------------------------------------------------
    # Primary token-span match.
    # --------------------------------------------------------

    if starts:
        spans = []

        for relative_start in starts:
            relative_end = (
                relative_start
                + len(entity_ids)
            )

            clip_start = (
                caption_positions[
                    relative_start
                ]
            )

            # End-exclusive position.
            clip_end = (
                caption_positions[
                    relative_end - 1
                ]
                + 1
            )

            spans.append(
                [
                    int(clip_start),
                    int(clip_end),
                ]
            )

        status = (
            "unique"
            if len(spans) == 1
            else "multiple"
        )

        return {
            "status": status,
            "caption_train": caption_train,
            "entity_clean": entity_clean,
            "span_length": len(entity_ids),
            "num_matches": len(spans),
            "spans": spans,
        }

    # --------------------------------------------------------
    # Diagnose truncation.
    # --------------------------------------------------------

    if caption_full is not None:

        (
            full_caption_ids,
            _,
        ) = tokenize_content(
            tokenizer,
            caption_full,
            sot_id,
            eot_id,
        )

        full_starts = (
            find_subsequence_starts(
                full_caption_ids,
                entity_ids,
            )
        )

        if full_starts:
            return {
                "status": "truncated",
                "caption_train": caption_train,
                "caption_full": caption_full,
                "entity_clean": entity_clean,
                "span_length": len(entity_ids),
            }

    # --------------------------------------------------------
    # String-level diagnostics.
    # --------------------------------------------------------

    string_in_train = (
        entity_clean
        in caption_train
    )

    string_in_full = (
        caption_full is not None
        and entity_clean
        in caption_full
    )

    return {
        "status": "unmatched",
        "caption_train": caption_train,
        "caption_full": caption_full,
        "entity_clean": entity_clean,
        "string_in_train": bool(
            string_in_train
        ),
        "string_in_full": bool(
            string_in_full
        ),
        "span_length": len(entity_ids),
    }


# ============================================================
# Main
# ============================================================


def run(
    input_path: str,
    model_name: str,
    max_words: int,
    output_path: Optional[str],
    max_examples: int,
) -> Dict[str, Any]:

    data = load_json(
        input_path
    )

    records = get_records(
        data
    )

    tokenizer = (
        open_clip.get_tokenizer(
            model_name
        )
    )

    (
        sot_id,
        eot_id,
    ) = get_special_token_ids(
        tokenizer
    )

    status_counter = Counter()
    span_length_counter = Counter()

    total_entities = 0
    captions_with_entities = 0
    zero_entity_captions = 0

    examples_by_status: Dict[
        str,
        List[Dict[str, Any]],
    ] = {
        "multiple": [],
        "truncated": [],
        "unmatched": [],
        "invalid_caption": [],
        "invalid_entity": [],
        "empty_entity_tokens": [],
    }

    for record_index, record in enumerate(
        records
    ):

        caption = record.get(
            "normalized_caption"
        )

        if not isinstance(
            caption,
            str,
        ) or not caption.strip():

            caption = record.get(
                "caption",
                "",
            )

        if not isinstance(
            caption,
            str,
        ):
            caption = str(
                caption
            )

        entities = get_entity_texts(
            record
        )

        if entities:
            captions_with_entities += 1
        else:
            zero_entity_captions += 1

        for entity_index, entity in enumerate(
            entities
        ):

            total_entities += 1

            result = analyze_entity_span(
                tokenizer=tokenizer,
                caption_raw=caption,
                entity_raw=entity,
                max_words=max_words,
                sot_id=sot_id,
                eot_id=eot_id,
            )

            status = result[
                "status"
            ]

            status_counter[
                status
            ] += 1

            if (
                "span_length"
                in result
            ):
                span_length_counter[
                    int(
                        result[
                            "span_length"
                        ]
                    )
                ] += 1

            if (
                status
                in examples_by_status
                and len(
                    examples_by_status[
                        status
                    ]
                ) < max_examples
            ):
                examples_by_status[
                    status
                ].append(
                    {
                        "record_index": (
                            record_index
                        ),
                        "source_index": record.get(
                            "source_index"
                        ),
                        "caption": caption,
                        "entity_index": (
                            entity_index
                        ),
                        "entity": entity,
                        **result,
                    }
                )

    unique_count = (
        status_counter[
            "unique"
        ]
    )

    multiple_count = (
        status_counter[
            "multiple"
        ]
    )

    matched_any_count = (
        unique_count
        + multiple_count
    )

    unmatched_count = (
        total_entities
        - matched_any_count
    )

    safe_unique_coverage = (
        unique_count
        / total_entities
        if total_entities
        else 0.0
    )

    any_match_coverage = (
        matched_any_count
        / total_entities
        if total_entities
        else 0.0
    )

    summary = {
        "records": len(
            records
        ),
        "captions_with_entities": (
            captions_with_entities
        ),
        "zero_entity_captions": (
            zero_entity_captions
        ),
        "total_entities": (
            total_entities
        ),
        "unique_token_span_matches": (
            unique_count
        ),
        "multiple_token_span_matches": (
            multiple_count
        ),
        "matched_any": (
            matched_any_count
        ),
        "unmatched_total": (
            unmatched_count
        ),
        "safe_unique_coverage": (
            safe_unique_coverage
        ),
        "any_match_coverage": (
            any_match_coverage
        ),
        "status_counts": dict(
            status_counter
        ),
        "span_length_histogram": {
            str(key): value
            for key, value in sorted(
                span_length_counter.items()
            )
        },
    }

    report = {
        "metadata": {
            "input": str(
                Path(
                    input_path
                )
            ),
            "model_name": (
                model_name
            ),
            "max_words": (
                max_words
            ),
            "sot_token_id": (
                sot_id
            ),
            "eot_token_id": (
                eot_id
            ),
            "matching_rule": (
                "contiguous exact CLIP token subsequence "
                "after datasets.utils.pre_caption preprocessing"
            ),
        },
        "summary": summary,
        "examples": examples_by_status,
    }

    print()
    print("=" * 78)
    print(
        "ENTITY -> CAPTION TOKEN SPAN COVERAGE"
    )
    print("=" * 78)

    print(
        f"Input records               : "
        f"{len(records)}"
    )

    print(
        f"Captions with entities      : "
        f"{captions_with_entities}"
    )

    print(
        f"Zero-entity captions        : "
        f"{zero_entity_captions}"
    )

    print(
        f"Total entities              : "
        f"{total_entities}"
    )

    print()
    print(
        f"Unique token-span matches   : "
        f"{unique_count}"
    )

    print(
        f"Multiple token-span matches : "
        f"{multiple_count}"
    )

    print(
        f"Any token-span match        : "
        f"{matched_any_count}"
    )

    print(
        f"Unmatched total             : "
        f"{unmatched_count}"
    )

    print()
    print(
        f"Safe unique coverage        : "
        f"{safe_unique_coverage * 100:.4f}%"
    )

    print(
        f"Any-match coverage          : "
        f"{any_match_coverage * 100:.4f}%"
    )

    print()
    print(
        "Status counts:"
    )

    for key, value in sorted(
        status_counter.items()
    ):
        print(
            f"  {key:<20}: {value}"
        )

    print()
    print(
        "Entity token-span length histogram:"
    )

    for key, value in sorted(
        span_length_counter.items()
    ):
        print(
            f"  {key:>2} token(s): {value}"
        )

    # --------------------------------------------------------
    # Show representative failure/ambiguity examples.
    # --------------------------------------------------------

    for status in (
        "multiple",
        "truncated",
        "unmatched",
    ):
        examples = examples_by_status[
            status
        ]

        if not examples:
            continue

        print()
        print("-" * 78)
        print(
            f"{status.upper()} EXAMPLES "
            f"(showing {len(examples)})"
        )
        print("-" * 78)

        for item in examples:
            print(
                f"[record={item['record_index']}] "
                f"entity={item['entity']!r}"
            )
            print(
                f"  caption: "
                f"{item['caption']}"
            )

            if status == "multiple":
                print(
                    f"  spans  : "
                    f"{item.get('spans')}"
                )

            if status == "truncated":
                print(
                    f"  train  : "
                    f"{item.get('caption_train')}"
                )

            if status == "unmatched":
                print(
                    f"  cleaned entity : "
                    f"{item.get('entity_clean')!r}"
                )
                print(
                    f"  string_in_train: "
                    f"{item.get('string_in_train')}"
                )

    if output_path:
        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(
            f"Saved report: {output}"
        )

    print("=" * 78)

    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many EAR entity phrases can be "
            "mapped to exact contiguous CLIP token spans "
            "inside their training captions."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Frozen EAR JSON or entity training-index JSON."
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32-quickgelu",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=30,
        help=(
            "Must match dataset.max_words used during training."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional JSON report path."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help=(
            "Maximum examples saved/printed per failure type."
        ),
    )

    return parser


def main():
    args = (
        build_arg_parser()
        .parse_args()
    )

    run(
        input_path=args.input,
        model_name=args.model_name,
        max_words=args.max_words,
        output_path=args.output,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
