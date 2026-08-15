import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredSemanticsCache:
    """
    Persistent JSONL cache for LLM structured semantics.

    Cache key is determined by:
        normalized_caption
        model
        prompt_version
        schema_version

    Only RAW structured semantics are cached.
    Sanitization and validation are intentionally NOT cached.
    """

    def __init__(
        self,
        cache_path: str,
        model: str,
        prompt_version: str,
        schema_version: str,
    ):
        self.cache_path = Path(cache_path)

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = model
        self.prompt_version = prompt_version
        self.schema_version = schema_version

        # cache_key -> record
        self._records: Dict[str, Dict[str, Any]] = {}

        self._load()

    # ========================================================
    # Cache key
    # ========================================================

    def make_key(
        self,
        normalized_caption: str,
    ) -> str:
        """
        Create deterministic SHA256 cache key.
        """

        payload = {
            "normalized_caption": normalized_caption,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # Load cache
    # ========================================================

    def _load(self) -> None:
        """
        Load existing JSONL cache into memory.

        Invalid lines are skipped rather than causing the whole
        extraction process to fail.
        """

        if not self.cache_path.exists():
            print(
                f"[Cache] New cache: "
                f"{self.cache_path}"
            )
            return

        loaded = 0
        skipped = 0

        with self.cache_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    skipped += 1

                    print(
                        f"[Cache warning] Invalid JSON "
                        f"at line {line_number}; skipped."
                    )

                    continue

                if not isinstance(record, dict):
                    skipped += 1
                    continue

                cache_key = record.get(
                    "cache_key"
                )

                if (
                    not isinstance(cache_key, str)
                    or not cache_key
                ):
                    skipped += 1
                    continue

                self._records[
                    cache_key
                ] = record

                loaded += 1

        print(
            f"[Cache] Loaded {loaded} records "
            f"from {self.cache_path}"
        )

        if skipped > 0:
            print(
                f"[Cache] Skipped {skipped} "
                f"invalid records."
            )

    # ========================================================
    # Lookup
    # ========================================================

    def has(
        self,
        normalized_caption: str,
    ) -> bool:
        """
        Return True if caption exists in cache.
        """

        key = self.make_key(
            normalized_caption
        )

        return key in self._records

    def get(
        self,
        normalized_caption: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return cached record or None.
        """

        key = self.make_key(
            normalized_caption
        )

        return self._records.get(
            key
        )

    # ========================================================
    # Write cache
    # ========================================================

    def put(
        self,
        normalized_caption: str,
        caption: str,
        raw_structured_semantics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Append one successful RAW EAR result.

        Duplicate cache records are not written twice.
        """

        key = self.make_key(
            normalized_caption
        )

        # Already cached.
        if key in self._records:
            return self._records[
                key
            ]

        record = {
            "cache_key": key,

            "normalized_caption": (
                normalized_caption
            ),

            "caption": caption,

            "model": self.model,

            "prompt_version": (
                self.prompt_version
            ),

            "schema_version": (
                self.schema_version
            ),

            "raw_structured_semantics": (
                raw_structured_semantics
            ),
        }

        # ----------------------------------------------------
        # Append immediately.
        #
        # If extraction crashes later, this successful API
        # result remains available for resume.
        # ----------------------------------------------------

        with self.cache_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            json.dump(
                record,
                f,
                ensure_ascii=False,
            )

            f.write("\n")

            # Force Python buffer flush.
            f.flush()

        self._records[
            key
        ] = record

        return record

    # ========================================================
    # Utility
    # ========================================================

    def __len__(self) -> int:
        return len(
            self._records
        )

    def stats(
        self,
    ) -> Dict[str, Any]:

        return {
            "cache_path": str(
                self.cache_path
            ),

            "num_records": len(
                self._records
            ),

            "model": self.model,

            "prompt_version": (
                self.prompt_version
            ),

            "schema_version": (
                self.schema_version
            ),
        }


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    test_cache = StructuredSemanticsCache(
        cache_path=(
            "tmp/structured_semantics/"
            "_cache_test.jsonl"
        ),

        model=(
            "qwen3.7-flash-2026-07-15"
        ),

        prompt_version="v3.0-open",
        schema_version="v1",
    )

    caption = (
        "many green trees are near "
        "several buildings ."
    )

    normalized_caption = (
        caption.strip().lower()
    )

    print()
    print(
        "Initial cache size:",
        len(test_cache),
    )

    print(
        "Has before put:",
        test_cache.has(
            normalized_caption
        ),
    )

    example_raw = {
        "entities": [
            {
                "id": "e1",
                "text": "trees",
                "attributes": [
                    {
                        "type": "count",
                        "value": "many",
                    },
                    {
                        "type": "color",
                        "value": "green",
                    },
                ],
            },
            {
                "id": "e2",
                "text": "buildings",
                "attributes": [
                    {
                        "type": "count",
                        "value": "several",
                    }
                ],
            },
        ],

        "relations": [
            {
                "subject": "e1",
                "predicate": "near",
                "object": "e2",
            }
        ],
    }

    test_cache.put(
        normalized_caption=normalized_caption,
        caption=caption,
        raw_structured_semantics=example_raw,
    )

    print(
        "Has after put:",
        test_cache.has(
            normalized_caption
        ),
    )

    print(
        "Final cache size:",
        len(test_cache),
    )

    print(
        json.dumps(
            test_cache.stats(),
            ensure_ascii=False,
            indent=2,
        )
    )