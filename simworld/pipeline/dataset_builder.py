"""Pipeline — Dataset Builder.

Reads raw JSONL training data, deduplicates, splits into train/eval,
and writes clean dataset files for fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]


class DatasetBuilder:
    """Builds deduplicated, split datasets from raw JSONL training data."""

    def __init__(self, seed: int = 42, train_split: float = 0.90) -> None:
        self._seed = seed
        self._train_split = train_split

    def build(self, category: str) -> tuple[int, int]:
        """Build train + eval datasets for a single category.

        Steps:
        1. Read all records from data/{category}/training_data.jsonl
        2. Deduplicate by hashing input (user content) + output (assistant content)
        3. Shuffle deterministically
        4. Split 90/10 into train.jsonl and eval.jsonl
        5. Write output files

        Returns:
            (train_count, eval_count)
        """
        source_path = DATA_DIR / category / "training_data.jsonl"

        if not source_path.exists():
            logger.warning("No data file for category '%s'", category)
            return 0, 0

        # ── Read all records ──────────────────────────────────────────
        raw_records: list[dict[str, Any]] = []
        line_num = 0

        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if self._validate_record(record):
                        raw_records.append(record)
                    else:
                        logger.debug(
                            "Skipping invalid record at line %d in %s",
                            line_num, category,
                        )
                except json.JSONDecodeError:
                    logger.debug(
                        "Skipping malformed JSON at line %d in %s",
                        line_num, category,
                    )

        if not raw_records:
            logger.warning("No valid records for category '%s'", category)
            return 0, 0

        logger.info(
            "Category '%s': %d raw records read from %d lines",
            category, len(raw_records), line_num,
        )

        # ── Deduplicate ───────────────────────────────────────────────
        unique_records = self._deduplicate(raw_records)
        dedup_removed = len(raw_records) - len(unique_records)
        if dedup_removed > 0:
            logger.info(
                "Category '%s': removed %d duplicates (%d unique)",
                category, dedup_removed, len(unique_records),
            )

        # ── Shuffle ───────────────────────────────────────────────────
        rng = random.Random(self._seed)
        rng.shuffle(unique_records)

        # ── Split ─────────────────────────────────────────────────────
        split_idx = int(len(unique_records) * self._train_split)
        train_set = unique_records[:split_idx]
        eval_set = unique_records[split_idx:]

        # Ensure at least 1 eval sample if we have enough data
        if len(eval_set) == 0 and len(train_set) > 1:
            eval_set = [train_set.pop()]

        # ── Write output files ────────────────────────────────────────
        output_dir = DATA_DIR / category
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.jsonl"
        eval_path = output_dir / "eval.jsonl"

        self._write_jsonl(train_path, train_set)
        self._write_jsonl(eval_path, eval_set)

        logger.info(
            "Category '%s': wrote %d train, %d eval samples",
            category, len(train_set), len(eval_set),
        )

        return len(train_set), len(eval_set)

    def build_all(self) -> dict[str, dict[str, int]]:
        """Build datasets for all categories.

        Returns a dict mapping category → {"train": N, "eval": M}.
        """
        results: dict[str, dict[str, int]] = {}

        for category in CATEGORIES:
            train_count, eval_count = self.build(category)
            results[category] = {
                "train": train_count,
                "eval": eval_count,
                "total": train_count + eval_count,
            }

        return results

    def _validate_record(self, record: dict[str, Any]) -> bool:
        """Validate a training record has proper structure."""
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        roles = [m.get("role") for m in messages]
        if "user" not in roles or "assistant" not in roles:
            return False

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) == 0:
                return False

        return True

    def _deduplicate(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate records by hashing user input + assistant output."""
        seen_hashes: set[str] = set()
        unique: list[dict[str, Any]] = []

        for record in records:
            messages = record.get("messages", [])
            user_content = ""
            assistant_content = ""

            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")

            hash_input = f"{user_content}|||{assistant_content}"
            content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(record)

        return unique

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        """Write records to a JSONL file (overwrite mode)."""
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    builder = DatasetBuilder()
    results = builder.build_all()

    print("\n" + "=" * 60)
    print("DATASET BUILD REPORT")
    print("=" * 60)
    print(f"  {'Category':<14s} {'Train':>7s} {'Eval':>7s} {'Total':>7s}")
    print("  " + "-" * 40)

    total_train = 0
    total_eval = 0

    for cat, counts in results.items():
        total_train += counts["train"]
        total_eval += counts["eval"]
        print(
            f"  {cat:<14s} {counts['train']:>7d} {counts['eval']:>7d} {counts['total']:>7d}"
        )

    print("  " + "-" * 40)
    print(
        f"  {'TOTAL':<14s} {total_train:>7d} {total_eval:>7d} {total_train + total_eval:>7d}"
    )
    print("=" * 60)
