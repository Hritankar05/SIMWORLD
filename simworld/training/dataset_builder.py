"""Training — Dataset Builder (convenience wrapper).

Re-exports the full DatasetBuilder from the pipeline module
so training scripts can import from the training package directly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset_builder import DatasetBuilder  # noqa: E402

__all__ = ["DatasetBuilder"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    builder = DatasetBuilder()

    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        category = sys.argv[1]
        train_count, eval_count = builder.build(category)
        print(f"Built {category}: {train_count} train, {eval_count} eval")
    else:
        results = builder.build_all()
        print("\nDataset Build Results:")
        for cat, counts in results.items():
            print(f"  {cat}: {counts['train']} train, {counts['eval']} eval")
