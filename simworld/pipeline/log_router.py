"""Pipeline — Log Router.

Wraps LogCollector + SituationClassifier to provide a single interface
for routing simulation tick data to the correct category.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pipeline.classifier import SituationClassifier
from pipeline.log_collector import LogCollector
from pipeline.model_registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class LogRouter:
    """Routes simulation tick data through classification → collection.

    Caches classification results per simulation_id so that category
    lookup only happens once per simulation.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._classifier = SituationClassifier(
            api_key=api_key or os.getenv("NVIDIA_API_KEY", "")
        )
        self._collector = LogCollector()
        self._sim_categories: dict[str, str] = {}

    def classify_once(self, simulation_id: str, situation: str) -> str:
        """Classify a simulation's situation, caching the result.

        Only calls the classifier on the first invocation per simulation_id.
        Subsequent calls return the cached category.
        """
        if simulation_id in self._sim_categories:
            return self._sim_categories[simulation_id]

        category = self._classifier.classify(situation)
        self._sim_categories[simulation_id] = category
        logger.info(
            "Simulation %s classified as '%s'", simulation_id[:8], category
        )
        return category

    def route(
        self,
        simulation_id: str,
        situation: str,
        tick_number: int,
        agent_id: str,
        role: str,
        goal: str,
        perception: str,
        thought: str,
        action: str,
        emotional_state: str,
        world_state: dict[str, Any],
    ) -> None:
        """Classify (if needed) and route tick data to the correct category.

        This is the main entry point for the tick loop to call after
        each agent processes a tick.
        """
        category = self.classify_once(simulation_id, situation)

        self._collector.log_tick(
            simulation_id=simulation_id,
            category=category,
            situation=situation,
            tick_number=tick_number,
            agent_id=agent_id,
            role=role,
            goal=goal,
            perception=perception,
            thought=thought,
            action=action,
            emotional_state=emotional_state,
            world_state=world_state,
        )

    def check_threshold(self, category: str) -> bool:
        """Check if a category has reached its training threshold.

        Returns True if the number of records meets or exceeds
        the threshold defined in the model registry.
        """
        count = self._collector.get_record_count(category)
        threshold = MODEL_REGISTRY.get(category, {}).get("threshold", 100)
        ready = count >= threshold

        if ready:
            logger.info(
                "Category '%s' reached threshold: %d / %d",
                category, count, threshold,
            )

        return ready

    def get_counts(self) -> dict[str, dict[str, Any]]:
        """Get record counts and threshold status for all categories.

        Returns a dict like:
        {
            "finance": {"count": 45, "threshold": 100, "ready": False},
            "corporate": {"count": 120, "threshold": 100, "ready": True},
            ...
        }
        """
        result: dict[str, dict[str, Any]] = {}
        all_counts = self._collector.get_all_counts()

        for category in MODEL_REGISTRY:
            count = all_counts.get(category, 0)
            threshold = MODEL_REGISTRY[category].get("threshold", 100)
            result[category] = {
                "count": count,
                "threshold": threshold,
                "ready": count >= threshold,
                "progress": round(count / threshold * 100, 1) if threshold > 0 else 0,
            }

        return result

    def get_simulation_category(self, simulation_id: str) -> str | None:
        """Get the cached category for a simulation, if classified."""
        return self._sim_categories.get(simulation_id)

    def print_status(self) -> None:
        """Print a formatted status report."""
        counts = self.get_counts()
        print("\n" + "=" * 50)
        print("LOG ROUTER — STATUS")
        print("=" * 50)
        for cat, info in counts.items():
            bar_len = int(info["progress"] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            status = "✅ READY" if info["ready"] else "📦 Collecting"
            print(
                f"  {cat:12s} [{bar}] {info['count']:4d}/{info['threshold']:4d}  {status}"
            )
        print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    router = LogRouter()

    # Test classification + routing
    test_sim_id = "test-router-001"
    test_situation = (
        "A major bank is facing a liquidity crisis after bad derivative bets. "
        "The stock market is crashing and investors are fleeing to bonds."
    )

    category = router.classify_once(test_sim_id, test_situation)
    print(f"\nClassified as: {category}")

    router.route(
        simulation_id=test_sim_id,
        situation=test_situation,
        tick_number=1,
        agent_id="agent-test-001",
        role="Bank CEO",
        goal="Prevent bank collapse",
        perception="Liquidity is drying up fast",
        thought="Need to secure emergency credit lines",
        action="call_emergency_board_meeting",
        emotional_state="stressed",
        world_state={"marketIndex": -8, "tick": 1},
    )

    print(f"Threshold check (finance): {router.check_threshold('finance')}")
    router.print_status()
