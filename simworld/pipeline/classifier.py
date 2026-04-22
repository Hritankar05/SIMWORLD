"""Pipeline — Situation Classifier.

Classifies simulation scenarios into one of 5 categories using
keyword scoring with optional NVIDIA NIM confirmation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from pipeline.model_registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

VALID_CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]


class SituationClassifier:
    """Classifies a situation text into a domain category.

    Strategy:
    1. Score each category by keyword hits.
    2. If top score has 2+ keyword matches → confident, return immediately.
    3. If top score has exactly 1 match → ambiguous, call NVIDIA NIM to confirm.
    4. If no matches → return "generic".
    Results are cached by situation text to avoid repeat work.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key: str = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._cache: dict[str, str] = {}
        self._keyword_map: dict[str, list[str]] = {
            cat: entry["keywords"]
            for cat, entry in MODEL_REGISTRY.items()
        }

    def classify(self, situation: str) -> str:
        """Classify a situation string into a category.

        Returns one of: finance, corporate, crisis, social, generic.
        """
        # ── Cache check ───────────────────────────────────────────────
        cache_key = situation.strip().lower()[:500]
        if cache_key in self._cache:
            logger.debug("Cache hit for classification")
            return self._cache[cache_key]

        # ── Keyword scoring ───────────────────────────────────────────
        scores = self._score_keywords(situation)
        best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_category]

        logger.debug("Keyword scores: %s", scores)

        # ── Decision logic ────────────────────────────────────────────
        if best_score >= 2:
            # Confident — 2+ keyword matches
            result = best_category
            logger.info(
                "Classified as '%s' (confident, %d keyword hits)",
                result, best_score,
            )

        elif best_score == 1:
            # Ambiguous — 1 match, call NIM to confirm
            result = self._confirm_with_nim(situation, best_category)
            logger.info(
                "Classified as '%s' (NIM-confirmed, keyword hint was '%s')",
                result, best_category,
            )

        else:
            # No matches at all
            result = "generic"
            logger.info("Classified as 'generic' (no keyword matches)")

        self._cache[cache_key] = result
        return result

    def _score_keywords(self, situation: str) -> dict[str, int]:
        """Score each category by counting keyword hits in the situation."""
        text_lower = situation.lower()
        scores: dict[str, int] = {}

        for category, keywords in self._keyword_map.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = hits

        return scores

    def _confirm_with_nim(self, situation: str, hint: str) -> str:
        """Call NVIDIA NIM to confirm or correct an ambiguous classification.

        Falls back to the keyword hint if the API call fails.
        """
        if not self._api_key:
            logger.warning("No NVIDIA_API_KEY — using keyword hint '%s'", hint)
            return hint

        system_prompt = (
            "You are a situation classifier. Classify the following scenario "
            "into exactly ONE category: finance, corporate, crisis, social, generic.\n\n"
            "Definitions:\n"
            "- finance: stock markets, trading, banking, investments, monetary policy\n"
            "- corporate: companies, leadership, mergers, startups, governance\n"
            "- crisis: disasters, emergencies, pandemics, conflicts, evacuations\n"
            "- social: communities, elections, protests, education, public opinion\n"
            "- generic: anything that doesn't clearly fit the above\n\n"
            "Respond with ONLY the category name in lowercase. Nothing else."
        )

        try:
            # Using sync httpx since this may be called from non-async contexts
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    NVIDIA_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": NVIDIA_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": situation[:2000]},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 20,
                    },
                )
                response.raise_for_status()
                data = response.json()
                raw = data["choices"][0]["message"]["content"].strip().lower()

                # Extract valid category from response
                for cat in VALID_CATEGORIES:
                    if cat in raw:
                        return cat

                logger.warning("NIM returned '%s' — falling back to hint '%s'", raw, hint)
                return hint

        except Exception as exc:
            logger.error("NIM classification failed: %s — using hint '%s'", exc, hint)
            return hint

    def get_cache(self) -> dict[str, str]:
        """Return the current classification cache."""
        return self._cache.copy()

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()


# ── Unit Tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    classifier = SituationClassifier()

    test_cases = [
        (
            "The stock market crashed 30% after the Fed raised interest rates. "
            "Hedge funds are liquidating positions and retail investors are panicking.",
            "finance",
        ),
        (
            "The CEO was fired after a board vote. The company is restructuring "
            "and laying off 2000 employees. A merger with a competitor is being discussed.",
            "corporate",
        ),
        (
            "A 7.2 earthquake hit downtown. Emergency services are overwhelmed, "
            "buildings have collapsed, and evacuation routes are blocked.",
            "crisis",
        ),
        (
            "The community is organizing a protest against the new housing development. "
            "The mayor is facing a recall election and social media is exploding.",
            "social",
        ),
        (
            "Five people are trapped in an elevator and must decide how to ration "
            "their limited water supply over the next 24 hours.",
            "generic",
        ),
        (
            "A single keyword test: inflation is rising.",
            "finance",
        ),
        (
            "Something vague with no clear domain at all.",
            "generic",
        ),
    ]

    print("\n" + "=" * 60)
    print("SITUATION CLASSIFIER — UNIT TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for situation, expected in test_cases:
        result = classifier.classify(situation)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"\n  {status} Expected: {expected:12s} Got: {result:12s}")
        print(f"     {situation[:80]}...")

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"  Cache entries: {len(classifier.get_cache())}")
    print("=" * 60)
