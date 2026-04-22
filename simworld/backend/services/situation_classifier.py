"""Situation Classifier — keyword scoring + NVIDIA NIM fallback."""

from __future__ import annotations

import json
import logging
from typing import Literal

import httpx

from core.config import get_settings
from core.redis import get_cached, set_cached

logger = logging.getLogger(__name__)

Category = Literal["finance", "corporate", "crisis", "social", "generic"]

CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    "finance": [
        "stock", "market", "trading", "portfolio", "investment", "bank",
        "hedge fund", "derivatives", "bonds", "crypto", "bitcoin",
        "interest rate", "inflation", "recession", "dividend", "equity",
        "forex", "commodity", "ipo", "valuation", "revenue", "profit",
    ],
    "corporate": [
        "company", "ceo", "board", "merger", "acquisition", "startup",
        "layoff", "restructuring", "corporate", "employee", "management",
        "shareholder", "strategy", "leadership", "hr", "promotion",
        "office politics", "quarterly", "stakeholder", "governance",
    ],
    "crisis": [
        "disaster", "emergency", "pandemic", "earthquake", "flood",
        "hurricane", "terrorism", "security breach", "cyber attack",
        "outbreak", "evacuation", "nuclear", "collapse", "explosion",
        "hostage", "war", "conflict", "famine", "refugee", "riot",
    ],
    "social": [
        "community", "election", "protest", "social media", "public opinion",
        "culture", "education", "housing", "immigration", "healthcare",
        "inequality", "climate", "environment", "activism", "voting",
        "neighborhood", "family", "relationship", "dating", "school",
    ],
}

AMBIGUITY_THRESHOLD = 0.35


async def classify_situation(
    situation: str,
    simulation_id: str | None = None,
) -> Category:
    """Classify a situation into one of five categories.

    1. Check Redis cache first.
    2. Score by keyword frequency.
    3. If ambiguous, call NVIDIA NIM for classification.
    4. Cache the result.
    """
    # ── Cache check ───────────────────────────────────────────────────
    if simulation_id:
        cached = await get_cached(f"category:{simulation_id}")
        if cached and cached in ("finance", "corporate", "crisis", "social", "generic"):
            return cached  # type: ignore[return-value]

    # ── Keyword scoring ───────────────────────────────────────────────
    situation_lower = situation.lower()
    scores: dict[Category, float] = {}
    total_hits = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in situation_lower)
        scores[category] = hits
        total_hits += hits

    if total_hits > 0:
        for cat in scores:
            scores[cat] /= total_hits

    best_category: Category = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_category]

    # ── If strong match, use keyword result ────────────────────────────
    if best_score >= AMBIGUITY_THRESHOLD and total_hits >= 2:
        result = best_category
    else:
        # ── NVIDIA NIM fallback ───────────────────────────────────────
        result = await _classify_via_nim(situation)

    # ── Cache ─────────────────────────────────────────────────────────
    if simulation_id:
        await set_cached(f"category:{simulation_id}", result, ttl=86400)

    logger.info(
        "Classified situation as '%s' (keyword_score=%.2f, hits=%d)",
        result, best_score, total_hits,
    )
    return result


async def _classify_via_nim(situation: str) -> Category:
    """Call NVIDIA NIM to classify an ambiguous situation."""
    settings = get_settings()

    system_prompt = (
        "You are a situation classifier. Classify the following situation "
        "into exactly ONE of these categories: finance, corporate, crisis, "
        "social, generic. Respond with ONLY the category name in lowercase, "
        "nothing else."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.NVIDIA_MODEL_TICK,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": situation},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 20,
                },
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"].strip().lower()

            valid: list[Category] = ["finance", "corporate", "crisis", "social", "generic"]
            if raw in valid:
                return raw  # type: ignore[return-value]

            # Try to extract a valid category from the response
            for cat in valid:
                if cat in raw:
                    return cat  # type: ignore[return-value]

            logger.warning("NIM returned unexpected category '%s', defaulting to generic", raw)
            return "generic"

    except Exception as exc:
        logger.error("NIM classification failed: %s — defaulting to generic", exc)
        return "generic"
