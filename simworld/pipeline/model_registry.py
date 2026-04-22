"""Pipeline — Model Registry.

Central registry of all 5 domain-specific model definitions.
Tracks paths, thresholds, record counts, and metadata for each category.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def _count_records(category: str) -> int:
    """Count JSONL records on disk for a category."""
    jsonl_path = DATA_DIR / category / "training_data.jsonl"
    if not jsonl_path.exists():
        return 0
    count = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _load_score(category: str) -> float:
    """Load the last evaluation score for a category."""
    score_path = MODELS_DIR / category / "score.txt"
    if not score_path.exists():
        return 0.0
    try:
        return float(score_path.read_text().strip())
    except (ValueError, OSError):
        return 0.0


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "finance": {
        "id": "simworld-finance-v1",
        "label": "Finance & Markets",
        "desc": (
            "Specialized in stock markets, trading, portfolio management, "
            "banking, hedge funds, derivatives, crypto, interest rates, "
            "inflation, and monetary policy simulations."
        ),
        "keywords": [
            "stock", "market", "trading", "portfolio", "investment", "bank",
            "hedge fund", "derivatives", "bonds", "crypto", "bitcoin",
            "interest rate", "inflation", "recession", "dividend", "equity",
            "forex", "commodity", "ipo", "valuation", "revenue", "profit",
            "bull", "bear", "volatility", "liquidity", "short selling",
            "options", "futures", "mutual fund", "etf", "fintech",
        ],
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "data_path": DATA_DIR / "finance",
        "model_path": MODELS_DIR / "finance",
        "score_path": MODELS_DIR / "finance" / "score.txt",
        "threshold": 100,
        "record_count": _count_records("finance"),
    },
    "corporate": {
        "id": "simworld-corporate-v1",
        "label": "Corporate & Business",
        "desc": (
            "Specialized in corporate governance, mergers & acquisitions, "
            "startups, layoffs, restructuring, leadership dynamics, "
            "shareholder disputes, and office politics simulations."
        ),
        "keywords": [
            "company", "ceo", "board", "merger", "acquisition", "startup",
            "layoff", "restructuring", "corporate", "employee", "management",
            "shareholder", "strategy", "leadership", "hr", "promotion",
            "office politics", "quarterly", "stakeholder", "governance",
            "founder", "venture capital", "board meeting", "executive",
            "supply chain", "competitor", "bankruptcy", "ipo", "revenue",
        ],
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "data_path": DATA_DIR / "corporate",
        "model_path": MODELS_DIR / "corporate",
        "score_path": MODELS_DIR / "corporate" / "score.txt",
        "threshold": 100,
        "record_count": _count_records("corporate"),
    },
    "crisis": {
        "id": "simworld-crisis-v1",
        "label": "Crisis & Emergency",
        "desc": (
            "Specialized in disaster response, pandemics, terrorism, "
            "natural disasters, evacuations, security breaches, military "
            "conflicts, hostage situations, and emergency management."
        ),
        "keywords": [
            "disaster", "emergency", "pandemic", "earthquake", "flood",
            "hurricane", "terrorism", "security breach", "cyber attack",
            "outbreak", "evacuation", "nuclear", "collapse", "explosion",
            "hostage", "war", "conflict", "famine", "refugee", "riot",
            "wildfire", "tsunami", "chemical spill", "blackout", "siege",
            "martial law", "quarantine", "triage", "rescue", "survivors",
        ],
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "data_path": DATA_DIR / "crisis",
        "model_path": MODELS_DIR / "crisis",
        "score_path": MODELS_DIR / "crisis" / "score.txt",
        "threshold": 100,
        "record_count": _count_records("crisis"),
    },
    "social": {
        "id": "simworld-social-v1",
        "label": "Social & Political",
        "desc": (
            "Specialized in community dynamics, elections, protests, "
            "social media influence, public opinion, education, housing, "
            "immigration, healthcare, and environmental activism."
        ),
        "keywords": [
            "community", "election", "protest", "social media", "public opinion",
            "culture", "education", "housing", "immigration", "healthcare",
            "inequality", "climate", "environment", "activism", "voting",
            "neighborhood", "family", "relationship", "school", "church",
            "rally", "campaign", "petition", "demonstration", "grassroots",
            "city council", "mayor", "zoning", "gentrification", "advocacy",
        ],
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "data_path": DATA_DIR / "social",
        "model_path": MODELS_DIR / "social",
        "score_path": MODELS_DIR / "social" / "score.txt",
        "threshold": 100,
        "record_count": _count_records("social"),
    },
    "generic": {
        "id": "simworld-generic-v1",
        "label": "General Purpose",
        "desc": (
            "Catch-all category for simulations that don't clearly fit "
            "into finance, corporate, crisis, or social domains. Handles "
            "mixed scenarios, hypothetical thought experiments, and "
            "cross-domain situations."
        ),
        "keywords": [
            "simulation", "scenario", "agent", "decision", "strategy",
            "negotiate", "cooperate", "compete", "resource", "team",
            "plan", "risk", "opportunity", "challenge", "outcome",
            "experiment", "hypothetical", "what if", "imagine", "role play",
        ],
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "data_path": DATA_DIR / "generic",
        "model_path": MODELS_DIR / "generic",
        "score_path": MODELS_DIR / "generic" / "score.txt",
        "threshold": 100,
        "record_count": _count_records("generic"),
    },
}


def get_registry() -> dict[str, dict[str, Any]]:
    """Return the full model registry with refreshed record counts."""
    for category in MODEL_REGISTRY:
        MODEL_REGISTRY[category]["record_count"] = _count_records(category)
    return MODEL_REGISTRY


def get_category(category: str) -> dict[str, Any]:
    """Get registry entry for a single category."""
    if category not in MODEL_REGISTRY:
        raise KeyError(f"Unknown category: {category}")
    entry = MODEL_REGISTRY[category].copy()
    entry["record_count"] = _count_records(category)
    return entry


def get_all_categories() -> list[str]:
    """Return list of all category names."""
    return list(MODEL_REGISTRY.keys())


def get_score(category: str) -> float:
    """Get the current evaluation score for a category."""
    return _load_score(category)


def is_graduated(category: str, quality_threshold: float = 0.80) -> bool:
    """Check if a category's model has graduated (score >= threshold)."""
    return _load_score(category) >= quality_threshold


def print_registry() -> None:
    """Print a formatted summary of the registry."""
    registry = get_registry()
    print("\n" + "=" * 70)
    print("SIMWORLD MODEL REGISTRY")
    print("=" * 70)
    for cat, entry in registry.items():
        score = _load_score(cat)
        graduated = "✅" if score >= 0.80 else "❌"
        print(f"\n  {graduated} {entry['label']} ({cat})")
        print(f"     ID:        {entry['id']}")
        print(f"     Base:      {entry['base_model']}")
        print(f"     Records:   {entry['record_count']} / {entry['threshold']}")
        print(f"     Score:     {score:.3f}")
        print(f"     Data:      {entry['data_path']}")
        print(f"     Model:     {entry['model_path']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_registry()
