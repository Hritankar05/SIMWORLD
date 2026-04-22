"""Model Registry — tracks fine-tuned model availability and quality scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.redis import get_cached, set_cached

logger = logging.getLogger(__name__)

# Root models directory (project level)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ModelRegistry:
    """Registry for fine-tuned models per category."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    async def register_model(
        self,
        category: str,
        model_path: str,
        quality_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a newly trained model for a category."""
        entry = {
            "category": category,
            "model_path": model_path,
            "quality_score": quality_score,
            "metadata": metadata or {},
            "active": quality_score >= 0.80,
        }
        self._cache[category] = entry
        await set_cached(
            f"model_registry:{category}",
            json.dumps(entry),
            ttl=86400 * 7,
        )
        logger.info(
            "Registered model for '%s' — score=%.3f active=%s path=%s",
            category, quality_score, entry["active"], model_path,
        )

    async def get_model(self, category: str) -> dict[str, Any] | None:
        """Get the registered model for a category, if any."""
        # Check in-memory cache
        if category in self._cache:
            return self._cache[category]

        # Check Redis
        cached = await get_cached(f"model_registry:{category}")
        if cached:
            entry = json.loads(cached)
            self._cache[category] = entry
            return entry

        # Check filesystem
        model_dir = MODELS_DIR / category
        if model_dir.exists():
            meta_file = model_dir / "model_meta.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    entry = json.loads(f.read())
                self._cache[category] = entry
                await set_cached(
                    f"model_registry:{category}",
                    json.dumps(entry),
                    ttl=86400 * 7,
                )
                return entry

        return None

    async def is_model_available(self, category: str) -> bool:
        """Check if an active (graduated) model exists for a category."""
        model = await self.get_model(category)
        return model is not None and model.get("active", False)

    async def get_quality_score(self, category: str) -> float:
        """Get the quality score of the model for a category."""
        model = await self.get_model(category)
        if model is None:
            return 0.0
        return model.get("quality_score", 0.0)

    async def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        models: list[dict[str, Any]] = []
        categories = ["finance", "corporate", "crisis", "social", "generic"]

        for category in categories:
            model = await self.get_model(category)
            if model:
                models.append(model)
            else:
                models.append({
                    "category": category,
                    "model_path": None,
                    "quality_score": 0.0,
                    "active": False,
                    "metadata": {},
                })

        return models

    async def deactivate_model(self, category: str) -> None:
        """Deactivate a model (e.g., if quality degrades)."""
        if category in self._cache:
            self._cache[category]["active"] = False
            await set_cached(
                f"model_registry:{category}",
                json.dumps(self._cache[category]),
                ttl=86400 * 7,
            )
            logger.warning("Deactivated model for category '%s'", category)


# ── Singleton ─────────────────────────────────────────────────────────
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get or create the singleton ModelRegistry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
