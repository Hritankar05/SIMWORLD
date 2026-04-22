"""Inference Router — decides between local fine-tuned model and NVIDIA NIM."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import get_settings
from ml.model_registry import get_model_registry
from schemas.agent import AgentPromptContext, AgentTickResult

logger = logging.getLogger(__name__)


class InferenceRouter:
    """Routes inference requests to local models or falls back to NIM API."""

    def __init__(self) -> None:
        self._registry = get_model_registry()

    async def try_local_inference(
        self, ctx: AgentPromptContext
    ) -> AgentTickResult | None:
        """Try to use a local fine-tuned model for inference.

        Returns None if no suitable local model exists, signaling the
        caller to fall back to NVIDIA NIM.
        """
        settings = get_settings()

        # ── Determine category from Redis cache ───────────────────────
        from core.redis import get_cached

        category = await get_cached(f"category:{str(ctx.agent_id)[:36]}")
        if not category:
            # Try simulation-level category (we don't have sim_id here,
            # so we check if any model is available for common categories)
            for cat in ["finance", "corporate", "crisis", "social", "generic"]:
                if await self._registry.is_model_available(cat):
                    score = await self._registry.get_quality_score(cat)
                    if score >= settings.MODEL_QUALITY_THRESHOLD:
                        category = cat
                        break

        if not category:
            return None

        # ── Check model availability and quality ──────────────────────
        if not await self._registry.is_model_available(category):
            return None

        score = await self._registry.get_quality_score(category)
        if score < settings.MODEL_QUALITY_THRESHOLD:
            logger.debug(
                "Local model for '%s' below threshold (%.2f < %.2f)",
                category, score, settings.MODEL_QUALITY_THRESHOLD,
            )
            return None

        # ── Run local inference ───────────────────────────────────────
        model_info = await self._registry.get_model(category)
        if model_info is None:
            return None

        logger.info("Using local fine-tuned model for category '%s'", category)
        result = await self._run_local_model(ctx, model_info)
        return result

    async def _run_local_model(
        self,
        ctx: AgentPromptContext,
        model_info: dict[str, Any],
    ) -> AgentTickResult | None:
        """Execute inference using a local fine-tuned model.

        This is a placeholder for actual model loading (e.g., via vLLM,
        llama.cpp, or a local FastAPI sidecar). In production, this would
        load the LoRA adapter and run inference.
        """
        model_path = model_info.get("model_path")
        if not model_path:
            return None

        try:
            # In a real deployment, this would call a local inference server.
            # For now, we simulate by returning a structured response that
            # indicates the local model was used — the actual integration
            # point for vLLM / TGI / llama.cpp goes here.
            logger.info(
                "Local inference for agent %s using model at %s",
                ctx.name, model_path,
            )

            # Return None to fall back to NIM — local inference requires
            # actual model deployment infrastructure.
            return None

        except Exception as exc:
            logger.warning("Local inference failed: %s — falling back to NIM", exc)
            return None


# ── Singleton ─────────────────────────────────────────────────────────
_router: InferenceRouter | None = None


def get_inference_router() -> InferenceRouter:
    """Get or create the singleton InferenceRouter."""
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router
