"""Outcome Predictor — uses NVIDIA NIM to predict 24h simulation outcomes."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)


async def predict_outcome(
    simulation_id: uuid.UUID,
    situation: str,
    category: str,
    tick_count: int,
    world_state: dict[str, Any],
    agent_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a 24-hour outcome prediction using NVIDIA NIM.

    Only available after ≥10 ticks to ensure enough data.
    """
    if tick_count < 10:
        return {
            "simulation_id": str(simulation_id),
            "tick_count": tick_count,
            "prediction": "Insufficient data — at least 10 ticks required.",
            "confidence": 0.0,
            "key_factors": [],
        }

    settings = get_settings()

    events = world_state.get("events", [])[-15:]
    events_text = "\n".join(events) if events else "No recorded events."

    agents_text = "\n".join(
        f"- {a['name']} ({a['role']}): emotion={a.get('emotional_state', 'unknown')}"
        for a in agent_summaries
    )

    system_prompt = (
        "You are an expert simulation analyst specializing in predictive "
        "modeling. Given a running multi-agent simulation, predict the most "
        "likely outcome in the next 24 simulated hours. "
        "Respond ONLY in valid JSON with keys: "
        "prediction (string, 2-3 paragraphs), "
        "confidence (float 0-1), "
        "key_factors (array of strings, 3-5 factors driving the outcome)."
    )

    user_prompt = (
        f"Category: {category}\n"
        f"Situation: {situation}\n"
        f"Ticks completed: {tick_count}\n"
        f"Current agents:\n{agents_text}\n"
        f"Recent events:\n{events_text}\n"
        f"World state summary: {json.dumps(world_state)[:800]}\n\n"
        f"Predict the 24-hour outcome."
    )

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.NVIDIA_MODEL_PREDICT,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"].strip()

            return _parse_prediction(simulation_id, tick_count, raw)

    except Exception as exc:
        logger.error("Outcome prediction failed: %s", exc)
        return {
            "simulation_id": str(simulation_id),
            "tick_count": tick_count,
            "prediction": f"Prediction unavailable — API error: {exc}",
            "confidence": 0.0,
            "key_factors": ["API communication failure"],
        }


def _parse_prediction(
    simulation_id: uuid.UUID,
    tick_count: int,
    raw: str,
) -> dict[str, Any]:
    """Parse the raw NIM prediction response."""
    cleaned = raw.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    return {
        "simulation_id": str(simulation_id),
        "tick_count": tick_count,
        "prediction": parsed.get("prediction", raw[:500]),
        "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
        "key_factors": parsed.get("key_factors", ["Analysis based on simulation data"]),
    }
