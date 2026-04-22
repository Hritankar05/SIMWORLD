"""Log Router — writes tick results to DB, JSONL files, and training table."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from core.config import get_settings
from core.redis import get_cached, publish_message, set_cached
from models.tick_log import TickLog
from models.training_data import TrainingData
from schemas.agent import AgentTickResult

logger = logging.getLogger(__name__)

# Project root → data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


async def route_tick_result(
    db: AsyncSession,
    simulation_id: uuid.UUID,
    agent_id: uuid.UUID,
    agent_name: str,
    agent_role: str,
    tick_number: int,
    situation: str,
    world_state: dict[str, Any],
    result: AgentTickResult,
) -> None:
    """Route a tick result to all three destinations.

    1. TickLog table in PostgreSQL
    2. data/{category}/training_data.jsonl
    3. TrainingData table
    Then check if training threshold is reached.
    """
    settings = get_settings()

    # ── Resolve category ──────────────────────────────────────────────
    category = await _get_category(str(simulation_id))

    # ── 1. Write TickLog ──────────────────────────────────────────────
    tick_log = TickLog(
        simulation_id=simulation_id,
        agent_id=agent_id,
        tick_number=tick_number,
        perception=f"Situation: {situation[:200]}",
        thought=result.thought,
        action=result.action,
        emotional_state=result.emotional_state,
        world_state_snapshot=world_state,
    )
    db.add(tick_log)

    # ── 2. Build training example ─────────────────────────────────────
    input_prompt = (
        f"Tick {tick_number}. Situation: {situation}. "
        f"World state: {json.dumps(world_state)}."
    )
    output_completion = json.dumps({
        "thought": result.thought,
        "action": result.action,
        "emotionalState": result.emotional_state,
        "message": result.message,
    })

    # ── 3. Write TrainingData ─────────────────────────────────────────
    training_entry = TrainingData(
        simulation_id=simulation_id,
        category=category,
        input_prompt=input_prompt,
        output_completion=output_completion,
    )
    db.add(training_entry)

    # ── 4. Write JSONL file ───────────────────────────────────────────
    await _write_jsonl(category, agent_name, agent_role, input_prompt, output_completion)

    # ── 5. Check training threshold ───────────────────────────────────
    await _check_training_threshold(db, category, settings.TRAINING_THRESHOLD)

    await db.flush()


async def _get_category(simulation_id: str) -> str:
    """Look up simulation category from Redis cache, default to 'generic'."""
    cached = await get_cached(f"category:{simulation_id}")
    if cached:
        return cached
    return "generic"


async def _write_jsonl(
    category: str,
    agent_name: str,
    agent_role: str,
    input_prompt: str,
    output_completion: str,
) -> None:
    """Append a training record as JSONL to the category data directory."""
    category_dir = DATA_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = category_dir / "training_data.jsonl"

    record = {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a {category} simulation agent named {agent_name}, "
                    f"acting as a {agent_role}. Respond in JSON with keys: "
                    f"thought, action, emotionalState, message."
                ),
            },
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": output_completion},
        ]
    }

    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("Failed to write JSONL for category %s: %s", category, exc)


async def _check_training_threshold(
    db: AsyncSession,
    category: str,
    threshold: int,
) -> None:
    """If unused records for a category exceed threshold, publish train signal.

    Uses a Redis cooldown key to avoid spamming train signals every tick.
    """
    # ── Cooldown check: only trigger once per 5 minutes per category ──
    cooldown_key = f"training_cooldown:{category}"
    cooldown_active = await get_cached(cooldown_key)
    if cooldown_active:
        return

    stmt = (
        select(func.count())
        .select_from(TrainingData)
        .where(
            TrainingData.category == category,
            TrainingData.used_in_training == False,  # noqa: E712
        )
    )
    result = await db.execute(stmt)
    count = result.scalar_one()

    if count >= threshold:
        logger.info(
            "Category '%s' hit training threshold (%d/%d) — publishing train signal",
            category, count, threshold,
        )
        await publish_message("training_channel", f"train:{category}")
        # Set cooldown for 5 minutes to avoid spam
        await set_cached(cooldown_key, "true", ttl=300)
