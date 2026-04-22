"""Simulation Engine — async tick loop with WebSocket push."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.config import get_settings
from core.database import async_session_factory
from core.redis import get_cached, publish_message, redis_client, set_cached
from models.agent import Agent
from models.simulation import Simulation, SimulationStatus
from schemas.agent import AgentPromptContext, AgentTickResult
from services.agent_service import process_agent_tick
from services.log_router import route_tick_result
from services.situation_classifier import classify_situation

logger = logging.getLogger(__name__)

# ── Active simulation loops ──────────────────────────────────────────
_active_loops: dict[str, asyncio.Task] = {}

# ── WebSocket connections ────────────────────────────────────────────
_ws_connections: dict[str, list[Any]] = {}

# ── Speed map ────────────────────────────────────────────────────────
SPEED_MAP: dict[str, float] = {
    "slow": 3.0,
    "normal": 1.5,
    "fast": 0.5,
}


def register_ws(simulation_id: str, ws: Any) -> None:
    """Register a WebSocket connection for a simulation."""
    if simulation_id not in _ws_connections:
        _ws_connections[simulation_id] = []
    _ws_connections[simulation_id].append(ws)
    logger.info("WS registered for sim %s (total: %d)", simulation_id, len(_ws_connections[simulation_id]))


def unregister_ws(simulation_id: str, ws: Any) -> None:
    """Unregister a WebSocket connection."""
    if simulation_id in _ws_connections:
        _ws_connections[simulation_id] = [
            c for c in _ws_connections[simulation_id] if c is not ws
        ]
        if not _ws_connections[simulation_id]:
            del _ws_connections[simulation_id]


async def broadcast_tick(simulation_id: str, payload: dict[str, Any]) -> None:
    """Broadcast a tick update to all connected WebSocket clients."""
    connections = _ws_connections.get(simulation_id, [])
    if not connections:
        return

    message = json.dumps(payload)
    dead: list[Any] = []

    for ws in connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)

    for ws in dead:
        unregister_ws(simulation_id, ws)


async def start_simulation_loop(simulation_id: uuid.UUID) -> None:
    """Start the async tick loop for a simulation as a background task."""
    sim_key = str(simulation_id)
    if sim_key in _active_loops and not _active_loops[sim_key].done():
        logger.warning("Simulation %s already has an active loop", sim_key)
        return

    task = asyncio.create_task(_tick_loop(simulation_id))
    _active_loops[sim_key] = task
    logger.info("Started tick loop for simulation %s", sim_key)


async def stop_simulation_loop(simulation_id: uuid.UUID) -> None:
    """Cancel the tick loop for a simulation."""
    sim_key = str(simulation_id)
    task = _active_loops.pop(sim_key, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("Stopped tick loop for simulation %s", sim_key)


async def _tick_loop(simulation_id: uuid.UUID) -> None:
    """Main simulation tick loop.

    Each tick:
    1. Check pause/resume flag in Redis
    2. Load simulation + agents from DB
    3. Process all agents concurrently via asyncio.gather
    4. Apply actions to world state
    5. Log results via log_router
    6. Push update via WebSocket
    7. Sleep for tick speed duration
    """
    settings = get_settings()
    sim_key = str(simulation_id)

    try:
        while True:
            # ── Check pause flag ──────────────────────────────────────
            paused = await get_cached(f"sim_paused:{sim_key}")
            if paused == "true":
                await asyncio.sleep(0.5)
                continue

            # ── Check speed override ──────────────────────────────────
            speed_override = await get_cached(f"sim_speed:{sim_key}")
            tick_delay = SPEED_MAP.get(speed_override or "normal", settings.tick_speed_seconds)

            # ── Load simulation ───────────────────────────────────────
            async with async_session_factory() as db:
                sim = await _load_simulation(db, simulation_id)
                if sim is None or sim.status == SimulationStatus.COMPLETED:
                    logger.info("Simulation %s not found or completed — exiting loop", sim_key)
                    break

                if sim.status != SimulationStatus.RUNNING:
                    await asyncio.sleep(0.5)
                    continue

                tick_number = sim.tick_count + 1
                # Build a list of "other agent" info for cross-referencing
                agent_info: list[dict[str, Any]] = []
                for agent in sim.agents:
                    agent_info.append({
                        "name": agent.name,
                        "role": agent.role,
                        "emotional_state": agent.emotional_state,
                        "last_action": "",
                        "last_message": "",
                    })

                # Populate last_action/last_message from recent events
                recent_events = _extract_recent_events(sim.world_state)
                for info in agent_info:
                    for ev in reversed(recent_events):
                        if ev.startswith(f"{info['name']} ("):
                            info["last_action"] = ev.split("): ", 1)[-1] if "): " in ev else ""
                            break

                # ── Classify on first tick ────────────────────────────
                if tick_number == 1:
                    category = await classify_situation(sim.situation, sim_key)
                    sim.category = category
                    await set_cached(f"category:{sim_key}", category, ttl=86400)

                # ── Get recent events ─────────────────────────────────
                world_summary = json.dumps(sim.world_state)[:1000]

                # ── Build agent contexts ──────────────────────────────
                contexts: list[AgentPromptContext] = []
                for agent in sim.agents:
                    ctx = AgentPromptContext(
                        agent_id=agent.id,
                        name=agent.name,
                        role=agent.role,
                        goal=agent.goal,
                        personality=agent.personality or {},
                        tick_number=tick_number,
                        situation=sim.situation,
                        world_state_summary=world_summary,
                        recent_events=recent_events,
                        risk_tolerance=agent.risk_tolerance,
                        emotional_state=agent.emotional_state,
                    )
                    contexts.append(ctx)

                # ── Assign talk targets (round-robin) ─────────────────
                talk_targets: list[str] = []
                for idx, agent in enumerate(sim.agents):
                    others = [a for a in sim.agents if a.id != agent.id]
                    target_idx = (tick_number + idx) % len(others) if others else 0
                    talk_targets.append(others[target_idx].name if others else "")

                # ── Build per-agent "other agents" context ────────────
                per_agent_others: list[list[dict[str, Any]]] = []
                for agent in sim.agents:
                    others = [info for info in agent_info if info["name"] != agent.name]
                    per_agent_others.append(others)

                # ── Process agents with rate limiting ─────────────────
                sem = asyncio.Semaphore(2)  # max 2 concurrent API calls

                async def _limited_tick(ctx, others, target):
                    async with sem:
                        result = await process_agent_tick(ctx, other_agents=others, talk_to=target)
                        await asyncio.sleep(0.3)  # small delay to avoid 429
                        return result

                results: list[AgentTickResult] = await asyncio.gather(
                    *[
                        _limited_tick(ctx, others, target)
                        for ctx, others, target in zip(contexts, per_agent_others, talk_targets)
                    ],
                    return_exceptions=False,
                )

                # ── Apply actions to world state ──────────────────────
                new_events: list[str] = []
                agent_updates: list[dict[str, Any]] = []

                for agent, result in zip(sim.agents, results):
                    event_text = f"{agent.name} ({agent.role}): {result.action}"
                    new_events.append(event_text)

                    agent.emotional_state = result.emotional_state

                    agent_updates.append({
                        "agentId": str(agent.id),
                        "agentName": agent.name,
                        "thought": result.thought,
                        "action": result.action,
                        "emotionalState": result.emotional_state,
                        "message": result.message,
                        "targetAgent": result.target_agent,
                    })

                    # ── Route to logs and training ────────────────────
                    await route_tick_result(
                        db=db,
                        simulation_id=simulation_id,
                        agent_id=agent.id,
                        agent_name=agent.name,
                        agent_role=agent.role,
                        tick_number=tick_number,
                        situation=sim.situation,
                        world_state=sim.world_state,
                        result=result,
                    )

                # ── Update world state ────────────────────────────────
                world_state = sim.world_state or {}
                existing_events = world_state.get("events", [])
                existing_events.extend(new_events)
                world_state["events"] = existing_events[-50:]  # Keep last 50
                world_state["tick"] = tick_number
                world_state["marketIndex"] = world_state.get("marketIndex", 0)

                sim.world_state = world_state
                sim.tick_count = tick_number
                await db.commit()

                # ── Broadcast via WebSocket ───────────────────────────
                ws_payload = {
                    "type": "tick_update",
                    "tick": tick_number,
                    "agentUpdates": agent_updates,
                    "worldState": world_state,
                    "newEvents": new_events,
                }
                await broadcast_tick(sim_key, ws_payload)

                logger.info("Tick %d completed for simulation %s", tick_number, sim_key)

            # ── Sleep ─────────────────────────────────────────────────
            await asyncio.sleep(tick_delay)

    except asyncio.CancelledError:
        logger.info("Tick loop cancelled for simulation %s", sim_key)
    except Exception as exc:
        logger.error("Tick loop error for simulation %s: %s", sim_key, exc, exc_info=True)
    finally:
        _active_loops.pop(sim_key, None)


async def inject_event(simulation_id: uuid.UUID, event: str) -> None:
    """Inject a breaking event into the simulation's world state."""
    async with async_session_factory() as db:
        sim = await _load_simulation(db, simulation_id)
        if sim is None:
            return
        world_state = sim.world_state or {}
        events = world_state.get("events", [])
        events.append(f"[INJECTED] {event}")
        world_state["events"] = events[-50:]
        sim.world_state = world_state
        await db.commit()
    logger.info("Injected event into simulation %s: %s", simulation_id, event[:100])


async def pause_simulation(simulation_id: uuid.UUID) -> None:
    """Pause a running simulation via Redis flag."""
    sim_key = str(simulation_id)
    await set_cached(f"sim_paused:{sim_key}", "true", ttl=86400)
    async with async_session_factory() as db:
        sim = await _load_simulation(db, simulation_id)
        if sim:
            sim.status = SimulationStatus.PAUSED
            await db.commit()
    await broadcast_tick(sim_key, {"type": "status_change", "status": "paused"})


async def resume_simulation(simulation_id: uuid.UUID) -> None:
    """Resume a paused simulation."""
    sim_key = str(simulation_id)
    await redis_client.delete(f"sim_paused:{sim_key}")
    async with async_session_factory() as db:
        sim = await _load_simulation(db, simulation_id)
        if sim:
            sim.status = SimulationStatus.RUNNING
            await db.commit()
    await broadcast_tick(sim_key, {"type": "status_change", "status": "running"})


async def stop_simulation(simulation_id: uuid.UUID) -> None:
    """Stop a simulation permanently — marks it as COMPLETED and kills the loop."""
    sim_key = str(simulation_id)

    # Cancel the tick loop task
    await stop_simulation_loop(simulation_id)

    # Clean up Redis flags
    await redis_client.delete(f"sim_paused:{sim_key}")
    await redis_client.delete(f"sim_speed:{sim_key}")

    # Update DB status
    async with async_session_factory() as db:
        sim = await _load_simulation(db, simulation_id)
        if sim:
            sim.status = SimulationStatus.COMPLETED
            sim.completed_at = datetime.now(timezone.utc)
            await db.commit()

    await broadcast_tick(sim_key, {"type": "status_change", "status": "completed"})


async def set_simulation_speed(simulation_id: uuid.UUID, speed: str) -> None:
    """Change the tick speed of a running simulation."""
    if speed not in SPEED_MAP:
        speed = "normal"
    sim_key = str(simulation_id)
    await set_cached(f"sim_speed:{sim_key}", speed, ttl=86400)


async def _load_simulation(
    db: AsyncSession, simulation_id: uuid.UUID
) -> Simulation | None:
    """Load a simulation with its agents eagerly loaded."""
    stmt = (
        select(Simulation)
        .options(selectinload(Simulation.agents))
        .where(Simulation.id == simulation_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def _extract_recent_events(world_state: dict[str, Any] | None) -> list[str]:
    """Get the last 3 events from world state."""
    if not world_state:
        return []
    events = world_state.get("events", [])
    return events[-3:]
