"""Simulation REST API routes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.deps import get_db_session
from core.redis import set_cached
from models.agent import Agent, AgentCreatedBy
from models.simulation import Simulation, SimulationStatus
from models.tick_log import TickLog
from schemas.simulation import (
    AgentResponse,
    PredictionResponse,
    SimulationCreate,
    SimulationCreateResponse,
    SimulationResponse,
    TickLogResponse,
)
from services.agent_service import generate_agents_for_situation
from services.situation_classifier import classify_situation
from services.simulation_engine import (
    pause_simulation,
    resume_simulation,
    start_simulation_loop,
    stop_simulation,
)
from ml.outcome_predictor import predict_outcome

router = APIRouter(prefix="/api/simulations", tags=["simulations"])


@router.post(
    "",
    response_model=SimulationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_simulation(
    body: SimulationCreate,
    db: AsyncSession = Depends(get_db_session),
) -> SimulationCreateResponse:
    """Create a new simulation.

    1. Classify the situation.
    2. Auto-generate agents if none provided.
    3. Persist simulation + agents.
    """
    # ── Classify ──────────────────────────────────────────────────────
    category = await classify_situation(body.situation)

    # ── Create simulation record ──────────────────────────────────────
    sim = Simulation(
        situation=body.situation,
        category=category,
        status=SimulationStatus.PENDING,
        world_state={"events": [], "tick": 0, "marketIndex": 0},
    )
    db.add(sim)
    await db.flush()

    # ── Cache category ────────────────────────────────────────────────
    await set_cached(f"category:{str(sim.id)}", category, ttl=86400)

    # ── Generate or use custom agents ─────────────────────────────────
    if body.custom_agents and len(body.custom_agents) > 0:
        agents_data = [a.model_dump() for a in body.custom_agents]
    else:
        agents_data = await generate_agents_for_situation(
            situation=body.situation,
            category=category,
            count=5,
        )

    agents: list[Agent] = []
    for a_data in agents_data:
        agent = Agent(
            simulation_id=sim.id,
            name=a_data["name"],
            role=a_data["role"],
            goal=a_data["goal"],
            personality=a_data.get("personality", {}),
            risk_tolerance=a_data.get("risk_tolerance", 0.5),
            color=a_data.get("color", "#6366f1"),
            created_by=(
                AgentCreatedBy.CUSTOM if body.custom_agents else AgentCreatedBy.AUTO
            ),
        )
        db.add(agent)
        agents.append(agent)

    await db.flush()

    agent_responses = [
        AgentResponse(
            id=a.id,
            name=a.name,
            role=a.role,
            goal=a.goal,
            personality=a.personality,
            emotional_state=a.emotional_state,
            color=a.color,
            created_by=a.created_by.value,
            risk_tolerance=a.risk_tolerance,
        )
        for a in agents
    ]

    return SimulationCreateResponse(
        simulation_id=sim.id,
        category=category,
        agents=agent_responses,
    )


@router.post("/{simulation_id}/start", status_code=status.HTTP_200_OK)
async def start_simulation(
    simulation_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Start a simulation's tick loop."""
    sim = await _get_simulation(db, simulation_id)

    if sim.status == SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is already running.",
        )
    if sim.status == SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is already completed.",
        )

    sim.status = SimulationStatus.RUNNING
    await db.flush()

    await start_simulation_loop(simulation_id)

    return {"status": "running", "simulation_id": str(simulation_id)}


@router.post("/{simulation_id}/pause", status_code=status.HTTP_200_OK)
async def pause_sim(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Pause a running simulation."""
    sim = await _get_simulation(db, simulation_id)

    if sim.status != SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot pause — simulation is {sim.status.value}.",
        )

    await pause_simulation(simulation_id)
    return {"status": "paused", "simulation_id": str(simulation_id)}


@router.post("/{simulation_id}/resume", status_code=status.HTTP_200_OK)
async def resume_sim(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Resume a paused simulation."""
    sim = await _get_simulation(db, simulation_id)

    if sim.status != SimulationStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot resume — simulation is {sim.status.value}.",
        )

    await resume_simulation(simulation_id)
    return {"status": "running", "simulation_id": str(simulation_id)}


@router.post("/{simulation_id}/stop", status_code=status.HTTP_200_OK)
async def stop_sim(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Stop a simulation permanently."""
    sim = await _get_simulation(db, simulation_id)

    if sim.status == SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation is already completed.",
        )

    await stop_simulation(simulation_id)
    return {"status": "completed", "simulation_id": str(simulation_id)}


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> SimulationResponse:
    """Get simulation details with agents."""
    sim = await _get_simulation(db, simulation_id)

    agent_responses = [
        AgentResponse(
            id=a.id,
            name=a.name,
            role=a.role,
            goal=a.goal,
            personality=a.personality,
            emotional_state=a.emotional_state,
            color=a.color,
            created_by=a.created_by.value,
            risk_tolerance=a.risk_tolerance,
        )
        for a in sim.agents
    ]

    return SimulationResponse(
        id=sim.id,
        situation=sim.situation,
        category=sim.category,
        status=sim.status.value,
        tick_count=sim.tick_count,
        world_state=sim.world_state,
        created_at=sim.created_at,
        completed_at=sim.completed_at,
        agents=agent_responses,
    )


@router.get("/{simulation_id}/logs", response_model=list[TickLogResponse])
async def get_simulation_logs(
    simulation_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session),
) -> list[TickLogResponse]:
    """Get tick logs for a simulation."""
    await _get_simulation(db, simulation_id)

    stmt = (
        select(TickLog)
        .where(TickLog.simulation_id == simulation_id)
        .order_by(TickLog.tick_number.desc(), TickLog.timestamp.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()

    return [
        TickLogResponse(
            id=log.id,
            simulation_id=log.simulation_id,
            agent_id=log.agent_id,
            tick_number=log.tick_number,
            perception=log.perception,
            thought=log.thought,
            action=log.action,
            emotional_state=log.emotional_state,
            world_state_snapshot=log.world_state_snapshot,
            timestamp=log.timestamp,
        )
        for log in logs
    ]


@router.get("/{simulation_id}/prediction", response_model=PredictionResponse)
async def get_prediction(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> PredictionResponse:
    """Get a 24-hour outcome prediction (requires ≥10 ticks)."""
    sim = await _get_simulation(db, simulation_id)

    if sim.tick_count < 10:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Need at least 10 ticks for prediction (current: {sim.tick_count}).",
        )

    agent_summaries = [
        {
            "name": a.name,
            "role": a.role,
            "emotional_state": a.emotional_state,
            "risk_tolerance": a.risk_tolerance,
        }
        for a in sim.agents
    ]

    result = await predict_outcome(
        simulation_id=simulation_id,
        situation=sim.situation,
        category=sim.category,
        tick_count=sim.tick_count,
        world_state=sim.world_state,
        agent_summaries=agent_summaries,
    )

    return PredictionResponse(
        simulation_id=simulation_id,
        tick_count=sim.tick_count,
        prediction=result["prediction"],
        confidence=result["confidence"],
        key_factors=result["key_factors"],
    )


async def _get_simulation(
    db: AsyncSession, simulation_id: uuid.UUID
) -> Simulation:
    """Load a simulation or raise 404."""
    stmt = (
        select(Simulation)
        .options(selectinload(Simulation.agents))
        .where(Simulation.id == simulation_id)
    )
    result = await db.execute(stmt)
    sim = result.scalar_one_or_none()

    if sim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found.",
        )
    return sim
