"""Agent REST API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db_session
from models.agent import Agent
from schemas.agent import AgentUpdate
from schemas.simulation import AgentResponse

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> AgentResponse:
    """Get details for a single agent."""
    agent = await _get_agent(db, agent_id)
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        goal=agent.goal,
        personality=agent.personality,
        emotional_state=agent.emotional_state,
        color=agent.color,
        created_by=agent.created_by.value,
        risk_tolerance=agent.risk_tolerance,
    )


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    body: AgentUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> AgentResponse:
    """Partially update an agent."""
    agent = await _get_agent(db, agent_id)

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agent, field, value)

    await db.flush()

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        goal=agent.goal,
        personality=agent.personality,
        emotional_state=agent.emotional_state,
        color=agent.color,
        created_by=agent.created_by.value,
        risk_tolerance=agent.risk_tolerance,
    )


@router.get("/simulation/{simulation_id}", response_model=list[AgentResponse])
async def get_agents_by_simulation(
    simulation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> list[AgentResponse]:
    """List all agents for a simulation."""
    stmt = select(Agent).where(Agent.simulation_id == simulation_id)
    result = await db.execute(stmt)
    agents = result.scalars().all()

    if not agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No agents found for simulation {simulation_id}.",
        )

    return [
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


async def _get_agent(db: AsyncSession, agent_id: uuid.UUID) -> Agent:
    """Load an agent or raise 404."""
    stmt = select(Agent).where(Agent.id == agent_id)
    result = await db.execute(stmt)
    agent = result.scalar_one_or_none()

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found.",
        )
    return agent
