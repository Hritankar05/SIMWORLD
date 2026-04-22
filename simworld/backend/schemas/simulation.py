"""Pydantic schemas for Simulation endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SimulationCreate(BaseModel):
    """Request body to create a new simulation."""

    situation: str = Field(..., min_length=10, max_length=5000)
    custom_agents: Optional[list["AgentCreate"]] = None


class AgentCreate(BaseModel):
    """Minimal agent definition for custom agent injection."""

    name: str = Field(..., min_length=1, max_length=200)
    role: str = Field(..., min_length=1, max_length=200)
    goal: str = Field(..., min_length=1, max_length=2000)
    personality: dict[str, Any] = Field(default_factory=dict)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    color: str = Field(default="#6366f1")


class SimulationResponse(BaseModel):
    """Serialised simulation returned to clients."""

    id: uuid.UUID
    situation: str
    category: str
    status: str
    tick_count: int
    world_state: dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    agents: list["AgentResponse"] = []

    model_config = {"from_attributes": True}


class AgentResponse(BaseModel):
    """Serialised agent within a simulation response."""

    id: uuid.UUID
    name: str
    role: str
    goal: str
    personality: dict[str, Any]
    emotional_state: str
    color: str
    created_by: str
    risk_tolerance: float

    model_config = {"from_attributes": True}


class TickLogResponse(BaseModel):
    """Serialised tick log entry."""

    id: uuid.UUID
    simulation_id: uuid.UUID
    agent_id: uuid.UUID
    tick_number: int
    perception: str
    thought: str
    action: str
    emotional_state: str
    world_state_snapshot: dict[str, Any]
    timestamp: datetime

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    """24-hour outcome prediction after ≥10 ticks."""

    simulation_id: uuid.UUID
    tick_count: int
    prediction: str
    confidence: float
    key_factors: list[str]


class SimulationCreateResponse(BaseModel):
    """Response after creating a simulation."""

    simulation_id: uuid.UUID
    category: str
    agents: list[AgentResponse]


# Rebuild forward refs
SimulationCreate.model_rebuild()
SimulationResponse.model_rebuild()
