"""Pydantic schemas for Agent endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentTickResult(BaseModel):
    """Result of a single agent's action in a tick."""

    agent_id: uuid.UUID
    thought: str = ""
    action: str = "idle"
    emotional_state: str = "neutral"
    message: str = ""
    target_agent: str = ""


class AgentPromptContext(BaseModel):
    """Context passed to the NVIDIA NIM for each agent's tick."""

    agent_id: uuid.UUID
    name: str
    role: str
    goal: str
    personality: dict[str, Any]
    tick_number: int
    situation: str
    world_state_summary: str
    recent_events: list[str] = Field(default_factory=list)
    risk_tolerance: float = 0.5
    emotional_state: str = "neutral"


class AgentUpdate(BaseModel):
    """Partial update for an agent."""

    name: str | None = None
    role: str | None = None
    goal: str | None = None
    personality: dict[str, Any] | None = None
    risk_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    color: str | None = None
