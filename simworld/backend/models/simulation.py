"""Simulation ORM model."""

import enum
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from core.types import PortableJSON, PortableUUID

if TYPE_CHECKING:
    from models.agent import Agent
    from models.tick_log import TickLog
    from models.training_data import TrainingData


class SimulationStatus(str, enum.Enum):
    """Lifecycle states for a simulation."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class Simulation(Base):
    """Represents a single simulation run."""

    __tablename__ = "simulations"

    id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, primary_key=True, default=uuid.uuid4
    )
    situation: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(
        String(50), nullable=False, default="generic"
    )
    status: Mapped[SimulationStatus] = mapped_column(
        Enum(SimulationStatus, name="simulation_status"),
        nullable=False,
        default=SimulationStatus.PENDING,
    )
    tick_count: Mapped[int] = mapped_column(Integer, default=0)
    world_state: Mapped[dict] = mapped_column(PortableJSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────
    agents: Mapped[List["Agent"]] = relationship(
        "Agent", back_populates="simulation", cascade="all, delete-orphan"
    )
    tick_logs: Mapped[List["TickLog"]] = relationship(
        "TickLog", back_populates="simulation", cascade="all, delete-orphan"
    )
    training_data: Mapped[List["TrainingData"]] = relationship(
        "TrainingData", back_populates="simulation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Simulation {self.id} [{self.status.value}] ticks={self.tick_count}>"
