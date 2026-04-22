"""Agent ORM model."""

import enum
import uuid
from typing import TYPE_CHECKING, List

from sqlalchemy import Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from core.types import PortableJSON, PortableUUID

if TYPE_CHECKING:
    from models.simulation import Simulation
    from models.tick_log import TickLog


class AgentCreatedBy(str, enum.Enum):
    """How an agent was created."""

    AUTO = "auto"
    CUSTOM = "custom"


class Agent(Base):
    """An autonomous agent within a simulation."""

    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, primary_key=True, default=uuid.uuid4
    )
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID,
        ForeignKey("simulations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(200), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    personality: Mapped[dict] = mapped_column(PortableJSON, default=dict)
    emotional_state: Mapped[str] = mapped_column(
        String(100), nullable=False, default="neutral"
    )
    color: Mapped[str] = mapped_column(String(20), nullable=False, default="#6366f1")
    created_by: Mapped[AgentCreatedBy] = mapped_column(
        Enum(AgentCreatedBy, name="agent_created_by"),
        nullable=False,
        default=AgentCreatedBy.AUTO,
    )
    risk_tolerance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # ── Relationships ─────────────────────────────────────────────────────
    simulation: Mapped["Simulation"] = relationship(
        "Simulation", back_populates="agents"
    )
    tick_logs: Mapped[List["TickLog"]] = relationship(
        "TickLog", back_populates="agent", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Agent {self.name} role={self.role}>"
