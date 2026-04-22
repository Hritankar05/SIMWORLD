"""TickLog ORM model — stores every agent action per tick."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from core.types import PortableJSON, PortableUUID

if TYPE_CHECKING:
    from models.agent import Agent
    from models.simulation import Simulation


class TickLog(Base):
    """Immutable log entry for a single agent action within a tick."""

    __tablename__ = "tick_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, primary_key=True, default=uuid.uuid4
    )
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID,
        ForeignKey("simulations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tick_number: Mapped[int] = mapped_column(Integer, nullable=False)
    perception: Mapped[str] = mapped_column(Text, nullable=False, default="")
    thought: Mapped[str] = mapped_column(Text, nullable=False, default="")
    action: Mapped[str] = mapped_column(String(500), nullable=False, default="idle")
    emotional_state: Mapped[str] = mapped_column(
        String(100), nullable=False, default="neutral"
    )
    world_state_snapshot: Mapped[dict] = mapped_column(PortableJSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # ── Relationships ─────────────────────────────────────────────────────
    simulation: Mapped["Simulation"] = relationship(
        "Simulation", back_populates="tick_logs"
    )
    agent: Mapped["Agent"] = relationship("Agent", back_populates="tick_logs")

    def __repr__(self) -> str:
        return f"<TickLog sim={self.simulation_id} tick={self.tick_number} agent={self.agent_id}>"
