"""TrainingData ORM model — stores NIM fine-tuning examples."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from core.types import PortableUUID


class TrainingData(Base):
    """A single training example derived from simulation ticks."""

    __tablename__ = "training_data"

    id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, primary_key=True, default=uuid.uuid4
    )
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID,
        ForeignKey("simulations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="generic")
    input_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    output_completion: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    used_in_training: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # ── Relationships ─────────────────────────────────────────────────────
    simulation = relationship("Simulation", back_populates="training_data")

    def __repr__(self) -> str:
        return f"<TrainingData {self.id} cat={self.category} score={self.quality_score}>"
