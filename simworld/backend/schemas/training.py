"""Pydantic schemas for Training / ML endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TrainingDataResponse(BaseModel):
    """Serialised training data record."""

    id: uuid.UUID
    simulation_id: uuid.UUID
    category: str
    input_prompt: str
    output_completion: str
    quality_score: float
    used_in_training: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class CategoryTrainingStatus(BaseModel):
    """Training status for a single category."""

    category: str
    total_records: int
    unused_records: int
    avg_quality_score: float
    threshold: int
    ready_for_training: bool
    currently_training: bool
    graduated: bool
    last_trained_at: Optional[datetime] = None


class TrainingStatusResponse(BaseModel):
    """Aggregate training status across all categories."""

    categories: list[CategoryTrainingStatus]
    total_records: int
    total_categories_graduated: int


class TrainingJobResponse(BaseModel):
    """Status of a training job."""

    job_id: str
    category: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_used: int = 0
    final_score: Optional[float] = None
