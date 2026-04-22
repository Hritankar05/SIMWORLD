"""Training status REST API routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db_session
from core.config import get_settings
from core.redis import get_cached, redis_client
from ml.model_registry import get_model_registry
from models.training_data import TrainingData
from schemas.training import CategoryTrainingStatus, TrainingStatusResponse

router = APIRouter(prefix="/api/training", tags=["training"])

CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(
    db: AsyncSession = Depends(get_db_session),
) -> TrainingStatusResponse:
    """Return training status for all categories.

    Powers the personal dashboard with record counts, quality scores,
    and graduation status.
    """
    settings = get_settings()
    registry = get_model_registry()
    categories: list[CategoryTrainingStatus] = []
    total_records = 0
    total_graduated = 0

    for category in CATEGORIES:
        # ── Count total records ───────────────────────────────────────
        total_stmt = (
            select(func.count())
            .select_from(TrainingData)
            .where(TrainingData.category == category)
        )
        total_result = await db.execute(total_stmt)
        cat_total = total_result.scalar_one()

        # ── Count unused records ──────────────────────────────────────
        unused_stmt = (
            select(func.count())
            .select_from(TrainingData)
            .where(
                TrainingData.category == category,
                TrainingData.used_in_training == False,  # noqa: E712
            )
        )
        unused_result = await db.execute(unused_stmt)
        cat_unused = unused_result.scalar_one()

        # ── Average quality score ─────────────────────────────────────
        avg_stmt = (
            select(func.avg(TrainingData.quality_score))
            .where(TrainingData.category == category)
        )
        avg_result = await db.execute(avg_stmt)
        avg_score = avg_result.scalar_one() or 0.0

        # ── Check if currently training ───────────────────────────────
        currently_training = await redis_client.sismember("training_active", category)

        # ── Check graduation (model quality ≥ threshold) ──────────────
        model_score = await registry.get_quality_score(category)
        graduated = model_score >= settings.MODEL_QUALITY_THRESHOLD

        # ── Get last training timestamp ───────────────────────────────
        job_raw = await get_cached(f"training_job:{category}")
        last_trained_at = None
        if job_raw:
            try:
                job_data = json.loads(job_raw)
                last_trained_at = job_data.get("completed_at")
            except (json.JSONDecodeError, KeyError):
                pass

        categories.append(
            CategoryTrainingStatus(
                category=category,
                total_records=cat_total,
                unused_records=cat_unused,
                avg_quality_score=round(float(avg_score), 3),
                threshold=settings.TRAINING_THRESHOLD,
                ready_for_training=cat_unused >= settings.TRAINING_THRESHOLD,
                currently_training=bool(currently_training),
                graduated=graduated,
                last_trained_at=last_trained_at,
            )
        )

        total_records += cat_total
        if graduated:
            total_graduated += 1

    return TrainingStatusResponse(
        categories=categories,
        total_records=total_records,
        total_categories_graduated=total_graduated,
    )
