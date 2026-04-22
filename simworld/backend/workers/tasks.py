"""Celery tasks for SIMWORLD background processing."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from workers.celery_app import celery_app

logger = logging.getLogger(__name__)

TRAINING_SCRIPT = Path(__file__).resolve().parents[2] / "training" / "train.py"
EVALUATE_SCRIPT = Path(__file__).resolve().parents[2] / "training" / "evaluate.py"


@celery_app.task(bind=True, name="simworld.train_category", max_retries=2)
def train_category(self, category: str) -> dict:
    """Launch a training job for a specific category.

    Runs the training script as a subprocess and returns the result.
    """
    logger.info("Celery task: training category '%s'", category)

    try:
        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT), "--category", category],
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode == 0:
            logger.info("Training for '%s' completed successfully", category)
            return {
                "category": category,
                "status": "completed",
                "stdout": result.stdout[-500:] if result.stdout else "",
            }
        else:
            logger.error(
                "Training for '%s' failed (code %d): %s",
                category, result.returncode, result.stderr[:500],
            )
            raise self.retry(
                exc=RuntimeError(f"Training failed: {result.stderr[:200]}"),
                countdown=60,
            )

    except subprocess.TimeoutExpired:
        logger.error("Training for '%s' timed out", category)
        return {"category": category, "status": "timeout"}

    except Exception as exc:
        logger.error("Training task error for '%s': %s", category, exc)
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="simworld.evaluate_model")
def evaluate_model(category: str) -> dict:
    """Evaluate a trained model's quality."""
    logger.info("Celery task: evaluating model for '%s'", category)

    try:
        result = subprocess.run(
            [sys.executable, str(EVALUATE_SCRIPT), "--category", category],
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode == 0:
            return {
                "category": category,
                "status": "evaluated",
                "stdout": result.stdout[-500:] if result.stdout else "",
            }
        else:
            return {
                "category": category,
                "status": "evaluation_failed",
                "stderr": result.stderr[:500] if result.stderr else "",
            }

    except subprocess.TimeoutExpired:
        return {"category": category, "status": "timeout"}

    except Exception as exc:
        logger.error("Evaluation task error for '%s': %s", category, exc)
        return {"category": category, "status": "error", "error": str(exc)}


@celery_app.task(name="simworld.cleanup_old_data")
def cleanup_old_data(days: int = 30) -> dict:
    """Clean up training data older than N days that has been used in training."""
    logger.info("Celery task: cleaning up data older than %d days", days)
    # In production, this would run a SQL DELETE query.
    # For now, log and return.
    return {"status": "completed", "days": days}
