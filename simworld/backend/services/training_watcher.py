"""Training Watcher — monitors for training signals and launches jobs.

Works with Redis pub/sub when available, falls back to periodic polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import get_settings
from core.redis import redis_client

logger = logging.getLogger(__name__)

TRAINING_SCRIPT = Path(__file__).resolve().parents[2] / "training" / "train.py"


class TrainingWatcher:
    """Watches for training signals and launches background jobs."""

    def __init__(self) -> None:
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._active_categories: set[str] = set()

    async def start(self) -> None:
        """Start the watcher loop as a background task."""
        if self._running:
            logger.warning("TrainingWatcher already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("TrainingWatcher started")

    async def stop(self) -> None:
        """Stop the watcher loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TrainingWatcher stopped")

    async def _watch_loop(self) -> None:
        """Main watch loop — tries Redis pub/sub, falls back to polling."""
        # Try Redis pub/sub first
        try:
            if hasattr(redis_client, 'pubsub'):
                pubsub = redis_client.pubsub()
                await pubsub.subscribe("training_channel")
                logger.info("Subscribed to Redis training_channel")

                while self._running:
                    try:
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=1.0
                        )
                        if message and message["type"] == "message":
                            data = message["data"]
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            if data.startswith("train:"):
                                category = data.split(":", 1)[1]
                                await self._handle_train_signal(category)
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)

                await pubsub.unsubscribe("training_channel")
                await pubsub.aclose()
                return
        except Exception:
            pass

        # Fallback: just idle (training triggered via API or pipeline watcher)
        logger.info("TrainingWatcher running in passive mode (no Redis pub/sub)")
        while self._running:
            await asyncio.sleep(5)

    async def _handle_train_signal(self, category: str) -> None:
        """Handle a training signal for a specific category."""
        if category in self._active_categories:
            logger.info("Category '%s' already training — skipping", category)
            return

        self._active_categories.add(category)
        logger.info("Launching training job for category '%s'", category)

        job_meta = {
            "category": category,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        await redis_client.set(
            f"training_job:{category}", json.dumps(job_meta), ex=86400
        )

        asyncio.create_task(self._run_training(category))

    async def _run_training(self, category: str) -> None:
        """Run the training script as a subprocess."""
        status = "failed"
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(TRAINING_SCRIPT),
                "--category", category,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("Training for '%s' completed", category)
                status = "completed"
            else:
                logger.error(
                    "Training for '%s' failed: %s", category,
                    stderr.decode("utf-8", errors="replace")[:500],
                )
        except Exception as exc:
            logger.error("Training error for '%s': %s", category, exc)
        finally:
            self._active_categories.discard(category)
            job_meta = {
                "category": category,
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.set(
                f"training_job:{category}", json.dumps(job_meta), ex=86400
            )


_watcher: TrainingWatcher | None = None


def get_training_watcher() -> TrainingWatcher:
    """Get or create the singleton TrainingWatcher."""
    global _watcher
    if _watcher is None:
        _watcher = TrainingWatcher()
    return _watcher
