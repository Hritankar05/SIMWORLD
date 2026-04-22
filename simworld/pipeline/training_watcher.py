"""Pipeline — Training Watcher.

Runs as a standalone daemon process that monitors record counts
and triggers training when thresholds are reached.

Usage:
    python pipeline/training_watcher.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train.py"
WATCHER_LOG = DATA_DIR / "watcher.log"
CHECK_INTERVAL_SECONDS = 300  # 5 minutes

CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]

# Set up file logging
DATA_DIR.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(str(WATCHER_LOG), encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
)

logger = logging.getLogger("training_watcher")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def _send_notification(title: str, message: str) -> None:
    """Send a desktop notification. Uses plyer if available, falls back to print."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            timeout=10,
        )
    except (ImportError, Exception):
        # Fallback: just print to console
        print(f"\n🔔 {title}: {message}\n")


def _get_lock_path(category: str) -> Path:
    """Get the lock file path for a category."""
    return DATA_DIR / category / ".training.lock"


def _is_locked(category: str) -> bool:
    """Check if a category is currently being trained."""
    lock_path = _get_lock_path(category)
    if not lock_path.exists():
        return False

    # Check if lock is stale (older than 4 hours)
    try:
        lock_age = time.time() - lock_path.stat().st_mtime
        if lock_age > 4 * 3600:
            logger.warning(
                "Stale lock for '%s' (%.1f hours old) — removing",
                category, lock_age / 3600,
            )
            lock_path.unlink()
            return False
    except OSError:
        return False

    return True


def _acquire_lock(category: str) -> bool:
    """Acquire a training lock for a category. Returns True if acquired."""
    if _is_locked(category):
        return False

    lock_path = _get_lock_path(category)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lock_data = {
            "category": category,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
        return True
    except OSError as exc:
        logger.error("Failed to acquire lock for '%s': %s", category, exc)
        return False


def _release_lock(category: str) -> None:
    """Release the training lock for a category."""
    lock_path = _get_lock_path(category)
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError as exc:
        logger.error("Failed to release lock for '%s': %s", category, exc)


def _count_records(category: str) -> int:
    """Count JSONL records for a category."""
    jsonl_path = DATA_DIR / category / "training_data.jsonl"
    if not jsonl_path.exists():
        return 0
    count = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _get_threshold(category: str) -> int:
    """Get the training threshold for a category."""
    try:
        from pipeline.model_registry import MODEL_REGISTRY
        return MODEL_REGISTRY.get(category, {}).get("threshold", 100)
    except ImportError:
        return 100


class TrainingWatcher:
    """Watches record counts and triggers training when thresholds are met.

    Designed to run as a standalone daemon process.
    """

    def __init__(self, check_interval: int = CHECK_INTERVAL_SECONDS) -> None:
        self._check_interval = check_interval
        self._running = False
        self._training_history: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start the watcher loop. Blocks until interrupted."""
        self._running = True
        logger.info("=" * 60)
        logger.info("SIMWORLD Training Watcher started")
        logger.info("Check interval: %d seconds", self._check_interval)
        logger.info("Monitoring categories: %s", ", ".join(CATEGORIES))
        logger.info("=" * 60)

        _send_notification(
            "SIMWORLD Watcher",
            "Training watcher started. Monitoring all categories.",
        )

        try:
            while self._running:
                self._check_all_categories()
                logger.info(
                    "Next check in %d seconds...", self._check_interval
                )
                time.sleep(self._check_interval)

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user (Ctrl+C)")
        except Exception as exc:
            logger.error("Watcher crashed: %s", exc, exc_info=True)
        finally:
            self._running = False
            logger.info("Training watcher stopped")

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._running = False

    def _check_all_categories(self) -> None:
        """Check all categories and trigger training if ready."""
        logger.info("-" * 40)
        logger.info("Checking all categories...")

        for category in CATEGORIES:
            count = _count_records(category)
            threshold = _get_threshold(category)

            if count >= threshold:
                if _is_locked(category):
                    logger.info(
                        "  %s: %d/%d records (training in progress)",
                        category, count, threshold,
                    )
                else:
                    logger.info(
                        "  %s: %d/%d records — TRIGGERING TRAINING",
                        category, count, threshold,
                    )
                    self._trigger_training(category)
            else:
                logger.info(
                    "  %s: %d/%d records", category, count, threshold
                )

    def _trigger_training(self, category: str) -> None:
        """Trigger a training job for a category."""
        if not _acquire_lock(category):
            logger.warning(
                "Could not acquire lock for '%s' — skipping", category
            )
            return

        logger.info("🚀 Starting training for '%s'...", category)
        _send_notification(
            "SIMWORLD Training",
            f"Training started for category: {category}",
        )

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(TRAINING_SCRIPT), "--category", category],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                cwd=str(PROJECT_ROOT),
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                logger.info(
                    "✅ Training for '%s' completed in %.1f minutes",
                    category, elapsed / 60,
                )
                _send_notification(
                    "SIMWORLD Training Complete",
                    f"Category '{category}' trained in {elapsed/60:.1f} min",
                )
                status = "completed"
            else:
                logger.error(
                    "❌ Training for '%s' failed (exit code %d)",
                    category, result.returncode,
                )
                logger.error("STDERR: %s", result.stderr[:1000])
                _send_notification(
                    "SIMWORLD Training Failed",
                    f"Category '{category}' training failed!",
                )
                status = "failed"

            self._training_history.append({
                "category": category,
                "status": status,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stdout_tail": result.stdout[-500:] if result.stdout else "",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            })

        except subprocess.TimeoutExpired:
            logger.error("Training for '%s' timed out after 2 hours", category)
            _send_notification(
                "SIMWORLD Training Timeout",
                f"Category '{category}' training timed out!",
            )
            self._training_history.append({
                "category": category,
                "status": "timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as exc:
            logger.error("Training error for '%s': %s", category, exc)

        finally:
            _release_lock(category)

    def get_history(self) -> list[dict[str, Any]]:
        """Get the training history log."""
        return self._training_history.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIMWORLD Training Watcher")
    parser.add_argument(
        "--interval",
        type=int,
        default=CHECK_INTERVAL_SECONDS,
        help=f"Check interval in seconds (default: {CHECK_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit (don't loop)",
    )
    args = parser.parse_args()

    watcher = TrainingWatcher(check_interval=args.interval)

    if args.once:
        watcher._check_all_categories()
    else:
        watcher.start()
