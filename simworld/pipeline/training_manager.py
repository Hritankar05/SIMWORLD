"""Pipeline — Training Manager.

Orchestrates training: checks GPU, runs train.py, evaluates,
and swaps models if the new one is better.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "training" / "evaluate.py"

CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]


class TrainingManager:
    """Manages the training lifecycle for all categories."""

    def __init__(self) -> None:
        self._active_jobs: dict[str, dict[str, Any]] = {}

    def check_gpu(self) -> dict[str, Any]:
        """Check GPU availability.

        Returns a dict with gpu_available, gpu_name, gpu_memory, and backend.
        """
        # ── Check CUDA via torch ──────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem
                gpu_mem_gb = round(gpu_mem / (1024 ** 3), 1)
                return {
                    "gpu_available": True,
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_mem_gb,
                    "backend": "cuda",
                }
        except ImportError:
            pass

        # ── Check MPS (Apple Silicon) ─────────────────────────────────
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return {
                    "gpu_available": True,
                    "gpu_name": "Apple Silicon (MPS)",
                    "gpu_memory_gb": 0,
                    "backend": "mps",
                }
        except (ImportError, AttributeError):
            pass

        # ── Check nvidia-smi directly ─────────────────────────────────
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                return {
                    "gpu_available": True,
                    "gpu_name": parts[0].strip(),
                    "gpu_memory": parts[1].strip() if len(parts) > 1 else "unknown",
                    "backend": "cuda",
                }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory_gb": 0,
            "backend": "cpu",
        }

    def trigger_training(self, category: str) -> dict[str, Any]:
        """Trigger training for a category.

        If GPU available: run train.py locally as subprocess.
        If no GPU: print Google Colab instructions and return.
        """
        if category not in CATEGORIES:
            return {"status": "error", "message": f"Unknown category: {category}"}

        gpu_info = self.check_gpu()

        if not gpu_info["gpu_available"]:
            colab_instructions = self._get_colab_instructions(category)
            print(colab_instructions)
            return {
                "status": "no_gpu",
                "category": category,
                "message": "No GPU available. See Colab instructions above.",
                "colab_instructions": colab_instructions,
            }

        logger.info(
            "Training '%s' on %s (%s)",
            category, gpu_info["gpu_name"], gpu_info["backend"],
        )

        self._active_jobs[category] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_info["gpu_name"],
        }

        try:
            result = subprocess.run(
                [sys.executable, str(TRAINING_SCRIPT), "--category", category],
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode == 0:
                logger.info("Training for '%s' completed successfully", category)

                # ── Find the latest candidate model ───────────────────
                model_dir = MODELS_DIR / category
                candidates = sorted(
                    model_dir.glob("candidate_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ) if model_dir.exists() else []

                score = 0.0
                if candidates:
                    candidate_path = candidates[0]
                    score = self._evaluate_candidate(category, str(candidate_path))
                    self.swap_if_better(category, str(candidate_path), score)

                self._active_jobs[category] = {
                    "status": "completed",
                    "score": score,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }

                return {
                    "status": "completed",
                    "category": category,
                    "score": score,
                    "stdout": result.stdout[-500:],
                }
            else:
                logger.error("Training for '%s' failed", category)
                self._active_jobs[category] = {"status": "failed"}
                return {
                    "status": "failed",
                    "category": category,
                    "stderr": result.stderr[-500:],
                }

        except subprocess.TimeoutExpired:
            self._active_jobs[category] = {"status": "timeout"}
            return {"status": "timeout", "category": category}

        except Exception as exc:
            self._active_jobs[category] = {"status": "error", "error": str(exc)}
            return {"status": "error", "category": category, "error": str(exc)}

    def swap_if_better(
        self, category: str, model_path: str, score: float
    ) -> bool:
        """Swap the active model if the candidate scores higher.

        Returns True if swapped.
        """
        current_score = self._get_current_score(category)

        if score > current_score:
            active_dir = MODELS_DIR / category / "active"

            # Back up current active model
            if active_dir.exists():
                backup_dir = MODELS_DIR / category / f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(active_dir), str(backup_dir))
                logger.info("Backed up previous model to %s", backup_dir)

            # Move candidate to active
            active_dir.mkdir(parents=True, exist_ok=True)
            candidate = Path(model_path)
            if candidate.exists():
                if candidate.is_dir():
                    shutil.copytree(str(candidate), str(active_dir), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(candidate), str(active_dir))

            # Save score
            score_path = MODELS_DIR / category / "score.txt"
            score_path.write_text(str(round(score, 4)))

            logger.info(
                "✅ Swapped model for '%s': %.4f → %.4f",
                category, current_score, score,
            )
            return True
        else:
            logger.info(
                "Candidate for '%s' not better (%.4f <= %.4f) — keeping current",
                category, score, current_score,
            )
            return False

    def _evaluate_candidate(self, category: str, model_path: str) -> float:
        """Run evaluate.py on a candidate model."""
        try:
            result = subprocess.run(
                [
                    sys.executable, str(EVALUATE_SCRIPT),
                    "--category", category,
                    "--model-path", model_path,
                ],
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode == 0:
                # Parse score from last line of stdout
                lines = result.stdout.strip().split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    try:
                        return float(line)
                    except ValueError:
                        if "score" in line.lower():
                            parts = line.split(":")
                            if len(parts) >= 2:
                                try:
                                    return float(parts[-1].strip())
                                except ValueError:
                                    continue
                # Try reading from score file
                return self._get_current_score(category)
            else:
                logger.error("Evaluation failed: %s", result.stderr[:300])
                return 0.0

        except Exception as exc:
            logger.error("Evaluation error: %s", exc)
            return 0.0

    def _get_current_score(self, category: str) -> float:
        """Get the current active model's score."""
        score_path = MODELS_DIR / category / "score.txt"
        if not score_path.exists():
            return 0.0
        try:
            return float(score_path.read_text().strip())
        except (ValueError, OSError):
            return 0.0

    def _get_colab_instructions(self, category: str) -> str:
        """Generate Google Colab instructions for training without a local GPU."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  NO GPU DETECTED — USE GOOGLE COLAB                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Upload these files to Colab:                                 ║
║     • data/{category}/train.jsonl                           ║
║     • data/{category}/eval.jsonl                            ║
║     • training/train.py                                          ║
║     • training/evaluate.py                                       ║
║     • training/configs/lora_config.yaml                          ║
║     • training/requirements.txt                                  ║
║                                                                  ║
║  2. In Colab, run:                                               ║
║     !pip install -r requirements.txt                             ║
║     !python train.py --category {category:<20s}             ║
║                                                                  ║
║  3. Download the trained model from:                             ║
║     models/{category}/candidate_*/                          ║
║                                                                  ║
║  4. Place it in your local:                                      ║
║     models/{category}/active/                               ║
║                                                                  ║
║  5. Run evaluation locally:                                      ║
║     python training/evaluate.py --category {category:<13s}  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

    def get_status(self) -> dict[str, Any]:
        """Get the current training status for all categories."""
        gpu = self.check_gpu()
        categories_status: dict[str, dict[str, Any]] = {}

        for category in CATEGORIES:
            score = self._get_current_score(category)
            data_path = DATA_DIR / category / "training_data.jsonl"
            record_count = 0
            if data_path.exists():
                with open(data_path) as f:
                    record_count = sum(1 for line in f if line.strip())

            has_active = (MODELS_DIR / category / "active").exists()

            categories_status[category] = {
                "record_count": record_count,
                "score": score,
                "graduated": score >= 0.80,
                "has_active_model": has_active,
                "job": self._active_jobs.get(category, {"status": "idle"}),
            }

        return {
            "gpu": gpu,
            "categories": categories_status,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    manager = TrainingManager()
    status = manager.get_status()

    print("\n" + "=" * 60)
    print("TRAINING MANAGER STATUS")
    print("=" * 60)

    gpu = status["gpu"]
    if gpu["gpu_available"]:
        print(f"  GPU: {gpu['gpu_name']} ({gpu['backend']})")
    else:
        print("  GPU: Not available (CPU only)")

    print()
    for cat, info in status["categories"].items():
        graduated = "✅" if info["graduated"] else "❌"
        active = "🟢" if info["has_active_model"] else "⚪"
        print(
            f"  {graduated} {active} {cat:<12s}  "
            f"records={info['record_count']:>5d}  "
            f"score={info['score']:.3f}  "
            f"job={info['job']['status']}"
        )

    print("=" * 60)
