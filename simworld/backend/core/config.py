"""SIMWORLD Configuration — reads from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    NVIDIA_API_KEY: str = "nvapi-your-key-here"

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://simworld:simworld@localhost:5432/simworld"
    )

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── NVIDIA NIM Models ─────────────────────────────────────────────────
    NVIDIA_MODEL_TICK: str = "nvidia/nvidia-nemotron-nano-9b-v2"
    NVIDIA_MODEL_SITUATION: str = "nvidia/nemotron-3-super-120b-a12b"
    NVIDIA_MODEL_PREDICT: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    # ── Simulation Tuning ─────────────────────────────────────────────────
    TICK_SPEED_MS: int = 1500
    TRAINING_THRESHOLD: int = 100
    MODEL_QUALITY_THRESHOLD: float = 0.80

    # ── General ───────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    CORS_ORIGINS: list[str] = ["*"]

    @property
    def tick_speed_seconds(self) -> float:
        """Convenience: tick speed in seconds."""
        return self.TICK_SPEED_MS / 1000.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor for Settings."""
    return Settings()
