"""SIMWORLD Backend — FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.simulations import router as simulations_router
from api.routes.agents import router as agents_router
from api.routes.training import router as training_router
from api.routes.ws import router as ws_router
from core.config import get_settings
from core.database import close_db, init_db
from core.redis import close_redis
from services.training_watcher import get_training_watcher

logger = logging.getLogger("simworld")
settings = get_settings()

# ── Configure logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown hooks."""
    logger.info("🚀 SIMWORLD starting up...")

    # ── Startup ───────────────────────────────────────────────────────
    await init_db()
    logger.info("Database initialized")

    watcher = get_training_watcher()
    await watcher.start()
    logger.info("Training watcher started")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("🛑 SIMWORLD shutting down...")
    await watcher.stop()
    await close_redis()
    await close_db()
    logger.info("All connections closed")


app = FastAPI(
    title="SIMWORLD",
    description="Multi-Agent Simulation Platform — Production Backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────
app.include_router(simulations_router)
app.include_router(agents_router)
app.include_router(training_router)
app.include_router(ws_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "simworld"}


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "service": "SIMWORLD",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
