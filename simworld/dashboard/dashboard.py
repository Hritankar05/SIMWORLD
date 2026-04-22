"""SIMWORLD Dashboard — Lightweight monitoring UI.

Uses Jinja2 templates for rendering. Can run standalone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
BACKEND_URL = "http://localhost:8000"

app = FastAPI(title="SIMWORLD Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


async def fetch_training_status() -> dict[str, Any]:
    """Fetch training status from the backend API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BACKEND_URL}/api/training/status")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch training status: %s", exc)
        return {
            "categories": [],
            "total_records": 0,
            "total_categories_graduated": 0,
            "error": str(exc),
        }


async def fetch_health() -> dict[str, str]:
    """Fetch backend health status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unreachable", "service": "simworld"}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the main dashboard page."""
    training_status = await fetch_training_status()
    health = await fetch_health()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "health": health,
            "training": training_status,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")
