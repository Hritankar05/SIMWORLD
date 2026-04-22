"""WebSocket route for real-time simulation streaming."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.simulation_engine import (
    inject_event,
    pause_simulation,
    register_ws,
    resume_simulation,
    set_simulation_speed,
    stop_simulation,
    unregister_ws,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/simulation/{simulation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    simulation_id: uuid.UUID,
) -> None:
    """WebSocket endpoint for real-time simulation updates.

    Server pushes tick_update messages.
    Client can send: inject_event, pause, resume, speed_change.
    """
    await websocket.accept()
    sim_key = str(simulation_id)
    register_ws(sim_key, websocket)

    logger.info("WebSocket connected for simulation %s", sim_key)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
                continue

            msg_type = data.get("type", "")

            if msg_type == "inject_event":
                event_text = data.get("event", "")
                if event_text:
                    await inject_event(simulation_id, event_text)
                    await websocket.send_text(
                        json.dumps({
                            "type": "event_injected",
                            "event": event_text,
                        })
                    )
                else:
                    await websocket.send_text(
                        json.dumps({
                            "type": "error",
                            "message": "Event text required.",
                        })
                    )

            elif msg_type == "pause":
                await pause_simulation(simulation_id)
                await websocket.send_text(
                    json.dumps({"type": "status_change", "status": "paused"})
                )

            elif msg_type == "resume":
                await resume_simulation(simulation_id)
                await websocket.send_text(
                    json.dumps({"type": "status_change", "status": "running"})
                )

            elif msg_type == "stop":
                await stop_simulation(simulation_id)
                await websocket.send_text(
                    json.dumps({"type": "status_change", "status": "completed"})
                )

            elif msg_type == "speed_change":
                speed = data.get("speed", "normal")
                if speed not in ("slow", "normal", "fast"):
                    speed = "normal"
                await set_simulation_speed(simulation_id, speed)
                await websocket.send_text(
                    json.dumps({"type": "speed_changed", "speed": speed})
                )

            else:
                await websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for simulation %s", sim_key)
    except Exception as exc:
        logger.error("WebSocket error for simulation %s: %s", sim_key, exc)
    finally:
        unregister_ws(sim_key, websocket)
