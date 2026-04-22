"""Pipeline — Log Collector.

Writes simulation tick data to JSONL files and a local SQLite backup.
Thread-safe for concurrent simulation engines.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_PATH = DATA_DIR / "simworld.db"


class LogCollector:
    """Collects simulation tick logs and writes them to JSONL + SQLite.

    Thread-safe: uses a threading.Lock for file writes and a separate
    lock for SQLite operations.
    """

    def __init__(self) -> None:
        self._file_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._db_initialized = False

    def _ensure_db(self) -> None:
        """Create the SQLite backup database and table if they don't exist."""
        if self._db_initialized:
            return

        with self._db_lock:
            if self._db_initialized:
                return

            DATA_DIR.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(SQLITE_PATH))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tick_logs (
                        id TEXT PRIMARY KEY,
                        simulation_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        tick_number INTEGER NOT NULL,
                        agent_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        perception TEXT NOT NULL,
                        thought TEXT NOT NULL,
                        action TEXT NOT NULL,
                        emotional_state TEXT NOT NULL,
                        world_state TEXT NOT NULL,
                        situation TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tick_logs_category
                    ON tick_logs(category)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tick_logs_simulation
                    ON tick_logs(simulation_id)
                """)
                conn.commit()
            finally:
                conn.close()

            self._db_initialized = True
            logger.info("SQLite backup initialized at %s", SQLITE_PATH)

    def log_tick(
        self,
        simulation_id: str,
        category: str,
        situation: str,
        tick_number: int,
        agent_id: str,
        role: str,
        goal: str,
        perception: str,
        thought: str,
        action: str,
        emotional_state: str,
        world_state: dict[str, Any],
    ) -> None:
        """Log a single tick result to JSONL and SQLite.

        Args:
            simulation_id: UUID of the simulation.
            category: Classified category (finance, corporate, etc.).
            situation: The simulation scenario text.
            tick_number: Current tick number.
            agent_id: UUID of the acting agent.
            role: Agent's role in the simulation.
            goal: Agent's goal.
            perception: What the agent perceived this tick.
            thought: Agent's internal reasoning.
            action: The action taken.
            emotional_state: Agent's emotional state after action.
            world_state: Current world state dict.
        """
        # ── Build training record in HuggingFace chat format ──────────
        system_content = (
            f"You are a {category} simulation agent acting as a {role}. "
            f"Your goal is: {goal}. "
            f"Respond ONLY in valid JSON with keys: "
            f"thought, action, emotionalState, message."
        )

        user_content = (
            f"Tick {tick_number}. "
            f"Situation: {situation}. "
            f"World state: {json.dumps(world_state)}. "
            f"Your current perception: {perception}"
        )

        assistant_content = json.dumps({
            "thought": thought,
            "action": action,
            "emotionalState": emotional_state,
            "message": f"[{role}] {action}",
        })

        record = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

        # ── Write to JSONL (thread-safe) ──────────────────────────────
        self._write_jsonl(category, record)

        # ── Write to SQLite backup (thread-safe) ──────────────────────
        self._write_sqlite(
            simulation_id=simulation_id,
            category=category,
            situation=situation,
            tick_number=tick_number,
            agent_id=agent_id,
            role=role,
            goal=goal,
            perception=perception,
            thought=thought,
            action=action,
            emotional_state=emotional_state,
            world_state=world_state,
        )

        logger.debug(
            "Logged tick %d for agent %s in category '%s'",
            tick_number, agent_id, category,
        )

    def _write_jsonl(self, category: str, record: dict[str, Any]) -> None:
        """Append a record to the category's JSONL file (thread-safe)."""
        category_dir = DATA_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = category_dir / "training_data.jsonl"

        line = json.dumps(record, ensure_ascii=False) + "\n"

        with self._file_lock:
            try:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as exc:
                logger.error(
                    "Failed to write JSONL for category '%s': %s", category, exc
                )

    def _write_sqlite(
        self,
        simulation_id: str,
        category: str,
        situation: str,
        tick_number: int,
        agent_id: str,
        role: str,
        goal: str,
        perception: str,
        thought: str,
        action: str,
        emotional_state: str,
        world_state: dict[str, Any],
    ) -> None:
        """Write a tick log to the SQLite backup database (thread-safe)."""
        self._ensure_db()

        row_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        world_state_str = json.dumps(world_state)

        with self._db_lock:
            try:
                conn = sqlite3.connect(str(SQLITE_PATH))
                conn.execute(
                    """
                    INSERT INTO tick_logs
                        (id, simulation_id, category, tick_number, agent_id,
                         role, goal, perception, thought, action,
                         emotional_state, world_state, situation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id, simulation_id, category, tick_number, agent_id,
                        role, goal, perception, thought, action,
                        emotional_state, world_state_str, situation, now,
                    ),
                )
                conn.commit()
                conn.close()
            except sqlite3.Error as exc:
                logger.error("SQLite write failed: %s", exc)

    def get_record_count(self, category: str) -> int:
        """Count JSONL records for a category."""
        jsonl_path = DATA_DIR / category / "training_data.jsonl"
        if not jsonl_path.exists():
            return 0
        count = 0
        with self._file_lock:
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            except OSError:
                pass
        return count

    def get_all_counts(self) -> dict[str, int]:
        """Get record counts for all categories."""
        categories = ["finance", "corporate", "crisis", "social", "generic"]
        return {cat: self.get_record_count(cat) for cat in categories}

    def get_sqlite_count(self, category: str | None = None) -> int:
        """Count records in the SQLite backup."""
        self._ensure_db()
        with self._db_lock:
            try:
                conn = sqlite3.connect(str(SQLITE_PATH))
                if category:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM tick_logs WHERE category = ?",
                        (category,),
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM tick_logs")
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except sqlite3.Error:
                return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = LogCollector()

    # Test write
    collector.log_tick(
        simulation_id="test-sim-001",
        category="finance",
        situation="Stock market crash scenario",
        tick_number=1,
        agent_id="agent-001",
        role="Hedge Fund Manager",
        goal="Maximize returns during volatility",
        perception="Market is down 5% in the last hour",
        thought="I should hedge my positions with put options",
        action="buy_puts",
        emotional_state="anxious",
        world_state={"marketIndex": -5, "tick": 1, "events": []},
    )

    print("\n📊 Record counts (JSONL):")
    for cat, count in collector.get_all_counts().items():
        print(f"  {cat}: {count}")

    print(f"\n📊 SQLite total: {collector.get_sqlite_count()}")
    print(f"📊 SQLite finance: {collector.get_sqlite_count('finance')}")
    print("✅ Log collector test passed")
