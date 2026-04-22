"""Seed script — creates one sample simulation with 5 agents."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from core.database import async_session_factory, init_db
from models.agent import Agent, AgentCreatedBy
from models.simulation import Simulation, SimulationStatus


async def seed() -> None:
    """Create a sample financial crisis simulation with 5 agents."""
    await init_db()

    async with async_session_factory() as db:
        sim = Simulation(
            situation=(
                "A major tech company's stock has plummeted 40% after a whistleblower "
                "revealed systematic accounting fraud. The SEC has launched an investigation, "
                "institutional investors are demanding a board shakeup, and retail investors "
                "are panicking. Meanwhile, the company's main competitor is considering a "
                "hostile takeover bid. The market is volatile with high uncertainty about "
                "whether the company can survive the scandal."
            ),
            category="finance",
            status=SimulationStatus.PENDING,
            world_state={
                "events": [],
                "tick": 0,
                "marketIndex": -40,
                "companyValuation": 12_000_000_000,
                "secInvestigation": True,
                "mediaAttention": "high",
            },
        )
        db.add(sim)
        await db.flush()

        agents_data = [
            {
                "name": "Victoria Sterling",
                "role": "CEO of the embattled company",
                "goal": "Save the company from collapse by managing the crisis, retaining key employees, and negotiating with regulators",
                "personality": {"composure": 0.7, "ambition": 0.9, "empathy": 0.4, "pragmatism": 0.8, "communication": 0.85},
                "emotional_state": "stressed",
                "color": "#f43f5e",
                "risk_tolerance": 0.6,
            },
            {
                "name": "Marcus Chen",
                "role": "Hedge Fund Manager at BlackRock Capital",
                "goal": "Exploit the market volatility to maximize returns, decide whether to short or buy the dip",
                "personality": {"aggression": 0.85, "analytical": 0.95, "patience": 0.3, "greed": 0.7, "risk_appetite": 0.9},
                "emotional_state": "excited",
                "color": "#6366f1",
                "risk_tolerance": 0.85,
            },
            {
                "name": "Dr. Sarah Williams",
                "role": "SEC Lead Investigator",
                "goal": "Uncover the full extent of the fraud, build a prosecutable case, and protect market integrity",
                "personality": {"diligence": 0.95, "skepticism": 0.8, "integrity": 0.9, "patience": 0.7, "authority": 0.75},
                "emotional_state": "determined",
                "color": "#10b981",
                "risk_tolerance": 0.2,
            },
            {
                "name": "James O'Brien",
                "role": "Financial Journalist at WSJ",
                "goal": "Break exclusive stories about the scandal, hold powerful people accountable, and win a Pulitzer",
                "personality": {"curiosity": 0.95, "tenacity": 0.85, "empathy": 0.5, "ambition": 0.8, "ethics": 0.7},
                "emotional_state": "energized",
                "color": "#f59e0b",
                "risk_tolerance": 0.65,
            },
            {
                "name": "Aisha Patel",
                "role": "Retail Investor and Social Media Influencer",
                "goal": "Protect personal savings, inform her 500K followers, and rally retail investors for collective action",
                "personality": {"passion": 0.9, "impulsiveness": 0.6, "empathy": 0.85, "influence": 0.8, "optimism": 0.5},
                "emotional_state": "anxious",
                "color": "#8b5cf6",
                "risk_tolerance": 0.45,
            },
        ]

        for data in agents_data:
            agent = Agent(
                simulation_id=sim.id,
                name=data["name"],
                role=data["role"],
                goal=data["goal"],
                personality=data["personality"],
                emotional_state=data["emotional_state"],
                color=data["color"],
                risk_tolerance=data["risk_tolerance"],
                created_by=AgentCreatedBy.AUTO,
            )
            db.add(agent)

        await db.commit()

        print(f"✅ Seed complete!")
        print(f"   Simulation ID: {sim.id}")
        print(f"   Category: {sim.category}")
        print(f"   Agents: {len(agents_data)}")
        print(f"   Status: {sim.status.value}")


if __name__ == "__main__":
    asyncio.run(seed())
