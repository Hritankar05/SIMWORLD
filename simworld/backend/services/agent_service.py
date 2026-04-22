"""Agent Service — NIM-powered agent reasoning and action generation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

import httpx

from core.config import get_settings
from ml.inference_router import get_inference_router
from schemas.agent import AgentPromptContext, AgentTickResult

logger = logging.getLogger(__name__)


async def process_agent_tick(
    ctx: AgentPromptContext,
    other_agents: list[dict[str, Any]] | None = None,
    talk_to: str | None = None,
) -> AgentTickResult:
    """Process a single agent's tick.

    1. Check if a local fine-tuned model is available via inference_router.
    2. If available and quality ≥ threshold, use local model.
    3. Otherwise, call NVIDIA NIM API.
    4. Parse the JSON response into an AgentTickResult.
    """
    settings = get_settings()

    # ── Try local model first ─────────────────────────────────────────
    inference_router = get_inference_router()
    local_result = await inference_router.try_local_inference(ctx)
    if local_result is not None:
        logger.debug("Using local model for agent %s", ctx.name)
        return local_result

    # ── Build NVIDIA NIM request ──────────────────────────────────────
    system_prompt = _build_system_prompt(ctx, other_agents, talk_to)
    user_prompt = _build_user_prompt(ctx)

    try:
        raw_response = await _call_nvidia_nim(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.NVIDIA_MODEL_TICK,
            api_key=settings.NVIDIA_API_KEY,
        )
        return _parse_agent_response(ctx.agent_id, raw_response, talk_to)

    except Exception as exc:
        logger.error(
            "NIM call failed for agent %s (tick %d): %s",
            ctx.name, ctx.tick_number, exc,
        )
        # Generate a meaningful fallback instead of "I'm unable to respond"
        fallback_msg = _generate_fallback_message(ctx, talk_to)
        return AgentTickResult(
            agent_id=ctx.agent_id,
            thought=f"[System: NIM call failed — {exc}]",
            action=f"Observing the situation and considering options",
            emotional_state=ctx.emotional_state,
            message=fallback_msg,
            target_agent=talk_to or "",
        )


def _build_system_prompt(
    ctx: AgentPromptContext,
    other_agents: list[dict[str, Any]] | None = None,
    talk_to: str | None = None,
) -> str:
    """Build the system prompt for agent reasoning — matches the proven frontend format."""
    personality_str = ", ".join(
        f"{k}={v}" for k, v in (ctx.personality or {}).items()
    )
    if not personality_str:
        personality_str = "balanced"

    # Build other agents context
    other_agents_ctx = ""
    if other_agents:
        lines = []
        for a in other_agents:
            state = a.get("emotional_state", "neutral")
            last_action = a.get("last_action", "No actions yet")
            last_msg = a.get("last_message", "")
            msg_part = f' Last said: "{last_msg}"' if last_msg else ""
            lines.append(
                f"- {a['name']} ({a['role']}, feeling {state}): "
                f"Last action: \"{last_action}\".{msg_part}"
            )
        other_agents_ctx = "\n\nOTHER AGENTS IN THIS SIMULATION:\n" + "\n".join(lines)

    talk_to_line = ""
    if talk_to:
        talk_to_line = f"\nYou MUST directly address {talk_to} in your message this tick."

    return (
        f"You are {ctx.name}, a {ctx.role} in a live simulation.\n"
        f"Goal: {ctx.goal}\n"
        f"Personality: {personality_str}\n"
        f"Risk tolerance: {ctx.risk_tolerance:.1f} | Emotional state: {ctx.emotional_state}\n"
        f"{other_agents_ctx}"
        f"{talk_to_line}\n\n"
        f"CRITICAL RULES:\n"
        f'1. The "message" field is MANDATORY — it must contain spoken dialogue (1-2 sentences). NEVER leave it empty or null.\n'
        f"2. You must address another agent BY NAME in your message. Talk to them directly.\n"
        f'3. Your message should be a reaction, proposal, question, agreement, disagreement, warning, or negotiation.\n'
        f'4. "targetAgent" must be the name of the agent you\'re speaking to.\n'
        f'5. "action" must describe what you are DOING (not "idle").\n\n'
        f"EXAMPLE RESPONSES:\n"
        f'{{"thought":"I need to convince Marcus to join my approach","action":"Proposes a joint strategy to Marcus",'
        f'"emotionalState":"determined","message":"Marcus, I think we should combine our resources here. What do you say?",'
        f'"targetAgent":"Marcus Chen","marketImpact":2}}\n'
        f'{{"thought":"Sarah\'s plan is too risky, I need to push back","action":"Challenges Sarah\'s proposal publicly",'
        f'"emotionalState":"frustrated","message":"Sarah, that\'s way too aggressive. We\'ll lose everything if the market turns.",'
        f'"targetAgent":"Sarah Williams","marketImpact":-1}}\n\n'
        f"Respond with ONLY valid JSON. No markdown fences, no explanation, no extra text."
    )


def _build_user_prompt(ctx: AgentPromptContext) -> str:
    """Build the user prompt with situational context."""
    recent = "\n".join(ctx.recent_events[-5:]) if ctx.recent_events else "No recent events."
    return (
        f"Tick {ctx.tick_number}. "
        f"Situation: {ctx.situation}. "
        f"Recent events: {recent}. "
        f"Speak to another agent now."
    )


async def _call_nvidia_nim(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    """Call the NVIDIA NIM API (OpenAI-compatible) with retry for rate limits."""
    last_exc = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )

                # Retry on 429 with backoff
                if response.status_code == 429:
                    wait = 2 ** attempt + 1  # 2s, 3s, 5s
                    logger.warning("Rate limited (429), retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Handle null content from API
                if content is None:
                    logger.warning("NVIDIA API returned null content — retrying (attempt %d/%d)", attempt + 1, max_retries)
                    await asyncio.sleep(1)
                    continue

                return content

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = 2 ** attempt + 1
                logger.warning("Rate limited (429), retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                await asyncio.sleep(wait)
                last_exc = exc
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    # All retries exhausted
    raise Exception(f"NVIDIA API failed after {max_retries} attempts: {last_exc or 'null content'}")


def _clean_response(raw: str | None) -> str:
    """Clean raw LLM response: strip think tags, markdown fences, and extra text."""
    if not raw:
        return ""

    cleaned = raw.strip()

    # Strip <think>...</think> reasoning blocks that Nemotron produces
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    return cleaned


def _parse_agent_response(
    agent_id: uuid.UUID, raw: str, talk_to: str | None = None
) -> AgentTickResult:
    """Parse the raw NIM response JSON into an AgentTickResult.

    Handles <think> tags, markdown-wrapped JSON, partial JSON, and failures.
    """
    cleaned = _clean_response(raw)

    parsed = None

    # Attempt 1: direct parse
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract JSON object from text
    if parsed is None:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Attempt 3: if still nothing, try to build from the raw text
    if parsed is None:
        logger.warning("Could not parse agent response JSON: %s", cleaned[:300])
        # Use the raw text as a free-form response
        return AgentTickResult(
            agent_id=agent_id,
            thought=cleaned[:300] if cleaned else "Processing...",
            action="Analyzing the situation and formulating a response",
            emotional_state="focused",
            message=_extract_readable_message(cleaned, talk_to),
            target_agent=talk_to or "",
        )

    return AgentTickResult(
        agent_id=agent_id,
        thought=parsed.get("thought", ""),
        action=parsed.get("action", "Observing the situation"),
        emotional_state=parsed.get("emotionalState", parsed.get("emotional_state", "neutral")),
        message=parsed.get("message", ""),
        target_agent=parsed.get("targetAgent", parsed.get("target_agent", talk_to or "")),
    )


def _extract_readable_message(text: str, talk_to: str | None = None) -> str:
    """Extract a readable message from unparseable text."""
    if not text:
        target = talk_to or "everyone"
        return f"{target}, I'm assessing the current situation. What's your take?"

    # Try to find any quoted text that looks like dialogue
    quotes = re.findall(r'"([^"]{10,})"', text)
    if quotes:
        return quotes[0]

    # Use first meaningful sentence
    sentences = re.split(r"[.!?]", text)
    for s in sentences:
        s = s.strip()
        if len(s) > 15:
            return s + "."

    target = talk_to or "everyone"
    return f"{target}, I'm analyzing the situation carefully. Let me share my thoughts."


def _generate_fallback_message(
    ctx: AgentPromptContext, talk_to: str | None = None
) -> str:
    """Generate a contextual fallback message when NIM fails."""
    target = talk_to or "everyone"
    fallbacks = [
        f"{target}, given the current {ctx.situation[:50]}... I think we need to act carefully. What's your assessment?",
        f"{target}, as a {ctx.role}, I'm seeing some important patterns here. Let's discuss our next move.",
        f"{target}, I've been thinking about our {ctx.goal[:40]}. We should coordinate our approach.",
        f"{target}, the situation is evolving. From my perspective as a {ctx.role}, we need to stay focused.",
    ]
    # Rotate based on tick number for variety
    return fallbacks[ctx.tick_number % len(fallbacks)]


async def generate_agents_for_situation(
    situation: str,
    category: str,
    count: int = 5,
) -> list[dict[str, Any]]:
    """Call NVIDIA NIM to auto-generate agents suited for a situation.

    Uses the larger nemotron-120b model for richer character generation.
    """
    settings = get_settings()

    colors = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]

    system_prompt = (
        "You are a simulation designer. Generate a JSON array of "
        f"{count} diverse characters for a {category} simulation. "
        "Each character must have: name (string), role (string), "
        "goal (string), personality (object with traits as keys and "
        "0-1 float values), risk_tolerance (float 0-1). "
        "Make them diverse with conflicting goals for interesting dynamics. "
        "Respond with ONLY the JSON array, nothing else."
    )

    try:
        raw = await _call_nvidia_nim(
            system_prompt=system_prompt,
            user_prompt=f"Situation: {situation}",
            model=settings.NVIDIA_MODEL_SITUATION,
            api_key=settings.NVIDIA_API_KEY,
            temperature=0.8,
            max_tokens=2048,
        )

        cleaned = _clean_response(raw)

        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start != -1 and end > start:
            agents_raw = json.loads(cleaned[start:end])
        else:
            agents_raw = json.loads(cleaned)

        agents: list[dict[str, Any]] = []
        for i, a in enumerate(agents_raw[:count]):
            agents.append({
                "name": a.get("name", f"Agent-{i+1}"),
                "role": a.get("role", "participant"),
                "goal": a.get("goal", "Observe and react"),
                "personality": a.get("personality", {"adaptability": 0.5}),
                "risk_tolerance": float(a.get("risk_tolerance", 0.5)),
                "color": colors[i % len(colors)],
            })
        return agents

    except Exception as exc:
        logger.error("Agent generation via NIM failed: %s — using defaults", exc)
        return _default_agents(category, count, colors)


def _default_agents(
    category: str,
    count: int,
    colors: list[str],
) -> list[dict[str, Any]]:
    """Fallback agents when NIM generation fails."""
    templates = {
        "finance": [
            {"name": "Marcus Chen", "role": "Hedge Fund Manager", "goal": "Maximize portfolio returns while managing risk exposure", "personality": {"aggression": 0.8, "analytical": 0.9, "caution": 0.3}, "risk_tolerance": 0.8},
            {"name": "Sarah Williams", "role": "Regulatory Analyst", "goal": "Ensure market compliance and flag irregularities", "personality": {"diligence": 0.9, "skepticism": 0.7, "caution": 0.8}, "risk_tolerance": 0.2},
            {"name": "Raj Patel", "role": "Retail Investor", "goal": "Grow personal savings through smart market plays", "personality": {"optimism": 0.7, "impulsiveness": 0.5, "adaptability": 0.6}, "risk_tolerance": 0.5},
            {"name": "Elena Volkov", "role": "Central Bank Economist", "goal": "Stabilize monetary policy and control inflation", "personality": {"analytical": 0.95, "patience": 0.8, "authority": 0.7}, "risk_tolerance": 0.15},
            {"name": "James O'Brien", "role": "Financial Journalist", "goal": "Uncover market manipulation and inform the public", "personality": {"curiosity": 0.9, "tenacity": 0.8, "skepticism": 0.85}, "risk_tolerance": 0.6},
        ],
        "corporate": [
            {"name": "Victoria Sterling", "role": "CEO", "goal": "Drive company growth and satisfy shareholders", "personality": {"leadership": 0.9, "ambition": 0.85, "composure": 0.7}, "risk_tolerance": 0.6},
            {"name": "David Kim", "role": "VP of Engineering", "goal": "Ship reliable products on time and retain top talent", "personality": {"pragmatism": 0.8, "empathy": 0.6, "perfectionism": 0.7}, "risk_tolerance": 0.4},
            {"name": "Aisha Johnson", "role": "HR Director", "goal": "Maintain company culture during rapid change", "personality": {"empathy": 0.9, "diplomacy": 0.85, "firmness": 0.5}, "risk_tolerance": 0.3},
            {"name": "Michael Torres", "role": "Board Member", "goal": "Protect shareholder value and ensure governance", "personality": {"analytical": 0.8, "conservatism": 0.7, "authority": 0.75}, "risk_tolerance": 0.25},
            {"name": "Lisa Chang", "role": "Disgruntled Employee", "goal": "Fight for better working conditions", "personality": {"passion": 0.9, "frustration": 0.7, "courage": 0.8}, "risk_tolerance": 0.7},
        ],
        "crisis": [
            {"name": "Commander Hayes", "role": "Emergency Director", "goal": "Coordinate disaster response and save lives", "personality": {"decisiveness": 0.95, "composure": 0.85, "authority": 0.9}, "risk_tolerance": 0.7},
            {"name": "Dr. Amara Okafor", "role": "Medical Lead", "goal": "Triage and treat casualties efficiently", "personality": {"compassion": 0.9, "resilience": 0.8, "pragmatism": 0.7}, "risk_tolerance": 0.5},
            {"name": "Carlos Mendez", "role": "Civilian Leader", "goal": "Protect the community and maintain order", "personality": {"bravery": 0.8, "empathy": 0.75, "stubbornness": 0.6}, "risk_tolerance": 0.6},
            {"name": "Agent Thompson", "role": "Intelligence Officer", "goal": "Assess threats and provide situational intelligence", "personality": {"analytical": 0.9, "paranoia": 0.5, "efficiency": 0.85}, "risk_tolerance": 0.4},
            {"name": "Nina Petrova", "role": "Journalist", "goal": "Report the truth and hold authorities accountable", "personality": {"curiosity": 0.9, "tenacity": 0.85, "empathy": 0.6}, "risk_tolerance": 0.65},
        ],
        "social": [
            {"name": "Jordan Rivera", "role": "Community Organizer", "goal": "Build grassroots support for social change", "personality": {"charisma": 0.85, "passion": 0.9, "patience": 0.5}, "risk_tolerance": 0.7},
            {"name": "Prof. Eleanor Wright", "role": "University Professor", "goal": "Educate the public and provide expert analysis", "personality": {"intellect": 0.9, "patience": 0.8, "idealism": 0.7}, "risk_tolerance": 0.3},
            {"name": "Mayor Ben Crawford", "role": "Local Politician", "goal": "Balance public interest with political survival", "personality": {"diplomacy": 0.85, "ambition": 0.7, "pragmatism": 0.8}, "risk_tolerance": 0.4},
            {"name": "Zara Ahmed", "role": "Social Media Influencer", "goal": "Amplify voices and shape public narrative", "personality": {"charisma": 0.8, "impulsiveness": 0.6, "empathy": 0.7}, "risk_tolerance": 0.6},
            {"name": "Officer Diana Reyes", "role": "Police Captain", "goal": "Maintain peace while respecting civil rights", "personality": {"authority": 0.8, "composure": 0.7, "empathy": 0.5}, "risk_tolerance": 0.35},
        ],
    }

    default = [
        {"name": f"Agent-{i+1}", "role": "Participant", "goal": "Observe, analyze, and take decisive action", "personality": {"adaptability": 0.5 + i*0.1, "curiosity": 0.6}, "risk_tolerance": 0.5}
        for i in range(count)
    ]

    selected = templates.get(category, default)[:count]
    for i, agent in enumerate(selected):
        agent["color"] = colors[i % len(colors)]
    return selected
