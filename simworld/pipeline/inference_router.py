"""Pipeline — Inference Router.

Routes inference requests to local fine-tuned models or NVIDIA NIM API.
Tracks usage statistics and automatically saves API responses as training data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from pipeline.model_registry import MODEL_REGISTRY, MODELS_DIR, DATA_DIR

logger = logging.getLogger(__name__)

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

# Estimated cost per 1K tokens (for stats tracking)
ESTIMATED_COST_PER_1K_TOKENS = 0.0003


class InferenceRouter:
    """Routes inference to local models or NVIDIA NIM API.

    Priority:
    1. If local fine-tuned model exists AND score >= 0.80 → use local.
    2. Otherwise → call NVIDIA NIM API.

    All API responses are automatically saved as training data to
    continuously grow the dataset.
    """

    def __init__(
        self,
        api_key: str | None = None,
        quality_threshold: float = 0.80,
    ) -> None:
        self._api_key: str = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._quality_threshold = quality_threshold

        # ── Stats tracking ────────────────────────────────────────────
        self._api_calls: int = 0
        self._local_calls: int = 0
        self._total_api_tokens: int = 0
        self._cost_saved: float = 0.0
        self._errors: int = 0
        self._start_time: float = time.time()

        # ── Model cache ───────────────────────────────────────────────
        self._model_scores: dict[str, float] = {}
        self._refresh_scores()

    def _refresh_scores(self) -> None:
        """Load current model scores from disk."""
        for category in MODEL_REGISTRY:
            score_path = MODELS_DIR / category / "score.txt"
            if score_path.exists():
                try:
                    self._model_scores[category] = float(
                        score_path.read_text().strip()
                    )
                except (ValueError, OSError):
                    self._model_scores[category] = 0.0
            else:
                self._model_scores[category] = 0.0

    def _has_local_model(self, category: str) -> bool:
        """Check if a qualified local model exists for a category."""
        score = self._model_scores.get(category, 0.0)
        if score < self._quality_threshold:
            return False

        active_dir = MODELS_DIR / category / "active"
        return active_dir.exists() and any(active_dir.iterdir())

    def get_response(self, category: str, prompt: str) -> dict[str, Any]:
        """Get an inference response for a given category and prompt.

        Tries local model first, falls back to NVIDIA NIM.
        Saves API responses as training data automatically.

        Args:
            category: The domain category (finance, corporate, etc.)
            prompt: The user prompt (tick context).

        Returns:
            Dict with keys: thought, action, emotionalState, message, source.
        """
        # ── Try local model ───────────────────────────────────────────
        if self._has_local_model(category):
            result = self._call_local(category, prompt)
            if result is not None:
                self._local_calls += 1
                # Estimate tokens saved
                estimated_tokens = len(prompt.split()) + len(str(result).split())
                self._cost_saved += (estimated_tokens / 1000) * ESTIMATED_COST_PER_1K_TOKENS
                result["source"] = "local"
                return result

        # ── Fall back to NVIDIA NIM ───────────────────────────────────
        result = self._call_nvidia_nim(category, prompt)
        self._api_calls += 1

        # ── Save API response as training data ────────────────────────
        if result.get("source") == "api":
            self._save_as_training_data(category, prompt, result)

        return result

    def _call_local(self, category: str, prompt: str) -> dict[str, Any] | None:
        """Call the local fine-tuned model.

        In production, this would load the model via transformers/vLLM.
        This method provides the integration point.
        """
        active_dir = MODELS_DIR / category / "active"

        try:
            # Check if model files exist
            if not active_dir.exists():
                return None

            # In a real deployment, this is where you'd:
            # 1. Load the model with AutoModelForCausalLM
            # 2. Apply the LoRA adapter
            # 3. Run inference with the tokenizer
            # 4. Parse the output
            #
            # For now, return None to fall back to API.
            # This gets replaced when you actually have a trained model.
            logger.debug("Local model check for '%s' — no runtime loaded", category)
            return None

        except Exception as exc:
            logger.warning("Local inference error for '%s': %s", category, exc)
            return None

    def _call_nvidia_nim(self, category: str, prompt: str) -> dict[str, Any]:
        """Call the NVIDIA NIM API for inference."""
        if not self._api_key:
            self._errors += 1
            return {
                "thought": "[No API key configured]",
                "action": "idle",
                "emotionalState": "neutral",
                "message": "Unable to respond — no API key.",
                "source": "error",
            }

        system_prompt = (
            f"You are a {category} simulation agent. "
            f"Respond ONLY in valid JSON with keys: "
            f"thought, action, emotionalState, message. "
            f"No markdown, no explanation."
        )

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    NVIDIA_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": NVIDIA_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 512,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Track token usage
                usage = data.get("usage", {})
                self._total_api_tokens += usage.get("total_tokens", 0)

                raw = data["choices"][0]["message"]["content"].strip()
                parsed = self._parse_response(raw)
                parsed["source"] = "api"
                return parsed

        except Exception as exc:
            self._errors += 1
            logger.error("NVIDIA NIM call failed: %s", exc)
            return {
                "thought": f"[API error: {exc}]",
                "action": "idle",
                "emotionalState": "confused",
                "message": "I'm having trouble responding right now.",
                "source": "error",
            }

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse a raw LLM response into structured fields."""
        cleaned = raw.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Strip <think> tags
        import re
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting JSON from surrounding text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    return {
                        "thought": cleaned[:300],
                        "action": "idle",
                        "emotionalState": "neutral",
                        "message": "[Parse error]",
                    }
            else:
                return {
                    "thought": cleaned[:300],
                    "action": "idle",
                    "emotionalState": "neutral",
                    "message": "[No JSON found]",
                }

        return {
            "thought": parsed.get("thought", ""),
            "action": parsed.get("action", "idle"),
            "emotionalState": parsed.get("emotionalState", "neutral"),
            "message": parsed.get("message", ""),
        }

    def _save_as_training_data(
        self, category: str, prompt: str, result: dict[str, Any]
    ) -> None:
        """Save an API response as a training data record."""
        record = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a {category} simulation agent. "
                        f"Respond in JSON with keys: thought, action, emotionalState, message."
                    ),
                },
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": result.get("thought", ""),
                        "action": result.get("action", "idle"),
                        "emotionalState": result.get("emotionalState", "neutral"),
                        "message": result.get("message", ""),
                    }),
                },
            ]
        }

        category_dir = DATA_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = category_dir / "training_data.jsonl"

        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to save training data: %s", exc)

    def get_stats(self) -> dict[str, Any]:
        """Get inference routing statistics."""
        total_calls = self._api_calls + self._local_calls
        uptime = time.time() - self._start_time

        return {
            "api_calls": self._api_calls,
            "local_calls": self._local_calls,
            "total_calls": total_calls,
            "local_percentage": (
                round(self._local_calls / total_calls * 100, 1)
                if total_calls > 0
                else 0.0
            ),
            "total_api_tokens": self._total_api_tokens,
            "estimated_cost_usd": round(
                self._total_api_tokens / 1000 * ESTIMATED_COST_PER_1K_TOKENS, 4
            ),
            "cost_saved_usd": round(self._cost_saved, 4),
            "errors": self._errors,
            "uptime_seconds": round(uptime, 1),
            "model_scores": self._model_scores.copy(),
        }

    def print_stats(self) -> None:
        """Print a formatted stats report."""
        stats = self.get_stats()
        print("\n" + "=" * 55)
        print("INFERENCE ROUTER — STATISTICS")
        print("=" * 55)
        print(f"  API calls:        {stats['api_calls']:>8d}")
        print(f"  Local calls:      {stats['local_calls']:>8d}")
        print(f"  Total calls:      {stats['total_calls']:>8d}")
        print(f"  Local %:          {stats['local_percentage']:>7.1f}%")
        print(f"  API tokens used:  {stats['total_api_tokens']:>8d}")
        print(f"  Est. API cost:    ${stats['estimated_cost_usd']:>7.4f}")
        print(f"  Cost saved:       ${stats['cost_saved_usd']:>7.4f}")
        print(f"  Errors:           {stats['errors']:>8d}")
        print(f"  Uptime:           {stats['uptime_seconds']:>7.1f}s")
        print()
        print("  Model scores:")
        for cat, score in stats["model_scores"].items():
            status = "✅ Active" if score >= 0.80 else "❌ API"
            print(f"    {cat:<12s}  {score:.3f}  {status}")
        print("=" * 55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    router = InferenceRouter()

    # Test inference
    result = router.get_response(
        "finance",
        "Tick 1. The stock market just dropped 10%. You are a hedge fund manager. "
        "What do you do?",
    )
    print(f"\nResponse: {json.dumps(result, indent=2)}")
    router.print_stats()
