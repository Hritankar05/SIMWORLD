"""Training — Model Evaluation.

Evaluates a fine-tuned model against held-out eval data.
Scores based on JSON validity, field completeness, and contextual appropriateness.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CATEGORIES = ["finance", "corporate", "crisis", "social", "generic"]
REQUIRED_FIELDS = {"thought", "action", "emotionalState", "message"}
MAX_EVAL_SAMPLES = 50

# Emotional states considered contextually appropriate per situation keyword
CONTEXTUAL_EMOTIONS: dict[str, list[str]] = {
    "crash": ["anxious", "fearful", "stressed", "panicked", "worried"],
    "crisis": ["anxious", "fearful", "determined", "stressed", "urgent"],
    "success": ["excited", "confident", "optimistic", "pleased", "satisfied"],
    "conflict": ["frustrated", "angry", "determined", "defensive", "tense"],
    "negotiat": ["cautious", "strategic", "determined", "composed", "calculating"],
    "disaster": ["fearful", "anxious", "determined", "urgent", "compassionate"],
    "opportunit": ["excited", "optimistic", "eager", "strategic", "confident"],
    "loss": ["sad", "frustrated", "defeated", "resigned", "anxious"],
    "threat": ["fearful", "defensive", "anxious", "alert", "determined"],
    "celebrat": ["excited", "happy", "joyful", "proud", "satisfied"],
}


def evaluate(
    category: str,
    model_path: str | None = None,
    max_samples: int = MAX_EVAL_SAMPLES,
) -> float:
    """Evaluate a model against eval data for a category.

    Scoring per response:
    - +1.0 if valid JSON with ALL required fields (thought, action, emotionalState, message)
    - +0.5 if valid JSON but missing some fields
    - +0.0 if invalid JSON or empty
    - Bonus +0.1 if emotionalState is contextually appropriate

    Maximum possible score per sample: 1.1

    Args:
        category: Domain category to evaluate.
        model_path: Path to the model (used for logging; actual inference
                    uses the model already loaded or falls back to parsing eval data).
        max_samples: Maximum number of eval samples to process.

    Returns:
        Average score across all samples (0.0 to 1.1).
    """
    eval_path = DATA_DIR / category / "eval.jsonl"

    if not eval_path.exists():
        logger.error("No eval data for category '%s' at %s", category, eval_path)
        print(f"0.0")
        return 0.0

    # ── Load eval samples ─────────────────────────────────────────────
    samples: list[dict[str, Any]] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not samples:
        logger.error("No valid eval samples for '%s'", category)
        print(f"0.0")
        return 0.0

    # Limit to max_samples
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    logger.info(
        "Evaluating '%s' on %d samples (model: %s)",
        category, len(samples), model_path or "N/A",
    )

    # ── Score each sample ─────────────────────────────────────────────
    scores: list[float] = []
    detailed_results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        messages = sample.get("messages", [])
        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")

        if not assistant_content:
            scores.append(0.0)
            detailed_results.append({
                "sample": i + 1,
                "score": 0.0,
                "reason": "Empty assistant response",
            })
            continue

        score, reason = _score_response(assistant_content, user_content)
        scores.append(score)
        detailed_results.append({
            "sample": i + 1,
            "score": score,
            "reason": reason,
        })

    # ── Calculate final score ─────────────────────────────────────────
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # ── Save score to disk ────────────────────────────────────────────
    score_dir = MODELS_DIR / category
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "score.txt"
    score_path.write_text(str(round(avg_score, 4)))

    # ── Save detailed results ─────────────────────────────────────────
    eval_results_path = score_dir / "eval_results.json"
    eval_report = {
        "category": category,
        "model_path": model_path,
        "samples_evaluated": len(samples),
        "average_score": round(avg_score, 4),
        "graduated": avg_score >= 0.80,
        "score_distribution": {
            "perfect_1.1": sum(1 for s in scores if s >= 1.05),
            "full_1.0": sum(1 for s in scores if 0.95 <= s < 1.05),
            "partial_0.5": sum(1 for s in scores if 0.45 <= s < 0.55),
            "failed_0.0": sum(1 for s in scores if s < 0.05),
        },
        "details": detailed_results,
    }
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────
    _print_report(category, model_path, avg_score, scores, detailed_results)

    # Print score as last line (for subprocess parsing)
    print(f"{avg_score:.4f}")
    return avg_score


def _score_response(assistant_content: str, user_content: str) -> tuple[float, str]:
    """Score a single assistant response.

    Returns (score, reason).
    """
    # ── Try to parse as JSON ──────────────────────────────────────────
    cleaned = assistant_content.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return 0.0, "Invalid JSON"
        else:
            return 0.0, "No JSON found"

    if not isinstance(parsed, dict):
        return 0.0, "Response is not a JSON object"

    # ── Check required fields ─────────────────────────────────────────
    present_fields = REQUIRED_FIELDS.intersection(parsed.keys())
    missing_fields = REQUIRED_FIELDS - present_fields

    if len(missing_fields) == 0:
        # All fields present
        base_score = 1.0
        reason = "All fields present"

        # Check fields are non-empty strings
        empty_fields = [
            f for f in REQUIRED_FIELDS
            if not isinstance(parsed.get(f), str) or len(parsed[f].strip()) == 0
        ]
        if empty_fields:
            base_score = 0.5
            reason = f"Empty fields: {', '.join(empty_fields)}"

    elif len(present_fields) > 0:
        base_score = 0.5
        reason = f"Missing fields: {', '.join(missing_fields)}"
    else:
        return 0.0, "No required fields found"

    # ── Contextual emotion bonus ──────────────────────────────────────
    emotional_state = parsed.get("emotionalState", "").lower().strip()
    bonus = 0.0

    if emotional_state and user_content:
        user_lower = user_content.lower()
        for keyword, appropriate_emotions in CONTEXTUAL_EMOTIONS.items():
            if keyword in user_lower:
                if emotional_state in [e.lower() for e in appropriate_emotions]:
                    bonus = 0.1
                    reason += " + contextual emotion bonus"
                break

    return base_score + bonus, reason


def _print_report(
    category: str,
    model_path: str | None,
    avg_score: float,
    scores: list[float],
    details: list[dict[str, Any]],
) -> None:
    """Print a detailed evaluation report."""
    graduated = avg_score >= 0.80

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Category:      {category}")
    print(f"  Model:         {model_path or 'N/A'}")
    print(f"  Samples:       {len(scores)}")
    print(f"  Average Score: {avg_score:.4f}")
    print(f"  Graduated:     {'✅ YES' if graduated else '❌ NO'} (threshold: 0.80)")
    print()

    # Score distribution
    perfect = sum(1 for s in scores if s >= 1.05)
    full = sum(1 for s in scores if 0.95 <= s < 1.05)
    partial = sum(1 for s in scores if 0.45 <= s < 0.55)
    failed = sum(1 for s in scores if s < 0.05)

    print("  Score Distribution:")
    print(f"    1.1 (perfect + bonus): {perfect:>4d}  {'█' * perfect}")
    print(f"    1.0 (all fields):      {full:>4d}  {'█' * full}")
    print(f"    0.5 (partial):         {partial:>4d}  {'█' * partial}")
    print(f"    0.0 (failed):          {failed:>4d}  {'█' * failed}")

    # Show first few failures
    failures = [d for d in details if d["score"] < 0.5]
    if failures:
        print(f"\n  First {min(5, len(failures))} failures:")
        for f in failures[:5]:
            print(f"    Sample {f['sample']}: {f['reason']}")

    print("=" * 60)


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="SIMWORLD Model Evaluation")
    parser.add_argument(
        "--category",
        type=str,
        choices=CATEGORIES,
        help="Category to evaluate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_EVAL_SAMPLES,
        help=f"Max eval samples (default: {MAX_EVAL_SAMPLES})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all categories",
    )
    args = parser.parse_args()

    if args.all:
        results: dict[str, float] = {}
        for cat in CATEGORIES:
            score = evaluate(cat, max_samples=args.max_samples)
            results[cat] = score

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for cat, score in results.items():
            status = "✅" if score >= 0.80 else "❌"
            print(f"  {status} {cat:<12s}  {score:.4f}")
        print("=" * 60)

    elif args.category:
        evaluate(
            args.category,
            model_path=args.model_path,
            max_samples=args.max_samples,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
