"""
Azure AI Foundry Model Evaluation & Benchmark Workflow.

Evaluates and compares model catalog candidates (e.g., Mistral Large vs LLaMA 3.1)
on schema adherence, grounded evidence traceability, and recommendation actionability.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from .client import AzureFoundryClient
from .inference import run_foundry_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_response_quality(response: Dict[str, Any]) -> Dict[str, Any]:
    """Score model response on schema compliance, grounding, and actionability."""
    scores = {
        "schema_valid": False,
        "grounded_hook_detected": False,
        "drop_offs_with_rationale": 0,
        "actionable_improvements": 0,
        "total_score": 0.0,
    }

    # 1. Schema check
    has_hook = "hook" in response and isinstance(response["hook"], str)
    has_drops = "drop_off_risks" in response and isinstance(response["drop_off_risks"], list)
    has_quality = "quality_score" in response and isinstance(response["quality_score"], (int, float))
    has_improvements = "improvements" in response and isinstance(response["improvements"], list)

    scores["schema_valid"] = has_hook and has_drops and has_quality and has_improvements

    # 2. Grounded hook
    if has_hook and ("[" in response["hook"] and "]" in response["hook"]):
        scores["grounded_hook_detected"] = True

    # 3. Drop-offs with rationale
    if has_drops:
        valid_drops = 0
        for d in response["drop_off_risks"]:
            if isinstance(d, dict) and "timestamp" in d and "reason" in d:
                valid_drops += 1
        scores["drop_offs_with_rationale"] = valid_drops

    # 4. Actionable improvements
    if has_improvements:
        scores["actionable_improvements"] = len(response["improvements"])

    # Composite score / 10.0
    comp = 0.0
    if scores["schema_valid"]:
        comp += 4.0
    if scores["grounded_hook_detected"]:
        comp += 2.0
    comp += min(2.0, scores["drop_offs_with_rationale"] * 0.7)
    comp += min(2.0, scores["actionable_improvements"] * 0.5)

    scores["total_score"] = round(comp, 2)
    return scores


def run_model_comparison(
    models: List[str] = ["mistral-large-2407", "Meta-Llama-3.1-70B-Instruct"],
    sample_transcript: Optional[str] = None,
    output_dir: str = "outputs/foundry_evaluation",
) -> Dict[str, Any]:
    """Compare multiple Azure AI Foundry model candidates on a standardized editorial test prompt."""
    test_transcript = sample_transcript or (
        "[00:00] In the next 60 seconds, I'll reveal the exact retention strategy top creators use.\n"
        "[00:24] Um, so let me pause here... uh, actually let me adjust my mic for a second.\n"
        "[00:38] But here is the truth: viewer drop-off happens when narrative pacing stagnates."
    )

    client = AzureFoundryClient()
    comparison_results = {}

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("🔬 AZURE AI FOUNDRY MODEL CANDIDATE EVALUATION")
    print("=" * 65)

    for model_name in models:
        logger.info(f"Evaluating candidate model: {model_name}...")
        start_t = time.time()
        resp = run_foundry_inference(transcript=test_transcript, model_name=model_name, client=client)
        latency = round((time.time() - start_t) * 1000.0, 1)

        eval_scores = evaluate_response_quality(resp)

        comparison_results[model_name] = {
            "model_name": model_name,
            "latency_ms": latency,
            "evaluation_scores": eval_scores,
            "response_preview": resp,
        }

        print(f"Candidate: {model_name:<30} | Score: {eval_scores['total_score']}/10.0 | Latency: {latency} ms")

    print("=" * 65)

    # Save results
    with open(out_p / "foundry_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"Comparison report saved to {out_p / 'foundry_comparison.json'}")
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Azure AI Foundry Model Candidates")
    parser.add_argument("--models", nargs="+", default=["mistral-large-2407", "Meta-Llama-3.1-70B-Instruct"], help="Model IDs to compare")
    parser.add_argument("--output-dir", type=str, default="outputs/foundry_evaluation", help="Output directory")
    args = parser.parse_args()

    run_model_comparison(models=args.models, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

