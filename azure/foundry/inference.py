"""
Azure AI Foundry Editorial Inference Runner.

Orchestrates prompt formatting and model catalog inference for video editorial intelligence.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .client import AzureFoundryClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_foundry_inference(
    transcript: str,
    metrics_summary: Optional[str] = None,
    model_name: Optional[str] = None,
    client: Optional[AzureFoundryClient] = None,
) -> Dict[str, Any]:
    """Execute editorial intelligence analysis through Azure AI Foundry."""
    foundry_client = client or AzureFoundryClient()

    # Load prompt template
    prompt_file = Path(__file__).parent / "prompts" / "editorial_analysis.json"
    with open(prompt_file, "r") as f:
        prompt_data = json.load(f)

    sys_msg = prompt_data.get("system_message", "")
    usr_template = prompt_data.get("user_template", "")

    usr_msg = usr_template.replace("{{transcript}}", transcript)
    usr_msg = usr_msg.replace(
        "{{metrics_summary}}",
        metrics_summary or "Pacing: 152 WPM | Average Silence: 4.2% | Lexical Density: 0.68",
    )

    logger.info("Executing Azure AI Foundry inference...")
    result = foundry_client.complete_editorial_chat(
        system_prompt=sys_msg,
        user_prompt=usr_msg,
        model_name=model_name,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Editorial Inference via Azure AI Foundry")
    parser.add_argument("--transcript", type=str, default=None, help="Transcript text or path to transcript file")
    parser.add_argument("--model", type=str, default=None, help="Model name in Foundry catalog (e.g. mistral-large-2407)")
    args = parser.parse_args()

    text = args.transcript
    if text and Path(text).exists():
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()

    sample_text = (
        "[00:00] In the next 60 seconds, I'll reveal the exact retention strategy top creators use.\n"
        "[00:24] Um, so let me pause here... uh, actually let me adjust my mic for a second.\n"
        "[00:38] But here is the truth: viewer drop-off happens when narrative pacing stagnates."
    )

    res = run_foundry_inference(transcript=text or sample_text, model_name=args.model)
    print("\n" + "=" * 60)
    print("🌐 AZURE AI FOUNDRY EDITORIAL INFERENCE RESULT")
    print("=" * 60)
    print(json.dumps(res, indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

