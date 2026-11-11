"""
Tests for Azure AI Foundry client, prompt formatting, and model comparison.
"""

from pathlib import Path
import pytest

from azure.foundry.client import AzureFoundryClient
from azure.foundry.inference import run_foundry_inference
from azure.foundry.evaluation import evaluate_response_quality, run_model_comparison


def test_foundry_client_simulation_mode():
    # Unconfigured client runs safely in simulation mode
    client = AzureFoundryClient(endpoint=None, api_key=None)
    assert not client.is_live

    res = client.complete_editorial_chat(
        system_prompt="Test system",
        user_prompt="Test user",
    )
    assert "hook" in res
    assert "drop_off_risks" in res
    assert "quality_score" in res


def test_foundry_inference_runner():
    res = run_foundry_inference(
        transcript="[00:00] In this video we examine pacing. [00:15] Notice the transition.",
    )
    assert isinstance(res, dict)
    assert res["quality_score"] > 0


def test_foundry_evaluation_scoring():
    sample_response = {
        "hook": "[00:00] This is a grounded hook.",
        "drop_off_risks": [
            {"timestamp": "[00:15]", "reason": "Silence", "explanation": "Dead air gap"},
            {"timestamp": "[00:30]", "reason": "Pacing", "explanation": "Slow tempo"}
        ],
        "quality_score": 8.5,
        "improvements": [
            "Trim pauses at [00:15]",
            "Accelerate speaking tempo at [00:30]"
        ],
    }
    score = evaluate_response_quality(sample_response)
    assert score["schema_valid"] is True
    assert score["grounded_hook_detected"] is True
    assert score["total_score"] >= 8.0


def test_foundry_model_comparison(tmp_path):
    results = run_model_comparison(
        models=["mistral-large-2407", "Meta-Llama-3.1-70B-Instruct"],
        output_dir=str(tmp_path),
    )
    assert len(results) == 2
    assert "mistral-large-2407" in results
    assert "Meta-Llama-3.1-70B-Instruct" in results
    assert (tmp_path / "foundry_comparison.json").exists()
