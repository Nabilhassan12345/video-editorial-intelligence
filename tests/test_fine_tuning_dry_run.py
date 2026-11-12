"""
Tests for fine-tuning pipeline dry run and LoRA parameter efficiency.
"""

from pathlib import Path
import pytest

from src.models.config import EditorialModelConfig, TrainingConfig
from src.train import run_training


def test_fine_tuning_dry_run_execution(tmp_path):
    model_config = EditorialModelConfig(base_model_name="distilbert-base-uncased", use_peft=True)
    train_config = TrainingConfig(
        output_dir=str(tmp_path / "checkpoints"),
        num_train_epochs=1,
        batch_size=4,
        dry_run=True,
    )

    result = run_training(model_config, train_config)
    assert result["status"] == "dry_run_verified"
    assert "loss" in result
    assert result["loss"] > 0.0

    # Verify artifacts were serialized
    dry_dir = tmp_path / "checkpoints" / "dry_run"
    assert (dry_dir / "editorial_config.json").exists()
    assert (dry_dir / "prediction_heads.pt").exists()

