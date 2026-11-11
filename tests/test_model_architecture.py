"""
Tests for VideoEditorialTransformer architecture, multi-head forward pass, and loss computation.
"""

import pytest
import torch

from src.models.config import EditorialModelConfig
from src.models.multi_head_transformer import VideoEditorialTransformer


def test_transformer_forward_pass_shapes():
    config = EditorialModelConfig(base_model_name="distilbert-base-uncased", use_peft=False)
    model = VideoEditorialTransformer(config)
    model.eval()

    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    feature_vector = torch.randn((batch_size, config.feature_dim))

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_vector=feature_vector,
        )

    assert "hook_prob" in out
    assert "drop_off_risk" in out
    assert "quality_score" in out

    assert out["hook_prob"].shape == (batch_size,)
    assert out["drop_off_risk"].shape == (batch_size,)
    assert out["quality_score"].shape == (batch_size,)

    # Values within defined bounds
    assert torch.all(out["hook_prob"] >= 0.0) and torch.all(out["hook_prob"] <= 1.0)
    assert torch.all(out["drop_off_risk"] >= 0.0) and torch.all(out["drop_off_risk"] <= 1.0)
    assert torch.all(out["quality_score"] >= 1.0) and torch.all(out["quality_score"] <= 10.0)


def test_transformer_multi_task_loss():
    config = EditorialModelConfig(base_model_name="distilbert-base-uncased", use_peft=False)
    model = VideoEditorialTransformer(config)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    feature_vector = torch.randn((batch_size, config.feature_dim))

    hook_labels = torch.tensor([1.0, 0.0])
    drop_off_labels = torch.tensor([0.2, 0.8])
    quality_labels = torch.tensor([8.5, 3.0])

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        feature_vector=feature_vector,
        hook_labels=hook_labels,
        drop_off_labels=drop_off_labels,
        quality_labels=quality_labels,
    )

    assert "loss" in out
    assert out["loss"].item() > 0.0
    assert "loss_hook" in out
    assert "loss_drop_off" in out
    assert "loss_quality" in out

