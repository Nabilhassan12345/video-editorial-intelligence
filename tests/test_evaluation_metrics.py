"""
Tests for evaluation metrics and error analysis calculation.
"""

import numpy as np
import pytest

from src.evaluate import compute_editorial_metrics, perform_error_analysis


def test_compute_editorial_metrics():
    # Synthetic ground truth and predictions
    hook_preds = np.array([0.9, 0.8, 0.2, 0.1])
    hook_targets = np.array([1.0, 1.0, 0.0, 0.0])

    drop_preds = np.array([0.1, 0.2, 0.8, 0.9])
    drop_targets = np.array([0.0, 0.0, 1.0, 1.0])

    quality_preds = np.array([8.5, 7.8, 4.2, 3.1])
    quality_targets = np.array([9.0, 8.0, 4.0, 3.0])

    res = compute_editorial_metrics(
        hook_preds=hook_preds,
        hook_targets=hook_targets,
        drop_preds=drop_preds,
        drop_targets=drop_targets,
        quality_preds=quality_preds,
        quality_targets=quality_targets,
        latency_ms_per_sample=15.0,
    )

    assert res["hook_detection"]["accuracy"] == 1.0
    assert res["hook_detection"]["f1_score"] == 1.0
    assert res["drop_off_prediction"]["roc_auc"] == 1.0
    assert res["quality_scoring"]["rmse"] < 0.5
    assert res["quality_scoring"]["pearson_correlation"] > 0.95
    assert res["efficiency"]["latency_ms_per_sample"] == 15.0


def test_perform_error_analysis():
    samples = [
        {"id": "s1", "text": "Good video segment", "drop_off_risk": 0.1, "quality_score": 9.0},
        {"id": "s2", "text": "Boring stalled segment", "drop_off_risk": 0.2, "quality_score": 3.0},
    ]
    hp = np.array([0.5, 0.1])
    dp = np.array([0.15, 0.75])
    qp = np.array([8.8, 7.5])  # s2 predicted 7.5 instead of 3.0 (large residual)

    analysis = perform_error_analysis(samples, hp, dp, qp, top_k=1)
    assert len(analysis["largest_quality_residuals"]) == 1
    assert analysis["largest_quality_residuals"][0]["sample_id"] == "s2"
    assert analysis["largest_quality_residuals"][0]["absolute_error"] == 4.5

