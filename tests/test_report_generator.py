"""
Tests for ReportGenerator output formats (JSON, Markdown, HTML).
"""

import json
from pathlib import Path
import pytest

from src.pipeline.report_generator import ReportGenerator
from src.pipeline.video_pipeline import PipelineResult, EditorialDropOffRisk


@pytest.fixture
def dummy_result():
    risk = EditorialDropOffRisk(
        timestamp="[01:15]",
        start_time=75.0,
        end_time=90.0,
        risk_score=0.62,
        reason="Pacing slowdown",
        explanation="Speaking rate dropped to 85 WPM with high filler words.",
        evidence={"silence_ratio": 22.0, "speaking_rate_wpm": 85.0, "filler_word_ratio": 12.0},
    )
    return PipelineResult(
        hook="[00:00] In this video we explore high-retention editing.",
        hook_segment_index=0,
        hook_confidence=0.85,
        drop_off_risks=[risk],
        quality_score=7.8,
        improvements=["Trim dead air at [01:15]"],
        segments=[
            {
                "index": 0,
                "start_time": 0.0,
                "end_time": 15.0,
                "start_timestamp": "[00:00]",
                "end_timestamp": "[00:15]",
                "text": "In this video we explore high-retention editing.",
                "hook_score": 0.85,
                "risk_score": 0.12,
                "quality_score": 8.5,
                "wpm": 155.0,
                "silence_ratio": 0.02,
            }
        ],
        retention_curve=[{"timestamp": "[00:00]", "seconds": 0.0, "modeled_retention_index": 100.0}],
        metadata={"video_filename": "test.mp4", "model_source": "test_model"},
    )


def test_report_generator_formats(dummy_result, tmp_path):
    # 1. JSON
    json_str = ReportGenerator.to_json(dummy_result)
    parsed = json.loads(json_str)
    assert parsed["quality_score"] == 7.8
    assert len(parsed["drop_off_risks"]) == 1

    # 2. Markdown
    md_str = ReportGenerator.to_markdown(dummy_result)
    assert "# 🎬 Video Editorial Intelligence Report" in md_str
    assert "[01:15]" in md_str
    assert "Pacing slowdown" in md_str

    # 3. HTML
    html_str = ReportGenerator.to_html(dummy_result)
    assert "<!DOCTYPE html>" in html_str
    assert "7.8" in html_str
    assert "Pacing slowdown" in html_str

    # 4. Save helper
    saved = ReportGenerator.save_reports(dummy_result, str(tmp_path), base_name="test_report")
    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert saved["html"].exists()

