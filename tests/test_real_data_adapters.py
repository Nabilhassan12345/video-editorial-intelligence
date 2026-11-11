"""Tests for real-dataset label derivation and the vRetention curve parser."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.real.labels import (
    drop_off_risk_from_curve,
    first_drop_off_second,
    hook_label_from_curve,
    quality_from_percentile,
)
from src.data.real.vretention import collect_vretention, parse_svg_retention_path


def test_hook_label_respects_threshold():
    strong = [100.0, 80.0, 70.0, 65.0]  # keeps 65% after 3s
    weak = [100.0, 40.0, 30.0, 10.0]     # keeps 10%
    assert hook_label_from_curve(strong) == 1.0
    assert hook_label_from_curve(weak) == 0.0
    assert hook_label_from_curve(weak, threshold=0.08) == 1.0


def test_drop_off_risk_is_largest_relative_loss():
    curve = [100.0, 90.0, 45.0, 40.0]  # 90 -> 45 = 50% relative loss
    assert drop_off_risk_from_curve(curve) == pytest.approx(0.5)
    assert drop_off_risk_from_curve([]) == 0.0
    assert drop_off_risk_from_curve([100.0]) == 0.0


def test_first_drop_off_second():
    curve = [100.0, 95.0, 30.0, 25.0]
    assert first_drop_off_second(curve) == 2


def test_quality_percentile_mapping():
    assert quality_from_percentile(99) == 10.0
    assert quality_from_percentile(0) == 1.0
    assert quality_from_percentile(None) == 5.0


def test_parse_svg_retention_path_uses_y_as_retention_pct():
    # Real vRetention format: y is the retention % directly, 100 at second 0.
    path = "M 0.0,100.0 C 1.0,80.9 2.0,13.3 5.0,4.5 L 10.0,3.0"
    curve = parse_svg_retention_path(path, num_points=10)
    assert curve is not None
    assert curve[0] == 100.0
    assert curve[-1] < 5.0
    assert all(0.0 <= v <= 100.0 for v in curve)


def test_collect_vretention_from_csv(tmp_path: Path):
    csv = tmp_path / "youtube.csv"
    csv.write_text(
        "id,publishedAt,title,categoryId,duration,viewCount,countryCode,date,retentionCurve\n"
        "abc123,2024-01-01,My video,20,60,100000,US,2024-02-01,M 0 0 L 300 100 L 600 200\n"
    )
    rows = list(collect_vretention(str(csv)))
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "vretention"
    assert r["features_complete"] is False
    assert 0.0 <= r["drop_off_risk"] <= 1.0
    assert r["group_id"].startswith("vret_")


def test_collected_schema_matches_training(tmp_path: Path):
    """Every produced sample must carry the fields VideoEditorialDataset consumes."""
    csv = tmp_path / "youtube.csv"
    csv.write_text("id,title,duration,viewCount,retentionCurve\nx,Vid,30,500,M 0 0 L 600 300\n")
    row = list(collect_vretention(str(csv)))[0]
    for key in ("id", "group_id", "text", "feature_vector", "hook_label",
                "drop_off_risk", "quality_score"):
        assert key in row
    assert len(row["feature_vector"]) == 8


def test_tiktok_hooks_quality_mapping_is_monotonic_and_bounded():
    from src.data.real.tiktok_hooks import _quality_from_outlier
    assert _quality_from_outlier(0) == 1.0
    assert _quality_from_outlier(1) == 2.5
    assert _quality_from_outlier(10) < _quality_from_outlier(100)
    assert _quality_from_outlier(1e6) == 10.0


def test_vtt_to_text_dedupes_and_strips_tags(tmp_path: Path):
    from src.data.real.youtube import _vtt_to_text
    vtt = tmp_path / "c.vtt"
    vtt.write_text(
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:00.000 --> 00:00:01.000\nhello <c>world</c>\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello world\nnext line\n",
        encoding="utf-8",
    )
    assert _vtt_to_text(str(vtt)) == "hello world next line"
