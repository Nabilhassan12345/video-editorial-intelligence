"""
End-to-End integration tests for VideoEditorialPipeline.
"""

from pathlib import Path
import pytest

from src.pipeline.video_pipeline import VideoEditorialPipeline, PipelineResult


def test_pipeline_with_text_transcript():
    pipeline = VideoEditorialPipeline(checkpoint_dir="checkpoints/editorial_model")
    transcript = (
        "[00:00] In the next 60 seconds I will reveal the single biggest editing mistake. "
        "[00:15] Let me pause here and find my notes... uh, wait a second. "
        "[00:35] Here is the secret framework that doubles your retention."
    )

    result = pipeline.run(transcript_input=transcript)

    assert isinstance(result, PipelineResult)
    assert result.quality_score >= 1.0 and result.quality_score <= 10.0
    assert result.hook.startswith("[00:00]")
    assert len(result.segments) == 3
    assert len(result.improvements) > 0
    assert len(result.retention_curve) == 3
    assert result.metadata["model_status"] in {"fine_tuned", "untrained_heads"}
    assert "modeled_retention_index" in result.retention_curve[0]


def test_pipeline_flags_untrained_heads(tmp_path):
    pipeline = VideoEditorialPipeline(checkpoint_dir=str(tmp_path / "missing"))
    result = pipeline.run(transcript_input="[00:00] Hello there. [00:10] Second part of the talk.")
    assert result.metadata["model_status"] == "untrained_heads"
    assert "UNTRAINED" in result.metadata["prediction_note"]


def test_pipeline_with_sample_video():
    sample_video = Path("examples/sample_video.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video media not present.")

    pipeline = VideoEditorialPipeline(checkpoint_dir="checkpoints/editorial_model")
    transcript = "[00:00] First segment with tone audio. [00:15] Second segment with tone audio."

    result = pipeline.run(video_path=str(sample_video), transcript_input=transcript)
    assert isinstance(result, PipelineResult)
    assert result.metadata["video_filename"] == "sample_video.mp4"
    assert len(result.segments) == 2

