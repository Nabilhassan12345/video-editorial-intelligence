"""
Tests for AudioExtractor and acoustic feature engineering.
"""

from pathlib import Path
import pytest
import numpy as np

from src.audio.extractor import AudioExtractor, AudioFeatures


def test_audio_extractor_validation(tmp_path):
    extractor = AudioExtractor()

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        extractor.validate_video_file(str(tmp_path / "non_existent.mp4"))

    # Unsupported format
    unsupported = tmp_path / "document.pdf"
    unsupported.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported media format"):
        extractor.validate_video_file(str(unsupported))

    # Supported format
    supported = tmp_path / "video.mp4"
    supported.write_bytes(b"\x00" * 100)
    p = extractor.validate_video_file(str(supported))
    assert p.exists()


def test_compute_segment_features():
    extractor = AudioExtractor()
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave signal
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feat = extractor.compute_segment_features(
        samples=samples,
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
        word_count=12,
    )

    assert isinstance(feat, AudioFeatures)
    assert feat.duration == pytest.approx(5.0)
    assert feat.rms_energy > 0.1
    assert feat.speaking_rate_wpm == pytest.approx((12 / (5.0 / 60.0)), rel=1e-2)
    assert 0.0 <= feat.silence_ratio <= 1.0


def test_fallback_acoustic_features_for_segments():
    extractor = AudioExtractor()
    segments = [
        {"start_time": 0.0, "end_time": 5.0, "text": "This is a fast intro segment!"},
        {"start_time": 5.0, "end_time": 15.0, "text": "And here, well, we pause... and wait."},
    ]

    features = extractor.extract_features_for_segments(wav_path=None, segments=segments)
    assert len(features) == 2
    assert features[0].duration == 5.0
    assert features[0].speaking_rate_wpm > 0.0
    assert features[1].silence_ratio > 0.0

