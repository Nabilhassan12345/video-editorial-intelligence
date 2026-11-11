"""
Tests for FeatureEngineer and multimodal fusion.
"""

import numpy as np
import pytest

from src.audio.extractor import AudioFeatures
from src.data.feature_engineering import FeatureEngineer, MultimodalSegment
from src.data.segmenter import VideoSegment


def test_extract_text_features():
    fe = FeatureEngineer()
    text = "Stop scrolling! Here is the secret truth about video retention and how to double views."
    feats = fe.extract_text_features(text, duration=6.0)

    assert feats["lexical_density"] > 0.0
    assert feats["type_token_ratio"] > 0.0
    assert feats["hook_cue_score"] > 0.0  # contains 'stop', 'secret', 'truth', 'how to'
    assert feats["words_per_second"] == pytest.approx(len(text.split()) / 6.0)


def test_fuse_features_vector_shape():
    fe = FeatureEngineer()
    seg = VideoSegment(index=0, start_time=0.0, end_time=10.0, text="Welcome to the video!")
    af = AudioFeatures(
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        rms_energy=0.12,
        silence_ratio=0.05,
        speaking_rate_wpm=150.0,
        energy_variance=0.002,
    )

    mm = fe.fuse_features(seg, af)
    assert isinstance(mm, MultimodalSegment)
    assert isinstance(mm.feature_vector, np.ndarray)
    assert mm.feature_vector.shape == (8,)
    assert mm.feature_vector.dtype == np.float32

