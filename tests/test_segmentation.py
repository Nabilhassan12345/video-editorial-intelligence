"""
Tests for TranscriptSegmenter and temporal normalization.
"""

import pytest
from src.data.segmenter import TranscriptSegmenter, VideoSegment


def test_parse_timestamp_str():
    assert TranscriptSegmenter.parse_timestamp_str("[00:15]") == 15.0
    assert TranscriptSegmenter.parse_timestamp_str("01:30") == 90.0
    assert TranscriptSegmenter.parse_timestamp_str("[01:05:20]") == 3920.0
    assert TranscriptSegmenter.parse_timestamp_str("invalid") == 0.0


def test_parse_timestamped_text():
    segmenter = TranscriptSegmenter()
    text = (
        "[00:00] In this video we learn retention hacks. "
        "[00:15] Notice how pacing changes here. "
        "[00:30] And here is the conclusion."
    )
    segments = segmenter.parse_raw_input(text)
    assert len(segments) == 3
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 15.0
    assert segments[0].start_timestamp == "[00:00]"
    assert "retention hacks" in segments[0].text
    assert segments[1].start_time == 15.0
    assert segments[1].end_time == 30.0


def test_parse_dict_list():
    segmenter = TranscriptSegmenter()
    raw = [
        {"start": 0.0, "end": 8.0, "text": "Segment one"},
        {"start": 8.0, "end": 20.0, "text": "Segment two"},
    ]
    segments = segmenter.parse_raw_input(raw)
    assert len(segments) == 2
    assert segments[0].duration == 8.0
    assert segments[1].duration == 12.0


def test_segment_plain_text():
    segmenter = TranscriptSegmenter(target_segment_duration=5.0)
    text = (
        "This is the first sentence that introduces our topic. "
        "Here is the second sentence providing additional context. "
        "Now we explain the entire framework in detail."
    )
    segments = segmenter.parse_raw_input(text)
    assert len(segments) >= 1
    for s in segments:
        assert s.duration >= segmenter.min_duration
        assert len(s.text) > 0

