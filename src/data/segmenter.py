"""
Adaptive transcript and video segmentation module.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Union


@dataclass
class VideoSegment:
    """Represents a coherent temporal chunk of video/audio content."""
    index: int
    start_time: float
    end_time: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.1, self.end_time - self.start_time)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def start_timestamp(self) -> str:
        return self._format_timestamp(self.start_time)

    @property
    def end_timestamp(self) -> str:
        return self._format_timestamp(self.end_time)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{mins:02d}:{secs:02d}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration": round(self.duration, 2),
            "text": self.text,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


class TranscriptSegmenter:
    """Parses, segments, and normalizes video transcripts for editorial sequence modeling."""

    def __init__(self, target_segment_duration: float = 10.0, min_segment_duration: float = 4.0):
        self.target_duration = target_segment_duration
        self.min_duration = min_segment_duration

    @staticmethod
    def parse_timestamp_str(ts_str: str) -> float:
        """Parse [MM:SS] or [HH:MM:SS] or MM:SS into seconds float."""
        clean = ts_str.strip("[]() ")
        parts = clean.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        try:
            return float(clean)
        except ValueError:
            return 0.0

    def parse_raw_input(self, input_data: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[VideoSegment]:
        """
        Parse diverse transcript formats into a standardized list of VideoSegments.
        Supports:
        - JSON list of segment dicts (with 'start', 'end', 'text' or 'start_time', 'end_time')
        - Timestamped text (e.g., "[00:12] Hello world [00:18] Next topic...")
        - Plain raw untimestamped text
        """
        if isinstance(input_data, list):
            return self._from_dict_list(input_data)
        elif isinstance(input_data, dict) and "segments" in input_data:
            return self._from_dict_list(input_data["segments"])

        text = str(input_data).strip()
        if not text:
            return []

        # Check if the text is JSON-formatted
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return self._from_dict_list(parsed)
                elif isinstance(parsed, dict) and "segments" in parsed:
                    return self._from_dict_list(parsed["segments"])
            except json.JSONDecodeError:
                pass

        # Check for bracketed timestamps like [01:23] or (01:23)
        timestamp_pattern = re.compile(r"(\[?\d{1,2}:\d{2}(?::\d{2})?\]?)")
        matches = list(timestamp_pattern.finditer(text))

        if matches:
            segments: List[VideoSegment] = []
            for i, match in enumerate(matches):
                ts_str = match.group(1)
                st = self.parse_timestamp_str(ts_str)

                text_start = match.end()
                text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                seg_text = text[text_start:text_end].strip()

                if i + 1 < len(matches):
                    next_ts = self.parse_timestamp_str(matches[i + 1].group(1))
                    et = max(st + 1.0, next_ts)
                else:
                    # Estimate based on word count (~2.5 words per sec)
                    word_cnt = len(seg_text.split())
                    est_dur = max(self.min_duration, word_cnt / 2.5)
                    et = st + est_dur

                if seg_text:
                    segments.append(
                        VideoSegment(
                            index=len(segments),
                            start_time=st,
                            end_time=et,
                            text=seg_text,
                        )
                    )
            return segments

        # Plain untimestamped text: sentence-based chunking with estimated speaking rate
        return self._segment_plain_text(text)

    def _from_dict_list(self, dict_list: List[Dict[str, Any]]) -> List[VideoSegment]:
        segments = []
        for i, item in enumerate(dict_list):
            st = float(item.get("start") or item.get("start_time") or 0.0)
            et = float(item.get("end") or item.get("end_time") or (st + 5.0))
            text = str(item.get("text") or item.get("content") or "").strip()
            segments.append(
                VideoSegment(
                    index=i,
                    start_time=st,
                    end_time=max(st + 0.5, et),
                    text=text,
                    metadata=item.get("metadata", {}),
                )
            )
        return self._normalize_continuity(segments)

    def _segment_plain_text(self, text: str) -> List[VideoSegment]:
        """Split plain text into sentences and cluster them into ~target_duration blocks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments: List[VideoSegment] = []
        current_text: List[str] = []
        current_words = 0
        current_time = 0.0

        for sentence in sentences:
            words = len(sentence.split())
            current_text.append(sentence)
            current_words += words

            est_duration = current_words / 2.5  # ~150 words per minute = 2.5 words/sec

            if est_duration >= self.target_duration:
                seg_text = " ".join(current_text)
                segments.append(
                    VideoSegment(
                        index=len(segments),
                        start_time=current_time,
                        end_time=current_time + est_duration,
                        text=seg_text,
                    )
                )
                current_time += est_duration
                current_text = []
                current_words = 0

        if current_text:
            seg_text = " ".join(current_text)
            est_duration = max(self.min_duration, current_words / 2.5)
            segments.append(
                VideoSegment(
                    index=len(segments),
                    start_time=current_time,
                    end_time=current_time + est_duration,
                    text=seg_text,
                )
            )

        return segments

    def _normalize_continuity(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Ensure segments are chronologically sorted and non-overlapping."""
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        normalized = []
        for i, s in enumerate(sorted_segs):
            if i > 0 and s.start_time < normalized[-1].end_time:
                # Resolve overlap
                s.start_time = normalized[-1].end_time
            if s.end_time <= s.start_time:
                s.end_time = s.start_time + max(2.0, s.word_count / 2.5)
            s.index = i
            normalized.append(s)
        return normalized

