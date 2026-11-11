"""
Lexical, syntactic, and multimodal feature engineering for video editorial sequence modeling.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..audio.extractor import AudioFeatures
from .segmenter import VideoSegment


HOOK_TRIGGER_WORDS = {
    "secret", "never", "revealed", "truth", "mistake", "warning", "how to",
    "stop", "why", "shocking", "insane", "million", "dollar", "today",
    "watch this", "don't", "proven", "hack", "strategy", "nobody", "everyone"
}

FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "actually", "literally",
    "sort of", "kind of", "so yeah", "i mean", "right"
}


@dataclass
class MultimodalSegment:
    """Unified multimodal representation of a video segment."""
    segment: VideoSegment
    audio_features: AudioFeatures
    lexical_density: float
    type_token_ratio: float
    hook_cue_score: float
    filler_word_ratio: float
    words_per_second: float
    question_count: int
    exclamation_count: int
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.segment.index,
            "start_time": round(self.segment.start_time, 2),
            "end_time": round(self.segment.end_time, 2),
            "start_timestamp": self.segment.start_timestamp,
            "end_timestamp": self.segment.end_timestamp,
            "duration": round(self.segment.duration, 2),
            "text": self.segment.text,
            "word_count": self.segment.word_count,
            "audio": self.audio_features.to_dict(),
            "lexical": {
                "lexical_density": round(self.lexical_density, 3),
                "type_token_ratio": round(self.type_token_ratio, 3),
                "hook_cue_score": round(self.hook_cue_score, 3),
                "filler_word_ratio": round(self.filler_word_ratio, 3),
                "words_per_second": round(self.words_per_second, 2),
                "question_count": self.question_count,
                "exclamation_count": self.exclamation_count,
            },
        }


class FeatureEngineer:
    """Extracts lexical and multimodal features and aligns them with video timestamps."""

    def __init__(self, optimal_wpm_range: tuple = (140.0, 180.0)):
        self.optimal_wpm_low, self.optimal_wpm_high = optimal_wpm_range

    def extract_text_features(self, text: str, duration: float) -> Dict[str, Any]:
        """Compute lexical, syntactic, and structural metrics from text."""
        words = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        total_words = len(words)
        duration = max(0.1, duration)

        if total_words == 0:
            return {
                "lexical_density": 0.0,
                "type_token_ratio": 0.0,
                "hook_cue_score": 0.0,
                "filler_word_ratio": 0.0,
                "words_per_second": 0.0,
                "question_count": 0,
                "exclamation_count": 0,
            }

        # Type-Token Ratio (vocabulary richness)
        unique_words = set(words)
        ttr = len(unique_words) / total_words

        # Lexical Density: proportion of non-filler content words
        content_words = [w for w in words if len(w) > 3 and w not in FILLER_WORDS]
        lexical_density = len(content_words) / total_words

        # Hook cues: presence of curiosity, urgency, and high-retention keywords
        text_lower = text.lower()
        hook_hits = sum(1 for hw in HOOK_TRIGGER_WORDS if hw in text_lower)
        # Normalize hook cue score into [0, 1] range
        hook_cue_score = min(1.0, (hook_hits * 0.25) + (0.2 if "?" in text else 0.0))

        # Filler words ratio
        filler_hits = sum(1 for fw in FILLER_WORDS if fw in text_lower)
        filler_ratio = min(1.0, filler_hits / total_words)

        # Pacing
        words_per_sec = total_words / duration

        # Punctuation signals
        question_count = text.count("?")
        exclamation_count = text.count("!")

        return {
            "lexical_density": lexical_density,
            "type_token_ratio": ttr,
            "hook_cue_score": hook_cue_score,
            "filler_word_ratio": filler_ratio,
            "words_per_second": words_per_sec,
            "question_count": question_count,
            "exclamation_count": exclamation_count,
        }

    def fuse_features(
        self,
        segment: VideoSegment,
        audio_features: AudioFeatures,
    ) -> MultimodalSegment:
        """Combine text metrics and acoustic signals into a unified MultimodalSegment."""
        text_feats = self.extract_text_features(segment.text, segment.duration)

        # 8-dimensional dense numerical vector for MLP heads:
        # [rms_energy, silence_ratio, wpm_norm, lexical_density, ttr, hook_cue, filler_ratio, words_per_sec]
        wpm_norm = float(np.clip(audio_features.speaking_rate_wpm / 200.0, 0.0, 2.0))
        vec = np.array([
            float(audio_features.rms_energy),
            float(audio_features.silence_ratio),
            wpm_norm,
            float(text_feats["lexical_density"]),
            float(text_feats["type_token_ratio"]),
            float(text_feats["hook_cue_score"]),
            float(text_feats["filler_word_ratio"]),
            float(min(5.0, text_feats["words_per_second"]) / 5.0),
        ], dtype=np.float32)

        return MultimodalSegment(
            segment=segment,
            audio_features=audio_features,
            lexical_density=text_feats["lexical_density"],
            type_token_ratio=text_feats["type_token_ratio"],
            hook_cue_score=text_feats["hook_cue_score"],
            filler_word_ratio=text_feats["filler_word_ratio"],
            words_per_second=text_feats["words_per_second"],
            question_count=text_feats["question_count"],
            exclamation_count=text_feats["exclamation_count"],
            feature_vector=vec,
        )

    def process_timeline(
        self,
        segments: List[VideoSegment],
        audio_features: List[AudioFeatures],
    ) -> List[MultimodalSegment]:
        """Align and fuse all segments across the video timeline."""
        results = []
        for i, seg in enumerate(segments):
            af = audio_features[i] if i < len(audio_features) else AudioFeatures(
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=seg.duration,
                rms_energy=0.08,
                silence_ratio=0.05,
                speaking_rate_wpm=(seg.word_count / (seg.duration / 60.0)) if seg.duration > 0 else 140.0,
                energy_variance=0.001,
            )
            mm = self.fuse_features(seg, af)
            results.append(mm)
        return results

