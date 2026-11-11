"""
Audio extraction and acoustic feature extraction using FFmpeg and standard audio processing.
"""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}


@dataclass
class AudioFeatures:
    """Acoustic feature representation for a temporal segment."""
    start_time: float
    end_time: float
    duration: float
    rms_energy: float
    silence_ratio: float
    speaking_rate_wpm: float
    energy_variance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "rms_energy": round(self.rms_energy, 4),
            "silence_ratio": round(self.silence_ratio, 4),
            "speaking_rate_wpm": round(self.speaking_rate_wpm, 1),
            "energy_variance": round(self.energy_variance, 4),
        }


class AudioExtractor:
    """Extracts audio tracks from video files and extracts pacing & acoustic signals."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self._ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Verify whether ffmpeg is available in the current environment."""
        try:
            res = subprocess.run(
                [self.ffmpeg_bin, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except FileNotFoundError:
            logger.warning("FFmpeg executable not found. Audio extraction will use fallback mock features.")
            return False

    @property
    def is_available(self) -> bool:
        return self._ffmpeg_available

    def validate_video_file(self, video_path: str) -> Path:
        """Validate that the file exists and is a supported video format."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {video_path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_VIDEO_FORMATS and suffix not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported media format '{suffix}'. Supported formats: {sorted(SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS)}"
            )
        return path

    def extract_audio(
        self,
        video_path: str,
        output_wav_path: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> Path:
        """
        Extract mono WAV audio at 16kHz from a video file using FFmpeg.
        
        Args:
            video_path: Path to the input video.
            output_wav_path: Optional destination WAV path.
            sample_rate: Target audio sample rate (default: 16000 Hz).
            
        Returns:
            Path to the extracted WAV file.
        """
        validated_path = self.validate_video_file(video_path)

        if not self._ffmpeg_available:
            raise RuntimeError(
                "FFmpeg is required to extract audio from video files but was not found in PATH. "
                "Please install FFmpeg or provide a pre-extracted WAV audio file."
            )

        if output_wav_path is None:
            output_dir = validated_path.parent / "extracted_audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_wav = output_dir / f"{validated_path.stem}_16k.wav"
        else:
            output_wav = Path(output_wav_path)
            output_wav.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",  # overwrite
            "-i", str(validated_path),
            "-vn",  # no video
            "-acodec", "pcm_s16le",
            "-ac", "1",  # mono
            "-ar", str(sample_rate),
            str(output_wav),
        ]

        logger.info(f"Extracting audio using FFmpeg: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed (code {proc.returncode}): {proc.stderr}"
            )

        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg completed but output audio file is missing or empty: {output_wav}")

        return output_wav

    def load_audio_samples(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Load PCM 16-bit mono WAV samples into a float32 NumPy array [-1.0, 1.0].
        Uses standard library `wave` to avoid heavy external dependencies.
        """
        wav_p = Path(wav_path)
        if not wav_p.exists():
            raise FileNotFoundError(f"WAV audio file not found: {wav_path}")

        with wave.open(str(wav_p), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            raw_bytes = wf.readframes(num_frames)

        if sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        samples = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        samples = samples / max_val
        return samples, sample_rate

    def compute_segment_features(
        self,
        samples: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        word_count: int = 0,
        silence_threshold_db: float = -38.0,
    ) -> AudioFeatures:
        """
        Compute acoustic and pacing metrics for a specific time window.
        
        Args:
            samples: Audio array [-1.0, 1.0].
            sample_rate: Sampling frequency in Hz.
            start_time: Window start in seconds.
            end_time: Window end in seconds.
            word_count: Number of spoken words in this segment.
            silence_threshold_db: Energy cutoff in dB for silence detection.
        """
        duration = max(0.1, end_time - start_time)
        start_idx = max(0, int(start_time * sample_rate))
        end_idx = min(len(samples), int(end_time * sample_rate))

        if start_idx >= len(samples) or end_idx <= start_idx:
            # Segment out of range or empty
            return AudioFeatures(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                rms_energy=0.0,
                silence_ratio=1.0,
                speaking_rate_wpm=0.0,
                energy_variance=0.0,
            )

        segment_samples = samples[start_idx:end_idx]

        # RMS Energy
        square_mean = np.mean(segment_samples ** 2) if len(segment_samples) > 0 else 0.0
        rms_energy = float(np.sqrt(max(1e-12, square_mean)))

        # Frame-level silence ratio (e.g., 20ms frames)
        frame_len = max(1, int(0.02 * sample_rate))
        num_frames = len(segment_samples) // frame_len
        if num_frames > 0:
            frames = segment_samples[:num_frames * frame_len].reshape(num_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            frame_db = 20.0 * np.log10(frame_rms)
            silence_frames = np.sum(frame_db < silence_threshold_db)
            silence_ratio = float(silence_frames / num_frames)
            energy_variance = float(np.var(frame_rms))
        else:
            silence_ratio = 0.0
            energy_variance = 0.0

        # Speaking rate (Words Per Minute)
        speaking_rate_wpm = (word_count / (duration / 60.0)) if duration > 0 else 0.0

        return AudioFeatures(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            rms_energy=rms_energy,
            silence_ratio=silence_ratio,
            speaking_rate_wpm=speaking_rate_wpm,
            energy_variance=energy_variance,
        )

    def extract_features_for_segments(
        self,
        wav_path: Optional[str],
        segments: List[Dict],
    ) -> List[AudioFeatures]:
        """
        Extract features for all segments. If wav_path is None or unavailable,
        computes text-derived approximations for pacing.
        """
        if wav_path and Path(wav_path).exists():
            try:
                samples, sr = self.load_audio_samples(wav_path)
                features_list = []
                for seg in segments:
                    st = float(seg.get("start_time", 0.0))
                    et = float(seg.get("end_time", st + 5.0))
                    text = seg.get("text", "")
                    words = len(text.split())
                    feat = self.compute_segment_features(samples, sr, st, et, word_count=words)
                    features_list.append(feat)
                return features_list
            except Exception as e:
                logger.warning(f"Failed to load WAV audio for features: {e}. Falling back to text heuristics.")

        # Fallback acoustic simulation based on transcript text
        features_list = []
        for seg in segments:
            st = float(seg.get("start_time", 0.0))
            et = float(seg.get("end_time", st + 5.0))
            dur = max(0.5, et - st)
            text = seg.get("text", "")
            words = len(text.split())
            wpm = (words / (dur / 60.0)) if dur > 0 else 140.0
            # Heuristic volume based on exclamation and punctuation
            rms = 0.08 + (0.04 if "!" in text else 0.0)
            # Heuristic silence based on pause commas and periods
            silence_ratio = min(0.4, (text.count(",") * 0.05 + text.count("...") * 0.1) / max(1, dur))
            features_list.append(
                AudioFeatures(
                    start_time=st,
                    end_time=et,
                    duration=dur,
                    rms_energy=round(rms, 4),
                    silence_ratio=round(silence_ratio, 4),
                    speaking_rate_wpm=round(wpm, 1),
                    energy_variance=0.002,
                )
            )
        return features_list

