"""
Collector that pairs vRetention YouTube videos with their real transcripts
(fetched via yt-dlp — the same mechanism Agent-Reach wraps, with no extra
service or API key needed for public captions).

For each video ID in a vRetention CSV we:
- fetch auto/manual captions with `yt-dlp --write-auto-subs --skip-download`
- optionally download the audio track for FFmpeg acoustic features (--audio)
- derive text features via FeatureEngineer
- derive labels from the video's measured retention curve (src.data.real.labels)

YouTube ToS note: caption download for personal/research use at modest rates is
fine; do not hammer. `--sleep` adds a delay between requests.
"""

import glob
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Dict, Iterator, Optional

from src.audio.extractor import AudioExtractor
from src.data.feature_engineering import FeatureEngineer
from src.data.real.labels import (
    curve_auc,
    drop_off_risk_from_curve,
    first_drop_off_second,
    hook_label_from_curve,
    quality_from_curve,
)
from src.data.real.vretention import parse_svg_retention_path

logger = logging.getLogger(__name__)

LICENSE_NOTE = (
    "vRetention academic dataset (retention curves, research-use only) + "
    "YouTube public captions fetched via yt-dlp."
)


def _run_ytdlp(args, timeout=60):
    import sys
    return subprocess.run(
        [sys.executable, "-m", "yt_dlp", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def fetch_transcript(video_id: str, tmpdir: str) -> Optional[str]:
    """Download auto/manual captions for a video and return plain text, or None."""
    out = os.path.join(tmpdir, "cap")
    proc = _run_ytdlp([
        "--skip-download", "--write-auto-subs", "--write-subs",
        "--sub-langs", "en.*,-live_chat", "--sub-format", "vtt",
        "-o", out, f"https://www.youtube.com/watch?v={video_id}",
    ])
    if proc.returncode != 0:
        logger.debug("captions failed for %s: %s", video_id, proc.stderr[-200:])
        return None
    vtts = [p for p in glob.glob(os.path.join(tmpdir, "cap*.vtt"))]
    if not vtts:
        return None
    return _vtt_to_text(sorted(vtts)[0])


def _vtt_to_text(path: str) -> Optional[str]:
    lines, seen = [], set()
    for raw in open(path, encoding="utf-8", errors="ignore"):
        line = raw.strip()
        if not line or "-->" in line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    text = " ".join(lines)
    return text or None


def fetch_audio(video_id: str, tmpdir: str) -> Optional[str]:
    """Download best audio to tmpdir (m4a→wav handled by FFmpeg extractor)."""
    proc = _run_ytdlp([
        "-f", "bestaudio", "-x", "--audio-format", "mp3",
        "-o", os.path.join(tmpdir, "audio.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ], timeout=180)
    hits = glob.glob(os.path.join(tmpdir, "audio.*"))
    return hits[0] if proc.returncode == 0 and hits else None


def collect_youtube(csv_path: str, limit: int = 200, audio: bool = False,
                    sleep: float = 1.0) -> Iterator[Dict]:
    """Yield samples combining vRetention curves with real YouTube transcripts."""
    import pandas as pd

    engineer = FeatureEngineer()
    extractor = AudioExtractor() if audio else None
    df = pd.read_csv(csv_path)
    emitted = 0

    for _, row in df.iterrows():
        if emitted >= limit:
            break
        vid = str(row.get("id", "")).strip()
        curve = parse_svg_retention_path(str(row.get("retentionCurve", "")))
        if not vid or not curve:
            continue

        duration = float(row.get("duration") or len(curve) - 1 or 1)
        text = None
        rms, silence = 0.0, 0.0
        with tempfile.TemporaryDirectory() as tmp:
            text = fetch_transcript(vid, tmp)
            if audio and extractor and extractor.is_available:
                aud = fetch_audio(vid, tmp)
                if aud:
                    try:
                        wav = extractor.extract_audio(aud, os.path.join(tmp, "a_16k.wav"))
                        arr, rate = extractor.load_audio_samples(wav)
                        af = extractor.compute_segment_features(
                            arr, rate, start_time=0.0, end_time=min(duration, 120.0))
                        rms, silence = af.rms_energy, af.silence_ratio
                    except Exception as e:
                        logger.debug("audio features failed for %s: %s", vid, e)

        if not text:
            continue  # no transcript -> no text features; skip
        emitted += 1
        tf = engineer.extract_text_features(text, duration)
        words = len(text.split())
        wpm = (words / (duration / 60.0)) if duration > 0 else 0.0
        vec = [rms, silence, min(wpm / 200.0, 2.0), tf["lexical_density"],
               tf["type_token_ratio"], tf["hook_cue_score"],
               tf["filler_word_ratio"], min(5.0, tf["words_per_second"]) / 5.0]
        yield {
            "id": f"yt_{vid}",
            "group_id": f"yt_{vid}",
            "video_id": vid,
            "text": text[:2000],
            "feature_vector": vec,
            "hook_label": hook_label_from_curve(curve),
            "drop_off_risk": drop_off_risk_from_curve(curve),
            "quality_score": quality_from_curve(curve),
            "duration": round(duration, 2),
            "retention_curve": curve,
            "first_drop_off_second": first_drop_off_second(curve),
            "source": "youtube_vretention",
            "transcribed": False,
            "has_captions": True,
            "features_complete": audio,
            "engagement": {"viewCount": row.get("viewCount"), "categoryId": row.get("categoryId")},
        }
        time.sleep(sleep)
