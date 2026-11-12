"""
Adapter for the TikTok Creative Center Top Ads corpus
(`liangyuch/ttcc-v0_2_0` on Hugging Face, 61,789 ads).

Each ad ships an embedded mp4 plus a per-second audience `retention_curve` and
engagement metrics (CTR global percentile, likes, comments, shares). We derive:

- audio features  -> FFmpeg on the embedded mp4 (rms_energy, silence_ratio)
- text            -> openai-whisper "tiny" transcription when installed, else
                     the ad's caption / highlight text / keywords
- text features   -> src.data.feature_engineering.FeatureEngineer
- labels          -> src.data.real.labels (hook, drop-off, quality)

LICENSE: research-use only (see the dataset card). Nothing here redistributes
the data — users download it themselves; do not commit collected output.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.audio.extractor import AudioExtractor
from src.data.feature_engineering import FeatureEngineer
from src.data.real.labels import (
    curve_auc,
    drop_off_risk_from_curve,
    first_drop_off_second,
    hook_label_from_curve,
    quality_from_curve,
)

logger = logging.getLogger(__name__)

DATASET_ID = "liangyuch/ttcc-v0_2_0"
LICENSE_NOTE = (
    "TikTok Creative Center Top Ads dataset — research-use only per dataset card. "
    "Not for commercial use or redistribution."
)


def _load_whisper(model_name: str):
    try:
        import whisper
    except ImportError:
        return None
    logger.info("Transcribing with openai-whisper model '%s'", model_name)
    return whisper.load_model(model_name)


def _write_mp4(row: Dict, tmpdir: str) -> Optional[str]:
    vid = row.get("video_local_path")
    blob = vid.get("bytes") if isinstance(vid, dict) else vid
    if not blob:
        return None
    path = os.path.join(tmpdir, f"{row['ad_id']}.mp4")
    with open(path, "wb") as f:
        f.write(blob)
    return path


def _fallback_text(row: Dict) -> str:
    parts = [row.get("caption"), row.get("highlight_text"), row.get("ad_title")]
    kw = row.get("keyword_list") or []
    parts.extend(str(k) for k in kw)
    return " ".join(p for p in parts if p)


def _iter_streams(load_dataset, splits, num_shards: int, shard_index: int):
    """Yield ``(split, streaming_dataset)`` pairs, optionally restricted to one shard of parquet files."""
    if num_shards <= 1:
        for split in splits:
            # Sequential streaming: shuffling here would pull many parquet shards at once.
            yield split, load_dataset(DATASET_ID, split=split, streaming=True)
        return
    from huggingface_hub import HfApi

    files = sorted(
        f for f in HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
        if f.endswith(".parquet") and f.rsplit("/", 1)[-1].split("-")[0] in splits
    )
    for path in files[shard_index::num_shards]:
        split = path.rsplit("/", 1)[-1].split("-")[0]
        yield split, load_dataset(DATASET_ID, data_files=path, split="train", streaming=True)


def collect_ttcc(
    limit: int = 1000,
    transcribe: bool = True,
    whisper_model: str = "tiny",
    splits: tuple = ("train", "val", "test"),
    include_audio: bool = True,
    progress_every: int = 50,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[Dict]:
    """Stream collected samples from the TTCC corpus (no dataset files persisted).

    With ``num_shards > 1`` only every ``num_shards``-th parquet file (offset ``shard_index``)
    is read, so several processes can collect disjoint parts of the corpus in parallel.
    """
    from datasets import load_dataset  # lazy: only needed when collecting

    extractor = AudioExtractor() if include_audio else None
    engineer = FeatureEngineer()
    whisper = _load_whisper(whisper_model) if transcribe else None

    seen = 0
    for split, ds in _iter_streams(load_dataset, splits, num_shards, shard_index):
        if seen >= limit:
            break
        for row in ds:
            if seen >= limit:
                break
            curve = row.get("retention_curve") or []
            duration = float(row.get("duration") or max(len(curve) - 1, 1))
            if not curve or duration <= 0:
                continue

            sample_id = f"ttcc_{row['ad_id']}"
            text = ""
            rms, silence, wpm = 0.0, 0.0, 0.0
            transcribed = False

            with tempfile.TemporaryDirectory() as tmp:
                mp4 = _write_mp4(row, tmp)
                wav = None
                if mp4 and extractor and extractor.is_available:
                    try:
                        wav = extractor.extract_audio(mp4, output_wav_path=os.path.join(tmp, "audio_16k.wav"))
                    except Exception as e:  # pragma: no cover - ffmpeg edge cases
                        logger.debug("audio extract failed for %s: %s", sample_id, e)
                if wav:
                    try:
                        samples_arr, rate = extractor.load_audio_samples(wav)
                        af = extractor.compute_segment_features(
                            samples_arr, rate, start_time=0.0, end_time=duration, word_count=0
                        )
                        rms, silence = af.rms_energy, af.silence_ratio
                    except Exception as e:
                        logger.debug("audio features failed for %s: %s", sample_id, e)
                if wav and whisper is not None:
                    try:
                        res = whisper.transcribe(wav, fp16=False)
                        text = (res.get("text") or "").strip()
                        transcribed = bool(text)
                    except Exception as e:
                        logger.debug("whisper failed for %s: %s", sample_id, e)

            if not text:
                text = _fallback_text(row)

            text_feats = engineer.extract_text_features(text, duration)
            words = len(text.split())
            wpm = (words / (duration / 60.0)) if duration > 0 else 0.0
            vec = [
                rms,
                silence,
                float(min(wpm / 200.0, 2.0)),
                float(text_feats["lexical_density"]),
                float(text_feats["type_token_ratio"]),
                float(text_feats["hook_cue_score"]),
                float(text_feats["filler_word_ratio"]),
                float(min(5.0, text_feats["words_per_second"]) / 5.0),
            ]

            ret_1s = curve[min(1, len(curve) - 1)] / (curve[0] or 100.0)
            ret_3s = curve[min(3, len(curve) - 1)] / (curve[0] or 100.0)

            yield {
                "id": sample_id,
                "group_id": f"ttcc_{row['ad_id']}",
                "video_id": str(row["ad_id"]),
                "text": text,
                "feature_vector": vec,
                "hook_label": hook_label_from_curve(curve),
                "drop_off_risk": drop_off_risk_from_curve(curve),
                "quality_score": quality_from_curve(curve),
                "duration": round(duration, 2),
                "retention_1s": round(ret_1s, 4),
                "retention_3s": round(ret_3s, 4),
                "retention_auc": round(curve_auc(curve), 4),
                "retention_curve": curve,
                "first_drop_off_second": first_drop_off_second(curve),
                "source": "ttcc",
                "hf_split": split,
                "transcribed": transcribed,
                "engagement": {
                    "percentile_global": row.get("percentile_global"),
                    "ctr_global": row.get("ctr_global"),
                    "like_cnt": row.get("like_cnt"),
                    "comments_cnt": row.get("comments_cnt"),
                    "shares_cnt": row.get("shares_cnt"),
                    "industry": row.get("industry_name"),
                },
            }
            seen += 1
            if seen % progress_every == 0:
                logger.info("collected %d/%d samples", seen, limit)
