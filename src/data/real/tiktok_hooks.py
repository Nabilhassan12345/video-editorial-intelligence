"""Collector for the benxh/tiktok-hooks-finetune dataset (46,605 rows).

Each row is a real TikTok post's text hook + engagement stats. There is no
retention curve, so labels are engagement-derived instead of retention-derived:

- hook_label   = 1 when `outlier_multiplier` >= HOOK_OUTLIER_THRESHOLD
                 (outlier_multiplier = this post's views / the account's median —
                 a strongly out-performing post had a hook that landed)
- drop_off_risk = 0.0 (no per-second signal in this dataset)
- quality_score = 1-10 mapped from the log-scaled outlier multiplier

Text features are real (FeatureEngineer on the hook + caption); acoustic
features are zeros (no video/audio in this dataset).
"""

import logging
from typing import Dict, Iterator

from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

DATASET_ID = "benxh/tiktok-hooks-finetune"
LICENSE_NOTE = "Public Hugging Face dataset (tiktok-hooks-finetune); verify terms before redistribution."
HOOK_OUTLIER_THRESHOLD = 2.0


def _quality_from_outlier(mult: float) -> float:
    import math
    if mult <= 0:
        return 1.0
    # log-scale: multiplier 1 ~ 2.5, 10 ~ 5.0, 100 ~ 7.5, 1000 ~ 10
    return float(max(1.0, min(10.0, 2.5 + 2.5 * math.log10(mult))))


def collect_tiktok_hooks(limit: int = 20000, lang: str = "en") -> Iterator[Dict]:
    from datasets import load_dataset

    engineer = FeatureEngineer()
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    emitted = 0
    for row in ds:
        if emitted >= limit:
            break
        hook = (row.get("text_hook") or "").strip().strip('"')
        if not hook:
            continue
        if lang and row.get("text_hook_lang") != lang:
            continue
        caption = (row.get("caption") or "").strip()
        text = f"{hook} {caption}".strip()
        duration = float(row.get("length") or 7.0)
        mult = float(row.get("outlier_multiplier") or 0.0)
        tf = engineer.extract_text_features(text, duration)
        wpm = (len(text.split()) / (duration / 60.0)) if duration > 0 else 0.0
        vec = [0.0, 0.0, min(wpm / 200.0, 2.0), tf["lexical_density"],
               tf["type_token_ratio"], tf["hook_cue_score"],
               tf["filler_word_ratio"], min(5.0, tf["words_per_second"]) / 5.0]
        emitted += 1
        yield {
            "id": f"hook_{emitted}",
            "group_id": row.get("username") or f"hook_{emitted}",
            "video_id": row.get("ad_link", "").rstrip("/").split("/")[-1] or f"hook_{emitted}",
            "text": text[:2000],
            "feature_vector": vec,
            "hook_label": 1.0 if mult >= HOOK_OUTLIER_THRESHOLD else 0.0,
            "drop_off_risk": 0.0,
            "quality_score": _quality_from_outlier(mult),
            "duration": duration,
            "retention_curve": None,
            "first_drop_off_second": None,
            "source": "tiktok_hooks",
            "transcribed": False,
            "features_complete": False,
            "engagement": {
                "views": row.get("views"), "likes": row.get("likes"),
                "comments": row.get("comments"), "shares": row.get("shares"),
                "outlier_multiplier": mult,
                "main_category": row.get("main_category"),
            },
        }
