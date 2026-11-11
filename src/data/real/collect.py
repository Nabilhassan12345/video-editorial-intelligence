"""
Collect real labeled training data from public research datasets.

Writes a JSONL dataset in the exact schema `src/train.py --dataset-path`
consumes, plus a `<out>.meta.json` file recording source, license and counts.

Usage:
    python -m src.data.real.collect --source ttcc --limit 1000 \
        --out data/real/ttcc_train.jsonl
    python -m src.data.real.collect --source vretention \
        --csv-path path/to/youtube.csv --limit 5000 --out data/real/vret.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.data.real.labels import HOOK_RETENTION_THRESHOLD, HOOK_WINDOW_SECONDS

logger = logging.getLogger(__name__)


def _iter_source(args):
    if args.source == "ttcc":
        from src.data.real.ttcc import DATASET_ID, LICENSE_NOTE, collect_ttcc
        return (
            collect_ttcc(
                limit=args.limit,
                transcribe=not args.no_transcribe,
                whisper_model=args.whisper_model,
                include_audio=not args.no_audio,
            ),
            {"dataset": DATASET_ID, "license": LICENSE_NOTE},
        )
    if args.source == "youtube":
        if not args.csv_path:
            raise SystemExit("--csv-path is required for source=youtube (vRetention youtube.csv)")
        from src.data.real.youtube import LICENSE_NOTE, collect_youtube
        return (
            collect_youtube(args.csv_path, limit=args.limit,
                            audio=not args.no_audio, sleep=args.sleep),
            {"dataset": "vRetention curves + YouTube captions (yt-dlp)", "license": LICENSE_NOTE},
        )
    if args.source == "vretention":
        if not args.csv_path:
            raise SystemExit("--csv-path is required for source=vretention")
        from src.data.real.vretention import LICENSE_NOTE, SOURCE_URL, collect_vretention
        return (
            collect_vretention(args.csv_path, limit=args.limit),
            {"dataset": SOURCE_URL, "license": LICENSE_NOTE},
        )
    raise SystemExit(f"Unknown source: {args.source}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="collect-real-data", description=__doc__)
    p.add_argument("--source", choices=["ttcc", "vretention", "youtube"], required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--csv-path", type=str, default=None,
                   help="vretention: local path to youtube.csv")
    p.add_argument("--no-transcribe", action="store_true",
                   help="ttcc: skip whisper, use caption/keywords as text")
    p.add_argument("--no-audio", action="store_true",
                   help="ttcc: skip FFmpeg audio features")
    p.add_argument("--whisper-model", type=str, default="tiny")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="youtube: seconds to wait between caption requests")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    samples, provenance = _iter_source(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = {"written": 0, "transcribed": 0, "hook_positive": 0, "drop_off_positive": 0}
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            counts["written"] += 1
            counts["transcribed"] += int(bool(s.get("transcribed")))
            counts["hook_positive"] += int(s.get("hook_label") == 1.0)
            counts["drop_off_positive"] += int(s.get("drop_off_risk", 0) > 0.5)
            f.write(json.dumps(s) + "\n")

    meta = {
        **provenance,
        "label_definition": {
            "hook_label": f"retained >= {HOOK_RETENTION_THRESHOLD:.0%} of viewers after {HOOK_WINDOW_SECONDS}s",
            "drop_off_risk": "largest single-second relative audience loss, [0,1]",
            "quality_score": "1-10 mapped from retention-curve AUC (mean retained fraction)",
        },
        **counts,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %d samples -> %s (meta: %s)", counts["written"], out, meta_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
