# Real Training Data

The model can be trained on **real audience-retention data**, not only the bundled
synthetic set. Adapters live in [`src/data/real/`](../src/data/real/) and emit the
exact per-sample schema `src/train.py --dataset-path` consumes.

## Sources

| Source | Size | What's real | Access |
|---|---|---|---|
| [TikTok Creative Center Top Ads](https://huggingface.co/datasets/liangyuch/ttcc-v0_2_0) (`ttcc`) | 61,789 ads | Per-second (1 Hz) **retention curves**, CTR percentiles, likes/comments/shares, embedded mp4 + audio | Public on Hugging Face. **Research-use only** — do not redistribute or use commercially. |
| [vRetention](https://github.com/flowtele/vRetention) (`vretention`) | thousands of videos | Real YouTube & Bilibili audience-retention curves | Academic dataset; download `youtube.csv` yourself (it's a Git-LFS file — `git lfs pull` or the media.githubusercontent.com URL). **Research-use only.** |
| vRetention + YouTube captions (`youtube`) | bounded by how many IDs you fetch | Same retention curves **plus real transcripts** fetched with yt-dlp (the mechanism Agent-Reach wraps); optional audio download for acoustic features | Public captions; keep `--sleep` ≥ 0.3 and modest limits. |
| [tiktok-hooks-finetune](https://huggingface.co/datasets/benxh/tiktok-hooks-finetune) (`tiktok-hooks`) | 46,605 posts (~12k English hooks collected) | Real TikTok hook text + caption + engagement (`outlier_multiplier` = views / account median) | Public HF dataset. Labels are **engagement-derived** (hook = outlier ≥ 2.0), not retention — no curves in this source. |

Labels are **derived from measured viewer retention**, not generated:

- `hook_label` — 1 if the clip keeps ≥ 20% of viewers after the first 3 seconds
  (≈ top quartile on short-form; the corpus median 3-second retention is ~10%).
- `drop_off_risk` — the largest single-second *relative* audience loss in the
  measured curve, in [0, 1].
- `quality_score` — 1–10 mapped from the curve's normalised area under it
  (mean retained fraction). The corpus's own `percentile_global` saturates at 99,
  so retention AUC is the discriminative signal.

Feature vectors (audio RMS/silence from FFmpeg, lexical density, TTR, hook cues,
filler ratio, pacing) are computed per sample the same way as inference, so
train/test features match what the pipeline produces at runtime.

## Collect

```bash
# TikTok corpus — streams from Hugging Face; writes JSONL + a .meta.json
# provenance/license file. Whisper "tiny" transcribes audio when installed.
.venv/bin/python -m src.data.real.collect --source ttcc \
    --limit 5000 --out data/real/ttcc_train.jsonl

# Faster collection without transcription/audio features:
.venv/bin/python -m src.data.real.collect --source ttcc \
    --no-transcribe --no-audio --limit 10000 --out data/real/ttcc_train.jsonl

# YouTube curves — download the dataset yourself first:
#   git clone https://github.com/flowtele/vRetention
.venv/bin/python -m src.data.real.collect --source vretention \
    --csv-path vRetention/dataset/youtube.csv --limit 5000 --out data/real/vret.jsonl

# Same curves + real transcripts fetched per video ID (slow — one yt-dlp call each):
.venv/bin/python -m src.data.real.collect --source youtube \
    --csv-path vRetention/dataset/youtube.csv --limit 200 --out data/real/yt.jsonl
# add --no-audio to skip audio downloads (transcripts only)

# TikTok hooks — text + engagement-derived labels (fast, no media downloads):
.venv/bin/python -m src.data.real.collect --source tiktok-hooks \
    --limit 20000 --lang en --out data/real/hooks.jsonl

# Combine corpora for a multi-source training set:
cat data/real/ttcc_train.jsonl data/real/vret.jsonl \
    data/real/youtube_transcripts.jsonl data/real/hooks.jsonl \
    > data/real/combined.jsonl
.venv/bin/python -m src.train --dataset-path data/real/combined.jsonl \
    --epochs 4 --batch-size 32 --output-dir checkpoints/editorial_model_combined
```

`data/real/` is git-ignored on purpose: the licenses allow research use, not
redistribution.

## Train

```bash
# Local (CPU works but is slow at scale)
.venv/bin/python -m src.train --dataset-path data/real/ttcc_train.jsonl \
    --epochs 3 --output-dir checkpoints/editorial_model_real

# Azure ML (dry-run by default; add --submit to actually submit — incurs cost)
.venv/bin/python -m azure.ml.submit_training \
    --dataset-path data/real/ttcc_train.jsonl --epochs 3
```

## Honest limits

- Retention curves are **platform-aggregated** (views across the ad's delivery
  window), not per-session telemetry. Labels measure ad engagement, which is a
  proxy for — not identical to — organic editorial quality.
- Whisper transcriptions are only as good as a `tiny` CPU model; many ads are
  music-only and produce no transcript (see `transcribed` count in the meta file).
- vRetention rows carry no audio/text features (`features_complete: false`);
  they supervise the drop-off and hook heads only.
