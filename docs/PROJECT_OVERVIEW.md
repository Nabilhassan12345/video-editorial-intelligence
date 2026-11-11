# Project Overview

Video Editorial Intelligence analyses a video or transcript and returns timestamped, evidence-linked editorial feedback: the strongest hook, segments at risk of losing viewers, an overall quality score and concrete edit suggestions.

## Two parts

| Part | Origin | What it does |
|---|---|---|
| Node.js/Express API + React/Vite UI (`server/`, `client/`) | Microsoft Türkiye AI Innovators Internship, Summer 2026 | Transcript upload, heuristic segmentation, GPT-4o analysis (or mock), dashboard/history |
| Python ML pipeline (`src/`, `demo/`, `azure/`, `tests/`) | Post-internship | FFmpeg audio features, DistilBERT + LoRA multi-head model, training/evaluation, reports, Streamlit demo, Azure ML / Foundry tooling |

The Node server now calls the Python pipeline first and falls back to the original OpenAI/mock flow.

## What is verified

- Python tests (`pytest tests`) and Node tests (`npm test` in `server/`) pass; client builds; CI runs all three without Azure credentials or GPUs.
- Training and evaluation run on CPU in seconds on the bundled synthetic dataset.

## What is not

- The default checkpoint's reported metrics are on a synthetic template benchmark (see [EVALUATION.md](EVALUATION.md)). Real labeled data IS obtainable now via `src/data/real/collect.py` (TikTok Creative Center retention curves + vRetention YouTube curves — see [DATASETS.md](DATASETS.md)); a full-scale training run on it has not been executed yet.
- No Azure ML job, registration, deployment or Foundry call has been executed from this repository.

See [ARCHITECTURE.md](ARCHITECTURE.md), [DATA_PIPELINE.md](DATA_PIPELINE.md), [FINE_TUNING.md](FINE_TUNING.md), [AZURE_MACHINE_LEARNING.md](AZURE_MACHINE_LEARNING.md), [AZURE_AI_FOUNDRY.md](AZURE_AI_FOUNDRY.md), [DEPLOYMENT.md](DEPLOYMENT.md).
