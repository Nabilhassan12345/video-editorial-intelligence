# Evaluation

## Methodology

0. (Optional) Collect real labeled data first: `python -m src.data.real.collect --source ttcc --limit N --out data/real/ttcc_train.jsonl` — see [DATASETS.md](DATASETS.md) for sources and licenses.
1. Generate (or load) the dataset; every sample carries a `group_id` (template or source video).
2. Split with `split_dataset`: whole groups per split, stratified by label type (hook / drop-off / other). `assert_no_leakage` raises if any group or identical text appears in two splits. `src/evaluate.py` recreates the exact same split as `src/train.py`.
3. Evaluate a baseline (`distilbert-base-uncased` with randomly initialised heads, no adapter) and the fine-tuned checkpoint on the test split.
4. Metrics (`scikit-learn` / `scipy`): Hook F1 + ROC-AUC, Drop-off ROC-AUC, PR-AUC and Brier score, Quality RMSE / MAE / R² / Pearson r, CPU latency and throughput. Ranking metrics are reported as `null` / "undefined" when the test split has a single class, instead of a fake 0.5.

## Results (real retention-labeled data — TikTok Creative Center)

Real data collected with `python -m src.data.real.collect --source ttcc --limit 1500 --out data/real/ttcc_train.jsonl` (1,500 ads, 1,207 distinct groups after text-merging; 500 hook-positive, 1,154 drop-off-positive). Labels are **measured per-second audience retention**: hook = ≥20% retained at 3s, drop_off_risk = largest single-second relative loss, quality = retention-curve AUC scaled 1–10. See [DATASETS.md](DATASETS.md).

Commands: `python -m src.train --dataset-path data/real/ttcc_train.jsonl --epochs 3 --batch-size 16 --output-dir checkpoints/editorial_model_real` (CPU, 374s, best val loss 1.5408) then `python -m src.evaluate --checkpoint-dir checkpoints/editorial_model_real --dataset-path data/real/ttcc_train.jsonl` (test split: 216 held-out ads).

| Task / Metric | Baseline (random heads) | Fine-tuned (LoRA) |
|:---|:---:|:---:|
| Hook F1 (threshold 0.5) | 0.4141 | 0.0000 |
| Hook ROC-AUC | 0.4496 | **0.6379** |
| Drop-off ROC-AUC | 0.4719 | **0.7269** |
| Drop-off PR-AUC | 0.7730 | **0.9127** |
| Drop-off Brier (lower is better) | 0.2487 | **0.1667** |
| Quality RMSE (lower is better) | 3.4145 | **0.9230** |
| Quality Pearson r | 0.0280 | -0.0104 |
| CPU latency / segment | – | 42.69 ms (23.4 segments/s) |

**How to read this honestly**

- **Drop-off head clearly learned something real:** ROC-AUC 0.73 / PR-AUC 0.91 on 216 held-out ads is well above the random-heads baseline and above chance — the model ranks high-churn ads correctly.
- **Hook ROC-AUC 0.64 is real but modest; Hook F1 is 0.00.** The hook probability output sits below 0.5 everywhere — the head is uncalibrated at this scale (1.5k samples, ~446k trainable params).
- **Quality head: low RMSE (0.92) but Pearson r ≈ 0.** It learned the mean of a heavily skewed quality distribution (median AUC ~11%), not ordering. Negative result worth reporting; larger data and per-segment (not whole-ad) labels are the fix.
- Dataset license: research-use only; the JSONL is not redistributed.

## Results (synthetic template benchmark)

> [!IMPORTANT]
> These numbers come from the bundled **synthetic** dataset (`generate_curated_editorial_dataset`): 70 samples generated from **10 hand-written sentence templates** with jittered acoustic features and labels. They are a pipeline sanity check, **not** evidence of real-world editorial or retention performance.

### Split (group-aware, stratified by label type)

Samples generated from the same template are near-duplicates, so whole templates are assigned to one split (`src/train.py::split_dataset`, checked by `assert_no_leakage`). An earlier version split by sample, which put copies of the same sentence in train and test and produced ROC-AUC = 1.0 — those numbers were leakage and have been removed.

| Split | Samples | Templates |
|---|---|---|
| Train | 42 | 6 |
| Validation | 7 | 1 |
| Test | 21 | 3 (one hook, one filler/drop-off, one neutral) |

### Metrics on the held-out templates

Command: `python -m src.train --epochs 5` then `python -m src.evaluate --checkpoint-dir checkpoints/editorial_model` (CPU, seed 42, best val loss 1.1412).

| Task / Metric | Baseline (random heads) | Fine-tuned (LoRA) |
|:---|:---:|:---:|
| Hook F1 (threshold 0.5) | 0.0000 | 0.0000 |
| Hook ROC-AUC | 1.0000 | 1.0000 |
| Drop-off ROC-AUC | 0.0000 | 1.0000 |
| Drop-off PR-AUC | 0.2124 | 1.0000 |
| Drop-off Brier (lower is better) | 0.2666 | 0.1432 |
| Quality RMSE (lower is better) | 2.4428 | 1.7399 |
| Quality Pearson r | -0.1506 | 0.9268 |
| CPU latency / segment | – | 17.34 ms (57.7 segments/s) |

**How to read this honestly**

- The test set contains only **3 distinct sentences** (7 jittered copies each). Ranking metrics over 3 sentences are close to meaningless: the random-init baseline also gets Hook ROC-AUC 1.0, and Drop-off ROC-AUC flips between 0 and 1 depending on how one sentence is ranked.
- **Hook F1 is 0** for both models: no hook probability crosses 0.5, so the hook head is uncalibrated. Thresholds were not tuned.
- Brier, RMSE and Pearson r do improve after fine-tuning, which shows the training loop learns *something* transferable across templates, but with 6 training templates this cannot be generalised.
- Largest residual: the fine-tuned model predicts quality ≈5.7 for the filler-heavy "And yeah, like I said before…" template whose label is ≈2.9.

The JSON / Markdown outputs are written to `outputs/evaluation/` (git-ignored; re-run the commands above to reproduce). Exact numbers may vary slightly across PyTorch versions/hardware.

---

## Evaluation Commands

### Run full evaluation (baseline + fine-tuned)

```bash
cd video-editorial-intelligence
.venv/bin/python -m src.evaluate
```

### Evaluate only the fine-tuned model

```bash
.venv/bin/python -m src.evaluate --model-only
```

### Save evaluation artefacts

```bash
.venv/bin/python -m src.evaluate --output-dir outputs/evaluation/
```

The evaluation report is saved to [`outputs/evaluation/evaluation_summary.md`](../outputs/evaluation/evaluation_summary.md).
