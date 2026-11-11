# Evaluation

## Methodology

1. Generate (or load) the dataset; every sample carries a `group_id` (template or source video).
2. Split with `split_dataset`: whole groups per split, stratified by label type (hook / drop-off / other). `assert_no_leakage` raises if any group or identical text appears in two splits. `src/evaluate.py` recreates the exact same split as `src/train.py`.
3. Evaluate a baseline (`distilbert-base-uncased` with randomly initialised heads, no adapter) and the fine-tuned checkpoint on the test split.
4. Metrics (`scikit-learn` / `scipy`): Hook F1 + ROC-AUC, Drop-off ROC-AUC, PR-AUC and Brier score, Quality RMSE / MAE / R² / Pearson r, CPU latency and throughput. Ranking metrics are reported as `null` / "undefined" when the test split has a single class, instead of a fake 0.5.

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
