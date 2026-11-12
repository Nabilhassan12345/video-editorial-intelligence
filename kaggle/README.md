# Free-GPU training on Kaggle

Kaggle gives ~30 free GPU hours/week (phone-verified account). This runs the full pipeline there:
all ~36.5k English TikTok hooks, as many TTCC ads as fit in the time budget (default 4 h, up to 61,789),
plus your locally collected vRetention / YouTube-caption / TTCC files, then trains + evaluates on CUDA.

```bash
pip install kaggle
export KAGGLE_USERNAME=... KAGGLE_KEY=...      # or ~/.kaggle/kaggle.json
python kaggle/launch.py push                   # private dataset + private GPU kernel
python kaggle/launch.py status
python kaggle/launch.py fetch                  # -> kaggle_output/{editorial_model_kaggle,reports,corpus_counts.json}
```

The uploaded corpus dataset is **private**: TTCC and vRetention are research-use only.
Tune via env vars in `kaggle/run_kaggle.py` (`TTCC_MINUTES` default 120, `TTCC_WORKERS` default 2× CPU count — parallel workers over disjoint TTCC parquet shards, `TTCC_LIMIT`, `EPOCHS`, `BATCH_SIZE`).
