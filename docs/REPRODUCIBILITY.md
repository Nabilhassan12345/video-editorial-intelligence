# Reproducibility Guide

## Environment Setup

### Requirements

- **Python 3.12** (tested; 3.10+ should also work)
- **Node.js 18+** (for the web UI)
- **FFmpeg** (for audio extraction — `brew install ffmpeg` on macOS)

### Python Virtual Environment

```bash
# From the project root
cd video-editorial-intelligence

python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Node.js Dependencies (web UI only)

```bash
npm install
```

---

## Random Seed

All stochastic operations use **seed = 42** by default, set in [`src/train.py`](../src/train.py):

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

The dataset shuffle in `split_dataset()` also uses `seed=42`. As long as the same Python and library versions are used, the train/val/test split and training trajectory are deterministic.

---

## Reproduce Training

### 1. Dry-run (verify setup)

```bash
.venv/bin/python -m src.train --dry-run
```

Expected output: a few forward/backward passes, then exit without saving checkpoints.

### 2. Full training run

```bash
.venv/bin/python -m src.train
```

Checkpoints are saved to `checkpoints/editorial_model/`. The best checkpoint (by validation loss) is saved as `best_model.pt`.

**Expected result:** best validation loss ≈ **1.4123** at epoch 3.

---

## Reproduce Evaluation

```bash
.venv/bin/python -m src.evaluate --output-dir outputs/evaluation/
```

**Expected metrics (on the 70-sample synthetic dataset):**

| Metric | Fine-Tuned |
|---|---|
| Hook ROC-AUC | 1.0000 |
| Drop-off ROC-AUC | 1.0000 |
| Drop-off Brier Score | 0.1451 |
| Quality RMSE | 2.1304 |
| Quality Pearson r | 0.8259 |

---

## Reproduce End-to-End Video Analysis

### Generate a sample video (demo)

```bash
.venv/bin/python demo/generate_sample_video.py --output demo/sample_video.mp4
```

### Run the pipeline on the sample video

```bash
.venv/bin/python -m src.cli analyze demo/sample_video.mp4 \
    --checkpoint checkpoints/editorial_model/best_model.pt \
    --output outputs/pipeline/
```

### Run via the web UI

```bash
# Terminal 1 — Python backend (if running separately)
.venv/bin/python -m src.cli serve

# Terminal 2 — Node.js server
npm start
# Open http://localhost:3000
```

---

## Run Tests

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

---

## Known Reproducibility Limitations

| Issue | Detail |
|---|---|
| **Library version sensitivity** | PyTorch and HuggingFace Transformers may produce slightly different floating-point results across minor versions. Pin versions with `pip install -r requirements.txt`. |
| **CPU vs. GPU** | Floating-point arithmetic differs between CPU and GPU, so loss values may vary slightly when switching hardware. |
| **Synthetic dataset** | The curated synthetic dataset is generated deterministically from seed=42, but the generation logic itself may change between versions. |
| **Non-deterministic CUDA ops** | Some CUDA kernels (e.g., atomics in scatter operations) are non-deterministic by default. Set `torch.use_deterministic_algorithms(True)` if strict reproducibility on GPU is required (may be slower). |
| **Small dataset variance** | With only 70 samples, metric estimates have high variance. Slight changes in data or preprocessing may produce noticeably different numbers. |
