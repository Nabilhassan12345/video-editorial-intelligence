# Fine-Tuning Guide (PEFT LoRA)

## Methodology

This project uses **Parameter-Efficient Fine-Tuning (PEFT)** with **Low-Rank Adaptation (LoRA)** to adapt a frozen `distilbert-base-uncased` backbone for three video editorial prediction tasks simultaneously.

### Why LoRA?

Full fine-tuning of a 66 M-parameter transformer requires significant GPU memory and risks catastrophic forgetting. LoRA instead decomposes weight updates into two small low-rank matrices:

```
W' = W + ΔW = W + B·A   where A ∈ R^{r×d}, B ∈ R^{d×r}
```

Only `A` and `B` are learned; `W` stays frozen. At rank `r=16`, this reduces trainable parameters from ~66 M to ~446 K — a **99.33 % reduction**.

### Target Modules

LoRA adapters are injected into:

| Module name | Layer type | Note |
|---|---|---|
| `q_lin` | Query projection | DistilBERT-specific naming |
| `v_lin` | Value projection | DistilBERT-specific naming |

> [!IMPORTANT]
> DistilBERT names its attention projections `q_lin` and `v_lin`, **not** `query` and `value` (which are BERT-base names). Using wrong target modules silently skips adapter injection.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha (`α`) | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_lin`, `v_lin` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient accumulation steps | 1 |
| Max gradient norm | 1.0 |
| Random seed | 42 |
| Loss weights | hook×1.0, drop-off×1.5, quality×0.5 |

The composite loss is:

```
L = 1.0·L_hook_BCE + 1.5·L_dropoff_BCE + 0.5·L_quality_MSE
```

---

## Actual Training Results

| Metric | Value |
|---|---|
| **Best validation loss** | **1.4123** (epoch 3) |
| Trainable parameters | 446,211 |
| Total parameters | 66,809,091 |
| % trainable | **0.67 %** |

> [!NOTE]
> These results use a **70-sample curated synthetic dataset** for demonstration. Real-world training requires large labeled datasets with ground-truth viewer retention labels (e.g., from YouTube Analytics or A/B testing platforms).

---

## Commands

### Dry-run (verify setup, no weights saved)

```bash
cd video-editorial-intelligence
.venv/bin/python -m src.train --dry-run
```

### Full training

```bash
.venv/bin/python -m src.train
```

Checkpoints are written to `checkpoints/editorial_model/`.

### Resume from checkpoint

```bash
.venv/bin/python -m src.train --checkpoint checkpoints/editorial_model/checkpoint_epoch_2.pt
```

### Loading a saved checkpoint for inference

```python
from src.models.peft_adapter import load_fine_tuned_model
from src.models.config import EditorialModelConfig

config = EditorialModelConfig()
model = load_fine_tuned_model("checkpoints/editorial_model/best_model.pt", config)
model.eval()
```

---

## File References

| File | Purpose |
|---|---|
| [`src/train.py`](../src/train.py) | Training entrypoint |
| [`src/models/config.py`](../src/models/config.py) | `EditorialModelConfig` and `TrainingConfig` dataclasses |
| [`src/models/multi_head_transformer.py`](../src/models/multi_head_transformer.py) | Model definition |
| [`src/models/peft_adapter.py`](../src/models/peft_adapter.py) | LoRA injection, checkpoint save/load |
| [`src/data/dataset.py`](../src/data/dataset.py) | Dataset and synthetic data generation |
