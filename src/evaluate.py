"""
Evaluation and benchmarking suite for Video Editorial Intelligence models.

Computes:
- Hook detection: F1, Precision, Recall, Accuracy, ROC-AUC
- Drop-off risk: ROC-AUC, PR-AUC, Brier Score, Cross-Entropy
- Quality scoring: RMSE, MAE, Pearson r correlation, R-squared
- Efficiency: Latency (ms/sample), throughput (samples/sec)
- Error analysis: Residuals, top false positives/negatives
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import VideoEditorialDataset, generate_curated_editorial_dataset
from src.models.config import EditorialModelConfig
from src.models.multi_head_transformer import VideoEditorialTransformer
from src.models.peft_adapter import load_fine_tuned_model
from src.train import assert_no_leakage, split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_auc(fn, targets, preds) -> Optional[float]:
    """Ranking metrics are undefined for a single-class split; return None instead of NaN/0.5."""
    if len(np.unique(targets)) < 2:
        return None
    return float(fn(targets, preds))


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if value is None else round(value, ndigits)


def compute_editorial_metrics(
    hook_preds: np.ndarray,
    hook_targets: np.ndarray,
    drop_preds: np.ndarray,
    drop_targets: np.ndarray,
    quality_preds: np.ndarray,
    quality_targets: np.ndarray,
    latency_ms_per_sample: float = 0.0,
) -> Dict[str, Any]:
    """Calculate comprehensive task-specific metrics with scikit-learn and scipy."""
    # 1. Hook Detection Metrics
    hook_binary_preds = (hook_preds >= 0.5).astype(int)
    hook_binary_targets = (hook_targets >= 0.5).astype(int)

    hook_acc = float(metrics.accuracy_score(hook_binary_targets, hook_binary_preds))
    hook_f1 = float(metrics.f1_score(hook_binary_targets, hook_binary_preds, zero_division=0))
    hook_precision = float(metrics.precision_score(hook_binary_targets, hook_binary_preds, zero_division=0))
    hook_recall = float(metrics.recall_score(hook_binary_targets, hook_binary_preds, zero_division=0))

    # ROC-AUC is undefined when the test split contains a single class; report None, not 0.5.
    hook_roc_auc = _safe_auc(metrics.roc_auc_score, hook_binary_targets, hook_preds)

    # 2. Retention Drop-off Churn Metrics
    drop_binary_targets = (drop_targets >= 0.5).astype(int)
    drop_roc_auc = _safe_auc(metrics.roc_auc_score, drop_binary_targets, drop_preds)
    drop_pr_auc = _safe_auc(metrics.average_precision_score, drop_binary_targets, drop_preds)

    drop_brier = float(metrics.brier_score_loss(drop_binary_targets, drop_preds))

    # 3. Holistic Quality Score Regression Metrics
    quality_mae = float(metrics.mean_absolute_error(quality_targets, quality_preds))
    quality_rmse = float(np.sqrt(metrics.mean_squared_error(quality_targets, quality_preds)))
    quality_r2 = float(metrics.r2_score(quality_targets, quality_preds))

    if len(quality_preds) > 1 and np.std(quality_preds) > 1e-6 and np.std(quality_targets) > 1e-6:
        pearson_r, p_val = stats.pearsonr(quality_preds, quality_targets)
        pearson_r = float(pearson_r)
        p_val = float(p_val)
    else:
        pearson_r, p_val = 0.0, 1.0

    return {
        "hook_detection": {
            "accuracy": round(hook_acc, 4),
            "f1_score": round(hook_f1, 4),
            "precision": round(hook_precision, 4),
            "recall": round(hook_recall, 4),
            "roc_auc": _round(hook_roc_auc),
        },
        "drop_off_prediction": {
            "roc_auc": _round(drop_roc_auc),
            "pr_auc": _round(drop_pr_auc),
            "brier_score": round(drop_brier, 4),
        },
        "quality_scoring": {
            "rmse": round(quality_rmse, 4),
            "mae": round(quality_mae, 4),
            "r2_score": round(quality_r2, 4),
            "pearson_correlation": round(pearson_r, 4),
            "p_value": round(p_val, 6),
        },
        "efficiency": {
            "latency_ms_per_sample": round(latency_ms_per_sample, 2),
            "throughput_samples_per_sec": round(1000.0 / max(0.1, latency_ms_per_sample), 1) if latency_ms_per_sample > 0 else 0.0,
        },
    }


def perform_error_analysis(
    samples: List[Dict],
    hook_preds: np.ndarray,
    drop_preds: np.ndarray,
    quality_preds: np.ndarray,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Identify largest residuals, false positives, and false negatives."""
    drop_targets = np.array([float(s.get("drop_off_risk", 0.0)) for s in samples])
    quality_targets = np.array([float(s.get("quality_score", 5.0)) for s in samples])

    # Quality residuals
    q_residuals = np.abs(quality_preds - quality_targets)
    top_q_err_indices = np.argsort(q_residuals)[::-1][:top_k]

    top_quality_errors = []
    for idx in top_q_err_indices:
        top_quality_errors.append({
            "sample_id": samples[idx].get("id", f"sample_{idx}"),
            "text": samples[idx].get("text", "")[:80] + "...",
            "target_quality": round(float(quality_targets[idx]), 2),
            "predicted_quality": round(float(quality_preds[idx]), 2),
            "absolute_error": round(float(q_residuals[idx]), 2),
        })

    # Drop-off false alarms (high pred, low target)
    drop_false_alarms = []
    drop_diff = drop_preds - drop_targets
    top_fa_indices = np.argsort(drop_diff)[::-1][:top_k]
    for idx in top_fa_indices:
        if drop_diff[idx] > 0.2:
            drop_false_alarms.append({
                "sample_id": samples[idx].get("id", f"sample_{idx}"),
                "text": samples[idx].get("text", "")[:80] + "...",
                "predicted_risk": round(float(drop_preds[idx]), 3),
                "actual_risk": round(float(drop_targets[idx]), 3),
            })

    return {
        "largest_quality_residuals": top_quality_errors,
        "notable_drop_off_false_alarms": drop_false_alarms,
    }


def evaluate_dataset(
    model: VideoEditorialTransformer,
    dataset: List[Dict],
    tokenizer: Optional[Any],
    device: str = "cpu",
    batch_size: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run full evaluation over test dataset and return metric dict and error analysis."""
    eval_ds = VideoEditorialDataset(dataset, tokenizer=tokenizer)
    dataloader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_hook_preds, all_hook_targets = [], []
    all_drop_preds, all_drop_targets = [], []
    all_quality_preds, all_quality_targets = [], []

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feat_vec = batch["feature_vector"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_vector=feat_vec,
            )

            all_hook_preds.extend(out["hook_prob"].cpu().numpy().tolist())
            all_hook_targets.extend(batch["hook_label"].numpy().tolist())

            all_drop_preds.extend(out["drop_off_risk"].cpu().numpy().tolist())
            all_drop_targets.extend(batch["drop_off_risk"].numpy().tolist())

            all_quality_preds.extend(out["quality_score"].cpu().numpy().tolist())
            all_quality_targets.extend(batch["quality_score"].numpy().tolist())

    total_time = time.time() - start_time
    n_samples = max(1, len(dataset))
    latency_ms = (total_time / n_samples) * 1000.0

    hp = np.array(all_hook_preds)
    ht = np.array(all_hook_targets)
    dp = np.array(all_drop_preds)
    dt = np.array(all_drop_targets)
    qp = np.array(all_quality_preds)
    qt = np.array(all_quality_targets)

    metrics_result = compute_editorial_metrics(hp, ht, dp, dt, qp, qt, latency_ms_per_sample=latency_ms)
    error_analysis = perform_error_analysis(dataset, hp, dp, qp)

    return metrics_result, error_analysis


def generate_markdown_summary(
    baseline_metrics: Optional[Dict[str, Any]],
    fine_tuned_metrics: Dict[str, Any],
    error_analysis: Dict[str, Any],
    model_name: str,
    dataset_note: str = "",
) -> str:
    """Format evaluation results into a Markdown report table."""
    md = [
        f"# Model Evaluation & Benchmark Report",
        f"\n**Evaluated Architecture:** `{model_name}`",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n> {dataset_note}" if dataset_note else "",
        "\n## 1. Quantitative Benchmark Comparison\n",
        "| Task / Metric | Zero-Shot Baseline | Fine-Tuned (PEFT LoRA) | Delta |",
        "| :--- | :---: | :---: | :---: |",
    ]

    ft_hook = fine_tuned_metrics["hook_detection"]
    ft_drop = fine_tuned_metrics["drop_off_prediction"]
    ft_qual = fine_tuned_metrics["quality_scoring"]

    base_hook = baseline_metrics.get("hook_detection", {}) if baseline_metrics else {}
    base_drop = baseline_metrics.get("drop_off_prediction", {}) if baseline_metrics else {}
    base_qual = baseline_metrics.get("quality_scoring", {}) if baseline_metrics else {}

    def fmt_row(name, base_val, ft_val, lower_is_better=False):
        if ft_val is None:
            b_str = "N/A" if base_val is None else f"{base_val:.4f}"
            return f"| **{name}** | {b_str} | undefined (single-class test split) | -- |"
        if base_val is not None and isinstance(base_val, (int, float)):
            delta = ft_val - base_val
            symbol = "🟢" if (delta > 0 and not lower_is_better) or (delta < 0 and lower_is_better) else "⚪"
            delta_str = f"{symbol} {delta:+.4f}"
            b_str = f"{base_val:.4f}"
        else:
            b_str = "N/A"
            delta_str = "--"
        return f"| **{name}** | {b_str} | **{ft_val:.4f}** | {delta_str} |"

    md.append(fmt_row("Hook F1-Score", base_hook.get("f1_score"), ft_hook["f1_score"]))
    md.append(fmt_row("Hook ROC-AUC", base_hook.get("roc_auc"), ft_hook["roc_auc"]))
    md.append(fmt_row("Drop-off ROC-AUC", base_drop.get("roc_auc"), ft_drop["roc_auc"]))
    md.append(fmt_row("Drop-off PR-AUC", base_drop.get("pr_auc"), ft_drop["pr_auc"]))
    md.append(fmt_row("Drop-off Brier Score", base_drop.get("brier_score"), ft_drop["brier_score"], lower_is_better=True))
    md.append(fmt_row("Quality RMSE", base_qual.get("rmse"), ft_qual["rmse"], lower_is_better=True))
    md.append(fmt_row("Quality Pearson r", base_qual.get("pearson_correlation"), ft_qual["pearson_correlation"]))

    eff = fine_tuned_metrics.get("efficiency", {})
    md.extend([
        f"\n## 2. Inference Profiling",
        f"- **Per-Segment Latency:** `{eff.get('latency_ms_per_sample', 0):.2f} ms`",
        f"- **Throughput:** `{eff.get('throughput_samples_per_sec', 0):.1f} segments/sec`",
        f"\n## 3. Error Analysis & Residuals",
        "### Largest Quality Residuals:",
    ])

    for err in error_analysis.get("largest_quality_residuals", []):
        md.append(f"- Sample `{err['sample_id']}`: Target `{err['target_quality']}` vs Predicted `{err['predicted_quality']}` (Error: `{err['absolute_error']}`)")
        md.append(f"  *Text:* \"{err['text']}\"")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Editorial Intelligence Models")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model for baseline comparison")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs/evaluation", help="Directory to save evaluation reports")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on ('cpu', 'mps', 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible test split")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    if args.dataset_path and Path(args.dataset_path).exists():
        if str(args.dataset_path).endswith(".jsonl"):
            with open(args.dataset_path, "r") as f:
                full_data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(args.dataset_path, "r") as f:
                full_data = json.load(f)
    else:
        full_data = generate_curated_editorial_dataset(num_samples=70, seed=args.seed)
    is_synthetic = not (args.dataset_path and Path(args.dataset_path).exists())

    # Re-create the exact group-aware split used by src.train so the test set never overlaps training.
    train_split, val_split, test_split = split_dataset(full_data, seed=args.seed)
    assert_no_leakage(train_split, val_split, test_split)
    test_groups = sorted({str(s.get("group_id") or s.get("text")) for s in test_split})
    logger.info(f"Evaluating on held-out test split: {len(test_split)} samples from {len(test_groups)} group(s).")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception:
        tokenizer = None

    # 2. Evaluate Baseline Model (Zero-Shot / Random init heads)
    logger.info("Evaluating baseline unadapted model...")
    base_model = VideoEditorialTransformer(EditorialModelConfig(base_model_name=args.base_model, use_peft=False))
    base_metrics, _ = evaluate_dataset(base_model, test_split, tokenizer=tokenizer, device=args.device)

    # 3. Evaluate Fine-Tuned Model
    if args.checkpoint_dir and Path(args.checkpoint_dir).exists():
        logger.info(f"Loading and evaluating fine-tuned model from {args.checkpoint_dir}...")
        ft_model = load_fine_tuned_model(args.checkpoint_dir, device=args.device)
    else:
        logger.info("No checkpoint provided; evaluating initialized PEFT model.")
        ft_model = VideoEditorialTransformer(EditorialModelConfig(base_model_name=args.base_model, use_peft=True))

    ft_metrics, error_analysis = evaluate_dataset(ft_model, test_split, tokenizer=tokenizer, device=args.device)

    # 4. Generate Reports
    dataset_note = (
        f"Synthetic template benchmark: {len(full_data)} samples generated from "
        f"{len({s.get('group_id') for s in full_data})} templates; test split = {len(test_split)} samples "
        f"from {len(test_groups)} unseen template(s). Not real viewer data; numbers are a pipeline sanity check only."
        if is_synthetic
        else f"Dataset: {args.dataset_path}; test split = {len(test_split)} samples from {len(test_groups)} group(s)."
    )
    md_report = generate_markdown_summary(
        base_metrics, ft_metrics, error_analysis, model_name=args.base_model, dataset_note=dataset_note
    )
    json_report = {
        "model_name": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": "synthetic_template_benchmark" if is_synthetic else args.dataset_path,
        "dataset_note": dataset_note,
        "split_sizes": {"train": len(train_split), "val": len(val_split), "test": len(test_split)},
        "test_groups": test_groups,
        "test_sample_count": len(test_split),
        "baseline_metrics": base_metrics,
        "fine_tuned_metrics": ft_metrics,
        "error_analysis": error_analysis,
    }

    with open(out_path / "evaluation_results.json", "w") as f:
        json.dump(json_report, f, indent=2)

    with open(out_path / "evaluation_summary.md", "w") as f:
        f.write(md_report)

    logger.info(f"Evaluation results successfully saved to {out_path / 'evaluation_results.json'}")
    logger.info(f"Summary markdown saved to {out_path / 'evaluation_summary.md'}")
    print("\n" + md_report + "\n")


if __name__ == "__main__":
    main()

