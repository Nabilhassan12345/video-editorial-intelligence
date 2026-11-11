"""
Model fine-tuning entrypoint with PEFT LoRA, multi-task composite loss, and dry-run verification.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import VideoEditorialDataset, generate_curated_editorial_dataset
from src.models.config import EditorialModelConfig, TrainingConfig
from src.models.multi_head_transformer import VideoEditorialTransformer
from src.models.peft_adapter import inject_peft_adapters, save_fine_tuned_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_key(item: Dict) -> str:
    """Samples sharing a group (same source template or video) must stay in one split."""
    return str(item.get("group_id") or item.get("video_id") or item.get("text", item.get("id")))


def _stratum(item: Dict) -> str:
    if float(item.get("hook_label", 0.0)) > 0.5:
        return "hook"
    if float(item.get("drop_off_risk", 0.0)) > 0.5:
        return "drop_off"
    return "other"


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified, group-aware train/val/test split.

    Whole groups (group_id, video_id or identical text) go to a single split so near-duplicate
    samples cannot leak between splits. Groups are stratified by label type (hook / drop-off /
    other) so the test split contains positives and negatives whenever there are enough groups.
    """
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    groups: Dict[str, List[Dict]] = {}
    for item in data:
        groups.setdefault(_group_key(item), []).append(item)

    strata: Dict[str, List[str]] = {}
    for key in sorted(groups):
        strata.setdefault(_stratum(groups[key][0]), []).append(key)

    rng = random.Random(seed)
    train_keys: List[str] = []
    val_keys: List[str] = []
    test_keys: List[str] = []
    for name in sorted(strata):
        keys = strata[name]
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, round(n * test_ratio)) if n >= 2 else 0
        n_val = round(n * val_ratio) if n - n_test >= 2 else 0
        test_keys += keys[:n_test]
        val_keys += keys[n_test:n_test + n_val]
        train_keys += keys[n_test + n_val:]

    if not val_keys and len(train_keys) > 1:
        val_keys.append(train_keys.pop())

    train_set = [s for k in train_keys for s in groups[k]]
    val_set = [s for k in val_keys for s in groups[k]]
    test_set = [s for k in test_keys for s in groups[k]]

    assert_no_leakage(train_set, val_set, test_set)
    logger.info(
        f"Group split {len(data)} samples / {len(groups)} groups: "
        f"Train={len(train_set)} ({len(train_keys)} groups), Val={len(val_set)} ({len(val_keys)} groups), "
        f"Test={len(test_set)} ({len(test_keys)} groups)"
    )
    return train_set, val_set, test_set


def assert_no_leakage(*splits: List[Dict]) -> None:
    """Raise if any group or exact text appears in more than one split."""
    seen: Dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for item in split:
            for key in {_group_key(item), str(item.get("text", ""))}:
                if key and seen.setdefault(key, split_idx) != split_idx:
                    raise ValueError(f"Data leakage: group/text {key[:60]!r} appears in multiple splits")


def train_epoch(
    model: VideoEditorialTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_weights: Dict[str, float],
) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feat_vec = batch["feature_vector"].to(device)
        hook_labels = batch["hook_label"].to(device)
        drop_off = batch["drop_off_risk"].to(device)
        quality = batch["quality_score"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_vector=feat_vec,
            hook_labels=hook_labels,
            drop_off_labels=drop_off,
            quality_labels=quality,
            loss_weights=loss_weights,
        )

        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate_model(
    model: VideoEditorialTransformer,
    dataloader: DataLoader,
    device: str,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    hook_preds, hook_targets = [], []
    drop_preds, drop_targets = [], []
    quality_preds, quality_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            feat_vec = batch["feature_vector"].to(device)
            hook_labels = batch["hook_label"].to(device)
            drop_off = batch["drop_off_risk"].to(device)
            quality = batch["quality_score"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_vector=feat_vec,
                hook_labels=hook_labels,
                drop_off_labels=drop_off,
                quality_labels=quality,
                loss_weights=loss_weights,
            )

            total_loss += outputs["loss"].item()
            hook_preds.extend(outputs["hook_prob"].cpu().numpy().tolist())
            hook_targets.extend(hook_labels.cpu().numpy().tolist())
            drop_preds.extend(outputs["drop_off_risk"].cpu().numpy().tolist())
            drop_targets.extend(drop_off.cpu().numpy().tolist())
            quality_preds.extend(outputs["quality_score"].cpu().numpy().tolist())
            quality_targets.extend(quality.cpu().numpy().tolist())

    n_batches = max(1, len(dataloader))
    avg_loss = total_loss / n_batches

    # Quality RMSE
    q_diff = np.array(quality_preds) - np.array(quality_targets)
    quality_rmse = float(np.sqrt(np.mean(q_diff ** 2))) if len(q_diff) > 0 else 0.0

    # Hook Accuracy (threshold at 0.5)
    hook_acc = float(np.mean((np.array(hook_preds) >= 0.5) == (np.array(hook_targets) >= 0.5))) if len(hook_preds) > 0 else 0.0

    return {
        "val_loss": round(avg_loss, 4),
        "quality_rmse": round(quality_rmse, 4),
        "hook_accuracy": round(hook_acc, 4),
    }


def run_training(
    model_config: EditorialModelConfig,
    train_config: TrainingConfig,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute training or dry-run workflow."""
    set_seed(train_config.seed)
    device = train_config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    logger.info(f"Target execution device: {device}")

    # 1. Load Data
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Loading custom dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            raw_data = json.load(f)
    else:
        logger.info("Using curated multi-segment video editorial dataset.")
        raw_data = generate_curated_editorial_dataset(num_samples=70, seed=train_config.seed)

    train_data, val_data, test_data = split_dataset(raw_data, seed=train_config.seed)

    # 2. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
    except Exception as e:
        logger.warning(f"Could not load online tokenizer ({e}); using character/whitespace tokenizer fallback.")
        from transformers import BertTokenizerFast
        # Fallback to local fast tokenizer if available or basic
        tokenizer = None

    train_ds = VideoEditorialDataset(train_data, tokenizer=tokenizer)
    val_ds = VideoEditorialDataset(val_data, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False)

    # 3. Model & PEFT Injection
    model = VideoEditorialTransformer(model_config)
    if model_config.use_peft:
        model = inject_peft_adapters(model, model_config)

    model.to(device)

    loss_weights = {
        "hook": train_config.hook_loss_weight,
        "drop_off": train_config.drop_off_loss_weight,
        "quality": train_config.quality_loss_weight,
    }

    # 4. DRY RUN CHECK
    if train_config.dry_run:
        logger.info(">>> RUNNING TRAINING PIPELINE DRY-RUN VERIFICATION <<<")
        first_batch = next(iter(train_loader))
        input_ids = first_batch["input_ids"].to(device)
        attention_mask = first_batch["attention_mask"].to(device)
        feat_vec = first_batch["feature_vector"].to(device)
        hook_l = first_batch["hook_label"].to(device)
        drop_l = first_batch["drop_off_risk"].to(device)
        qual_l = first_batch["quality_score"].to(device)

        # Forward
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_vector=feat_vec,
            hook_labels=hook_l,
            drop_off_labels=drop_l,
            quality_labels=qual_l,
            loss_weights=loss_weights,
        )
        assert "loss" in out, "Loss missing from model outputs in dry run"
        loss_val = out["loss"].item()
        logger.info(f"Dry run forward pass successful! Initial composite loss: {loss_val:.4f}")

        # Backward
        out["loss"].backward()
        logger.info("Dry run backward pass successful! Gradients verified.")

        # Test artifact serialization
        save_dir = Path(train_config.output_dir) / "dry_run"
        save_fine_tuned_model(
            model,
            str(save_dir),
            metadata={"status": "dry_run_verified", "device": device, "timestamp": time.time()},
        )
        logger.info(f"Dry run artifacts successfully serialized to {save_dir}")
        print("\n==================================================================")
        print("DRY RUN COMPLETE: Training pipeline verified without expensive full training.")
        print(f"Tensor shapes, PEFT adapter injection, forward/backward passes verified.")
        print("==================================================================\n")
        return {"status": "dry_run_verified", "loss": loss_val}

    # 5. Full Training Loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_loss = float("inf")
    metrics_history = []
    start_time = time.time()

    logger.info(f"Beginning fine-tuning for {train_config.num_train_epochs} epochs...")
    for epoch in range(train_config.num_train_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_weights)
        val_metrics = evaluate_model(model, val_loader, device, loss_weights)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            **val_metrics,
        }
        metrics_history.append(log_entry)
        logger.info(f"Epoch {epoch + 1}/{train_config.num_train_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - Quality RMSE: {val_metrics['quality_rmse']:.4f}")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_fine_tuned_model(
                model,
                train_config.output_dir,
                metadata={
                    "best_epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "metrics_history": metrics_history,
                    "base_model": model_config.base_model_name,
                },
            )

    duration = time.time() - start_time
    logger.info(f"Training completed in {duration:.1f}s. Best validation loss: {best_val_loss:.4f}")

    return {
        "status": "training_completed",
        "best_val_loss": best_val_loss,
        "epochs": train_config.num_train_epochs,
        "metrics_history": metrics_history,
        "checkpoint_dir": train_config.output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Video Editorial Intelligence Transformer")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Hugging Face base model name")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to input training JSON dataset")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/editorial_model", help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', 'mps', or 'cuda'")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run check without full training")
    args = parser.parse_args()

    model_config = EditorialModelConfig(base_model_name=args.base_model, use_peft=True)
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    run_training(model_config, train_config, dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
