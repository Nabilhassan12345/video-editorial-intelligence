"""
PyTorch Dataset and data loader utilities for Video Editorial Intelligence.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VideoEditorialDataset(Dataset):
    """
    PyTorch Dataset for multi-task video editorial intelligence.
    
    Predicts:
    1. Hook suitability probability (binary classification or ranking)
    2. Retention drop-off risk (binary/continuous probability)
    3. Structural quality score (continuous 1.0 - 10.0 regression)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        include_labels: bool = True,
    ):
        """
        Args:
            samples: List of dictionaries containing 'text', 'feature_vector', and optionally targets:
                     'hook_label', 'drop_off_risk', 'quality_score'.
            tokenizer: PreTrainedTokenizer instance. If None, tokenization is deferred.
            max_length: Maximum sequence length for truncation/padding.
            include_labels: Whether ground truth labels are present.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        text = str(item.get("text", ""))

        # Feature vector (8 dimensions)
        feat_vec = item.get("feature_vector")
        if feat_vec is None:
            feat_vec = np.zeros(8, dtype=np.float32)
        elif not isinstance(feat_vec, np.ndarray):
            feat_vec = np.array(feat_vec, dtype=np.float32)

        data = {
            "text": text,
            "feature_vector": torch.tensor(feat_vec, dtype=torch.float32),
        }

        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            data["input_ids"] = tokenized["input_ids"].squeeze(0)
            data["attention_mask"] = tokenized["attention_mask"].squeeze(0)

        # Labels
        if self.include_labels:
            data["hook_label"] = torch.tensor(float(item.get("hook_label", 0.0)), dtype=torch.float32)
            data["drop_off_risk"] = torch.tensor(float(item.get("drop_off_risk", 0.0)), dtype=torch.float32)
            data["quality_score"] = torch.tensor(float(item.get("quality_score", 5.0)), dtype=torch.float32)

        return data


def generate_curated_editorial_dataset(num_samples: int = 60, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a curated, reproducible synthetic dataset matching real video editorial dynamics.
    
    Produces realistic:
    - High-performing intro hooks (curiosity, urgency, high retention)
    - Monotonous filler drop-off segments (silence, low lexical density, high churn risk)
    - High-quality balanced body segments
    - Low-quality unstructured rambling segments
    """
    np.random.seed(seed)
    dataset = []

    templates = [
        # (text, hook_prob, drop_off_prob, quality_score, is_hook, base_wpm, base_silence)
        (
            "In the next three minutes, I will show you the exact strategy that changed my entire workflow forever.",
            1.0, 0.08, 9.2, 1, 165.0, 0.02
        ),
        (
            "Stop scrolling right now if you want to know the biggest mistake 90% of editors make every single day.",
            1.0, 0.12, 8.8, 1, 172.0, 0.03
        ),
        (
            "Why is nobody talking about this hidden trick? It literally saves four hours every week.",
            1.0, 0.15, 8.5, 1, 155.0, 0.04
        ),
        (
            "Um, so basically in this video we are kind of going to look at... well, let me adjust my camera first.",
            0.0, 0.85, 2.8, 0, 95.0, 0.38
        ),
        (
            "And yeah, like I said before, you know, it's just something that sort of happens sometimes.",
            0.0, 0.78, 3.2, 0, 110.0, 0.28
        ),
        (
            "Let me pause here for a second and search for that link... just give me a moment guys.",
            0.0, 0.92, 1.9, 0, 70.0, 0.45
        ),
        (
            "First, let's break down the three fundamental pillars of high-retention video architecture.",
            0.2, 0.22, 8.4, 0, 148.0, 0.06
        ),
        (
            "Notice how the visual transition synchronizes with the dynamic frequency shift in the narration.",
            0.1, 0.18, 8.9, 0, 152.0, 0.05
        ),
        (
            "Now let's compare the before and after retention metrics across a 50,000 viewer sample.",
            0.2, 0.20, 8.7, 0, 158.0, 0.05
        ),
        (
            "If you found this useful, hit the subscribe button and download the free workflow checklist below.",
            0.0, 0.42, 7.5, 0, 145.0, 0.08
        ),
    ]

    for i in range(num_samples):
        base = templates[i % len(templates)]
        # Add slight natural perturbation to simulate real-world noise
        text = base[0]
        hook_label = float(base[4])
        drop_off = float(np.clip(base[2] + np.random.normal(0, 0.04), 0.0, 1.0))
        quality = float(np.clip(base[3] + np.random.normal(0, 0.2), 1.0, 10.0))
        wpm = float(np.clip(base[5] + np.random.normal(0, 10.0), 60.0, 220.0))
        silence = float(np.clip(base[6] + np.random.normal(0, 0.02), 0.0, 0.6))
        rms = float(np.clip(0.12 - (silence * 0.15) + np.random.normal(0, 0.01), 0.01, 0.25))

        words = text.split()
        dur = max(1.0, len(words) / (wpm / 60.0))
        ttr = len(set(words)) / max(1, len(words))

        vec = np.array([
            rms,
            silence,
            wpm / 200.0,
            0.65 if hook_label > 0 else 0.45,
            ttr,
            hook_label * 0.8,
            0.35 if drop_off > 0.6 else 0.05,
            len(words) / dur / 5.0,
        ], dtype=np.float32)

        dataset.append({
            "id": f"sample_{i:04d}",
            "group_id": f"template_{i % len(templates):02d}",
            "text": text,
            "feature_vector": vec,
            "hook_label": hook_label,
            "drop_off_risk": drop_off,
            "quality_score": quality,
            "duration": round(dur, 2),
        })

    return dataset

