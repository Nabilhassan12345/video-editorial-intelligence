"""
Tests for the group-aware train/val/test split (data-leakage safeguard).
"""

import pytest

from src.data.dataset import generate_curated_editorial_dataset
from src.train import assert_no_leakage, split_dataset


def test_split_keeps_templates_disjoint():
    data = generate_curated_editorial_dataset(num_samples=70, seed=42)
    train, val, test = split_dataset(data, seed=42)

    assert len(train) + len(val) + len(test) == len(data)
    assert train and val and test
    train_texts = {s["text"] for s in train}
    assert not train_texts & {s["text"] for s in val}
    assert not train_texts & {s["text"] for s in test}


def test_split_is_deterministic():
    data = generate_curated_editorial_dataset(num_samples=70, seed=42)
    a = [s["id"] for s in split_dataset(data, seed=7)[2]]
    b = [s["id"] for s in split_dataset(data, seed=7)[2]]
    assert a == b


def test_assert_no_leakage_detects_overlap():
    item = {"id": "x", "text": "same text", "group_id": "g1"}
    with pytest.raises(ValueError):
        assert_no_leakage([item], [dict(item, id="y")])
