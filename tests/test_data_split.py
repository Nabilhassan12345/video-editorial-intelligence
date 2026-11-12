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


def test_split_merges_groups_transitively_through_shared_text():
    data = []
    for i in range(30):
        data.append({"id": f"a{i}", "group_id": f"g{i}", "text": f"unique {i}", "hook_label": float(i % 2),
                     "drop_off_risk": 0.0})
    # g_x shares a group with one item and a text ("brand") with another group: all must co-locate.
    data += [
        {"id": "x1", "group_id": "gx1", "text": "brand", "hook_label": 1.0, "drop_off_risk": 0.0},
        {"id": "x2", "group_id": "gx2", "text": "other text", "hook_label": 0.0, "drop_off_risk": 0.0},
        {"id": "x3", "group_id": "gx2", "text": "brand", "hook_label": 0.0, "drop_off_risk": 0.0},
    ]
    for seed in range(5):
        splits = split_dataset(data, seed=seed)
        where = {s["id"]: i for i, split in enumerate(splits) for s in split}
        assert where["x1"] == where["x2"] == where["x3"]
        assert_no_leakage(*splits)
