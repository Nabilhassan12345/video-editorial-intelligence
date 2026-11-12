"""
Tests for Azure Machine Learning configuration templates and job specifications.
"""

import json
from pathlib import Path
import pytest
import yaml

from azure.ml.submit_training import submit_training_job
from azure.ml.submit_evaluation import submit_evaluation_job
from azure.ml.register_model import register_model_asset
from azure.ml.deploy_model import configure_deployment


def test_azure_ml_config_templates():
    cfg_example = Path("azure/ml/config.json.example")
    assert cfg_example.exists()
    with open(cfg_example, "r") as f:
        data = json.load(f)
    assert "subscription_id" in data
    assert "resource_group" in data
    assert "workspace_name" in data

    compute_yaml = Path("azure/ml/jobs/compute_config.yaml")
    assert compute_yaml.exists()
    with open(compute_yaml, "r") as f:
        comp_data = yaml.safe_load(f)
    assert "cpu_cluster" in comp_data
    assert "gpu_cluster" in comp_data
    assert comp_data["cpu_cluster"]["min_instances"] == 0


def test_azure_ml_training_job_dry_run():
    job = submit_training_job(
        base_model="distilbert-base-uncased",
        compute_cluster="gpu-cluster-t4",
        epochs=1,
        dry_run=True,
    )
    assert job is not None
    assert job.experiment_name == "video-editorial-fine-tuning"
    assert "model_output" in job.outputs


def test_azure_ml_evaluation_job_dry_run():
    job = submit_evaluation_job(
        checkpoint_asset="azureml:video-editorial-transformer:1",
        dry_run=True,
    )
    assert job is not None
    assert job.experiment_name == "video-editorial-evaluation"


def test_azure_ml_model_registration_dry_run(tmp_path):
    (tmp_path / "training_metadata.json").write_text(json.dumps({"best_val_loss": 1.0}))
    asset = register_model_asset(
        checkpoint_dir=str(tmp_path),
        dry_run=True,
    )
    assert asset.name == "video-editorial-transformer"


def test_azure_ml_deployment_dry_run():
    endpoint, deployment = configure_deployment(
        endpoint_name="test-endpoint",
        dry_run=True,
    )
    assert endpoint.name == "test-endpoint"
    assert deployment.instance_count == 1



def test_azure_ml_workspace_bootstrap_dry_run():
    from azure.ml.setup_workspace import setup_workspace
    specs = setup_workspace("", "", "aml-test", dry_run=True)
    names = {c.name: c for c in specs["computes"]}
    assert set(names) == {"gpu-cluster-t4", "cpu-cluster-standard"}
    assert all(c.min_instances == 0 for c in names.values())
    assert names["gpu-cluster-t4"].size == "Standard_NC4as_T4_v3"
    assert specs["environment"].name == "video-editorial-env"


def test_azure_ml_training_job_picks_device_from_compute():
    gpu = submit_training_job(compute_cluster="gpu-cluster-t4", dataset_path="data/x.jsonl", dry_run=True)
    cpu = submit_training_job(compute_cluster="cpu-cluster-standard", dry_run=True)
    assert "--device cuda" in gpu.command and "--dataset-path" in gpu.command
    assert "--device cpu" in cpu.command and "--dataset-path" not in cpu.command
