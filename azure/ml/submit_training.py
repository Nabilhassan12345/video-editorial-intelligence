"""
Azure Machine Learning SDK v2 Fine-Tuning Job Submission.

Configures and submits a PEFT LoRA fine-tuning job to an Azure ML compute target.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ml_client(config_file: Optional[str] = None) -> MLClient:
    """Connect to Azure ML workspace using DefaultAzureCredential and environment or config file."""
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

    if not (subscription_id and resource_group and workspace_name):
        cfg_path = Path(config_file or "azure/ml/config.json")
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            subscription_id = cfg.get("subscription_id")
            resource_group = cfg.get("resource_group")
            workspace_name = cfg.get("workspace_name")

    if not (subscription_id and resource_group and workspace_name):
        raise ValueError(
            "Azure ML credentials missing. Please set AZURE_SUBSCRIPTION_ID, "
            "AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE_NAME, or provide azure/ml/config.json."
        )

    credential = DefaultAzureCredential()
    return MLClient(credential, subscription_id, resource_group, workspace_name)


def submit_training_job(
    base_model: str = "distilbert-base-uncased",
    compute_cluster: str = "gpu-cluster-v100",
    environment_name: str = "video-editorial-env:1",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-4,
    dataset_path: Optional[str] = None,
    experiment_name: str = "video-editorial-fine-tuning",
    dry_run: bool = True,
):
    """Build and submit an Azure ML Command Job."""
    logger.info("Initializing Azure ML Client...")

    try:
        ml_client = get_ml_client()
        logger.info(f"Connected to Workspace: {ml_client.workspace_name}")
    except Exception as e:
        logger.warning(f"Could not connect to live Azure ML workspace ({e}).")
        if not dry_run:
            raise
        ml_client = None

    # Construct SDK v2 Command Job
    project_root = Path(__file__).resolve().parent.parent.parent

    job = command(
        code=str(project_root),
        command=(
            "python -m src.train "
            "--base-model ${{inputs.base_model}} "
            "--output-dir ${{outputs.model_output}} "
            "--epochs ${{inputs.epochs}} "
            "--batch-size ${{inputs.batch_size}} "
            "--lr ${{inputs.learning_rate}} "
            "--device cuda"
            + (" --dataset-path ${{inputs.dataset}}" if dataset_path else "")
        ),
        inputs={
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            **({"dataset": Input(type=AssetTypes.URI_FILE, path=dataset_path)} if dataset_path else {}),
        },
        outputs={
            "model_output": Output(type=AssetTypes.CUSTOM_MODEL),
        },
        environment=f"azureml:{environment_name}" if not environment_name.startswith("azureml:") else environment_name,
        compute=compute_cluster,
        experiment_name=experiment_name,
        display_name=f"peft-lora-{base_model}-fine-tuning",
        description="Fine-tunes multi-head video editorial transformer on Azure ML compute cluster.",
    )

    print("\n" + "=" * 65)
    print("🚀 AZURE MACHINE LEARNING SDK v2 JOB SPECIFICATION")
    print("=" * 65)
    print(f"Experiment Name:  {job.experiment_name}")
    print(f"Display Name:     {job.display_name}")
    print(f"Compute Cluster:  {job.compute}")
    print(f"Environment:      {job.environment}")
    print(f"Code Root:        {job.code}")
    print(f"Base Model:       {base_model}")
    print(f"Dataset:          {dataset_path or 'bundled synthetic (70 samples)'}")
    print(f"Epochs:           {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 65)

    if dry_run or ml_client is None:
        print("\n[NOTICE] Dry-run mode enabled. Cloud job was NOT submitted to avoid cloud charges.")
        print("To submit live, ensure Azure credentials are configured and pass --submit.\n")
        return job

    logger.info("Submitting job to Azure Machine Learning...")
    returned_job = ml_client.jobs.create_or_update(job)
    logger.info(f"Job submitted successfully! Studio URL: {returned_job.studio_url}")
    return returned_job


def main():
    parser = argparse.ArgumentParser(description="Submit Fine-Tuning Job to Azure Machine Learning")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--compute", type=str, default="gpu-cluster-v100", help="Azure ML compute cluster name")
    parser.add_argument("--environment", type=str, default="video-editorial-env:1", help="Azure ML environment")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Local JSONL dataset to upload and train on (e.g. data/real/ttcc_train.jsonl "
                             "produced by `python -m src.data.real.collect --source ttcc`). Omit for the bundled synthetic set.")
    parser.add_argument("--submit", action="store_true", help="Submit job to Azure ML (defaults to dry-run preview)")
    args = parser.parse_args()

    submit_training_job(
        base_model=args.base_model,
        compute_cluster=args.compute,
        environment_name=args.environment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_path=args.dataset_path,
        dry_run=not args.submit,
    )


if __name__ == "__main__":
    main()
