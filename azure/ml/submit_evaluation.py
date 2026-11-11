"""
Azure Machine Learning SDK v2 Model Evaluation Job Submission.

Submits an evaluation job to benchmark editorial intelligence models on an Azure ML compute target.
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

from .submit_training import get_ml_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def submit_evaluation_job(
    checkpoint_asset: str = "azureml:video-editorial-transformer:1",
    base_model: str = "distilbert-base-uncased",
    compute_cluster: str = "cpu-cluster-standard",
    environment_name: str = "video-editorial-env:1",
    experiment_name: str = "video-editorial-evaluation",
    dry_run: bool = True,
):
    """Build and submit an Azure ML Evaluation Command Job."""
    logger.info("Initializing Azure ML Client for evaluation...")

    try:
        ml_client = get_ml_client()
    except Exception as e:
        logger.warning(f"Could not connect to live Azure ML workspace ({e}).")
        if not dry_run:
            raise
        ml_client = None

    project_root = Path(__file__).resolve().parent.parent.parent

    job = command(
        code=str(project_root),
        command=(
            "python -m src.evaluate "
            "--checkpoint-dir ${{inputs.checkpoint_dir}} "
            "--base-model ${{inputs.base_model}} "
            "--output-dir ${{outputs.eval_output}} "
            "--device cpu"
        ),
        inputs={
            "checkpoint_dir": Input(type=AssetTypes.URI_FOLDER, path=checkpoint_asset),
            "base_model": base_model,
        },
        outputs={
            "eval_output": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=f"azureml:{environment_name}" if not environment_name.startswith("azureml:") else environment_name,
        compute=compute_cluster,
        experiment_name=experiment_name,
        display_name="editorial-model-evaluation",
        description="Evaluates video editorial models across ROC-AUC, F1, RMSE, and Pearson correlation.",
    )

    print("\n" + "=" * 65)
    print("📊 AZURE MACHINE LEARNING SDK v2 EVALUATION SPECIFICATION")
    print("=" * 65)
    print(f"Experiment Name:   {job.experiment_name}")
    print(f"Checkpoint Asset:  {checkpoint_asset}")
    print(f"Compute Target:    {job.compute}")
    print(f"Environment:       {job.environment}")
    print("=" * 65)

    if dry_run or ml_client is None:
        print("\n[NOTICE] Dry-run mode enabled. Cloud job was NOT submitted.")
        print("To submit live, configure Azure credentials and pass --submit.\n")
        return job

    returned_job = ml_client.jobs.create_or_update(job)
    logger.info(f"Evaluation job submitted! Studio URL: {returned_job.studio_url}")
    return returned_job


def main():
    parser = argparse.ArgumentParser(description="Submit Model Evaluation Job to Azure ML")
    parser.add_argument("--checkpoint-asset", type=str, default="azureml:video-editorial-transformer:1", help="Azure ML model asset")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--compute", type=str, default="cpu-cluster-standard", help="Compute cluster")
    parser.add_argument("--environment", type=str, default="video-editorial-env:1", help="Environment")
    parser.add_argument("--submit", action="store_true", help="Submit live to Azure ML")
    args = parser.parse_args()

    submit_evaluation_job(
        checkpoint_asset=args.checkpoint_asset,
        base_model=args.base_model,
        compute_cluster=args.compute,
        environment_name=args.environment,
        dry_run=not args.submit,
    )


if __name__ == "__main__":
    main()

