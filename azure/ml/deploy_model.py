"""
Azure Machine Learning Managed Online Endpoint Deployment.

Defines ManagedOnlineEndpoint, ManagedOnlineDeployment, and scoring script for scalable real-time inference.
"""

import argparse
import logging
import os
from pathlib import Path

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.ai.ml.constants import AssetTypes

from .submit_training import get_ml_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_scoring_script(output_path: Path):
    """Generate Azure ML inference scoring script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_content = '''"""
Azure ML Online Endpoint Scoring Script.
"""
import json
import os
import torch
from src.pipeline.video_pipeline import VideoEditorialPipeline

pipeline = None

def init():
    global pipeline
    model_dir = os.getenv("AZUREML_MODEL_DIR", "checkpoints/editorial_model")
    pipeline = VideoEditorialPipeline(checkpoint_dir=model_dir)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        transcript = data.get("transcript", "")
        result = pipeline.run(transcript_input=transcript)
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(score_content)


def configure_deployment(
    endpoint_name: str = "video-editorial-endpoint",
    deployment_name: str = "blue",
    model_asset_name: str = "video-editorial-transformer:1",
    instance_type: str = "Standard_DS3_v2",
    instance_count: int = 1,
    dry_run: bool = True,
):
    """Configure Managed Online Endpoint and Deployment."""
    scoring_script_path = Path("azure/ml/scoring/score.py")
    generate_scoring_script(scoring_script_path)

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Real-time Video Editorial Intelligence inference endpoint",
        auth_mode="key",
    )

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_asset_name}",
        code_configuration=CodeConfiguration(
            code="../../",
            scoring_script="azure/ml/scoring/score.py",
        ),
        environment="azureml:video-editorial-env:1",
        instance_type=instance_type,
        instance_count=instance_count,
    )

    print("\n" + "=" * 65)
    print("🌐 AZURE ML MANAGED ONLINE ENDPOINT SPECIFICATION")
    print("=" * 65)
    print(f"Endpoint Name:     {endpoint.name}")
    print(f"Deployment Name:   {deployment.name}")
    print(f"Model Asset:       {deployment.model}")
    print(f"Instance Type:     {deployment.instance_type} (Count: {deployment.instance_count})")
    print(f"Scoring Script:    {scoring_script_path}")
    print("=" * 65)

    if dry_run:
        print("\n[NOTICE] Dry-run preview mode. Cloud endpoint was NOT deployed to prevent cloud billing.")
        print("To deploy live, run with --deploy.\n")
        return endpoint, deployment

    ml_client = get_ml_client()
    logger.info(f"Creating endpoint '{endpoint_name}'...")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    logger.info(f"Deploying model to '{deployment_name}'...")
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(f"Deployment complete! Traffic routed 100% to '{deployment_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Deploy Video Editorial Model to Azure ML Online Endpoint")
    parser.add_argument("--endpoint", type=str, default="video-editorial-endpoint", help="Endpoint name")
    parser.add_argument("--instance-type", type=str, default="Standard_DS3_v2", help="VM instance type")
    parser.add_argument("--deploy", action="store_true", help="Perform live cloud deployment")
    args = parser.parse_args()

    configure_deployment(
        endpoint_name=args.endpoint,
        instance_type=args.instance_type,
        dry_run=not args.deploy,
    )


if __name__ == "__main__":
    main()

