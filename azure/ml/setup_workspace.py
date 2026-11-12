"""
One-command Azure ML workspace bootstrap.

Creates (or reuses) everything the training/evaluation jobs reference:
  * Azure ML workspace
  * gpu-cluster-t4   (Standard_NC4as_T4_v3, scale-to-zero, ~$0.5/h list price)
  * cpu-cluster-standard (Standard_DS3_v2, low-priority, scale-to-zero) - fallback when GPU quota is 0
  * video-editorial-env:1 environment (CUDA base image + azure/ml/environments/video-editorial-env.yml)

Dry-run by default: prints the resources it would create. Pass --create to provision.
Prerequisites: `az login` and an existing resource group (`az group create -n <rg> -l <region>`).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, Workspace
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENV_NAME = "video-editorial-env"
ENV_VERSION = "1"
ENV_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
CONDA_FILE = Path(__file__).resolve().parent / "environments" / "video-editorial-env.yml"


def build_compute_specs(gpu_size: str = "Standard_NC4as_T4_v3", gpu_low_priority: bool = False) -> List[AmlCompute]:
    return [
        AmlCompute(
            name="gpu-cluster-t4",
            size=gpu_size,
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=180,
            tier="low_priority" if gpu_low_priority else "dedicated",
        ),
        AmlCompute(
            name="cpu-cluster-standard",
            size="Standard_DS3_v2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
            tier="low_priority",
        ),
    ]


def build_environment_spec() -> Environment:
    return Environment(
        name=ENV_NAME,
        version=ENV_VERSION,
        image=ENV_BASE_IMAGE,
        conda_file=str(CONDA_FILE),
        description="PyTorch + PEFT + FFmpeg runtime for video-editorial-intelligence training/eval.",
    )


def setup_workspace(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    location: str = "eastus",
    gpu_size: str = "Standard_NC4as_T4_v3",
    gpu_low_priority: bool = False,
    dry_run: bool = True,
) -> Dict[str, object]:
    workspace = Workspace(name=workspace_name, location=location,
                          description="Video Editorial Intelligence - training, registry, endpoints")
    computes = build_compute_specs(gpu_size=gpu_size, gpu_low_priority=gpu_low_priority)
    env = build_environment_spec()

    print("\n" + "=" * 65)
    print("AZURE ML WORKSPACE BOOTSTRAP")
    print("=" * 65)
    print(f"Subscription:   {subscription_id or '<AZURE_SUBSCRIPTION_ID>'}")
    print(f"Resource group: {resource_group or '<AZURE_RESOURCE_GROUP>'}")
    print(f"Workspace:      {workspace.name} ({location})")
    for c in computes:
        print(f"Compute:        {c.name}: {c.size}, {c.min_instances}-{c.max_instances} nodes, tier={c.tier}")
    print(f"Environment:    {env.name}:{env.version} on {env.image}")
    print("=" * 65)

    specs = {"workspace": workspace, "computes": computes, "environment": env}
    if dry_run:
        print("\n[NOTICE] Dry-run: nothing was created. Pass --create to provision (clusters scale to 0 when idle).\n")
        return specs

    if not (subscription_id and resource_group):
        raise ValueError("Set AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP (or pass --subscription/--resource-group).")

    credential = DefaultAzureCredential()
    sub_client = MLClient(credential, subscription_id, resource_group)
    try:
        sub_client.workspaces.get(workspace_name)
        logger.info(f"Workspace '{workspace_name}' already exists; reusing it.")
    except Exception:
        logger.info(f"Creating workspace '{workspace_name}' (takes a few minutes)...")
        sub_client.workspaces.begin_create(workspace).result()

    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    for c in computes:
        logger.info(f"Creating/updating compute '{c.name}' ({c.size})...")
        try:
            ml_client.compute.begin_create_or_update(c).result()
        except Exception as e:  # most common: 0 GPU quota on new/free subscriptions
            logger.warning(f"Compute '{c.name}' failed: {e}\n"
                           "  -> Request quota: Azure portal > Quotas > Machine Learning > <region> > "
                           "'Standard NCASv3_T4 Family' (4 vCPUs). Meanwhile use --compute cpu-cluster-standard.")
    logger.info(f"Registering environment '{env.name}:{env.version}'...")
    ml_client.environments.create_or_update(env)
    logger.info("Workspace ready. Next: python -m azure.ml.submit_training --dataset-path <jsonl> --submit")
    return specs


def main():
    p = argparse.ArgumentParser(description="Bootstrap Azure ML workspace, compute and environment")
    p.add_argument("--subscription", default=os.getenv("AZURE_SUBSCRIPTION_ID", ""))
    p.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP", ""))
    p.add_argument("--workspace", default=os.getenv("AZURE_WORKSPACE_NAME", "aml-video-editorial"))
    p.add_argument("--location", default=os.getenv("AZURE_LOCATION", "eastus"))
    p.add_argument("--gpu-size", default="Standard_NC4as_T4_v3",
                   help="GPU VM size (e.g. Standard_NC6s_v3 for V100)")
    p.add_argument("--gpu-low-priority", action="store_true", help="Use low-priority (spot) GPU nodes")
    p.add_argument("--create", action="store_true", help="Actually provision resources")
    a = p.parse_args()
    setup_workspace(a.subscription, a.resource_group, a.workspace, a.location,
                    a.gpu_size, a.gpu_low_priority, dry_run=not a.create)


if __name__ == "__main__":
    main()
