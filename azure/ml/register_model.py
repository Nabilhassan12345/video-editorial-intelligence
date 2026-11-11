"""
Azure Machine Learning Model Registry Utility.

Registers a fine-tuned model checkpoint as a versioned asset in the Azure ML Model Registry.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

from .submit_training import get_ml_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def register_model_asset(
    checkpoint_dir: str = "checkpoints/editorial_model",
    model_name: str = "video-editorial-transformer",
    description: str = "Fine-tuned multi-head transformer with PEFT LoRA for video editorial intelligence.",
    dry_run: bool = True,
):
    """Register local checkpoint directory as a versioned model asset in Azure ML."""
    chk_path = Path(checkpoint_dir)
    if not chk_path.exists():
        raise FileNotFoundError(f"Local checkpoint directory not found: {checkpoint_dir}")

    # Inspect metadata if available
    metadata = {}
    meta_file = chk_path / "training_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)

    model_asset = Model(
        name=model_name,
        path=str(chk_path),
        type=AssetTypes.CUSTOM_MODEL,
        description=description,
        tags={
            "framework": "PyTorch",
            "tuning": "PEFT_LoRA",
            "task": "video-editorial-intelligence",
            "best_val_loss": str(metadata.get("best_val_loss", "N/A")),
        },
    )

    print("\n" + "=" * 65)
    print("📦 AZURE MACHINE LEARNING MODEL REGISTRATION SPECIFICATION")
    print("=" * 65)
    print(f"Model Name:       {model_asset.name}")
    print(f"Local Path:       {model_asset.path}")
    print(f"Model Type:       {model_asset.type}")
    print(f"Tags:             {model_asset.tags}")
    print("=" * 65)

    if dry_run:
        print("\n[NOTICE] Dry-run preview mode. Model was NOT registered in Azure ML.")
        print("To register live in Azure ML Model Registry, pass --register.\n")
        return model_asset

    ml_client = get_ml_client()
    registered = ml_client.models.create_or_update(model_asset)
    logger.info(f"Model '{registered.name}' registered successfully! Version: {registered.version}")
    return registered


def main():
    parser = argparse.ArgumentParser(description="Register Model in Azure ML Model Registry")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/editorial_model", help="Path to checkpoint directory")
    parser.add_argument("--model-name", type=str, default="video-editorial-transformer", help="Registered model name")
    parser.add_argument("--register", action="store_true", help="Perform live model registration")
    args = parser.parse_args()

    register_model_asset(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        dry_run=not args.register,
    )


if __name__ == "__main__":
    main()

