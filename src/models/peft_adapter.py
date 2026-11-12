"""
Parameter-Efficient Fine-Tuning (PEFT) and LoRA adapter integration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .config import EditorialModelConfig
from .multi_head_transformer import VideoEditorialTransformer

logger = logging.getLogger(__name__)


def inject_peft_adapters(
    model: VideoEditorialTransformer,
    config: EditorialModelConfig,
) -> VideoEditorialTransformer:
    """
    Inject Low-Rank Adaptation (LoRA) adapters into the transformer encoder attention projections.
    Freezes base encoder parameters and enables efficient fine-tuning.
    """
    if not config.use_peft:
        logger.info("PEFT disabled in config; using full parameter training.")
        return model

    # Determine target modules according to architecture
    base_name = config.base_model_name.lower()
    if "distilbert" in base_name:
        target_modules = ["q_lin", "v_lin"]
    elif "mistral" in base_name or "llama" in base_name or "qwen" in base_name:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "bert" in base_name:
        target_modules = ["query", "value"]
    else:
        target_modules = config.lora_target_modules

    try:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        model.encoder = get_peft_model(model.encoder, lora_config)
        trainable_params, all_param = count_parameters(model)
        pct = 100 * trainable_params / all_param if all_param > 0 else 0.0
        logger.info(
            f"PEFT LoRA injected successfully. "
            f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {pct:.2f}%"
        )
    except Exception as e:
        logger.warning(
            f"Could not inject PEFT LoRA into encoder ({e}). "
            f"Falling back to selective freezing of base encoder layers."
        )
        for param in model.encoder.parameters():
            param.requires_grad = False
        # Keep prediction heads trainable
        for head in [model.hook_head, model.drop_off_head, model.quality_head]:
            for param in head.parameters():
                param.requires_grad = True

    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters in the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_fine_tuned_model(
    model: VideoEditorialTransformer,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Save fine-tuned model artifacts:
    1. PEFT adapter weights (if PEFT is active)
    2. Multi-head prediction layer weights (`prediction_heads.pt`)
    3. Model configuration (`editorial_config.json`)
    4. Training & evaluation metadata (`training_metadata.json`)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save encoder / PEFT adapters
    if isinstance(model.encoder, PeftModel):
        model.encoder.save_pretrained(str(out_path / "adapter"))
        logger.info(f"Saved PEFT LoRA adapter to {out_path / 'adapter'}")
    else:
        torch.save(model.encoder.state_dict(), str(out_path / "encoder.pt"))

    # 2. Save prediction heads
    heads_state = {
        "hook_head": model.hook_head.state_dict(),
        "drop_off_head": model.drop_off_head.state_dict(),
        "quality_head": model.quality_head.state_dict(),
    }
    torch.save(heads_state, str(out_path / "prediction_heads.pt"))
    logger.info(f"Saved multi-head weights to {out_path / 'prediction_heads.pt'}")

    # 3. Save config
    config_dict = {
        "base_model_name": model.config.base_model_name,
        "hidden_dim": model.hidden_dim,
        "feature_dim": model.feature_dim,
        "mlp_hidden_dim": model.config.mlp_hidden_dim,
        "dropout": model.config.dropout,
        "use_peft": model.config.use_peft,
        "lora_r": model.config.lora_r,
        "lora_alpha": model.config.lora_alpha,
    }
    with open(out_path / "editorial_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Save metadata
    if metadata:
        with open(out_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return out_path


def load_fine_tuned_model(
    checkpoint_dir: str,
    device: str = "cpu",
) -> VideoEditorialTransformer:
    """
    Load a fine-tuned model from checkpoint directory.
    Reconstructs architecture, loads PEFT adapter or encoder weights, and loads prediction heads.
    """
    chk_path = Path(checkpoint_dir)
    if not chk_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    config_file = chk_path / "editorial_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg_data = json.load(f)
        config = EditorialModelConfig(
            base_model_name=cfg_data.get("base_model_name", "distilbert-base-uncased"),
            hidden_dim=cfg_data.get("hidden_dim", 768),
            feature_dim=cfg_data.get("feature_dim", 8),
            mlp_hidden_dim=cfg_data.get("mlp_hidden_dim", 128),
            use_peft=cfg_data.get("use_peft", True),
            lora_r=cfg_data.get("lora_r", 8),
            lora_alpha=cfg_data.get("lora_alpha", 16),
        )
    else:
        config = EditorialModelConfig()

    model = VideoEditorialTransformer(config)

    # Load adapter if present
    adapter_dir = chk_path / "adapter"
    if adapter_dir.exists():
        try:
            model.encoder = PeftModel.from_pretrained(model.encoder, str(adapter_dir))
            logger.info(f"Loaded PEFT LoRA adapter from {adapter_dir}")
        except Exception as e:
            logger.warning(f"Failed to load PEFT adapter ({e}). Using base encoder.")
    elif (chk_path / "encoder.pt").exists():
        model.encoder.load_state_dict(torch.load(chk_path / "encoder.pt", map_location=device))

    # Load prediction heads
    heads_file = chk_path / "prediction_heads.pt"
    if heads_file.exists():
        heads_state = torch.load(heads_file, map_location=device)
        model.hook_head.load_state_dict(heads_state["hook_head"])
        model.drop_off_head.load_state_dict(heads_state["drop_off_head"])
        model.quality_head.load_state_dict(heads_state["quality_head"])
        logger.info(f"Loaded multi-head prediction layers from {heads_file}")

    model.to(device)
    model.eval()
    return model
