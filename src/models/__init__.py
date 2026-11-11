"""
Model architecture, multi-head neural layers, and PEFT LoRA adapters.
"""

from .config import EditorialModelConfig, TrainingConfig
from .multi_head_transformer import VideoEditorialTransformer
from .peft_adapter import inject_peft_adapters, load_fine_tuned_model

__all__ = [
    "EditorialModelConfig",
    "TrainingConfig",
    "VideoEditorialTransformer",
    "inject_peft_adapters",
    "load_fine_tuned_model",
]

