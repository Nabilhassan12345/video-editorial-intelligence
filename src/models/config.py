"""
Model and Training configuration definitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EditorialModelConfig:
    """Configuration for the Multi-Head Video Editorial Transformer."""
    base_model_name: str = "distilbert-base-uncased"
    hidden_dim: int = 768
    feature_dim: int = 8
    mlp_hidden_dim: int = 128
    dropout: float = 0.1
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])
    quantization_bits: Optional[int] = None  # 4 for QLoRA NF4, None for fp32/bf16


@dataclass
class TrainingConfig:
    """Hyperparameters and execution configuration for fine-tuning."""
    output_dir: str = "./checkpoints/editorial_model"
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    hook_loss_weight: float = 1.0
    drop_off_loss_weight: float = 1.5
    quality_loss_weight: float = 0.5
    eval_steps: int = 50
    save_total_limit: int = 2
    device: str = "cpu"
    seed: int = 42
    dry_run: bool = False

