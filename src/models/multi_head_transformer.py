"""
Multi-Head Transformer Architecture for Video Editorial Intelligence.

Implements token aggregation and three task-specific prediction heads:
1. Hook Prediction Head (BCE / ranking probability)
2. Drop-off Risk Head (BCE retention churn probability)
3. Quality Score Head (MSE continuous rating regression)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import EditorialModelConfig

logger = logging.getLogger(__name__)


class LightweightTransformerEncoder(nn.Module):
    """
    Self-contained fallback transformer encoder for offline environments and fast unit testing.
    Guarantees that the architecture can be verified, trained, and evaluated anywhere without network access.
    """

    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 768, num_layers: int = 2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.config = type("Config", (), {"hidden_size": hidden_dim})()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        x = self.embeddings(input_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return type("Output", (), {"last_hidden_state": out})()


class VideoEditorialTransformer(nn.Module):
    """
    Multi-Head Video Editorial Intelligence Neural Architecture.
    
    Combines transformer sequence representations with multimodal acoustic/lexical features
    to output joint editorial predictions.
    """

    def __init__(self, config: EditorialModelConfig):
        super().__init__()
        self.config = config

        # 1. Base Transformer Backbone
        self.encoder, self.hidden_dim = self._init_encoder(config.base_model_name, config.hidden_dim)

        # 2. Token Aggregation & Feature Projection
        self.feature_dim = config.feature_dim
        self.dropout = nn.Dropout(config.dropout)

        # 3. Task Head 1: Hook Score Head
        self.hook_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.feature_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, 1),
        )

        # 4. Task Head 2: Drop-off Churn Probability Head
        self.drop_off_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.feature_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, 1),
        )

        # 5. Task Head 3: Holistic Quality Regression Head (1.0 to 10.0)
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.feature_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, 1),
            nn.Sigmoid(),  # Rescaled to 1.0 - 10.0
        )

    def _init_encoder(self, model_name: str, fallback_dim: int) -> Tuple[nn.Module, int]:
        """Initialize pretrained encoder with graceful local fallback if offline."""
        try:
            hf_config = AutoConfig.from_pretrained(model_name, local_files_only=False)
            hidden_dim = getattr(hf_config, "hidden_size", getattr(hf_config, "dim", fallback_dim))
            encoder = AutoModel.from_pretrained(model_name)
            logger.info(f"Loaded pretrained backbone '{model_name}' (hidden_dim={hidden_dim})")
            return encoder, hidden_dim
        except Exception as e:
            logger.warning(
                f"Could not load '{model_name}' from Hugging Face Hub ({e}). "
                f"Using local LightweightTransformerEncoder (hidden_dim={fallback_dim}) for standalone execution."
            )
            return LightweightTransformerEncoder(hidden_dim=fallback_dim), fallback_dim

    def mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Token aggregation via attention-weighted mean pooling."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        feature_vector: Optional[torch.Tensor] = None,
        hook_labels: Optional[torch.Tensor] = None,
        drop_off_labels: Optional[torch.Tensor] = None,
        quality_labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and multi-head prediction layers.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            feature_vector: (batch_size, feature_dim) multimodal features
            hook_labels: (batch_size,) target binary hook labels [0, 1]
            drop_off_labels: (batch_size,) target retention churn probabilities [0, 1]
            quality_labels: (batch_size,) target quality ratings [1.0, 10.0]
            loss_weights: Optional dictionary specifying loss weighting
            
        Returns:
            Dictionary with 'hook_prob', 'drop_off_risk', 'quality_score', and optionally 'loss'.
        """
        # Pass through backbone transformer
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(encoder_outputs, attention_mask)
        pooled = self.dropout(pooled)

        # Fuse pooled text embeddings with acoustic/lexical feature vector
        batch_size = input_ids.size(0)
        if feature_vector is None:
            feature_vector = torch.zeros(batch_size, self.feature_dim, device=input_ids.device)

        combined = torch.cat([pooled, feature_vector], dim=-1)

        # Prediction Heads
        hook_logits = self.hook_head(combined).squeeze(-1)
        drop_off_logits = self.drop_off_head(combined).squeeze(-1)
        quality_raw = self.quality_head(combined).squeeze(-1)  # in [0, 1]

        hook_prob = torch.sigmoid(hook_logits)
        drop_off_prob = torch.sigmoid(drop_off_logits)
        # Scale quality score to [1.0, 10.0] range
        quality_score = 1.0 + (quality_raw * 9.0)

        output = {
            "hook_logits": hook_logits,
            "hook_prob": hook_prob,
            "drop_off_logits": drop_off_logits,
            "drop_off_risk": drop_off_prob,
            "quality_score": quality_score,
        }

        # Multi-task Loss Computation
        if hook_labels is not None or drop_off_labels is not None or quality_labels is not None:
            weights = loss_weights or {"hook": 1.0, "drop_off": 1.5, "quality": 0.5}
            total_loss = torch.tensor(0.0, device=input_ids.device)

            if hook_labels is not None:
                bce_hook = nn.BCEWithLogitsLoss()(hook_logits, hook_labels.float())
                total_loss = total_loss + weights.get("hook", 1.0) * bce_hook
                output["loss_hook"] = bce_hook

            if drop_off_labels is not None:
                bce_drop = nn.BCEWithLogitsLoss()(drop_off_logits, drop_off_labels.float())
                total_loss = total_loss + weights.get("drop_off", 1.5) * bce_drop
                output["loss_drop_off"] = bce_drop

            if quality_labels is not None:
                # Normalize targets to [0, 1] for stable MSE with sigmoid head
                normalized_targets = (quality_labels.float() - 1.0) / 9.0
                mse_quality = nn.MSELoss()(quality_raw, normalized_targets)
                total_loss = total_loss + weights.get("quality", 0.5) * mse_quality
                output["loss_quality"] = mse_quality

            output["loss"] = total_loss

        return output

