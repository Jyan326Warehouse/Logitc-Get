from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class SST5ContentAEConfig:
    k: int
    fusion_hidden_dim: int = 64
    encoder_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    residual_fusion: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


def make_k_to_k_mlp(k: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(k),
        nn.Linear(k, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, k),
    )


class SST5ContentAE(nn.Module):
    """
    AE in frozen SST-5 train-derived token K-space.
    This is a fresh K-to-K implementation for the SST-5 experiment; it does
    not import the archived original AE modules.

    External tensors stay in the same K-space and may carry any leading shape:
        input_logitc_K: [B,T,K]
        fused_logits:   [B,T,K]
        latent_logits:  [B,T,K]
        recon_logits:   [B,T,K]

    Token CE is computed over the K dimension. recon_logits are decoded from
    latent_logits and trained to match input_logitc_K over the same K tokens.
    The model never emits full-vocabulary logits.
    """

    def __init__(self, config: SST5ContentAEConfig) -> None:
        super().__init__()
        if config.k <= 0:
            raise ValueError(f"k must be positive, got {config.k}")
        for name, value in (
            ("fusion_hidden_dim", config.fusion_hidden_dim),
            ("encoder_hidden_dim", config.encoder_hidden_dim),
            ("decoder_hidden_dim", config.decoder_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.config = config
        self.feature_fusion = make_k_to_k_mlp(
            config.k,
            config.fusion_hidden_dim,
            config.dropout,
        )
        self.encoder = make_k_to_k_mlp(
            config.k,
            config.encoder_hidden_dim,
            config.dropout,
        )
        self.decoder = make_k_to_k_mlp(
            config.k,
            config.decoder_hidden_dim,
            config.dropout,
        )

    @property
    def k(self) -> int:
        return int(self.config.k)

    def forward(self, input_logitc_K: torch.Tensor) -> Dict[str, torch.Tensor]:
        if input_logitc_K.ndim != 3:
            raise ValueError(
                f"input_logitc_K must have shape [B,T,K], got {list(input_logitc_K.shape)}"
            )
        if input_logitc_K.shape[-1] != self.k:
            raise ValueError(
                f"Expected input_logitc_K last dim K={self.k}, got {input_logitc_K.shape[-1]}"
            )
        fused_delta = self.feature_fusion(input_logitc_K)
        if self.config.residual_fusion:
            fused_logits = input_logitc_K + fused_delta
        else:
            fused_logits = fused_delta
        latent_logits = self.encoder(fused_logits)
        recon_logits = self.decoder(latent_logits)
        return {
            "fused_logits": fused_logits,
            "latent_logits": latent_logits,
            "recon_logits": recon_logits,
            "latent_logits_K": latent_logits,
            "recon_logitc_K": recon_logits,
        }


def build_model(
    k: int,
    fusion_hidden_dim: int = 64,
    encoder_hidden_dim: int = 128,
    decoder_hidden_dim: int = 128,
    dropout: float = 0.1,
    residual_fusion: bool = True,
) -> SST5ContentAE:
    return SST5ContentAE(
        SST5ContentAEConfig(
            k=int(k),
            fusion_hidden_dim=int(fusion_hidden_dim),
            encoder_hidden_dim=int(encoder_hidden_dim),
            decoder_hidden_dim=int(decoder_hidden_dim),
            dropout=float(dropout),
            residual_fusion=bool(residual_fusion),
        )
    )
