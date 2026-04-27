from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class GSMTokenListAEConfig:
    k: int
    hidden_dim: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> Dict:
        return asdict(self)


def make_tokenlist_mlp(k: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(k, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, k),
    )


class GSMTokenListAE(nn.Module):
    """
    Autoencoder in GSM token-list space.

    The latent dimension is exactly K, and latent_logits[:, i] always aligns to
    token_ids[i] from gsm_token_list.json.
    """

    def __init__(self, config: GSMTokenListAEConfig):
        super().__init__()
        if config.k <= 0:
            raise ValueError(f"k must be positive, got {config.k}")
        self.config = config
        self.encoder = make_tokenlist_mlp(config.k, config.hidden_dim, config.dropout)
        self.decoder = make_tokenlist_mlp(config.k, config.hidden_dim, config.dropout)

    @property
    def k(self) -> int:
        return int(self.config.k)

    def forward(self, input_logits_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        if input_logits_k.shape[-1] != self.k:
            raise ValueError(
                f"Expected input last dim K={self.k}, got {input_logits_k.shape[-1]}"
            )
        latent_logits_k = self.encoder(input_logits_k)
        recon_logits_k = self.decoder(latent_logits_k)
        return {
            "latent_logits": latent_logits_k,
            "recon_logits": recon_logits_k,
        }


def build_model(k: int, hidden_dim: int = 1024, dropout: float = 0.1) -> GSMTokenListAE:
    return GSMTokenListAE(
        GSMTokenListAEConfig(
            k=int(k),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
    )
