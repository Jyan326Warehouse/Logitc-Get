from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LogitsAEConfig:
    input_dim: int = 151936
    latent_dim: int = 256


class LogitsEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TiedLogitsDecoder(nn.Module):
    def __init__(self, encoder: LogitsEncoder):
        super().__init__()
        self.encoder = encoder
        self.bias = nn.Parameter(torch.zeros(encoder.proj.in_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.encoder.proj.weight.t(), self.bias)


class LogitsAutoEncoder(nn.Module):
    def __init__(self, config: LogitsAEConfig):
        super().__init__()
        self.config = config
        self.encoder = LogitsEncoder(config.input_dim, config.latent_dim)
        self.decoder = TiedLogitsDecoder(self.encoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def build_model(input_dim: int = 151936, latent_dim: int = 256) -> LogitsAutoEncoder:
    config = LogitsAEConfig(input_dim=input_dim, latent_dim=latent_dim)
    return LogitsAutoEncoder(config)
