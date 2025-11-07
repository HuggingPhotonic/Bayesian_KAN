"""
Reusable prior distributions for photonic Bayesian models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class GaussianPrior:
    mean: float = 0.0
    std: float = 1.0

    def kl(self, posterior_mean: torch.Tensor, posterior_log_var: torch.Tensor) -> torch.Tensor:
        var = torch.exp(posterior_log_var)
        kl = 0.5 * (
            ((posterior_mean - self.mean) ** 2 + var) / (self.std**2)
            - 1.0
            - posterior_log_var
            + 2 * math.log(self.std)
        )
        return kl.sum()


@dataclass
class BetaPrior:
    alpha: float = 1.0
    beta: float = 1.0

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return (self.alpha - 1) * torch.log(x.clamp(min=1e-6)) + (self.beta - 1) * torch.log(1 - x.clamp(max=1 - 1e-6)) - torch.lgamma(
            torch.tensor(self.alpha)
        ) - torch.lgamma(torch.tensor(self.beta)) + torch.lgamma(torch.tensor(self.alpha + self.beta))


@dataclass
class VonMisesPrior:
    mu: float = 0.0
    kappa: float = 1.0

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return self.kappa * torch.cos(theta - self.mu) - math.log(2 * math.pi) - torch.log(torch.i0(torch.tensor(self.kappa)))


@dataclass
class HalfNormalPrior:
    scale: float = 1.0

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=1e-6)
        return torch.log(2 / (math.sqrt(2 * math.pi) * self.scale)) - 0.5 * (x / self.scale) ** 2 + torch.log(x)


@dataclass
class LogNormalPrior:
    mean: float = 0.0
    std: float = 1.0

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=1e-6)
        return -torch.log(x * self.std * math.sqrt(2 * math.pi)) - (torch.log(x) - self.mean) ** 2 / (2 * self.std**2)
