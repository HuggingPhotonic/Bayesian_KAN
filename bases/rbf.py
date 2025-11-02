from __future__ import annotations

import torch
import torch.nn as nn


class RBFBasis(nn.Module):
    """
    Radial basis function layer with optional variational coefficients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 16,
        *,
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.variational = variational

        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.log_sigma = nn.Parameter(torch.zeros(num_centers))

        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, num_centers) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, num_centers), -5.0)
            )
            self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
            self.register_buffer(
                "prior_var",
                torch.ones_like(self.coeff_mean) * prior_scale ** 2,
            )
        else:
            self.coeffs = nn.Parameter(
                torch.randn(in_features, out_features, num_centers) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)
        centers = self.centers.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        dist_sq = torch.sum((x_expanded - centers) ** 2, dim=-1)
        rbf_vals = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))
        return rbf_vals.unsqueeze(1).repeat(1, self.in_features, 1)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._basis(x)
        if self.variational:
            if sample:
                std = torch.exp(0.5 * self.coeff_log_var)
                eps = torch.randn(n_samples, *std.shape, device=x.device)
                coeffs = self.coeff_mean + eps * std
                outputs = []
                for s in range(n_samples):
                    outputs.append(torch.einsum("bin,ion->bo", basis, coeffs[s]))
                rbf_out = torch.stack(outputs).mean(0)
            else:
                rbf_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
            kl = self._kl_divergence()
        else:
            rbf_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
            kl = torch.zeros(1, device=x.device)
        residual = x @ self.base_weight
        return rbf_out + residual, kl

    def _kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl
