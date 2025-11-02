from __future__ import annotations

import torch
import torch.nn as nn


class WaveletBasis(nn.Module):
    """
    Haar wavelet basis layer with optional variational coefficients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        levels: int = 3,
        *,
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.levels = levels
        self.variational = variational
        self.num_basis = 1 + sum(2 ** j for j in range(levels))

        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, self.num_basis) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, self.num_basis), -5.0)
            )
            self.register_buffer(
                "prior_mean", torch.zeros_like(self.coeff_mean)
            )
            self.register_buffer(
                "prior_var",
                torch.ones_like(self.coeff_mean) * prior_scale ** 2,
            )
        else:
            self.coeffs = nn.Parameter(
                torch.randn(in_features, out_features, self.num_basis) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x) + 1.0) / 2.0

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        bases = []
        for i in range(self.in_features):
            xi = x[:, i:i+1]
            funcs = [torch.ones_like(xi)]
            for level in range(self.levels):
                freq = 2 ** level
                left_edges = torch.arange(freq, device=xi.device, dtype=xi.dtype) / freq
                mid = left_edges + 0.5 / freq
                right_edges = (torch.arange(freq, device=xi.device, dtype=xi.dtype) + 1) / freq
                for k in range(freq):
                    left_mask = ((xi >= left_edges[k]) & (xi < mid[k])).float()
                    right_mask = ((xi >= mid[k]) & (xi < right_edges[k])).float()
                    if k == freq - 1:
                        right_mask = torch.where(
                            xi == right_edges[k],
                            torch.ones_like(right_mask),
                            right_mask,
                        )
                    psi = (left_mask - right_mask) * (2.0 ** (level / 2))
                    funcs.append(psi)
            basis_stack = torch.cat(funcs, dim=1)
            bases.append(basis_stack)
        return torch.stack(bases, dim=1)

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
                wave_out = torch.stack(outputs).mean(0)
            else:
                wave_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
            kl = self._kl_divergence()
        else:
            wave_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
            kl = torch.zeros(1, device=x.device)
        residual = x @ self.base_weight
        return wave_out + residual, kl

    def _kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl
