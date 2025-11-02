from __future__ import annotations

import torch
import torch.nn as nn


class ChebyshevBasis(nn.Module):
    """
    Chebyshev polynomial basis layer supporting deterministic and variational
    (mean/log-var) coefficients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 10,
        *,
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.variational = variational

        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, degree + 1) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, degree + 1), -5.0)
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
                torch.randn(in_features, out_features, degree + 1) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def _normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def _polynomials(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalized_input(x)
        polys = []
        for i in range(self.in_features):
            xi = x[:, i:i+1]
            T0 = torch.ones_like(xi)
            T1 = xi
            basis = [T0, T1]
            for k in range(2, self.degree + 1):
                Tk = 2 * xi * basis[-1] - basis[-2]
                basis.append(Tk)
            basis_stack = torch.stack(basis, dim=-1).squeeze(1)
            polys.append(basis_stack)
        return torch.stack(polys, dim=1)  # (batch, in_features, degree+1)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._polynomials(x)
        if self.variational:
            if sample:
                std = torch.exp(0.5 * self.coeff_log_var)
                eps = torch.randn(n_samples, *std.shape, device=x.device)
                coeffs = self.coeff_mean + eps * std
                outputs = []
                for s in range(n_samples):
                    outputs.append(torch.einsum("bin,ion->bo", basis, coeffs[s]))
                poly_out = torch.stack(outputs).mean(0)
            else:
                poly_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
            kl = self._kl_divergence()
        else:
            poly_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
            kl = torch.zeros(1, device=x.device)
        residual = x @ self.base_weight
        return poly_out + residual, kl

    def _kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl
