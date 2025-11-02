from __future__ import annotations

import torch
import torch.nn as nn


class BSplineBasis(nn.Module):
    """
    Uniform B-spline basis with optional variational coefficients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        *,
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.variational = variational

        num_ctrl_pts = grid_size + spline_order
        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, num_ctrl_pts) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, num_ctrl_pts), -5.0)
            )
            self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
            self.register_buffer(
                "prior_var",
                torch.ones_like(self.coeff_mean) * prior_scale ** 2,
            )
        else:
            self.coeffs = nn.Parameter(
                torch.randn(in_features, out_features, num_ctrl_pts) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)
        h = 2.0 / grid_size
        self.register_buffer(
            "grid",
            torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                           grid_size + 2 * spline_order + 1)
        )

    def _basis(self, x, k=0):
        if k == 0:
            x_expanded = x.unsqueeze(-1)
            return ((x_expanded >= self.grid[:-1].unsqueeze(0).unsqueeze(0)) &
                    (x_expanded < self.grid[1:].unsqueeze(0).unsqueeze(0))).float()

        prev = self._basis(x, k - 1)
        left_num = x.unsqueeze(-1) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
        left_den = self.grid[k:-1].unsqueeze(0).unsqueeze(0) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
        left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
        left = (left_num / left_den) * prev[:, :, :-1]

        right_num = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - x.unsqueeze(-1)
        right_den = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - self.grid[1:-k].unsqueeze(0).unsqueeze(0)
        right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
        right = (right_num / right_den) * prev[:, :, 1:]

        return left + right

    def forward(self, x, sample=True, n_samples=1):
        basis = []
        for i in range(self.in_features):
            basis_vals = self._basis(torch.tanh(x[:, i:i+1]), self.spline_order)
            if basis_vals.dim() == 3:
                basis_vals = basis_vals.squeeze(-2)
            basis.append(basis_vals)
        basis = torch.stack(basis, dim=1)

        if self.variational:
            if sample:
                std = torch.exp(0.5 * self.coeff_log_var)
                eps = torch.randn(n_samples, *std.shape, device=x.device)
                coeffs = self.coeff_mean + eps * std
                outputs = []
                for s in range(n_samples):
                    outputs.append(torch.einsum("bin,ion->bo", basis, coeffs[s]))
                spline_out = torch.stack(outputs).mean(0)
            else:
                spline_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
            kl = self._kl_divergence()
        else:
            spline_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
            kl = torch.zeros(1, device=x.device)

        residual = x @ self.base_weight
        return spline_out + residual, kl

    def _kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl
