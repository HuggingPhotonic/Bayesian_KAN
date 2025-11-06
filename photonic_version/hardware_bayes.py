"""
Variational (Bayesian) photonic bases using hardware-style mixers.

These modules mirror :mod:`photonic_version.hardware_basis` but endow the
hardware parameters (VOA attenuations, MZI phase shifters) with Gaussian
posteriors so that variational inference can capture parametric uncertainty.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn

from .basic_function1 import PhotonicCoherentBasis
from .basic_function2 import PhotonicIncoherentBasis


def _sample_diag_gaussian(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    sample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reparameterised sample along with KL to unit Gaussian prior."""

    var = torch.exp(log_var)
    if sample:
        std = torch.sqrt(var + 1e-12)
        eps = torch.randn_like(std)
        draw = mean + eps * std
    else:
        draw = mean

    kl = 0.5 * torch.sum(var + mean**2 - 1.0 - log_var)
    return draw, kl


def _softplus_gain(param: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(param) + 1e-6


class BayesianVOAArray(nn.Module):
    """
    Variational VOA bank for non-coherent photonic weighting.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        shape = (in_features, out_features, num_rings)
        self.gain_mean = nn.Parameter(torch.zeros(shape))
        self.gain_log_var = nn.Parameter(torch.full(shape, -4.0))

    def forward(self, basis_power: torch.Tensor, *, sample: bool, n_samples: int = 1):
        draws = []
        kl_total = torch.zeros(1, device=basis_power.device, dtype=basis_power.dtype)
        for _ in range(n_samples):
            gain_raw, kl = _sample_diag_gaussian(self.gain_mean, self.gain_log_var, sample)
            gains = _softplus_gain(gain_raw)
            mixed = torch.einsum("bin,ion->bo", basis_power, gains)
            draws.append(mixed)
            kl_total = kl_total + kl.to(basis_power.device)

        stacked = torch.stack(draws)
        return stacked.mean(dim=0), kl_total / n_samples


class BayesianPhaseShifter(nn.Module):
    """Variational phase shifter with bounded output."""

    def __init__(self, shape: torch.Size):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(shape))
        self.log_var = nn.Parameter(torch.full(shape, -4.0))

    def forward(self, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        raw, kl = _sample_diag_gaussian(self.mean, self.log_var, sample)
        return math.pi * torch.tanh(raw), kl


class BayesianMZIArray(nn.Module):
    """
    Variational MZI weight bank for coherent photonic bases.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        shape = (in_features, out_features, num_rings)
        self.theta_mean = nn.Parameter(torch.zeros(shape))
        self.theta_log_var = nn.Parameter(torch.full(shape, -4.0))
        self.phase = BayesianPhaseShifter(shape)

    def _amplitude(self, theta_sample: torch.Tensor) -> torch.Tensor:
        # Use sigmoid to map to (0, 1), providing smoother gradients near 0.
        return torch.sigmoid(theta_sample)

    def forward(
        self,
        basis_field: torch.Tensor,
        *,
        sample: bool,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if basis_field.dim() != 4 or basis_field.size(-1) != 2:
            raise ValueError("basis_field must be [batch, in_features, num_rings, 2].")

        batch, in_features, num_rings, _ = basis_field.shape
        outputs = []
        kl_total = torch.zeros(1, device=basis_field.device, dtype=basis_field.dtype)
        basis_complex = torch.view_as_complex(basis_field.contiguous())

        for _ in range(n_samples):
            theta_raw, kl_theta = _sample_diag_gaussian(self.theta_mean, self.theta_log_var, sample)
            amp = self._amplitude(theta_raw)
            phi, kl_phi = self.phase(sample=sample)
            weights = amp * torch.exp(1j * phi)

            combined = torch.einsum("bin,ion->bo", basis_complex, weights)
            outputs.append(torch.view_as_real(combined))
            kl_total = kl_total + (kl_theta + kl_phi).to(basis_field.device, dtype=basis_field.dtype)

        stacked = torch.stack(outputs)
        return stacked.mean(0), kl_total / n_samples


class BayesianHardwarePhotonicIncoherentBasis(PhotonicIncoherentBasis):
    """
    Photonic incoherent basis with variational VOA weighting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_rings: int = 16,
        wl_nm_range: Sequence[float] = (1546.0, 1554.0),
        R_um: float = 30.0,
        neff: float = 2.34,
        ng: float = 4.2,
        loss_dB_cm: float = 3.0,
        kappa: float = 0.2,
        phase_offsets: Iterable[float] | None = None,
    ):
        super().__init__(
            in_features,
            out_features,
            num_rings=num_rings,
            wl_nm_range=wl_nm_range,
            R_um=R_um,
            neff=neff,
            ng=ng,
            loss_dB_cm=loss_dB_cm,
            kappa=kappa,
            phase_offsets=phase_offsets,
            variational=False,
        )
        if hasattr(self, "coeffs"):
            del self.coeffs
        self.voa_mixer = BayesianVOAArray(in_features, out_features, num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)
        mixed, kl = self.voa_mixer(basis, sample=sample, n_samples=n_samples)
        residual = x @ self.base_weight
        return mixed + residual, kl


class BayesianHardwarePhotonicCoherentBasis(PhotonicCoherentBasis):
    """
    Photonic coherent basis with variational MZI weighting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_rings: int = 16,
        wl_nm_range: Sequence[float] = (1546.0, 1554.0),
        R_um: float = 30.0,
        neff: float = 2.34,
        ng: float = 4.2,
        loss_dB_cm: float = 3.0,
        kappa: float = 0.2,
        phase_offsets: Iterable[float] | None = None,
    ):
        super().__init__(
            in_features,
            out_features,
            num_rings=num_rings,
            wl_nm_range=wl_nm_range,
            R_um=R_um,
            neff=neff,
            ng=ng,
            loss_dB_cm=loss_dB_cm,
            kappa=kappa,
            phase_offsets=phase_offsets,
            variational=False,
        )
        if hasattr(self, "coeffs"):
            del self.coeffs
        self.mzi_mixer = BayesianMZIArray(in_features, out_features, num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)
        batch, in_features, feat = basis.shape
        num_rings = self.num_rings
        basis_field = basis.view(batch, in_features, num_rings, -1)
        mixed_complex, kl = self.mzi_mixer(basis_field, sample=sample, n_samples=n_samples)
        residual = x @ self.base_weight

        # Combine residual (real) with the real component of the coherent sum.
        mixed_real = mixed_complex[..., 0]
        combined = mixed_real + residual
        return combined, kl
