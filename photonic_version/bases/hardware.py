"""
Hardware-enhanced microring bases, deterministic and Bayesian.

This module wraps the core microring bases with VOA/MZI mixers (defined in
``bases.mixers``) so that experiments can swap between abstract coefficient
tensors and hardware-aware photonic weighting networks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .modes import PhotonicCoherentBasis, PhotonicIncoherentBasis
from .mixers import (
    HardwareCoherentMixer,
    HardwareIncoherentMixer,
    BayesianMZIArray,
    BayesianVOAArray,
    _sample_diag_gaussian,
)


class _BayesianRingParameterMixin:
    """Utility mixin that places variational distributions on ring parameters."""

    def _setup_ring_posteriors(self, num_rings: int) -> None:
        phase_vals = []
        coupling_vals = []
        for ring in self.rings:
            phase_param = ring.wg.phase.detach().float()
            phase_vals.append(phase_param)
            ring.wg.phase.requires_grad_(False)
            coupling_param = ring.dc._coupling.detach().float().clamp(1e-3, 1 - 1e-3)
            coupling_vals.append(torch.logit(coupling_param))
            ring.dc._coupling.requires_grad_(False)
        self.wg_phase_mean = nn.Parameter(torch.stack(phase_vals))
        self.wg_phase_log_var = nn.Parameter(torch.full((num_rings,), -6.0))
        self.dc_coupling_mean = nn.Parameter(torch.stack(coupling_vals))
        self.dc_coupling_log_var = nn.Parameter(torch.full((num_rings,), -6.0))

    def _sample_ring_parameters(self, sample: bool) -> torch.Tensor:
        phase_draw, kl_phase = _sample_diag_gaussian(
            self.wg_phase_mean, self.wg_phase_log_var, sample
        )
        coupling_raw, kl_coupling = _sample_diag_gaussian(
            self.dc_coupling_mean, self.dc_coupling_log_var, sample
        )
        coupling_draw = torch.sigmoid(coupling_raw).clamp_(1e-3, 1 - 1e-3)
        total_kl = kl_phase + kl_coupling
        for idx, ring in enumerate(self.rings):
            ring.wg.phase.data = phase_draw[idx].detach().cpu()
            ring.dc._coupling.data = coupling_draw[idx].detach().cpu()
        return total_kl


class HardwarePhotonicIncoherentBasis(PhotonicIncoherentBasis):
    """Incoherent microring basis with VOA-style weighting network."""

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
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        if variational:
            raise ValueError("Hardware mixers currently support variational=False only.")
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
            prior_scale=prior_scale,
        )
        if hasattr(self, "coeffs"):
            del self.coeffs
        self.voa_mixer = HardwareIncoherentMixer(in_features, out_features, num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)
        mixed = self.voa_mixer(basis)
        residual = x @ self.base_weight
        kl = torch.zeros(1, device=x.device, dtype=x.dtype)
        return mixed + residual, kl


class HardwarePhotonicCoherentBasis(PhotonicCoherentBasis):
    """Coherent microring basis with MZI-style weighting network."""

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
        variational: bool = False,
        prior_scale: float = 1.0,
    ):
        if variational:
            raise ValueError("Hardware mixers currently support variational=False only.")
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
            prior_scale=prior_scale,
        )
        if hasattr(self, "coeffs"):
            del self.coeffs
        self.mzi_mixer = HardwareCoherentMixer(in_features, out_features, num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)
        mixed = self.mzi_mixer(basis)
        residual = x @ self.base_weight.to(dtype=x.dtype, device=x.device)
        kl = torch.zeros(1, device=x.device, dtype=x.dtype)
        return mixed + residual, kl


class BayesianHardwarePhotonicIncoherentBasis(_BayesianRingParameterMixin, PhotonicIncoherentBasis):
    """Photonic incoherent basis with variational VOA weighting."""

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
        self._setup_ring_posteriors(num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        ring_kl = self._sample_ring_parameters(sample)
        basis = self._evaluate_basis(x)
        mixed, kl = self.voa_mixer(basis, sample=sample, n_samples=n_samples)
        residual = x @ self.base_weight
        return mixed + residual, kl + ring_kl

class BayesianHardwarePhotonicCoherentBasis(_BayesianRingParameterMixin, PhotonicCoherentBasis):
    """Photonic coherent basis with variational MZI weighting."""

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
        self._setup_ring_posteriors(num_rings)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        ring_kl = self._sample_ring_parameters(sample)
        basis = self._evaluate_basis(x)
        batch, in_features, feat = basis.shape
        basis_field = basis.view(batch, in_features, self.num_rings, -1)
        mixed, kl = self.mzi_mixer(basis_field, sample=sample, n_samples=n_samples)
        residual = x @ self.base_weight
        mixed_real = mixed[..., 0]
        return mixed_real + residual, kl + ring_kl
