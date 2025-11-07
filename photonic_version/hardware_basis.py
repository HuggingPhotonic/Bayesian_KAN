"""
Photonic KAN bases that use hardware-inspired weighting components.

These variants reuse the microring simulation logic from ``basic_function1`` /
``basic_function2`` but replace the abstract linear coefficient tensors with
hardware-motivated mixers defined in ``hardware_components``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .basic_function1 import MicroringBasisBase, PhotonicCoherentBasis
from .basic_function2 import PhotonicIncoherentBasis
from .hardware_components import HardwareCoherentMixer, HardwareIncoherentMixer


class HardwarePhotonicIncoherentBasis(PhotonicIncoherentBasis):
    """
    Incoherent microring basis with VOA-style weighting network.
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
    """
    Coherent microring basis with MZI-style weighting network.
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
