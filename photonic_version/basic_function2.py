"""
Incoherent photonic basis functions using Photontorch microring resonators.

This module complements :mod:`photonic_version.basic_function1` by providing an
intensity-only (incoherent) encoding of the microring response.  The power
notch of each ring is used as the photonic analogue of a B-spline basis
function.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .basic_function1 import AllPassRing, MicroringBasisBase


class PhotonicIncoherentBasis(MicroringBasisBase):
    """
    Incoherent microring basis that operates on optical power.

    Each ring contributes a single positive channel corresponding to the notch
    depth (1 - transmitted power).
    """

    response_channels = 1

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
            variational=variational,
            prior_scale=prior_scale,
        )

    def _ring_response(
        self,
        ring: AllPassRing,
        wavelengths: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        # Non-coherent regime: we only care about transmitted power.
        power = ring(source=source, power=True)
        power = power.squeeze(-1).squeeze(-1)  # -> (n_wl,)
        # Convert to a notch-like basis function (higher = deeper resonance).
        notch = 1.0 - power
        return notch.unsqueeze(-1)  # align with response_channels=1

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        # Clamp to [0, ∞) to match physical power constraints.
        return torch.clamp(basis, min=0.0)
