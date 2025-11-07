"""
Microring basis variants (coherent and incoherent) built on the shared core.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .core import AllPassRing, MicroringBasisBase


class PhotonicCoherentBasis(MicroringBasisBase):
    """Coherent microring basis that keeps the complex field response."""

    response_channels = 2

    def _ring_response(
        self,
        ring: AllPassRing,
        wavelengths: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        field = ring(source=source, power=False)
        field = field.squeeze(-1).squeeze(-1)

        if torch.is_complex(field):
            return torch.view_as_real(field)

        if field.ndim >= 2 and field.shape[0] == 2:
            real, imag = field[0], field[1]
            return torch.stack((real, imag), dim=-1)

        imag = torch.zeros_like(field)
        return torch.stack((field, imag), dim=-1)

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        return basis


class PhotonicIncoherentBasis(MicroringBasisBase):
    """
    Incoherent microring basis that operates on optical power.
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
        power = ring(source=source, power=True)
        power = power.squeeze(-1).squeeze(-1)
        notch = 1.0 - power
        return notch.unsqueeze(-1)

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        return torch.clamp(basis, min=0.0)


class PhotonicTorchBasis(PhotonicIncoherentBasis):
    """
    Backward-compatible alias that reuses the incoherent basis implementation.
    """

    pass

