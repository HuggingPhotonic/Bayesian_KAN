"""
Photonic hardware-inspired weighting modules for microring bases.

These components replace the abstract learnable coefficient tensors used in
``PhotonicIncoherentBasis`` and ``PhotonicCoherentBasis`` with counterparts
that mimic physically implementable photonic elements:

* TrainableVOAArray – models an array of variable optical attenuators that
  combine multiple power-domain ring responses (non-coherent case).
* TrainableMZIArray – models a bank of Mach–Zehnder interferometer (MZI)
  weight units followed by phase shifters to realise complex-valued
  amplitude/phase tuning (coherent case).

Both modules expose drop-in replacements that operate on the basis tensors
computed in ``basic_function1``/``basic_function2``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _softplus_gain(param: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameter to strictly-positive gain."""
    return torch.nn.functional.softplus(param) + 1e-6


class TrainableVOAArray(nn.Module):
    """
    Trainable VOA (variable optical attenuator) bank.

    Each input ring response is attenuated by a non-negative gain realised
    through a softplus parameterisation that emulates VOA loss tuning.

    Parameters
    ----------
    in_features: int
        Number of input features (KAN layer inputs).
    out_features: int
        Number of output features.
    num_rings: int
        Number of microring basis functions per input feature.
    shared_over_inputs: bool
        Whether the VOA setting is shared across input features.  When False
        (default) each input feature owns its own attenuator bank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_rings: int,
        *,
        shared_over_inputs: bool = False,
    ):
        super().__init__()
        if shared_over_inputs:
            shape = (out_features, num_rings)
        else:
            shape = (in_features, out_features, num_rings)
        self.raw_gain = nn.Parameter(torch.zeros(shape))

    def forward(self, basis_power: torch.Tensor) -> torch.Tensor:
        """
        Combine power-domain ring responses using VOA gains.

        Parameters
        ----------
        basis_power: torch.Tensor
            Tensor with shape (batch, in_features, num_rings) containing power
            responses from microring notches (e.g. output of
            PhotonicIncoherentBasis._evaluate_basis).
        """

        if basis_power.dim() != 3:
            raise ValueError("basis_power must be [batch, in_features, num_rings].")

        gains = _softplus_gain(self.raw_gain)
        if gains.dim() == 2:
            gains = gains.unsqueeze(0).expand(basis_power.size(1), -1, -1)

        # einsum: b i n , i o n -> b o
        mixed = torch.einsum("bin,ion->bo", basis_power, gains)
        return mixed


class TrainablePhaseShifter(nn.Module):
    """
    Trainable phase shifter producing phases in [-pi, pi].

    The parameterisation uses tanh to keep phases bounded while still covering
    the full 2π range required for coherent weighting.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.raw_phase = nn.Parameter(torch.zeros(in_features, out_features, num_rings))

    def forward(self) -> torch.Tensor:
        return math.pi * torch.tanh(self.raw_phase)


class TrainableMZIArray(nn.Module):
    """
    Trainable Mach–Zehnder interferometer (MZI) weight bank.

    Each coefficient is realised by a 2x2 MZI block that controls amplitude via
    an internal phase ``theta`` and overall phase via an output phase shifter
    ``phi``.  The underlying formulation mirrors common photonic neural-network
    building blocks.

    References
    ----------
    * Shen et al., "Deep learning with coherent nanophotonic circuits",
      Nature Photonics 2017.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(in_features, out_features, num_rings))
        self.phase = TrainablePhaseShifter(in_features, out_features, num_rings)
        # Initialise MZI so that the effective coupling starts weak, reducing
        # large initial interference spikes during training.
        nn.init.constant_(self.theta, -3.0)

    def _complex_weights(self) -> torch.Tensor:
        """
        Compute complex weights from the current MZI settings.

        Returns
        -------
        torch.Tensor
            Complex weights with shape (in_features, out_features, num_rings).
        """

        amp = torch.sin(torch.nn.functional.sigmoid(self.theta) * math.pi * 0.5) ** 2
        phase = self.phase()
        return amp * torch.exp(1j * phase)

    def forward(self, basis_field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the MZI array to coherent basis responses.

        Parameters
        ----------
        basis_field: torch.Tensor
            Tensor with shape (batch, in_features, num_rings, 2) containing
            real/imag parts of microring field responses.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Real and imaginary components of the weighted sum, each with shape
            (batch, out_features).
        """

        if basis_field.dim() != 4 or basis_field.size(-1) != 2:
            raise ValueError("basis_field must be [batch, in_features, num_rings, 2].")

        basis_complex = torch.view_as_complex(basis_field.contiguous())
        weights = self._complex_weights()

        combined = torch.einsum("bin,ion->bo", basis_complex, weights)
        combined_realimag = torch.view_as_real(combined)  # (batch, out_features, 2)
        real = combined_realimag[..., 0]
        imag = combined_realimag[..., 1]
        return real, imag


class HardwareIncoherentMixer(nn.Module):
    """
    Drop-in replacement for the coefficient tensor in PhotonicIncoherentBasis.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.voa_bank = TrainableVOAArray(in_features, out_features, num_rings)

    def forward(self, basis_power: torch.Tensor) -> torch.Tensor:
        return self.voa_bank(basis_power)


class HardwareCoherentMixer(nn.Module):
    """
    Drop-in replacement for the coefficient tensor in PhotonicCoherentBasis.

    Combines coherent ring responses using an MZI array, returning real-valued
    outputs by taking the in-phase component after combination.
    """

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.mzi_bank = TrainableMZIArray(in_features, out_features, num_rings)
        self.num_rings = num_rings

    def forward(self, basis_field: torch.Tensor) -> torch.Tensor:
        if basis_field.dim() != 3:
            raise ValueError("basis_field must be [batch, in_features, 2*num_rings].")
        batch, in_features, features = basis_field.shape
        if features != 2 * self.num_rings:
            raise ValueError(
                f"Expected last dimension 2*num_rings={2*self.num_rings}, "
                f"got {features}."
            )
        reshaped = basis_field.view(batch, in_features, self.num_rings, 2).contiguous()
        real, imag = self.mzi_bank(reshaped)
        # Here we return only the real component; applications that require both
        # quadratures can combine them as needed.
        return real
