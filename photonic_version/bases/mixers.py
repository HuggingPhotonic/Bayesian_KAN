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
produced by :mod:`photonic_version.bases.modes`.
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
        nn.init.uniform_(self.theta, -3.0, -1.0)

    def _weight_components(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute real/imaginary weight components from the current MZI settings.
        """

        theta = self.theta.to(device=device, dtype=dtype)
        amp = torch.sin(torch.nn.functional.sigmoid(theta) * math.pi * 0.5) ** 2
        phase = self.phase().to(device=device, dtype=dtype)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return real, imag

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

        dtype = basis_field.dtype
        device = basis_field.device
        basis_real = basis_field[..., 0]
        basis_imag = basis_field[..., 1]
        weight_real, weight_imag = self._weight_components(dtype=dtype, device=device)

        real = torch.einsum("bin,ion->bo", basis_real, weight_real) - torch.einsum(
            "bin,ion->bo", basis_imag, weight_imag
        )
        imag = torch.einsum("bin,ion->bo", basis_real, weight_imag) + torch.einsum(
            "bin,ion->bo", basis_imag, weight_real
        )
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
        dtype = self.mzi_bank.theta.dtype
        device = self.mzi_bank.theta.device
        basis_field = basis_field.to(dtype=dtype, device=device)
        reshaped = basis_field.view(batch, in_features, self.num_rings, 2).contiguous()
        real, imag = self.mzi_bank(reshaped)
        # Here we return only the real component; applications that require both
        # quadratures can combine them as needed.
        return real


# --------------------------------------------------------------------------- #
# Bayesian counterparts used by variational hardware bases.
# --------------------------------------------------------------------------- #

def _sample_diag_gaussian(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    sample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    var = torch.exp(log_var)
    if sample:
        std = torch.sqrt(var + 1e-12)
        eps = torch.randn_like(std)
        draw = mean + eps * std
    else:
        draw = mean

    # Use a more relaxed prior: N(0, prior_var²) instead of N(0, 1)
    # This allows posterior means to deviate more without high KL penalty
    prior_var = 5.0  # Prior std=5.0, allowing means in roughly [-15, 15]
    prior_log_var = 2.0 * torch.log(torch.tensor(prior_var))

    # KL(q||p) for diagonal Gaussians with different priors
    kl = 0.5 * torch.mean(
        var / (prior_var ** 2) +
        (mean ** 2) / (prior_var ** 2) -
        1.0 -
        log_var +
        prior_log_var
    )
    return draw, kl


class BayesianVOAArray(nn.Module):
    """Variational VOA bank for non-coherent photonic weighting."""

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        shape = (in_features, out_features, num_rings)
        self.gain_mean = nn.Parameter(torch.randn(shape) * 0.05)
        self.gain_log_var = nn.Parameter(torch.full(shape, -6.0))

    def forward(self, basis_power: torch.Tensor, *, sample: bool, n_samples: int = 1):
        draws = []
        kl_total = torch.zeros(1, device=basis_power.device, dtype=basis_power.dtype)
        for _ in range(n_samples):
            gain_raw, kl = _sample_diag_gaussian(self.gain_mean, self.gain_log_var, sample)
            gains = _softplus_gain(gain_raw)
            mixed = torch.einsum("bin,ion->bo", basis_power, gains)
            draws.append(mixed)
            kl_total = kl_total + kl.to(basis_power.device, dtype=basis_power.dtype)
        stacked = torch.stack(draws)
        return stacked.mean(dim=0), kl_total / n_samples


class BayesianPhaseShifter(nn.Module):
    """Variational phase shifter with bounded output."""

    def __init__(self, shape: torch.Size):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(shape))
        self.log_var = nn.Parameter(torch.full(shape, -6.0))

    def forward(self, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        raw, kl = _sample_diag_gaussian(self.mean, self.log_var, sample)
        return math.pi * torch.tanh(raw), kl


class BayesianMZIArray(nn.Module):
    """Variational MZI weight bank for coherent photonic bases."""

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        shape = (in_features, out_features, num_rings)
        self.theta_mean = nn.Parameter(torch.randn(shape) * 0.05)
        self.theta_log_var = nn.Parameter(torch.full(shape, -6.0))
        self.phase = BayesianPhaseShifter(shape)

    def _amplitude(self, theta_sample: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(theta_sample)

    def forward(
        self,
        basis_field: torch.Tensor,
        *,
        sample: bool,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if basis_field.dim() != 4 or basis_field.size(-1) != 2:
            raise ValueError("basis_field must be [batch, in_features, num_rings, 2].")

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
