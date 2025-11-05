"""Analytic photonic basis functions for KAN models.

This reimplementation avoids relying on Photontorch's internal autograd path
by explicitly modelling single-bus microring resonators with differentiable
PyTorch expressions.  Every parameter (effective index, coupling, loss, phase
offset, residual weights) is exposed as an `nn.Parameter`, so gradients are
available end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class RingSpec:
    R_um: float = 30.0
    neff: float = 2.34
    loss_dB_cm: float = 3.0
    kappa: float = 0.2
    phase_offset: float = 0.0


class AnalyticRing(nn.Module):
    """Single-bus microring resonator with differentiable parameters."""

    def __init__(self, spec: RingSpec):
        super().__init__()
        self.length = 2 * math.pi * (spec.R_um * 1e-6)

        # Effective index as base value + learnable delta.
        self.neff_base = nn.Parameter(torch.tensor(spec.neff), requires_grad=False)
        self.neff_delta = nn.Parameter(torch.zeros(1))

        # Coupling coefficient in (0, 1). Use logit parameterisation.
        kappa = torch.tensor(spec.kappa, dtype=torch.float32)
        kappa = torch.clamp(kappa, 1e-3, 1 - 1e-3)
        self.kappa_raw = nn.Parameter(torch.logit(kappa))

        # Amplitude loss per round trip via sigmoid parameter.
        loss_amp = 10 ** (-spec.loss_dB_cm * (self.length * 100.0) / 20.0)
        loss_amp = torch.tensor(loss_amp, dtype=torch.float32)
        loss_amp = torch.clamp(loss_amp, 1e-6, 0.9999)
        self.loss_raw = nn.Parameter(torch.logit(loss_amp))

        # Optional extra phase shift (radians).
        self.phase_offset = nn.Parameter(torch.tensor(spec.phase_offset, dtype=torch.float32))

    def forward(self, wl: torch.Tensor) -> torch.Tensor:
        neff = self.neff_base + self.neff_delta
        kappa = torch.sigmoid(self.kappa_raw)
        r = torch.sqrt(torch.clamp(1 - kappa ** 2, min=1e-6))
        a = torch.sigmoid(self.loss_raw)

        phi = 2 * math.pi * neff * self.length / wl + self.phase_offset
        exp_term = torch.exp(-1j * phi)

        numerator = r - a * exp_term
        denominator = 1 - r * a * exp_term
        transfer = numerator / denominator
        return transfer


class PhotonicMixerPower(nn.Module):
    """Power combiner with signed, normalised weights."""

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(in_features, out_features, num_rings))
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self, powers: torch.Tensor) -> torch.Tensor:
        weights = torch.tanh(self.raw_weights)
        weights = weights / (weights.norm(p=2, dim=-1, keepdim=True) + self.eps)
        return torch.einsum("bin,ion->bo", powers, weights)


class PhotonicMixerCoherent(nn.Module):
    """Coherent combiner with amplitude + phase control."""

    def __init__(self, in_features: int, out_features: int, num_rings: int):
        super().__init__()
        self.amp_raw = nn.Parameter(torch.zeros(in_features, out_features, num_rings))
        self.phase = nn.Parameter(torch.zeros(in_features, out_features, num_rings))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        amplitude = torch.nn.functional.softplus(self.amp_raw)
        coeff = amplitude * torch.exp(1j * self.phase)
        summed = torch.einsum("bin,ion->bo", fields, coeff)
        return summed.abs() ** 2


class PhotonicTorchBasis(nn.Module):
    """Photonic basis layer built from analytic microring resonators."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_rings: int = 8,
        wl_nm_range: Tuple[float, float] = (1546.0, 1554.0),
        phase_offsets: Iterable[float] | None = None,
        mixing: Literal["power", "coherent"] = "power",
        residual: bool = True,
    ):
        super().__init__()
        if num_rings <= 0:
            raise ValueError("num_rings must be positive.")

        wl_min_nm, wl_max_nm = wl_nm_range
        if wl_max_nm <= wl_min_nm:
            raise ValueError("wl_max_nm must exceed wl_min_nm.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_rings = num_rings
        self.mixing = mixing
        self.enable_residual = residual

        if phase_offsets is None:
            phase_offsets = torch.linspace(-math.pi, math.pi, num_rings)
        else:
            phase_offsets = torch.tensor(list(phase_offsets), dtype=torch.float32)
            if phase_offsets.numel() != num_rings:
                raise ValueError("phase_offsets length must match num_rings.")

        rings = []
        for offset in phase_offsets:
            spec = RingSpec(phase_offset=float(offset))
            rings.append(AnalyticRing(spec))
        self.rings = nn.ModuleList(rings)

        self.register_buffer("wl_min", torch.tensor(wl_min_nm * 1e-9, dtype=torch.float32))
        self.register_buffer("wl_span", torch.tensor((wl_max_nm - wl_min_nm) * 1e-9, dtype=torch.float32))

        if mixing == "power":
            self.mixer = PhotonicMixerPower(in_features, out_features, num_rings)
        elif mixing == "coherent":
            self.mixer = PhotonicMixerCoherent(in_features, out_features, num_rings)
        else:
            raise ValueError("mixing must be 'power' or 'coherent'.")

        if residual:
            self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)
        if self.mixing == "power":
            out = self.mixer(basis)
        else:
            out = self.mixer(basis)

        if self.enable_residual:
            out = out + x @ self.base_weight

        kl = torch.zeros(1, device=x.device, dtype=x.dtype)
        return out, kl

    def _evaluate_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_features = x.shape
        wl = self.wl_min.to(x.device, x.dtype) + 0.5 * (torch.clamp(x, -1.0, 1.0) + 1.0) * self.wl_span.to(x.device, x.dtype)
        wl = wl.reshape(-1)

        responses = []
        for ring in self.rings:
            transfer = ring(wl)
            if self.mixing == "power":
                responses.append(transfer.abs() ** 2)
            else:
                responses.append(transfer)

        stacked = torch.stack(responses, dim=0)
        stacked = stacked.view(self.num_rings, batch, in_features).permute(1, 2, 0).contiguous()

        if self.mixing == "power":
            mean = stacked.mean(dim=-1, keepdim=True)
            return (stacked - mean)
        return stacked
