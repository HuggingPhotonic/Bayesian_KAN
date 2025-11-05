"""
Photontorch-backed basis functions for photonic Kolmogorov–Arnold Networks.

This module builds a single-bus (all-pass) microring resonator using
Photontorch components and exposes it as a differentiable basis layer that
mimics the B-spline role in classical KANs.  The only trainable parameters are
the linear combination weights that mix multiple ring responses; all device
physics (ring radius, coupling, loss, etc.) are kept as fixed buffers.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import photontorch as pt


class AllPassRing(pt.Network):
    """
    Single-bus all-pass microring resonator constructed with Photontorch.
    """

    def __init__(
        self,
        *,
        R_um: float,
        neff: float,
        ng: float,
        loss_dB_cm: float,
        kappa: float,
    ):
        super().__init__()
        length = 2 * math.pi * (R_um * 1e-6)
        self.wg = pt.Waveguide(length=length, neff=neff, ng=ng, loss=loss_dB_cm, trainable=True)
        self.dc = pt.DirectionalCoupler(coupling=kappa, trainable=True)

        # Wiring: dc:2 -> wg -> dc:3 closes the ring.
        self.link("dc:2", "0:wg:1", "3:dc")

        # External ports: source feeds dc port 0, detector reads dc port 1.
        self.src = pt.Source()
        self.det = pt.Detector()
        self.link("src:0", "0:dc:1", "0:det")

    def throughput(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Simulate power transmission for the provided wavelengths (in metres).
        """
        wl_cpu = wavelengths.detach().cpu()
        with pt.Environment(wl=wl_cpu, freqdomain=True):
            power = self(source=1.0, power=True)  # (n_wl, n_det=1, batch=1)
        squeezed = power.squeeze(-1).squeeze(-1)  # (n_wl,)
        return squeezed.to(wavelengths.device, dtype=wavelengths.dtype)


class PhotonicTorchBasis(nn.Module):
    """
    Basis layer that mixes multiple microring notch responses.

    Trainable parameters:
        * Photontorch waveguide / coupler settings (length, neff, ng, loss, coupling)
        * coeffs (or coeff_mean / coeff_log_var when variational=True)
        * base_weight (global linear skip connection)

    Fixed buffers:
        * Pre-computed phase offsets realised via effective-index shifts
        * Wavelength normalisation bounds
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
        super().__init__()
        if num_rings <= 0:
            raise ValueError("num_rings must be positive.")
        wl_min_nm, wl_max_nm = wl_nm_range
        if wl_max_nm <= wl_min_nm:
            raise ValueError("wl_max_nm must exceed wl_min_nm.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_rings = num_rings
        self.variational = variational

        ring_length = 2 * math.pi * (R_um * 1e-6)
        center_wl = 0.5 * (wl_min_nm + wl_max_nm) * 1e-9

        if phase_offsets is None:
            offsets = torch.linspace(-math.pi, math.pi, num_rings)
        else:
            offsets = torch.tensor(list(phase_offsets), dtype=torch.float32)
            if offsets.numel() != num_rings:
                raise ValueError("phase_offsets length must match num_rings.")
        self.register_buffer("phase_offsets", offsets)

        # Convert phase offsets into effective index perturbations at centre λ.
        delta_neff = offsets.double() * center_wl / (2 * math.pi * ring_length)
        base_neff = torch.full((num_rings,), float(neff), dtype=torch.float64)
        neff_each = base_neff + delta_neff

        rings = []
        for neff_i in neff_each:
            ring = AllPassRing(
                R_um=R_um,
                neff=float(neff_i),
                ng=ng,
                loss_dB_cm=loss_dB_cm,
                kappa=kappa,
            )
            rings.append(ring)
        self.rings = nn.ModuleList(rings)

        self.register_buffer("wl_min", torch.tensor(wl_min_nm * 1e-9, dtype=torch.float32))
        self.register_buffer("wl_span", torch.tensor((wl_max_nm - wl_min_nm) * 1e-9, dtype=torch.float32))

        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, num_rings) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, num_rings), -5.0)
            )
            self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
            self.register_buffer(
                "prior_var",
                torch.ones_like(self.coeff_mean) * prior_scale ** 2,
            )
        else:
            self.coeffs = nn.Parameter(
                torch.randn(in_features, out_features, num_rings) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)

        if self.variational:
            if sample:
                std = torch.exp(0.5 * self.coeff_log_var)
                eps = torch.randn(n_samples, *std.shape, device=x.device, dtype=std.dtype)
                coeffs = self.coeff_mean + eps * std
                outputs = torch.einsum("bin,sion->sbo", basis, coeffs)
                basis_out = outputs.mean(dim=0)
            else:
                basis_out = torch.einsum("bin,ion->bo", basis, self.coeff_mean)
            kl = self._kl_divergence()
        else:
            basis_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
            kl = torch.zeros(1, device=x.device, dtype=x.dtype)

        residual = x @ self.base_weight
        return basis_out + residual, kl

    def _evaluate_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_features = x.shape
        dtype = x.dtype
        device = x.device

        wl = self.wl_min.to(device=device, dtype=dtype) + 0.5 * (torch.clamp(x, -1.0, 1.0) + 1.0) * self.wl_span.to(device=device, dtype=dtype)
        wl_flat = wl.reshape(-1)

        wl_env = wl_flat.to(dtype=torch.float32, device=torch.device("cpu"))
        if wl_flat.requires_grad:
            wl_env.requires_grad_()

        responses = []
        with pt.Environment(wl=wl_env, freqdomain=True):
            for ring in self.rings:
                power = ring(source=torch.tensor(1.0, dtype=wl_env.dtype, device=wl_env.device, requires_grad=False), power=True)
                notch = 1.0 - power.squeeze(-1).squeeze(-1)
                responses.append(notch)

        stacked = torch.stack(responses, dim=0)  # (num_rings, n_wl)
        stacked = stacked.to(device=device, dtype=dtype)
        basis = stacked.view(self.num_rings, batch, in_features).permute(1, 2, 0).contiguous()
        basis = torch.clamp(basis, min=0.0)
        return basis

    def _kl_divergence(self) -> torch.Tensor:
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl

