"""Common microring components reused by photonic basis modules.

This module hosts the Photontorch-based microring simulator and the
``MicroringBasisBase`` class so that both coherent and incoherent variants can
share a single implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import photontorch as pt


class AllPassRing(pt.Network):
    """Single-bus (all-pass) microring resonator network."""

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

        self.wg = pt.Waveguide(
            length=length,
            neff=neff,
            ng=ng,
            loss=loss_dB_cm,
            trainable=True,
        )
        self.dc = pt.DirectionalCoupler(coupling=kappa, trainable=True)

        self.link("dc:2", "0:wg:1", "3:dc")

        self.src = pt.Source()
        self.det = pt.Detector()
        self.link("src:0", "0:dc:1", "0:det")

    def throughput(self, wavelengths: torch.Tensor, *, power: bool = True) -> torch.Tensor:
        """Simulate the microring response for the provided wavelengths."""

        wl_cpu = wavelengths.detach().cpu()
        with pt.Environment(wl=wl_cpu, freqdomain=True):
            response = self(source=1.0, power=power)
        squeezed = response.squeeze(-1).squeeze(-1)
        return squeezed.to(wavelengths.device, dtype=wavelengths.dtype)


class MicroringBasisBase(nn.Module):
    """Base class for constructing differentiable microring bases."""

    response_channels: int = 1

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

        offsets = (
            torch.linspace(-math.pi, math.pi, num_rings, dtype=torch.float32)
            if phase_offsets is None
            else torch.tensor(list(phase_offsets), dtype=torch.float32)
        )
        if offsets.numel() != num_rings:
            raise ValueError("phase_offsets length must match num_rings.")
        self.register_buffer("phase_offsets", offsets)

        ring_length = 2 * math.pi * (R_um * 1e-6)
        center_wl = torch.tensor(
            0.5 * (wl_min_nm + wl_max_nm) * 1e-9, dtype=torch.float32
        )

        delta_neff = offsets * center_wl / (2 * math.pi * ring_length)
        base_neff = torch.full((num_rings,), float(neff), dtype=torch.float32)
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

        self.register_buffer(
            "wl_min",
            torch.tensor(wl_min_nm * 1e-9, dtype=torch.float32),
        )
        self.register_buffer(
            "wl_span",
            torch.tensor((wl_max_nm - wl_min_nm) * 1e-9, dtype=torch.float32),
        )

        self.num_basis = self.response_channels * num_rings

        if variational:
            self.coeff_mean = nn.Parameter(
                torch.randn(in_features, out_features, self.num_basis) * 0.05
            )
            self.coeff_log_var = nn.Parameter(
                torch.full((in_features, out_features, self.num_basis), -5.0)
            )
            self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
            self.register_buffer(
                "prior_var",
                torch.ones_like(self.coeff_mean) * prior_scale**2,
            )
        else:
            self.coeffs = nn.Parameter(
                torch.randn(in_features, out_features, self.num_basis) * 0.05
            )

        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        basis = self._evaluate_basis(x)

        if self.variational:
            if sample:
                std = torch.exp(0.5 * self.coeff_log_var)
                eps = torch.randn(
                    n_samples, *std.shape, device=x.device, dtype=std.dtype
                )
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

        wl = self.wl_min.to(device=device, dtype=dtype) + 0.5 * (
            torch.clamp(x, -1.0, 1.0) + 1.0
        ) * self.wl_span.to(device=device, dtype=dtype)
        wl_flat = wl.reshape(-1)

        wl_env = wl_flat.to(dtype=torch.float32, device=torch.device("cpu"))
        if wl_flat.requires_grad:
            wl_env.requires_grad_()

        responses = self._simulate_rings(wl_env)
        responses = responses.to(device=device, dtype=dtype)

        responses = responses.view(
            self.num_rings,
            batch,
            in_features,
            self.response_channels,
        )
        responses = responses.permute(1, 2, 0, 3).contiguous()
        responses = responses.view(batch, in_features, self.num_basis)
        return self._postprocess_basis(responses)

    def _simulate_rings(self, wavelengths: torch.Tensor) -> torch.Tensor:
        responses = []
        with pt.Environment(wl=wavelengths, freqdomain=True):
            source = torch.ones(
                (1, wavelengths.shape[0], 1, 1),
                dtype=wavelengths.dtype,
                device=wavelengths.device,
            ).refine_names("t", "w", "s", "b")
            for ring in self.rings:
                resp = self._ring_response(ring, wavelengths, source)
                responses.append(resp)
        return torch.stack(responses, dim=0)

    def _ring_response(
        self,
        ring: AllPassRing,
        wavelengths: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        return basis

    def _kl_divergence(self) -> torch.Tensor:
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl
