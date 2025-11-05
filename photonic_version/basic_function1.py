"""
Coherent photonic basis functions built from Photontorch microring resonators.

This module exposes a reusable ``MicroringBasisBase`` together with a coherent
variant that keeps the complex field response of each ring (encoded as real and
imaginary components).  The implementation mirrors the B-spline basis used in
classical KANs but replaces the analytical basis with an all-pass microring
transfer function simulated by Photontorch.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import photontorch as pt


class AllPassRing(pt.Network):
    """
    Single-bus (all-pass) microring resonator network.

    Parameters
    ----------
    R_um: float
        Ring radius in micrometres.
    neff: float
        Effective refractive index.
    ng: float
        Group index.
    loss_dB_cm: float
        Propagation loss in dB/cm.
    kappa: float
        Power coupling coefficient of the directional coupler (0..1).
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

        # Device building blocks – kept non-trainable so that only the mixing
        # coefficients of the basis layer are optimised during learning.
        self.wg = pt.Waveguide(
            length=length,
            neff=neff,
            ng=ng,
            loss=loss_dB_cm,
            trainable=False,
        )
        self.dc = pt.DirectionalCoupler(coupling=kappa, trainable=False)

        # Close the ring: dc:2 -> wg -> dc:3.
        self.link("dc:2", "0:wg:1", "3:dc")

        # External ports – source feeds port 0, detector reads port 1.
        self.src = pt.Source()
        self.det = pt.Detector()
        self.link("src:0", "0:dc:1", "0:det")

    def throughput(self, wavelengths: torch.Tensor, *, power: bool = True) -> torch.Tensor:
        """
        Simulate the microring response for the provided wavelengths (metres).

        Parameters
        ----------
        wavelengths: torch.Tensor
            1D tensor of wavelengths in metres.
        power: bool
            Whether to return optical power (True) or complex field (False).
        """
        # Photontorch expects CPU tensors; detach to avoid leaking graph.
        wl_cpu = wavelengths.detach().cpu()
        with pt.Environment(wl=wl_cpu, freqdomain=True):
            response = self(source=1.0, power=power)
        squeezed = response.squeeze(-1).squeeze(-1)
        return squeezed.to(wavelengths.device, dtype=wavelengths.dtype)


class MicroringBasisBase(nn.Module):
    """
    Base class for constructing differentiable microring bases.

    Subclasses customise how the raw Photontorch response is encoded by
    overriding :meth:`_ring_response` (and optionally
    :meth:`_postprocess_basis`).
    """

    response_channels: int = 1  # number of channels per ring after encoding

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
            torch.linspace(-math.pi, math.pi, num_rings)
            if phase_offsets is None
            else torch.tensor(list(phase_offsets), dtype=torch.float32)
        )
        if offsets.numel() != num_rings:
            raise ValueError("phase_offsets length must match num_rings.")
        self.register_buffer("phase_offsets", offsets)

        ring_length = 2 * math.pi * (R_um * 1e-6)
        center_wl = 0.5 * (wl_min_nm + wl_max_nm) * 1e-9

        # Translate desired phase offsets into effective-index perturbations.
        delta_neff = offsets.double() * center_wl / (2 * math.pi * ring_length)
        base_neff = torch.full((num_rings,), float(neff), dtype=torch.float64)
        neff_each = base_neff + delta_neff

        # Instantiate one ring per phase-offset (effective-index shift).
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

        # Store scaling terms that map neural activations to wavelengths.
        self.register_buffer(
            "wl_min",
            torch.tensor(wl_min_nm * 1e-9, dtype=torch.float32),
        )
        self.register_buffer(
            "wl_span",
            torch.tensor((wl_max_nm - wl_min_nm) * 1e-9, dtype=torch.float32),
        )

        # Each ring contributes `response_channels` features (coherent case = 2).
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

    # --------------------------------------------------------------------- #
    # Autograd interface
    # --------------------------------------------------------------------- #
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

        # Linear skip connection mirroring classical KAN residual branches.
        residual = x @ self.base_weight
        return basis_out + residual, kl

    # --------------------------------------------------------------------- #
    # Basis construction
    # --------------------------------------------------------------------- #
    def _evaluate_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_features = x.shape
        dtype = x.dtype
        device = x.device

        # Map inputs in [-1, 1] to physical wavelengths inside the sweep window.
        wl = self.wl_min.to(device=device, dtype=dtype) + 0.5 * (
            torch.clamp(x, -1.0, 1.0) + 1.0
        ) * self.wl_span.to(device=device, dtype=dtype)
        wl_flat = wl.reshape(-1)

        wl_env = wl_flat.to(dtype=torch.float32, device=torch.device("cpu"))
        if wl_flat.requires_grad:
            wl_env.requires_grad_()

        responses = self._simulate_rings(wl_env)
        responses = responses.to(device=device, dtype=dtype)

        # Reshape back to [batch, in_features, num_rings * response_channels].
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
        # Each entry in `responses` will be [num_wl, response_channels].
        responses = []
        with pt.Environment(wl=wavelengths, freqdomain=True):
            # Drive each wavelength sample independently with unit amplitude.
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
        """
        Return the per-wavelength response for a single ring.

        Subclasses must override this method.
        """

        raise NotImplementedError

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Optional hook for subclasses to modify the stacked basis features.
        """

        return basis

    # --------------------------------------------------------------------- #
    # Variational inference helpers
    # --------------------------------------------------------------------- #
    def _kl_divergence(self) -> torch.Tensor:
        # Closed-form KL divergence between two diagonal-covariance Gaussians.
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0
            - torch.log(var / self.prior_var)
        )
        return kl


class PhotonicCoherentBasis(MicroringBasisBase):
    """
    Coherent microring basis that keeps the complex field response.

    Each ring contributes two basis channels (real and imaginary parts),
    allowing subsequent linear combinations to emulate coherent interference.
    """

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

        # Directional couplers in Photontorch often return the real/imag pair
        # along the leading dimension even when dtype is real.
        if field.ndim >= 2 and field.shape[0] == 2:
            real, imag = field[0], field[1]
            return torch.stack((real, imag), dim=-1)

        # Photontorch may return purely real fields depending on component setup;
        # append a synthetic zero-imaginary channel to preserve the coherent layout.
        imag = torch.zeros_like(field)
        return torch.stack((field, imag), dim=-1)

    def _postprocess_basis(self, basis: torch.Tensor) -> torch.Tensor:
        # No clamping – coherent responses can take positive/negative values.
        return basis
