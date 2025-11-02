from __future__ import annotations

from typing import Sequence
from pathlib import Path
import sys
import torch
import torch.nn as nn

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from bases import (
    ChebyshevBasis,
    WaveletBasis,
    RBFBasis,
    BSplineBasis,
)

BASIS_REGISTRY = {
    "chebyshev": ChebyshevBasis,
    "wavelet": WaveletBasis,
    "rbf": RBFBasis,
    "bspline": BSplineBasis,
}


class KAN(nn.Module):
    """Generic Kolmogorov–Arnold Network stacking basis layers."""

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, sample: bool = True, n_samples: int = 1):
        total_kl = torch.zeros(1, device=x.device)
        for layer in self.layers:
            x, kl = layer(x, sample=sample, n_samples=n_samples)
            total_kl = total_kl + kl.to(x.device)
        return x, total_kl


def build_kan(
    input_dim: int,
    basis_name: str,
    layer_sizes: Sequence[int],
    *,
    variational: bool = False,
    **basis_kwargs,
) -> KAN:
    """
    Build a KAN given basis name and hidden sizes.

    Parameters
    ----------
    input_dim: int
        Dimensionality of input features.
    basis_name: str
        One of 'chebyshev', 'wavelet', 'rbf', 'bspline'.
    layer_sizes: Sequence[int]
        Hidden/output sizes excluding input (e.g. [16, 16, 1]).
    variational: bool
        Whether to use variational coefficients (for VI).
    basis_kwargs:
        Extra kwargs forwarded to basis constructor.
    """
    try:
        basis_cls = BASIS_REGISTRY[basis_name]
    except KeyError as exc:
        raise ValueError(f"Unknown basis '{basis_name}'.") from exc

    sizes = [input_dim] + list(layer_sizes)
    layers = []
    for in_f, out_f in zip(sizes[:-1], sizes[1:]):
        layer = basis_cls(
            in_features=in_f,
            out_features=out_f,
            variational=variational,
            **basis_kwargs,
        )
        layers.append(layer)
    return KAN(layers)
