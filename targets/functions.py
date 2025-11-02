"""
Collection of analytical target functions used to benchmark KAN models.

The module provides:
    * A registry mapping (dimension, name) -> callable
    * Metadata describing domains and textual descriptions
    * Utilities for listing and sampling from the registered functions
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple, List, Optional, Sequence

import torch


# ==================== Toy Dataset Functions ====================
# Functions from Section 3.1 of the paper

def target_1d_sin_cos(x: torch.Tensor) -> torch.Tensor:
    """Original 1D toy function: sin(3x) + 0.3*cos(10x)"""
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


def target_2d_wave(x: torch.Tensor) -> torch.Tensor:
    """Original 2D wave function with multiple components"""
    # x: (batch, 2)
    xi, yi = x[:, 0:1], x[:, 1:2]
    return (torch.sin(torch.pi * xi) * torch.cos(torch.pi * yi) +
            0.3 * torch.exp(-(xi ** 2 + yi ** 2)) +
            0.2 * xi * yi +
            0.1 * torch.sin(3 * xi) * torch.sin(3 * yi))


def bessel_toy(x: torch.Tensor) -> torch.Tensor:
    """
    Bessel function J_0(20x) toy example
    Note: PyTorch doesn't have native Bessel functions, so we use series approximation
    Domain: [-1, 1], Shape: [2, 1]
    """
    # Use series approximation for J_0
    z = 20 * x.squeeze(-1) if x.dim() > 1 else 20 * x
    # J_0(z) ≈ 1 - z²/4 + z⁴/64 - z⁶/2304 (first few terms)
    z2 = z ** 2
    result = 1 - z2/4 + z2**2/64 - z2**3/2304
    return result.unsqueeze(-1) if x.dim() > 1 else result


def exp_sin_toy(x: torch.Tensor) -> torch.Tensor:
    """
    exp(sin(πx₁) + x₂² + x₃)
    Domain: [-1, 1]³, Shape: [3, 2, 1]
    """
    return torch.exp(
        torch.sin(math.pi * x[:, 0]) + x[:, 1]**2 + x[:, 2]
    )


def sin_quadratic_toy(x: torch.Tensor) -> torch.Tensor:
    """
    sin(π/2(x₁²+x₂²)) + x₃²(x₁²+x₂²)
    Domain: [-1, 1]³, Shape: [3, 3, 2, 1]
    """
    r_sq = x[:, 0]**2 + x[:, 1]**2
    return torch.sin(math.pi/2 * r_sq) + x[:, 2]**2 * r_sq


def high_dim_toy(x: torch.Tensor) -> torch.Tensor:
    """
    High-dimensional: exp((sin(πx₁)+sin(πx₂))/2 + (sin(πx₃)+sin(πx₄))/2)
    Domain: [-1, 1]⁴, Shape: [4, 3, 2, 1]
    """
    term1 = (torch.sin(math.pi * x[:, 0]) + torch.sin(math.pi * x[:, 1])) / 2
    term2 = (torch.sin(math.pi * x[:, 2]) + torch.sin(math.pi * x[:, 3])) / 2
    return torch.exp(term1 + term2)


def sqrt_composition(x: torch.Tensor) -> torch.Tensor:
    """
    sqrt(x₁² + x₂² + x₃²) - Euclidean norm
    Domain: [0, 1]³, Shape: [3, 2, 2, 1]
    """
    return torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)


# ==================== Feynman Dataset Functions ====================
# Functions from Section 3.3 of the paper

def feynman_I_6_2(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian: exp(-θ²/2)/√(2π)
    Domain: [-3, 3], Formula: exp(-theta²/2)/sqrt(2*pi)
    """
    return torch.exp(-x[:, 0]**2 / 2) / math.sqrt(2 * math.pi)


def feynman_I_6_2b(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian with mean: exp(-(θ-θ₁)²/2σ²)/√(2πσ²)
    Domain: [-3, 3]², [0.5, 2]
    """
    return torch.exp(-(x[:, 0] - x[:, 1])**2 / (2 * x[:, 2]**2)) / torch.sqrt(2 * math.pi * x[:, 2]**2)


def feynman_I_12_11(x: torch.Tensor) -> torch.Tensor:
    """
    Coulomb force: q₁q₂r/(4πε₀r³)
    Domain: [1e-9, 1e-6]², [1e-10, 1e-8]
    """
    epsilon_0 = 8.854e-12
    return x[:, 0] * x[:, 1] * x[:, 2] / (4 * math.pi * epsilon_0 * x[:, 2]**3)


def feynman_I_13_12(x: torch.Tensor) -> torch.Tensor:
    """
    Gravitational force: Gm₁m₂/r²
    Domain: [1e20, 1e30]², [1e8, 1e12]
    """
    G = 6.674e-11
    return G * x[:, 0] * x[:, 1] / x[:, 2]**2


def feynman_I_15_3x(x: torch.Tensor) -> torch.Tensor:
    """
    Lorentz transformation: (x-ut)/√(1-u²/c²)
    Domain: [0, 1e6]², [0, 100]
    """
    c = 3e8
    return (x[:, 0] - x[:, 1] * x[:, 2]) / torch.sqrt(1 - x[:, 1]**2 / c**2)


def feynman_I_18_4(x: torch.Tensor) -> torch.Tensor:
    """
    Kinetic energy: mv²/2
    Domain: [0.1, 10], [0.1, 100]
    """
    return x[:, 0] * x[:, 1]**2 / 2


def feynman_I_27_6(x: torch.Tensor) -> torch.Tensor:
    """
    Planck distribution: 1/(exp(ℏω/kT)-1)
    Domain: [0.1, 5], [0.5, 3]
    """
    return 1 / (torch.exp(x[:, 0] / x[:, 1]) - 1)


def feynman_I_30_5(x: torch.Tensor) -> torch.Tensor:
    """
    Snell's law: arcsin(n·sin(θ))
    Domain: [0.5, 0.9], [0, π/4]
    """
    return torch.asin(x[:, 0] * torch.sin(x[:, 1]))


def feynman_I_37_4(x: torch.Tensor) -> torch.Tensor:
    """
    Interference pattern: I₁+I₂+2√(I₁I₂)cos(δ)
    Domain: [0.1, 5]², [0, 2π]
    """
    return x[:, 0] + x[:, 1] + 2 * torch.sqrt(x[:, 0] * x[:, 1]) * torch.cos(x[:, 2])


def feynman_I_40_1(x: torch.Tensor) -> torch.Tensor:
    """
    Barometric formula: n₀exp(-mgx/kT)
    Domain: [1e20, 1e25], [1e-27, 1e-25], [0, 1000], [200, 400]
    """
    k_b = 1.38e-23
    g = 9.8
    return x[:, 0] * torch.exp(-x[:, 1] * g * x[:, 2] / (k_b * x[:, 3]))


def feynman_I_44_4(x: torch.Tensor) -> torch.Tensor:
    """
    Entropy change: nk_bT·ln(V₂/V₁)
    Domain: [1e20, 1e24], [200, 400], [1, 10], [0.1, 1]
    """
    k_b = 1.38e-23
    return x[:, 0] * k_b * x[:, 1] * torch.log(x[:, 2] / x[:, 3])


def relativistic_velocity(x: torch.Tensor) -> torch.Tensor:
    """
    Relativistic velocity addition: (u+v)/(1+uv/c²)
    Domain: [0, 2e8]²
    """
    c = 3e8
    return (x[:, 0] + x[:, 1]) / (1 + x[:, 0] * x[:, 1] / c**2)


# ==================== PDE Related Functions ====================
# Functions from Section 3.4 of the paper

def poisson_2d_solution(x: torch.Tensor) -> torch.Tensor:
    """
    Poisson 2D solution: sin(πx)sin(πy)
    Domain: [-1, 1]², PDE: -Δu = 2π²sin(πx)sin(πy)
    """
    return torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1])


def poisson_2d_source(x: torch.Tensor) -> torch.Tensor:
    """
    Poisson 2D source term: 2π²sin(πx)sin(πy)
    Domain: [-1, 1]², Source term for -Δu = f
    """
    return 2 * math.pi**2 * torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1])


# ==================== Additional Example Functions ====================

def multiplication_via_log(x: torch.Tensor) -> torch.Tensor:
    """
    Multiplication via logarithm: x₁ × x₂ = exp(ln(x₁) + ln(x₂))
    Domain: positive reals
    """
    return torch.exp(torch.log(torch.abs(x[:, 0]) + 1e-10) + torch.log(torch.abs(x[:, 1]) + 1e-10))


def division_via_log(x: torch.Tensor) -> torch.Tensor:
    """
    Division via logarithm: x₁ / x₂ = exp(ln(x₁) - ln(x₂))
    Domain: positive reals
    """
    return torch.exp(torch.log(torch.abs(x[:, 0]) + 1e-10) - torch.log(torch.abs(x[:, 1]) + 1e-10))


def phase_transition(x: torch.Tensor) -> torch.Tensor:
    """
    Phase transition function: tanh((x₁² + x₂² + x₃² + x₄²) - 2)
    Domain: [-2, 2]⁴
    """
    order_param = x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 + x[:, 3]**2
    return torch.tanh(order_param - 2)


# ==================== Function Registry ====================

TARGETS: Dict[Tuple[int, str], Callable[[torch.Tensor], torch.Tensor]] = {
    # Original functions
    (1, "sin_cos"): target_1d_sin_cos,
    (2, "wave"): target_2d_wave,

    # Toy dataset functions
    (1, "bessel_toy"): bessel_toy,
    (3, "exp_sin_toy"): exp_sin_toy,
    (3, "sin_quadratic_toy"): sin_quadratic_toy,
    (4, "high_dim_toy"): high_dim_toy,
    (3, "sqrt_composition"): sqrt_composition,

    # Feynman dataset functions
    (1, "feynman_I_6_2"): feynman_I_6_2,
    (3, "feynman_I_6_2b"): feynman_I_6_2b,
    (3, "feynman_I_12_11"): feynman_I_12_11,
    (3, "feynman_I_13_12"): feynman_I_13_12,
    (3, "feynman_I_15_3x"): feynman_I_15_3x,
    (2, "feynman_I_18_4"): feynman_I_18_4,
    (2, "feynman_I_27_6"): feynman_I_27_6,
    (2, "feynman_I_30_5"): feynman_I_30_5,
    (3, "feynman_I_37_4"): feynman_I_37_4,
    (4, "feynman_I_40_1"): feynman_I_40_1,
    (4, "feynman_I_44_4"): feynman_I_44_4,
    (2, "relativistic_velocity"): relativistic_velocity,

    # PDE related functions
    (2, "poisson_2d_solution"): poisson_2d_solution,
    (2, "poisson_2d_source"): poisson_2d_source,

    # Additional examples
    (2, "multiplication_via_log"): multiplication_via_log,
    (2, "division_via_log"): division_via_log,
    (4, "phase_transition"): phase_transition,
}


# Function metadata for data generation
FUNCTION_INFO: Dict[str, Dict] = {
    # Toy functions
    "bessel_toy": {
        "n_vars": 1,
        "domain": [[-1, 1]],
        "description": "Bessel function J_0(20x)",
        "true_shape": [2, 1]
    },
    "exp_sin_toy": {
        "n_vars": 3,
        "domain": [[-1, 1]] * 3,
        "description": "exp(sin(πx₁) + x₂² + x₃)",
        "true_shape": [3, 2, 1]
    },
    "sin_quadratic_toy": {
        "n_vars": 3,
        "domain": [[-1, 1]] * 3,
        "description": "sin(π/2(x₁²+x₂²)) + x₃²(x₁²+x₂²)",
        "true_shape": [3, 3, 2, 1]
    },
    "high_dim_toy": {
        "n_vars": 4,
        "domain": [[-1, 1]] * 4,
        "description": "High-dim: exp((sin(πx₁)+sin(πx₂))/2 + (sin(πx₃)+sin(πx₄))/2)",
        "true_shape": [4, 3, 2, 1]
    },
    "sqrt_composition": {
        "n_vars": 3,
        "domain": [[0, 1]] * 3,
        "description": "sqrt(x₁² + x₂² + x₃²)",
        "true_shape": [3, 2, 2, 1]
    },

    # Feynman functions
    "feynman_I_6_2": {
        "n_vars": 1,
        "domain": [[-3, 3]],
        "description": "Gaussian: exp(-θ²/2)/√(2π)",
        "formula": "exp(-theta²/2)/sqrt(2*pi)"
    },
    "feynman_I_6_2b": {
        "n_vars": 3,
        "domain": [[-3, 3], [-3, 3], [0.5, 2]],
        "description": "Gaussian with mean: exp(-(θ-θ₁)²/2σ²)/√(2πσ²)",
        "formula": "exp(-(theta-theta1)²/(2*sigma²))/sqrt(2*pi*sigma²)"
    },
    "feynman_I_12_11": {
        "n_vars": 3,
        "domain": [[1e-9, 1e-6], [1e-9, 1e-6], [1e-10, 1e-8]],
        "description": "Coulomb force: q₁q₂r/(4πε₀r³)",
        "formula": "q1*q2*r/(4*pi*epsilon*r³)"
    },
    "feynman_I_13_12": {
        "n_vars": 3,
        "domain": [[1e20, 1e30], [1e20, 1e30], [1e8, 1e12]],
        "description": "Gravitational force: Gm₁m₂/r²",
        "formula": "G*m1*m2/r²"
    },
    "feynman_I_15_3x": {
        "n_vars": 3,
        "domain": [[0, 1e6], [0, 1e6], [0, 100]],
        "description": "Lorentz transformation: (x-ut)/√(1-u²/c²)",
        "formula": "(x-u*t)/sqrt(1-u²/c²)"
    },
    "feynman_I_18_4": {
        "n_vars": 2,
        "domain": [[0.1, 10], [0.1, 100]],
        "description": "Kinetic energy: mv²/2",
        "formula": "m*v²/2"
    },
    "feynman_I_27_6": {
        "n_vars": 2,
        "domain": [[0.1, 5], [0.5, 3]],
        "description": "Planck distribution: 1/(exp(ℏω/kT)-1)",
        "formula": "1/(exp(h*omega/(k*T))-1)"
    },
    "feynman_I_30_5": {
        "n_vars": 2,
        "domain": [[0.5, 0.9], [0, math.pi/4]],
        "description": "Snell's law: arcsin(n·sin(θ))",
        "formula": "arcsin(n*sin(theta))"
    },
    "feynman_I_37_4": {
        "n_vars": 3,
        "domain": [[0.1, 5], [0.1, 5], [0, 2*math.pi]],
        "description": "Interference pattern: I₁+I₂+2√(I₁I₂)cos(δ)",
        "formula": "I1+I2+2*sqrt(I1*I2)*cos(delta)"
    },
    "feynman_I_40_1": {
        "n_vars": 4,
        "domain": [[1e20, 1e25], [1e-27, 1e-25], [0, 1000], [200, 400]],
        "description": "Barometric formula: n₀exp(-mgx/kT)",
        "formula": "n0*exp(-m*g*x/(k*T))"
    },
    "feynman_I_44_4": {
        "n_vars": 4,
        "domain": [[1e20, 1e24], [200, 400], [1, 10], [0.1, 1]],
        "description": "Entropy change: nk_bT·ln(V₂/V₁)",
        "formula": "n*k_b*T*ln(V2/V1)"
    },
    "relativistic_velocity": {
        "n_vars": 2,
        "domain": [[0, 2e8], [0, 2e8]],
        "description": "Relativistic velocity: (u+v)/(1+uv/c²)",
        "formula": "(u+v)/(1+u*v/c²)"
    },

    # PDE functions
    "poisson_2d_solution": {
        "n_vars": 2,
        "domain": [[-1, 1], [-1, 1]],
        "description": "Poisson 2D solution: sin(πx)sin(πy)",
        "pde": "-Δu = 2π²sin(πx)sin(πy)"
    },
    "poisson_2d_source": {
        "n_vars": 2,
        "domain": [[-1, 1], [-1, 1]],
        "description": "Poisson 2D source: 2π²sin(πx)sin(πy)",
        "pde": "Source term for -Δu = f"
    },

    # Additional functions
    "sin_cos": {
        "n_vars": 1,
        "domain": [[-1, 1]],
        "description": "Original 1D toy: sin(3x) + 0.3cos(10x)"
    },
    "wave": {
        "n_vars": 2,
        "domain": [[-1, 1], [-1, 1]],
        "description": "Original 2D wave function with multiple components"
    },
    "multiplication_via_log": {
        "n_vars": 2,
        "domain": [[0.1, 10], [0.1, 10]],
        "description": "Multiplication via logarithm: exp(ln(x₁) + ln(x₂))"
    },
    "division_via_log": {
        "n_vars": 2,
        "domain": [[0.1, 10], [0.1, 10]],
        "description": "Division via logarithm: exp(ln(x₁) - ln(x₂))"
    },
    "phase_transition": {
        "n_vars": 4,
        "domain": [[-2, 2]] * 4,
        "description": "Phase transition: tanh((x₁² + x₂² + x₃² + x₄²) - 2)"
    }
}


def _available_targets() -> Dict[str, List[int]]:
    """Return mapping from function name to registered dimensions."""
    mapping: Dict[str, List[int]] = {}
    for (dim, name) in TARGETS:
        mapping.setdefault(name, []).append(dim)
    return {name: sorted(dims) for name, dims in mapping.items()}


TARGET_DIMS: Dict[str, List[int]] = _available_targets()
TARGETS_BY_NAME: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    name: TARGETS[(dims[0], name)]
    for name, dims in TARGET_DIMS.items()
    if len(dims) == 1
}


def get_target_function(dim: int, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Retrieve a target function by dimension and name."""
    key = (dim, name)
    if key in TARGETS:
        return TARGETS[key]
    if name in TARGET_DIMS:
        dims = ", ".join(str(d) for d in TARGET_DIMS[name])
        raise ValueError(
            f"Function '{name}' is registered for dimensions [{dims}] "
            f"but not for dim={dim}."
        )
    available = ", ".join(f"{d}D:{n}" for d, n in sorted(TARGETS))
    raise ValueError(
        f"Unknown target function for dim={dim}, name='{name}'. "
        f"Available entries: {available}"
    )


def resolve_target(name: str, dim: Optional[int] = None) -> Tuple[int, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Resolve a target function by name, optionally checking the requested dimension.

    Returns a tuple (dimension, callable).
    """
    if name not in TARGET_DIMS:
        available = ", ".join(sorted(TARGET_DIMS))
        raise ValueError(f"Unknown target '{name}'. Registered names: {available}")
    dims = TARGET_DIMS[name]
    if dim is None:
        dim = dims[0]
    if dim not in dims:
        dims_str = ", ".join(str(d) for d in dims)
        raise ValueError(f"Target '{name}' does not support dim={dim} (available: {dims_str}).")
    return dim, TARGETS[(dim, name)]


def get_function_info(name: str) -> Dict:
    """Return metadata about a target function."""
    if name not in FUNCTION_INFO:
        raise ValueError(f"No metadata available for function '{name}'.")
    return FUNCTION_INFO[name]


def list_functions(dim: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[int, str]]:
    """
    List registered targets, optionally filtering by dimension or category.
    """
    entries = list(TARGETS.keys())
    if dim is not None:
        entries = [(d, n) for d, n in entries if d == dim]

    if category is not None:
        category_filters = {
            "toy": lambda n: any(k in n for k in ("bessel_toy", "exp_sin", "sin_quadratic", "high_dim", "sqrt_composition")),
            "feynman": lambda n: "feynman" in n or "relativistic" in n,
            "pde": lambda n: "poisson" in n,
            "other": lambda n: n in {"sin_cos", "wave", "multiplication_via_log", "division_via_log", "phase_transition"},
        }
        predicate = category_filters.get(category)
        if predicate:
            entries = [(d, n) for d, n in entries if predicate(n)]

    return sorted(entries)


def _sample_uniform(domain: Sequence[Sequence[float]], n_samples: int, device: torch.device) -> torch.Tensor:
    lows = torch.tensor([interval[0] for interval in domain], device=device)
    highs = torch.tensor([interval[1] for interval in domain], device=device)
    return torch.rand(n_samples, len(domain), device=device) * (highs - lows) + lows


def generate_data(
    func_name: str,
    n_samples: int = 1000,
    noise_level: float = 0.0,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (X, y) pairs for the specified analytical function.
    """
    info = get_function_info(func_name)
    domain = info["domain"]
    device = torch.device(device)

    _, func = resolve_target(func_name, dim=len(domain))

    torch.manual_seed(seed)
    X = _sample_uniform(domain, n_samples, device=device)

    y = func(X)
    if y.dim() > 1:
        y = y.view(y.size(0), -1)
        if y.size(1) != 1:
            raise ValueError(
                f"Function '{func_name}' returns {y.size(1)} outputs; "
                "multi-output targets are not supported by the data generator."
            )
        y = y.squeeze(1)

    if noise_level > 0:
        y = y + torch.randn_like(y) * noise_level

    return X, y
