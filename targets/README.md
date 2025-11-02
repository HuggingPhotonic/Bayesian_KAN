# KAN Test Functions

This module contains PyTorch implementations of test functions from the KAN paper (arxiv:2404.19756).

## Overview

The functions are organized into several categories:

- **Toy Functions**: Simple test cases from Section 3.1 of the paper
- **Feynman Functions**: Physics formulas from Section 3.3
- **PDE Functions**: Partial differential equation solutions from Section 3.4
- **Additional Functions**: Other example functions

## Usage

### Basic Function Evaluation

```python
import torch
from targets.functions import get_target_function

# Get a 2D function
func = get_target_function(dim=2, name="feynman_I_18_4")

# Evaluate on some inputs
x = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
y = func(x)
print(y)  # Output: tensor([50., 400.])
```

### Generate Training Data

```python
from targets.functions import generate_data

# Generate 1000 samples for a function
X, y = generate_data(
    func_name="exp_sin_toy",
    n_samples=1000,
    noise_level=0.01,  # Add 1% noise
    seed=42,
    device='cpu'
)

print(X.shape)  # torch.Size([1000, 3])
print(y.shape)  # torch.Size([1000])
```

### List Available Functions

```python
from targets.functions import list_functions

# List all functions
all_funcs = list_functions()

# List functions by dimension
funcs_2d = list_functions(dim=2)

# List functions by category
toy_funcs = list_functions(category='toy')
feynman_funcs = list_functions(category='feynman')
pde_funcs = list_functions(category='pde')
```

### Get Function Metadata

```python
from targets.functions import get_function_info

info = get_function_info("exp_sin_toy")
print(info)
# {
#     'n_vars': 3,
#     'domain': [[-1, 1], [-1, 1], [-1, 1]],
#     'description': 'exp(sin(πx₁) + x₂² + x₃)',
#     'true_shape': [3, 2, 1]
# }
```

## Available Functions

### Toy Dataset Functions (Section 3.1)

| Name | Dim | Formula | Domain | Description |
|------|-----|---------|--------|-------------|
| `bessel_toy` | 1D | J₀(20x) | [-1, 1] | Bessel function approximation |
| `exp_sin_toy` | 3D | exp(sin(πx₁) + x₂² + x₃) | [-1, 1]³ | Exponential with trigonometric components |
| `sin_quadratic_toy` | 3D | sin(π/2(x₁²+x₂²)) + x₃²(x₁²+x₂²) | [-1, 1]³ | Sine and quadratic composition |
| `high_dim_toy` | 4D | exp((sin(πx₁)+sin(πx₂))/2 + (sin(πx₃)+sin(πx₄))/2) | [-1, 1]⁴ | High-dimensional trigonometric |
| `sqrt_composition` | 3D | √(x₁² + x₂² + x₃²) | [0, 1]³ | Euclidean norm |

### Feynman Dataset Functions (Section 3.3)

| Name | Dim | Formula | Description |
|------|-----|---------|-------------|
| `feynman_I_6_2` | 1D | exp(-θ²/2)/√(2π) | Gaussian distribution |
| `feynman_I_6_2b` | 3D | exp(-(θ-θ₁)²/2σ²)/√(2πσ²) | Gaussian with mean |
| `feynman_I_12_11` | 3D | q₁q₂r/(4πε₀r³) | Coulomb force |
| `feynman_I_13_12` | 3D | Gm₁m₂/r² | Gravitational force |
| `feynman_I_15_3x` | 3D | (x-ut)/√(1-u²/c²) | Lorentz transformation |
| `feynman_I_18_4` | 2D | mv²/2 | Kinetic energy |
| `feynman_I_27_6` | 2D | 1/(exp(ℏω/kT)-1) | Planck distribution |
| `feynman_I_30_5` | 2D | arcsin(n·sin(θ)) | Snell's law |
| `feynman_I_37_4` | 3D | I₁+I₂+2√(I₁I₂)cos(δ) | Interference pattern |
| `feynman_I_40_1` | 4D | n₀exp(-mgx/kT) | Barometric formula |
| `feynman_I_44_4` | 4D | nk_bT·ln(V₂/V₁) | Entropy change |
| `relativistic_velocity` | 2D | (u+v)/(1+uv/c²) | Relativistic velocity addition |

### PDE Functions (Section 3.4)

| Name | Dim | Formula | Description |
|------|-----|---------|-------------|
| `poisson_2d_solution` | 2D | sin(πx)sin(πy) | Poisson equation solution |
| `poisson_2d_source` | 2D | 2π²sin(πx)sin(πy) | Poisson source term |

### Additional Functions

| Name | Dim | Formula | Description |
|------|-----|---------|-------------|
| `multiplication_via_log` | 2D | exp(ln(x₁) + ln(x₂)) | Multiplication via logarithm |
| `division_via_log` | 2D | exp(ln(x₁) - ln(x₂)) | Division via logarithm |
| `phase_transition` | 4D | tanh(Σx_i² - 2) | Phase transition function |

### Original Functions

| Name | Dim | Description |
|------|-----|-------------|
| `sin_cos` | 1D | sin(3x) + 0.3cos(10x) |
| `wave` | 2D | Complex 2D wave pattern |

## Integration with KAN Models

These functions can be used directly with the KAN models in this repository:

```python
from models.kan import BayesianKAN
from targets.functions import generate_data

# Generate training data
X_train, y_train = generate_data("exp_sin_toy", n_samples=1000)

# Create and train KAN model
model = BayesianKAN(
    input_dim=3,
    hidden_dims=[5, 3],
    output_dim=1,
    spline_order=3,
    num_knots=10
)

# Train the model
# ... (training code)
```

## Notes

- All functions are implemented in PyTorch and support automatic differentiation
- The Bessel function `bessel_toy` uses a series approximation since PyTorch doesn't have native Bessel functions
- For special functions (elliptic integrals, spherical harmonics, etc.) that require scipy, use the NumPy version or implement custom approximations
- Domains are specified in `FUNCTION_INFO` and are used by `generate_data()` to sample inputs

## Testing

Run the test script to verify all functions work correctly:

```bash
python test_functions.py
```

## Reference

Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
