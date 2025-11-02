"""
Test script for KAN test functions
Verifies that all functions work correctly with PyTorch tensors
"""

import torch
from targets.functions import (
    list_functions,
    get_target_function,
    get_function_info,
    generate_data
)


def test_all_functions():
    """Test all registered functions"""
    print("=" * 70)
    print("Testing KAN Test Functions")
    print("=" * 70)

    # List all functions by category
    categories = ['toy', 'feynman', 'pde', 'other']

    for category in categories:
        print(f"\n{category.upper()} Functions:")
        funcs = list_functions(category=category)
        for dim, name in funcs:
            print(f"  - {dim}D: {name}")

    # Test data generation for a few functions
    print("\n" + "=" * 70)
    print("Testing Data Generation")
    print("=" * 70)

    test_functions = [
        "bessel_toy",
        "exp_sin_toy",
        "feynman_I_6_2",
        "feynman_I_18_4",
        "poisson_2d_solution"
    ]

    for func_name in test_functions:
        try:
            print(f"\n{func_name}:")
            info = get_function_info(func_name)
            print(f"  Description: {info['description']}")
            print(f"  Variables: {info['n_vars']}")
            print(f"  Domain: {info['domain']}")

            # Generate sample data
            X, y = generate_data(func_name, n_samples=10, seed=42)
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"  y range: [{y.min():.3e}, {y.max():.3e}]")
            print(f"  ✓ Success")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test function evaluation directly
    print("\n" + "=" * 70)
    print("Testing Direct Function Evaluation")
    print("=" * 70)

    # Test 1D function
    print("\n1D Function (bessel_toy):")
    func_1d = get_target_function(1, "bessel_toy")
    x_1d = torch.linspace(-1, 1, 5).unsqueeze(1)
    y_1d = func_1d(x_1d)
    print(f"  Input: {x_1d.squeeze().tolist()}")
    print(f"  Output: {y_1d.squeeze().tolist()}")

    # Test 2D function
    print("\n2D Function (feynman_I_18_4):")
    func_2d = get_target_function(2, "feynman_I_18_4")
    x_2d = torch.tensor([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])
    y_2d = func_2d(x_2d)
    print(f"  Input shape: {x_2d.shape}")
    print(f"  Output: {y_2d.tolist()}")
    print(f"  Expected (mv²/2): {(x_2d[:, 0] * x_2d[:, 1]**2 / 2).tolist()}")

    # Test 3D function
    print("\n3D Function (exp_sin_toy):")
    func_3d = get_target_function(3, "exp_sin_toy")
    x_3d = torch.rand(3, 3) * 2 - 1  # Random in [-1, 1]
    y_3d = func_3d(x_3d)
    print(f"  Input shape: {x_3d.shape}")
    print(f"  Output shape: {y_3d.shape}")
    print(f"  Output range: [{y_3d.min():.3f}, {y_3d.max():.3f}]")

    # Test 4D function
    print("\n4D Function (high_dim_toy):")
    func_4d = get_target_function(4, "high_dim_toy")
    x_4d = torch.rand(3, 4) * 2 - 1  # Random in [-1, 1]
    y_4d = func_4d(x_4d)
    print(f"  Input shape: {x_4d.shape}")
    print(f"  Output shape: {y_4d.shape}")
    print(f"  Output range: [{y_4d.min():.3f}, {y_4d.max():.3f}]")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_functions()
