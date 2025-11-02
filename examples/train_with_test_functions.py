"""
Example: Training KAN on Test Functions
Demonstrates how to use the test functions with KAN models
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from targets.functions import generate_data, get_function_info, list_functions
from models.kan import build_kan


def train_simple_model(func_name: str, n_epochs: int = 100, plot: bool = True):
    """
    Train a simple KAN model on a test function

    Args:
        func_name: Name of the test function
        n_epochs: Number of training epochs
        plot: Whether to plot results
    """
    print(f"\n{'='*70}")
    print(f"Training KAN on: {func_name}")
    print(f"{'='*70}")

    # Get function info
    info = get_function_info(func_name)
    print(f"Description: {info['description']}")
    print(f"Variables: {info['n_vars']}")
    print(f"Domain: {info['domain']}")

    # Generate training data
    X_train, y_train = generate_data(
        func_name=func_name,
        n_samples=1000,
        noise_level=0.01,
        seed=42
    )

    # Generate test data
    X_test, y_test = generate_data(
        func_name=func_name,
        n_samples=200,
        noise_level=0.0,
        seed=123
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Create model
    input_dim = info['n_vars']

    # Use suggested shape if available
    if 'true_shape' in info:
        hidden_dims = info['true_shape'][1:-1] if len(info['true_shape']) > 2 else [5]
    else:
        hidden_dims = [5, 3]

    # Build layer sizes: hidden_dims + [output_dim]
    layer_sizes = hidden_dims + [1]

    print(f"Model architecture: [{input_dim}] -> {layer_sizes}")

    model = build_kan(
        input_dim=input_dim,
        basis_name='bspline',
        layer_sizes=layer_sizes,
        variational=False,
        grid_size=10,
        spline_order=3
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    test_losses = []

    print("\nTraining...")
    for epoch in range(n_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()

        y_pred_train, kl = model(X_train, sample=False)
        y_pred_train = y_pred_train.squeeze()
        loss_train = criterion(y_pred_train, y_train)

        loss_train.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            y_pred_test, _ = model(X_test, sample=False)
            y_pred_test = y_pred_test.squeeze()
            loss_test = criterion(y_pred_test, y_test)

        train_losses.append(loss_train.item())
        test_losses.append(loss_test.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {loss_train.item():.6f}, "
                  f"Test Loss: {loss_test.item():.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test, _ = model(X_test, sample=False)
        y_pred_test = y_pred_test.squeeze()
        final_loss = criterion(y_pred_test, y_test)

        # Calculate R² score
        ss_tot = torch.sum((y_test - y_test.mean())**2)
        ss_res = torch.sum((y_test - y_pred_test)**2)
        r2_score = 1 - ss_res / ss_tot

    print(f"\nFinal Results:")
    print(f"Test Loss (MSE): {final_loss.item():.6f}")
    print(f"R² Score: {r2_score.item():.6f}")

    # Plotting
    if plot and input_dim <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Training history
        axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
        axes[0].plot(test_losses, label='Test Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Plot 2: Predictions vs Actual
        if input_dim == 1:
            # Sort for better visualization
            sorted_idx = torch.argsort(X_test.squeeze())
            X_sorted = X_test[sorted_idx]
            y_sorted = y_test[sorted_idx]
            y_pred_sorted = y_pred_test[sorted_idx]

            axes[1].plot(X_sorted.numpy(), y_sorted.numpy(),
                        'b-', label='True Function', linewidth=2)
            axes[1].plot(X_sorted.numpy(), y_pred_sorted.numpy(),
                        'r--', label='KAN Prediction', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
        else:  # 2D case
            axes[1].scatter(y_test.numpy(), y_pred_test.numpy(), alpha=0.5)
            min_val = min(y_test.min(), y_pred_test.min()).item()
            max_val = max(y_test.max(), y_pred_test.max()).item()
            axes[1].plot([min_val, max_val], [min_val, max_val],
                        'r--', label='Perfect Prediction')
            axes[1].set_xlabel('True Values')
            axes[1].set_ylabel('Predicted Values')

        axes[1].set_title(f'Predictions (R²={r2_score.item():.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'KAN Training on {func_name}')
        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent.parent / 'results' / 'test_functions'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{func_name}_training.png', dpi=150)
        print(f"Plot saved to: {output_dir / f'{func_name}_training.png'}")

        plt.show()

    return model, train_losses, test_losses


def compare_functions():
    """Compare KAN performance on multiple test functions"""
    print("\n" + "="*70)
    print("Comparing KAN Performance on Multiple Functions")
    print("="*70)

    # Select a few representative functions
    test_functions = [
        "exp_sin_toy",
        "feynman_I_6_2",
        "feynman_I_18_4",
        "sqrt_composition"
    ]

    results = {}

    for func_name in test_functions:
        try:
            model, train_losses, test_losses = train_simple_model(
                func_name,
                n_epochs=100,
                plot=False
            )
            results[func_name] = {
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1],
                'model': model
            }
        except Exception as e:
            print(f"Error training on {func_name}: {e}")

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Function':<30} {'Train Loss':<15} {'Test Loss':<15}")
    print("-"*70)
    for func_name, res in results.items():
        print(f"{func_name:<30} {res['final_train_loss']:<15.6f} {res['final_test_loss']:<15.6f}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train KAN on test functions')
    parser.add_argument('--function', type=str, default='exp_sin_toy',
                      help='Test function name')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--list', action='store_true',
                      help='List all available functions')
    parser.add_argument('--compare', action='store_true',
                      help='Compare multiple functions')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Test Functions:")
        print("="*70)
        for category in ['toy', 'feynman', 'pde', 'other']:
            funcs = list_functions(category=category)
            if funcs:
                print(f"\n{category.upper()}:")
                for dim, name in funcs:
                    info = get_function_info(name)
                    print(f"  {name:30s} ({dim}D) - {info['description']}")
        return

    if args.compare:
        compare_functions()
    else:
        train_simple_model(args.function, args.epochs, plot=True)


if __name__ == "__main__":
    main()
