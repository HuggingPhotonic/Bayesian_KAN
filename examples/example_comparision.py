"""
Example: Comparing all Bayesian inference methods
VI, MCMC, and Laplace Approximation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from bayesian_kan import (
    BayesianKAN,
    VariationalInference,
    MCMCSampler,
    LaplaceApproximation,
    BayesianKANVisualizer,
    generate_synthetic_data,
    set_seed,
    create_data_loaders,
    compute_metrics,
    DEVICE
)


def train_vi(X_train, y_train, X_test, y_test):
    """Train model using Variational Inference"""
    print("\n" + "="*70)
    print("VARIATIONAL INFERENCE")
    print("="*70)
    
    model = BayesianKAN(
        layer_sizes=[1, 8, 8, 1],
        n_basis=6,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=32
    )
    
    history = VariationalInference.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=80,
        lr=1e-3,
        kl_weight=0.01
    )
    
    # Get predictions
    mean_pred, std_pred = model.predict_with_uncertainty(X_test, n_samples=100)
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    return model, metrics, history


def train_mcmc(X_train, y_train, X_test, y_test):
    """Train model using MCMC"""
    print("\n" + "="*70)
    print("MCMC (Hamiltonian Monte Carlo)")
    print("="*70)
    
    model = BayesianKAN(
        layer_sizes=[1, 8, 8, 1],
        n_basis=6,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    # Pre-train with VI
    print("Pre-training with VI...")
    train_loader, _ = create_data_loaders(X_train, y_train, batch_size=32)
    VariationalInference.train(model, train_loader, epochs=30, lr=1e-3, kl_weight=0.01)
    
    # MCMC sampling
    print("Running MCMC...")
    X_mcmc = X_train[:100]  # Use subset for speed
    y_mcmc = y_train[:100]
    
    mcmc_results = MCMCSampler.sample(
        model=model,
        data=(X_mcmc, y_mcmc),
        n_samples=300,
        warmup=50,
        step_size=0.001,
        n_leapfrog=5
    )
    
    print(f"Acceptance rate: {mcmc_results['final_acceptance']:.2%}")
    
    # Make predictions using samples
    predictions = []
    n_pred_samples = min(50, len(mcmc_results['samples']))
    
    with torch.no_grad():
        for i in range(n_pred_samples):
            sample = mcmc_results['samples'][i]
            for param, sample_param in zip(model.parameters(), sample):
                param.data = sample_param.to(DEVICE)
            pred, _ = model(X_test, sample=False)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    return model, metrics, mcmc_results


def train_laplace(X_train, y_train, X_test, y_test):
    """Train model using Laplace Approximation"""
    print("\n" + "="*70)
    print("LAPLACE APPROXIMATION")
    print("="*70)
    
    model = BayesianKAN(
        layer_sizes=[1, 8, 8, 1],
        n_basis=6,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    # Pre-train with VI
    print("Pre-training with VI...")
    train_loader, _ = create_data_loaders(X_train, y_train, batch_size=32)
    VariationalInference.train(model, train_loader, epochs=30, lr=1e-3, kl_weight=0.01)
    
    # Laplace approximation
    print("Computing Laplace approximation...")
    laplace_results = LaplaceApproximation.approximate(
        model=model,
        data=(X_train, y_train),
        n_iterations=10,
        prior_weight=0.01
    )
    
    # Sample from posterior
    posterior_samples = LaplaceApproximation.sample_from_posterior(
        map_params=laplace_results['map_params'],
        posterior_var=laplace_results['posterior_variance'],
        n_samples=50
    )
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for sample in posterior_samples:
            for param, sample_param in zip(model.parameters(), sample):
                param.data = sample_param.to(DEVICE)
            pred, _ = model(X_test, sample=False)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    return model, metrics, laplace_results


def compare_methods(vi_metrics, mcmc_metrics, laplace_metrics):
    """Create comparison visualization"""
    print("\n" + "="*70)
    print("COMPARISON OF METHODS")
    print("="*70)
    
    methods = ['VI', 'MCMC', 'Laplace']
    metrics_dict = {
        'VI': vi_metrics,
        'MCMC': mcmc_metrics,
        'Laplace': laplace_metrics
    }
    
    # Print comparison table
    print(f"\n{'Method':<15} {'MSE':<12} {'RMSE':<12} {'Calibration':<15} {'Mean Unc.':<12}")
    print("-" * 70)
    for method in methods:
        m = metrics_dict[method]
        print(f"{method:<15} {m['mse']:<12.6f} {m['rmse']:<12.6f} "
              f"{m['calibration_95']:<15.2%} {m['mean_uncertainty']:<12.6f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE comparison
    mse_values = [metrics_dict[m]['mse'] for m in methods]
    axes[0, 0].bar(methods, mse_values, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    rmse_values = [metrics_dict[m]['rmse'] for m in methods]
    axes[0, 1].bar(methods, rmse_values, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Squared Error')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Calibration comparison
    calib_values = [metrics_dict[m]['calibration_95'] * 100 for m in methods]
    axes[1, 0].bar(methods, calib_values, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[1, 0].axhline(95, color='red', linestyle='--', alpha=0.5, label='Target (95%)')
    axes[1, 0].set_ylabel('Calibration (%)')
    axes[1, 0].set_title('95% Uncertainty Calibration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Uncertainty comparison
    unc_values = [metrics_dict[m]['mean_uncertainty'] for m in methods]
    axes[1, 1].bar(methods, unc_values, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[1, 1].set_ylabel('Mean Uncertainty')
    axes[1, 1].set_title('Average Prediction Uncertainty')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results_comparison/method_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comparison plot to: results_comparison/method_comparison.png")
    plt.show()


def main():
    print("="*70)
    print("COMPREHENSIVE COMPARISON: VI vs MCMC vs Laplace")
    print("="*70)
    
    # Set seed
    set_seed(42)
    
    # Generate data
    print("\nGenerating data...")
    X_train, y_train = generate_synthetic_data(n_samples=300, input_dim=1, 
                                               noise_std=0.1, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=150, input_dim=1, 
                                             noise_std=0.1, seed=43)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train with each method
    vi_model, vi_metrics, vi_history = train_vi(X_train, y_train, X_test, y_test)
    mcmc_model, mcmc_metrics, mcmc_results = train_mcmc(X_train, y_train, X_test, y_test)
    laplace_model, laplace_metrics, laplace_results = train_laplace(X_train, y_train, X_test, y_test)
    
    # Compare results
    compare_methods(vi_metrics, mcmc_metrics, laplace_metrics)
    
    # Visualize all methods
    print("\nGenerating individual visualizations...")
    
    # VI visualization
    viz_vi = BayesianKANVisualizer(save_dir="results_comparison/vi")
    viz_vi.plot_training_history(vi_history)
    viz_vi.plot_predictions_with_uncertainty(vi_model, X_test, y_test, n_samples=50)
    
    # MCMC visualization
    viz_mcmc = BayesianKANVisualizer(save_dir="results_comparison/mcmc")
    viz_mcmc.plot_mcmc_diagnostics(mcmc_results)
    
    # Laplace visualization
    viz_laplace = BayesianKANVisualizer(save_dir="results_comparison/laplace")
    viz_laplace.plot_parameter_distributions(laplace_model)
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("Check 'results_comparison' folder for all visualizations.")
    print("="*70)


if __name__ == "__main__":
    import os
    os.makedirs("results_comparison", exist_ok=True)
    main()