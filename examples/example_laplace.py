"""
Example: Training Bayesian KAN with Laplace Approximation
"""

import torch

from bayesian_kan import (
    BayesianKAN,
    LaplaceApproximation,
    BayesianKANVisualizer,
    generate_synthetic_data,
    set_seed,
    compute_metrics,
    DEVICE
)


def main():
    print("="*70)
    print("Bayesian KAN - Laplace Approximation Example")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=300, input_dim=1, 
                                               noise_std=0.1, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=150, input_dim=1, 
                                             noise_std=0.1, seed=43)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Initialize model
    print("\n2. Initializing Bayesian KAN model...")
    model = BayesianKAN(
        layer_sizes=[1, 10, 10, 1],
        n_basis=8,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    print(f"   Architecture: {model.layer_sizes}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Compute Laplace Approximation
    print("\n3. Computing Laplace approximation...")
    laplace_results = LaplaceApproximation.approximate(
        model=model,
        data=(X_train, y_train),
        n_iterations=10,
        prior_weight=0.01
    )
    
    print(f"   MAP loss: {laplace_results['map_loss']:.6f}")
    print(f"   Posterior variance range: [{laplace_results['posterior_variance'].min():.6f}, "
          f"{laplace_results['posterior_variance'].max():.6f}]")
    
    # 4. Sample from Laplace posterior
    print("\n4. Sampling from Laplace posterior...")
    posterior_samples = LaplaceApproximation.sample_from_posterior(
        map_params=laplace_results['map_params'],
        posterior_var=laplace_results['posterior_variance'],
        n_samples=100
    )
    print(f"   Generated {len(posterior_samples)} posterior samples")
    
    # 5. Make predictions using Laplace samples
    print("\n5. Making predictions with Laplace posterior...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i, sample in enumerate(posterior_samples):
            # Load sample parameters
            for param, sample_param in zip(model.parameters(), sample):
                param.data = sample_param.to(DEVICE)
            
            # Make prediction
            pred, _ = model(X_test, sample=False)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    
    # 6. Compute metrics
    print("\n6. Computing evaluation metrics...")
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   95% Calibration: {metrics['calibration_95']:.2%}")
    print(f"   Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
    
    # 7. Reload MAP model for visualization
    print("\n7. Reloading MAP model...")
    for param, map_param in zip(model.parameters(), laplace_results['map_params']):
        param.data = map_param.to(DEVICE)
    
    # 8. Visualize results
    print("\n8. Visualizing results...")
    viz = BayesianKANVisualizer(save_dir="results_laplace")
    
    # Plot predictions with uncertainty using Laplace samples
    viz.plot_predictions_from_stats(
        X_test,
        mean_pred,
        std_pred,
        y_test=y_test
    )
    
    # Plot parameter distributions
    viz.plot_parameter_distributions(model)
    
    # 9. Visualize Laplace posterior
    print("\n9. Visualizing Laplace posterior properties...")
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Posterior variance histogram
    post_var = laplace_results['posterior_variance'].numpy()
    axes[0].hist(np.log10(post_var + 1e-10), bins=50, alpha=0.7, 
                color='blue', edgecolor='black')
    axes[0].set_xlabel('log10(Posterior Variance)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Posterior Variances')
    axes[0].grid(True, alpha=0.3)
    
    # Hessian diagonal
    hess_diag = laplace_results['hessian_diagonal'].numpy()
    axes[1].hist(np.log10(hess_diag + 1e-10), bins=50, alpha=0.7, 
                color='green', edgecolor='black')
    axes[1].set_xlabel('log10(Hessian Diagonal)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Hessian Diagonal Elements')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{viz.save_dir}/laplace_posterior_{viz.timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.show()
    
    # 10. Save results
    print("\n10. Saving results...")
    laplace_summary = {
        'map_loss': laplace_results['map_loss'],
        'posterior_var_stats': {
            'min': float(post_var.min()),
            'max': float(post_var.max()),
            'mean': float(post_var.mean()),
            'std': float(post_var.std())
        },
        'metrics': metrics
    }
    laplace_history = {
        'map_loss': [laplace_results['map_loss']],
        'posterior_variance_mean': [float(post_var.mean())]
    }
    viz.save_results_summary(model, laplace_history, laplace_summary)
    
    print("\n" + "="*70)
    print("Laplace approximation complete!")
    print("Check the 'results_laplace' folder for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()
