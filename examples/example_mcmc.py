"""
Example: Training Bayesian KAN with MCMC (HMC) Sampling
"""

import torch

from bayesian_kan import (
    BayesianKAN,
    MCMCSampler,
    BayesianKANVisualizer,
    generate_synthetic_data,
    set_seed,
    DEVICE
)


def main():
    print("="*70)
    print("Bayesian KAN - MCMC (Hamiltonian Monte Carlo) Sampling Example")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # 1. Generate data (smaller dataset for MCMC efficiency)
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=200, input_dim=1, 
                                               noise_std=0.1, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=100, input_dim=1, 
                                             noise_std=0.1, seed=43)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Initialize model
    print("\n2. Initializing Bayesian KAN model...")
    model = BayesianKAN(
        layer_sizes=[1, 8, 8, 1],  # Smaller for faster MCMC
        n_basis=6,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    print(f"   Architecture: {model.layer_sizes}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. MCMC Sampling
    print("\n3. Running MCMC sampling (this may take a while)...")
    print("   Note: For faster results, reduce n_samples or use a subset of data")
    
    # Use smaller subset for MCMC
    X_mcmc = X_train[:100]
    y_mcmc = y_train[:100]
    
    mcmc_results = MCMCSampler.sample(
        model=model,
        data=(X_mcmc, y_mcmc),
        n_samples=500,
        warmup=100,
        step_size=0.001,
        n_leapfrog=5
    )
    
    print(f"   Total samples: {len(mcmc_results['samples'])}")
    print(f"   Final acceptance rate: {mcmc_results['final_acceptance']:.2%}")
    
    # 4. Compute sample statistics
    print("\n4. Computing MCMC sample statistics...")
    sample_stats = MCMCSampler.compute_statistics(mcmc_results['samples'])
    print(f"   Number of samples: {sample_stats['n_samples']}")
    
    # 5. Visualize MCMC diagnostics
    print("\n5. Visualizing MCMC diagnostics...")
    viz = BayesianKANVisualizer(save_dir="results_mcmc")
    viz.plot_mcmc_diagnostics(mcmc_results)
    
    # 6. Make predictions using MCMC samples
    print("\n6. Making predictions with MCMC posterior samples...")
    model.eval()
    predictions = []
    
    # Use first 100 samples for prediction
    n_pred_samples = min(100, len(mcmc_results['samples']))
    
    with torch.no_grad():
        for i in range(n_pred_samples):
            # Load sample parameters
            sample = mcmc_results['samples'][i]
            for param, sample_param in zip(model.parameters(), sample):
                param.data = sample_param.to(DEVICE)
            
            # Make prediction
            pred, _ = model(X_test, sample=False)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(0)
    std_pred = predictions.std(0)
    
    # 7. Compute metrics
    print("\n7. Computing evaluation metrics...")
    from bayesian_kan.utils import compute_metrics
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   95% Calibration: {metrics['calibration_95']:.2%}")
    print(f"   Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
    
    # 8. Visualize predictions
    print("\n8. Visualizing predictions with MCMC uncertainty...")
    viz.plot_predictions_from_stats(
        X_test,
        mean_pred,
        std_pred,
        y_test=y_test
    )
    
    # 9. Save results
    print("\n9. Saving MCMC results...")
    mcmc_history = {
        'log_probs': mcmc_results['log_probs'],
        'acceptance_rate': mcmc_results['acceptance_rate']
    }
    mcmc_summary = {
        'n_samples': len(mcmc_results['samples']),
        'acceptance_rate': mcmc_results['final_acceptance'],
        'metrics': metrics
    }
    viz.save_results_summary(model, mcmc_history, mcmc_summary)
    
    print("\n" + "="*70)
    print("MCMC sampling complete!")
    print("Check the 'results_mcmc' folder for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()
