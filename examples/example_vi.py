"""
Example: Training Bayesian KAN with Variational Inference
"""

import torch
import torch.nn.functional as F

from bayesian_kan import (
    BayesianKAN,
    VariationalInference,
    BayesianKANVisualizer,
    generate_synthetic_data,
    set_seed,
    create_data_loaders,
    compute_metrics,
    DEVICE
)


def main():
    print("="*70)
    print("Bayesian KAN - Variational Inference Example")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=3000, input_dim=1, 
                                               noise_std=0.1, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=2000, input_dim=1, 
                                             noise_std=0.1, seed=43)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, 
        batch_size=32, shuffle=True
    )
    
    # 3. Initialize model
    print("\n3. Initializing Bayesian KAN model...")
    model = BayesianKAN(
        layer_sizes=[1, 10, 10, 1],
        n_basis=8,
        degree=3,
        prior_scale=1.0
    ).to(DEVICE)
    
    print(f"   Architecture: {model.layer_sizes}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Train with Variational Inference
    print("\n4. Training with Variational Inference...")
    history = VariationalInference.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=1e-3,
        kl_weight=0.1
    )
    
    print(f"   Final training loss: {history['loss'][-1]:.6f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # 5. Visualize training
    print("\n5. Visualizing training history...")
    viz = BayesianKANVisualizer(save_dir="results_vi")
    viz.plot_training_history(history)
    
    # 6. Make predictions with uncertainty
    print("\n6. Making predictions with uncertainty...")
    model.eval()
    with torch.no_grad():
        mean_pred, std_pred = model.predict_with_uncertainty(X_test, n_samples=500)
    
    # 7. Compute metrics
    print("\n7. Computing evaluation metrics...")
    metrics = compute_metrics(mean_pred, y_test, std_pred)
    
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   95% Calibration: {metrics['calibration_95']:.2%}")
    print(f"   Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
    
    # 8. Visualize predictions
    print("\n8. Visualizing predictions with uncertainty...")
    viz.plot_predictions_with_uncertainty(model, X_test, y_test, n_samples=100)
    
    # 9. Visualize parameter distributions
    print("\n9. Visualizing parameter distributions...")
    viz.plot_parameter_distributions(model)
    
    # 10. Save results
    print("\n10. Saving results...")
    viz.save_results_summary(model, history, metrics)
    
    print("\n" + "="*70)
    print("Variational Inference training complete!")
    print("Check the 'results_vi' folder for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()