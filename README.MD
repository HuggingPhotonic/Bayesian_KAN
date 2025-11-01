# Bayesian Kolmogorov-Arnold Networks (B-KAN)

A comprehensive implementation of Bayesian KAN with multiple inference methods including Variational Inference (VI), Markov Chain Monte Carlo (MCMC), and Laplace Approximation.

## Features

- **Bayesian KAN Architecture**: Probabilistic Kolmogorov-Arnold Networks with B-spline basis functions
- **Multiple Inference Methods**:
  - Variational Inference (VI) - Fast and scalable
  - MCMC with Hamiltonian Monte Carlo - Accurate posterior sampling
  - Laplace Approximation - Quick posterior approximation
- **Uncertainty Quantification**: Built-in prediction uncertainty estimation
- **Comprehensive Visualization**: Training curves, predictions with uncertainty bands, MCMC diagnostics
- **GPU Support**: Automatic GPU/CPU detection and usage

## Project Structure

```
bayesian_kan/
├── __init__.py              # Package initialization
├── config.py                # Configuration and device setup
├── bspline.py              # B-spline basis functions
├── model.py                # Bayesian KAN model architecture
├── variational_inference.py # VI training
├── mcmc_sampling.py        # HMC sampling
├── laplace_approximation.py # Laplace approximation
├── visualization.py        # Visualization tools
├── utils.py                # Utility functions
└── examples/
    ├── example_vi.py       # VI example
    ├── example_mcmc.py     # MCMC example
    ├── example_laplace.py  # Laplace example
    └── example_comparison.py # Compare all methods
```

## Installation

```bash
# Clone or download the repository
cd bayesian_kan

# Install dependencies
pip install torch numpy matplotlib seaborn tqdm
```

## Quick Start

### 1. Variational Inference (Recommended for most cases)

```python
from bayesian_kan import (
    BayesianKAN, VariationalInference, 
    generate_synthetic_data, create_data_loaders
)

# Generate data
X_train, y_train = generate_synthetic_data(n_samples=500)
train_loader, val_loader = create_data_loaders(X_train, y_train)

# Create model
model = BayesianKAN(layer_sizes=[1, 10, 10, 1])

# Train with VI
history = VariationalInference.train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)

# Make predictions with uncertainty
mean, std = model.predict_with_uncertainty(X_test, n_samples=100)
```

### 2. MCMC Sampling

```python
from bayesian_kan import MCMCSampler

# Run MCMC
mcmc_results = MCMCSampler.sample(
    model=model,
    data=(X_train, y_train),
    n_samples=1000,
    warmup=100
)

# Compute statistics
sample_stats = MCMCSampler.compute_statistics(mcmc_results['samples'])
```

### 3. Laplace Approximation

```python
from bayesian_kan import LaplaceApproximation

# Compute approximation
laplace_results = LaplaceApproximation.approximate(
    model=model,
    data=(X_train, y_train)
)

# Sample from posterior
samples = LaplaceApproximation.sample_from_posterior(
    map_params=laplace_results['map_params'],
    posterior_var=laplace_results['posterior_variance'],
    n_samples=100
)
```

## Running Examples

### Run individual examples:

```bash
# Variational Inference
python -m bayesian_kan.examples.example_vi

# MCMC Sampling
python -m bayesian_kan.examples.example_mcmc

# Laplace Approximation
python -m bayesian_kan.examples.example_laplace

# Compare all methods
python -m bayesian_kan.examples.example_comparison
```

## Method Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **VI** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | General purpose, large datasets |
| **MCMC** | 🐌 Slow | ⭐⭐⭐⭐⭐ Excellent | Small datasets, need accurate posterior |
| **Laplace** | ⚡⚡ Medium | ⭐⭐⭐ Good | Quick approximation, medium datasets |

## Model Architecture

The Bayesian KAN uses:
- **B-spline basis functions** for smooth, learnable transformations
- **Variational parameters** (mean and log-variance) for each coefficient
- **KL divergence** regularization to stay close to prior
- **Uncertainty propagation** through the network

## Key Parameters

### Model Initialization
- `layer_sizes`: List of layer sizes (e.g., [1, 10, 10, 1])
- `n_basis`: Number of B-spline basis functions (default: 10)
- `degree`: B-spline degree (default: 3)
- `prior_scale`: Standard deviation of prior (default: 1.0)

### Training Parameters
- **VI**: 
  - `epochs`: Number of training epochs
  - `lr`: Learning rate
  - `kl_weight`: Weight for KL divergence term
- **MCMC**: 
  - `n_samples`: Number of samples to draw
  - `warmup`: Number of warmup iterations
  - `step_size`: HMC step size
- **Laplace**: 
  - `n_iterations`: MAP optimization iterations
  - `prior_weight`: Weight for prior term

## Visualization

The package includes comprehensive visualization tools:

```python
from bayesian_kan import BayesianKANVisualizer

viz = BayesianKANVisualizer(save_dir="results")

# Plot training history
viz.plot_training_history(history)

# Plot predictions with uncertainty
viz.plot_predictions_with_uncertainty(model, X_test, y_test)

# Plot parameter distributions
viz.plot_parameter_distributions(model)

# Plot MCMC diagnostics
viz.plot_mcmc_diagnostics(mcmc_results)

# Save results summary
viz.save_results_summary(model, history, metrics)
```

## Performance Tips

1. **Use VI for initial training**: It's fast and provides a good initialization
2. **Pre-train before MCMC**: Use VI to find a good starting point
3. **Adjust KL weight**: Lower for more flexible models, higher for more regularization
4. **Use smaller networks for MCMC**: Reduces computation time
5. **GPU acceleration**: Automatically used if available

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bayesian_kan,
  title={Bayesian Kolmogorov-Arnold Networks},
  author={Bayesian KAN Team},
  year={2024},
  url={https://github.com/yourusername/bayesian-kan}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. Kolmogorov, A. N. (1957). On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables.
2. Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks.
3. Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks.
4. Neal, R. M. (2011). MCMC using Hamiltonian dynamics.