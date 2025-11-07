"""
Check if training has converged by analyzing loss curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the most recent training results
results_dir = Path("/Users/fans/code/Bayesian_KAN/photonic_version/bayesian_models/results/photonic_incoherent_hw_vi")
latest_run = sorted(results_dir.glob("*"))[-1]
print(f"Analyzing: {latest_run}")

# You would need to save losses during training
# For now, this is a template

def analyze_convergence(losses, window=100):
    """
    Analyze if training has converged.

    Returns:
        converged: bool
        recommended_epochs: int
    """
    losses = np.array(losses)

    # Calculate moving average
    if len(losses) < window:
        return False, len(losses) * 2

    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')

    # Calculate derivative (rate of change)
    derivative = np.diff(moving_avg)

    # Check if derivative is near zero in the last 20% of training
    last_20_percent = int(0.2 * len(derivative))
    recent_derivative = np.abs(derivative[-last_20_percent:])

    # Convergence criteria: average change < 0.0001 per epoch
    is_converged = np.mean(recent_derivative) < 0.0001

    if is_converged:
        # Find when it actually converged
        convergence_threshold = 0.0001
        for i in range(len(derivative) - 1, 0, -1):
            if np.abs(derivative[i]) > convergence_threshold:
                # Converged at epoch i + window + safety margin
                recommended = int((i + window) * 1.2)  # 20% safety margin
                return True, recommended

    # Not converged, recommend more epochs
    return False, int(len(losses) * 1.5)

# Example usage (you'd load actual losses from your training)
print("""
To use this script properly:

1. Modify your VI training to save losses:
   ```python
   import pickle
   with open(output_dir / 'losses.pkl', 'wb') as f:
       pickle.dump({
           'losses': losses,
           'recon_losses': recon_losses,
           'kl_terms': kl_terms
       }, f)
   ```

2. Then run this script to analyze convergence

3. Based on the analysis:
   - If converged early: reduce epochs
   - If not converged: increase epochs
   - If just right: keep current setting
""")
