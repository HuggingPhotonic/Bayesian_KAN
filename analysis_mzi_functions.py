"""
Analyze the gradient properties of different MZI amplitude functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def original_amp(theta):
    """Original trainable amplitude function."""
    sigmoid_theta = torch.sigmoid(torch.tensor(theta, dtype=torch.float32))
    return (torch.sin(sigmoid_theta * np.pi * 0.5) ** 2).numpy()

def physical_amp(theta):
    """Physical MZI amplitude function."""
    sigmoid_theta = torch.sigmoid(torch.tensor(theta, dtype=torch.float32))
    theta_physical = np.pi * sigmoid_theta.numpy()
    return np.cos(theta_physical / 2.0)

# Compute gradients numerically
theta_range = np.linspace(-10, 10, 1000)
original_vals = [original_amp(t) for t in theta_range]
physical_vals = [physical_amp(t) for t in theta_range]

# Numerical gradients
original_grads = np.gradient(original_vals, theta_range)
physical_grads = np.gradient(physical_vals, theta_range)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amplitude functions
axes[0, 0].plot(theta_range, original_vals, label='Original: sin²(sigmoid(θ)·π/2)', linewidth=2)
axes[0, 0].plot(theta_range, physical_vals, label='Physical: cos(π·sigmoid(θ)/2)', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('θ (raw parameter)', fontsize=12)
axes[0, 0].set_ylabel('Amplitude', fontsize=12)
axes[0, 0].set_title('MZI Amplitude Functions', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)

# Gradients
axes[0, 1].plot(theta_range, original_grads, label='Original gradient', linewidth=2)
axes[0, 1].plot(theta_range, physical_grads, label='Physical gradient', linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('θ (raw parameter)', fontsize=12)
axes[0, 1].set_ylabel('dAmp/dθ', fontsize=12)
axes[0, 1].set_title('Gradient Comparison', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].axvline(0, color='gray', linestyle=':', alpha=0.5)

# Zoom in on training region [-5, 5]
axes[1, 0].plot(theta_range, original_vals, label='Original', linewidth=2)
axes[1, 0].plot(theta_range, physical_vals, label='Physical', linewidth=2, linestyle='--')
axes[1, 0].set_xlim(-5, 5)
axes[1, 0].set_xlabel('θ (training range)', fontsize=12)
axes[1, 0].set_ylabel('Amplitude', fontsize=12)
axes[1, 0].set_title('Amplitude (zoomed to training range)', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].fill_between([-3, -1], 0, 1, alpha=0.2, color='green', label='Init range')

# Gradient magnitude
axes[1, 1].plot(theta_range, np.abs(original_grads), label='|Original gradient|', linewidth=2)
axes[1, 1].plot(theta_range, np.abs(physical_grads), label='|Physical gradient|', linewidth=2, linestyle='--')
axes[1, 1].set_xlabel('θ (raw parameter)', fontsize=12)
axes[1, 1].set_ylabel('|dAmp/dθ|', fontsize=12)
axes[1, 1].set_title('Gradient Magnitude', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')
axes[1, 1].axvline(-3, color='green', linestyle=':', alpha=0.5, label='Init lower')
axes[1, 1].axvline(-1, color='green', linestyle=':', alpha=0.5, label='Init upper')

plt.tight_layout()
plt.savefig('/Users/fans/code/Bayesian_KAN/mzi_function_analysis.png', dpi=150, bbox_inches='tight')
print("Analysis saved to mzi_function_analysis.png")

# Print statistics
print("\n=== Function Analysis ===")
print(f"Original amp range: [{min(original_vals):.4f}, {max(original_vals):.4f}]")
print(f"Physical amp range: [{min(physical_vals):.4f}, {max(physical_vals):.4f}]")
print(f"\nOriginal gradient range: [{min(original_grads):.6f}, {max(original_grads):.6f}]")
print(f"Physical gradient range: [{min(physical_grads):.6f}, {max(physical_grads):.6f}]")

# At initialization range [-3, -1]
init_mask = (theta_range >= -3) & (theta_range <= -1)
print(f"\n=== At Initialization (θ ∈ [-3, -1]) ===")
print(f"Original amp: [{min(np.array(original_vals)[init_mask]):.4f}, {max(np.array(original_vals)[init_mask]):.4f}]")
print(f"Physical amp: [{min(np.array(physical_vals)[init_mask]):.4f}, {max(np.array(physical_vals)[init_mask]):.4f}]")
print(f"Original grad: [{min(original_grads[init_mask]):.6f}, {max(original_grads[init_mask]):.6f}]")
print(f"Physical grad: [{min(physical_grads[init_mask]):.6f}, {max(physical_grads[init_mask]):.6f}]")
