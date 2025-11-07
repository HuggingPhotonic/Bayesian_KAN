"""
Diagnose why KL divergence is so high (80-90).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# Simulate what might be happening in your model
def compute_kl_components(mean, log_var):
    """Compute KL divergence and its components."""
    var = torch.exp(log_var)

    # KL components
    var_term = var
    mean_term = mean ** 2
    entropy_term = -1.0 - log_var

    kl_per_param = 0.5 * (var_term + mean_term + entropy_term)
    kl_mean = torch.mean(kl_per_param)

    return {
        'kl_mean': kl_mean.item(),
        'var_term_mean': torch.mean(var_term).item(),
        'mean_term_mean': torch.mean(mean_term).item(),
        'entropy_term_mean': torch.mean(entropy_term).item(),
        'var_mean': torch.mean(var).item(),
        'mean_abs_mean': torch.mean(torch.abs(mean)).item(),
        'log_var_mean': torch.mean(log_var).item(),
        'log_var_max': torch.max(log_var).item(),
        'log_var_min': torch.min(log_var).item(),
    }

print("=== KL Divergence Diagnosis ===\n")

# Case 1: Healthy posterior (should have low KL)
print("Case 1: Healthy VI (small variance, small mean)")
mean = torch.randn(1000) * 0.1
log_var = torch.full((1000,), -4.0)
stats = compute_kl_components(mean, log_var)
print(f"  KL_mean: {stats['kl_mean']:.4f}")
print(f"  Mean contribution: {stats['mean_term_mean']:.4f}")
print(f"  Var contribution: {stats['var_term_mean']:.4f}")
print(f"  Entropy contribution: {stats['entropy_term_mean']:.4f}\n")

# Case 2: Large variance
print("Case 2: Large variance (log_var = 2.0)")
mean = torch.randn(1000) * 0.1
log_var = torch.full((1000,), 2.0)
stats = compute_kl_components(mean, log_var)
print(f"  KL_mean: {stats['kl_mean']:.4f} ⚠️")
print(f"  Mean contribution: {stats['mean_term_mean']:.4f}")
print(f"  Var contribution: {stats['var_term_mean']:.4f}")
print(f"  Entropy contribution: {stats['entropy_term_mean']:.4f}\n")

# Case 3: Large mean
print("Case 3: Large mean (mean ~ 5.0)")
mean = torch.randn(1000) * 5.0
log_var = torch.full((1000,), -4.0)
stats = compute_kl_components(mean, log_var)
print(f"  KL_mean: {stats['kl_mean']:.4f} ⚠️")
print(f"  Mean contribution: {stats['mean_term_mean']:.4f}")
print(f"  Var contribution: {stats['var_term_mean']:.4f}")
print(f"  Entropy contribution: {stats['entropy_term_mean']:.4f}\n")

# Case 4: Both large (your likely case)
print("Case 4: Both large (mean ~ 3.0, log_var ~ 1.0)")
mean = torch.randn(1000) * 3.0
log_var = torch.randn(1000) * 0.5 + 1.0
stats = compute_kl_components(mean, log_var)
print(f"  KL_mean: {stats['kl_mean']:.4f} ⚠️⚠️")
print(f"  Mean contribution: {stats['mean_term_mean']:.4f}")
print(f"  Var contribution: {stats['var_term_mean']:.4f}")
print(f"  Entropy contribution: {stats['entropy_term_mean']:.4f}\n")

# Case 5: Extreme case (KL ~ 85)
print("Case 5: Extreme case to reach KL ~ 85")
# Solve: 0.5 * (var + mean²) ≈ 85
# If var = 20, mean² = 150 → mean ≈ 12
mean = torch.randn(1000) * 12.0
log_var = torch.randn(1000) * 0.5 + 3.0
stats = compute_kl_components(mean, log_var)
print(f"  KL_mean: {stats['kl_mean']:.4f} 🔥")
print(f"  Mean contribution: {stats['mean_term_mean']:.4f}")
print(f"  Var contribution: {stats['var_term_mean']:.4f}")
print(f"  Entropy contribution: {stats['entropy_term_mean']:.4f}")
print(f"  Mean of |mean|: {stats['mean_abs_mean']:.4f}")
print(f"  Mean of log_var: {stats['log_var_mean']:.4f}\n")

print("\n=== Diagnosis Conclusion ===")
print("If your KL ≈ 85, likely causes:")
print("1. Posterior means have grown to |mean| ~ 10-15")
print("2. Or log_var has grown to ~ 3-5 (variance ~ 20-150)")
print("3. Or both")
print("\nSolutions:")
print("1. Increase kl_max to allow more deviation")
print("2. Reduce learning rate to slow parameter growth")
print("3. Add stronger weight decay")
print("4. Check if prior scale is appropriate")
print("5. Use a more informative prior")

# Visualize KL vs mean and variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KL vs mean (fixed var)
means = np.linspace(-15, 15, 100)
log_var_fixed = -4.0
kls_vs_mean = []
for m in means:
    mean_tensor = torch.tensor([m])
    log_var_tensor = torch.tensor([log_var_fixed])
    kl = 0.5 * (torch.exp(log_var_tensor) + mean_tensor**2 - 1.0 - log_var_tensor)
    kls_vs_mean.append(kl.item())

axes[0].plot(means, kls_vs_mean, linewidth=2)
axes[0].axhline(85, color='red', linestyle='--', label='Your KL = 85')
axes[0].axhline(1, color='green', linestyle='--', label='Healthy KL ~ 1')
axes[0].set_xlabel('Posterior Mean')
axes[0].set_ylabel('KL Divergence')
axes[0].set_title('KL vs Posterior Mean (log_var = -4)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_ylim(0, 100)

# KL vs log_var (fixed mean)
log_vars = np.linspace(-6, 6, 100)
mean_fixed = 0.0
kls_vs_var = []
for lv in log_vars:
    mean_tensor = torch.tensor([mean_fixed])
    log_var_tensor = torch.tensor([lv])
    kl = 0.5 * (torch.exp(log_var_tensor) + mean_tensor**2 - 1.0 - log_var_tensor)
    kls_vs_var.append(kl.item())

axes[1].plot(log_vars, kls_vs_var, linewidth=2)
axes[1].axhline(85, color='red', linestyle='--', label='Your KL = 85')
axes[1].axhline(1, color='green', linestyle='--', label='Healthy KL ~ 1')
axes[1].axvline(-4, color='blue', linestyle=':', alpha=0.5, label='Init value')
axes[1].set_xlabel('log(Posterior Variance)')
axes[1].set_ylabel('KL Divergence')
axes[1].set_title('KL vs log(Variance) (mean = 0)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('/Users/fans/code/Bayesian_KAN/kl_diagnosis.png', dpi=150)
print("\nPlot saved to kl_diagnosis.png")
