"""
Debug script to analyze VI training loss behavior.
"""
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import VIConfig, train_vi
from photonic_version.bases.hardware import BayesianHardwarePhotonicIncoherentBasis
from photonic_version.photonic_kan import target_function
from photonic_version.utils import get_device
import torch.nn as nn

class SimpleBayesianKAN(nn.Module):
    def __init__(self, layer_sizes, basis_kwargs):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(BayesianHardwarePhotonicIncoherentBasis(in_f, out_f, **basis_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True, n_samples=1):
        kl_total = torch.zeros(1, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, kl = layer(x, sample=sample, n_samples=n_samples)
            kl_total = kl_total + kl.to(x.device)
        return x, kl_total

torch.manual_seed(42)
device = get_device()

# Training data
n_train = 512
X_train = torch.rand(n_train, 1, device=device) * 2 - 1
y_train = target_function(X_train)

# Simplified model
layer_sizes = (1, 16, 16, 1)  # Smaller for debugging
basis_kwargs = {
    "num_rings": 12,
    "wl_nm_range": (1546.0, 1554.0),
    "R_um": 30.0,
    "neff": 2.34,
    "ng": 4.2,
    "loss_dB_cm": 3.0,
    "kappa": 0.2,
}

model = SimpleBayesianKAN(layer_sizes, basis_kwargs).to(device)

# Custom training loop with detailed logging
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
mse_loss = nn.MSELoss()

epochs = 500
kl_max = 0.5
kl_warmup = 200

losses = []
recons = []
kls = []
kl_weights_list = []

print("Starting debug training...")
print(f"{'Epoch':>6} {'Loss':>10} {'Recon':>10} {'KL':>10} {'KL_weight':>10} {'KL*weight':>10}")
print("-" * 66)

for epoch in range(epochs):
    optimizer.zero_grad()

    preds, kl = model(X_train, sample=True, n_samples=4)
    recon = mse_loss(preds, y_train)

    # KL annealing
    warmup_frac = min(1.0, (epoch + 1) / kl_warmup)
    kl_weight = kl_max * warmup_frac

    # ELBO loss
    loss = recon + kl_weight * kl / X_train.size(0)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    recons.append(recon.item())
    kls.append(kl.item())
    kl_weights_list.append(kl_weight)

    if epoch % 50 == 0 or epoch < 10:
        print(f"{epoch:6d} {loss.item():10.4f} {recon.item():10.4f} {kl.item():10.2f} "
              f"{kl_weight:10.4f} {(kl_weight * kl / X_train.size(0)).item():10.4f}")

print("\n=== Analysis ===")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Final recon: {recons[-1]:.4f}")
print(f"Final KL: {kls[-1]:.2f}")
print(f"Final KL contribution: {(kl_weights_list[-1] * kls[-1] / X_train.size(0)):.4f}")

# Check if loss is increasing
if losses[-1] > losses[kl_warmup]:
    print("\n⚠️  WARNING: Loss increased after warmup!")
    print(f"   Loss at warmup end (epoch {kl_warmup}): {losses[kl_warmup]:.4f}")
    print(f"   Loss at end (epoch {epochs-1}): {losses[-1]:.4f}")
    print(f"   Increase: {losses[-1] - losses[kl_warmup]:.4f}")
else:
    print("\n✓ Loss decreased overall")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(losses, label='Total Loss', linewidth=2)
axes[0, 0].axvline(kl_warmup, color='red', linestyle='--', alpha=0.5, label='Warmup end')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Total Loss (ELBO)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Reconstruction
axes[0, 1].plot(recons, label='Reconstruction Loss', color='green', linewidth=2)
axes[0, 1].axvline(kl_warmup, color='red', linestyle='--', alpha=0.5, label='Warmup end')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Reconstruction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# KL divergence
axes[1, 0].plot(kls, label='KL Divergence', color='orange', linewidth=2)
axes[1, 0].axvline(kl_warmup, color='red', linestyle='--', alpha=0.5, label='Warmup end')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('KL')
axes[1, 0].set_title('KL Divergence (raw)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# KL weight schedule
axes[1, 1].plot(kl_weights_list, label='KL Weight', color='purple', linewidth=2)
axes[1, 1].axvline(kl_warmup, color='red', linestyle='--', alpha=0.5, label='Warmup end')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_title('KL Weight Schedule')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/fans/code/Bayesian_KAN/vi_debug_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to vi_debug_analysis.png")

# Check parameter statistics
print("\n=== Parameter Statistics ===")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:60s} mean={param.mean().item():8.4f} std={param.std().item():8.4f} "
              f"min={param.min().item():8.4f} max={param.max().item():8.4f}")
