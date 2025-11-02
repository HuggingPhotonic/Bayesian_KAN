"""
RBF KAN 2D Laplace approximation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
from tqdm.auto import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_laplace_rbf_2d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class RBFKernelLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_centers: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.log_sigma = nn.Parameter(torch.zeros(num_centers))
        self.coeffs = nn.Parameter(
            torch.randn(in_features, out_features, num_centers) * 0.05
        )
        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

    def _basis(self, x):
        x_expanded = x.unsqueeze(1)
        centers = self.centers.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        dist_sq = torch.sum((x_expanded - centers) ** 2, dim=-1)
        rbf_vals = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))
        return rbf_vals.unsqueeze(1).repeat(1, self.in_features, 1)

    def forward(self, x):
        basis = self._basis(x)
        rbf_out = torch.einsum("bin,ion->bo", basis, self.coeffs)
        residual = x @ self.base_weight
        return rbf_out + residual


class RBFKAN(nn.Module):
    def __init__(self, layers_hidden, num_centers=16):
        super().__init__()
        self.layers = nn.ModuleList([
            RBFKernelLayer(layers_hidden[i], layers_hidden[i + 1], num_centers=num_centers)
            for i in range(len(layers_hidden) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def target_function(x, y):
    return (torch.sin(np.pi * x) * torch.cos(np.pi * y) +
            0.3 * torch.exp(-(x ** 2 + y ** 2)) +
            0.2 * x * y +
            0.1 * torch.sin(3 * x) * torch.sin(3 * y))


def negative_log_posterior(model, x, y, noise_var=0.05, prior_var=1.0):
    preds = model(x)
    sse = torch.sum((preds - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def train_map(model, x, y, epochs=2000, lr=5e-4,
              noise_var=0.05, prior_var=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    losses = []
    progress = trange(epochs, desc="RBF MAP 2D", leave=True)
    for epoch in progress:
        optimizer.zero_grad()
        loss = negative_log_posterior(model, x, y, noise_var, prior_var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    return losses


def hessian_diag(model, x, y, noise_var=0.05, prior_var=0.5):
    loss = negative_log_posterior(model, x, y, noise_var, prior_var)
    params = [p for p in model.parameters()]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = parameters_to_vector(grads)
    diag = []
    for i in range(grad_vec.numel()):
        comp = grad_vec[i]
        second = torch.autograd.grad(comp, params, retain_graph=True)
        diag.append(parameters_to_vector(second)[i])
    return torch.stack(diag).detach()


def laplace_samples(model, theta_map, var_diag, x_eval, n_samples=200):
    preds = []
    std = torch.sqrt(var_diag.clamp(min=1e-6))
    with torch.no_grad():
        for _ in range(n_samples):
            sample_vec = theta_map + torch.randn_like(theta_map) * std
            vector_to_parameters(sample_vec, model.parameters())
            preds.append(model(x_eval))
        vector_to_parameters(theta_map, model.parameters())
    return torch.stack(preds)


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train = 2000
    x_train = torch.rand(n_train, 1) * 4 - 2
    y_train = torch.rand(n_train, 1) * 4 - 2
    X_train = torch.cat([x_train, y_train], dim=1).to(device)
    z_train = target_function(x_train, y_train).to(device)

    n_test = 60
    x_test = torch.linspace(-2, 2, n_test)
    y_test = torch.linspace(-2, 2, n_test)
    X_grid, Y_grid = torch.meshgrid(x_test, y_test, indexing="ij")
    X_eval = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(device)
    Z_true = target_function(X_grid.to(device), Y_grid.to(device)).cpu()

    model = RBFKAN(layers_hidden=[2, 12, 12, 1], num_centers=24).to(device)

    print("Training MAP estimate with RBF basis...")
    map_losses = train_map(model, X_train, z_train)

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("RBF MAP Training Loss (2D)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "rbf_map_loss.png", dpi=150, bbox_inches="tight")

    print("\nComputing Laplace approximation...")
    theta_map = parameters_to_vector(model.parameters()).detach()
    diag = hessian_diag(model, X_train, z_train)
    var_diag = 1.0 / (diag + 1e-2)  # damping for stability
    preds = laplace_samples(model, theta_map, var_diag, X_eval, n_samples=150)
    mean_pred = preds.mean(0).cpu().reshape(n_test, n_test)
    std_pred = preds.std(0).cpu().reshape(n_test, n_test)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, cmap="viridis", alpha=0.8)
    ax1.set_title("Ground Truth")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X_grid.cpu(), Y_grid.cpu(), mean_pred, cmap="viridis", alpha=0.8)
    ax2.set_title("Laplace Mean (RBF)")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X_grid.cpu(), Y_grid.cpu(), std_pred, cmap="hot", alpha=0.8)
    ax3.set_title("Predictive Std Dev")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rbf_laplace_surfaces.png", dpi=150, bbox_inches="tight")
    print("Surface plots saved!")

    mse = torch.mean((mean_pred - Z_true) ** 2).item()
    mae = torch.mean(torch.abs(mean_pred - Z_true)).item()
    print(f"\nLaplace metrics: MSE={mse:.6f}, MAE={mae:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    levels = 20
    c1 = axes[0].contourf(X_grid.cpu(), Y_grid.cpu(), Z_true, levels=levels, cmap="viridis")
    axes[0].set_title("Ground Truth")
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X_grid.cpu(), Y_grid.cpu(), mean_pred, levels=levels, cmap="viridis")
    axes[1].set_title("Laplace Mean")
    plt.colorbar(c2, ax=axes[1])

    c3 = axes[2].contourf(X_grid.cpu(), Y_grid.cpu(), std_pred, levels=levels, cmap="hot")
    axes[2].set_title("Predictive Std Dev")
    plt.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rbf_laplace_contours.png", dpi=150, bbox_inches="tight")
    print("Contour plots saved!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("RBF KAN 2D Laplace Approximation")
    print("=" * 60)

    run_experiment()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
