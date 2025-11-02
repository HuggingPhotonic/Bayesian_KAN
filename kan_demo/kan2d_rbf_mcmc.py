"""
RBF KAN 2D MCMC comparison (Random-Walk Metropolis & HMC).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
from tqdm.auto import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_mcmc_rbf_2d"
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


def train_map(model, x, y, epochs=1000, lr=5e-4,
              noise_var=0.05, prior_var=1.0):
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


def log_posterior_and_grad(model, theta_vec, x, y, noise_var=0.05, prior_var=1.0):
    vector_to_parameters(theta_vec, model.parameters())
    model.zero_grad()
    nlp = negative_log_posterior(model, x, y, noise_var, prior_var)
    grads = torch.autograd.grad(nlp, model.parameters())
    grad_vec = -parameters_to_vector(grads)
    return (-nlp).detach(), grad_vec.detach()


def random_walk_metropolis(model, x, y, n_samples=800, burn_in=600,
                           step_size=2e-4, noise_var=0.05, prior_var=1.0):
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_lp, _ = log_posterior_and_grad(model, current, x, y, noise_var, prior_var)
    samples = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="RWM Sampling 2D", leave=True):
        proposal = current + step_size * torch.randn_like(current)
        proposal_lp, _ = log_posterior_and_grad(model, proposal, x, y, noise_var, prior_var)
        log_alpha = proposal_lp - current_lp
        if torch.log(torch.rand(1)) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())
        if step >= burn_in:
            samples.append(current.clone())
    acceptance = accept / total
    return torch.stack(samples) if samples else torch.empty(0), acceptance


def hmc_sampling(model, x, y, n_samples=800, burn_in=600,
                 step_size=3e-4, n_leapfrog=35,
                 noise_var=0.05, prior_var=1.0):
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_logp, current_grad = log_posterior_and_grad(model, current, x, y, noise_var, prior_var)
    samples = []
    accept = 0
    total = burn_in + n_samples
    for step in trange(total, desc="HMC Sampling 2D", leave=True):
        theta = current.clone()
        grad = current_grad.clone()
        momentum = torch.randn_like(theta)
        current_H = -current_logp + 0.5 * torch.sum(momentum ** 2)

        theta_new = theta.clone()
        momentum_new = momentum.clone()
        momentum_new = momentum_new + 0.5 * step_size * grad
        for l in range(n_leapfrog):
            theta_new = theta_new + step_size * momentum_new
            logp_new, grad_new = log_posterior_and_grad(model, theta_new, x, y, noise_var, prior_var)
            if l != n_leapfrog - 1:
                momentum_new = momentum_new + step_size * grad_new
        momentum_new = momentum_new + 0.5 * step_size * grad_new
        momentum_new = -momentum_new

        new_H = -logp_new + 0.5 * torch.sum(momentum_new ** 2)
        log_alpha = -(new_H - current_H)
        if torch.log(torch.rand(1)) < log_alpha:
            current = theta_new.detach()
            current_logp = logp_new.detach()
            current_grad = grad_new.detach()
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())

        if step >= burn_in:
            samples.append(current.clone())
    acceptance = accept / total
    return torch.stack(samples) if samples else torch.empty(0), acceptance


def posterior_stats(model, sample_set, x_eval):
    if sample_set.numel() == 0:
        raise RuntimeError("Sampler returned no samples.")
    theta_ref = parameters_to_vector(model.parameters()).detach()
    preds = []
    with torch.no_grad():
        for sample in sample_set:
            vector_to_parameters(sample, model.parameters())
            preds.append(model(x_eval))
        vector_to_parameters(theta_ref, model.parameters())
    preds = torch.stack(preds)
    return preds.mean(0).cpu(), preds.std(0).cpu()


def visualise_results(X_grid, Y_grid, Z_true, stats):
    xg = X_grid.cpu()
    yg = Y_grid.cpu()
    fig = plt.figure(figsize=(18, 5 * (len(stats) + 1)))
    ax = fig.add_subplot(len(stats) + 1, 1, 1, projection="3d")
    ax.plot_surface(xg, yg, Z_true, cmap="viridis", alpha=0.8)
    ax.set_title("Ground Truth")

    for idx, (name, (mean, std)) in enumerate(stats.items(), start=2):
        ax_mean = fig.add_subplot(len(stats) + 1, 2, 2 * idx - 2, projection="3d")
        ax_std = fig.add_subplot(len(stats) + 1, 2, 2 * idx - 1, projection="3d")
        ax_mean.plot_surface(xg, yg, mean.reshape(xg.shape), cmap="viridis", alpha=0.8)
        ax_mean.set_title(f"{name} Mean")
        ax_std.plot_surface(xg, yg, std.reshape(xg.shape), cmap="hot", alpha=0.8)
        ax_std.set_title(f"{name} Std Dev")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rbf_mcmc_surfaces.png", dpi=150, bbox_inches="tight")
    print("Surface comparison saved!")


def visualise_contours(X_grid, Y_grid, Z_true, stats):
    xg = X_grid.cpu()
    yg = Y_grid.cpu()
    fig, axes = plt.subplots(len(stats) + 1, 2, figsize=(16, 5 * (len(stats) + 1)))
    c1 = axes[0, 0].contourf(xg, yg, Z_true, levels=20, cmap="viridis")
    axes[0, 0].set_title("Ground Truth")
    plt.colorbar(c1, ax=axes[0, 0])
    axes[0, 1].axis("off")

    for idx, (name, (mean, std)) in enumerate(stats.items(), start=1):
        mean_np = mean.reshape(xg.shape)
        std_np = std.reshape(xg.shape)
        c_mean = axes[idx, 0].contourf(xg, yg, mean_np, levels=20, cmap="viridis")
        axes[idx, 0].set_title(f"{name} Mean")
        plt.colorbar(c_mean, ax=axes[idx, 0])
        c_std = axes[idx, 1].contourf(xg, yg, std_np, levels=20, cmap="hot")
        axes[idx, 1].set_title(f"{name} Std Dev")
        plt.colorbar(c_std, ax=axes[idx, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rbf_mcmc_contours.png", dpi=150, bbox_inches="tight")
    print("Contour plots saved!")


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

    model = RBFKAN(layers_hidden=[2, 12, 12, 1], num_centers=12).to(device)
    map_losses = train_map(model, X_train, z_train)

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("RBF MAP Training Loss (2D)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "rbf_map_loss.png", dpi=150, bbox_inches="tight")

    stats = {}
    print("\nRunning Random-Walk Metropolis...")
    rwm_samples, acc_rwm = random_walk_metropolis(model, X_train, z_train)
    print(f"RWM acceptance rate: {acc_rwm:.2%}")
    stats["RandomWalkMetropolis"] = posterior_stats(model, rwm_samples, X_eval)

    print("\nRunning Hamiltonian Monte Carlo...")
    hmc_samples, acc_hmc = hmc_sampling(model, X_train, z_train)
    print(f"HMC acceptance rate: {acc_hmc:.2%}")
    stats["HamiltonianMonteCarlo"] = posterior_stats(model, hmc_samples, X_eval)

    visualise_results(X_grid, Y_grid, Z_true, stats)
    visualise_contours(X_grid, Y_grid, Z_true, stats)

    for name, (mean, _) in stats.items():
        mse = torch.mean((mean.reshape(n_test, n_test) - Z_true) ** 2).item()
        mae = torch.mean(torch.abs(mean.reshape(n_test, n_test) - Z_true)).item()
        print(f"{name} -> MSE: {mse:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("RBF KAN 2D MCMC (RWM & HMC)")
    print("=" * 60)

    run_experiment()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
