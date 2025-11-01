import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
from tqdm import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_mcmc_2d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BSpline(nn.Module):
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        num_ctrl = grid_size + spline_order
        self.coeffs = nn.Parameter(
            torch.randn(in_features, out_features, num_ctrl) * 0.05
        )
        self.base_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)

        h = 2.0 / grid_size
        self.register_buffer(
            "grid",
            torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                           grid_size + 2 * spline_order + 1)
        )

    def b_splines(self, x, k=0):
        if k == 0:
            x_expanded = x.unsqueeze(-1)
            return ((x_expanded >= self.grid[:-1].unsqueeze(0).unsqueeze(0)) &
                    (x_expanded < self.grid[1:].unsqueeze(0).unsqueeze(0))).float()

        prev = self.b_splines(x, k - 1)

        left_num = x.unsqueeze(-1) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
        left_den = self.grid[k:-1].unsqueeze(0).unsqueeze(0) - self.grid[:-k-1].unsqueeze(0).unsqueeze(0)
        left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
        left = (left_num / left_den) * prev[:, :, :-1]

        right_num = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - x.unsqueeze(-1)
        right_den = self.grid[k+1:].unsqueeze(0).unsqueeze(0) - self.grid[1:-k].unsqueeze(0).unsqueeze(0)
        right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
        right = (right_num / right_den) * prev[:, :, 1:]

        return left + right

    def forward(self, x):
        bases = []
        for i in range(self.in_features):
            basis_vals = self.b_splines(torch.tanh(x[:, i:i+1]), self.spline_order)
            if basis_vals.dim() == 3:
                basis_vals = basis_vals.squeeze(-2)
            bases.append(basis_vals)
        bases = torch.stack(bases, dim=1)
        activation = torch.einsum("bin,ion->bo", bases, self.coeffs)
        residual = x @ self.base_weight
        return activation + residual


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=8, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList([
            BSpline(layers_hidden[i], layers_hidden[i + 1], grid_size, spline_order)
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


def train_map(model, x, y, epochs=600, lr=1e-3, noise_var=0.05, prior_var=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
    losses = []
    progress = trange(epochs, desc="MAP Training 2D", leave=True)
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
    grad_vec = parameters_to_vector(grads)
    logp = (-nlp).detach()
    return logp, (-grad_vec).detach()


def random_walk_metropolis(model, x, y, n_samples=300, burn_in=150,
                           step_size=0.0015, noise_var=0.05, prior_var=1.0):
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
    acceptance_rate = accept / total
    return torch.stack(samples), acceptance_rate


def hmc_sampling(model, x, y, n_samples=300, burn_in=150,
                 step_size=0.002, n_leapfrog=10,
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
        for lf in range(n_leapfrog):
            theta_new = theta_new + step_size * momentum_new
            logp_new, grad_new = log_posterior_and_grad(model, theta_new, x, y, noise_var, prior_var)
            if lf != n_leapfrog - 1:
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

    acceptance_rate = accept / total
    return torch.stack(samples), acceptance_rate


def run_mcmc_2d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    model = KAN(layers_hidden=[2, 16, 16, 1], grid_size=8, spline_order=3).to(device)

    print("\nTraining MAP estimate...")
    map_losses = train_map(model, X_train, z_train)

    print("\nRunning Random-Walk Metropolis sampling...")
    rwm_samples, rwm_acc = random_walk_metropolis(
        model, X_train, z_train,
        n_samples=350, burn_in=200, step_size=0.0015,
        noise_var=0.05, prior_var=1.0
    )
    print(f"RWM acceptance rate: {rwm_acc:.2%}")

    print("\nRunning Hamiltonian Monte Carlo sampling...")
    hmc_samples, hmc_acc = hmc_sampling(
        model, X_train, z_train,
        n_samples=350, burn_in=200, step_size=0.0015, n_leapfrog=15,
        noise_var=0.05, prior_var=1.0
    )
    print(f"HMC acceptance rate: {hmc_acc:.2%}")

    theta_map = parameters_to_vector(model.parameters()).detach()
    def reconstruct(sample_set):
        preds = []
        with torch.no_grad():
            for sample in sample_set:
                vector_to_parameters(sample, model.parameters())
                preds.append(model(X_eval))
            vector_to_parameters(theta_map, model.parameters())
        preds = torch.stack(preds)
        return preds.mean(0).cpu().reshape(n_test, n_test), preds.std(0).cpu().reshape(n_test, n_test)

    mean_rwm, std_rwm = reconstruct(rwm_samples)
    mean_hmc, std_hmc = reconstruct(hmc_samples)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, cmap="viridis", alpha=0.8)
    ax1.set_title("Ground Truth")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X_grid.cpu(), Y_grid.cpu(), mean_rwm, cmap="viridis", alpha=0.8)
    ax2.set_title("RWM Mean")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X_grid.cpu(), Y_grid.cpu(), std_rwm, cmap="hot", alpha=0.8)
    ax3.set_title("RWM Std Dev")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_surfaces_rwm.png", dpi=150, bbox_inches="tight")
    print("\nRWM surface plots saved!")

    fig = plt.figure(figsize=(18, 5))
    bx1 = fig.add_subplot(131, projection="3d")
    bx1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, cmap="viridis", alpha=0.8)
    bx1.set_title("Ground Truth")

    bx2 = fig.add_subplot(132, projection="3d")
    bx2.plot_surface(X_grid.cpu(), Y_grid.cpu(), mean_hmc, cmap="viridis", alpha=0.8)
    bx2.set_title("HMC Mean")

    bx3 = fig.add_subplot(133, projection="3d")
    bx3.plot_surface(X_grid.cpu(), Y_grid.cpu(), std_hmc, cmap="hot", alpha=0.8)
    bx3.set_title("HMC Std Dev")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_surfaces_hmc.png", dpi=150, bbox_inches="tight")
    print("HMC surface plots saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Neg Log Posterior")
    plt.title("MAP Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_map_loss.png", dpi=150, bbox_inches="tight")
    print("MAP loss curve saved!")

    mse_rwm = torch.mean((mean_rwm - Z_true) ** 2).item()
    mae_rwm = torch.mean(torch.abs(mean_rwm - Z_true)).item()
    mse_hmc = torch.mean((mean_hmc - Z_true) ** 2).item()
    mae_hmc = torch.mean(torch.abs(mean_hmc - Z_true)).item()
    print(f"\nRWM metrics: MSE={mse_rwm:.6f}, MAE={mae_rwm:.6f}")
    print(f"HMC metrics: MSE={mse_hmc:.6f}, MAE={mae_hmc:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    levels = 20
    c1 = axes[0, 0].contourf(X_grid.cpu(), Y_grid.cpu(), Z_true, levels=levels, cmap="viridis")
    axes[0, 0].set_title("Ground Truth")
    plt.colorbar(c1, ax=axes[0, 0])

    c2 = axes[0, 1].contourf(X_grid.cpu(), Y_grid.cpu(), mean_rwm, levels=levels, cmap="viridis")
    axes[0, 1].set_title("RWM Mean")
    plt.colorbar(c2, ax=axes[0, 1])

    c3 = axes[0, 2].contourf(X_grid.cpu(), Y_grid.cpu(), std_rwm, levels=levels, cmap="hot")
    axes[0, 2].set_title("RWM Std Dev")
    plt.colorbar(c3, ax=axes[0, 2])

    c4 = axes[1, 0].contourf(X_grid.cpu(), Y_grid.cpu(), mean_hmc, levels=levels, cmap="viridis")
    axes[1, 0].set_title("HMC Mean")
    plt.colorbar(c4, ax=axes[1, 0])

    c5 = axes[1, 1].contourf(X_grid.cpu(), Y_grid.cpu(), std_hmc, levels=levels, cmap="hot")
    axes[1, 1].set_title("HMC Std Dev")
    plt.colorbar(c5, ax=axes[1, 1])

    diff = torch.abs(mean_hmc - mean_rwm)
    c6 = axes[1, 2].contourf(X_grid.cpu(), Y_grid.cpu(), diff, levels=levels, cmap="coolwarm")
    axes[1, 2].set_title("|HMC Mean - RWM Mean|")
    plt.colorbar(c6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_mcmc_contours_comparison.png", dpi=150, bbox_inches="tight")
    print("Contour plots saved!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 2D MCMC Inference")
    print("=" * 60)

    run_mcmc_2d()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
