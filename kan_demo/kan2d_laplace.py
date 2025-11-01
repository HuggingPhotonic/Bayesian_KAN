import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
from tqdm import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_laplace_2d"
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


def hessian_diag(model, x, y, noise_var, prior_var):
    loss = negative_log_posterior(model, x, y, noise_var, prior_var)
    params = [p for p in model.parameters()]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = parameters_to_vector(grads)
    diag = []
    for i in range(grad_vec.numel()):
        grad_comp = grad_vec[i]
        second = torch.autograd.grad(grad_comp, params, retain_graph=True)
        diag.append(parameters_to_vector(second)[i])
    return torch.stack(diag).detach()


def run_laplace_2d():
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

    print("\nComputing Laplace approximation...")
    noise_var = 0.05
    prior_var = 1.0
    diag = hessian_diag(model, X_train, z_train, noise_var, prior_var)
    post_var = 1.0 / diag.clamp(min=1e-6)

    theta_map = parameters_to_vector(model.parameters()).detach()
    preds = []
    with torch.no_grad():
        for _ in range(150):
            sample_vec = theta_map + torch.randn_like(theta_map) * torch.sqrt(post_var)
            vector_to_parameters(sample_vec, model.parameters())
            preds.append(model(X_eval))
        vector_to_parameters(theta_map, model.parameters())
    preds = torch.stack(preds)
    mean_pred = preds.mean(0).cpu().reshape(n_test, n_test)
    std_pred = preds.std(0).cpu().reshape(n_test, n_test)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X_grid.cpu(), Y_grid.cpu(), Z_true, cmap="viridis", alpha=0.8)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X_grid.cpu(), Y_grid.cpu(), mean_pred, cmap="viridis", alpha=0.8)
    ax2.set_title("Laplace Mean")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X_grid.cpu(), Y_grid.cpu(), std_pred, cmap="hot", alpha=0.8)
    ax3.set_title("Predictive Std Dev")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("σ")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_2d_laplace_surfaces.png", dpi=150, bbox_inches="tight")
    print("\nSurface plots saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Neg Log Posterior")
    plt.title("MAP Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_2d_laplace_map_loss.png", dpi=150, bbox_inches="tight")
    print("MAP loss curve saved!")

    mse = torch.mean((mean_pred - Z_true) ** 2).item()
    mae = torch.mean(torch.abs(mean_pred - Z_true)).item()
    print(f"\nEvaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

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
    plt.savefig(OUTPUT_DIR / "kan_2d_laplace_contours.png", dpi=150, bbox_inches="tight")
    print("Contour plots saved!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 2D Laplace Approximation")
    print("=" * 60)

    run_laplace_2d()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
