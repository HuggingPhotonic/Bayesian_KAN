import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_laplace_1d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BSpline1D(nn.Module):
    """Deterministic 1D B-spline with trainable coefficients."""

    def __init__(self, n_basis=8, spline_order=3):
        super().__init__()
        self.n_basis = n_basis
        self.spline_order = spline_order

        self.num_bases = n_basis + spline_order
        self.coeffs = nn.Parameter(torch.randn(self.num_bases) * 0.05)
        self.linear_weight = nn.Parameter(torch.randn(1) * 0.05)
        self.bias = nn.Parameter(torch.zeros(1))

        h = 2.0 / n_basis
        self.register_buffer(
            "knots",
            torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                           n_basis + 2 * spline_order + 1)
        )

    def _basis(self, x, degree):
        x_flat = x.squeeze(-1)
        knots = self.knots
        device = x.device
        n_knots = knots.numel()

        basis = torch.zeros(x_flat.shape[0], n_knots - 1, device=device)
        for i in range(n_knots - 1):
            left = knots[i]
            right = knots[i + 1]
            mask = (x_flat >= left) & (x_flat < right)
            if i == n_knots - 2:
                mask |= (x_flat == right)
            basis[:, i] = mask.float()

        for k in range(1, degree + 1):
            new_basis = torch.zeros(x_flat.shape[0], n_knots - k - 1, device=device)
            for i in range(n_knots - k - 1):
                denom1 = knots[i + k] - knots[i]
                term1 = torch.zeros_like(x_flat)
                if denom1 != 0:
                    term1 = ((x_flat - knots[i]) / denom1) * basis[:, i]

                denom2 = knots[i + k + 1] - knots[i + 1]
                term2 = torch.zeros_like(x_flat)
                if denom2 != 0:
                    term2 = ((knots[i + k + 1] - x_flat) / denom2) * basis[:, i + 1]

                new_basis[:, i] = term1 + term2
            basis = new_basis
        return basis

    def forward(self, x):
        bases = self._basis(x, self.spline_order)
        spline = torch.matmul(bases, self.coeffs).unsqueeze(-1)
        residual = self.linear_weight * x + self.bias
        return spline + residual


class DeterministicKAN1D(nn.Module):
    """Stacked spline layers for 1D regression."""

    def __init__(self, n_layers=2, n_basis=8, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [BSpline1D(n_basis, spline_order) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def target_function(x):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


def negative_log_posterior(model, x, y, noise_var, prior_var):
    preds = model(x)
    sse = torch.sum((preds - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def train_map(model, x, y, noise_var=0.05, prior_var=1.0, epochs=800, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    losses = []
    progress = trange(epochs, desc="MAP Training", leave=True)
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
        second_grads = torch.autograd.grad(grad_comp, params, retain_graph=True)
        diag.append(parameters_to_vector(second_grads)[i])
    return torch.stack(diag).detach()


def laplace_posterior_samples(model, cov_diag, n_samples, device):
    theta_map = parameters_to_vector([p for p in model.parameters()]).detach()
    std = torch.sqrt(1.0 / cov_diag.clamp(min=1e-6))
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            sample_vec = theta_map + torch.randn_like(theta_map) * std
            vector_to_parameters(sample_vec, model.parameters())
            preds.append(model(device))
    vector_to_parameters(theta_map, model.parameters())
    return torch.stack(preds)


def run_laplace_1d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train = torch.linspace(-2, 2, 512).unsqueeze(-1).to(device)
    y_train = target_function(x_train).to(device)
    x_test = torch.linspace(-2.5, 2.5, 400).unsqueeze(-1).to(device)
    y_test = target_function(x_test.cpu()).to(device)

    model = DeterministicKAN1D(n_layers=2, n_basis=8, spline_order=3).to(device)

    print("\nTraining MAP estimate...")
    map_losses = train_map(model, x_train, y_train)

    print("\nComputing Laplace approximation...")
    noise_var = 0.05
    prior_var = 1.0
    diag = hessian_diag(model, x_train, y_train, noise_var, prior_var)
    posterior_var = 1.0 / diag.clamp(min=1e-6)

    with torch.no_grad():
        theta_map = parameters_to_vector(model.parameters()).detach()
        preds = []
        for _ in range(200):
            sample_vec = theta_map + torch.randn_like(theta_map) * torch.sqrt(posterior_var)
            vector_to_parameters(sample_vec, model.parameters())
            preds.append(model(x_test))
        vector_to_parameters(theta_map, model.parameters())
        preds = torch.stack(preds)
        mean_pred = preds.mean(0).cpu().squeeze().numpy()
        std_pred = preds.std(0).cpu().squeeze().numpy()

    x_train_cpu = x_train.cpu().squeeze().numpy()
    y_train_cpu = y_train.cpu().squeeze().numpy()
    x_test_cpu = x_test.cpu().squeeze().numpy()
    y_test_cpu = y_test.cpu().squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x_test_cpu, y_test_cpu, label="Ground Truth", color="black", linewidth=2)
    plt.plot(x_test_cpu, mean_pred, label="Laplace Mean", color="blue", linewidth=2)
    plt.fill_between(x_test_cpu,
                     mean_pred - 2 * std_pred,
                     mean_pred + 2 * std_pred,
                     alpha=0.2, color="blue", label="±2σ")
    plt.scatter(x_train_cpu, y_train_cpu, s=10, alpha=0.4, label="Train Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KAN 1D Laplace Approximation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_laplace_fit.png", dpi=150, bbox_inches="tight")
    print("\nLaplace fit visualisation saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("MAP Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_laplace_map_loss.png", dpi=150, bbox_inches="tight")
    print("MAP loss curve saved!")

    mse = float(np.mean((mean_pred - y_test_cpu) ** 2))
    mae = float(np.mean(np.abs(mean_pred - y_test_cpu)))
    print(f"\nEvaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 1D Laplace Approximation")
    print("=" * 60)

    run_laplace_1d()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
