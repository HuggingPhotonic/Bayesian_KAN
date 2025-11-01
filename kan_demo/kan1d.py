import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

OUTPUT_DIR = Path(__file__).parent / "results_1d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DeterministicBSpline1D(nn.Module):
    """Deterministic 1D B-spline module with trainable coefficients."""

    def __init__(self, n_basis=8, spline_order=3):
        super().__init__()
        self.n_basis = n_basis
        self.spline_order = spline_order

        total_bases = n_basis + spline_order
        self.coeffs = nn.Parameter(torch.randn(total_bases) * 0.05)
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
        bases = self._basis(x, self.spline_order)  # (batch, total_bases)
        spline = torch.matmul(bases, self.coeffs).unsqueeze(-1)
        residual = self.linear_weight * x + self.bias
        return spline + residual


class DeterministicKAN1D(nn.Module):
    """Stack of deterministic spline layers for 1D regression."""

    def __init__(self, n_layers=2, n_basis=8, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [DeterministicBSpline1D(n_basis, spline_order) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def target_function(x):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


def train_kan_1d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_train = 5000
    x_train = torch.linspace(-2, 2, n_train).unsqueeze(-1)
    y_train = target_function(x_train)
    X_train = x_train.to(device)
    y_train = y_train.to(device)

    x_test = torch.linspace(-2.5, 2.5, 100).unsqueeze(-1).to(device)
    y_test = target_function(x_test.cpu()).to(device)

    model = DeterministicKAN1D(n_layers=2, n_basis=10, spline_order=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    criterion = nn.MSELoss()

    epochs = 2000
    losses = []
    print("\nStart deterministic 1D training...")
    progress = trange(epochs, desc="KAN 1D Training", leave=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()

        preds = model(X_train)
        loss = criterion(preds, y_train)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    print("\nOptimised parameters (name -> shape):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {tuple(param.shape)}")

    model.eval()
    with torch.no_grad():
        y_mean = model(x_test).cpu().squeeze().numpy()

    x_test_cpu = x_test.cpu().squeeze().numpy()
    y_test_cpu = y_test.cpu().squeeze().numpy()
    x_train_cpu = X_train.cpu().squeeze().numpy()
    y_train_cpu = y_train.cpu().squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x_test_cpu, y_test_cpu, label="Ground Truth", linewidth=2, color="black")
    plt.plot(x_test_cpu, y_mean, label="KAN Prediction", linewidth=2, color="blue")
    plt.scatter(x_train_cpu, y_train_cpu, s=10, alpha=0.4, label="Train Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Deterministic KAN 1D Fit")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_fit.png", dpi=150, bbox_inches="tight")
    print("\n1D fit visualisation saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Deterministic KAN 1D Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_loss.png", dpi=150, bbox_inches="tight")
    print("Training loss curve saved!")

    mse = float(np.mean((y_mean - y_test_cpu) ** 2))
    mae = float(np.mean(np.abs(y_mean - y_test_cpu)))
    print(f"\nEvaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    return model, losses, mse, mae


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Deterministic KAN 1D Regression")
    print("=" * 60)

    train_kan_1d()

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)
