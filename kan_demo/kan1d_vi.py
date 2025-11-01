import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

OUTPUT_DIR = Path(__file__).parent / "results_vi_1d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BayesianBSpline1D(nn.Module):
    """1D B-spline with variational coefficients."""

    def __init__(self, n_basis=10, spline_order=5, prior_scale=5):
        super().__init__()
        self.n_basis = n_basis + spline_order
        self.spline_order = spline_order

        self.coeff_mean = nn.Parameter(torch.randn(self.n_basis) * 0.05)
        self.coeff_log_var = nn.Parameter(torch.full((self.n_basis,), -5.0))
        self.register_buffer("prior_mean", torch.zeros_like(self.coeff_mean))
        self.register_buffer("prior_var", torch.ones_like(self.coeff_mean) * prior_scale ** 2)

        h = 2.0 / n_basis
        self.register_buffer(
            "grid",
            torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                           n_basis + 2 * spline_order + 1)
        )

    def b_splines(self, x, k=0):
        x_flat = x.squeeze(-1)
        knots = self.grid
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

        for degree in range(1, k + 1):
            new_basis = torch.zeros(x_flat.shape[0], n_knots - degree - 1, device=device)
            for i in range(n_knots - degree - 1):
                denom1 = knots[i + degree] - knots[i]
                term1 = torch.zeros_like(x_flat)
                if denom1 != 0:
                    term1 = ((x_flat - knots[i]) / denom1) * basis[:, i]

                denom2 = knots[i + degree + 1] - knots[i + 1]
                term2 = torch.zeros_like(x_flat)
                if denom2 != 0:
                    term2 = ((knots[i + degree + 1] - x_flat) / denom2) * basis[:, i + 1]

                new_basis[:, i] = term1 + term2
            basis = new_basis

        return basis

    def forward(self, x, sample=True, n_samples=1):
        x_normalized = torch.tanh(x)
        bases = self.b_splines(x_normalized, self.spline_order)  # (batch, n_basis)

        if sample:
            std = torch.exp(0.5 * self.coeff_log_var)
            eps = torch.randn(n_samples, *std.shape, device=x.device)
            coeffs = self.coeff_mean + eps * std
            outputs = torch.matmul(bases, coeffs.T)  # (batch, n_samples)
            spline_out = outputs.mean(dim=-1, keepdim=True)
        else:
            spline_out = torch.matmul(bases, self.coeff_mean).unsqueeze(-1)

        kl = self.kl_divergence()
        return spline_out, kl

    def kl_divergence(self):
        var = torch.exp(self.coeff_log_var)
        kl = 0.5 * torch.sum(
            (var + (self.coeff_mean - self.prior_mean) ** 2) / self.prior_var
            - 1.0 - torch.log(var / self.prior_var)
        )
        return kl


class BayesianKAN1D(nn.Module):
    def __init__(self, layer_sizes, n_basis=6, spline_order=3, prior_scale=1.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(BayesianBSpline1D(n_basis, spline_order, prior_scale))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True, n_samples=1):
        total_kl = 0.0
        for layer in self.layers:
            x, kl = layer(x, sample=sample, n_samples=n_samples)
            total_kl = total_kl + kl
        return x, total_kl

    def predict_with_uncertainty(self, x, n_samples=100):
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.forward(x, sample=True, n_samples=1)
                preds.append(pred)
        stacked = torch.stack(preds)
        return stacked.mean(0), stacked.std(0)


def target_function(x):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


def train_vi_1d():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_train = 5000
    x_train = torch.linspace(-2, 2, n_train).unsqueeze(-1)
    y_train = target_function(x_train)
    X_train = x_train.to(device)
    y_train = y_train.to(device)

    x_test = torch.linspace(-2.5, 2.5, 1000).unsqueeze(-1).to(device)
    y_test = target_function(x_test.cpu()).to(device)

    model = BayesianKAN1D(layer_sizes=[1, 1], n_basis=50, spline_order=10, prior_scale=0.5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    mse_loss = nn.MSELoss()
    kl_max = 5e-3
    kl_warmup_epochs = 100

    epochs = 2000
    losses = []
    print("\nStart 1D VI training...")
    progress = trange(epochs, desc="VI Training 1D", leave=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()

        preds, kl = model(X_train, sample=True, n_samples=100)
        recon = mse_loss(preds, y_train)
        # alpha = 0.8
        kl_weight = kl_max * min(1.0, (epoch + 1) / kl_warmup_epochs)
        loss = recon + kl_weight * kl / X_train.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        progress.set_postfix(recon=recon.item(),
                             kl=kl.item(),
                             beta=kl_weight,
                             loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])

    print("\nOptimised variational parameters (name -> shape):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {tuple(param.shape)}")

    model.eval()
    with torch.no_grad():
        mean_pred, std_pred = model.predict_with_uncertainty(x_test, n_samples=200)

    x_test_cpu = x_test.cpu().squeeze().numpy()
    mean_cpu = mean_pred.cpu().squeeze().numpy()
    std_cpu = std_pred.cpu().squeeze().numpy()
    y_test_cpu = y_test.cpu().squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x_test_cpu, y_test_cpu, label="Ground Truth", linewidth=2, color="black")
    plt.plot(x_test_cpu, mean_cpu, label="Mean Prediction", linewidth=2, color="blue")
    plt.fill_between(x_test_cpu,
                     mean_cpu - 2 * std_cpu,
                     mean_cpu + 2 * std_cpu,
                     color="blue", alpha=0.2, label="±2σ")
    plt.scatter(X_train.cpu().squeeze().numpy(),
                y_train.cpu().squeeze().numpy(),
                s=10, alpha=0.5, label="Train Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Bayesian KAN 1D Fit with Uncertainty")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "bayesian_kan_1d_fit.png", dpi=150, bbox_inches="tight")
    print("\n1D fit visualisation saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("ELBO Loss")
    plt.title("Bayesian KAN 1D VI Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "bayesian_kan_1d_loss.png", dpi=150, bbox_inches="tight")
    print("Training loss curve saved!")

    mse = torch.mean((mean_pred - y_test) ** 2).item()
    mae = torch.mean(torch.abs(mean_pred - y_test)).item()
    print(f"\nEvaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    return model, losses, mse, mae


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Bayesian KAN 1D Variational Inference")
    print("=" * 60)

    train_vi_1d()

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)
