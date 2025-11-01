import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from torch.nn.utils import parameters_to_vector, vector_to_parameters

OUTPUT_DIR = Path(__file__).parent / "results_mcmc_1d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BSpline1D(nn.Module):
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


def negative_log_posterior(model, x, y, noise_var=0.05, prior_var=1.0):
    preds = model(x)
    sse = torch.sum((preds - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def train_map(model, x, y, epochs=800, lr=1e-3, noise_var=0.05, prior_var=1.0):
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


def log_posterior(model, theta_vec, x, y, noise_var=0.05, prior_var=1.0):
    vector_to_parameters(theta_vec, model.parameters())
    with torch.no_grad():
        preds = model(x)
        sse = torch.sum((preds - y) ** 2)
        prior = torch.sum(theta_vec ** 2) / prior_var
        return -0.5 * sse / noise_var - 0.5 * prior


def random_walk_metropolis(model, x, y, n_samples=500, burn_in=200, step_size=0.01,
                           noise_var=0.05, prior_var=1.0):
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    current_lp = log_posterior(model, current, x, y, noise_var, prior_var)
    samples = []

    total_steps = burn_in + n_samples
    accept = 0
    for step in trange(total_steps, desc="MCMC Sampling", leave=True):
        proposal = current + step_size * torch.randn_like(current)
        proposal_lp = log_posterior(model, proposal, x, y, noise_var, prior_var)
        log_alpha = proposal_lp - current_lp
        if torch.log(torch.rand(1)) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accept += 1
        else:
            vector_to_parameters(current, model.parameters())
        if step >= burn_in:
            samples.append(current.clone())
    acceptance_rate = accept / total_steps
    return torch.stack(samples) if samples else torch.empty(0), acceptance_rate


def hmc_sampling(model, x, y, n_samples=400, burn_in=200,
                 step_size=0.01, n_leapfrog=15,
                 noise_var=0.05, prior_var=1.0):
    theta_map = parameters_to_vector(model.parameters()).detach()
    current = theta_map.clone()
    model.zero_grad()
    loss = negative_log_posterior(model, x, y, noise_var, prior_var)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    current_logp = (-loss).detach()
    current_grad = (-parameters_to_vector(grads)).detach()

    def logp_and_grad(theta_vec):
        vector_to_parameters(theta_vec, model.parameters())
        model.zero_grad()
        nlp = negative_log_posterior(model, x, y, noise_var, prior_var)
        grads = torch.autograd.grad(nlp, model.parameters())
        grad_vec = -parameters_to_vector(grads)
        return (-nlp).detach(), grad_vec.detach()

    samples = []
    accept = 0
    total_steps = burn_in + n_samples

    for step in trange(total_steps, desc="HMC Sampling", leave=True):
        theta = current.clone()
        grad = current_grad.clone()
        momentum = torch.randn_like(theta)
        current_H = -current_logp + 0.5 * torch.sum(momentum ** 2)

        theta_new = theta.clone()
        momentum_new = momentum.clone()

        momentum_new = momentum_new + 0.5 * step_size * grad
        for l in range(n_leapfrog):
            theta_new = theta_new + step_size * momentum_new
            logp_new, grad_new = logp_and_grad(theta_new)
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

    acceptance_rate = accept / total_steps
    return torch.stack(samples) if samples else torch.empty(0), acceptance_rate


def run_mcmc_1d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train = torch.linspace(-2, 2, 512).unsqueeze(-1).to(device)
    y_train = target_function(x_train).to(device)
    x_test = torch.linspace(-2.5, 2.5, 400).unsqueeze(-1).to(device)
    y_test = target_function(x_test.cpu()).to(device)

    model = DeterministicKAN1D(n_layers=2, n_basis=8, spline_order=3).to(device)

    print("\nTraining MAP estimate...")
    map_losses = train_map(model, x_train, y_train)

    print("\nRunning Random-Walk Metropolis sampling...")
    rwm_samples, rwm_acc = random_walk_metropolis(
        model, x_train, y_train,
        n_samples=600, burn_in=300, step_size=0.005,
        noise_var=0.05, prior_var=1.0
    )
    print(f"RWM acceptance rate: {rwm_acc:.2%}")

    print("\nRunning Hamiltonian Monte Carlo sampling...")
    hmc_samples, hmc_acc = hmc_sampling(
        model, x_train, y_train,
        n_samples=500, burn_in=250, step_size=0.01, n_leapfrog=20,
        noise_var=0.05, prior_var=1.0
    )
    print(f"HMC acceptance rate: {hmc_acc:.2%}")

    theta_map = parameters_to_vector(model.parameters()).detach()

    def reconstruct_from_samples(sample_set, label):
        if sample_set.numel() == 0:
            return None, None
        preds = []
        with torch.no_grad():
            for sample in sample_set:
                vector_to_parameters(sample, model.parameters())
                preds.append(model(x_test))
            vector_to_parameters(theta_map, model.parameters())
        preds = torch.stack(preds)
        mean_pred = preds.mean(0).cpu().squeeze().numpy()
        std_pred = preds.std(0).cpu().squeeze().numpy()
        return mean_pred, std_pred

    mean_rwm, std_rwm = reconstruct_from_samples(rwm_samples, "rwm")
    mean_hmc, std_hmc = reconstruct_from_samples(hmc_samples, "hmc")

    if mean_rwm is None or mean_hmc is None:
        print("Sampling failed; no samples collected.")
        return

    x_train_cpu = x_train.cpu().squeeze().numpy()
    y_train_cpu = y_train.cpu().squeeze().numpy()
    x_test_cpu = x_test.cpu().squeeze().numpy()
    y_test_cpu = y_test.cpu().squeeze().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_test_cpu, y_test_cpu, label="Ground Truth", color="black", linewidth=2)
    plt.plot(x_test_cpu, mean_rwm, label="RWM Mean", color="green", linewidth=2)
    plt.fill_between(x_test_cpu,
                     mean_rwm - 2 * std_rwm,
                     mean_rwm + 2 * std_rwm,
                     alpha=0.2, color="green", label="RWM ±2σ")
    plt.scatter(x_train_cpu, y_train_cpu, s=10, alpha=0.3)
    plt.legend()
    plt.title("Random-Walk Metropolis Posterior Predictive")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(x_test_cpu, y_test_cpu, label="Ground Truth", color="black", linewidth=2)
    plt.plot(x_test_cpu, mean_hmc, label="HMC Mean", color="blue", linewidth=2)
    plt.fill_between(x_test_cpu,
                     mean_hmc - 2 * std_hmc,
                     mean_hmc + 2 * std_hmc,
                     alpha=0.2, color="blue", label="HMC ±2σ")
    plt.scatter(x_train_cpu, y_train_cpu, s=10, alpha=0.3)
    plt.legend()
    plt.title("Hamiltonian Monte Carlo Posterior Predictive")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kan_1d_mcmc_comparison.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved!")

    plt.figure(figsize=(8, 4))
    plt.plot(map_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Posterior")
    plt.title("MAP Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "kan_1d_mcmc_map_loss.png", dpi=150, bbox_inches="tight")
    print("MAP loss curve saved!")

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("KAN 1D MCMC Inference")
    print("=" * 60)

    run_mcmc_1d()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
