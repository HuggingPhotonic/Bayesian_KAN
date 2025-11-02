from __future__ import annotations

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Tuple


def _model_output(model, x):
    output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    return output


def negative_log_posterior(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: float,
    prior_var: float,
) -> torch.Tensor:
    output = _model_output(model, x)
    sse = torch.sum((output - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def negative_log_posterior_grad(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: float,
    prior_var: float,
) -> torch.Tensor:
    preds = _model_output(model, x)
    sse = torch.sum((preds - y) ** 2)
    theta = parameters_to_vector(model.parameters())
    prior = torch.sum(theta ** 2) / prior_var
    return 0.5 * sse / noise_var + 0.5 * prior


def log_posterior_and_grad(
    model: torch.nn.Module,
    theta_vec: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_var: float,
    prior_var: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vector_to_parameters(theta_vec, model.parameters())
    model.zero_grad()
    nlp = negative_log_posterior_grad(model, x, y, noise_var, prior_var)
    grads = torch.autograd.grad(nlp, model.parameters())
    grad_vec = -parameters_to_vector(grads)
    return (-nlp).detach(), grad_vec.detach()


def posterior_stats(
    model: torch.nn.Module,
    samples: torch.Tensor,
    x_eval: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples.numel() == 0:
        raise RuntimeError("Sampler produced no samples.")
    theta_ref = parameters_to_vector(model.parameters()).detach()
    preds = []
    with torch.no_grad():
        for sample in samples:
            vector_to_parameters(sample, model.parameters())
            preds.append(_model_output(model, x_eval))
        vector_to_parameters(theta_ref, model.parameters())
    stacked = torch.stack(preds)
    return stacked.mean(0), stacked.std(0)
