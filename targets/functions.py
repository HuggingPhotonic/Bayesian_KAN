import torch
from typing import Callable, Dict, Tuple


def target_1d_sin_cos(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)


def target_2d_wave(x: torch.Tensor) -> torch.Tensor:
    # x: (batch, 2)
    xi, yi = x[:, 0:1], x[:, 1:2]
    return (torch.sin(torch.pi * xi) * torch.cos(torch.pi * yi) +
            0.3 * torch.exp(-(xi ** 2 + yi ** 2)) +
            0.2 * xi * yi +
            0.1 * torch.sin(3 * xi) * torch.sin(3 * yi))


TARGETS: Dict[Tuple[int, str], Callable[[torch.Tensor], torch.Tensor]] = {
    (1, "sin_cos"): target_1d_sin_cos,
    (2, "wave"): target_2d_wave,
}


def get_target_function(dim: int, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        return TARGETS[(dim, name)]
    except KeyError as exc:
        raise ValueError(f"Unknown target function for dim={dim}, name={name}") from exc
