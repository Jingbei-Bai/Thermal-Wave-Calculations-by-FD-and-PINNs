"""
Left-boundary heat flux q_x*(0, y*, t*) with a Gaussian time pulse and y*-layered peak amplitude.

    q_x*|_{x*=0} = A(y*) * exp( -0.5 ((t*-t_c*)/w*)^2 ) * q_time_ref

No material layering — only the **influx peak** varies by y* (piecewise constant bands).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

# Default: three bands, strong pulse in the middle strip (contrasts with weak top/bottom).
DEFAULT_FLUX_Y_EDGES: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
DEFAULT_FLUX_LAYER_AMPS: tuple[float, ...] = (0.38, 1.62, 0.42)


def layer_amplitude_numpy(y: np.ndarray, y_edges: tuple[float, ...], amps: tuple[float, ...]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    edges = np.array(y_edges, dtype=float)
    n = len(amps)
    idx = np.searchsorted(edges, y, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    return np.array(amps, dtype=float)[idx]


def layer_amplitude_torch(
    y: torch.Tensor,
    y_edges: tuple[float, ...],
    amps: tuple[float, ...],
) -> torch.Tensor:
    dev, dtype = y.device, y.dtype
    edges = torch.tensor(y_edges, device=dev, dtype=dtype)
    a_t = torch.tensor(amps, device=dev, dtype=dtype)
    idx = torch.searchsorted(edges, y, right=True) - 1
    idx = torch.clamp(idx, 0, len(amps) - 1)
    return a_t[idx]


def build_q_left_torch(
    pulse_center: float,
    pulse_width: float,
    y_edges: tuple[float, ...],
    layer_amps: tuple[float, ...],
    time_amp: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns ``q_left_torch(t, y)`` for PINN Dirichlet BC on x*=0."""

    def q_left_torch(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = layer_amplitude_torch(y, y_edges, layer_amps)
        g = time_amp * torch.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2)
        return a * g

    return q_left_torch


def uniform_flux_edges_amps() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.0, 1.0), (1.0,)
