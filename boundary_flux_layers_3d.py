"""
Left-face 3D boundary flux on x*=0 with separable yz-layering and Gaussian time pulse.

q_x*(0, y*, z*, t*) = A_y(y*) * A_z(z*) * exp(-0.5 * ((t*-t_c*)/w*)^2)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


DEFAULT_FLUX_Y_EDGES_3D: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
DEFAULT_FLUX_Y_AMPS_3D: tuple[float, ...] = (0.40, 1.60, 0.45)
DEFAULT_FLUX_Z_EDGES_3D: tuple[float, ...] = (0.0, 0.5, 1.0)
DEFAULT_FLUX_Z_AMPS_3D: tuple[float, ...] = (0.70, 1.30)


def _layer_amp_numpy(v: np.ndarray, edges: tuple[float, ...], amps: tuple[float, ...]) -> np.ndarray:
    vv = np.asarray(v, dtype=float)
    ee = np.asarray(edges, dtype=float)
    idx = np.searchsorted(ee, vv, side="right") - 1
    idx = np.clip(idx, 0, len(amps) - 1)
    return np.asarray(amps, dtype=float)[idx]


def _layer_amp_torch(v: torch.Tensor, edges: tuple[float, ...], amps: tuple[float, ...]) -> torch.Tensor:
    ee = torch.tensor(edges, device=v.device, dtype=v.dtype)
    aa = torch.tensor(amps, device=v.device, dtype=v.dtype)
    idx = torch.searchsorted(ee, v, right=True) - 1
    idx = torch.clamp(idx, 0, len(amps) - 1)
    return aa[idx]


def yz_amp_numpy(
    y: np.ndarray,
    z: np.ndarray,
    y_edges: tuple[float, ...],
    y_amps: tuple[float, ...],
    z_edges: tuple[float, ...],
    z_amps: tuple[float, ...],
) -> np.ndarray:
    ay = _layer_amp_numpy(y, y_edges, y_amps)[:, None]
    az = _layer_amp_numpy(z, z_edges, z_amps)[None, :]
    return ay * az


def build_q_left_torch_3d(
    pulse_center: float,
    pulse_width: float,
    y_edges: tuple[float, ...],
    y_amps: tuple[float, ...],
    z_edges: tuple[float, ...],
    z_amps: tuple[float, ...],
    time_amp: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def q_left_torch(t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ay = _layer_amp_torch(y, y_edges, y_amps)
        az = _layer_amp_torch(z, z_edges, z_amps)
        g = time_amp * torch.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2)
        return ay * az * g

    return q_left_torch


def uniform_flux_edges_amps_3d() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return (0.0, 1.0), (1.0,), (0.0, 1.0), (1.0,)
