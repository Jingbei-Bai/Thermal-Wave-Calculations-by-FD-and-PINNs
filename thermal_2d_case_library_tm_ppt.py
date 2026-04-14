"""
2D finite-difference reference on [0,1]^2, aligned with `thermal_1d_case_library_tm_ppt.py` time marching.

Same nondimensional PDE structure as the 2D PINN (`pinn_2d_cv_tm_fourier_core.py`):

    θ*_t + ∂q_x*/∂x* + ∂q_y*/∂y* = 0

Boundary influx may be **y*-layered** (piecewise amplitude of the time Gaussian); medium stays isotropic.

Boundary (Case 0 / single_pulse style):
    x*=0: q_x* = q_L(t*) or A(y*) q_L(t*) if ``flux_y_edges`` / ``flux_layer_amps`` are set;
    x*=1: q_x* = 0;  y*=0,1: q_y* = 0.

Models: Fourier, CV, DPL, Thermomass — explicit sub-stepping analogous to the 1D script.

Example:
    python thermal_2d_case_library_tm_ppt.py
    from thermal_2d_case_library_tm_ppt import simulate_model_2d
    res = simulate_model_2d("Fourier", "single_pulse", nx=81, ny=81)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from boundary_flux_layers_2d import layer_amplitude_numpy
from thermal_1d_case_library_tm_ppt import grad_upwind, make_case


@dataclass
class Result2D:
    model: str
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    T: np.ndarray  # (n_saved, nx, ny) — θ*
    qx: np.ndarray
    qy: np.ndarray
    meta: dict


def grad_center_x(T: np.ndarray, dx: float) -> np.ndarray:
    gx = np.zeros_like(T)
    gx[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dx)
    gx[0, :] = (T[1, :] - T[0, :]) / dx
    gx[-1, :] = (T[-1, :] - T[-2, :]) / dx
    return gx


def grad_center_y(T: np.ndarray, dy: float) -> np.ndarray:
    gy = np.zeros_like(T)
    gy[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dy)
    gy[:, 0] = (T[:, 1] - T[:, 0]) / dy
    gy[:, -1] = (T[:, -1] - T[:, -2]) / dy
    return gy


def laplace_2d(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
    out[1:-1, 1:-1] += (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def grad_upwind_x(q: np.ndarray, w: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(q)
    for j in range(q.shape[1]):
        out[:, j] = grad_upwind(q[:, j], w[:, j], dx)
    return out


def grad_upwind_y(q: np.ndarray, w: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(q)
    for i in range(q.shape[0]):
        out[i, :] = grad_upwind(q[i, :], w[i, :], dy)
    return out


def div_q_2d(
    qx: np.ndarray,
    qy: np.ndarray,
    dx: float,
    dy: float,
    q_left: np.ndarray | float,
    q_right: float,
) -> np.ndarray:
    """∂q_x/∂x + ∂q_y/∂y; ``q_left`` may be scalar or length-ny vector (x=0 face)."""
    nx, ny = qx.shape
    if isinstance(q_left, (float, int, np.floating)):
        qL = np.full(ny, float(q_left), dtype=float)
    else:
        qL = np.asarray(q_left, dtype=float).reshape(ny)

    dqx = np.zeros_like(qx)
    dqy = np.zeros_like(qy)

    # interior (both directions central)
    dqx[1:-1, 1:-1] = (qx[2:, 1:-1] - qx[:-2, 1:-1]) / (2.0 * dx)
    dqy[1:-1, 1:-1] = (qy[1:-1, 2:] - qy[1:-1, :-2]) / (2.0 * dy)

    # x = 0, interior y
    dqx[0, 1:-1] = (qx[1, 1:-1] - qL[1:-1]) / dx
    dqy[0, 1:-1] = (qy[0, 2:] - qy[0, :-2]) / (2.0 * dy)

    # x = nx-1, interior y
    dqx[-1, 1:-1] = (q_right - qx[-2, 1:-1]) / dx
    dqy[-1, 1:-1] = (qy[-1, 2:] - qy[-1, :-2]) / (2.0 * dy)

    # y = 0, interior x
    dqy[1:-1, 0] = (qy[1:-1, 1] - 0.0) / dy
    dqx[1:-1, 0] = (qx[2:, 0] - qx[:-2, 0]) / (2.0 * dx)

    # y = ny-1, interior x
    dqy[1:-1, -1] = (0.0 - qy[1:-1, -2]) / dy
    dqx[1:-1, -1] = (qx[2:, -1] - qx[:-2, -1]) / (2.0 * dx)

    # corners
    dqx[0, 0] = (qx[1, 0] - qL[0]) / dx
    dqy[0, 0] = (qy[0, 1] - 0.0) / dy
    dqx[0, -1] = (qx[1, -1] - qL[-1]) / dx
    dqy[0, -1] = (0.0 - qy[0, -2]) / dy
    dqx[-1, 0] = (q_right - qx[-2, 0]) / dx
    dqy[-1, 0] = (qy[-1, 1] - 0.0) / dy
    dqx[-1, -1] = (q_right - qx[-2, -1]) / dx
    dqy[-1, -1] = (0.0 - qy[-1, -2]) / dy

    return dqx + dqy


def recommend_substeps_2d(model: str, dt: float, dx: float, dy: float, p: dict) -> int:
    inv_lap = 1.0 / (dx * dx) + 1.0 / (dy * dy)
    dxm = min(dx, dy)
    if model == "Thermomass":
        ch = math.sqrt(1.0 / max(p["tau_tm"], 1.0e-30))
        limits = [
            0.32 * p["tau_tm"],
            0.30 * dxm / max(ch, 1.0e-12),
            0.22 * dxm / max(p.get("tm_u_clip", 3.0), 0.1),
        ]
        return max(1, int(math.ceil(dt / min(limits))))

    limits = [0.20 / (2.0 * inv_lap)]
    if model in ("CV", "DPL"):
        limits.append(0.35 * p["tau_q"])
        ch = math.sqrt(1.0 / max(p["tau_q"], 1e-30))
        if ch > 0.0:
            limits.append(0.35 * dxm / ch)
    dt_lim = min(limits)
    return max(1, int(math.ceil(dt / dt_lim)))


def simulate_model_2d(
    model: str,
    case: str,
    nx: int = 81,
    ny: int = 81,
    flux_y_edges: tuple[float, ...] | None = None,
    flux_layer_amps: tuple[float, ...] | None = None,
) -> Result2D:
    """
    Run 2D FD with the same material / case setup as the 1D library (via ``make_case``).

    flux_y_edges / flux_layer_amps
        If both set: ``q_x*|_{x*=0} = A(y*) * q_L(t*)`` (Gaussian in time from case).
        If both None: uniform influx in ``y*`` (same as 1D scalar boundary).
    """
    p, qL_fn, qR_fn = make_case(case)
    if model not in ("Fourier", "CV", "DPL", "Thermomass"):
        raise ValueError(f"2D FD supports Fourier/CV/DPL/Thermomass, got {model}")
    if (flux_y_edges is None) ^ (flux_layer_amps is None):
        raise ValueError("Provide both flux_y_edges and flux_layer_amps, or neither.")

    dt = p["dt"]
    t_end = p["t_end"]
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    nt = int(np.floor(t_end / dt)) + 1
    t_full = np.linspace(0.0, dt * (nt - 1), nt)

    T = np.full((nx, ny), float(p["T0"]))
    qx = np.zeros((nx, ny))
    qy = np.zeros((nx, ny))
    gx_prev = np.zeros_like(T)
    gy_prev = np.zeros_like(T)

    nsub = recommend_substeps_2d(model, dt, dx, dy, p)
    dt_sub = dt / nsub
    p = dict(p)
    p["nsub_2d"] = nsub
    p["dt_sub_2d"] = dt_sub
    p["nx_2d"] = nx
    p["ny_2d"] = ny
    if flux_y_edges is not None:
        p["flux_y_edges"] = list(flux_y_edges)
        p["flux_layer_amps"] = list(flux_layer_amps)

    T_hist: list[np.ndarray] = []
    qx_hist: list[np.ndarray] = []
    qy_hist: list[np.ndarray] = []
    t_hist: list[float] = []

    for n, tn in enumerate(t_full):
        for j in range(nsub):
            t_now = tn + j * dt_sub
            base_flux = float(qL_fn(t_now))
            qR = float(qR_fn(t_now))
            if flux_layer_amps is None:
                qL_row = np.full(ny, base_flux, dtype=float)
            else:
                assert flux_y_edges is not None
                amp = layer_amplitude_numpy(y, flux_y_edges, flux_layer_amps)
                qL_row = amp * base_flux

            gx = grad_center_x(T, dx)
            gy = grad_center_y(T, dy)

            if model == "Fourier":
                qx_new = -gx
                qy_new = -gy
                qx_new[0, :] = qL_row
                qx_new[-1, :] = qR
                qy_new[:, 0] = 0.0
                qy_new[:, -1] = 0.0
                divv = div_q_2d(qx_new, qy_new, dx, dy, qL_row, qR)
                T_new = T - dt_sub * divv
            else:
                if model == "CV":
                    dqx_dt = (-gx - qx) / p["tau_q"]
                    dqy_dt = (-gy - qy) / p["tau_q"]
                elif model == "DPL":
                    dgrad_x = (gx - gx_prev) / dt_sub
                    dgrad_y = (gy - gy_prev) / dt_sub
                    dqx_dt = (-(gx + p["tau_T"] * dgrad_x) - qx) / p["tau_q"]
                    dqy_dt = (-(gy + p["tau_T"] * dgrad_y) - qy) / p["tau_q"]
                elif model == "Thermomass":
                    # strict PPT thermomass (2D component form); keep existing output paths unchanged.
                    # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): not used here.
                    eps = float(p["tm_epsilon"])
                    G_tm = float(p["tau_tm"])
                    tiny = 1.0e-12
                    Theta = 1.0 + eps * T
                    Theta_s = np.maximum(Theta, tiny)
                    ux = qx / Theta_s
                    uy = qy / Theta_s
                    div_theta = div_q_2d(qx, qy, dx, dy, qL_row, qR)
                    Theta_t = -eps * div_theta
                    Theta_x = eps * gx
                    Theta_y = eps * gy
                    qx_ux = grad_upwind_x(qx, ux, dx)
                    qy_uy = grad_upwind_y(qy, uy, dy)
                    rqx = ux * Theta_t - gx / G_tm - eps * ux * (qx_ux - ux * Theta_x)
                    rqy = uy * Theta_t - gy / G_tm - eps * uy * (qy_uy - uy * Theta_y)
                    qx_s = (qx + dt_sub * rqx) / (1.0 + dt_sub / G_tm)
                    qy_s = (qy + dt_sub * rqy) / (1.0 + dt_sub / G_tm)
                    dqx_dt = (qx_s - qx) / dt_sub
                    dqy_dt = (qy_s - qy) / dt_sub
                else:
                    raise ValueError(model)

                qx_new = qx + dt_sub * dqx_dt
                qy_new = qy + dt_sub * dqy_dt
                qx_new[0, :] = qL_row
                qx_new[-1, :] = qR
                qy_new[:, 0] = 0.0
                qy_new[:, -1] = 0.0
                divv = div_q_2d(qx_new, qy_new, dx, dy, qL_row, qR)
                T_new = T - dt_sub * divv

            T = T_new
            qx, qy = qx_new, qy_new
            gx_prev = gx.copy()
            gy_prev = gy.copy()

            if (not np.all(np.isfinite(T))) or (not np.all(np.isfinite(qx))) or (not np.all(np.isfinite(qy))):
                raise FloatingPointError(f"{model} 2D became non-finite; try smaller dt or coarser grid.")

        if n % p["save_every"] == 0:
            T_hist.append(T.copy())
            qx_hist.append(qx.copy())
            qy_hist.append(qy.copy())
            t_hist.append(float(tn))

    return Result2D(
        model=model,
        x=x,
        y=y,
        t=np.array(t_hist),
        T=np.stack(T_hist, axis=0),
        qx=np.stack(qx_hist, axis=0),
        qy=np.stack(qy_hist, axis=0),
        meta=p,
    )


def nearest_time_index(t_arr: np.ndarray, t_star: float) -> int:
    return int(np.argmin(np.abs(t_arr - t_star)))


def _main_demo() -> None:
    from pathlib import Path

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures.")
        return

    out = Path("results") / "case0_2d_fd_reference"
    out.mkdir(parents=True, exist_ok=True)

    for model in ("Fourier", "CV", "Thermomass"):
        res = simulate_model_2d(model, "single_pulse", nx=81, ny=81)
        np.savez_compressed(
            out / f"fd_2d_case0_{model.lower()}.npz",
            x=res.x,
            y=res.y,
            t=res.t,
            T=res.T,
            qx=res.qx,
            qy=res.qy,
        )
        print(f"saved {out / f'fd_2d_case0_{model.lower()}.npz'}")

        X, Y = np.meshgrid(res.x, res.y, indexing="ij")
        for ts in (0.04, 0.10, 0.20):
            it = min(nearest_time_index(res.t, ts), res.T.shape[0] - 1)
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            pcm = ax.pcolormesh(X.T, Y.T, res.T[it].T, shading="auto", cmap="inferno")
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$y^*$")
            ax.set_title(rf"FD 2D {model}  $\theta^*$ at $t^*={res.t[it]:.3f}$")
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out / f"fd2d_{model.lower()}_t_{res.t[it]:.3f}.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    _main_demo()
