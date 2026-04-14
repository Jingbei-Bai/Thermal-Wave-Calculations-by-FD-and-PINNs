"""
3D finite-difference reference on [0,1]^3, same nondimensional models as `thermal_2d_case_library_tm_ppt.py`.

    θ*_t + ∂q_x*/∂x* + ∂q_y*/∂y* + ∂q_z*/∂z* = 0

Boundary (Case 0 extension):
    x*=0: q_x* = q_L(t*) or A(y*) q_L(t*) (same y*-layering as 2D; uniform in z*),
    x*=1: q_x* = 0;  y*,z* ∈ {0,1}: q_y* = q_z* = 0 (insulated).

Models: Fourier, CV, DPL, Thermomass — explicit sub-steps like 1D/2D.

Example:
    from thermal_3d_case_library_tm_ppt import simulate_model_3d
    res = simulate_model_3d("Fourier", "single_pulse", nx=41, ny=41, nz=41)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from boundary_flux_layers_2d import layer_amplitude_numpy
from thermal_1d_case_library_tm_ppt import grad_upwind, make_case


@dataclass
class Result3D:
    model: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray
    T: np.ndarray  # (n_saved, nx, ny, nz)
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    meta: dict


def grad_center_x(T: np.ndarray, dx: float) -> np.ndarray:
    gx = np.zeros_like(T)
    gx[1:-1, :, :] = (T[2:, :, :] - T[:-2, :, :]) / (2.0 * dx)
    gx[0, :, :] = (T[1, :, :] - T[0, :, :]) / dx
    gx[-1, :, :] = (T[-1, :, :] - T[-2, :, :]) / dx
    return gx


def grad_center_y(T: np.ndarray, dy: float) -> np.ndarray:
    gy = np.zeros_like(T)
    gy[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2.0 * dy)
    gy[:, 0, :] = (T[:, 1, :] - T[:, 0, :]) / dy
    gy[:, -1, :] = (T[:, -1, :] - T[:, -2, :]) / dy
    return gy


def grad_center_z(T: np.ndarray, dz: float) -> np.ndarray:
    gz = np.zeros_like(T)
    gz[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2.0 * dz)
    gz[:, :, 0] = (T[:, :, 1] - T[:, :, 0]) / dz
    gz[:, :, -1] = (T[:, :, -1] - T[:, :, -2]) / dz
    return gz


def laplace_3d(u: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1, 1:-1] = (
        (u[2:, 1:-1, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / (dx * dx)
        + (u[1:-1, 2:, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / (dy * dy)
        + (u[1:-1, 1:-1, 2:] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / (dz * dz)
    )
    out[0, :, :] = out[1, :, :]
    out[-1, :, :] = out[-2, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -1, :] = out[:, -2, :]
    out[:, :, 0] = out[:, :, 1]
    out[:, :, -1] = out[:, :, -2]
    return out


def grad_upwind_x(q: np.ndarray, w: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(q)
    for j in range(q.shape[1]):
        for k in range(q.shape[2]):
            out[:, j, k] = grad_upwind(q[:, j, k], w[:, j, k], dx)
    return out


def grad_upwind_y(q: np.ndarray, w: np.ndarray, dy: float) -> np.ndarray:
    out = np.zeros_like(q)
    for i in range(q.shape[0]):
        for k in range(q.shape[2]):
            out[i, :, k] = grad_upwind(q[i, :, k], w[i, :, k], dy)
    return out


def grad_upwind_z(q: np.ndarray, w: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(q)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            out[i, j, :] = grad_upwind(q[i, j, :], w[i, j, :], dz)
    return out


def div_q_3d(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    q_left_face: np.ndarray,
    q_right: float,
) -> np.ndarray:
    """∇·q with q_x prescribed on x=0 face (array shape (ny, nz))."""
    nx, ny, nz = qx.shape
    qL = np.asarray(q_left_face, dtype=float)
    if qL.shape == (ny, nz):
        pass
    elif qL.size == ny * nz:
        qL = qL.reshape(ny, nz)
    elif qL.shape == (ny, 1) or (qL.ndim == 1 and qL.size == ny):
        qL = np.broadcast_to(qL.reshape(ny, 1), (ny, nz))
    else:
        raise ValueError(f"q_left_face must be (ny,nz)=({ny},{nz}) or broadcastable from (ny,), got {qL.shape}")

    dqx_dx = np.zeros_like(qx)
    dqx_dx[1:-1, :, :] = (qx[2:, :, :] - qx[:-2, :, :]) / (2.0 * dx)
    dqx_dx[0, :, :] = (qx[1, :, :] - qL) / dx
    dqx_dx[-1, :, :] = (q_right - qx[-2, :, :]) / dx

    dqy_dy = np.zeros_like(qy)
    dqy_dy[:, 1:-1, :] = (qy[:, 2:, :] - qy[:, :-2, :]) / (2.0 * dy)
    dqy_dy[:, 0, :] = (qy[:, 1, :] - 0.0) / dy
    dqy_dy[:, -1, :] = (0.0 - qy[:, -2, :]) / dy

    dqz_dz = np.zeros_like(qz)
    dqz_dz[:, :, 1:-1] = (qz[:, :, 2:] - qz[:, :, :-2]) / (2.0 * dz)
    dqz_dz[:, :, 0] = (qz[:, :, 1] - 0.0) / dz
    dqz_dz[:, :, -1] = (0.0 - qz[:, :, -2]) / dz

    return dqx_dx + dqy_dy + dqz_dz


def recommend_substeps_3d(model: str, dt: float, dx: float, dy: float, dz: float, p: dict) -> int:
    inv_lap = 1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)
    dxm = min(dx, dy, dz)
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


def simulate_model_3d(
    model: str,
    case: str,
    nx: int = 41,
    ny: int = 41,
    nz: int = 41,
    flux_y_edges: tuple[float, ...] | None = None,
    flux_layer_amps: tuple[float, ...] | None = None,
) -> Result3D:
    """
    3D FD on the unit cube. Left face influx uses the same optional ``A(y*)`` as 2D (constant along z*).
    """
    p, qL_fn, qR_fn = make_case(case)
    if model not in ("Fourier", "CV", "DPL", "Thermomass"):
        raise ValueError(f"3D FD supports Fourier/CV/DPL/Thermomass, got {model}")
    if (flux_y_edges is None) ^ (flux_layer_amps is None):
        raise ValueError("Provide both flux_y_edges and flux_layer_amps, or neither.")

    dt = p["dt"]
    t_end = p["t_end"]
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    nt = int(np.floor(t_end / dt)) + 1
    t_full = np.linspace(0.0, dt * (nt - 1), nt)

    T = np.full((nx, ny, nz), float(p["T0"]))
    qx = np.zeros((nx, ny, nz))
    qy = np.zeros((nx, ny, nz))
    qz = np.zeros((nx, ny, nz))
    gx_prev = np.zeros_like(T)
    gy_prev = np.zeros_like(T)
    gz_prev = np.zeros_like(T)

    nsub = recommend_substeps_3d(model, dt, dx, dy, dz, p)
    dt_sub = dt / nsub
    p = dict(p)
    p["nsub_3d"] = nsub
    p["dt_sub_3d"] = dt_sub
    p["nx_3d"] = nx
    p["ny_3d"] = ny
    p["nz_3d"] = nz
    if flux_y_edges is not None:
        p["flux_y_edges"] = list(flux_y_edges)
        p["flux_layer_amps"] = list(flux_layer_amps)

    T_hist: list[np.ndarray] = []
    qx_hist: list[np.ndarray] = []
    qy_hist: list[np.ndarray] = []
    qz_hist: list[np.ndarray] = []
    t_hist: list[float] = []

    for n, tn in enumerate(t_full):
        for j in range(nsub):
            t_now = tn + j * dt_sub
            base_flux = float(qL_fn(t_now))
            qR = float(qR_fn(t_now))
            if flux_layer_amps is None:
                qL_face = np.full((ny, nz), base_flux, dtype=float)
            else:
                assert flux_y_edges is not None
                amp_y = layer_amplitude_numpy(y, flux_y_edges, flux_layer_amps)
                # (ny, 1) * scalar is only (ny, 1); div_q_3d needs full (ny, nz) face
                qL_face = np.broadcast_to(
                    (amp_y * base_flux)[:, np.newaxis],
                    (ny, nz),
                )

            gx = grad_center_x(T, dx)
            gy = grad_center_y(T, dy)
            gz = grad_center_z(T, dz)

            if model == "Fourier":
                qx_new = -gx
                qy_new = -gy
                qz_new = -gz
                qx_new[0, :, :] = qL_face
                qx_new[-1, :, :] = qR
                qy_new[:, 0, :] = 0.0
                qy_new[:, -1, :] = 0.0
                qz_new[:, :, 0] = 0.0
                qz_new[:, :, -1] = 0.0
                divv = div_q_3d(qx_new, qy_new, qz_new, dx, dy, dz, qL_face, qR)
                T_new = T - dt_sub * divv
            else:
                if model == "CV":
                    dqx_dt = (-gx - qx) / p["tau_q"]
                    dqy_dt = (-gy - qy) / p["tau_q"]
                    dqz_dt = (-gz - qz) / p["tau_q"]
                elif model == "DPL":
                    dgrad_x = (gx - gx_prev) / dt_sub
                    dgrad_y = (gy - gy_prev) / dt_sub
                    dgrad_z = (gz - gz_prev) / dt_sub
                    dqx_dt = (-(gx + p["tau_T"] * dgrad_x) - qx) / p["tau_q"]
                    dqy_dt = (-(gy + p["tau_T"] * dgrad_y) - qy) / p["tau_q"]
                    dqz_dt = (-(gz + p["tau_T"] * dgrad_z) - qz) / p["tau_q"]
                elif model == "Thermomass":
                    # strict PPT thermomass (3D component form); keep existing output paths unchanged.
                    # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): not used here.
                    eps = float(p["tm_epsilon"])
                    G_tm = float(p["tau_tm"])
                    tiny = 1.0e-12
                    Theta = 1.0 + eps * T
                    Theta_s = np.maximum(Theta, tiny)
                    ux = qx / Theta_s
                    uy = qy / Theta_s
                    uz = qz / Theta_s
                    div_theta = div_q_3d(qx, qy, qz, dx, dy, dz, qL_face, qR)
                    Theta_t = -eps * div_theta
                    Theta_x = eps * gx
                    Theta_y = eps * gy
                    Theta_z = eps * gz
                    qx_ux = grad_upwind_x(qx, ux, dx)
                    qy_uy = grad_upwind_y(qy, uy, dy)
                    qz_uz = grad_upwind_z(qz, uz, dz)
                    rqx = ux * Theta_t - gx / G_tm - eps * ux * (qx_ux - ux * Theta_x)
                    rqy = uy * Theta_t - gy / G_tm - eps * uy * (qy_uy - uy * Theta_y)
                    rqz = uz * Theta_t - gz / G_tm - eps * uz * (qz_uz - uz * Theta_z)
                    qx_s = (qx + dt_sub * rqx) / (1.0 + dt_sub / G_tm)
                    qy_s = (qy + dt_sub * rqy) / (1.0 + dt_sub / G_tm)
                    qz_s = (qz + dt_sub * rqz) / (1.0 + dt_sub / G_tm)
                    dqx_dt = (qx_s - qx) / dt_sub
                    dqy_dt = (qy_s - qy) / dt_sub
                    dqz_dt = (qz_s - qz) / dt_sub
                else:
                    raise ValueError(model)

                qx_new = qx + dt_sub * dqx_dt
                qy_new = qy + dt_sub * dqy_dt
                qz_new = qz + dt_sub * dqz_dt
                qx_new[0, :, :] = qL_face
                qx_new[-1, :, :] = qR
                qy_new[:, 0, :] = 0.0
                qy_new[:, -1, :] = 0.0
                qz_new[:, :, 0] = 0.0
                qz_new[:, :, -1] = 0.0
                divv = div_q_3d(qx_new, qy_new, qz_new, dx, dy, dz, qL_face, qR)
                T_new = T - dt_sub * divv

            T = T_new
            qx, qy, qz = qx_new, qy_new, qz_new
            gx_prev = gx.copy()
            gy_prev = gy.copy()
            gz_prev = gz.copy()

            if (
                (not np.all(np.isfinite(T)))
                or (not np.all(np.isfinite(qx)))
                or (not np.all(np.isfinite(qy)))
                or (not np.all(np.isfinite(qz)))
            ):
                raise FloatingPointError(f"{model} 3D became non-finite; try smaller dt or coarser grid.")

        if n % p["save_every"] == 0:
            T_hist.append(T.copy())
            qx_hist.append(qx.copy())
            qy_hist.append(qy.copy())
            qz_hist.append(qz.copy())
            t_hist.append(float(tn))

    return Result3D(
        model=model,
        x=x,
        y=y,
        z=z,
        t=np.array(t_hist),
        T=np.stack(T_hist, axis=0),
        qx=np.stack(qx_hist, axis=0),
        qy=np.stack(qy_hist, axis=0),
        qz=np.stack(qz_hist, axis=0),
        meta=p,
    )


def nearest_time_index(t_arr: np.ndarray, t_star: float) -> int:
    return int(np.argmin(np.abs(t_arr - t_star)))


if __name__ == "__main__":
    from pathlib import Path

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; demo saves npz only.")
        res = simulate_model_3d("Fourier", "single_pulse", nx=25, ny=25, nz=25)
        print("Fourier 3D done.", res.T.shape)
        raise SystemExit(0)

    from boundary_flux_layers_2d import DEFAULT_FLUX_LAYER_AMPS, DEFAULT_FLUX_Y_EDGES

    out = Path("results") / "case0_3d_fd_demo"
    out.mkdir(parents=True, exist_ok=True)
    res = simulate_model_3d(
        "Fourier",
        "single_pulse",
        nx=33,
        ny=33,
        nz=33,
        flux_y_edges=DEFAULT_FLUX_Y_EDGES,
        flux_layer_amps=DEFAULT_FLUX_LAYER_AMPS,
    )
    np.savez_compressed(
        out / "fd_3d_fourier_demo.npz",
        x=res.x,
        y=res.y,
        z=res.z,
        t=res.t,
        T=res.T,
    )
    print(f"saved {out / 'fd_3d_fourier_demo.npz'}  T shape={res.T.shape}")

    kz = res.z.size // 2
    X, Y = np.meshgrid(res.x, res.y, indexing="ij")
    for ts in (0.04, 0.10):
        it = min(nearest_time_index(res.t, ts), res.T.shape[0] - 1)
        sl = res.T[it, :, :, kz]
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        pcm = ax.pcolormesh(X.T, Y.T, sl.T, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$y^*$")
        ax.set_title(rf"FD 3D Fourier  $\theta^*$ at $z^*={res.z[kz]:.3f}$, $t^*={res.t[it]:.3f}$")
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"fd3d_slice_z_{res.z[kz]:.2f}_t_{res.t[it]:.3f}.png", dpi=200)
        plt.close(fig)
