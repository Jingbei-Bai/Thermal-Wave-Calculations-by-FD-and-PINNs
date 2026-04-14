from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["Fourier", "CV", "DPL", "GK", "Thermomass"]


@dataclass
class Result:
    model: str
    x: np.ndarray
    t: np.ndarray
    T: np.ndarray               # dimensionless excess temperature theta
    q: np.ndarray               # dimensionless heat flux
    q_left_hist: np.ndarray
    q_right_hist: np.ndarray
    meta: dict


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def laplace_1d(u: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def grad_center(u: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    out[0] = (u[1] - u[0]) / dx
    out[-1] = (u[-1] - u[-2]) / dx
    return out


def grad_upwind(phi: np.ndarray, vel: np.ndarray, dx: float) -> np.ndarray:
    """First-order upwind derivative for advective terms d(phi)/dx.
    Positive vel uses backward diff, negative vel uses forward diff.
    """
    out = np.zeros_like(phi)
    # interior
    pos = vel[1:-1] >= 0.0
    neg = ~pos
    out_mid = out[1:-1]
    out_mid[pos] = (phi[1:-1][pos] - phi[:-2][pos]) / dx
    out_mid[neg] = (phi[2:][neg] - phi[1:-1][neg]) / dx
    out[1:-1] = out_mid
    # boundaries: one-sided consistent with flow direction; these points are overwritten by BC after update
    out[0] = (phi[1] - phi[0]) / dx
    out[-1] = (phi[-1] - phi[-2]) / dx
    return out


def dqdx_from_flux(q: np.ndarray, q_left: float, q_right: float, dx: float) -> np.ndarray:
    qext = q.copy()
    qext[0] = q_left
    qext[-1] = q_right
    out = np.zeros_like(q)
    out[1:-1] = (qext[2:] - qext[:-2]) / (2.0 * dx)
    out[0] = (qext[1] - q_left) / dx
    out[-1] = (q_right - qext[-2]) / dx
    return out


def dqdx_from_flux_rusanov(
    T: np.ndarray, q: np.ndarray, q_left: float, q_right: float, dx: float, wave_speed: float
) -> np.ndarray:
    """Rusanov flux for θ_t + q_x = 0 on Thermomass (reduces spurious Gibbs at steep q-fronts)."""
    n = len(T)
    out = np.zeros_like(T)
    a = float(max(wave_speed, 1.0e-12))
    F = np.zeros(n + 1, dtype=float)
    F[0] = q_left
    for i in range(n - 1):
        F[i + 1] = 0.5 * (q[i] + q[i + 1]) - 0.5 * a * (T[i + 1] - T[i])
    F[n] = q_right
    out[:] = (F[1:] - F[:-1]) / dx
    return out


def thermomass_rusanov_wave_speed(q: np.ndarray, theta: np.ndarray, eps: float, tau_tm: float) -> float:
    """Linearized Cattaneo-scale speed √(1/τ_tm) plus |u|; avoids under-dissipation (spurious peaks)."""
    tiny = 1.0e-12
    Theta_s = np.maximum(1.0 + eps * theta, tiny)
    c_tm = math.sqrt(1.0 / max(float(tau_tm), 1.0e-30))
    u_mag = float(np.max(np.abs(q / Theta_s)))
    return float(max(c_tm, u_mag, 1.0))


def gaussian_pulse(t: float, amp: float, center: float, width: float) -> float:
    return float(amp * np.exp(-0.5 * ((t - center) / width) ** 2))


def double_gaussian_pulse(t: float, amp1: float, c1: float, w1: float,
                          amp2: float, c2: float, w2: float) -> float:
    return gaussian_pulse(t, amp1, c1, w1) + gaussian_pulse(t, amp2, c2, w2)


def default_material_params() -> dict:
    return {
        "rho": 2330.0,
        "k": 163.0,
        "Cv": 657.0,
        "gamma": 1.96,
        "vs": 8433.0,
        "tau_tm": 1.38e-10,
        "tau_q": 5.0e-11,
        "tau_T": 8.0e-12,
        "ell_gk": 1.76e-7,
        # Θ = 1 + ε θ, ε = ΔT_ref / T_ref_abs; T = T_ref_abs + ΔT_ref * θ. When θ* is O(1) in cases, take ΔT_ref as the
        # physical ΔT matching θ*=1 (default 1 K → weak nonlinearity; override in material if your scaling differs).
        "T_ref_abs": 300.0,
        "Delta_T_ref": 1.0,
    }


def nondim_groups(material: dict | None = None, L_ref: float | None = None) -> dict:
    m = default_material_params()
    if material:
        m.update(material)

    alpha = m["k"] / (m["rho"] * m["Cv"])
    if L_ref is None:
        L_ref = m["ell_gk"]
    t_ref = L_ref * L_ref / alpha
    T_ref_abs = float(m.get("T_ref_abs", 300.0))
    Delta_T_ref = float(m.get("Delta_T_ref", 1.0))
    tm_epsilon = Delta_T_ref / max(T_ref_abs, 1.0e-30)

    return {
        "alpha": alpha,
        "L_ref": L_ref,
        "t_ref": t_ref,
        "tau_q": m["tau_q"] / t_ref,
        "tau_T": m["tau_T"] / t_ref,
        "tau_tm": m["tau_tm"] / t_ref,
        "l2_gk": (m["ell_gk"] / L_ref) ** 2,
        # strict PPT thermomass: Θ = 1 + ε θ, ε = ΔT_ref / T_ref_abs.
        "tm_epsilon": tm_epsilon,
        # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): kept in dict for meta/compat only.
        "tm_theta_base": 1.0,
        "tm_eta": 2.0e-3,
        "tm_theta_floor": 5.0e-2,
        "tm_u_clip": 3.0,
        "gamma": m["gamma"],
        "rho": m["rho"],
        "k": m["k"],
        "Cv": m["Cv"],
        "vs": m["vs"],
        "ell_gk": m["ell_gk"],
    }


def make_case(case: str):
    p = {
        "nx": 401,
        "t_end": 0.28,
        "dt": 2.0e-5,
        "save_every": 60,
        "T0": 0.0,
        "material": default_material_params(),
    }
    p.update(nondim_groups(p["material"]))

    if case == "single_pulse":
        p["title"] = "Case 0: Single heat pulse"
        def qL(t):
            return gaussian_pulse(t, 1.0, 0.055, 0.012)
        def qR(t):
            return 0.0
        p["snap_times"] = [0.04, 0.07, 0.10, 0.14, 0.20, 0.26]
        p["probes"] = [0.10, 0.25, 0.45]
    elif case == "double_pulse":
        p["title"] = "Case 1: Double-pulse interference"
        def qL(t):
            return double_gaussian_pulse(t, 1.0, 0.05, 0.012, 0.85, 0.11, 0.014)
        def qR(t):
            return 0.0
        p["snap_times"] = [0.05, 0.09, 0.13, 0.18, 0.24]
        p["probes"] = [0.10, 0.25, 0.45]
    elif case == "heat_cool":
        p["title"] = "Case 2: Heating then cooling"
        def qL(t):
            return gaussian_pulse(t, 1.0, 0.055, 0.012) - gaussian_pulse(t, 0.90, 0.13, 0.016)
        def qR(t):
            return 0.0
        p["snap_times"] = [0.05, 0.09, 0.13, 0.18, 0.24]
        p["probes"] = [0.10, 0.25, 0.45]
    elif case == "bilateral_heating":
        p["title"] = "Case 4: Bilateral inward heating"
        def qin(t):
            return gaussian_pulse(t, 0.8, 0.07, 0.02)
        def qL(t):
            return qin(t)
        def qR(t):
            return -qin(t)
        p["snap_times"] = [0.05, 0.09, 0.13, 0.18, 0.24]
        p["probes"] = [0.25, 0.50, 0.75]
    elif case == "tail_recovery":
        p["title"] = "Case 6: Post-pulse tail recovery"
        p["t_end"] = 0.72
        p["dt"] = 2.0e-6
        p["save_every"] = 400
        def qL(t):
            return gaussian_pulse(t, 1.0, 0.05, 0.011)
        def qR(t):
            return 0.0
        p["snap_times"] = [0.05, 0.09, 0.15, 0.24, 0.36, 0.48, 0.60, 0.72]
        p["probes"] = [0.10, 0.25, 0.45]
    else:
        raise ValueError(case)
    return p, qL, qR


def recommend_substeps(model: str, dt: float, dx: float, p: dict) -> int:
    if model == "Thermomass":
        # Hyperbolic-led substeps: Fourier-style 0.4*dx^2 caps dt_sub ~ O(dx^2) → many substeps/macro and
        # (with Rusanov) wipes finite-speed wave fronts. Use τ_tm + Cattaneo CFL + |u| cap only.
        c_wave = math.sqrt(1.0 / max(p["tau_tm"], 1.0e-30))
        limits = [
            0.32 * p["tau_tm"],
            0.30 * dx / max(c_wave, 1.0e-12),
            0.22 * dx / max(p.get("tm_u_clip", 3.0), 0.1),
        ]
        dt_lim = min(limits)
        return max(1, int(math.ceil(dt / dt_lim)))

    limits = [0.40 * dx * dx]
    if model in ("CV", "DPL"):
        limits.append(0.35 * p["tau_q"])
    if model == "GK":
        limits.append(0.20 * p["tau_q"] * dx * dx / max(p["l2_gk"], 1.0e-14))
    dt_lim = min(limits)
    return max(1, int(math.ceil(dt / dt_lim)))


def simulate_model(model: str, case: str) -> Result:
    p, qL_fn, qR_fn = make_case(case)
    nx = p["nx"]
    dt = p["dt"]
    t_end = p["t_end"]
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    nt = int(np.floor(t_end / dt)) + 1
    t_full = np.linspace(0.0, dt * (nt - 1), nt)

    T = np.full(nx, p["T0"], dtype=float)
    q = np.zeros(nx, dtype=float)
    grad_prev = np.zeros(nx, dtype=float)

    nsub = recommend_substeps(model, dt, dx, p)
    dt_sub = dt / nsub
    p["nsub"] = nsub
    p["dt_sub"] = dt_sub

    T_hist, q_hist, t_hist, ql_hist, qr_hist = [], [], [], [], []
    for n, tn in enumerate(t_full):
        for j in range(nsub):
            t_now = tn + j * dt_sub
            qL = float(qL_fn(t_now))
            qR = float(qR_fn(t_now))
            gradT = grad_center(T, dx)

            if model == "Fourier":
                q_new = -gradT
                q_new[0] = qL
                q_new[-1] = qR
                dqdx = dqdx_from_flux(q_new, qL, qR, dx)
                T_new = T - dt_sub * dqdx
            else:
                if model == "CV":
                    dqdt = (-gradT - q) / p["tau_q"]
                elif model == "DPL":
                    dgrad = (gradT - grad_prev) / dt_sub
                    dqdt = (-(gradT + p["tau_T"] * dgrad) - q) / p["tau_q"]
                elif model == "GK":
                    lap_q = laplace_1d(q, dx)
                    dqdt = (-gradT - q + p["l2_gk"] * lap_q) / p["tau_q"]
                elif model == "Thermomass":
                    # strict PPT thermomass equation (dimensionless): q + Γ_tm[q_t - u Θ_t]
                    #   + Γ_tm ε u [q_x - u Θ_x] = -θ_x,  Θ = 1 + ε θ,  q = Θ u.
                    # keep existing output paths unchanged (this branch only replaces the closure).
                    # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): tm_eta / tm_theta_* / clip not used here.
                    eps = float(p["tm_epsilon"])
                    G_tm = float(p["tau_tm"])
                    tiny = 1.0e-12
                    Theta = 1.0 + eps * T
                    Theta_s = np.maximum(Theta, tiny)
                    u = q / Theta_s
                    dqdx_inst = dqdx_from_flux(q, qL, qR, dx)
                    # damp BC spike in ∂q/∂x used only for Θ_t (not energy): reduces false u·Θ_t blow-up
                    if nx > 2:
                        dqdx_inst[0] = dqdx_inst[1]
                        dqdx_inst[-1] = dqdx_inst[-2]
                    Theta_t = -eps * dqdx_inst
                    Theta_x = eps * gradT
                    qx_up = grad_upwind(q, u, dx)
                    # semi-implicit on -q/Γ_tm: q_new = (q + dt*rhs) / (1 + dt/Γ_tm), rhs = u·Θ_t - θ_x/Γ - ε u(…)
                    rhs_tm = u * Theta_t - gradT / G_tm - eps * u * (qx_up - u * Theta_x)
                    q_star = (q + dt_sub * rhs_tm) / (1.0 + dt_sub / G_tm)
                    dqdt = (q_star - q) / dt_sub
                else:
                    raise ValueError(model)

                q_new = q + dt_sub * dqdt
                q_new[0] = qL
                q_new[-1] = qR
                # Centered θ_t+q_x; Thermomass uses hyperbolic substeps (no Rusanov stack-up) so q can lead θ (finite-speed front).
                dqdx = dqdx_from_flux(q_new, qL, qR, dx)
                T_new = T - dt_sub * dqdx

            T = T_new
            q = q_new
            grad_prev = gradT.copy()

            if (not np.all(np.isfinite(T))) or (not np.all(np.isfinite(q))):
                raise FloatingPointError(f"{model} became non-finite. Try a smaller dt.")

        if n % p["save_every"] == 0:
            T_hist.append(T.copy())
            q_hist.append(q.copy())
            t_hist.append(tn)
            ql_hist.append(float(qL_fn(tn)))
            qr_hist.append(float(qR_fn(tn)))

    return Result(
        model=model,
        x=x,
        t=np.array(t_hist),
        T=np.array(T_hist),
        q=np.array(q_hist),
        q_left_hist=np.array(ql_hist),
        q_right_hist=np.array(qr_hist),
        meta=p,
    )


def simulate_all(case: str):
    return {m: simulate_model(m, case) for m in MODELS}


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def probe_history(res: Result, x_probe: float):
    ix = nearest_idx(res.x, x_probe)
    return res.T[:, ix]


def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_summary(results: dict, outpath: Path):
    probes = results["Fourier"].meta["probes"]
    rows = []
    for model, res in results.items():
        row = {
            "model": model,
            "max_T": float(np.max(res.T)),
            "min_T": float(np.min(res.T)),
            "final_left_T": float(res.T[-1, 1]),
            "final_mid_T": float(res.T[-1, len(res.x) // 2]),
            "final_right_T": float(res.T[-1, -2]),
            "nsub": int(res.meta.get("nsub", 1)),
            "tau_q_star": float(res.meta["tau_q"]),
            "tau_T_star": float(res.meta["tau_T"]),
            "tau_tm_star": float(res.meta["tau_tm"]),
            "l2_gk": float(res.meta["l2_gk"]),
            "tm_eta": float(res.meta.get("tm_eta", 0.0)),
            "tm_u_clip": float(res.meta.get("tm_u_clip", 0.0)),
        }
        for xp in probes:
            row[f"probe_{xp:.2f}_peak"] = float(np.max(probe_history(res, xp)))
        rows.append(row)

    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_boundary_flux(results: dict, outdir: Path, case: str):
    base = next(iter(results.values()))
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    ax.plot(base.t, base.q_left_hist, linewidth=2.0, label="left boundary flux")
    if np.max(np.abs(base.q_right_hist)) > 1e-12:
        ax.plot(base.t, base.q_right_hist, linewidth=2.0, label="right boundary flux")
    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel(r"$q_b^*$")
    ax.set_title(base.meta["title"] + " | boundary flux")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"{case}_boundary_flux.png", dpi=220)
    plt.close(fig)


def plot_same_model_snapshots(results: dict, outdir: Path, case: str):
    base = next(iter(results.values()))
    times = base.meta["snap_times"]
    for model, res in results.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for ts in times:
            it = nearest_idx(res.t, ts)
            ax.plot(res.x[1:], res.T[it, 1:], linewidth=2.0, label=fr"$t^*={res.t[it]:.3f}$")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(f"{base.meta['title']} | {model} | same model, different times")
        style_axes(ax)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / f"{case}_same_model_{model.lower()}.png", dpi=220)
        plt.close(fig)


def plot_same_time_all_models(results: dict, outdir: Path, case: str):
    base = next(iter(results.values()))
    for ts in base.meta["snap_times"]:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for model, res in results.items():
            it = nearest_idx(res.t, ts)
            ax.plot(res.x[1:], res.T[it, 1:], linewidth=2.0, label=model)
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(base.meta["title"] + fr" | same time: $t^*={ts:.3f}$")
        style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"{case}_same_time_t_{ts:.3f}.png", dpi=220)
        plt.close(fig)


def plot_probe_histories(results: dict, outdir: Path, case: str):
    probes = next(iter(results.values())).meta["probes"]
    for xp in probes:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for model, res in results.items():
            ax.plot(res.t, probe_history(res, xp), linewidth=2.0, label=model)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(fr"$\theta(x^*={xp:.2f},t^*)$")
        ax.set_title(next(iter(results.values())).meta["title"] + fr" | probe at $x^*={xp:.2f}$")
        style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"{case}_probe_x_{xp:.2f}.png", dpi=220)
        plt.close(fig)


def plot_spacetime_cv(results: dict, outdir: Path, case: str):
    res = results["CV"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(
        res.T[:, 1:],
        origin="lower",
        aspect="auto",
        extent=[res.x[1], res.x[-1], res.t[0], res.t[-1]],
    )
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$t^*$")
    ax.set_title(res.meta["title"] + " | CV spacetime map")
    fig.colorbar(im, ax=ax, label=r"$\theta$")
    fig.tight_layout()
    fig.savefig(outdir / f"{case}_cv_spacetime.png", dpi=220)
    plt.close(fig)


def run_case(case: str, outdir: str | Path):
    outdir = ensure_dir(outdir)
    results = simulate_all(case)
    save_summary(results, outdir / f"{case}_summary.csv")
    plot_boundary_flux(results, outdir, case)
    plot_same_model_snapshots(results, outdir, case)
    plot_same_time_all_models(results, outdir, case)
    plot_probe_histories(results, outdir, case)
    plot_spacetime_cv(results, outdir, case)
    return results
