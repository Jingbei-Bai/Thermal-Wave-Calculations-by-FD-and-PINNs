from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray
MODELS = ["Fourier", "CV", "DPL", "GK", "Thermomass"]


@dataclass
class Case8Result:
    model: str
    heating_strength: float
    x: Array
    t: Array
    theta: Array
    q: Array
    meta: Dict


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def laplace_1d(u: Array, dx: float) -> Array:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def grad_center(u: Array, dx: float) -> Array:
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    out[0] = (u[1] - u[0]) / dx
    out[-1] = (u[-1] - u[-2]) / dx
    return out


def grad_upwind(phi: Array, vel: Array, dx: float) -> Array:
    out = np.zeros_like(phi)
    pos = vel[1:-1] >= 0.0
    neg = ~pos
    out_mid = out[1:-1]
    out_mid[pos] = (phi[1:-1][pos] - phi[:-2][pos]) / dx
    out_mid[neg] = (phi[2:][neg] - phi[1:-1][neg]) / dx
    out[1:-1] = out_mid
    out[0] = (phi[1] - phi[0]) / dx
    out[-1] = (phi[-1] - phi[-2]) / dx
    return out


def dqdx_from_flux(q: Array, q_left: float, q_right: float, dx: float) -> Array:
    """∂q/∂x with prescribed face fluxes (cf. ``thermal_1d_case_library_tm_ppt``)."""
    qext = q.copy()
    qext[0] = q_left
    qext[-1] = q_right
    out = np.zeros_like(q)
    out[1:-1] = (qext[2:] - qext[:-2]) / (2.0 * dx)
    out[0] = (qext[1] - q_left) / dx
    out[-1] = (q_right - qext[-2]) / dx
    return out


def thermomass_rusanov_wave_speed(q: Array, theta: Array, eps: float, tau_tm: float) -> float:
    tiny = 1.0e-12
    Theta_s = np.maximum(1.0 + eps * theta, tiny)
    c_tm = math.sqrt(1.0 / max(float(tau_tm), 1.0e-30))
    u_mag = float(np.max(np.abs(q / Theta_s)))
    return float(max(c_tm, u_mag, 1.0e-12))


def dqdx_from_flux_rusanov(theta: Array, q: Array, q_left: float, q_right: float, dx: float, wave_speed: float) -> Array:
    """Finite-volume divergence using a Rusanov/Lax-Friedrichs numerical flux.

    This mirrors the protection used in the earlier main script for the temperature
    equation theta_t + q_x = 0, reducing centered-grid oscillations fed back from q into theta.

    Toggle: ``cv_dpl_rusanov_div_q`` (default True).
    """
    n = len(theta)
    out = np.zeros_like(theta)
    a = float(max(wave_speed, 1.0e-12))
    F = np.zeros(n + 1, dtype=float)
    F[0] = q_left
    for i in range(n - 1):
        qL_i = q[i]
        qR_i = q[i + 1]
        thetaL_i = theta[i]
        thetaR_i = theta[i + 1]
        F[i + 1] = 0.5 * (qL_i + qR_i) - 0.5 * a * (thetaR_i - thetaL_i)
    F[n] = q_right
    out[:] = (F[1:] - F[:-1]) / dx
    return out


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
        "l2_gk": (m["ell_gk"] / L_ref) ** 2,
        "tau_tm": m["tau_tm"] / t_ref,
        "tm_epsilon": tm_epsilon,
        # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): meta/compat only.
        "tm_theta_base": 1.0,
        "tm_theta_floor": 5.0e-2,
        "tm_u_clip": 3.0,
        "tm_eta": 2.0e-3,
        "gamma": m["gamma"],
        "rho": m["rho"],
        "k": m["k"],
        "Cv": m["Cv"],
        "vs": m["vs"],
        "ell_gk": m["ell_gk"],
        "material": m,
    }


def default_case8_params() -> Dict[str, float]:
    p = {
        "nx": 241,
        "t_end": 0.10,
        # Explicit stability ~0.4*dx^2 ~ 7e-6; dt=2e-7 gave nt≈5e5 per run → hours for 5×4 cases.
        "dt": 1.0e-6,
        "save_every": 400,
        "snapshot_times": [0.008, 0.016, 0.032, 0.048, 0.072],
        "heating_strengths": [0.6, 0.9, 1.2, 1.5],
        "cv_dpl_rusanov_div_q": True,
        "thermomass_rusanov_theta": True,
        "material": default_material_params(),
    }
    p.update(nondim_groups(p["material"]))
    return p


def recommend_substeps(model: str, dt: float, dx: float, p: dict) -> int:
    if model == "Thermomass":
        c_wave = math.sqrt(max(1.0 / max(p["tau_tm"], 1.0e-30), 0.0))
        limits = [
            0.32 * p["tau_tm"],
            0.30 * dx / max(c_wave, 1.0e-12),
            0.22 * dx / max(p.get("tm_u_clip", 3.0), 0.1),
        ]
        return max(1, int(math.ceil(dt / min(limits))))

    limits = [0.40 * dx * dx]
    if model in ("CV", "DPL"):
        limits.append(0.35 * p["tau_q"])
    if model == "GK":
        limits.append(0.20 * p["tau_q"] * dx * dx / max(p["l2_gk"], 1.0e-14))
    dt_lim = min(limits)
    return max(1, int(math.ceil(dt / dt_lim)))


def simulate_case8(model: str, heating_strength: float = 1.0, params: Optional[Dict] = None) -> Case8Result:
    p = default_case8_params()
    if params:
        p.update(params)

    nx = int(p["nx"])
    t_end = float(p["t_end"])
    dt = float(p["dt"])
    tau_q = float(p["tau_q"])
    tau_T = float(p["tau_T"])
    l2_gk = float(p["l2_gk"])
    tau_tm = float(p["tau_tm"])
    save_every = int(p["save_every"])
    cv_dpl_rusanov = bool(p.get("cv_dpl_rusanov_div_q", True))

    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    nt = int(np.floor(t_end / dt)) + 1
    t_full = np.linspace(0.0, dt * (nt - 1), nt)

    theta = np.zeros(nx, dtype=float)
    theta[0] = heating_strength
    theta[-1] = heating_strength
    q = np.zeros(nx, dtype=float)
    grad_prev = np.zeros(nx, dtype=float)

    nsub = recommend_substeps(model, dt, dx, p)
    dt_sub = dt / nsub
    p["nsub"] = nsub
    p["dt_sub"] = dt_sub

    theta_hist = []
    q_hist = []
    t_hist = []

    for n, tn in enumerate(t_full):
        if n % save_every == 0:
            theta_hist.append(theta.copy())
            q_hist.append(q.copy())
            t_hist.append(float(tn))

        for _ in range(nsub):
            grad_theta = grad_center(theta, dx)
            tm_bnd_flux: tuple[float, float] | None = None

            if model == "Fourier":
                q_new = -grad_theta
                div_q = grad_center(q_new, dx)
            elif model == "CV":
                dqdt = (-grad_theta - q) / tau_q
                q_new = q + dt_sub * dqdt
                q_new[0] = q_new[1]
                q_new[-1] = q_new[-2]
                c_wave = math.sqrt(max(1.0 / max(tau_q, 1.0e-14), 1.0e-14))
                if cv_dpl_rusanov:
                    div_q = dqdx_from_flux_rusanov(theta, q_new, q_new[0], q_new[-1], dx, c_wave)
                else:
                    div_q = grad_center(q_new, dx)
            elif model == "DPL":
                dgrad = (grad_theta - grad_prev) / dt_sub
                dqdt = (-(grad_theta + tau_T * dgrad) - q) / tau_q
                q_new = q + dt_sub * dqdt
                c_wave = math.sqrt(max(1.0 / max(tau_q, 1.0e-14), 1.0e-14))
                q_new[0] = q_new[1]
                q_new[-1] = q_new[-2]
                if cv_dpl_rusanov:
                    div_q = dqdx_from_flux_rusanov(theta, q_new, q_new[0], q_new[-1], dx, c_wave)
                else:
                    div_q = grad_center(q_new, dx)
            elif model == "GK":
                lap_q = laplace_1d(q, dx)
                dqdt = (-grad_theta - q + l2_gk * lap_q) / tau_q
                q_new = q + dt_sub * dqdt
                div_q = grad_center(q_new, dx)
            elif model == "Thermomass":
                # Same θ*,x*,t* as thermal_1d_case_library_tm_ppt (closure uses θ_x*, not alpha*θ_x*).
                eps = float(p["tm_epsilon"])
                tiny = 1.0e-12
                Theta = 1.0 + eps * theta
                Theta_s = np.maximum(Theta, tiny)
                u = q / Theta_s
                dqdx_inst = grad_center(q, dx)
                if nx > 2:
                    dqdx_inst[0] = dqdx_inst[1]
                    dqdx_inst[-1] = dqdx_inst[-2]
                Theta_t = -eps * dqdx_inst
                Theta_x = eps * grad_theta
                qx_up = grad_upwind(q, u, dx)
                rhs_tm = u * Theta_t - grad_theta / tau_tm - eps * u * (qx_up - u * Theta_x)
                q_star = (q + dt_sub * rhs_tm) / (1.0 + dt_sub / tau_tm)
                q_new = q_star
                q_face_L = -float(grad_theta[0])
                q_face_R = -float(grad_theta[-1])
                tm_bnd_flux = (q_face_L, q_face_R)
                q_new[0] = q_face_L
                q_new[-1] = q_face_R
                if p.get("thermomass_rusanov_theta", True):
                    c_tm = thermomass_rusanov_wave_speed(q_new, theta, eps, tau_tm)
                    div_q = dqdx_from_flux_rusanov(theta, q_new, q_face_L, q_face_R, dx, c_tm)
                else:
                    div_q = dqdx_from_flux(q_new, q_face_L, q_face_R, dx)
            else:
                raise ValueError(f"Unsupported model: {model}")
            theta_new = theta - dt_sub * div_q
            theta_new[0] = heating_strength
            theta_new[-1] = heating_strength
            if tm_bnd_flux is not None:
                q_new[0], q_new[-1] = tm_bnd_flux
            else:
                q_new[0] = q_new[1]
                q_new[-1] = q_new[-2]

            theta = theta_new
            q = q_new
            grad_prev = grad_theta.copy()

            if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(q)):
                raise FloatingPointError(f"{model} became non-finite for S={heating_strength}.")

    return Case8Result(
        model=model,
        heating_strength=heating_strength,
        x=x,
        t=np.asarray(t_hist),
        theta=np.asarray(theta_hist),
        q=np.asarray(q_hist),
        meta=dict(p),
    )


def summarize_result(res: Case8Result) -> dict:
    idx = int(np.argmax(res.theta))
    it, ix = np.unravel_index(idx, res.theta.shape)
    center_idx = len(res.x) // 2
    return {
        "model": res.model,
        "heating_strength": res.heating_strength,
        "global_max_theta": float(np.max(res.theta)),
        "time_of_max": float(res.t[it]),
        "x_of_max": float(res.x[ix]),
        "final_center_theta": float(res.theta[-1, center_idx]),
        "nsub": int(res.meta.get("nsub", 1)),
        "tau_q_star": float(res.meta["tau_q"]),
        "tau_T_star": float(res.meta["tau_T"]),
        "tau_tm_star": float(res.meta["tau_tm"]),
    }


def save_summary(rows: list[dict], path: str | Path) -> None:
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_max_history_cross_group(results: dict[float, dict[str, Case8Result]], outdir: Path) -> None:
    strengths = sorted(results.keys())
    ncols = 2
    nrows = int(np.ceil(len(strengths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.2 * nrows), squeeze=False)
    for ax, qs in zip(axes.ravel(), strengths):
        for model in MODELS:
            res = results[qs][model]
            ax.plot(res.t, np.max(res.theta, axis=1), linewidth=2.0, label=model)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\max_x\theta(x,t)$")
        ax.set_title(fr"Maximum-temperature history, $S={qs:.1f}$")
        _style_axes(ax)
    for ax in axes.ravel()[len(strengths):]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "case8_max_history_cross_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_center_history_cross_group(results: dict[float, dict[str, Case8Result]], outdir: Path) -> None:
    strengths = sorted(results.keys())
    ncols = 2
    nrows = int(np.ceil(len(strengths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.2 * nrows), squeeze=False)
    for ax, qs in zip(axes.ravel(), strengths):
        for model in MODELS:
            res = results[qs][model]
            center_idx = len(res.x) // 2
            ax.plot(res.t, res.theta[:, center_idx], linewidth=2.0, label=model)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\theta(0.5,t)$")
        ax.set_title(fr"Center-temperature history, $S={qs:.1f}$")
        _style_axes(ax)
    for ax in axes.ravel()[len(strengths):]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "case8_center_history_cross_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_profiles_at_cv_max_time(results: dict[float, dict[str, Case8Result]], outdir: Path) -> None:
    strengths = sorted(results.keys())
    ncols = 2
    nrows = int(np.ceil(len(strengths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.2 * nrows), squeeze=False)
    for ax, qs in zip(axes.ravel(), strengths):
        cv_res = results[qs]["CV"]
        idx = int(np.argmax(cv_res.theta))
        it, _ = np.unravel_index(idx, cv_res.theta.shape)
        t_star = cv_res.t[it]
        for model in MODELS:
            res = results[qs][model]
            jt = int(np.argmin(np.abs(res.t - t_star)))
            ax.plot(res.x, res.theta[jt], linewidth=2.0, label=model)
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(fr"Profiles at CV maximum time, $S={qs:.1f}$, $t^*={t_star:.4f}$")
        _style_axes(ax)
    for ax in axes.ravel()[len(strengths):]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "case8_profiles_at_cv_max_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_same_model_snapshots(results: dict[float, dict[str, Case8Result]], outdir: Path, strongest: float | None = None, target_times: list[float] | None = None) -> None:
    if strongest is None:
        strongest = max(results.keys())
    if target_times is None:
        target_times = [0.008, 0.016, 0.032, 0.048, 0.072]
    for model in MODELS:
        res = results[strongest][model]
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for tt in target_times:
            jt = int(np.argmin(np.abs(res.t - tt)))
            ax.plot(res.x, res.theta[jt], linewidth=2.0, label=fr"$t^*={res.t[jt]:.3f}$")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(fr"{model} profile evolution under strongest bilateral heating ($S={strongest:.1f}$)")
        _style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"case8_same_model_{model.lower()}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_same_time_all_models(results: dict[float, dict[str, Case8Result]], outdir: Path, strongest: float | None = None, target_times: list[float] | None = None) -> None:
    if strongest is None:
        strongest = max(results.keys())
    if target_times is None:
        target_times = [0.016, 0.032, 0.048, 0.072]
    for tt in target_times:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for model in MODELS:
            res = results[strongest][model]
            jt = int(np.argmin(np.abs(res.t - tt)))
            ax.plot(res.x, res.theta[jt], linewidth=2.0, label=model)
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(fr"All models at same time under strongest bilateral heating ($S={strongest:.1f}$, $t^*\approx{tt:.2f}$)")
        _style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"case8_same_time_all_models_t_{tt:.3f}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_cv_spacetime(results: dict[float, dict[str, Case8Result]], outdir: Path, strongest: float | None = None) -> None:
    if strongest is None:
        strongest = max(results.keys())
    res = results[strongest]["CV"]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    im = ax.imshow(res.theta, origin='lower', aspect='auto', extent=[res.x[0], res.x[-1], res.t[0], res.t[-1]], vmin=float(np.min(res.theta)), vmax=float(np.max(res.theta)))
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$t^*$")
    ax.set_title(fr"CV spacetime map under strongest bilateral heating ($S={strongest:.1f}$)")
    fig.colorbar(im, ax=ax, label=r"$\theta$")
    fig.tight_layout()
    fig.savefig(outdir / "case8_cv_spacetime_strongest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_case8(outdir: str | Path = "results_case8_bilateral_heating", heating_strengths: tuple[float, ...] = (0.6, 0.9, 1.2, 1.5), params: Optional[Dict] = None) -> dict[float, dict[str, Case8Result]]:
    p = default_case8_params()
    if params:
        p.update(params)
    outdir = ensure_dir(outdir)
    all_results: dict[float, dict[str, Case8Result]] = {}
    rows: list[dict] = []
    total_runs = len(heating_strengths) * len(MODELS)
    run_i = 0
    for hs in heating_strengths:
        case_results: dict[str, Case8Result] = {}
        for model in MODELS:
            run_i += 1
            print(f"[case8] {run_i}/{total_runs}  S={hs}  model={model}  (dt={p['dt']}, nx={p['nx']})")
            res = simulate_case8(model=model, heating_strength=hs, params=p)
            case_results[model] = res
            rows.append(summarize_result(res))
        all_results[hs] = case_results

    save_summary(rows, outdir / "case8_bilateral_heating_summary.csv")
    plot_max_history_cross_group(all_results, outdir)
    plot_center_history_cross_group(all_results, outdir)
    plot_profiles_at_cv_max_time(all_results, outdir)
    plot_same_model_snapshots(all_results, outdir, target_times=p["snapshot_times"])
    plot_same_time_all_models(all_results, outdir, target_times=p["snapshot_times"][:-1])
    plot_cv_spacetime(all_results, outdir)
    return all_results


if __name__ == "__main__":
    run_case8()
