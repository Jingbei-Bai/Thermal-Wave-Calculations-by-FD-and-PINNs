from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from thermal_1d_case_library_tm_ppt import (
    MODELS,
    Result,
    ensure_dir,
    laplace_1d,
    grad_center,
    grad_upwind,
    dqdx_from_flux,
    gaussian_pulse,
    default_material_params,
    nondim_groups,
    style_axes,
)


@dataclass
class SweepConfig:
    amplitudes: tuple[float, ...] = (0.5, 10.0, 100.0)
    labels: tuple[str, ...] = ("low", "medium", "high")
    center: float = 0.055
    width: float = 0.012
    nx: int = 401
    t_end: float = 0.28
    dt: float = 2.0e-5
    save_every: int = 60
    T0: float = 0.0
    snap_times: tuple[float, ...] = (0.07, 0.10, 0.14, 0.20)
    probes: tuple[float, ...] = (0.10, 0.25, 0.45)
    tm_eta: float = 2.0e-3
    tm_theta_floor: float = 5.0e-2
    tm_u_clip: float = 3.0
    material: dict | None = None


def nearest_idx(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def recommend_substeps(model: str, dt: float, dx: float, p: dict) -> int:
    if model == "Thermomass":
        c_wave = math.sqrt(1.0 / max(p["tau_tm"], 1.0e-30))
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


def build_params(amplitude: float, cfg: SweepConfig) -> dict:
    p = {
        "nx": cfg.nx,
        "t_end": cfg.t_end,
        "dt": cfg.dt,
        "save_every": cfg.save_every,
        "T0": cfg.T0,
        "material": default_material_params() if cfg.material is None else dict(cfg.material),
    }
    p.update(nondim_groups(p["material"]))
    p["title"] = f"Single heat pulse amplitude sweep | A={amplitude:.2f}"
    p["snap_times"] = list(cfg.snap_times)
    p["probes"] = list(cfg.probes)
    p["amplitude"] = float(amplitude)
    p["pulse_center"] = float(cfg.center)
    p["pulse_width"] = float(cfg.width)
    p["tm_eta"] = float(cfg.tm_eta)
    p["tm_theta_floor"] = float(cfg.tm_theta_floor)
    p["tm_u_clip"] = float(cfg.tm_u_clip)
    return p


def simulate_model_amplitude(model: str, amplitude: float, cfg: SweepConfig) -> Result:
    p = build_params(amplitude, cfg)
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
            qL = gaussian_pulse(t_now, amplitude, cfg.center, cfg.width)
            qR = 0.0
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
                    # strict PPT thermomass; keep existing output paths unchanged.
                    # deprecated old practical tm terms (Gamma_T, Pi_q, eta_tm): sweep overrides ignored in closure.
                    eps = float(p["tm_epsilon"])
                    G_tm = float(p["tau_tm"])
                    tiny = 1.0e-12
                    Theta = 1.0 + eps * T
                    Theta_s = np.maximum(Theta, tiny)
                    u = q / Theta_s
                    dqdx_inst = dqdx_from_flux(q, qL, qR, dx)
                    if nx > 2:
                        dqdx_inst[0] = dqdx_inst[1]
                        dqdx_inst[-1] = dqdx_inst[-2]
                    Theta_t = -eps * dqdx_inst
                    Theta_x = eps * gradT
                    qx_up = grad_upwind(q, u, dx)
                    rhs_tm = u * Theta_t - gradT / G_tm - eps * u * (qx_up - u * Theta_x)
                    q_star = (q + dt_sub * rhs_tm) / (1.0 + dt_sub / G_tm)
                    dqdt = (q_star - q) / dt_sub
                else:
                    raise ValueError(model)

                q_new = q + dt_sub * dqdt
                q_new[0] = qL
                q_new[-1] = qR
                dqdx = dqdx_from_flux(q_new, qL, qR, dx)
                T_new = T - dt_sub * dqdx

            T = T_new
            q = q_new
            grad_prev = gradT.copy()

            if (not np.all(np.isfinite(T))) or (not np.all(np.isfinite(q))):
                raise FloatingPointError(f"{model} became non-finite for amplitude {amplitude}.")

        if n % p["save_every"] == 0:
            T_hist.append(T.copy())
            q_hist.append(q.copy())
            t_hist.append(tn)
            ql_hist.append(gaussian_pulse(tn, amplitude, cfg.center, cfg.width))
            qr_hist.append(0.0)

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


def peak_position_history(res: Result) -> np.ndarray:
    return res.x[np.argmax(res.T, axis=1)]


def peak_value_history(res: Result) -> np.ndarray:
    return np.max(res.T, axis=1)


def probe_history(res: Result, x_probe: float) -> np.ndarray:
    return res.T[:, nearest_idx(res.x, x_probe)]


def save_summary(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outpath: Path) -> None:
    rows = []
    for level, model_map in results.items():
        for model, res in model_map.items():
            xpk = peak_position_history(res)
            Tpk = peak_value_history(res)
            rows.append({
                "level": level,
                "amplitude": labels_to_amp[level],
                "model": model,
                "peak_theta_max": float(np.max(Tpk)),
                "peak_theta_final": float(Tpk[-1]),
                "peak_x_at_t0.10": float(xpk[nearest_idx(res.t, 0.10)]),
                "peak_x_at_t0.14": float(xpk[nearest_idx(res.t, 0.14)]),
                "peak_x_at_t0.20": float(xpk[nearest_idx(res.t, 0.20)]),
                "nsub": int(res.meta["nsub"]),
            })
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_same_time_all_models(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outdir: Path, snap_times: tuple[float, ...]) -> None:
    for level, amp in labels_to_amp.items():
        for ts in snap_times:
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            for model in MODELS:
                res = results[level][model]
                it = nearest_idx(res.t, ts)
                ax.plot(res.x[1:], res.T[it, 1:], linewidth=2.0, label=model)
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$\theta$")
            ax.set_title(fr"{level.capitalize()} amplitude ($A={amp:.2f}$) | all models | $t^*={ts:.3f}$")
            style_axes(ax)
            ax.legend(frameon=False, ncol=3)
            fig.tight_layout()
            fig.savefig(outdir / f"{level}_all_models_t_{ts:.3f}.png", dpi=220)
            plt.close(fig)


def plot_same_model_across_amplitudes(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outdir: Path, snap_times: tuple[float, ...]) -> None:
    for model in MODELS:
        for ts in snap_times:
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            for level, amp in labels_to_amp.items():
                res = results[level][model]
                it = nearest_idx(res.t, ts)
                ax.plot(res.x[1:], res.T[it, 1:], linewidth=2.0, label=fr"{level} ($A={amp:.2f}$)")
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$\theta$")
            ax.set_title(fr"{model} | same frequency, different amplitudes | $t^*={ts:.3f}$")
            style_axes(ax)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(outdir / f"{model.lower()}_amplitude_compare_t_{ts:.3f}.png", dpi=220)
            plt.close(fig)


def plot_peak_position_by_amplitude(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outdir: Path) -> None:
    for level, amp in labels_to_amp.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for model in MODELS:
            res = results[level][model]
            ax.plot(res.t, peak_position_history(res), linewidth=2.0, label=model)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$x^*_{\mathrm{peak}}$")
        ax.set_title(fr"Peak position history | all models | {level} amplitude ($A={amp:.2f}$)")
        style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"{level}_peak_position_all_models.png", dpi=220)
        plt.close(fig)


def plot_peak_value_by_amplitude(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outdir: Path) -> None:
    for level, amp in labels_to_amp.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for model in MODELS:
            res = results[level][model]
            ax.plot(res.t, peak_value_history(res), linewidth=2.0, label=model)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\theta_{\max}(t^*)$")
        ax.set_title(fr"Peak amplitude history | all models | {level} amplitude ($A={amp:.2f}$)")
        style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / f"{level}_peak_value_all_models.png", dpi=220)
        plt.close(fig)


def plot_probe_histories(results: dict[str, dict[str, Result]], labels_to_amp: dict[str, float], outdir: Path, probes: tuple[float, ...]) -> None:
    for level, amp in labels_to_amp.items():
        for xp in probes:
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            for model in MODELS:
                res = results[level][model]
                ax.plot(res.t, probe_history(res, xp), linewidth=2.0, label=model)
            ax.set_xlabel(r"$t^*$")
            ax.set_ylabel(fr"$\theta(x^*={xp:.2f},t^*)$")
            ax.set_title(fr"Probe history at $x^*={xp:.2f}$ | all models | {level} amplitude ($A={amp:.2f}$)")
            style_axes(ax)
            ax.legend(frameon=False, ncol=3)
            fig.tight_layout()
            fig.savefig(outdir / f"{level}_probe_x_{xp:.2f}_all_models.png", dpi=220)
            plt.close(fig)


def run_amplitude_sweep_all_models(outdir: str | Path = "results/case_amp_sweep_all_models_tm_ppt") -> dict[str, dict[str, Result]]:
    cfg = SweepConfig()
    outdir = ensure_dir(outdir)
    labels_to_amp = {cfg.labels[i]: cfg.amplitudes[i] for i in range(len(cfg.amplitudes))}

    results: dict[str, dict[str, Result]] = {}
    for level, amp in labels_to_amp.items():
        results[level] = {model: simulate_model_amplitude(model, amp, cfg) for model in MODELS}

    save_summary(results, labels_to_amp, outdir / "amplitude_sweep_all_models_summary.csv")
    plot_same_time_all_models(results, labels_to_amp, outdir, cfg.snap_times)
    plot_same_model_across_amplitudes(results, labels_to_amp, outdir, cfg.snap_times)
    plot_peak_position_by_amplitude(results, labels_to_amp, outdir)
    plot_peak_value_by_amplitude(results, labels_to_amp, outdir)
    plot_probe_histories(results, labels_to_amp, outdir, cfg.probes)
    return results


if __name__ == "__main__":
    run_amplitude_sweep_all_models()
    print("done amplitude sweep all models")
