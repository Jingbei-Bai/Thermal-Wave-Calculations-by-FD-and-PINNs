"""
Case 0 (single left-boundary heat pulse) — Cattaneo–Vernotte (CV) model,
physics-informed neural network (PINN) **without** any observation/data loss.

Shared implementation: `pinn_cv_tm_ppt_core.py`.

Governing equations (same nondimensional form as `thermal_1d_case_library_tm_ppt.py`):

    theta_t + q_x = 0
    tau_q_star * q_t + q + theta_x = 0

Domain: x* in [0, 1], t* in [0, t_end] with t_end = 0.28.

Example:
    python pinn_case0_cv_unsupervised.py              # default: high-quality (float64, 45k steps, L-BFGS, …)
    python pinn_case0_cv_unsupervised.py --fast       # lighter / faster
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from thermal_1d_case_library_tm_ppt import simulate_model

from pinn_cv_tm_ppt_core import (
    evaluate_on_grid,
    gaussian_torch,
    material_tau_q_star,
    train_pinn_cv,
)

T_END = 0.28
PULSE_CENTER = 0.055
PULSE_WIDTH = 0.012

TIME_WINDOWS_CASE0 = [
    (max(0.0, PULSE_CENTER - 4.0 * PULSE_WIDTH), min(T_END, PULSE_CENTER + 4.0 * PULSE_WIDTH))
]


def q_left_star(t: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return 1.0 * np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_WIDTH) ** 2)


def q_left_torch(t: torch.Tensor) -> torch.Tensor:
    return gaussian_torch(t, 1.0, PULSE_CENTER, PULSE_WIDTH)


def q_right_torch(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def train_pinn(device: str | torch.device = "cpu", **kwargs):
    """Delegates to `train_pinn_cv` with Case 0 boundary fluxes."""
    tau_q = material_tau_q_star()
    return train_pinn_cv(
        tau_q_star=tau_q,
        t_end=T_END,
        q_left_torch=q_left_torch,
        q_right_torch=q_right_torch,
        time_windows=TIME_WINDOWS_CASE0,
        device=device,
        **kwargs,
    )


def plot_comparison(
    outdir: Path,
    pinn_theta: np.ndarray,
    fd_theta: np.ndarray,
    x_pinn: np.ndarray,
    x_fd: np.ndarray,
    t_pinn: np.ndarray,
    t_fd: np.ndarray,
    t_snap: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    def nearest_t(arr: np.ndarray, ts: float) -> int:
        return int(np.argmin(np.abs(arr - ts)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ts in t_snap:
        it_p = min(nearest_t(t_pinn, ts), pinn_theta.shape[1] - 1)
        it_f = nearest_t(t_fd, ts)
        axes[0].plot(x_pinn, pinn_theta[:, it_p], label=fr"PINN $t^*={t_pinn[it_p]:.3f}$")
        axes[1].plot(x_fd[1:], fd_theta[it_f, 1:], label=fr"FD CV $t^*={t_fd[it_f]:.3f}$")
    axes[0].set_title("PINN (unsupervised)")
    axes[1].set_title("Finite difference CV (reference, not used in training)")
    for ax in axes:
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "case0_cv_pinn_vs_fd_profiles.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 0 CV PINN (physics-only, no data).")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster run: 20k steps, smaller net, float32, no L-BFGS (default is high-quality preset).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (experimental for PINNs; sets functorch donated_buffer=False).",
    )
    p.add_argument(
        "--lbfgs",
        action="store_true",
        help="After Adam, run L-BFGS fine-tuning (slow on CPU; disables compile).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tau_q = material_tau_q_star()
    print(f"Case 0 CV PINN (unsupervised).  tau_q* = {tau_q:.8f}  (from `nondim_groups`)")

    outdir = Path("results") / "case0_cv_pinn_unsupervised"
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_kw: dict = {}
    if args.fast:
        train_kw.update(
            n_steps=20_000,
            n_colloc=4_000,
            n_ic=500,
            n_bc=500,
            log_every=2_000,
            hidden=80,
            n_trunk_layers=4,
            use_float64=False,
            use_lbfgs=False,
        )
    else:
        train_kw.update(
            n_steps=45_000,
            n_colloc=22_000,
            n_ic=1_600,
            n_bc=1_600,
            use_float64=True,
            use_lbfgs=True,
            lbfgs_rounds=55,
            compile_model=False,
            hidden=128,
            n_trunk_layers=5,
            log_every=1_000,
        )
    if args.compile:
        train_kw["compile_model"] = True
    if args.lbfgs:
        train_kw["use_lbfgs"] = True
        train_kw["compile_model"] = False

    print(
        f"Training preset: fast={args.fast} (default=high-quality) "
        f"compile={args.compile} device={device} extra={train_kw}"
    )
    model, meta = train_pinn(device=device, **train_kw)

    nx, nt = 241, 180
    x, t_grid, theta_pinn, _q_pinn = evaluate_on_grid(model, nx, nt, T_END, device=device)
    np.savez_compressed(
        outdir / "pinn_case0_cv_field.npz",
        x=x,
        t=t_grid,
        theta=theta_pinn,
        tau_q_star=tau_q,
        meta=meta,
    )
    torch.save(model.state_dict(), outdir / "pinn_case0_cv_weights.pt")
    print(f"Saved PINN field to {outdir / 'pinn_case0_cv_field.npz'}")

    try:
        fd = simulate_model("CV", "single_pulse")
        t_snap = [0.04, 0.10, 0.20]
        plot_comparison(
            outdir,
            theta_pinn,
            fd.T,
            x,
            fd.x,
            t_grid,
            fd.t,
            t_snap,
        )
        print(f"Optional comparison figure: {outdir / 'case0_cv_pinn_vs_fd_profiles.png'}")
    except Exception as exc:
        print("Skipping FD comparison plot:", exc)


if __name__ == "__main__":
    main()
