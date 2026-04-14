"""
Case 0 — Dual-phase-lag (DPL) flux law, physics-only PINN (no data).

Same boundary / IC as `thermal_1d_case_library_tm_ppt.py` single_pulse.
Shared core: `pinn_cv_tm_ppt_core.py`.

PDE (continuous form aligned with FD branch `model == "DPL"`):

    theta*_t + q*_x = 0
    tau_q* q*_t + q* + theta*_x + tau_T* d(theta*_x)/dt* = 0

Example:
    python pinn_case0_dpl_unsupervised.py
    python pinn_case0_dpl_unsupervised.py --fast
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
    material_dpl_params,
    train_pinn_dpl,
)

T_END = 0.28
PULSE_CENTER = 0.055
PULSE_WIDTH = 0.012

TIME_WINDOWS_CASE0 = [
    (max(0.0, PULSE_CENTER - 4.0 * PULSE_WIDTH), min(T_END, PULSE_CENTER + 4.0 * PULSE_WIDTH))
]


def q_left_torch(t: torch.Tensor) -> torch.Tensor:
    return gaussian_torch(t, 1.0, PULSE_CENTER, PULSE_WIDTH)


def q_right_torch(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def train_pinn(device: str | torch.device = "cpu", **kwargs):
    dpl = material_dpl_params()
    return train_pinn_dpl(
        tau_q_star=dpl["tau_q"],
        tau_T_star=dpl["tau_T"],
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
        axes[1].plot(x_fd[1:], fd_theta[it_f, 1:], label=fr"FD DPL $t^*={t_fd[it_f]:.3f}$")
    axes[0].set_title("PINN DPL (unsupervised)")
    axes[1].set_title("Finite difference DPL (reference)")
    for ax in axes:
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "case0_dpl_pinn_vs_fd_profiles.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Case 0 DPL PINN (physics-only).")
    p.add_argument("--fast", action="store_true", help="Faster run (default is high-quality).")
    p.add_argument("--compile", action="store_true", help="Optional torch.compile.")
    p.add_argument("--lbfgs", action="store_true", help="L-BFGS after Adam (disables compile).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dpl = material_dpl_params()
    print(
        f"Case 0 DPL PINN.  tau_q* = {dpl['tau_q']:.8f}  tau_T* = {dpl['tau_T']:.8f}  (from `nondim_groups`)"
    )

    outdir = Path("results") / "case0_dpl_pinn_unsupervised"
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

    print(f"device={device} train_kw keys={list(train_kw.keys())}")
    model, meta = train_pinn(device=device, **train_kw)

    nx, nt = 241, 180
    x, t_grid, theta_pinn, _q = evaluate_on_grid(model, nx, nt, T_END, device=device)
    np.savez_compressed(
        outdir / "pinn_case0_dpl_field.npz",
        x=x,
        t=t_grid,
        theta=theta_pinn,
        meta=meta,
    )
    torch.save(model.state_dict(), outdir / "pinn_case0_dpl_weights.pt")
    print(f"Saved {outdir / 'pinn_case0_dpl_field.npz'}")

    try:
        fd = simulate_model("DPL", "single_pulse")
        plot_comparison(
            outdir,
            theta_pinn,
            fd.T,
            x,
            fd.x,
            t_grid,
            fd.t,
            [0.04, 0.10, 0.20],
        )
        print(f"Figure: {outdir / 'case0_dpl_pinn_vs_fd_profiles.png'}")
    except Exception as exc:
        print("Skipping FD comparison:", exc)


if __name__ == "__main__":
    main()
