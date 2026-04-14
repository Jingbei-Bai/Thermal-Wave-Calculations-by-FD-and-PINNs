"""
Case 1 / 2 / 4 — Thermomass (PPT-style) PINN, same boundary fluxes as `thermal_1d_case_library_tm_ppt.py`.

Reuses flux definitions and time windows from `pinn_cv_cases_124_tm_ppt.py`.
Default training matches high-quality preset (see `pinn_case0_thermomass_unsupervised.py`); use `--fast` for a shorter run.

Examples:
    python pinn_thermomass_cases_124_tm_ppt.py --case double_pulse
    python pinn_thermomass_cases_124_tm_ppt.py --all
    python pinn_thermomass_cases_124_tm_ppt.py --case heat_cool --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from thermal_1d_case_library_tm_ppt import make_case, simulate_model

from pinn_cv_cases_124_tm_ppt import CASE_SPECS
from pinn_cv_tm_ppt_core import (
    evaluate_on_grid,
    material_thermomass_params,
    train_pinn_thermomass,
)

T_END_DEFAULT = 0.28


def plot_comparison(
    outdir: Path,
    case_tag: str,
    title: str,
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
        axes[1].plot(x_fd[1:], fd_theta[it_f, 1:], label=fr"FD Thermomass $t^*={t_fd[it_f]:.3f}$")
    axes[0].set_title(f"{title} | PINN Thermomass (unsupervised)")
    axes[1].set_title("Finite difference Thermomass (reference)")
    for ax in axes:
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$\theta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{case_tag}_thermomass_pinn_vs_fd.png", dpi=200)
    plt.close(fig)


def run_one_case(
    case_key: str,
    device: str,
    train_kw: dict,
    t_end: float,
) -> None:
    spec = CASE_SPECS[case_key]
    lib = spec["library_name"]
    tag = spec["tag"]
    p_meta, _, _ = make_case(lib)
    assert abs(float(p_meta["t_end"]) - t_end) < 1e-9, (p_meta["t_end"], t_end)

    tm = material_thermomass_params()
    print(
        f"\n=== {spec['title']} | Thermomass "
        f"tau_tm*={tm['tau_tm']:.6e} epsilon*={tm['tm_epsilon']:.6e} ==="
    )

    outdir = Path("results") / f"pinn_thermomass_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    model, meta = train_pinn_thermomass(
        t_end=t_end,
        q_left_torch=spec["q_left"],
        q_right_torch=spec["q_right"],
        time_windows=spec["time_windows"],
        tau_tm=tm["tau_tm"],
        tm_eta=tm["tm_eta"],
        theta_base=tm["tm_theta_base"],
        theta_floor=tm["tm_theta_floor"],
        u_clip=tm["tm_u_clip"],
        device=device,
        **train_kw,
    )
    meta["case"] = case_key
    meta["title"] = spec["title"]

    nx, nt = 241, 180
    x, t_grid, theta_pinn, _q = evaluate_on_grid(model, nx, nt, t_end, device=device)
    np.savez_compressed(
        outdir / f"pinn_thermomass_{tag}_field.npz",
        x=x,
        t=t_grid,
        theta=theta_pinn,
        meta=meta,
    )
    torch.save(model.state_dict(), outdir / f"pinn_thermomass_{tag}_weights.pt")
    print(f"Saved: {outdir / f'pinn_thermomass_{tag}_field.npz'}")

    try:
        fd = simulate_model("Thermomass", lib)
        plot_comparison(
            outdir,
            tag,
            spec["title"],
            theta_pinn,
            fd.T,
            x,
            fd.x,
            t_grid,
            fd.t,
            spec["plot_times"],
        )
        print(f"Figure: {outdir / f'{tag}_thermomass_pinn_vs_fd.png'}")
    except Exception as exc:
        print("Skipping FD plot:", exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thermomass PINN for cases 1, 2, 4 (library tm_ppt).")
    p.add_argument(
        "--case",
        choices=list(CASE_SPECS.keys()),
        default="double_pulse",
        help="Library case name.",
    )
    p.add_argument("--all", action="store_true", help="Run all three cases in order.")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster run (20k steps, smaller net, float32, no L-BFGS). Default is high-quality.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_end = T_END_DEFAULT

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

    keys = list(CASE_SPECS.keys()) if args.all else [args.case]
    print(f"Thermomass cases 1/2/4 | device={device} fast={args.fast} (default=high-quality) cases={keys}")

    for k in keys:
        run_one_case(k, device, train_kw, t_end)


if __name__ == "__main__":
    main()
