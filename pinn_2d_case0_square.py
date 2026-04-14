"""
2D Case 0 — left-face Gaussian heat pulse on [0,1]^2 with optional y*-layered peak amplitude.

Physics-only PINN; --law {fourier,cv,thermomass}. Medium is isotropic; only q_x*(0,y*,t*) may vary in y*.

Example:
    python pinn_2d_case0_square.py --law fourier --fast
    python pinn_2d_case0_square.py --law cv --compare-fd
    python pinn_2d_case0_square.py --law thermomass --uniform-flux
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from boundary_flux_layers_2d import (
    DEFAULT_FLUX_LAYER_AMPS,
    DEFAULT_FLUX_Y_EDGES,
    build_q_left_torch,
    uniform_flux_edges_amps,
)
from pinn_2d_cv_tm_fourier_core import (
    FluxLaw2D,
    evaluate_on_grid_2d,
    material_tau_q_star,
    material_thermomass_params,
    plot_theta_slices_2d,
    train_pinn_2d,
)

T_END = 0.28
PULSE_CENTER = 0.055
PULSE_WIDTH = 0.012

TIME_WINDOWS_CASE0 = [
    (max(0.0, PULSE_CENTER - 4.0 * PULSE_WIDTH), min(T_END, PULSE_CENTER + 4.0 * PULSE_WIDTH))
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D PINN Case 0 on unit square (Fourier / CV / Thermomass).")
    p.add_argument(
        "--law",
        type=str,
        default="fourier",
        choices=["fourier", "cv", "thermomass"],
        help="Constitutive law for heat flux (2D vector extension).",
    )
    p.add_argument("--fast", action="store_true", help="Smaller budget: fewer steps/collocation, float32.")
    p.add_argument(
        "--compare-fd",
        action="store_true",
        help="After PINN, run matching 2D FD and save comparison figures.",
    )
    p.add_argument(
        "--uniform-flux",
        action="store_true",
        help="Left boundary flux uniform in y* (default: y-layered peak amplitudes).",
    )
    p.add_argument("--flux-y-edges", type=str, default=None, help="Override: comma-separated y* edges, e.g. 0,0.33,0.66,1")
    p.add_argument(
        "--flux-layer-amps",
        type=str,
        default=None,
        help="Override: comma-separated amplitudes A(y*) per band, len = len(edges)-1.",
    )
    return p.parse_args()


def _flux_spec_from_args(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if args.uniform_flux:
        return uniform_flux_edges_amps()
    if args.flux_y_edges is not None and args.flux_layer_amps is not None:
        edges = tuple(float(x.strip()) for x in args.flux_y_edges.split(","))
        amps = tuple(float(x.strip()) for x in args.flux_layer_amps.split(","))
        return edges, amps
    return DEFAULT_FLUX_Y_EDGES, DEFAULT_FLUX_LAYER_AMPS


def _plot_pinn_vs_fd(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    theta_pinn: np.ndarray,
    t_pinn: np.ndarray,
    theta_fd: np.ndarray,
    t_fd: np.ndarray,
    law_label: str,
    t_snap: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def nearest_t(arr: np.ndarray, ts: float) -> int:
        return int(np.argmin(np.abs(arr - ts)))

    X, Y = np.meshgrid(x, y, indexing="ij")
    n = len(t_snap)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 7.8), squeeze=False)
    for k, ts in enumerate(t_snap):
        it_p = min(nearest_t(t_pinn, ts), theta_pinn.shape[2] - 1)
        it_f = min(nearest_t(t_fd, ts), theta_fd.shape[0] - 1)
        tp = theta_pinn[:, :, it_p]
        tf = theta_fd[it_f]
        vmin = float(min(tp.min(), tf.min()))
        vmax = float(max(tp.max(), tf.max()))
        for row, (dat, title, t_show) in enumerate(
            (
                (tp, f"PINN {law_label}", t_pinn[it_p]),
                (tf, f"FD 2D {law_label}", t_fd[it_f]),
            )
        ):
            ax = axes[row, k]
            pcm = ax.pcolormesh(X.T, Y.T, dat.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$y^*$")
            ax.set_title(rf"{title}  $t^*={t_show:.3f}$")
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        diff = tp - tf
        err = float(np.sqrt(np.mean(diff**2)))
        axes[1, k].text(
            0.02,
            0.02,
            rf"RMSE$_{{\mathrm{{PINN-FD}}}}$={err:.3e}",
            transform=axes[1, k].transAxes,
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    law_map = {
        "fourier": FluxLaw2D.FOURIER,
        "cv": FluxLaw2D.CV,
        "thermomass": FluxLaw2D.THERMOMASS,
    }
    law = law_map[args.law]

    flux_edges, flux_amps = _flux_spec_from_args(args)
    q_left_torch = build_q_left_torch(PULSE_CENTER, PULSE_WIDTH, flux_edges, flux_amps, time_amp=1.0)

    flux_tag = "_uniformflux" if args.uniform_flux else "_layeredflux"
    outdir = Path("results") / f"case0_2d_square_{args.law}{flux_tag}_pinn"
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2D PINN Case 0  law={args.law}  device={device}")
    print(f"  left flux: y_edges={flux_edges}  A(y*)={flux_amps}")

    tau_q: float | None = None
    tm_params: dict[str, float] | None = None
    if law == FluxLaw2D.CV:
        tau_q = material_tau_q_star()
        print(f"  tau_q* = {tau_q:.8f}")
    elif law == FluxLaw2D.THERMOMASS:
        tm_params = material_thermomass_params()
        print(
            f"  thermomass params: tau_tm*={tm_params['tau_tm']:.6g}  epsilon*={tm_params['tm_epsilon']:.6g}  "
            f"(eta* deprecated in strict PPT closure)"
        )

    train_kw: dict = {}
    if args.fast:
        train_kw.update(
            n_steps=8_000,
            n_colloc=5_000,
            n_ic=400,
            n_bc=400,
            log_every=2_000,
            hidden=56,
            n_trunk_layers=4,
            use_float64=False,
            use_lbfgs=True,
            lbfgs_rounds=18,
            lbfgs_colloc=4_000,
            lbfgs_ic=400,
            lbfgs_bc=400,
        )
    else:
        train_kw.update(
            n_steps=28_000,
            n_colloc=14_000,
            n_ic=1_200,
            n_bc=1_200,
            log_every=2_000,
            hidden=80,
            n_trunk_layers=5,
            use_float64=True,
            use_lbfgs=True,
            lbfgs_rounds=45,
            lbfgs_colloc=12_000,
            lbfgs_ic=1_200,
            lbfgs_bc=1_200,
        )

    model, meta = train_pinn_2d(
        law=law,
        t_end=T_END,
        q_left_torch=q_left_torch,
        time_windows=TIME_WINDOWS_CASE0,
        tau_q_star=tau_q,
        thermomass_params=tm_params,
        device=device,
        **train_kw,
    )
    meta["flux_y_edges"] = list(flux_edges)
    meta["flux_layer_amps"] = list(flux_amps)
    meta["uniform_flux"] = bool(args.uniform_flux)

    nx, ny, nt = 48, 48, 60
    x, y, t_grid, theta, qx, qy = evaluate_on_grid_2d(model, nx, ny, nt, T_END, device=device)

    if args.compare_fd:
        from thermal_2d_case_library_tm_ppt import simulate_model_2d

        fd_name = {"fourier": "Fourier", "cv": "CV", "thermomass": "Thermomass"}[args.law]
        fd_kw: dict = {"nx": nx, "ny": ny}
        if not args.uniform_flux:
            fd_kw["flux_y_edges"] = flux_edges
            fd_kw["flux_layer_amps"] = flux_amps
        fd_res = simulate_model_2d(fd_name, "single_pulse", **fd_kw)
        np.savez_compressed(
            outdir / f"fd_2d_ref_{args.law}.npz",
            x=fd_res.x,
            y=fd_res.y,
            t=fd_res.t,
            T=fd_res.T,
            qx=fd_res.qx,
            qy=fd_res.qy,
        )
        cmp_path = str(outdir / f"case0_2d_pinn_vs_fd_{args.law}.png")
        _plot_pinn_vs_fd(
            cmp_path,
            x,
            y,
            theta,
            t_grid,
            fd_res.T,
            fd_res.t,
            args.law.upper(),
            [0.04, 0.10, 0.20],
        )
        print(f"FD reference + comparison: {cmp_path}")
    np.savez_compressed(
        outdir / f"pinn_2d_case0_{args.law}.npz",
        x=x,
        y=y,
        t=t_grid,
        theta=theta,
        qx=qx,
        qy=qy,
    )
    (outdir / f"pinn_2d_case0_{args.law}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), outdir / f"pinn_2d_case0_{args.law}_weights.pt")
    print(f"Saved {outdir / f'pinn_2d_case0_{args.law}.npz'}")

    mark_y = [e for e in flux_edges if 0.0 < e < 1.0] if not args.uniform_flux else None
    plot_theta_slices_2d(
        str(outdir / f"case0_2d_theta_{args.law}.png"),
        x,
        y,
        theta,
        t_grid,
        [0.04, 0.10, 0.20],
        title_prefix=args.law.upper(),
        horizontal_mark_y=mark_y,
    )
    print(f"Figure: {outdir / f'case0_2d_theta_{args.law}.png'}")


if __name__ == "__main__":
    main()
