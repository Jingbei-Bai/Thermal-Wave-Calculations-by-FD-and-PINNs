"""
Finite-difference comparison: Fourier vs CV vs Thermomass on [0,1]^2, same Case 0 setup.

Left boundary: Gaussian in t* with **y*-layered peak amplitude** A(y*) (default three bands),
same for all three models. No material layering.

Outputs under ``results/case0_2d_compare_fourier_cv_tm_fd/``:
    combined snapshots (1x3 panels), per-model npz, optional boundary-flux sketch.

Example:
    python compare_2d_fourier_cv_thermomass_fd.py
    python compare_2d_fourier_cv_thermomass_fd.py --nx 101 --ny 101
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boundary_flux_layers_2d import DEFAULT_FLUX_LAYER_AMPS, DEFAULT_FLUX_Y_EDGES
from thermal_2d_case_library_tm_ppt import nearest_time_index, simulate_model_2d

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare FD 2D: Fourier, CV, Thermomass (layered boundary flux).")
    p.add_argument("--nx", type=int, default=81)
    p.add_argument("--ny", type=int, default=81)
    p.add_argument("--case", type=str, default="single_pulse")
    p.add_argument(
        "--uniform-flux",
        action="store_true",
        help="Use uniform A(y*)=1 on x*=0 instead of layered amplitudes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path("results") / "case0_2d_compare_fourier_cv_tm_fd"
    out.mkdir(parents=True, exist_ok=True)

    flux_kw: dict = {}
    if not args.uniform_flux:
        flux_kw["flux_y_edges"] = DEFAULT_FLUX_Y_EDGES
        flux_kw["flux_layer_amps"] = DEFAULT_FLUX_LAYER_AMPS

    models = ("Fourier", "CV", "Thermomass")
    results: dict = {}
    for m in models:
        results[m] = simulate_model_2d(m, args.case, nx=args.nx, ny=args.ny, **flux_kw)
        np.savez_compressed(
            out / f"fd_2d_{m.lower()}.npz",
            x=results[m].x,
            y=results[m].y,
            t=results[m].t,
            T=results[m].T,
            qx=results[m].qx,
            qy=results[m].qy,
        )
        print(f"saved {out / f'fd_2d_{m.lower()}.npz'}")

    if plt is None:
        return

    y = results["Fourier"].y
    if not args.uniform_flux:
        from boundary_flux_layers_2d import layer_amplitude_numpy

        amp_line = layer_amplitude_numpy(y, DEFAULT_FLUX_Y_EDGES, DEFAULT_FLUX_LAYER_AMPS)
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        ax.plot(y, amp_line, "k-", lw=2.0)
        ax.set_xlabel(r"$y^*$")
        ax.set_ylabel(r"$A(y^*)$")
        ax.set_title(r"Left boundary flux prefactor $q_x^*|_{x^*=0} \propto A(y^*)\,q_L(t^*)$")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "boundary_flux_A_of_y.png", dpi=200)
        plt.close(fig)

    t_snap = [0.04, 0.10, 0.20]
    X, Y = np.meshgrid(results["Fourier"].x, results["Fourier"].y, indexing="ij")

    for ts in t_snap:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), squeeze=False)
        vmin, vmax = 1e30, -1e30
        Ts: dict[str, np.ndarray] = {}
        for m in models:
            it = min(nearest_time_index(results[m].t, ts), results[m].T.shape[0] - 1)
            Ts[m] = results[m].T[it]
            vmin = min(vmin, float(Ts[m].min()))
            vmax = max(vmax, float(Ts[m].max()))
        for ax, m in zip(axes[0], models):
            it = min(nearest_time_index(results[m].t, ts), results[m].T.shape[0] - 1)
            pcm = ax.pcolormesh(X.T, Y.T, Ts[m].T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$y^*$")
            ax.set_title(rf"{m}  $\theta^*$  $t^*={results[m].t[it]:.3f}$")
            if not args.uniform_flux:
                for ye in DEFAULT_FLUX_Y_EDGES:
                    if 0.0 < ye < 1.0:
                        ax.axhline(ye, color="cyan", ls="--", lw=0.9, alpha=0.65)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("FD 2D: Fourier vs CV vs Thermomass (same layered boundary flux)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"compare_theta_t_{ts:.2f}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"figure: {out / f'compare_theta_t_{ts:.2f}.png'}")


if __name__ == "__main__":
    main()
