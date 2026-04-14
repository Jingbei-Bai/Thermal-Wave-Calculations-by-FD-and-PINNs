"""
FD comparison on [0,1]^3: Fourier vs CV vs Thermomass (same setup as 2D compare script).

Saves mid- z* slice figures (x*–y*) at selected times. Default grid 28^3 — increase with care (memory).

Example:
    python compare_3d_fourier_cv_thermomass_fd.py
    python compare_3d_fourier_cv_thermomass_fd.py --n 36 --uniform-flux
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boundary_flux_layers_2d import DEFAULT_FLUX_LAYER_AMPS, DEFAULT_FLUX_Y_EDGES
from thermal_3d_case_library_tm_ppt import nearest_time_index, simulate_model_3d

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare FD 3D: Fourier, CV, Thermomass.")
    p.add_argument("--n", type=int, default=28, help="Use nx=ny=nz=n (cube).")
    p.add_argument("--case", type=str, default="single_pulse")
    p.add_argument("--uniform-flux", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path("results") / "case0_3d_compare_fourier_cv_tm_fd"
    out.mkdir(parents=True, exist_ok=True)

    flux_kw: dict = {}
    if not args.uniform_flux:
        flux_kw["flux_y_edges"] = DEFAULT_FLUX_Y_EDGES
        flux_kw["flux_layer_amps"] = DEFAULT_FLUX_LAYER_AMPS

    n = args.n
    models = ("Fourier", "CV", "Thermomass")
    results: dict = {}
    for m in models:
        results[m] = simulate_model_3d(m, args.case, nx=n, ny=n, nz=n, **flux_kw)
        np.savez_compressed(
            out / f"fd_3d_{m.lower()}.npz",
            x=results[m].x,
            y=results[m].y,
            z=results[m].z,
            t=results[m].t,
            T=results[m].T,
        )
        print(f"saved {out / f'fd_3d_{m.lower()}.npz'}  Tshape={results[m].T.shape}")

    if plt is None:
        return

    kz = n // 2
    X, Y = np.meshgrid(results["Fourier"].x, results["Fourier"].y, indexing="ij")
    t_snap = [0.04, 0.10, 0.20]

    for ts in t_snap:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), squeeze=False)
        vmin, vmax = 1e30, -1e30
        Ts: dict[str, np.ndarray] = {}
        for m in models:
            it = min(nearest_time_index(results[m].t, ts), results[m].T.shape[0] - 1)
            Ts[m] = results[m].T[it, :, :, kz]
            vmin = min(vmin, float(Ts[m].min()))
            vmax = max(vmax, float(Ts[m].max()))
        for ax, m in zip(axes[0], models):
            it = min(nearest_time_index(results[m].t, ts), results[m].T.shape[0] - 1)
            pcm = ax.pcolormesh(X.T, Y.T, Ts[m].T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x^*$")
            ax.set_ylabel(r"$y^*$")
            ax.set_title(rf"{m}  $z^*={results[m].z[kz]:.3f}$  $t^*={results[m].t[it]:.3f}$")
            if not args.uniform_flux:
                for ye in DEFAULT_FLUX_Y_EDGES:
                    if 0.0 < ye < 1.0:
                        ax.axhline(ye, color="cyan", ls="--", lw=0.9, alpha=0.65)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("FD 3D: Fourier vs CV vs Thermomass (mid- z* slice)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"compare_xy_zmid_t_{ts:.2f}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"figure: {out / f'compare_xy_zmid_t_{ts:.2f}.png'}")


if __name__ == "__main__":
    main()
