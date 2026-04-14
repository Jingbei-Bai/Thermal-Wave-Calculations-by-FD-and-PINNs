"""
Run 2D finite-difference reference for Case 0 on [0,1]^2.
No PyTorch — NumPy + `thermal_2d_case_library_tm_ppt.py`.

Default: y*-layered Gaussian peak on x*=0 (same as `compare_2d_fourier_cv_thermomass_fd.py`).
Use --uniform-flux for A(y*)=1.

Example:
    python run_2d_fd_case0.py --model Fourier --nx 81 --ny 81
    python run_2d_fd_case0.py --model CV
    python run_2d_fd_case0.py --model Thermomass --uniform-flux
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
    p = argparse.ArgumentParser(description="2D FD Case 0 reference (unit square).")
    p.add_argument("--model", type=str, default="Fourier", choices=["Fourier", "CV", "DPL", "Thermomass"])
    p.add_argument("--nx", type=int, default=81)
    p.add_argument("--ny", type=int, default=81)
    p.add_argument("--case", type=str, default="single_pulse", help="Case key passed to make_case (same as 1D library).")
    p.add_argument("--uniform-flux", action="store_true", help="Uniform influx in y* (no layered peak).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sub = "_layeredflux" if not args.uniform_flux else "_uniformflux"
    out = Path("results") / f"case0_2d_fd_reference{sub}"
    out.mkdir(parents=True, exist_ok=True)

    fd_kw: dict = {"nx": args.nx, "ny": args.ny}
    if not args.uniform_flux:
        fd_kw["flux_y_edges"] = DEFAULT_FLUX_Y_EDGES
        fd_kw["flux_layer_amps"] = DEFAULT_FLUX_LAYER_AMPS

    res = simulate_model_2d(args.model, args.case, **fd_kw)
    np.savez_compressed(
        out / f"fd_2d_{args.case}_{args.model.lower()}.npz",
        x=res.x,
        y=res.y,
        t=res.t,
        T=res.T,
        qx=res.qx,
        qy=res.qy,
    )
    print(f"Saved {out / f'fd_2d_{args.case}_{args.model.lower()}.npz'}")
    print(f"  nsub={res.meta.get('nsub_2d')}  dt_sub={res.meta.get('dt_sub_2d'):.4e}")

    if plt is None:
        return

    X, Y = np.meshgrid(res.x, res.y, indexing="ij")
    for ts in (0.04, 0.10, 0.20):
        it = min(nearest_time_index(res.t, ts), res.T.shape[0] - 1)
        fig, ax = plt.subplots(figsize=(4.9, 4.2))
        pcm = ax.pcolormesh(X.T, Y.T, res.T[it].T, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$y^*$")
        ax.set_title(rf"FD 2D {args.model}  $\theta^*$ at $t^*={res.t[it]:.3f}$")
        if not args.uniform_flux:
            for ye in DEFAULT_FLUX_Y_EDGES:
                if 0.0 < ye < 1.0:
                    ax.axhline(ye, color="cyan", ls="--", lw=0.9, alpha=0.65)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"fd2d_{args.model.lower()}_t_{res.t[it]:.3f}.png", dpi=200)
        plt.close(fig)
        print(f"  figure t*≈{ts}: {out / f'fd2d_{args.model.lower()}_t_{res.t[it]:.3f}.png'}")


if __name__ == "__main__":
    main()
