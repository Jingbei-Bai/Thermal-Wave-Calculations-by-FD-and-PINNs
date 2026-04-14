"""
Run 3D finite-difference Case 0 on [0,1]^3 (NumPy only).

Same boundary pattern as 2D: x*=0 Gaussian pulse with optional A(y*) layering (uniform in z*).

Example:
    python run_3d_fd_case0.py --model Fourier --nx 33 --ny 33 --nz 33
    python run_3d_fd_case0.py --model CV --uniform-flux
    python run_3d_fd_case0.py --model Thermomass --fast-grid
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
    p = argparse.ArgumentParser(description="3D FD Case 0 on unit cube.")
    p.add_argument("--model", type=str, default="Fourier", choices=["Fourier", "CV", "DPL", "Thermomass"])
    p.add_argument("--nx", type=int, default=33)
    p.add_argument("--ny", type=int, default=33)
    p.add_argument("--nz", type=int, default=33)
    p.add_argument("--case", type=str, default="single_pulse")
    p.add_argument("--uniform-flux", action="store_true", help="A(y*)=1 on x*=0.")
    p.add_argument(
        "--fast-grid",
        action="store_true",
        help="Use 25^3 grid (overrides nx,ny,nz) for a quicker smoke run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nx, ny, nz = (25, 25, 25) if args.fast_grid else (args.nx, args.ny, args.nz)

    sub = "_uniformflux" if args.uniform_flux else "_layeredflux"
    out = Path("results") / f"case0_3d_fd_reference{sub}"
    out.mkdir(parents=True, exist_ok=True)

    fd_kw: dict = {"nx": nx, "ny": ny, "nz": nz}
    if not args.uniform_flux:
        fd_kw["flux_y_edges"] = DEFAULT_FLUX_Y_EDGES
        fd_kw["flux_layer_amps"] = DEFAULT_FLUX_LAYER_AMPS

    res = simulate_model_3d(args.model, args.case, **fd_kw)
    np.savez_compressed(
        out / f"fd_3d_{args.case}_{args.model.lower()}.npz",
        x=res.x,
        y=res.y,
        z=res.z,
        t=res.t,
        T=res.T,
        qx=res.qx,
        qy=res.qy,
        qz=res.qz,
    )
    print(f"Saved {out / f'fd_3d_{args.case}_{args.model.lower()}.npz'}")
    print(f"  T shape={res.T.shape}  nsub={res.meta.get('nsub_3d')}  dt_sub={res.meta.get('dt_sub_3d'):.4e}")

    if plt is None:
        return

    kz = nz // 2
    X, Y = np.meshgrid(res.x, res.y, indexing="ij")
    for ts in (0.04, 0.10, 0.20):
        it = min(nearest_time_index(res.t, ts), res.T.shape[0] - 1)
        sl = res.T[it, :, :, kz]
        fig, ax = plt.subplots(figsize=(4.9, 4.2))
        pcm = ax.pcolormesh(X.T, Y.T, sl.T, shading="auto", cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$y^*$")
        ax.set_title(
            rf"FD 3D {args.model}  $\theta^*$  $z^*\approx{res.z[kz]:.2f}$  $t^*={res.t[it]:.3f}$"
        )
        if not args.uniform_flux:
            for ye in DEFAULT_FLUX_Y_EDGES:
                if 0.0 < ye < 1.0:
                    ax.axhline(ye, color="cyan", ls="--", lw=0.9, alpha=0.65)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"fd3d_{args.model.lower()}_xy_zmid_t_{res.t[it]:.3f}.png", dpi=200)
        plt.close(fig)
        print(f"  figure: {out / f'fd3d_{args.model.lower()}_xy_zmid_t_{res.t[it]:.3f}.png'}")


if __name__ == "__main__":
    main()
