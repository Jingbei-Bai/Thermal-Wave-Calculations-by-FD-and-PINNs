"""
3D PINN Case 0 on unit cube for Fourier/CV/Thermomass with yz-layered left-face influx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from boundary_flux_layers_3d import (
    DEFAULT_FLUX_Y_AMPS_3D,
    DEFAULT_FLUX_Y_EDGES_3D,
    DEFAULT_FLUX_Z_AMPS_3D,
    DEFAULT_FLUX_Z_EDGES_3D,
    build_q_left_torch_3d,
    uniform_flux_edges_amps_3d,
)
from pinn_3d_cv_tm_fourier_core import (
    FluxLaw3D,
    evaluate_on_grid_3d,
    material_tau_q_star,
    material_thermomass_params,
    train_pinn_3d,
)

T_END = 0.10
PULSE_CENTER = 0.055
PULSE_WIDTH = 0.012


def _plot_3d_slices(
    out_dir: Path,
    law: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    theta: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t_snap = [0.04, 0.07, 0.10]
    ix = len(x) // 2
    iz = len(z) // 2
    Xxy, Yxy = np.meshgrid(x, y, indexing="ij")
    Yyz, Zyz = np.meshgrid(y, z, indexing="ij")
    for ts in t_snap:
        it = int(np.argmin(np.abs(t - ts)))
        sl_xy = theta[:, :, iz, it]
        sl_yz = theta[ix, :, :, it]
        vmin = float(min(sl_xy.min(), sl_yz.min()))
        vmax = float(max(sl_xy.max(), sl_yz.max()))
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), squeeze=False)
        ax0 = axes[0, 0]
        pcm0 = ax0.pcolormesh(Xxy.T, Yxy.T, sl_xy.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax0.set_aspect("equal")
        ax0.set_xlabel(r"$x^*$")
        ax0.set_ylabel(r"$y^*$")
        ax0.set_title(rf"{law.upper()} XY @ $z^*={z[iz]:.3f}$, $t^*={t[it]:.3f}$")
        fig.colorbar(pcm0, ax=ax0, fraction=0.046, pad=0.04)
        ax1 = axes[0, 1]
        pcm1 = ax1.pcolormesh(Yyz.T, Zyz.T, sl_yz.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax1.set_aspect("equal")
        ax1.set_xlabel(r"$y^*$")
        ax1.set_ylabel(r"$z^*$")
        ax1.set_title(rf"{law.upper()} YZ @ $x^*={x[ix]:.3f}$, $t^*={t[it]:.3f}$")
        fig.colorbar(pcm1, ax=ax1, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"case0_3d_{law}_slices_t_{t[it]:.3f}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D PINN Case0 cube (Fourier/CV/Thermomass, yz-layered flux).")
    p.add_argument("--law", type=str, default="fourier", choices=["fourier", "cv", "thermomass"])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--uniform-flux", action="store_true")
    p.add_argument("--flux-y-edges", type=str, default=None)
    p.add_argument("--flux-y-amps", type=str, default=None)
    p.add_argument("--flux-z-edges", type=str, default=None)
    p.add_argument("--flux-z-amps", type=str, default=None)
    return p.parse_args()


def _spec(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if args.uniform_flux:
        return uniform_flux_edges_amps_3d()
    if all(v is not None for v in (args.flux_y_edges, args.flux_y_amps, args.flux_z_edges, args.flux_z_amps)):
        ye = tuple(float(x.strip()) for x in args.flux_y_edges.split(","))
        ya = tuple(float(x.strip()) for x in args.flux_y_amps.split(","))
        ze = tuple(float(x.strip()) for x in args.flux_z_edges.split(","))
        za = tuple(float(x.strip()) for x in args.flux_z_amps.split(","))
        return ye, ya, ze, za
    return DEFAULT_FLUX_Y_EDGES_3D, DEFAULT_FLUX_Y_AMPS_3D, DEFAULT_FLUX_Z_EDGES_3D, DEFAULT_FLUX_Z_AMPS_3D


def main() -> None:
    args = parse_args()
    law = {"fourier": FluxLaw3D.FOURIER, "cv": FluxLaw3D.CV, "thermomass": FluxLaw3D.THERMOMASS}[args.law]
    y_edges, y_amps, z_edges, z_amps = _spec(args)
    q_left = build_q_left_torch_3d(PULSE_CENTER, PULSE_WIDTH, y_edges, y_amps, z_edges, z_amps)

    tau_q = material_tau_q_star() if law == FluxLaw3D.CV else None
    tm = material_thermomass_params() if law == FluxLaw3D.THERMOMASS else None

    out = Path("results") / f"case0_3d_cube_{args.law}_{'uniformflux' if args.uniform_flux else 'yzlayeredflux'}_pinn"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"3D PINN law={args.law} device={device} t_end={T_END}")

    kw = (
        dict(n_steps=5000, n_colloc=4500, n_ic=400, n_bc=500, hidden=64, n_layers=4, log_every=500, use_float64=False, lbfgs_rounds=10)
        if args.fast
        else dict(n_steps=22000, n_colloc=20000, n_ic=1600, n_bc=1600, hidden=88, n_layers=6, log_every=1000, use_float64=False, lbfgs_rounds=35)
    )

    model, meta = train_pinn_3d(
        law=law,
        t_end=T_END,
        q_left_torch=q_left,
        tau_q_star=tau_q,
        thermomass_params=tm,
        device=device,
        **kw,
    )

    nx, ny, nz, nt = (22, 22, 22, 36) if args.fast else (28, 28, 28, 44)
    x, y, z, t, theta = evaluate_on_grid_3d(model, nx, ny, nz, nt, T_END, device=device)
    np.savez_compressed(out / f"pinn_3d_case0_{args.law}.npz", x=x, y=y, z=z, t=t, theta=theta)
    meta.update(
        {
            "flux_y_edges": list(y_edges),
            "flux_y_amps": list(y_amps),
            "flux_z_edges": list(z_edges),
            "flux_z_amps": list(z_amps),
            "uniform_flux": bool(args.uniform_flux),
            "nx_eval": nx,
            "ny_eval": ny,
            "nz_eval": nz,
            "nt_eval": nt,
        }
    )
    (out / f"pinn_3d_case0_{args.law}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out / f"pinn_3d_case0_{args.law}_weights.pt")
    _plot_3d_slices(out, args.law, x, y, z, t, theta)
    print(f"saved {out / f'pinn_3d_case0_{args.law}.npz'}")


if __name__ == "__main__":
    main()
