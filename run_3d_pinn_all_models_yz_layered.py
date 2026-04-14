from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 3D PINN for Fourier/CV/Thermomass with yz-layered flux.")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--uniform-flux", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    driver = base / "pinn_3d_case0_cube.py"
    laws = ("fourier", "cv", "thermomass")
    for i, law in enumerate(laws, start=1):
        cmd = [sys.executable, str(driver), "--law", law]
        if args.fast:
            cmd.append("--fast")
        if args.uniform_flux:
            cmd.append("--uniform-flux")
        print(f"[3D PINN yz-layered] {i}/{len(laws)} running: {law}")
        subprocess.run(cmd, check=True, cwd=str(base))
    print("done 3d pinn: fourier/cv/thermomass")


if __name__ == "__main__":
    main()
