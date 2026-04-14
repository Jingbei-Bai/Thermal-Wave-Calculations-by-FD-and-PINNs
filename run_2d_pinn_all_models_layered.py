from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 2D PINN for Fourier/CV/Thermomass with layered boundary flux."
    )
    p.add_argument("--fast", action="store_true", help="Pass --fast to each PINN run.")
    p.add_argument(
        "--compare-fd",
        action="store_true",
        help="Pass --compare-fd to each PINN run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    laws = ("fourier", "cv", "thermomass")
    base_dir = Path(__file__).resolve().parent
    driver = base_dir / "pinn_2d_case0_square.py"
    if not driver.exists():
        raise FileNotFoundError(f"Missing driver: {driver}")

    for i, law in enumerate(laws, start=1):
        cmd = [sys.executable, str(driver), "--law", law]
        # Default of pinn_2d_case0_square is layered flux unless --uniform-flux is given.
        if args.fast:
            cmd.append("--fast")
        if args.compare_fd:
            cmd.append("--compare-fd")

        print(f"[2D PINN layered] {i}/{len(laws)} running: {law}")
        subprocess.run(cmd, check=True, cwd=str(base_dir))

    print("done 2d pinn layered: fourier/cv/thermomass")


if __name__ == "__main__":
    main()
