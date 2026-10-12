#!/usr/bin/env python3
"""Overplot dK/dlogr shell-width curves for different random point counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
DEFAULT_RUNS = [
    BASE_DIR / "synthetic_projected_shell_random_points_vbulk0p0",
    BASE_DIR / "synthetic_projected_shell_random_points_vbulk0p0_n10000",
]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def point_count_from_metadata(run_dir: Path) -> int:
    metadata = run_dir / "random_point_expanding_shell_metadata.json"
    text = metadata.read_text(encoding="utf-8")
    import json

    return int(json.loads(text)["n_points_per_shell"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "synthetic_projected_shell_random_points_vbulk0p0")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    colors = ["C0", "C3", "C2", "C4", "C1", "C5"]
    run_dirs = sorted(args.run_dirs, key=point_count_from_metadata)
    for i, run_dir in enumerate(run_dirs):
        csv_path = run_dir / "kurtosis_dK_dlogr_slope_vs_shell_width_aperture08_overlay.csv"
        data = load_csv(csv_path)
        n_points = point_count_from_metadata(run_dir)
        color = colors[i % len(colors)]
        x = data["width_over_rout"]
        ax.plot(
            x,
            data["full_dK_dlogr"],
            "o-",
            color=color,
            alpha=0.35,
            lw=1.2,
            ms=4.5,
            label=f"full, N={n_points:,}",
        )
        ax.plot(
            x,
            data["aperture_dK_dlogr"],
            "s-",
            color=color,
            alpha=0.95,
            lw=1.7,
            ms=5.0,
            label=fr"$R_{{proj}}\leq0.8r_{{out}}$, N={n_points:,}",
        )

    ax.axvline(0.1, color="0.25", ls=":", lw=1.1, label=r"$\Delta R/r_{out}=0.1$")
    ax.axhline(0, color="0.35", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$(r_{out}-r_{in})/r_{out}$")
    ax.set_ylabel(r"Linear slope, $dK/d\log_{10}r$")
    ax.set_title(r"Effect of random point count on $dK/d\log r$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = args.output_dir / "random_point_count_dK_dlogr_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
