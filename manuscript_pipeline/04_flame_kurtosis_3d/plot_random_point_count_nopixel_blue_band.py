#!/usr/bin/env python3
"""Mean +/- 1 sigma across N of the no-pixelization dK/dlogr sweep (blue band).

Companion to the (pixelized) purple-band comparison: same style, same N
sweep, but each run was generated with
`generate_random_point_expanding_shell_grid.py --no-pixelize`, which feeds
every random point's own projected (y, z, v) straight into the VSF/kurtosis
pair statistics instead of first averaging points that land in the same sky
pixel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
DEFAULT_RUN_DIRS = [
    BASE_DIR / "synthetic_projected_shell_random_points_nopixel_n5000",
    BASE_DIR / "synthetic_projected_shell_random_points_nopixel_n20000",
    BASE_DIR / "synthetic_projected_shell_random_points_nopixel_n50000",
    BASE_DIR / "synthetic_projected_shell_random_points_nopixel_n100000",
]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def logistic(log_w: np.ndarray, floor: float, x0: float, scale: float) -> np.ndarray:
    """Sigmoid rising from `floor` (small width) to 0 (large width)."""
    return floor / (1.0 + np.exp((log_w - x0) / scale))


def n_points_from_metadata(run_dir: Path) -> int:
    import json

    metadata = run_dir / "random_point_expanding_shell_metadata.json"
    return int(json.loads(metadata.read_text(encoding="utf-8"))["n_points_per_shell"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "random_point_count_dK_dlogr_comparison_blue_band.png",
    )
    args = parser.parse_args()

    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass

    run_dirs = sorted(args.run_dirs, key=n_points_from_metadata)
    cmap = plt.colormaps["Blues"]

    widths = None
    slope_matrix = []
    fallback_matrix = []
    labels = []
    for i, run_dir in enumerate(run_dirs):
        csv_path = run_dir / "kurtosis_dK_dlogr_slope_vs_shell_width_aperture08_overlay.csv"
        data = load_csv(csv_path)
        n_points = n_points_from_metadata(run_dir)
        x = data["width_over_rout"]
        if widths is None:
            widths = x
        elif not np.allclose(x, widths):
            raise ValueError(f"{csv_path} has a different width grid than the first run")
        slope_matrix.append(data["full_dK_dlogr"])
        fallback_matrix.append(data["full_zero_fallback"].astype(bool))
        labels.append(f"N={n_points:,}")

    slope_matrix = np.array(slope_matrix)
    fallback_matrix = np.array(fallback_matrix)
    mean_slope = np.nanmean(slope_matrix, axis=0)
    std_slope = np.nanstd(slope_matrix, axis=0)

    # Exclude widths where every run fell back to slope=0 (too few kurtosis
    # bins above the Gaussian floor to fit) -- those aren't real measurements.
    has_fit = ~np.all(fallback_matrix, axis=0)
    log_w_fit = np.log10(widths[has_fit])
    fit_params, _ = curve_fit(
        logistic,
        log_w_fit,
        mean_slope[has_fit],
        p0=[mean_slope[has_fit].min(), np.median(log_w_fit), 0.5],
    )
    log_w_dense = np.linspace(log_w_fit.min(), log_w_fit.max(), 200)
    fit_curve = logistic(log_w_dense, *fit_params)

    fig, ax = plt.subplots(figsize=(9.0, 6.75))

    ax.fill_between(
        widths,
        mean_slope - std_slope,
        mean_slope + std_slope,
        color="0.75",
        alpha=0.6,
        label=r"mean $\pm 1\sigma$ across N",
        zorder=1,
    )
    ax.plot(widths, mean_slope, "-", color="black", lw=2.2, label="mean across N", zorder=4)
    ax.plot(
        10**log_w_dense,
        fit_curve,
        "--",
        color="crimson",
        lw=1.8,
        label=fr"logistic fit (floor$={fit_params[0]:.2f}$)",
        zorder=5,
    )

    n_runs = len(run_dirs)
    for i, (slopes, label) in enumerate(zip(slope_matrix, labels)):
        shade = 0.3 + 0.6 * (i / max(n_runs - 1, 1))
        ax.plot(
            widths,
            slopes,
            "o--",
            ms=5.0,
            lw=1.3,
            color=cmap(shade),
            alpha=0.9,
            label=label,
            zorder=2,
        )

    ax.axvline(0.1, color="0.25", ls=":", lw=1.1)
    ax.axhline(0, color="0.25", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    ax.set_ylabel(r"$dK/d\log_{10} r$")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=11, ncol=2, loc="lower right")
    fig.tight_layout()

    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Wrote {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
