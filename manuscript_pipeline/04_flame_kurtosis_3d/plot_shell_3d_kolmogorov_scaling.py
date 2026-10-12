#!/usr/bin/env python3
"""Plot true 3-D shell VSF scaling for all synthetic shell widths."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from generate_shell_thickness_grid import (  # noqa: E402
    N_WIDTHS,
    R_OUTER_FIXED,
    WIDTH_MAX,
    WIDTH_MIN,
    width_grid,
)
from projected_shell_kurtosis import shell_projected_velocity  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402


SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
OUT_FIG = SHELL_DIR / "shell_3d_vsf_kolmogorov_scaling.png"
OUT_CSV = SHELL_DIR / "shell_3d_vsf_kolmogorov_slopes.csv"

N_GRID = 160
BOX_SIZE = 2.4
SEED = 7
N_BINS = 24
N_PAIRS = 1_500_000
RNG_SEED = 92653


def binned_vsf_3d(
    coords: np.ndarray,
    velocity: np.ndarray,
    n_bins: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = coords.shape[0]
    i = rng.integers(0, n_points, size=n_pairs)
    j = rng.integers(0, n_points, size=n_pairs)
    good_pair = i != j
    i = i[good_pair]
    j = j[good_pair]

    dr = coords[i] - coords[j]
    dv = velocity[i] - velocity[j]
    sep = np.linalg.norm(dr, axis=1)
    dv2 = dv**2

    good = sep > 0
    sep = sep[good]
    dv2 = dv2[good]

    edges = np.geomspace(np.nanmin(sep), np.nanpercentile(sep, 85), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    s2 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    which = np.digitize(sep, edges) - 1

    for idx_bin in range(n_bins):
        idx = np.flatnonzero(which == idx_bin)
        counts[idx_bin] = idx.size
        if idx.size:
            s2[idx_bin] = np.nanmean(dv2[idx])

    return centers, s2, counts


def shell_cells(vx: np.ndarray, box_size: float, r_inner: float, r_outer: float) -> tuple[np.ndarray, np.ndarray]:
    # Reuse the same shell mask definition as the projection scripts.
    coords_1d, _, _, _, shell_mask = shell_projected_velocity(vx, box_size, r_inner, r_outer)
    x, y, z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = np.column_stack([x[shell_mask], y[shell_mask], z[shell_mask]])
    velocity = vx[shell_mask]
    return coords, velocity


def main() -> None:
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass

    rng_field = np.random.default_rng(SEED)
    vx, _, _ = kolmogorov_velocity_grid(N_GRID, BOX_SIZE, rng_field)
    rng_pairs = np.random.default_rng(RNG_SEED)

    widths = width_grid(WIDTH_MIN, WIDTH_MAX, N_WIDTHS)
    cmap = plt.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    rows = []

    for width in widths:
        r_inner = R_OUTER_FIXED - float(width)
        r_outer = R_OUTER_FIXED
        coords, velocity = shell_cells(vx, BOX_SIZE, r_inner, r_outer)
        r, s2, counts = binned_vsf_3d(coords, velocity, N_BINS, N_PAIRS, rng_pairs)

        fit_mask = np.isfinite(s2) & (s2 > 0) & (counts > 50)
        fit_mask &= (r > 0.08) & (r < 0.65)
        slope, intercept = np.polyfit(np.log10(r[fit_mask]), np.log10(s2[fit_mask]), 1)

        color = cmap((float(width) - WIDTH_MIN) / (WIDTH_MAX - WIDTH_MIN))
        ax.loglog(
            r,
            s2,
            "o-",
            ms=3.4,
            lw=1.25,
            color=color,
            alpha=0.88,
            label=fr"$\Delta R={width:.3f}$",
        )
        rows.append((float(width), r_inner, r_outer, slope, coords.shape[0]))
        print(f"width={width:.4f} slope={slope:.3f} n_cells={coords.shape[0]}")

    ref_r = None
    ref_s2 = None
    ref_counts = None
    coords_ref, vel_ref = shell_cells(vx, BOX_SIZE, R_OUTER_FIXED - float(widths[len(widths) // 2]), R_OUTER_FIXED)
    ref_r, ref_s2, ref_counts = binned_vsf_3d(coords_ref, vel_ref, N_BINS, N_PAIRS, rng_pairs)
    ref_mask = np.isfinite(ref_s2) & (ref_s2 > 0) & (ref_counts > 50) & (ref_r > 0.08) & (ref_r < 0.65)
    anchor = np.nanmedian(ref_s2[ref_mask] / ref_r[ref_mask] ** (2.0 / 3.0))
    r_line = np.geomspace(np.nanmin(ref_r[ref_mask]), np.nanmax(ref_r[ref_mask]), 200)
    ax.loglog(r_line, anchor * r_line ** (2.0 / 3.0), "k--", lw=1.8, label=r"$r^{2/3}$")

    ax.set_xlabel(r"3D separation, $r$")
    ax.set_ylabel(r"$S_2(r)=\langle(\Delta v_x)^2\rangle$")
    ax.set_title("True 3D shell VSFs compared with Kolmogorov scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, title="Shell width")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with OUT_CSV.open("w", encoding="utf-8") as fobj:
        fobj.write("width,r_inner,r_outer,slope,n_shell_cells\n")
        for width, r_inner, r_outer, slope, n_cells in rows:
            fobj.write(f"{width:.8f},{r_inner:.8f},{r_outer:.8f},{slope:.8f},{n_cells}\n")

    print(f"Wrote {OUT_FIG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
