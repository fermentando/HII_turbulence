#!/usr/bin/env python3
"""Three-panel diagnostic for the N/pixel-size-dependent dK/dlogr artifact.

Companion to plot_fixed_rmin_comparison.py. That script showed the artifact
by coarsening the bin FLOOR (r_min) on raw, un-pixelized points -- it never
touches the points themselves, just where the smallest separation bin
starts. This script instead actually PIXELIZES the raw points: points
landing in the same (y, z) grid cell of a given pixel size are averaged
together into one point (one velocity) before anything else is computed --
the way a real telescope beam/fiber merges a sightline's near-side and
far-side contributions. See rmin_projection_artifact_explanation.md
(particularly "Connection to the pixelized pipeline") for why this averaging
is expected to suppress the near/far-side projection-contamination artifact
rather than just relabeling it.

Left panel: the N-sweep (5,000/20,000/50,000/100,000), with every N
pixelized at the SAME fixed pixel size -- N=5,000's own per-width r_min
(the coarsest/"honest" one, since N=5,000 needs no subsampling at all) --
isolating genuine N-dependence from the pixelization-scale choice.

Middle panel: N fixed at 100,000 (same raw points throughout), pixelized at
each size in PIXEL_SWEEP (0.001-0.1 r_HII, geometric, identical range to
RMIN_SWEEP in plot_fixed_rmin_comparison.py) -- demonstrates how the trend
responds to pixel size directly, on identical underlying data. Each pixel
size's points are joined by a line (no fit), colored by a continuous
pixel_size/r_HII colorbar.

Right panel: the deepest (most negative) raw dK/dlogr value reached by each
pixel-size curve in the middle panel, plotted against pixel_size/r_HII,
log-log, points joined by a line -- no curve fitting anywhere in this
figure, per request.

Both panels reuse the already-generated raw point data (y, z, projected_v
saved in each npz); no new random points are drawn -- only the pixelization
step (grouping and averaging) is new.

Caching: separate cache files from plot_fixed_rmin_comparison.py
(_fixed_pixelation_{left,right}_cache.npz), so running this script never
touches that one's cache or output, and vice versa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402
from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from generate_shell_thickness_grid import width_grid, shell_output_path, R_OUTER_FIXED  # noqa: E402
from plot_shell_dk_dlogr_aperture_comparison import fit_k_vs_logr_above_floor  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
N_VALUES = [5000, 20000, 50000, 100000]
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
PIXEL_SWEEP = np.geomspace(0.001, 0.1, 6)  # same range/spacing as RMIN_SWEEP

LEFT_CACHE = BASE_DIR / "_fixed_pixelation_left_cache.npz"
RIGHT_CACHE = BASE_DIR / "_fixed_pixelation_right_cache.npz"


def run_dir(n_points: int) -> Path:
    return BASE_DIR / f"synthetic_projected_shell_random_points_nopixel_n{n_points}"


def load_shell_points(n_points: int, widths: np.ndarray):
    """(y, z, v) raw arrays and r_min, one tuple per width, for an existing run."""
    d = run_dir(n_points)
    points, r_min = [], []
    for width in widths:
        r_inner = R_OUTER_FIXED - float(width)
        path = shell_output_path(d, r_inner, R_OUTER_FIXED)
        with np.load(path) as data:
            r_min.append(float(data["r_min"]))
            points.append((
                np.asarray(data["y"], dtype=float),
                np.asarray(data["z"], dtype=float),
                np.asarray(data["projected_v"], dtype=float),
            ))
    return points, r_min


def pixelize_points(y: np.ndarray, z: np.ndarray, v: np.ndarray, pixel_size: float):
    """Average every raw point landing in the same (y, z) grid cell into one.

    Grid is anchored at (0, 0) with square cells of side `pixel_size`. This
    is the same averaging a real telescope beam/fiber does, and the same
    operation project_random_points() in generate_random_point_expanding_
    shell_grid.py performs on a fixed box grid -- here parameterized
    directly by physical pixel size instead of a grid cell count, so it can
    be swept continuously like RMIN_SWEEP.
    """
    iy = np.floor(y / pixel_size).astype(np.int64)
    iz = np.floor(z / pixel_size).astype(np.int64)
    cell_ids = np.stack([iy, iz], axis=1)
    uniq_cells, inverse = np.unique(cell_ids, axis=0, return_inverse=True)
    n_cells = uniq_cells.shape[0]
    sums = np.zeros(n_cells)
    counts = np.zeros(n_cells)
    np.add.at(sums, inverse, v)
    np.add.at(counts, inverse, 1.0)
    v_mean = sums / counts
    y_cell = (uniq_cells[:, 0] + 0.5) * pixel_size
    z_cell = (uniq_cells[:, 1] + 0.5) * pixel_size
    return y_cell, z_cell, v_mean


def slope_for_width(y: np.ndarray, z: np.ndarray, v: np.ndarray, pixel_size: float) -> tuple[float, bool]:
    y_pix, z_pix, v_pix = pixelize_points(y, z, v, pixel_size)
    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y_pix, z_pix, v_pix, bin_resolution=BIN_RESOLUTION, min_separation=pixel_size
    )
    # Forced serial -- see plot_fixed_rmin_comparison.py for why (loky hang
    # in backgrounded/piped shells). Pixelization already collapses the
    # point count well below the 10,000/chunk subsample cap for any of the
    # pixel sizes swept here, so serial execution is still fast.
    n_jobs = 1
    vel_err = np.zeros_like(v_pix)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y_pix, z_pix, v_pix, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=n_jobs,
        )
    kk = kurtosis_from_pdf(pdf, dv_centers)
    record = {"r": 10**log_centers, "kk": kk}
    fit = fit_k_vs_logr_above_floor(record, floor=KURTOSIS_FLOOR)
    return fit["slope"], bool(fit["used_zero_fallback"])


def load_or_compute_left() -> tuple[dict, dict]:
    if LEFT_CACHE.exists():
        with np.load(LEFT_CACHE) as c:
            if (
                list(c["n_values"]) == N_VALUES
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
            ):
                print(f"Left panel: loading cached slopes from {LEFT_CACHE.relative_to(REPO_ROOT)}...")
                slopes = {n: c["slopes"][i] for i, n in enumerate(N_VALUES)}
                fallback = {n: c["fallback"][i].astype(bool) for i, n in enumerate(N_VALUES)}
                return slopes, fallback
        print("Left panel: cache doesn't match current N_VALUES/WIDTHS/BIN_RESOLUTION -- recomputing...")

    print("Left panel: vary N, pixel size fixed to N=5,000's per-width r_min...")
    ref_points, ref_r_min = load_shell_points(5000, WIDTHS)
    slopes = {n: [] for n in N_VALUES}
    fallback = {n: [] for n in N_VALUES}
    for n in N_VALUES:
        points = ref_points if n == 5000 else load_shell_points(n, WIDTHS)[0]
        for i in range(len(WIDTHS)):
            y, z, v = points[i]
            s, fb = slope_for_width(y, z, v, pixel_size=ref_r_min[i])
            slopes[n].append(s)
            fallback[n].append(fb)
        print(f"  N={n:,}: {['%.3f' % s for s in slopes[n][:5]]} ...")

    np.savez_compressed(
        LEFT_CACHE,
        n_values=np.array(N_VALUES),
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        slopes=np.array([slopes[n] for n in N_VALUES]),
        fallback=np.array([fallback[n] for n in N_VALUES]),
    )
    print(f"Left panel: cached to {LEFT_CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback


def load_or_compute_right() -> tuple[dict, dict]:
    if RIGHT_CACHE.exists():
        with np.load(RIGHT_CACHE) as c:
            if (
                np.allclose(c["pixel_sweep"], PIXEL_SWEEP)
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
            ):
                print(f"Middle panel: loading cached slopes from {RIGHT_CACHE.relative_to(REPO_ROOT)}...")
                slopes = {float(r): c["slopes"][i] for i, r in enumerate(c["pixel_sweep"])}
                fallback = {float(r): c["fallback"][i].astype(bool) for i, r in enumerate(c["pixel_sweep"])}
                return slopes, fallback
        print("Middle panel: cache doesn't match current PIXEL_SWEEP/WIDTHS/BIN_RESOLUTION -- recomputing...")

    print("Middle panel: fixed N=100,000, pixel size swept explicitly across PIXEL_SWEEP...")
    points_100k, _ = load_shell_points(100000, WIDTHS)
    slopes = {}
    fallback = {}
    for pixel_size in PIXEL_SWEEP:
        s_list, fb_list = [], []
        for i in range(len(WIDTHS)):
            y, z, v = points_100k[i]
            s, fb = slope_for_width(y, z, v, pixel_size=float(pixel_size))
            s_list.append(s)
            fb_list.append(fb)
        slopes[float(pixel_size)] = s_list
        fallback[float(pixel_size)] = fb_list
        print(f"  pixel_size={pixel_size:.4f}: {['%.3f' % s for s in s_list[:5]]} ...")

    np.savez_compressed(
        RIGHT_CACHE,
        pixel_sweep=PIXEL_SWEEP,
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        slopes=np.array([slopes[float(p)] for p in PIXEL_SWEEP]),
        fallback=np.array([fallback[float(p)] for p in PIXEL_SWEEP]),
    )
    print(f"Middle panel: cached to {RIGHT_CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback


left_slopes, left_fallback = load_or_compute_left()
right_slopes, right_fallback = load_or_compute_right()

style_path = REPO_ROOT / "custom_plot.mplstyle"
try:
    plt.style.use(str(style_path))
except OSError:
    pass

cmap = plt.colormaps["Blues"]
fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(21.0, 6.0))

# --- Left: vary N (shared pixel size) ---
for i, n in enumerate(N_VALUES):
    shade = 0.3 + 0.6 * (i / (len(N_VALUES) - 1))
    axL.plot(WIDTHS, left_slopes[n], "o--", ms=5.0, lw=1.3, color=cmap(shade), label=f"N={n:,}")

left_matrix = np.array([left_slopes[n] for n in N_VALUES])
left_fallback_matrix = np.array([left_fallback[n] for n in N_VALUES])
log_widths = np.log10(WIDTHS)

spread_lo = np.nanmin(left_matrix, axis=0)
spread_hi = np.nanmax(left_matrix, axis=0)
axL.fill_between(WIDTHS, spread_lo, spread_hi, color="0.5", alpha=0.3, zorder=2, label=r"spread across $N$")

axL.axvline(0.2, color="0.25", ls=":", lw=1.1)
axL.axhline(0, color="0.25", ls="--", lw=1.1)
axL.set_xscale("log")
axL.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axL.grid(True, axis="y", ls="--", alpha=0.4)
axL.legend(fontsize=9)
axL.set_ylabel(r"$dK/d\log_{10} r$")

# --- Middle: fixed N=100,000, vary pixel size -- raw points joined by lines ---
ratios = PIXEL_SWEEP / R_OUTER_FIXED
norm = matplotlib.colors.LogNorm(vmin=ratios.min(), vmax=ratios.max())
right_cmap = plt.colormaps["viridis"]

min_slope = np.full(len(PIXEL_SWEEP), np.nan)
for j, pixel_size in enumerate(PIXEL_SWEEP):
    color = right_cmap(norm(pixel_size / R_OUTER_FIXED))
    y_arr = np.asarray(right_slopes[float(pixel_size)])
    fb_arr = np.asarray(right_fallback[float(pixel_size)])
    axR.plot(WIDTHS, y_arr, "o-", ms=5.0, lw=1.5, color=color, alpha=0.9, zorder=3)

    if np.any(~fb_arr):
        min_slope[j] = np.min(y_arr[~fb_arr])

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=right_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axR)
cbar.set_label(r"pixel size$/r_{\rm HII}$")

axR.axvline(0.2, color="0.25", ls=":", lw=1.1)
axR.axhline(0, color="0.25", ls="--", lw=1.1)
axR.set_xscale("log")
axR.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axR.grid(True, axis="y", ls="--", alpha=0.4)
axR.set_ylim(axL.get_ylim())

# --- Right: deepest (most negative) raw slope reached by each pixel-size curve,
# vs pixel_size/r_HII -- no fit, just the data points joined by a line ---
good = np.isfinite(min_slope)
abs_min_slope = np.abs(min_slope)
axS.plot(
    ratios[good], abs_min_slope[good],
    "o-", ms=6.0, lw=1.5, color="#31688e", zorder=3,
)
axS.set_xscale("log")
axS.set_yscale("log")
ylim = (0.1, 10.0)
for j in np.flatnonzero(~good):
    axS.scatter(ratios[j], ylim[0], marker="x", s=60, color="0.5", zorder=5, clip_on=False)
axS.set_ylim(ylim)
axS.set_xlabel(r"pixel size$/r_{\rm HII}$")
axS.set_ylabel(r"$|{\rm deepest\ raw}|$ $dK/d\log_{10}r$")
axS.grid(True, which="both", axis="y", ls="--", alpha=0.4)

fig.tight_layout()

out = BASE_DIR / "random_point_count_dK_dlogr_fixed_pixelation_comparison.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print(f"\nWrote {out.relative_to(REPO_ROOT)}")
