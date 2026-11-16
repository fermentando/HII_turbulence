#!/usr/bin/env python3
"""Three-panel diagnostic for the N/r_min-dependent dK/dlogr artifact.

Companion to plot_fixed_rmin_comparison.py and plot_fixed_pixelation_
comparison.py. Those scripts isolated the artifact by holding one knob
fixed while sweeping the other (fixed N while sweeping r_min/pixel size, or
fixed r_min/pixel size while sweeping N). This script instead shows the
ORIGINAL, un-isolated effect: each N's curve computed at its own NATURAL
r_min (no floor imposed, no pixelization) -- i.e. the raw, uncontrolled
N-sweep -- color-coded by the minimum r_min that N's curve actually reached
across the width sweep. Since r_min shrinks with N roughly as 1/sqrt(N),
this panel shows directly that "varying N" and "varying r_min" are, at
natural resolution, the same knob.

Left panel: same as plot_fixed_rmin_comparison.py's left panel -- the
N-sweep with r_min FLOORED to N=5,000's own per-width value, isolating
genuine N-dependence from the r_min artifact. Reuses that script's cache
file directly (identical computation, so no recomputation needed).

Middle panel: ten N curves (50/100/250/500/1,000/2,500/5,000/20,000/50,000/
100,000), each at its own NATURAL r_min (no floor), colored by a continuous
colorbar keyed to the minimum r_min/r_HII that N's curve reached (instead
of a discrete N legend) -- showing that lower-r_min curves are the ones
with the deeper, steeper dK/dlogr trend, regardless of which N produced
that r_min. The N range extends down to N=50 specifically so the natural
r_min reaches ~0.1 r_HII (r_min ~ 1/sqrt(N)), matching the top of the
RMIN_SWEEP/PIXEL_SWEEP range used in the other two comparison scripts.

Right panel: the deepest (most negative) raw dK/dlogr value reached by each
N's curve, plotted against that N's minimum r_min/r_HII, log-log, points
joined by a line -- no curve fitting anywhere in this figure.

Caching: the left panel reuses plot_fixed_rmin_comparison.py's own cache
(_fixed_rmin_left_cache.npz, read-only, never written here). The middle/
right panels use their own new cache (_fixed_n_natural_right_cache.npz).
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
N_VALUES = [5000, 20000, 50000, 100000]  # left panel only -- matches plot_fixed_rmin_comparison.py's cache
# Middle/right panels only: extended down to N=50 so natural r_min reaches
# ~0.1 r_HII (r_min ~ 1/sqrt(N), so matching the top of RMIN_SWEEP/
# PIXEL_SWEEP's range needs N two orders of magnitude below N=5,000).
N_VALUES_NATURAL = [50, 100, 250, 500, 1000, 2500, 5000, 20000, 50000, 100000]
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20

LEFT_CACHE = BASE_DIR / "_fixed_rmin_left_cache.npz"  # shared read-only with plot_fixed_rmin_comparison.py
RIGHT_CACHE = BASE_DIR / "_fixed_n_natural_right_cache.npz"


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


def slope_for_width(y: np.ndarray, z: np.ndarray, v: np.ndarray, min_separation: float | None) -> tuple[float, bool, float]:
    """Also returns the actual r_min compute_bins ended up using (natural
    nearest-neighbor spacing when min_separation=None, or the floored value
    otherwise) -- spatial_bin_edges computes this internally regardless, so
    compute_bins hands it back via its bins_dict rather than recomputing it.
    """
    log_centers, bin_edges, dv_bins, bins_dict = compute_bins(
        y, z, v, bin_resolution=BIN_RESOLUTION, min_separation=min_separation
    )
    # Forced serial -- see plot_fixed_rmin_comparison.py for why (loky hang
    # in backgrounded/piped shells).
    n_jobs = 1
    vel_err = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y, z, v, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=n_jobs,
        )
    kk = kurtosis_from_pdf(pdf, dv_centers)
    record = {"r": 10**log_centers, "kk": kk}
    fit = fit_k_vs_logr_above_floor(record, floor=KURTOSIS_FLOOR)
    return fit["slope"], bool(fit["used_zero_fallback"]), float(bins_dict["r_min"])


def load_or_compute_left() -> tuple[dict, dict]:
    if not LEFT_CACHE.exists():
        raise FileNotFoundError(
            f"{LEFT_CACHE} not found -- run plot_fixed_rmin_comparison.py first "
            "to generate it (this script reuses that cache read-only)."
        )
    with np.load(LEFT_CACHE) as c:
        if not (
            list(c["n_values"]) == N_VALUES
            and np.allclose(c["widths"], WIDTHS)
            and int(c["bin_resolution"]) == BIN_RESOLUTION
        ):
            raise ValueError(
                f"{LEFT_CACHE} doesn't match current N_VALUES/WIDTHS/BIN_RESOLUTION -- "
                "rerun plot_fixed_rmin_comparison.py to refresh it."
            )
        print(f"Left panel: loading cached slopes from {LEFT_CACHE.relative_to(REPO_ROOT)} (shared with plot_fixed_rmin_comparison.py)...")
        slopes = {n: c["slopes"][i] for i, n in enumerate(N_VALUES)}
        fallback = {n: c["fallback"][i].astype(bool) for i, n in enumerate(N_VALUES)}
        return slopes, fallback


def load_or_compute_right() -> tuple[dict, dict, dict]:
    if RIGHT_CACHE.exists():
        with np.load(RIGHT_CACHE) as c:
            if (
                list(c["n_values"]) == N_VALUES_NATURAL
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
            ):
                print(f"Middle panel: loading cached slopes from {RIGHT_CACHE.relative_to(REPO_ROOT)}...")
                slopes = {n: c["slopes"][i] for i, n in enumerate(N_VALUES_NATURAL)}
                fallback = {n: c["fallback"][i].astype(bool) for i, n in enumerate(N_VALUES_NATURAL)}
                r_min_used = {n: c["r_min_used"][i] for i, n in enumerate(N_VALUES_NATURAL)}
                return slopes, fallback, r_min_used
        print("Middle panel: cache doesn't match current N_VALUES_NATURAL/WIDTHS/BIN_RESOLUTION -- recomputing...")

    print("Middle panel: each N at its own natural r_min (no floor)...")
    slopes = {n: [] for n in N_VALUES_NATURAL}
    fallback = {n: [] for n in N_VALUES_NATURAL}
    r_min_used = {n: [] for n in N_VALUES_NATURAL}
    for n in N_VALUES_NATURAL:
        points, _ = load_shell_points(n, WIDTHS)
        for i in range(len(WIDTHS)):
            y, z, v = points[i]
            s, fb, r_min = slope_for_width(y, z, v, min_separation=None)
            slopes[n].append(s)
            fallback[n].append(fb)
            r_min_used[n].append(r_min)
        print(f"  N={n:,}: slopes={['%.3f' % s for s in slopes[n][:5]]} ... min(r_min)={min(r_min_used[n]):.5f}")

    np.savez_compressed(
        RIGHT_CACHE,
        n_values=np.array(N_VALUES_NATURAL),
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        slopes=np.array([slopes[n] for n in N_VALUES_NATURAL]),
        fallback=np.array([fallback[n] for n in N_VALUES_NATURAL]),
        r_min_used=np.array([r_min_used[n] for n in N_VALUES_NATURAL]),
    )
    print(f"Middle panel: cached to {RIGHT_CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback, r_min_used


left_slopes, left_fallback = load_or_compute_left()
right_slopes, right_fallback, right_r_min_used = load_or_compute_right()

style_path = REPO_ROOT / "custom_plot.mplstyle"
try:
    plt.style.use(str(style_path))
except OSError:
    pass

cmap = plt.colormaps["Blues"]
fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(21.0, 6.0))

# --- Left: vary N, r_min floored to N=5,000's per-width value (reused cache) ---
for i, n in enumerate(N_VALUES):
    shade = 0.3 + 0.6 * (i / (len(N_VALUES) - 1))
    axL.plot(WIDTHS, left_slopes[n], "o--", ms=5.0, lw=1.3, color=cmap(shade), label=f"N={n:,}")

left_matrix = np.array([left_slopes[n] for n in N_VALUES])
left_fallback_matrix = np.array([left_fallback[n] for n in N_VALUES])

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

# --- Middle: vary N at NATURAL r_min, colored by that N's min(r_min)/r_HII ---
min_ratio_per_n = {n: min(right_r_min_used[n]) / R_OUTER_FIXED for n in N_VALUES_NATURAL}
all_ratios = np.array([min_ratio_per_n[n] for n in N_VALUES_NATURAL])
norm = matplotlib.colors.LogNorm(vmin=all_ratios.min(), vmax=all_ratios.max())
right_cmap = plt.colormaps["viridis"]

deepest_slope = np.full(len(N_VALUES_NATURAL), np.nan)
for j, n in enumerate(N_VALUES_NATURAL):
    color = right_cmap(norm(min_ratio_per_n[n]))
    y_arr = np.asarray(right_slopes[n])
    fb_arr = np.asarray(right_fallback[n])
    axR.plot(WIDTHS, y_arr, "o-", ms=5.0, lw=1.5, color=color, alpha=0.9, zorder=3)

    if np.any(~fb_arr):
        deepest_slope[j] = np.min(y_arr[~fb_arr])

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=right_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axR)
cbar.set_label(r"min$(r_{\rm min})/r_{\rm HII}$")

axR.axvline(0.2, color="0.25", ls=":", lw=1.1)
axR.axhline(0, color="0.25", ls="--", lw=1.1)
axR.set_xscale("log")
axR.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axR.grid(True, axis="y", ls="--", alpha=0.4)
axR.set_ylim(axL.get_ylim())

# --- Right: deepest (most negative) raw slope reached by each N's curve,
# vs that N's minimum r_min/r_HII -- no fit, just the data points joined by a line ---
good = np.isfinite(deepest_slope)
abs_deepest_slope = np.abs(deepest_slope)
order = np.argsort(all_ratios[good])
axS.plot(
    all_ratios[good][order], abs_deepest_slope[good][order],
    "o-", ms=6.0, lw=1.5, color="#31688e", zorder=3,
)
axS.set_xscale("log")
axS.set_yscale("log")
ylim = (0.1, 10.0)
for j in np.flatnonzero(~good):
    axS.scatter(all_ratios[j], ylim[0], marker="x", s=60, color="0.5", zorder=5, clip_on=False)
axS.set_ylim(ylim)
axS.set_xlabel(r"min$(r_{\rm min})/r_{\rm HII}$")
axS.set_ylabel(r"$|{\rm deepest\ raw}|$ $dK/d\log_{10}r$")
axS.grid(True, which="both", axis="y", ls="--", alpha=0.4)

fig.tight_layout()

out = BASE_DIR / "random_point_count_dK_dlogr_fixed_n_natural_rmin_comparison.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print(f"\nWrote {out.relative_to(REPO_ROOT)}")
