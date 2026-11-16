#!/usr/bin/env python3
"""Three-panel dK/dlogr diagnostic with NO BULK MOTION (v_bulk=0, turbulence only).

Identical to plot_fixed_rmin_comparison.py (same panels, pair-kurtosis
estimator, floor-fixed logistic fits) but run on the raw-point datasets
generated with --v-bulk 0.0 (synthetic_projected_shell_random_points_nopixel_nobulk_n*).
Uses its own caches (_fixed_rmin_nobulk_{left,right}_cache.npz) and output
image, so it never touches the v_bulk=10 version's cache or figure.
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
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402
from generate_shell_thickness_grid import width_grid, shell_output_path, R_OUTER_FIXED  # noqa: E402
from plot_shell_dk_dlogr_aperture_comparison import fit_k_vs_logr_above_floor  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
N_VALUES = [5000, 20000, 50000, 100000]
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
RMIN_SWEEP = np.geomspace(0.001, 0.1, 6)
MIN_FIT_POINTS_FOR_LINE = 3  # don't trust a line fit through fewer non-fallback widths
# Logistic floor is FREE here (unlike the v_bulk=10 script's fixed -2): with no
# bulk flow the slopes never get below ~-0.6, so a -2 floor is meaningless.
FIT_BOUNDS = ([-5.0, -4.0, 0.02], [0.0, 2.0, 5.0])  # (floor, x0, scale)

LEFT_CACHE = BASE_DIR / "_fixed_rmin_nobulk_left_cache.npz"
RIGHT_CACHE = BASE_DIR / "_fixed_rmin_nobulk_right_cache.npz"


def run_dir(n_points: int) -> Path:
    return BASE_DIR / f"synthetic_projected_shell_random_points_nopixel_nobulk_n{n_points}"


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


def slope_for_width(y: np.ndarray, z: np.ndarray, v: np.ndarray, min_separation: float) -> tuple[float, bool]:
    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y, z, v, bin_resolution=BIN_RESOLUTION, min_separation=min_separation
    )
    # Forced serial (not an auto-scaled n_jobs): joblib's default loky
    # (process-based) backend was hanging indefinitely trying to spawn
    # worker processes in a backgrounded/piped shell -- a known failure mode
    # in sandboxed/non-TTY environments. Points are capped at 10,000/chunk
    # either way, so serial execution is still fast per width.
    n_jobs = 1
    vel_err = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y, z, v, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=n_jobs,
        )
    kk = kk_pairs
    record = {"r": 10**log_centers, "kk": kk}
    fit = fit_k_vs_logr_above_floor(record, floor=KURTOSIS_FLOOR)
    return fit["slope"], bool(fit["used_zero_fallback"])


def logistic(log_w: np.ndarray, floor: float, x0: float, scale: float) -> np.ndarray:
    """Sigmoid rising from `floor` (small width) to 0 (large width); floor is fitted."""
    return floor / (1.0 + np.exp((log_w - x0) / scale))


def logistic_fit_with_band(log_w: np.ndarray, y: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Free-floor logistic fit over masked points, + MC-propagated 1-sigma bands.

    Returns (dense_x, dense_y, band_lo, band_hi, floor, floor_err, x0, x0_err,
    scale, scale_err) -- None if there aren't enough unmasked points to fit, or
    the fit doesn't converge.
    """
    n_valid = int(np.count_nonzero(mask))
    if n_valid < MIN_FIT_POINTS_FOR_LINE:
        return None
    try:
        params, cov = curve_fit(
            logistic,
            log_w[mask],
            y[mask],
            p0=[min(float(np.min(y[mask])), -0.05), np.median(log_w[mask]), 0.3],
            bounds=FIT_BOUNDS,
            maxfev=20000,
        )
    except RuntimeError:
        return None
    dense = np.linspace(log_w[mask].min(), log_w[mask].max(), 200)
    curve = logistic(dense, *params)
    has_cov = np.all(np.isfinite(cov))
    if has_cov:
        draws = rng.multivariate_normal(params, cov, size=1000)
        draw_curves = np.array([logistic(dense, *p) for p in draws])
        band_lo = np.percentile(draw_curves, 16, axis=0)
        band_hi = np.percentile(draw_curves, 84, axis=0)
    else:
        band_lo = band_hi = None
    errs = np.sqrt(np.diag(cov)) if has_cov else np.full(3, np.nan)
    return (
        dense, curve, band_lo, band_hi,
        float(params[0]), float(errs[0]),
        float(params[1]), float(errs[1]),
        float(params[2]), float(errs[2]),
    )


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

    print("Left panel: vary N, r_min floored to N=5,000's per-width value...")
    ref_points, ref_r_min = load_shell_points(5000, WIDTHS)
    slopes = {n: [] for n in N_VALUES}
    fallback = {n: [] for n in N_VALUES}
    for n in N_VALUES:
        points = ref_points if n == 5000 else load_shell_points(n, WIDTHS)[0]
        for i in range(len(WIDTHS)):
            y, z, v = points[i]
            s, fb = slope_for_width(y, z, v, min_separation=ref_r_min[i])
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
                np.allclose(c["rmin_sweep"], RMIN_SWEEP)
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
            ):
                print(f"Middle panel: loading cached slopes from {RIGHT_CACHE.relative_to(REPO_ROOT)}...")
                slopes = {float(r): c["slopes"][i] for i, r in enumerate(c["rmin_sweep"])}
                if "fallback" in c:
                    fallback = {float(r): c["fallback"][i].astype(bool) for i, r in enumerate(c["rmin_sweep"])}
                else:
                    # Older cache, computed before fallback tracking was added. The
                    # fallback sentinel in fit_k_vs_logr_above_floor is an exact
                    # 0.0 (a genuine fit landing there is practically impossible
                    # with real float data), so it can be recovered from the
                    # slopes alone -- no need to recompute just to backfill this.
                    print("  (no 'fallback' field in this cache; inferring it from slope==0.0 instead of recomputing)")
                    fallback = {float(r): (np.asarray(c["slopes"][i]) == 0.0) for i, r in enumerate(c["rmin_sweep"])}
                return slopes, fallback
        print("Middle panel: cache doesn't match current RMIN_SWEEP/WIDTHS/BIN_RESOLUTION -- recomputing...")

    print("Middle panel: fixed N=100,000, r_min swept explicitly across RMIN_SWEEP...")
    points_100k, _ = load_shell_points(100000, WIDTHS)
    slopes = {}
    fallback = {}
    for floor in RMIN_SWEEP:
        s_list, fb_list = [], []
        for i in range(len(WIDTHS)):
            y, z, v = points_100k[i]
            s, fb = slope_for_width(y, z, v, min_separation=float(floor))
            s_list.append(s)
            fb_list.append(fb)
        slopes[float(floor)] = s_list
        fallback[float(floor)] = fb_list
        print(f"  r_min={floor:.4f}: {['%.3f' % s for s in s_list[:5]]} ...")

    np.savez_compressed(
        RIGHT_CACHE,
        rmin_sweep=RMIN_SWEEP,
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        slopes=np.array([slopes[float(r)] for r in RMIN_SWEEP]),
        fallback=np.array([fallback[float(r)] for r in RMIN_SWEEP]),
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

rng = np.random.default_rng(0)
cmap = plt.colormaps["Blues"]
fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(21.0, 6.0))

# --- Left: vary N (shared r_min), straight-line fit + propagated 1-sigma band ---
for i, n in enumerate(N_VALUES):
    shade = 0.3 + 0.6 * (i / (len(N_VALUES) - 1))
    axL.plot(WIDTHS, left_slopes[n], "o--", ms=5.0, lw=1.3, color=cmap(shade), label=f"N={n:,}")

left_matrix = np.array([left_slopes[n] for n in N_VALUES])
left_fallback_matrix = np.array([left_fallback[n] for n in N_VALUES])
mean_slope = np.nanmean(left_matrix, axis=0)
left_has_fit = ~np.all(left_fallback_matrix, axis=0)
log_widths = np.log10(WIDTHS)

left_fit = logistic_fit_with_band(log_widths, mean_slope, left_has_fit, rng)
# Spread across the N curves themselves (min/max at each width), rather than
# the fit's own propagated-covariance band -- requested to be wider, to show
# how much the individual N lines actually disagree, not just the fit's
# formal uncertainty on the mean.
spread_lo = np.nanmin(left_matrix, axis=0)
spread_hi = np.nanmax(left_matrix, axis=0)
axL.fill_between(WIDTHS, spread_lo, spread_hi, color="0.5", alpha=0.3, zorder=2, label=r"spread across $N$")
if left_fit is not None:
    dense, curve, band_lo, band_hi, floor, floor_err, x0, x0_err, scale, scale_err = left_fit
    axL.plot(10**dense, curve, "-", color="crimson", lw=2.0, label=fr"logistic fit (floor$={floor:.2f}\pm{floor_err:.2f}$)", zorder=5)
    print(f"Left fit: floor={floor:.3f}+/-{floor_err:.3f}, x0={x0:.3f}+/-{x0_err:.3f}, scale={scale:.3f}+/-{scale_err:.3f}")

axL.axvline(0.2, color="0.25", ls=":", lw=1.1)
axL.axhline(0, color="0.25", ls="--", lw=1.1)
axL.set_xscale("log")
axL.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axL.grid(True, axis="y", ls="--", alpha=0.4)
axL.legend(fontsize=9)
axL.set_ylabel(r"$dK/d\log_{10} r$")

# --- Middle: fixed N=100,000, vary r_min -- per-r_min line fits, colored by r_min/r_HII ---
ratios = RMIN_SWEEP / R_OUTER_FIXED
norm = matplotlib.colors.LogNorm(vmin=ratios.min(), vmax=ratios.max())
right_cmap = plt.colormaps["viridis"]

fit_scale = np.full(len(RMIN_SWEEP), np.nan)
fit_scale_err = np.full(len(RMIN_SWEEP), np.nan)
fit_x0 = np.full(len(RMIN_SWEEP), np.nan)
fit_x0_err = np.full(len(RMIN_SWEEP), np.nan)
for j, rmin_val in enumerate(RMIN_SWEEP):
    color = right_cmap(norm(rmin_val / R_OUTER_FIXED))
    y_arr = np.asarray(right_slopes[float(rmin_val)])
    fb_arr = np.asarray(right_fallback[float(rmin_val)])
    axR.plot(WIDTHS, y_arr, "o", ms=5.0, color=color, alpha=0.9, zorder=3)

    fit = logistic_fit_with_band(log_widths, y_arr, ~fb_arr, rng)
    if fit is not None:
        dense, curve, band_lo, band_hi, floor, floor_err, x0, x0_err, scale, scale_err = fit
        axR.plot(10**dense, curve, "-", lw=1.8, color=color, zorder=4)
        print(
            f"r_min={rmin_val:.4f}: floor={floor:.3f}+/-{floor_err:.3f}, "
            f"x0={x0:.3f}+/-{x0_err:.3f}, scale={scale:.3f}+/-{scale_err:.3f}"
        )
        fit_scale[j] = scale
        fit_scale_err[j] = scale_err
        fit_x0[j] = x0
        fit_x0_err[j] = x0_err

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=right_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axR)
cbar.set_label(r"$r_{\rm min}/r_{\rm HII}$")

axR.axvline(0.2, color="0.25", ls=":", lw=1.1)
axR.axhline(0, color="0.25", ls="--", lw=1.1)
axR.set_xscale("log")
axR.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axR.grid(True, axis="y", ls="--", alpha=0.4)
axR.set_ylim(axL.get_ylim())

# --- Right (new): fitted logistic `scale` (left axis) and `x0` (right axis)
# vs r_min/r_HII, both from the free-floor fits -- scale sets how steep/wide the
# logistic transition is, x0 sets where (in log-width) it's centered;
# plotting both shows whether r_min moves the transition's location, its
# steepness, or both.
good = np.isfinite(fit_scale)
p1 = axS.errorbar(
    ratios[good], fit_scale[good], yerr=fit_scale_err[good],
    fmt="o", ms=6.0, lw=1.5, capsize=3, color="#31688e", zorder=3, label=r"scale (left axis)",
)
if np.count_nonzero(good) >= 2:
    log_ratio = np.log10(ratios[good])
    log_scale = np.log10(fit_scale[good])
    power, intercept = np.polyfit(log_ratio, log_scale, 1)
    dense_ratio = np.geomspace(ratios[good].min(), ratios[good].max(), 100)
    scale_fit_line, = axS.plot(
        dense_ratio, 10 ** (intercept + power * np.log10(dense_ratio)),
        "-", lw=1.5, color="#31688e", alpha=0.6, zorder=2,
        label=fr"scale $\propto (r_{{\rm min}}/r_{{\rm HII}})^{{{power:.2f}}}$",
    )
axS.set_xscale("log")
axS.set_yscale("log")
ylim = axS.get_ylim()
for j in np.flatnonzero(~good):
    axS.scatter(ratios[j], ylim[0], marker="x", s=60, color="0.5", zorder=5, clip_on=False)
axS.set_ylim(ylim)
axS.set_xlabel(r"$r_{\rm min}/r_{\rm HII}$")
axS.set_ylabel(r"logistic growth rate ($r$)", color="#31688e")
axS.tick_params(axis="y", labelcolor="#31688e")
axS.grid(True, which="both", axis="y", ls="--", alpha=0.4)

axS2 = axS.twinx()
good_x0 = np.isfinite(fit_x0)
p2 = axS2.errorbar(
    ratios[good_x0], fit_x0[good_x0], yerr=fit_x0_err[good_x0],
    fmt="s", ms=6.0, lw=1.5, capsize=3, color="#d95f02", zorder=3, label=r"$x_0$ (right axis)",
)
if np.count_nonzero(good_x0) >= 2:
    log_ratio_x0 = np.log10(ratios[good_x0])
    slope_x0, intercept_x0 = np.polyfit(log_ratio_x0, fit_x0[good_x0], 1)
    dense_ratio_x0 = np.geomspace(ratios[good_x0].min(), ratios[good_x0].max(), 100)
    x0_fit_line, = axS2.plot(
        dense_ratio_x0, intercept_x0 + slope_x0 * np.log10(dense_ratio_x0),
        "--", lw=1.5, color="#d95f02", alpha=0.6, zorder=2,
        label=fr"$x_0 \propto {slope_x0:.2f}\,\log_{{10}}(r_{{\rm min}}/r_{{\rm HII}})$",
    )
axS2.set_ylabel(r"logistic $x_0$ ($\log_{10}\Delta r_{\rm shell}/r_{\rm HII}$)", color="#d95f02")
axS2.tick_params(axis="y", labelcolor="#d95f02")

lines = [p1, scale_fit_line, p2, x0_fit_line]
axS.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="lower center")

fig.suptitle(r"$v_{\rm bulk}=0$ (turbulence only)", fontsize=13)
fig.tight_layout()

out = BASE_DIR / "random_point_count_dK_dlogr_fixed_rmin_comparison_nobulk.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print(f"\nWrote {out.relative_to(REPO_ROOT)}")
