#!/usr/bin/env python3
"""How does v_bulk/sigma_turb control the shape of dK/dlogr vs shell width?

Follow-up to the v_bulk=0 test in rmin_projection_artifact_explanation.md
("Isolating the source: a v_bulk=0 test"), which showed that removing bulk
motion only partially suppresses the near/far-side projection-contamination
artifact (floor drops from ~-2.0 to ~-1.4, not to ~0) -- meaning turbulent
decorrelation across the shell's 3D depth, not bulk flow alone, sources
most of the effect. This script sweeps v_bulk continuously (at fixed
sigma_turb=1, N=100,000, natural r_min) to see how the artifact's shape
evolves between the no-bulk and strong-bulk limits, rather than just
comparing the two endpoints.

Left panel: dK/dlogr vs shell width for each v_bulk/sigma_turb in
VBULK_SWEEP, colored by a continuous v_bulk/sigma_turb colorbar, with a
free 3-parameter logistic fit (floor, x0, scale all free -- unlike the
r_min sweep, here each curve is a physically distinct dataset rather than
a reparametrization of the same one, so there's no reason to expect a
shared floor and no corresponding degeneracy to fix).

Right panel: the fitted floor, x0, and scale plotted against
v_bulk/sigma_turb, showing which parameter(s) the bulk-to-turbulence ratio
actually controls.

Caching: own cache file (_vbulk_sweep_cache.npz), separate from every other
comparison script's cache.
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
from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from generate_shell_thickness_grid import width_grid, shell_output_path, R_OUTER_FIXED  # noqa: E402
from plot_shell_dk_dlogr_aperture_comparison import fit_k_vs_logr_above_floor  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
N_POINTS = 100_000
SIGMA_TURB = 1.0  # fixed across the sweep (random_point_expanding_shell_metadata.json)
VBULK_SWEEP = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
MIN_FIT_POINTS_FOR_LINE = 3

CACHE = BASE_DIR / "_vbulk_sweep_cache.npz"

# v_bulk=0 and v_bulk=10 reuse the directories generated for the earlier
# no-bulk test and the original production runs; the rest were generated
# specifically for this sweep.
_DIR_OVERRIDE = {
    0.0: "synthetic_projected_shell_random_points_nopixel_nobulk_n100000",
    10.0: "synthetic_projected_shell_random_points_nopixel_n100000",
}


def run_dir(v_bulk: float) -> Path:
    name = _DIR_OVERRIDE.get(v_bulk, f"synthetic_projected_shell_random_points_nopixel_vbulk{v_bulk:g}_n100000")
    return BASE_DIR / name


def load_shell_points(v_bulk: float, widths: np.ndarray):
    d = run_dir(v_bulk)
    points = []
    for width in widths:
        r_inner = R_OUTER_FIXED - float(width)
        path = shell_output_path(d, r_inner, R_OUTER_FIXED)
        with np.load(path) as data:
            points.append((
                np.asarray(data["y"], dtype=float),
                np.asarray(data["z"], dtype=float),
                np.asarray(data["projected_v"], dtype=float),
            ))
    return points


def slope_for_width(y: np.ndarray, z: np.ndarray, v: np.ndarray) -> tuple[float, bool]:
    # Natural r_min (no floor): N is fixed across the whole sweep, and point
    # positions are drawn independently of the velocity field, so the
    # natural r_min is essentially identical for every v_bulk anyway.
    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y, z, v, bin_resolution=BIN_RESOLUTION, min_separation=None
    )
    n_jobs = 1  # forced serial -- see plot_fixed_rmin_comparison.py for why
    vel_err = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y, z, v, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=n_jobs,
        )
    kk = kurtosis_from_pdf(pdf, dv_centers)
    record = {"r": 10**log_centers, "kk": kk}
    fit = fit_k_vs_logr_above_floor(record, floor=KURTOSIS_FLOOR)
    return fit["slope"], bool(fit["used_zero_fallback"])


def logistic(log_w: np.ndarray, floor: float, x0: float, scale: float) -> np.ndarray:
    return floor / (1.0 + np.exp((log_w - x0) / scale))


def logistic_fit(log_w: np.ndarray, y: np.ndarray, mask: np.ndarray):
    n_valid = int(np.count_nonzero(mask))
    if n_valid < MIN_FIT_POINTS_FOR_LINE:
        return None
    try:
        params, cov = curve_fit(
            logistic, log_w[mask], y[mask],
            p0=[y[mask].min(), np.median(log_w[mask]), 0.3], maxfev=5000,
        )
    except RuntimeError:
        return None
    dense = np.linspace(log_w[mask].min(), log_w[mask].max(), 200)
    curve = logistic(dense, *params)
    has_cov = np.all(np.isfinite(cov))
    perr = np.sqrt(np.diag(cov)) if has_cov else [np.nan, np.nan, np.nan]
    return dense, curve, float(params[0]), float(perr[0]), float(params[1]), float(perr[1]), float(params[2]), float(perr[2])


def load_or_compute() -> tuple[dict, dict]:
    if CACHE.exists():
        with np.load(CACHE) as c:
            if (
                np.allclose(c["vbulk_sweep"], VBULK_SWEEP)
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
                and int(c["n_points"]) == N_POINTS
            ):
                print(f"Loading cached slopes from {CACHE.relative_to(REPO_ROOT)}...")
                slopes = {float(v): c["slopes"][i] for i, v in enumerate(c["vbulk_sweep"])}
                fallback = {float(v): c["fallback"][i].astype(bool) for i, v in enumerate(c["vbulk_sweep"])}
                return slopes, fallback
        print("Cache doesn't match current VBULK_SWEEP/WIDTHS/BIN_RESOLUTION/N_POINTS -- recomputing...")

    print(f"Computing: N={N_POINTS:,}, v_bulk swept across {VBULK_SWEEP}...")
    slopes = {}
    fallback = {}
    for v_bulk in VBULK_SWEEP:
        points = load_shell_points(v_bulk, WIDTHS)
        s_list, fb_list = [], []
        for i in range(len(WIDTHS)):
            y, z, v = points[i]
            s, fb = slope_for_width(y, z, v)
            s_list.append(s)
            fb_list.append(fb)
        slopes[float(v_bulk)] = s_list
        fallback[float(v_bulk)] = fb_list
        print(f"  v_bulk={v_bulk:g} (v_bulk/sigma_turb={v_bulk/SIGMA_TURB:g}): {['%.3f' % s for s in s_list[:5]]} ...")

    np.savez_compressed(
        CACHE,
        vbulk_sweep=np.array(VBULK_SWEEP), widths=WIDTHS, bin_resolution=BIN_RESOLUTION, n_points=N_POINTS,
        slopes=np.array([slopes[float(v)] for v in VBULK_SWEEP]),
        fallback=np.array([fallback[float(v)] for v in VBULK_SWEEP]),
    )
    print(f"Cached to {CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback


slopes, fallback = load_or_compute()

style_path = REPO_ROOT / "custom_plot.mplstyle"
try:
    plt.style.use(str(style_path))
except OSError:
    pass

log_widths = np.log10(WIDTHS)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 6.0))

# --- Left: dK/dlogr vs width for each v_bulk/sigma_turb ---
ratios = np.array(VBULK_SWEEP) / SIGMA_TURB
# v_bulk=0 has no well-defined log, so colorbar/norm are built on the
# nonzero values and v_bulk=0 gets a distinct marker/label instead.
nonzero = ratios > 0
norm = matplotlib.colors.LogNorm(vmin=ratios[nonzero].min(), vmax=ratios[nonzero].max())
cmap = plt.colormaps["plasma"]

fit_floor = np.full(len(VBULK_SWEEP), np.nan)
fit_floor_err = np.full(len(VBULK_SWEEP), np.nan)
fit_x0 = np.full(len(VBULK_SWEEP), np.nan)
fit_x0_err = np.full(len(VBULK_SWEEP), np.nan)
fit_scale = np.full(len(VBULK_SWEEP), np.nan)
fit_scale_err = np.full(len(VBULK_SWEEP), np.nan)

for j, v_bulk in enumerate(VBULK_SWEEP):
    ratio = v_bulk / SIGMA_TURB
    color = "0.2" if ratio == 0 else cmap(norm(ratio))
    y_arr = np.asarray(slopes[float(v_bulk)])
    fb_arr = np.asarray(fallback[float(v_bulk)])
    label = r"$v_{\rm bulk}/\sigma_{\rm turb}=0$" if ratio == 0 else None
    axL.plot(WIDTHS, y_arr, "o", ms=5.0, color=color, alpha=0.9, zorder=3, label=label)

    fit = logistic_fit(log_widths, y_arr, ~fb_arr)
    if fit is not None:
        dense, curve, floor, floor_err, x0, x0_err, scale, scale_err = fit
        axL.plot(10**dense, curve, "-", lw=1.8, color=color, zorder=4)
        fit_floor[j], fit_floor_err[j] = floor, floor_err
        fit_x0[j], fit_x0_err[j] = x0, x0_err
        fit_scale[j], fit_scale_err[j] = scale, scale_err

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axL)
cbar.set_label(r"$v_{\rm bulk}/\sigma_{\rm turb}$ (nonzero only; $v_{\rm bulk}=0$ in black)")

axL.axhline(0, color="0.25", ls="--", lw=1.1)
axL.set_xscale("log")
axL.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
axL.set_ylabel(r"$dK/d\log_{10} r$")
axL.grid(True, axis="y", ls="--", alpha=0.4)
axL.legend(fontsize=9, loc="lower right")

# --- Right: fitted floor/x0/scale vs v_bulk/sigma_turb ---
good = np.isfinite(fit_floor)
axR.errorbar(
    ratios[good] + 1e-3, fit_floor[good], yerr=fit_floor_err[good],
    fmt="o-", ms=6.0, lw=1.5, capsize=3, color="#31688e", label="floor",
)
axR.errorbar(
    ratios[good] + 1e-3, fit_x0[good], yerr=fit_x0_err[good],
    fmt="s-", ms=6.0, lw=1.5, capsize=3, color="#d95f02", label=r"$x_0$",
)
axR.errorbar(
    ratios[good] + 1e-3, fit_scale[good], yerr=fit_scale_err[good],
    fmt="^-", ms=6.0, lw=1.5, capsize=3, color="#1b9e77", label="scale",
)
axR.axhline(0, color="0.25", ls="--", lw=1.0)
axR.set_xscale("symlog", linthresh=1.0)
axR.set_xlabel(r"$v_{\rm bulk}/\sigma_{\rm turb}$")
axR.set_ylabel("fitted logistic parameter")
axR.grid(True, axis="y", ls="--", alpha=0.4)
axR.legend(fontsize=9)

fig.tight_layout()

out = BASE_DIR / "random_point_count_dK_dlogr_vbulk_sweep.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print(f"\nWrote {out.relative_to(REPO_ROOT)}")
