#!/usr/bin/env python3
"""2D response model for dK/dlogr versus v_bulk/v_turb and r_min/r_HII.

This is the controlled decomposition companion to
plot_vbulk_ratio_sweep_n5000_logistic.py:

1. Reuse the same N=5,000 random-point shell data generated for the
   v_bulk/v_turb sweep. Those runs use the same turbulent field seed and
   reset the point RNG for every bulk ratio, so the sky coordinates are
   matched across v_bulk/v_turb.
2. Sweep r_min only as a post-processing min_separation floor when building
   the VSF/kurtosis bins.
3. Fit each dK/dlogr(shell width) curve with the same fixed-floor logistic
   form, then fit simple 2D surfaces for x0 and scale:

       x0 = a0 + a1 log10(r_min/r_HII)
              + a2 log10(1 + v_bulk/v_turb)
              + a3 log10(r_min/r_HII) log10(1 + v_bulk/v_turb)

   and the same form for log10(scale).

The interaction terms a3/b3 are the useful degeneracy diagnostic: if they
are small, r_min and v_bulk/v_turb are close to separable nuisance axes; if
they are large, one parameter changes the inferred effect of the other.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from generate_shell_thickness_grid import R_OUTER_FIXED, shell_output_path, width_grid  # noqa: E402
from plot_shell_dk_dlogr_aperture_comparison import fit_k_vs_logr_above_floor  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402
from plot_vbulk_ratio_sweep_n5000_logistic import (  # noqa: E402
    BASE_DIR,
    BIN_RESOLUTION,
    N_POINTS,
    VBULK_RATIOS,
    ensure_shell_data,
    run_dir,
)
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402

WIDTHS = width_grid(0.02, 1.0, 14)

# N=5,000 has a natural nearest-neighbor floor of about 0.011 r_HII, so
# floors below ~0.01 cannot actually lower r_min. This grid samples the
# useful coarser-floor regime while keeping the natural-resolution endpoint.
FLOOR_FIXED = -2.0
RMIN_RATIOS = np.geomspace(0.01, 0.15, 8)  # upper end covers the observed regions (max ~0.123)
KK_ESTIMATOR = "pairs"  # <dv^4>/<dv^2>^2 over pairs (was the binned-PDF kurtosis)

MIN_FIT_POINTS_FOR_LINE = 3
CACHE = BASE_DIR / "_vbulk_rmin_2d_response_model_cache.npz"


def load_shell_points(ratio: float, widths: np.ndarray):
    records = []
    for width in widths:
        r_inner = R_OUTER_FIXED - float(width)
        path = shell_output_path(run_dir(float(ratio)), r_inner, R_OUTER_FIXED)
        with np.load(path) as data:
            records.append(
                (
                    np.asarray(data["y"], dtype=float),
                    np.asarray(data["z"], dtype=float),
                    np.asarray(data["projected_v"], dtype=float),
                )
            )
    return records


def slope_for_width(
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    min_separation: float,
) -> tuple[float, bool, float]:
    log_centers, bin_edges, dv_bins, bins = compute_bins(
        y,
        z,
        v,
        bin_resolution=BIN_RESOLUTION,
        min_separation=min_separation,
    )
    vel_err = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        _pdf, _c2, _c4, _cnt, _dv_centers, kk_pairs = compute_pdf_and_statistics(
            y,
            z,
            v,
            vel_err,
            log_centers,
            bin_edges,
            dv_bins,
            order=1,
            n_jobs=1,
        )
    fit = fit_k_vs_logr_above_floor({"r": 10**log_centers, "kk": kk_pairs}, floor=KURTOSIS_FLOOR)
    return float(fit["slope"]), bool(fit["used_zero_fallback"]), float(bins["r_min"])


def logistic(log_w: np.ndarray, x0: float, scale: float) -> np.ndarray:
    return FLOOR_FIXED / (1.0 + np.exp((log_w - x0) / scale))


def fit_logistic(widths: np.ndarray, slopes: np.ndarray, fallback: np.ndarray):
    mask = np.isfinite(slopes) & ~fallback
    if int(np.count_nonzero(mask)) < MIN_FIT_POINTS_FOR_LINE:
        return np.nan, np.nan, np.nan, np.nan
    log_w = np.log10(widths)
    try:
        params, cov = curve_fit(
            logistic,
            log_w[mask],
            slopes[mask],
            p0=[float(np.nanmedian(log_w[mask])), 0.35],
            bounds=([-4.0, 0.03], [2.0, 5.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan
    if np.all(np.isfinite(cov)):
        err = np.sqrt(np.diag(cov))
    else:
        err = np.full(2, np.nan)
    return float(params[0]), float(err[0]), float(params[1]), float(err[1])


def compute_ratio_row(ratio: float):
    """All (r_min, width) slopes for one v_bulk/v_turb; identical points across ratios."""
    shell_points = load_shell_points(ratio, WIDTHS)
    slopes = np.full((len(RMIN_RATIOS), len(WIDTHS)), np.nan)
    fallback = np.zeros_like(slopes, dtype=bool)
    effective_rmin = np.full_like(slopes, np.nan)
    for j, rmin_ratio in enumerate(RMIN_RATIOS):
        min_sep = float(rmin_ratio * R_OUTER_FIXED)
        for k, (y, z, v) in enumerate(shell_points):
            slopes[j, k], fallback[j, k], effective_rmin[j, k] = slope_for_width(
                y,
                z,
                v,
                min_separation=min_sep,
            )
    return slopes, fallback, effective_rmin


def load_or_compute():
    if CACHE.exists():
        with np.load(CACHE) as c:
            if (
                np.allclose(c["vbulk_ratios"], VBULK_RATIOS)
                and np.allclose(c["rmin_ratios"], RMIN_RATIOS)
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
                and int(c["n_points"]) == N_POINTS
                and "kk_estimator" in c and str(c["kk_estimator"]) == KK_ESTIMATOR
            ):
                print(f"Loading cached 2D grid from {CACHE.relative_to(REPO_ROOT)}...")
                return c["slopes"], c["fallback"], c["effective_rmin"]
        print("2D grid cache does not match current constants; recomputing.")

    ensure_shell_data()
    slopes = np.full((len(VBULK_RATIOS), len(RMIN_RATIOS), len(WIDTHS)), np.nan)
    fallback = np.zeros_like(slopes, dtype=bool)
    effective_rmin = np.full_like(slopes, np.nan)

    print("Computing 2D v_bulk/v_turb x r_min/r_HII grid...")
    n_workers = min(len(VBULK_RATIOS), max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, (ratio_slopes, ratio_fallback, ratio_rmin) in enumerate(
            pool.map(compute_ratio_row, [float(r) for r in VBULK_RATIOS])
        ):
            slopes[i], fallback[i], effective_rmin[i] = ratio_slopes, ratio_fallback, ratio_rmin
            print(f"  vratio={VBULK_RATIOS[i]:g} done: {['%.3f' % s for s in slopes[i, 0, :5]]} ...")

    np.savez_compressed(
        CACHE,
        vbulk_ratios=VBULK_RATIOS,
        rmin_ratios=RMIN_RATIOS,
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        n_points=N_POINTS,
        kk_estimator=KK_ESTIMATOR,
        slopes=slopes,
        fallback=fallback,
        effective_rmin=effective_rmin,
    )
    print(f"Cached to {CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback, effective_rmin


def design_matrix(rmin_ratios: np.ndarray, vbulk_ratios: np.ndarray) -> np.ndarray:
    r = np.log10(rmin_ratios)
    v = np.log10(1.0 + vbulk_ratios)
    return np.column_stack((np.ones_like(r), r, v, r * v))


def fit_surface(
    rmin_grid: np.ndarray,
    vbulk_grid: np.ndarray,
    values: np.ndarray,
    transform_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    good = np.isfinite(values)
    x = design_matrix(rmin_grid[good], vbulk_grid[good])
    y = values[good]
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = np.full_like(values, np.nan, dtype=float)
    pred[good] = design_matrix(rmin_grid[good], vbulk_grid[good]) @ coeff
    resid = values[good] - pred[good]
    rms = float(np.sqrt(np.nanmean(resid**2)))
    print(f"{transform_name} surface RMS residual: {rms:.4f}")
    return coeff, pred, rms


def two_way_variance_fractions(values: np.ndarray, name: str) -> dict[str, float]:
    """Additive main-effect / interaction split of a fully crossed (v_bulk, r_min) grid.

    Uses only r_min columns with a finite value at every v_bulk so the design
    stays balanced; does not assume any functional form along either axis.
    """
    cols = np.all(np.isfinite(values), axis=0)
    grid = values[:, cols]
    if grid.shape[1] < 2:
        return {"n_rmin_used": int(grid.shape[1])}
    grand = grid.mean()
    row = grid.mean(axis=1, keepdims=True) - grand
    col = grid.mean(axis=0, keepdims=True) - grand
    inter = grid - grand - row - col
    ss_v = float(grid.shape[1] * np.sum(row**2))
    ss_r = float(grid.shape[0] * np.sum(col**2))
    ss_i = float(np.sum(inter**2))
    total = ss_v + ss_r + ss_i
    out = {
        "n_rmin_used": int(grid.shape[1]),
        "frac_vbulk": ss_v / total,
        "frac_rmin": ss_r / total,
        "frac_interaction": ss_i / total,
        "range_vbulk_effect": float(np.ptp(row)),
        "range_rmin_effect": float(np.ptp(col)),
    }
    print(
        f"{name}: variance from v_bulk/v_turb {out['frac_vbulk']:.1%}, "
        f"r_min/r_HII {out['frac_rmin']:.1%}, interaction {out['frac_interaction']:.1%} "
        f"(main-effect ranges {out['range_vbulk_effect']:.3f} vs {out['range_rmin_effect']:.3f}; "
        f"{out['n_rmin_used']}/{len(RMIN_RATIOS)} r_min columns)"
    )
    return out


def write_tables(
    x0: np.ndarray,
    x0_err: np.ndarray,
    scale: np.ndarray,
    scale_err: np.ndarray,
    effective_rmin: np.ndarray,
    x0_pred: np.ndarray,
    log_scale_pred: np.ndarray,
    x0_coeff: np.ndarray,
    log_scale_coeff: np.ndarray,
    x0_rms: float,
    log_scale_rms: float,
) -> None:
    out_csv = BASE_DIR / "vbulk_rmin_2d_response_model_fits.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "v_bulk_over_v_turb",
                "requested_r_min_over_R_HII",
                "median_effective_r_min_over_R_HII",
                "floor_fixed",
                "x0",
                "x0_err",
                "scale",
                "scale_err",
                "x0_surface_model",
                "scale_surface_model",
            ]
        )
        for i, ratio in enumerate(VBULK_RATIOS):
            for j, rmin_ratio in enumerate(RMIN_RATIOS):
                writer.writerow(
                    [
                        f"{ratio:.8g}",
                        f"{rmin_ratio:.8g}",
                        f"{np.nanmedian(effective_rmin[i, j] / R_OUTER_FIXED):.8g}",
                        f"{FLOOR_FIXED:.8g}",
                        f"{x0[i, j]:.8g}",
                        f"{x0_err[i, j]:.8g}",
                        f"{scale[i, j]:.8g}",
                        f"{scale_err[i, j]:.8g}",
                        f"{x0_pred[i, j]:.8g}",
                        f"{10 ** log_scale_pred[i, j]:.8g}" if np.isfinite(log_scale_pred[i, j]) else "nan",
                    ]
                )

    out_coeff = BASE_DIR / "vbulk_rmin_2d_response_model_coefficients.csv"
    with out_coeff.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "parameter",
                "intercept",
                "log10_rmin_coeff",
                "log10_1plus_vbulk_coeff",
                "interaction_coeff",
                "rms_residual",
            ]
        )
        writer.writerow(["x0", *[f"{v:.8g}" for v in x0_coeff], f"{x0_rms:.8g}"])
        writer.writerow(["log10_scale", *[f"{v:.8g}" for v in log_scale_coeff], f"{log_scale_rms:.8g}"])

    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_coeff.relative_to(REPO_ROOT)}")


def plot_response(
    x0: np.ndarray,
    scale: np.ndarray,
    x0_resid: np.ndarray,
    log_scale_resid: np.ndarray,
) -> None:
    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    x = np.log10(1.0 + VBULK_RATIOS)
    y = RMIN_RATIOS
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], x0, r"logistic $x_0$", "viridis", None),
        (axes[0, 1], scale, "logistic scale", "magma", matplotlib.colors.LogNorm()),
        (axes[1, 0], x0_resid, r"$x_0$ residual", "coolwarm", None),
        (axes[1, 1], log_scale_resid, r"$\log_{10}$(scale) residual", "coolwarm", None),
    ]

    for ax, values, title, cmap, norm in panels:
        if norm is None and "residual" in title:
            vmax = np.nanmax(np.abs(values))
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        mesh = ax.pcolormesh(xx, yy, values, shading="nearest", cmap=cmap, norm=norm)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(title)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.grid(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"requested $r_{\rm min}/r_{\rm HII}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(1 + v_{\rm bulk}/v_{\rm turb})$")

    fig.tight_layout()
    out_png = BASE_DIR / "vbulk_rmin_2d_response_model.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")


def main() -> None:
    slopes, fallback, effective_rmin = load_or_compute()

    x0 = np.full((len(VBULK_RATIOS), len(RMIN_RATIOS)), np.nan)
    x0_err = np.full_like(x0, np.nan)
    scale = np.full_like(x0, np.nan)
    scale_err = np.full_like(x0, np.nan)

    for i in range(len(VBULK_RATIOS)):
        for j in range(len(RMIN_RATIOS)):
            x0[i, j], x0_err[i, j], scale[i, j], scale_err[i, j] = fit_logistic(
                WIDTHS,
                slopes[i, j],
                fallback[i, j],
            )

    v_grid, r_grid = np.meshgrid(VBULK_RATIOS, RMIN_RATIOS, indexing="ij")
    x0_coeff, x0_pred, x0_rms = fit_surface(r_grid, v_grid, x0, "x0")
    log_scale = np.full_like(scale, np.nan)
    good_scale = np.isfinite(scale) & (scale > 0)
    log_scale[good_scale] = np.log10(scale[good_scale])
    log_scale_coeff, log_scale_pred, log_scale_rms = fit_surface(
        r_grid,
        v_grid,
        log_scale,
        "log10(scale)",
    )

    write_tables(
        x0,
        x0_err,
        scale,
        scale_err,
        effective_rmin,
        x0_pred,
        log_scale_pred,
        x0_coeff,
        log_scale_coeff,
        x0_rms,
        log_scale_rms,
    )
    out_var = BASE_DIR / "vbulk_rmin_2d_response_model_variance_fractions.csv"
    keys = ["n_rmin_used", "frac_vbulk", "frac_rmin", "frac_interaction", "range_vbulk_effect", "range_rmin_effect"]
    with out_var.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", *keys])
        for name, values in (("x0", x0), ("log10_scale", log_scale)):
            fractions = two_way_variance_fractions(values, name)
            writer.writerow([name, *[f"{fractions.get(k, np.nan):.8g}" for k in keys]])
    print(f"Wrote {out_var.relative_to(REPO_ROOT)}")

    plot_response(x0, scale, x0 - x0_pred, log_scale - log_scale_pred)


if __name__ == "__main__":
    main()
