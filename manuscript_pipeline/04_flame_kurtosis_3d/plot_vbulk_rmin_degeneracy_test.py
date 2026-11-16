#!/usr/bin/env python3
"""Test whether v_bulk/v_turb mimics r_min/r_HII in dK/dlogr shell-width curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"

FLOOR_FIXED = -2.0
VBULK_CACHE = BASE_DIR / "_vbulk_ratio_sweep_n5000_logistic_cache.npz"
RMIN_CACHE = BASE_DIR / "_fixed_rmin_right_cache.npz"


def logistic(log_w: np.ndarray, x0: float, scale: float) -> np.ndarray:
    return FLOOR_FIXED / (1.0 + np.exp((log_w - x0) / scale))


def fit_logistic(widths: np.ndarray, slopes: np.ndarray, fallback: np.ndarray | None = None) -> tuple[float, float]:
    mask = np.isfinite(slopes)
    if fallback is not None:
        mask &= ~fallback
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan
    log_w = np.log10(widths)
    try:
        params, _cov = curve_fit(
            logistic,
            log_w[mask],
            slopes[mask],
            p0=[float(np.median(log_w[mask])), 0.35],
            bounds=([-4.0, 0.03], [2.0, 5.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan
    return float(params[0]), float(params[1])


def load_vbulk() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(VBULK_CACHE) as cache:
        return (
            np.asarray(cache["vbulk_ratios"], dtype=float),
            np.asarray(cache["widths"], dtype=float),
            np.asarray(cache["slopes"], dtype=float),
            np.asarray(cache["fallback"], dtype=bool),
        )


def load_rmin() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(RMIN_CACHE) as cache:
        rmin_ratios = np.asarray(cache["rmin_sweep"], dtype=float)
        widths = np.asarray(cache["widths"], dtype=float)
        slopes = np.asarray(cache["slopes"], dtype=float)
        fallback = np.asarray(cache["fallback"], dtype=bool) if "fallback" in cache else slopes == 0.0
    return rmin_ratios, widths, slopes, fallback


def interpolated_rmin_curve(log_rmin_ratio: float, log_rmin_grid: np.ndarray, rmin_slopes: np.ndarray) -> np.ndarray:
    return np.array([
        interp1d(log_rmin_grid, rmin_slopes[:, i], kind="linear", fill_value="extrapolate")(log_rmin_ratio)
        for i in range(rmin_slopes.shape[1])
    ])


def best_effective_rmin(v_curve: np.ndarray, log_rmin_grid: np.ndarray, rmin_slopes: np.ndarray) -> tuple[float, float]:
    dense_log_rmin = np.linspace(log_rmin_grid.min(), log_rmin_grid.max(), 2000)
    rms = np.full_like(dense_log_rmin, np.nan, dtype=float)
    mask = np.isfinite(v_curve)
    for i, log_rmin in enumerate(dense_log_rmin):
        model_curve = interpolated_rmin_curve(log_rmin, log_rmin_grid, rmin_slopes)
        good = mask & np.isfinite(model_curve)
        if np.count_nonzero(good) >= 3:
            rms[i] = float(np.sqrt(np.nanmean((v_curve[good] - model_curve[good]) ** 2)))
    idx = int(np.nanargmin(rms))
    return float(10 ** dense_log_rmin[idx]), float(rms[idx])


def main() -> None:
    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    v_ratios, widths, v_slopes, v_fallback = load_vbulk()
    rmin_ratios, rmin_widths, rmin_slopes, rmin_fallback = load_rmin()
    if not np.allclose(widths, rmin_widths):
        raise ValueError("v_bulk and r_min caches use different shell-width grids.")

    log_widths = np.log10(widths)
    log_rmin_grid = np.log10(rmin_ratios)
    rmin_valid_slopes = np.where(rmin_fallback, np.nan, rmin_slopes)

    v_params = np.array([
        fit_logistic(widths, curve, fallback)
        for curve, fallback in zip(v_slopes, v_fallback)
    ])
    rmin_params = np.array([
        fit_logistic(widths, curve, fallback)
        for curve, fallback in zip(rmin_slopes, rmin_fallback)
    ])

    matches = []
    for ratio, curve in zip(v_ratios, v_slopes):
        best_rmin, rms = best_effective_rmin(curve, log_rmin_grid, rmin_valid_slopes)
        matches.append((ratio, best_rmin, rms))
    matches = np.asarray(matches, dtype=float)

    rmin_dynamic_rms = float(np.sqrt(np.nanmean(np.nanvar(rmin_valid_slopes, axis=0))))
    vbulk_dynamic_rms = float(np.sqrt(np.nanmean(np.nanvar(v_slopes, axis=0))))

    fig, (axC, axP) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    nonzero = v_ratios > 0
    norm_v = matplotlib.colors.LogNorm(vmin=v_ratios[nonzero].min(), vmax=v_ratios[nonzero].max())
    cmap_v = plt.colormaps["plasma"]
    norm_r = matplotlib.colors.LogNorm(vmin=rmin_ratios.min(), vmax=rmin_ratios.max())
    cmap_r = plt.colormaps["viridis"]

    for ratio, curve in zip(v_ratios, v_slopes):
        color = "0.25" if ratio == 0 else cmap_v(norm_v(ratio))
        axC.plot(widths, curve, "o-", ms=3.8, lw=1.0, color=color, alpha=0.6)
    for rmin_ratio, curve in zip(rmin_ratios, rmin_slopes):
        axC.plot(widths, curve, "-", lw=2.0, color=cmap_r(norm_r(rmin_ratio)), alpha=0.9)
    axC.axvline(0.2, color="0.25", ls=":", lw=1.1)
    axC.axhline(0, color="0.25", ls="--", lw=1.1)
    axC.set_xscale("log")
    axC.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    axC.set_ylabel(r"$dK/d\log_{10} r$")
    axC.set_title("Curve families: bulk sweep vs r_min sweep", fontsize=11)
    axC.grid(True, axis="y", ls="--", alpha=0.4)

    sm_v = matplotlib.cm.ScalarMappable(norm=norm_v, cmap=cmap_v)
    sm_v.set_array([])
    cbar_v = fig.colorbar(sm_v, ax=axC, fraction=0.046, pad=0.02)
    cbar_v.set_label(r"$v_{\rm bulk}/v_{\rm turb}$")

    axP.plot(rmin_params[:, 0], rmin_params[:, 1], "o-", color="#1b9e77", lw=1.8, label=r"$r_{\rm min}$ sweep")
    axP.scatter(
        v_params[nonzero, 0], v_params[nonzero, 1],
        c=v_ratios[nonzero], cmap=cmap_v, norm=norm_v, marker="s", s=48,
        edgecolor="white", linewidth=0.5, label=r"$v_{\rm bulk}/v_{\rm turb}$ sweep",
    )
    axP.scatter(v_params[~nonzero, 0], v_params[~nonzero, 1], color="0.25", marker="s", s=48, label="bulk ratio 0")
    for rmin_ratio, (x0, scale) in zip(rmin_ratios, rmin_params):
        axP.annotate(f"{rmin_ratio:.3g}", (x0, scale), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#1b9e77")
    axP.set_xlabel(r"logistic $x_0$")
    axP.set_ylabel("logistic scale")
    axP.set_title(fr"Parameter-space test, floor fixed$={FLOOR_FIXED:.1f}$", fontsize=11)
    axP.grid(True, ls="--", alpha=0.35)
    axP.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_png = BASE_DIR / "vbulk_vs_rmin_degeneracy_test.png"
    out_csv = BASE_DIR / "vbulk_vs_rmin_degeneracy_test.csv"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(
            "v_bulk_over_v_turb,best_effective_r_min_over_R_HII,curve_rms_to_best_rmin,"
            "vbulk_x0,vbulk_scale,rmin_family_dynamic_rms,vbulk_family_dynamic_rms\n"
        )
        for (ratio, best_rmin, rms), (x0, scale) in zip(matches, v_params):
            f.write(
                f"{ratio:.8g},{best_rmin:.8g},{rms:.8g},{x0:.8g},{scale:.8g},"
                f"{rmin_dynamic_rms:.8g},{vbulk_dynamic_rms:.8g}\n"
            )

    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    print(f"Typical RMS variation across r_min family: {rmin_dynamic_rms:.4f}")
    print(f"Typical RMS variation across v_bulk family: {vbulk_dynamic_rms:.4f}")


if __name__ == "__main__":
    main()
