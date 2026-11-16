#!/usr/bin/env python3
"""Infer observed shell-width ratios from an analytic logistic slope model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"

FLOOR = -2.0
SLOPE_CLIP_EPS_FLOOR = 0.01
SLOPE_CLIP_EPS_ZERO = 0.1
WIDTH_PLOT_MIN = 0.003
WIDTH_PLOT_MAX = 3.0
OBSERVED_POINTS = [
    ("BFS65 [S II]", 0.60631, 0.10278, -0.49, 0.20, "sii"),
    ("RCW60 [N II]", 0.39894, 0.02467, -0.66, 0.25, "nii"),
    ("RCW60 [S II]", 0.39894, 0.02467, -0.36, 0.28, "sii"),
    ("RCW87 [S II]", 0.48433, 0.12333, -0.06, 0.28, "sii"),
    ("RCW79 [N II]", 0.75340, 0.05139, -0.80, 0.33, "nii"),
    ("RCW79 [S II]", 0.75340, 0.05139, -0.29, 0.34, "sii"),
    ("Sh2-311 [S II]", 0.94710, 0.02056, -0.31, 0.42, "sii"),
    ("Gum31 [S II]", np.nan, 0.039, -1.55, 0.42, "sii"),
]


def bulk_switch(v_ratio: float, v_c: float, n: float) -> float:
    if np.isinf(v_ratio):
        return 1.0
    return v_ratio**n / (v_ratio**n + v_c**n)


def x0_from_rmin_ratio(rmin_ratio: float, v_ratio: float) -> float:
    rho = np.log10(rmin_ratio / 0.01)
    return -1.42 + 0.37 * rho + 0.35 * bulk_switch(v_ratio, 3.8, 0.87)


def scale_from_rmin_ratio(rmin_ratio: float, v_ratio: float) -> float:
    rho = np.log10(rmin_ratio / 0.01)
    return 10 ** (-0.38 - (0.45 + 0.21 * rho) * bulk_switch(v_ratio, 5.5, 1.3))


def slope_model(width_ratio: np.ndarray, rmin_ratio: float, v_ratio: float) -> np.ndarray:
    x0 = x0_from_rmin_ratio(rmin_ratio, v_ratio)
    scale = scale_from_rmin_ratio(rmin_ratio, v_ratio)
    return FLOOR / (1.0 + np.exp((np.log10(width_ratio) - x0) / scale))


def infer_width_ratio(slope: float, rmin_ratio: float, v_ratio: float) -> float:
    if not FLOOR < slope < 0.0:
        return np.nan
    x0 = x0_from_rmin_ratio(rmin_ratio, v_ratio)
    scale = scale_from_rmin_ratio(rmin_ratio, v_ratio)
    return float(10 ** (x0 + scale * np.log(FLOOR / slope - 1.0)))


def clip_slope_to_model_range(slope: float) -> tuple[float, bool]:
    clipped = float(np.clip(slope, FLOOR + SLOPE_CLIP_EPS_FLOOR, -SLOPE_CLIP_EPS_ZERO))
    return clipped, not np.isclose(clipped, slope)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v-ratio", type=float, default=0.0,
        help="v_bulk/v_turb used in the logistic parameters (use inf for the strong-flow limit)",
    )
    args = parser.parse_args()
    v_ratio = args.v_ratio
    tag = "vinf" if np.isinf(v_ratio) else f"v{v_ratio:g}"

    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    rows = []
    for name, rmin_pc, rmin_ratio, slope, slope_err, tracer in OBSERVED_POINTS:
        width = infer_width_ratio(slope, rmin_ratio, v_ratio)
        slope_plus_raw = slope + slope_err
        slope_minus_raw = slope - slope_err
        slope_plus, clipped_plus = clip_slope_to_model_range(slope_plus_raw)
        slope_minus, clipped_minus = clip_slope_to_model_range(slope_minus_raw)
        width_from_plus_err = infer_width_ratio(slope_plus, rmin_ratio, v_ratio)
        width_from_minus_err = infer_width_ratio(slope_minus, rmin_ratio, v_ratio)
        finite_bounds = [w for w in [width_from_plus_err, width_from_minus_err] if np.isfinite(w)]
        width_lo = min(finite_bounds) if finite_bounds else np.nan
        width_hi = max(finite_bounds) if finite_bounds else np.nan
        rows.append((
            name, tracer, rmin_pc, rmin_ratio, slope, slope_err,
            x0_from_rmin_ratio(rmin_ratio, v_ratio), scale_from_rmin_ratio(rmin_ratio, v_ratio),
            width, width_lo, width_hi, width_from_plus_err, width_from_minus_err,
            slope_plus_raw, slope_minus_raw, slope_plus, slope_minus, clipped_plus, clipped_minus,
        ))

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ratios = np.array([row[3] for row in rows])
    norm = matplotlib.colors.LogNorm(vmin=min(0.001, ratios.min()), vmax=max(0.13, ratios.max()))
    cmap = plt.colormaps["viridis"]

    width_grid = np.geomspace(WIDTH_PLOT_MIN, WIDTH_PLOT_MAX, 500)
    for rmin_ratio in np.geomspace(0.001, 0.13, 8):
        ax.plot(
            width_grid, slope_model(width_grid, rmin_ratio, v_ratio),
            color=cmap(norm(rmin_ratio)), lw=1.3, alpha=0.75,
        )

    for (
        name, tracer, _rmin_pc, rmin_ratio, slope, slope_err, _x0, _scale,
        width, width_lo, width_hi, _w_plus, _w_minus, _slope_plus_raw,
        _slope_minus_raw, _slope_plus, _slope_minus, clipped_plus, clipped_minus,
    ) in rows:
        color = {"sii": "#c51b7d", "nii": "#d95f02", "siii": "#7570b3"}[tracer]
        marker = {"sii": "D", "nii": "s", "siii": "^"}[tracer]
        width_lo_plot = width_lo if np.isfinite(width_lo) else WIDTH_PLOT_MIN
        width_hi_plot = width_hi if np.isfinite(width_hi) else WIDTH_PLOT_MAX
        xerr = np.array([[max(width - width_lo_plot, 0.0)], [max(width_hi_plot - width, 0.0)]])
        ax.errorbar(
            width, slope, xerr=xerr, yerr=slope_err, fmt=marker, ms=6.0,
            color=color, ecolor=color, capsize=3, markeredgecolor="white",
            markeredgewidth=0.7, zorder=8,
        )
        if clipped_plus or clipped_minus:
            ax.scatter(
                [width_hi if clipped_plus else width_lo],
                [slope],
                marker="|", s=90, color=color, zorder=9,
            )
        ax.annotate(
            name, xy=(width, slope), xytext=(5, 5), textcoords="offset points",
            color=color, fontsize=8, ha="left", va="bottom",
        )

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$r_{\rm min}/r_{\rm HII}$")

    ax.axvline(0.2, color="0.25", ls=":", lw=1.1)
    ax.axhline(0, color="0.25", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel(r"inferred $\Delta r_{\rm shell}/r_{\rm HII}$")
    ax.set_ylabel(r"$dK/d\log_{10} r$")
    v_label = r"\infty" if np.isinf(v_ratio) else f"{v_ratio:g}"
    ax.set_title(
        rf"Observed shell widths from logistic model ($v_{{\rm bulk}}/v_{{\rm turb}}={v_label}$)",
        fontsize=11,
    )
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_xlim(WIDTH_PLOT_MIN, WIDTH_PLOT_MAX)
    ax.set_ylim(-2.1, 0.1)
    fig.tight_layout()

    out_png = BASE_DIR / f"observed_sii_shell_width_from_logistic_{tag}.png"
    out_csv = BASE_DIR / f"observed_sii_shell_width_from_logistic_{tag}.csv"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(
            "region,tracer,r_min_pc,r_min_over_R_HII,slope,slope_err,x0,scale,"
            "inferred_delta_r_shell_over_R_HII,width_lo_from_slope_err,"
            "width_hi_from_slope_err,width_from_plus_err,width_from_minus_err,"
            "slope_plus_raw,slope_minus_raw,slope_plus_used,slope_minus_used,"
            "clipped_plus,clipped_minus\n"
        )
        for row in rows:
            (
                name, tracer, rmin_pc, rmin_ratio, slope, slope_err, x0, scale,
                width, width_lo, width_hi, w_plus, w_minus, slope_plus_raw,
                slope_minus_raw, slope_plus, slope_minus, clipped_plus, clipped_minus,
            ) = row
            f.write(
                f"{name},{tracer},{rmin_pc:.5f},{rmin_ratio:.5f},{slope:.4f},{slope_err:.4f},"
                f"{x0:.6f},{scale:.6f},{width:.6f},{width_lo:.6f},{width_hi:.6f},"
                f"{w_plus:.6f},{w_minus:.6f},{slope_plus_raw:.6f},{slope_minus_raw:.6f},"
                f"{slope_plus:.6f},{slope_minus:.6f},{clipped_plus},{clipped_minus}\n"
            )

    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
