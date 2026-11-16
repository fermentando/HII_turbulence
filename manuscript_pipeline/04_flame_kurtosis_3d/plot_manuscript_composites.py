#!/usr/bin/env python3
"""Manuscript composites built from the fixed-r_min comparison and the v_bulk/v_turb sweep.

Figure 1 (random_point_count_kurtosis_and_N_comparison.png):
  left   -- kurtosis curves K(l) behind panel 1 of the fixed-r_min comparison, one per shell width
            (colour = Delta r_shell/r_HII), N = N_KCURVES points, r_min floored to the N = 5,000 per-width value
            (the floor panel 1 uses for every N); dashed line: K = 3
  bottom -- plateau level K_plat vs shell width: fitted plateaus of the v_bulk/v_turb = 10, N = 100,000 runs
            vs the thin-shell model (plateau formula at lambda = 10) and the bulk limit K_b(0) = C_p - ln 2 Delta
            (compare_kolmogorov_model_lambda10.py; data read from kolmogorov_model_vs_lambda10_runs.csv)
  bottom right -- bend scale l_b vs width, log-log (joint softplus fits, bulk_limit_kurtosis_derivation.csv):
            simulation at lambda = 10, bulk-limit theory, and the thin-shell constant 2e^{-3/2} = 0.446

Figure 2 (random_point_count_rmin_and_vbulk_sweeps.png), one row:
  left   -- fitted logistic floor F (left axis) and scale s (right axis) vs v_bulk/v_turb, points coloured by
            v_bulk/v_turb (mako), switch fits in grey; v_bulk/v_turb = 0 plotted at 0.03
  middle -- dK/dlog r vs shell width: r_min sweep at N = 100,000 (rocket, solid, circles) overlaid with the
            N = 5,000 v_bulk/v_turb sweep (mako, dashed, squares); one colourbar each
  right  -- logistic s and x0 vs r_min (twin y axes, rocket tones)

The panels are drawn by draw_panels() of plot_fixed_rmin_comparison.py and
plot_vbulk_ratio_sweep_n5000_logistic.py (same caches, same fits), so they match the standalone figures.

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_manuscript_composites.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO_ROOT / "manuscript_pipeline" / "00_core")]

import plot_fixed_rmin_comparison as fixed_rmin  # noqa: E402
import plot_vbulk_ratio_sweep_n5000_logistic as vbulk_sweep  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402

N_KCURVES = 5000
BASE_DIR = fixed_rmin.BASE_DIR


def kurtosis_curves(n_points: int):
    """K(l) per shell width for n_points, with r_min floored to the N = 5,000 per-width value (as panel 1)."""
    points, _ = fixed_rmin.load_shell_points(n_points, fixed_rmin.WIDTHS)
    _, ref_r_min = fixed_rmin.load_shell_points(5000, fixed_rmin.WIDTHS)
    curves = []
    for (y, z, v), r_min in zip(points, ref_r_min):
        log_c, edges, dv_bins, _ = compute_bins(y, z, v, bin_resolution=fixed_rmin.BIN_RESOLUTION,
                                                min_separation=r_min)
        with np.errstate(invalid="ignore", divide="ignore"):
            *_, kk = compute_pdf_and_statistics(y, z, v, np.zeros_like(v), log_c, edges, dv_bins, order=1, n_jobs=1)
        curves.append((log_c, np.asarray(kk, dtype=float)))
    return curves


ZOOM_WIDTH = 0.1          # shell shown in the insets: the simulated width nearest to this
SLICE_HALF_THICKNESS = 0.1  # |z| < this (units of r_HII): slab through the shell centre containing the line of sight


def shell_points_3d(n_points: int, width: float):
    """3-D positions and line-of-sight (x) velocities of the random points of one simulated shell."""
    path = fixed_rmin.shell_output_path(fixed_rmin.run_dir(n_points), fixed_rmin.R_OUTER_FIXED - float(width),
                                        fixed_rmin.R_OUTER_FIXED)
    with np.load(path) as d:
        return d["point_x"], d["point_y"], d["point_z"], d["point_vx_total"]


PLATEAU_CSV = BASE_DIR / "kolmogorov_model_vs_lambda10_runs.csv"  # written by compare_kolmogorov_model_lambda10.py
PLATEAU_LAM = 10.0


def draw_plateau_model(ax, norm, cmap) -> None:
    """Plateau level K_plat vs shell width: simulation (lambda = 10, N = 100,000) vs the thin-shell model.

    Points: free softplus-fit K_plat of each simulated K(l) curve (coloured by width, as the panel above).
    Lines: plateau formula at lambda = 10 and the bulk limit K_b(0) = C_p - ln(2 Delta) (lambda -> inf),
    both evaluated on a fine width grid with compare_kolmogorov_model_lambda10.plateau().
    """
    import compare_kolmogorov_model_lambda10 as kmodel

    if not PLATEAU_CSV.exists():
        kmodel.main()
    d = np.genfromtxt(PLATEAU_CSV, delimiter=",", names=True)
    w_fine = np.geomspace(d["width"].min(), d["width"].max(), 60)
    k_lam = np.array([kmodel.plateau(w, PLATEAU_LAM) for w in w_fine])
    k_bulk = np.array([kmodel.plateau(w, np.inf) for w in w_fine])

    ax.plot(w_fine, k_bulk, "--", color="0.35", lw=1.4, zorder=2,
            label=r"$K_{\rm plat}=C_p-\ln 2\Delta r_{\rm shell}$")
    ax.plot(w_fine, k_lam, "-", color="#7B2CBF", lw=2.2, zorder=3,
            label="prediction")
    ax.errorbar(d["width"], d["K_plat_fit_data"], yerr=d["K_plat_fit_err"], fmt="none", ecolor="0.2",
                elinewidth=1.0, capsize=2, zorder=4)
    ax.scatter(d["width"], d["K_plat_fit_data"], c=d["width"], cmap=cmap, norm=norm, s=55,
               edgecolors="black", linewidths=0.8, zorder=5)
    ax.axhline(KURTOSIS_FLOOR, color="0.3", ls=":", lw=1.1)
    ax.set_xscale("log")
    ax.set_ylim(2.4, 6.6)
    ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    ax.set_ylabel(r"$K_{\rm plat}$")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)


BEND_CSV = BASE_DIR / "bulk_limit_kurtosis_derivation.csv"  # written by derive_bulk_limit_kurtosis.py
BEND_RATIO = 2 * np.exp(-1.5)  # l_b / Delta = exp(C_d - C_p), thin-shell bulk limit (kurtosis_bulk_limit_constants.md)


def draw_bend_scale(ax, norm, cmap) -> None:
    """Bend scale l_b / Delta r_shell vs shell width (joint softplus fits): simulation, bulk-limit theory, 2e^{-3/2}."""
    if not BEND_CSV.exists():
        import derive_bulk_limit_kurtosis

        derive_bulk_limit_kurtosis.main()
    d = np.genfromtxt(BEND_CSV, delimiter=",", names=True)
    w = d["width"]
    w_line = np.geomspace(0.8 * w.min(), 1.25 * w.max(), 50)
    ax.plot(w_line, BEND_RATIO * w_line, ":", color="0.3", lw=1.4, zorder=1,
            label=rf"$\ell_b=2e^{{-3/2}}\,\Delta r_{{\rm shell}}\simeq{BEND_RATIO:.2f}\,\Delta r_{{\rm shell}}$")
    ax.plot(w, d["l_b_theory"], "--", color="0.35", lw=1.4, zorder=2, label="bulk limit")
    ax.scatter(w, d["l_b_sim10"], c=w, cmap=cmap, norm=norm, s=55, edgecolors="black", linewidths=0.8,
               zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    ax.set_ylabel(r"$\ell_b/r_{\rm HII}$")
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)


def figure_1() -> None:
    from matplotlib.patches import ConnectionPatch, Rectangle

    curves = kurtosis_curves(N_KCURVES)
    widths = fixed_rmin.WIDTHS
    norm = matplotlib.colors.LogNorm(vmin=widths.min(), vmax=widths.max())
    cmap = plt.colormaps["viridis"]
    i_zoom = int(np.argmin(np.abs(np.log10(widths / ZOOM_WIDTH))))
    w_zoom = float(widths[i_zoom])

    fig = plt.figure(figsize=(10.0, 13.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.85], hspace=0.2, wspace=0.32)
    axK = fig.add_subplot(gs[0, :])
    axL = fig.add_subplot(gs[1, 0])
    axB = fig.add_subplot(gs[1, 1], sharex=axL)
    for k, (w, (log_c, kk)) in enumerate(zip(widths, curves)):
        ok = np.isfinite(kk)
        if k == i_zoom:
            axK.plot(10 ** log_c[ok], kk[ok], "-", color="black", lw=3.6, zorder=4)
        axK.plot(10 ** log_c[ok], kk[ok], "-", color=cmap(norm(w)), lw=2.2 if k == i_zoom else 1.6,
                 zorder=5 if k == i_zoom else 3)
    axK.axhline(KURTOSIS_FLOOR, color="0.3", ls="--", lw=1.1)
    axK.set_xscale("log")
    axK.set_xlabel(r"$\ell/r_{\rm HII}$")
    axK.set_ylabel(r"$K(\ell)$")
    axK.grid(True, axis="y", ls="--", alpha=0.4)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axK, location="top", orientation="horizontal", pad=0.02, fraction=0.05, aspect=40)
    cb.set_label(r"$\Delta r_{\rm shell}/r_{\rm HII}$", labelpad=8)
    cb.set_ticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    cb.set_ticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    cb.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # inset boxes (axes fractions of the K panel): 3-D shell on top, slice below, one column at the top right
    INSET_X, INSET_W, INSET_H = 0.67, 0.31, 0.36
    box3 = [INSET_X, 0.615, INSET_W, INSET_H]
    boxS = [INSET_X, 0.215, INSET_W, INSET_H]

    # --- insets: the zoomed shell in 3-D, and a slice through its centre ---
    px, py, pz, pv = shell_points_3d(N_KCURVES, w_zoom)
    vlim = np.percentile(np.abs(pv), 98)
    vcmap = sns.color_palette("vlag", as_cmap=True)
    ax3 = axK.inset_axes(box3, projection="3d")
    ax3.scatter(px, py, pz, c=pv, cmap=vcmap, vmin=-vlim, vmax=vlim, s=1.2, alpha=0.8, depthshade=False,
                rasterized=True)
    ax3.set_box_aspect((1, 1, 1), zoom=1.35)
    ax3.view_init(elev=18, azim=35)
    ax3.set_axis_off()
    ax3.set_zorder(7)
    ax3.set_facecolor("none")
    axK.add_patch(Rectangle(box3[:2], box3[2], box3[3], transform=axK.transAxes, fc="white", ec="black",
                            lw=1.6, zorder=6))

    axS = axK.inset_axes(boxS)
    axS.set_zorder(7)
    for spine in axS.spines.values():
        spine.set_linewidth(1.6)
    m = np.abs(pz) < SLICE_HALF_THICKNESS
    axS.scatter(px[m], py[m], c=pv[m], cmap=vcmap, vmin=-vlim, vmax=vlim, s=7, edgecolors="none", rasterized=True)
    t = np.linspace(0, 2 * np.pi, 300)
    for rad in (fixed_rmin.R_OUTER_FIXED, fixed_rmin.R_OUTER_FIXED - w_zoom):
        axS.plot(rad * np.cos(t), rad * np.sin(t), "-", color="0.35", lw=0.7)
    axS.set_aspect("equal", adjustable="datalim")
    axS.set_xticks([])
    axS.set_yticks([])
    # annotate the geometry: shell thickness (dimension arrows pointing at the wall from both sides, on the
    # +x side where the inset box has room) and outer radius (single arrow from the centre)
    r_out, r_in = fixed_rmin.R_OUTER_FIXED, fixed_rmin.R_OUTER_FIXED - w_zoom
    a_r = np.deg2rad(215.0)
    dim = dict(arrowstyle="-|>", color="black", lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=8)
    axS.annotate("", xy=(r_in, 0), xytext=(r_in - 0.22, 0), arrowprops=dim)
    axS.annotate("", xy=(r_out, 0), xytext=(r_out + 0.22, 0), arrowprops=dim)
    axS.text(r_out - 0.04, 0.12, r"$\Delta r_{\rm shell}$", ha="center", va="bottom", fontsize=11, color="black",
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    axS.plot(0, 0, "o", ms=3, color="black")
    axS.annotate("", xy=(r_out * np.cos(a_r), r_out * np.sin(a_r)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=0, shrinkB=0, mutation_scale=9))
    perp = a_r + np.pi / 2
    axS.text(0.5 * np.cos(a_r) + 0.13 * np.cos(perp), 0.5 * np.sin(a_r) + 0.13 * np.sin(perp), r"$r_{\rm HII}$",
             ha="center", va="center", fontsize=11, color="black")

    # connectors: highlighted curve -> 3-D inset -> slice
    log_c, kk = curves[i_zoom]
    ok = np.isfinite(kk)
    j0 = np.flatnonzero(ok)[np.argmin(np.abs(log_c[ok] - np.log10(0.05)))]
    axK.plot(10 ** log_c[j0], kk[j0], "o", ms=7, mfc="none", mec="black", mew=1.4, zorder=6)
    xmid = INSET_X + INSET_W / 2
    for xyA, coordsA, xyB in (((10 ** log_c[j0], kk[j0]), axK.transData, (INSET_X, box3[1] + INSET_H / 2)),
                              ((xmid, box3[1]), axK.transAxes, (xmid, boxS[1] + INSET_H))):
        con = ConnectionPatch(xyA=xyA, coordsA=coordsA, xyB=xyB, coordsB=axK.transAxes, color="0.3", lw=1.0,
                              arrowstyle="-|>", mutation_scale=10, zorder=10)
        fig.add_artist(con)

    draw_plateau_model(axL, norm, cmap)
    draw_bend_scale(axB, norm, cmap)

    out = BASE_DIR / "random_point_count_kurtosis_and_N_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


# Manuscript palettes (seaborn), trimmed so the near-black / near-white ends do not vanish:
#   mako   -- everything coloured by v_bulk/v_turb (v-sweep curves and the fitted F, s vs v_bulk/v_turb)
#   rocket -- everything coloured by r_min/r_HII (r_min-sweep curves; the s and x0 vs r_min panel)
_MAKO = sns.color_palette("mako", as_cmap=True)
_ROCKET = sns.color_palette("rocket", as_cmap=True)
V_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("mako_trim", _MAKO(np.linspace(0.10, 0.85, 256)))
RMIN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("rocket_trim", _ROCKET(np.linspace(0.15, 0.85, 256)))
ROCKET_DARK, ROCKET_LIGHT = matplotlib.colors.to_hex(_ROCKET(0.30)), matplotlib.colors.to_hex(_ROCKET(0.70))
FIT_GREY = "0.25"
V_ZERO_COLOR = "0.6"
V_ZERO_PLOT = 0.03  # where v_bulk/v_turb = 0 is placed on the log axis


def blacken(ax) -> None:
    """Black axis labels and tick labels, no title."""
    ax.set_title("")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="both", which="both", labelcolor="black")


def twin_of(ax):
    return next(a for a in ax.figure.axes if a is not ax and a.get_shared_x_axes().joined(a, ax))


def colorbar_axes(fig, before):
    return [a for a in fig.axes if a not in before and hasattr(a, "_colorbar")]


FIG2_LAYOUTS = {
    # one row, mirrored: v-sweep curves | F, s vs v | s, x0 vs r_min | r_min-sweep curves
    "mirror": dict(figsize=(21.0, 5.4), out="random_point_count_rmin_and_vbulk_sweeps_mirror.png",
                   boxes={"V": [0.035, 0.12, 0.190, 0.70], "C": [0.272, 0.12, 0.160, 0.70],
                          "D": [0.522, 0.12, 0.160, 0.70], "A": [0.762, 0.12, 0.190, 0.70]},
                   cbar_gap=0.03, cbar_h=0.025, divider=None),
    # one row: [F, s | v-sweep] | [r_min sweep | s, x0], vertical divider between the halves
    "row": dict(figsize=(21.0, 5.4), out="random_point_count_rmin_and_vbulk_sweeps.png",
                boxes={"C": [0.035, 0.12, 0.165, 0.70], "V": [0.285, 0.12, 0.190, 0.70],
                       "A": [0.590, 0.12, 0.190, 0.70], "D": [0.845, 0.12, 0.165, 0.70]},
                cbar_gap=0.03, cbar_h=0.025, divider=([0.53, 0.53], [0.12, 0.82])),
    # 2 x 2: sweep curves (with colourbars) in the left column, fitted parameters in the right;
    # top row v_bulk/v_turb, bottom row r_min; no divider
    "grid": dict(figsize=(11.0, 11.0), out="random_point_count_rmin_and_vbulk_sweeps_2x2.png",
                 boxes={"V": [0.08, 0.585, 0.37, 0.34], "C": [0.535, 0.585, 0.37, 0.34],
                        "A": [0.08, 0.065, 0.37, 0.34], "D": [0.535, 0.065, 0.37, 0.34]},
                 cbar_gap=0.02, cbar_h=0.014, divider=None),
}


def figure_2(layout: str = "row") -> None:
    L = FIG2_LAYOUTS[layout]
    fig = plt.figure(figsize=L["figsize"])
    B = L["boxes"]
    axC = fig.add_axes(B["C"])   # F, s vs v_bulk/v_turb
    axV = fig.add_axes(B["V"])   # v_bulk/v_turb sweep curves
    axA = fig.add_axes(B["A"])   # r_min sweep curves
    axD = fig.add_axes(B["D"])   # s, x0 vs r_min

    # --- v_bulk/v_turb sweep (mako), colourbar to the right ---
    before = list(fig.axes)
    dummy = plt.figure()
    dF, dS = dummy.subplots(1, 2)
    fit_rows, switch_fits, _ = vbulk_sweep.draw_panels(fig, axV, dF, dS, v_cmap=V_CMAP, zero_color=V_ZERO_COLOR)
    plt.close(dummy)
    cb_v = colorbar_axes(fig, before)

    # --- r_min sweep (rocket) and its s, x0 panel; colourbar to the left ---
    before = list(fig.axes)
    dummy = plt.figure()
    dL = dummy.add_subplot(111)
    fixed_rmin.draw_panels(fig, dL, axA, axD, rmin_cmap=RMIN_CMAP, scale_color=ROCKET_DARK, x0_color=ROCKET_LIGHT)
    plt.close(dummy)
    cb_r = colorbar_axes(fig, before)

    for ax in (axV, axA):
        ax.set_ylabel(r"$dK/d\log_{10} r$")
    if axV.get_legend() is not None:
        axV.get_legend().remove()
        axV.legend(handles=[plt.Line2D([], [], color=V_ZERO_COLOR, ls="none", marker="o", ms=5,
                                       label=r"$v_{\rm bulk}/v_{\rm turb}=0$")], fontsize=9, loc="lower right")

    # --- left: fitted F (left axis) and s (right axis) vs v_bulk/v_turb, points coloured by v_bulk/v_turb ---
    rows = np.asarray(fit_rows, dtype=float)  # ratio, floor, floor_err, scale, scale_err
    ratio = rows[:, 0]
    x = np.where(ratio > 0, ratio, V_ZERO_PLOT)
    vnorm = matplotlib.colors.LogNorm(vmin=ratio[ratio > 0].min(), vmax=ratio.max())
    colors = [V_ZERO_COLOR if r == 0 else V_CMAP(vnorm(r)) for r in ratio]
    axC2 = axC.twinx()
    v_dense = np.concatenate(([0.0], np.geomspace(0.05, ratio.max(), 300)))
    x_dense = np.where(v_dense > 0, v_dense, V_ZERO_PLOT)
    for ax, col, mk, ls, name in ((axC, 1, "o", "-", "floor"), (axC2, 3, "^", "--", "scale")):
        params = switch_fits[name][0]
        ax.plot(x_dense, vbulk_sweep.bulk_switch(v_dense, *params), ls, color=FIT_GREY, lw=1.4, zorder=2)
        for xi, yi, ei, ci in zip(x, rows[:, col], rows[:, col + 1], colors):
            ax.errorbar(xi, yi, yerr=ei, fmt=mk, ms=6.5, color=ci, mec="black", mew=0.5, capsize=2.5, lw=1.1,
                        zorder=3)
    axC.set_xscale("log")
    axC.axhline(0, color="0.3", ls=":", lw=1.0)
    axC.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    axC.set_ylabel(r"$F$")
    axC2.set_ylabel(r"$s$")
    axC.grid(True, axis="y", ls="--", alpha=0.4)
    axC.legend(handles=[
        plt.Line2D([], [], color=FIT_GREY, ls="-", marker="o", mfc="0.6", mec="black", ms=6, label=r"$F$"),
        plt.Line2D([], [], color=FIT_GREY, ls="--", marker="^", mfc="0.6", mec="black", ms=6, label=r"$s$"),
    ], fontsize=10, loc="center right")

    # --- right: logistic s and x0 vs r_min (twin y axes, rocket tones) ---
    axD2 = twin_of(axD)
    axD.set_ylabel(r"$s$")
    axD.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.2, 0.3, 0.4, 0.6, 1.0]))
    axD.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axD.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axD2.set_ylabel(r"$x_0$")
    leg = axD.get_legend()
    for text in leg.get_texts():
        t = text.get_text()
        t = t.replace("scale (left axis)", "$s$").replace("$x_0$ (right axis)", "$x_0$")
        if t.startswith("scale "):
            t = "$s$ " + t[len("scale "):]
        text.set_text(t)

    for ax in (axC, axC2, axV, axA, axD, axD2):
        blacken(ax)
    for ax in fig.axes:  # colorbars too
        ax.yaxis.label.set_color("black")

    # fixed positions (colorbar creation shrinks the parent axes): restore, then place the colourbars
    for ax, box in ((axC, B["C"]), (axV, B["V"]), (axA, B["A"]), (axD, B["D"])):
        ax.set_position(box)
    axC2.set_position(axC.get_position())
    axD2.set_position(axD.get_position())
    # replace the vertical colourbars by horizontal ones above their panels
    for old, ax, label in ((cb_v, axV, r"$v_{\rm bulk}/v_{\rm turb}$"), (cb_r, axA, r"$r_{\rm min}/r_{\rm HII}$")):
        mappable = old[0]._colorbar.mappable
        for cax in old:
            cax.remove()
        b = ax.get_position()
        cax = fig.add_axes([b.x0, b.y1 + L["cbar_gap"], b.width, L["cbar_h"]])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(label, color="black", labelpad=10)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelcolor="black")
    if L["divider"] is not None:
        fig.add_artist(plt.Line2D(*L["divider"], transform=fig.transFigure, color="black", lw=1.6))

    out = BASE_DIR / L["out"]
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    figure_1()
    figure_2("row")
    figure_2("grid")
    figure_2("mirror")


if __name__ == "__main__":
    main()
