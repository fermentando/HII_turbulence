#!/usr/bin/env python3
"""Composite: example segmented sqrt(S2) fits + first-segment slope vs l_break / R_HII, all tracers.

Rebuild of ShareSpace/LVM/vsf_fits_and_slope_vs_radius_sqrtS2_dBIC6_snr20.png (same canvas,
2309 x 1421 px at 200 dpi, same panel geometry) from NPZ/VSFORD_<region>_<LINE>.npz
(full FOV, S/N > 20, block-jackknife replicates; 06_vsf_orders/vsf_order_vs3_allregions.py).

Fit: continuous segmented power law on log10 sqrt(S2) vs log10 l (fit_segmented_vsf_breakpoints.py),
<= 3 segments, >= 5 points/segment, Delta-BIC >= 6, bins with >= 1000 pairs.
Slope errors: block jackknife, refitting every replicate with the selected break positions fixed
(the covariance errors of the fit ignore the correlation between lag bins and are ~2x too small).

Main panel: filled symbols = a break was selected (x = first break / R_HII, y = first-segment slope);
open symbols = no break (x = largest fitted lag / R_HII, y = single power-law slope).
Open symbols are left out of the marginal distributions.
Regions too sparse for the 1000-pair cut (RCW87) get a single power law over bins with
>= SPARSE_MIN_PAIRS pairs and are always drawn open.

Usage:
    python manuscript_pipeline/03_slopes_turnover/plot_vsf_fits_and_slope_vs_radius.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_segmented_vsf_breakpoints import (  # noqa: E402
    best_fit_by_segment_count,
    break_log_positions,
    select_fit,
    weighted_least_squares,
)

REPO_ROOT = HERE.parents[1]
RADIUS_CSV = REPO_ROOT / "manuscript_notes" / "adopted_hii_region_radii.csv"
NPZ_DIR = REPO_ROOT / "NPZ"
OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "03_slopes_turnover"
STYLE = REPO_ROOT / "custom_plot.mplstyle"
OUT_NAME = "vsf_fits_and_slope_vs_radius_sqrtS2_dBIC6_snr20_jackknife"

MIN_PAIRS = 1000
SPARSE_MIN_PAIRS = 50
MIN_DELTA_BIC = 6.0
MAX_SEGMENTS = 3
MIN_POINTS_PER_SEGMENT = 5
LINES = ["NII", "SII", "SIII"]
LINE_LABEL = {"NII": "[NII]", "SII": "[SII]", "SIII": "[SIII]"}
LINE_COLOR = {"NII": "#8C2D5A", "SII": "#D6164F", "SIII": "#F39A2B"}
EXAMPLES = ["BFS65", "Gum38a", "RCW79"]
REGIONS = ["BFS65", "Flame_nebula", "Gum31", "Gum38a", "RCW52", "RCW60", "RCW79", "RCW87",
           "Rosette_nebula", "Sh2-311"]
MARKER = {"BFS65": "o", "Flame_nebula": "s", "Gum31": "D", "Gum38a": "^", "RCW52": "v",
          "RCW60": "P", "RCW79": "X", "RCW87": "*", "Rosette_nebula": "h", "Sh2-311": "<"}
LABEL = {"Flame_nebula": "Flame nebula", "Rosette_nebula": "Rosette nebula"}

# Panel boxes measured on the original PNG: (x0, x1, y_top, y_bottom) in pixels.
W_PX, H_PX, DPI = 2309, 1421, 200
SCALE = 1.22 # canvas enlarged relative to the original; panel layout kept in proportion, fonts fixed in pt
# Fits forced to a given number of segments (overrides the Delta-BIC selection), as in the original figure.
FORCE_SEGMENTS = {("RCW79", "SIII"): 3}
BOX = {
    "left0": (155, 904, 35, 394), "left1": (155, 904, 480, 839), "left2": (155, 904, 925, 1284),
    "top": (1078, 2021, 35, 281), "main": (1078, 2021, 309, 1284), "right": (2047, 2285, 309, 1284),
}

TICKS = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50]


def rect(name: str) -> list[float]:
    x0, x1, yt, yb = BOX[name]
    return [x0 / W_PX, 1 - yb / H_PX, (x1 - x0) / W_PX, (yb - yt) / H_PX]


def plain_log_axis(axis, ticks=TICKS) -> None:
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def fit_region(region: str, line: str, r_hii: float) -> dict | None:
    path = NPZ_DIR / f"VSFORD_{region}_{line}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    i2 = list(d["orders"]).index(2)
    s2, s2_err, reps = d["S"][i2], d["S_err"][i2], d["S_jackknife_replicates"][:, i2]
    good = np.isfinite(s2) & (s2 > 0) & (s2_err > 0)
    ok = good & (d["counts"] >= MIN_PAIRS)
    sparse = ok.sum() < 2 * MIN_POINTS_PER_SEGMENT
    if sparse:
        ok = good & (d["counts"] >= SPARSE_MIN_PAIRS)
        if ok.sum() < MIN_POINTS_PER_SEGMENT:
            # jackknife errors can be all zero when every fibre falls in one block: unweighted fit instead
            ok = np.isfinite(s2) & (s2 > 0) & (d["counts"] >= SPARSE_MIN_PAIRS)
            if ok.sum() < MIN_POINTS_PER_SEGMENT:
                print(f"  {line:4s} {region:15s} skipped: {ok.sum()} bins with >= {SPARSE_MIN_PAIRS} pairs")
                return None
            r = 10 ** d["log_centers"][ok]
            x, y = np.log10(r), 0.5 * np.log10(s2[ok])
            coef, cov = np.polyfit(x, y, 1, cov=True)
            fit = select_fit(best_fit_by_segment_count(x, y, np.full_like(y, 0.05), 1, MIN_POINTS_PER_SEGMENT),
                             MIN_DELTA_BIC)
            print(f"  {line:4s} {region:15s} no jackknife errors: unweighted single fit on {ok.sum()} bins")
            return {
                "region": region, "line": line, "r_hii": r_hii, "r": r, "sig": 10**y,
                "sig_err": np.zeros_like(y), "fit": fit, "has_break": False, "sparse": True, "breaks_pc": [],
                "x_plot": r.max() / r_hii, "slopes": [float(coef[0])],
                "slope_errs": [float(np.sqrt(cov[0, 0]))], "delta_bic": np.nan,
            }
    r = 10 ** d["log_centers"][ok]
    x = np.log10(r)
    y = 0.5 * np.log10(s2[ok])
    ye = 0.5 * s2_err[ok] / (s2[ok] * np.log(10))
    max_seg = 1 if sparse else MAX_SEGMENTS
    fits = best_fit_by_segment_count(x, y, ye, max_seg, MIN_POINTS_PER_SEGMENT)
    forced = FORCE_SEGMENTS.get((region, line))
    fit = fits[forced] if forced in fits else select_fit(fits, MIN_DELTA_BIC)
    if forced:
        print(f"  {line:4s} {region:15s} forced to {forced} segments "
              f"(dBIC vs 1 segment = {fits[1].bic - fits[forced].bic:.1f})")
    delta_bic = fits[1].bic - min(f.bic for k, f in fits.items() if k > 1) if len(fits) > 1 else np.nan

    # jackknife slope errors with the selected breaks held fixed
    taus = break_log_positions(x, fit.break_indices)
    rep_slopes = []
    for rep in reps:
        c = rep[ok]
        if not np.all(np.isfinite(c) & (c > 0)):
            continue
        params, *_ = weighted_least_squares(x, 0.5 * np.log10(c), ye, taus)
        rep_slopes.append(np.cumsum(params[1:]))
    rep_slopes = np.asarray(rep_slopes)
    n = rep_slopes.shape[0]
    slope_err = np.sqrt((n - 1) / n * np.sum((rep_slopes - rep_slopes.mean(axis=0)) ** 2, axis=0))

    has_break = fit.n_segments > 1
    return {
        "region": region, "line": line, "r_hii": r_hii, "r": r, "sig": 10**y,
        "sig_err": 0.5 * s2_err[ok] / np.sqrt(s2[ok]), "fit": fit, "has_break": has_break,
        "sparse": sparse, "breaks_pc": [10**t for t in fit.break_log_r],
        "x_plot": (10 ** fit.break_log_r[0] if has_break else r.max()) / r_hii,
        "slopes": list(fit.slopes), "slope_errs": list(slope_err), "delta_bic": delta_bic,
    }


def model_curve(res: dict, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    fit = res["fit"]
    xx = np.linspace(np.log10(res["r"].min()), np.log10(res["r"].max()), n)
    cols = [np.ones_like(xx), xx] + [np.maximum(0.0, xx - t) for t in fit.break_log_r]
    return 10**xx, 10 ** (np.column_stack(cols) @ fit.params)


def kde_log(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(np.log10(values))(np.log10(grid)) if values.size >= 2 else np.zeros_like(grid)


def kde_lin(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(values)(grid) if values.size >= 2 else np.zeros_like(grid)


def write_csv(results: dict, out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "region", "has_break", "sparse_single_fit", "lbreak_pc", "lbreak_over_RHII_or_lmax",
                    "m", "m_err_jk", "m_err_cov", "all_slopes", "delta_bic"])
        for (region, line), res in sorted(results.items(), key=lambda t: (t[0][1], t[0][0])):
            w.writerow([line, region, int(res["has_break"]), int(res["sparse"]),
                        f"{res['breaks_pc'][0]:.3f}" if res["has_break"] else "",
                        f"{res['x_plot']:.4f}", f"{res['slopes'][0]:.4f}", f"{res['slope_errs'][0]:.4f}",
                        f"{res['fit'].slope_errs[0]:.4f}", ";".join(f"{s:.3f}" for s in res["slopes"]),
                        f"{res['delta_bic']:.1f}"])
            flag = "break" if res["has_break"] else ("sparse" if res["sparse"] else "none ")
            print(f"  {line:4s} {region:15s} {flag:6s} x={res['x_plot']:6.2f}  m={res['slopes'][0]:.3f}"
                  f" ± {res['slope_errs'][0]:.3f}  dBIC={res['delta_bic']:.1f}")


def draw_examples(fig, results: dict) -> None:
    for k, region in enumerate(EXAMPLES):
        ax = fig.add_axes(rect(f"left{k}"))
        handles = []
        for line in LINES:
            res = results.get((region, line))
            if res is None:
                continue
            c = LINE_COLOR[line]
            lag = res["r"] / res["r_hii"]
            ax.fill_between(lag, res["sig"] - res["sig_err"], res["sig"] + res["sig_err"], color=c, alpha=0.2, lw=0)
            ax.plot(lag, res["sig"], "o", color=c, ms=3.5, alpha=0.75, zorder=2)
            mx, my = model_curve(res)
            ax.plot(mx / res["r_hii"], my, color=c, lw=2.2, zorder=3)
            for b in res["breaks_pc"]:
                ax.axvline(b / res["r_hii"], color=c, ls=":", lw=1.6)
            handles.append(Line2D([], [], color=c, lw=2.5, label=rf"$m = {res['slopes'][0]:.2f}$"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axis(ax.xaxis)
        plain_log_axis(ax.yaxis, [1, 2, 3, 5, 10, 20, 30, 50])
        ax.tick_params(labelsize=13)
        ax.legend(handles=handles, loc="upper left", fontsize=9.5, framealpha=0.85, handlelength=2.0)
        ax.plot([0.745], [0.12], MARKER[region], transform=ax.transAxes, color="0.75", mec="k", ms=10, mew=1)
        ax.text(0.795, 0.12, LABEL.get(region, region), transform=ax.transAxes, fontsize=14, va="center")
        if k == 2:
            ax.set_xlabel(r"$\ell/R_{\rm HII}$")
        if k == 1:
            ax.set_ylabel(r"$\sqrt{S_2}=\langle|v(\vec r+\vec\ell)-v(\vec r)|^2\rangle^{1/2}$ [km/s]", labelpad=12)


def main() -> None:
    if STYLE.exists():
        plt.style.use(str(STYLE))
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"],
                         "axes.grid": True, "grid.alpha": 0.4})
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radii", choices=("adopted", "lvm-for-wiki", "lvm"), default="adopted",
                   help="adopted: manuscript_notes/adopted_hii_region_radii.csv (WISE + Wikipedia for "
                        "Flame/Orion/Rosette); lvm-for-wiki: same but LVM-catalogue radii for the Wikipedia ones; "
                        "lvm: LVM catalogue (vsf_module.get_hii_radius_pc) for every region")
    args = p.parse_args()
    out_name = OUT_NAME + {"adopted": "", "lvm-for-wiki": "_radiiWISE_LVMforWiki", "lvm": "_radiiLVM"}[args.radii]
    with RADIUS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    radii = {row["object"]: float(row["radius_pc"]) for row in rows}
    if args.radii != "adopted":
        sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "00_core"))
        from vsf_module import get_hii_radius_pc  # noqa: E402
        for row in rows:
            if args.radii == "lvm" or "wikipedia" in row["source_url"].lower():
                new = float(get_hii_radius_pc(f"LVM_{row['object']}.fits"))
                print(f"  R_HII {row['object']:15s} {radii[row['object']]:6.2f} -> {new:6.2f} pc (LVM catalogue)")
                radii[row["object"]] = new

    results = {}
    for line in LINES:
        for region in REGIONS:
            res = fit_region(region, line, radii[region])
            if res is not None:
                results[(region, line)] = res
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(results, OUT_DIR / f"{out_name}.csv")

    fig = plt.figure(figsize=(SCALE * W_PX / DPI, SCALE * H_PX / DPI))
    draw_examples(fig, results)

    ax_main = fig.add_axes(rect("main"))
    ax_top = fig.add_axes(rect("top"), sharex=ax_main)
    ax_right = fig.add_axes(rect("right"), sharey=ax_main)

    # ---- main scatter
    for (region, line), res in results.items():
        c = LINE_COLOR[line]
        filled = res["has_break"]
        ms = 15 if MARKER[region] == "*" else 11
        ax_main.errorbar(res["x_plot"], res["slopes"][0], yerr=res["slope_errs"][0], fmt="none",
                         ecolor=c, elinewidth=1.5, capsize=3, zorder=2)
        ax_main.plot(res["x_plot"], res["slopes"][0], MARKER[region], ms=ms, zorder=3,
                     mfc=c if filled else "white", mec="k" if filled else c, mew=0.9 if filled else 2.0)
    ax_main.axhline(1 / 3, color="0.45", ls="--", lw=1.5)
    ax_main.axvline(1, color="0.45", ls=":", lw=1.5)
    ax_main.set_xscale("log")
    plain_log_axis(ax_main.xaxis)
    ax_main.set_xlim(0.07, 13)
    ax_main.set_ylim(-0.03, 0.62)
    ax_main.set_xlabel(r"$\ell_{\rm break}/R_{\rm HII}$")
    ax_main.set_ylabel(r"$m\ (=\log_r\sqrt{S_2})$", fontsize=20)
    ax_main.tick_params(labelsize=17)
    line_handles = [Line2D([], [], marker="o", ls="", mfc=LINE_COLOR[l], mec="k", ms=11, label=LINE_LABEL[l])
                    for l in LINES]
    line_handles += [Line2D([], [], marker="o", ls="", mfc="0.6", mec="k", ms=11, label="break"),
                     Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", mew=2.0, ms=11,
                            label=r"no break ($\ell_{\max}$)")]
    ax_main.legend(handles=line_handles, loc="lower left", fontsize=15, framealpha=0.9)

    # ---- marginals (break-selected points only)
    gx = np.logspace(np.log10(0.07), np.log10(13), 400)
    gy = np.linspace(-0.03, 0.62, 400)
    all_x, all_m = [], []
    for line in LINES:
        sel = [r for (_, l), r in results.items() if l == line and r["has_break"]]
        xs = np.array([r["x_plot"] for r in sel])
        ms = np.array([r["slopes"][0] for r in sel])
        all_x.append(xs)
        all_m.append(ms)
        c = LINE_COLOR[line]
        px = kde_log(xs, gx)
        ax_top.fill_between(gx, px, color=c, alpha=0.5, lw=0)
        ax_top.plot(gx, px, color=c, lw=1.4)
        pm = kde_lin(ms, gy)
        ax_right.fill_betweenx(gy, pm, color=c, alpha=0.5, lw=0)
        ax_right.plot(pm, gy, color=c, lw=1.4)
    ax_top.plot(gx, kde_log(np.concatenate(all_x), gx), color="k", lw=1.6)
    ax_right.plot(kde_lin(np.concatenate(all_m), gy), gy, color="k", lw=1.6)
    ax_top.set_ylim(bottom=0)
    ax_right.set_xlim(left=0)
    ax_top.set_ylabel(r"$\mathcal{P}(\ell_{\rm break}/R_{\rm HII})$", fontsize=17)
    ax_top.tick_params(labelbottom=False, labelsize=17)
    ax_top.legend(handles=[Patch(facecolor="0.75", edgecolor="k", label="all lines combined")],
                  loc="upper left", fontsize=13, framealpha=0.9)
    ax_right.set_xlabel(r"$\mathcal{P}(m)$", fontsize=20)
    ax_right.tick_params(labelleft=False, labelsize=17)

    region_handles = [Line2D([], [], marker=MARKER[r], ls="", mfc="0.75", mec="k", mew=0.8,
                             ms=7 if MARKER[r] != "*" else 9.5, label=LABEL.get(r, r)) for r in REGIONS]
    # Region key in the free corner above the P(m) panel: use the largest line spacing that still fits.
    renderer = fig.canvas.get_renderer()
    key_bottom = ax_right.get_window_extent(renderer).y1 + 6
    for spacing in np.arange(1.0, -0.01, -0.05):
        key = fig.legend(handles=region_handles, loc="upper left", bbox_to_anchor=(2035 / W_PX, 0.998),
                         fontsize=11, frameon=False, handletextpad=0.5, labelspacing=spacing,
                         borderaxespad=0.0, borderpad=0.0, handlelength=1.0)
        if key.get_window_extent(renderer).y0 >= key_bottom:
            break
        key.remove()

    out = OUT_DIR / f"{out_name}.png"
    fig.savefig(out, dpi=DPI)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
