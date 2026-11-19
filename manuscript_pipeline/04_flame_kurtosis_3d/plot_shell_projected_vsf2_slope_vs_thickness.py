#!/usr/bin/env python3
"""Projected second-order VSF slope of the synthetic shells vs shell thickness.

Reads the x-projected shell NPZs written by generate_shell_thickness_grid.py
(same cube, fixed r_outer, varying r_inner) and uses the stored pair sums
S2(r) = c2 / cnt to measure

  * the local logarithmic slope d log S2 / d log r as a function of lag, and
  * power-law slopes fitted in fixed lag windows,

as a function of shell width. The intrinsic 3-D slope (from
shell_3d_vsf_kolmogorov_slopes.csv, if present) is overplotted for reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from plot_shell_kurtosis_slope_vs_thickness import FNAME_RE, SHELL_DIR, shell_paths_in_dir  # noqa: E402

STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
PIXEL = 2.4 / 159  # box_size / (n_grid - 1) of the default cube
FIT_WINDOWS = [(3 * PIXEL, 0.1), (0.1, 0.3), (0.3, 0.8)]
MIN_PAIRS = 20_000


def load_s2(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    width = float(d["r_outer"] - d["r_inner"])
    r = 10 ** d["log_centers"]
    cnt = d["cnt"]
    ok = cnt >= MIN_PAIRS
    s2 = np.where(ok, d["c2"] / np.where(cnt > 0, cnt, 1), np.nan)
    return width, r, s2, ok


def fit_slope(r: np.ndarray, s2: np.ndarray, lo: float, hi: float) -> float:
    sel = np.isfinite(s2) & (s2 > 0) & (r >= lo) & (r <= hi)
    if np.count_nonzero(sel) < 3:
        return np.nan
    return float(np.polyfit(np.log10(r[sel]), np.log10(s2[sel]), 1)[0])


def local_slope(r: np.ndarray, s2: np.ndarray, half_window: int = 2) -> np.ndarray:
    """Slope of a running linear fit in log-log over 2*half_window+1 bins."""
    lr, ls = np.log10(r), np.log10(s2)
    out = np.full_like(lr, np.nan)
    for i in range(lr.size):
        sl = slice(max(i - half_window, 0), i + half_window + 1)
        m = np.isfinite(ls[sl])
        if m.sum() >= 3:
            out[i] = np.polyfit(lr[sl][m], ls[sl][m], 1)[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=SHELL_DIR)
    args = parser.parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    paths = shell_paths_in_dir(args.shell_dir)
    data = [load_s2(p) for p in paths]
    widths = np.array([w for w, *_ in data])

    rows = []
    for w, r, s2, _ in data:
        rows.append([w] + [fit_slope(r, s2, lo, hi) for lo, hi in FIT_WINDOWS])
    rows = np.array(rows)

    out_csv = args.shell_dir / "shell_projected_vsf2_slopes.csv"
    header = "width," + ",".join(f"slope_{lo:.3f}_{hi:.3f}" for lo, hi in FIT_WINDOWS)
    np.savetxt(out_csv, rows, delimiter=",", header=header, comments="", fmt="%.5f")

    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, len(data)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    for (w, r, s2, _), c in zip(data, colors):
        ax.loglog(r, s2 / np.nanmax(s2), color=c, lw=1.4, label=f"{w:.3f}")
    ax.set_xlabel(r"lag r  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"$S_2(r)/\max S_2$")
    ax.set_title("Projected S2")
    ax.legend(title="width", fontsize=7, ncol=2)

    ax = axes[1]
    for (w, r, s2, _), c in zip(data, colors):
        ax.semilogx(r, local_slope(r, s2), color=c, lw=1.4)
        ax.axvline(w, color=c, ls=":", lw=0.8)
    for val, lab in [(5 / 3, r"5/3 ($r \ll$ depth)"), (2 / 3, r"2/3 ($r \gg$ depth)")]:
        ax.axhline(val, color="k", ls="--", lw=0.8)
        ax.text(0.02, val + 0.03, lab, fontsize=8)
    ax.set_ylim(-0.5, 2.2)
    ax.set_xlabel("lag r")
    ax.set_ylabel(r"local slope $d\log S_2/d\log r$")
    ax.set_title("Local slope (dotted: shell width)")

    ax = axes[2]
    for k, (lo, hi) in enumerate(FIT_WINDOWS):
        ax.semilogx(widths, rows[:, k + 1], "o-", label=f"fit {lo:.3f}–{hi:.2f}")
    ref = args.shell_dir / "shell_3d_vsf_kolmogorov_slopes.csv"
    if ref.exists():
        r3 = np.genfromtxt(ref, delimiter=",", names=True)
        ax.semilogx(r3["width"], r3["slope"], "k^--", label="intrinsic 3D")
    ax.axhline(5 / 3, color="gray", ls=":", lw=0.8)
    ax.axhline(2 / 3, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"shell width  [$R_{\rm out}=1$]")
    ax.set_ylabel("S2 slope")
    ax.set_title("Fitted projected S2 slope")
    ax.legend(fontsize=8)

    out_fig = args.shell_dir / "shell_projected_vsf2_slope_vs_thickness.png"
    fig.savefig(out_fig, dpi=200)
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    print(header)
    for row in rows:
        print(",".join(f"{v:.3f}" for v in row))

    make_single_fit_figure(data, widths, args.shell_dir, ref if ref.exists() else None)
    make_sqrt_s2_figure(paths, data, widths, args.shell_dir, ref if ref.exists() else None)


def per_layer_depth(path: Path) -> float:
    """Mean line-of-sight path through the shell per projected pixel, divided by 2 (front+back)."""
    counts = np.load(path)["shell_counts"]
    counts = counts[counts > 0]
    return float(np.mean(counts) * PIXEL / 2)


def make_sqrt_s2_figure(paths, data, widths, shell_dir: Path, ref: Path | None) -> None:
    """Slope of sqrt(S2) (= half the S2 slope) vs width, its derivative, and a depth/fit-window collapse."""
    n_fit = len(SINGLE_FIT_UPPER)
    slopes = np.full((len(data), n_fit), np.nan)
    errs = np.full_like(slopes, np.nan)
    rmin = np.full_like(slopes, np.nan)
    rmax = np.full_like(slopes, np.nan)
    for i, (w, r, s2, _) in enumerate(data):
        sig = np.sqrt(s2)
        for k, hi in enumerate(SINGLE_FIT_UPPER):
            coef, err, hi_val, sel = single_powerlaw_fit(r, sig, hi)
            slopes[i, k], errs[i, k] = coef[0], err
            rmin[i, k], rmax[i, k] = r[sel].min(), hi_val
    depth = np.array([per_layer_depth(p) for p in paths])
    lw = np.log10(widths)
    dslope = np.gradient(slopes, lw, axis=0)

    slope3d = None
    if ref is not None:
        r3 = np.genfromtxt(ref, delimiter=",", names=True)
        slope3d = 0.5 * r3["slope"]

    labels = [r"$r_{\min}\to$ S2 peak", r"$r_{\min}\to 0.4$", r"$r_{\min}\to 0.25$"]
    colors = ["C0", "C1", "C2"]
    rising = widths <= 0.21

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    # (a) slope vs width with linear-in-log fits on the rising part
    ax = axes[0]
    wfine = np.logspace(np.log10(widths.min()), np.log10(0.21), 50)
    for k in range(n_fit):
        c = np.polyfit(lw[rising], slopes[rising, k], 1, w=1 / errs[rising, k])
        plateau = np.mean(slopes[widths >= 0.2, k])
        ax.errorbar(widths, slopes[:, k], yerr=errs[:, k], fmt="o", color=colors[k], ms=4, capsize=2,
                    label=rf"{labels[k]}: ${c[1]:.2f}{c[0]:+.2f}\log_{{10}}w$")
        ax.plot(wfine, np.polyval(c, np.log10(wfine)), "--", color=colors[k], lw=1.2)
        ax.plot([0.2, widths.max()], [plateau, plateau], ":", color=colors[k], lw=1.2)
    if slope3d is not None:
        ax.plot(r3["width"], slope3d, "k^-", ms=4, lw=0.8, label="intrinsic 3D (1/2 of S2 slope)")
    ax.axhline(1 / 3, color="gray", ls=":", lw=0.8)
    ax.axhline(5 / 6, color="gray", ls=":", lw=0.8)
    ax.text(0.3, 5 / 6 - 0.025, r"5/6 (fully projected)", fontsize=8, color="gray", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"shell width $w$  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"fitted slope of $\sqrt{S_2}$")
    ax.set_title(r"$\sqrt{S_2}$ slope vs shell width")
    ax.legend(fontsize=7.5, loc="upper left")

    # (b) derivative, with predicted inflection where depth = log-centre of fit window
    ax = axes[1]
    for k in range(n_fit):
        ax.plot(widths, dslope[:, k], "o-", color=colors[k], ms=4, label=labels[k])
        centre = np.sqrt(np.median(rmin[:, k]) * np.median(rmax[:, k]))
        w_pred = np.interp(np.log10(centre), np.log10(depth), widths)
        ax.axvline(w_pred, color=colors[k], ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"shell width $w$  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"$d\,\mathrm{slope} / d\log_{10} w$")
    ax.set_title(r"Rate of change (dashed: depth $=\sqrt{r_{\min} r_{\max}}$)")
    ax.legend(fontsize=8)

    # (c) collapse onto fraction of the fit window (in log r) lying below the LOS depth
    ax = axes[2]
    for k in range(n_fit):
        frac = np.log(depth / rmin[:, k]) / np.log(rmax[:, k] / rmin[:, k])
        ax.plot(frac, slopes[:, k], "o-", color=colors[k], ms=4, label=labels[k])
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(1, color="gray", ls=":", lw=0.8)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$f = \log(D/r_{\min}) / \log(r_{\max}/r_{\min})$")
    ax.set_ylabel(r"fitted slope of $\sqrt{S_2}$")
    ax.set_title("Fraction of fit window below LOS depth")
    ax.legend(fontsize=8)

    out_fig = shell_dir / "shell_projected_sqrtS2_slope_vs_width.png"
    fig.savefig(out_fig, dpi=200)
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    for k in range(n_fit):
        centre = np.sqrt(np.median(rmin[:, k]) * np.median(rmax[:, k]))
        w_pred = np.interp(np.log10(centre), np.log10(depth), widths)
        print(f"{SINGLE_FIT_UPPER[k]}: r_min={np.median(rmin[:, k]):.3f} r_max={np.median(rmax[:, k]):.3f} "
              f"log-centre={centre:.3f} -> predicted inflection w={w_pred:.3f}, "
              f"max-derivative w={widths[np.argmax(dslope[:, k])]:.3f}")


SINGLE_FIT_UPPER = ["peak", 0.4, 0.25]


def single_powerlaw_fit(r, s2, hi):
    """Fit log S2 = a log r + b from the smallest valid lag up to r = hi."""
    ok = np.isfinite(s2) & (s2 > 0)
    if hi == "peak":
        m = ok & (r > 0.2) & (r < 1.0)
        hi = float(r[m][np.argmax(s2[m])])
    sel = ok & (r <= hi)
    coef, cov = np.polyfit(np.log10(r[sel]), np.log10(s2[sel]), 1, cov=True)
    return coef, float(np.sqrt(cov[0, 0])), hi, sel


def make_single_fit_figure(data, widths, shell_dir: Path, ref: Path | None) -> None:
    n_fit = len(SINGLE_FIT_UPPER)
    slopes = np.full((len(data), n_fit), np.nan)
    errs = np.full_like(slopes, np.nan)
    fits = []
    for i, (w, r, s2, _) in enumerate(data):
        row = []
        for k, hi in enumerate(SINGLE_FIT_UPPER):
            coef, err, hi_val, sel = single_powerlaw_fit(r, s2, hi)
            slopes[i, k], errs[i, k] = coef[0], err
            row.append((coef, hi_val, sel))
        fits.append(row)

    lw = np.log10(widths)
    dslope = np.gradient(slopes, lw, axis=0)  # d alpha / d log10(width)

    labels = [r"$r_{\min}\to$ S2 peak", r"$r_{\min}\to 0.4$", r"$r_{\min}\to 0.25$"]
    fit_colors = ["C0", "C2", "C3"]
    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, len(data)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    show = [0, len(data) // 3, 2 * len(data) // 3, len(data) - 1]
    for idx, i in enumerate(show):
        w, r, s2, _ = data[i]
        off = 10 ** (-0.35 * idx)
        ax.loglog(r, s2 * off, "o", ms=3, color=colors[i], label=f"w = {w:.3f}")
        coef, hi_val, sel = fits[i][0]
        rr = r[sel]
        ax.loglog(rr, off * 10 ** np.polyval(coef, np.log10(rr)), "-", color="k", lw=1)
        ax.text(rr[-1] * 1.1, off * 10 ** np.polyval(coef, np.log10(rr[-1])),
                rf"$\alpha={coef[0]:.2f}$", fontsize=8, va="center")
    ax.set_xlabel(r"lag r  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"$S_2(r)$  (offset)")
    ax.set_title(r"Single power-law fits ($r_{\min}\to$ peak)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for k in range(n_fit):
        ax.errorbar(widths, slopes[:, k], yerr=errs[:, k], fmt="o-", color=fit_colors[k],
                    ms=4, capsize=2, label=labels[k])
    if ref is not None:
        r3 = np.genfromtxt(ref, delimiter=",", names=True)
        ax.plot(r3["width"], r3["slope"], "k^--", ms=4, label="intrinsic 3D")
    ax.axhline(5 / 3, color="gray", ls=":", lw=0.8)
    ax.axhline(2 / 3, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"shell width $w$  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"fitted S2 slope $\alpha$")
    ax.set_title("Fitted slope vs shell width")
    ax.legend(fontsize=8)

    ax = axes[2]
    for k in range(n_fit):
        ax.plot(widths, dslope[:, k], "o-", color=fit_colors[k], ms=4, label=labels[k])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"shell width $w$  [$R_{\rm out}=1$]")
    ax.set_ylabel(r"$d\alpha / d\log_{10} w$")
    ax.set_title("Change in slope per dex of width")
    ax.legend(fontsize=8)

    out_fig = shell_dir / "shell_projected_vsf2_single_fit_slope_vs_width.png"
    fig.savefig(out_fig, dpi=200)
    out_csv = shell_dir / "shell_projected_vsf2_single_fit_slopes.csv"
    tags = ["peak", "0p40", "0p25"]
    header = "width," + ",".join(f"slope_{t},err_{t},dslope_dlogw_{t}" for t in tags)
    table = np.column_stack(
        [widths] + [np.column_stack([slopes[:, k], errs[:, k], dslope[:, k]]) for k in range(n_fit)]
    )
    np.savetxt(out_csv, table, delimiter=",", header=header, comments="", fmt="%.5f")
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    print(header)
    for row in table:
        print(",".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
