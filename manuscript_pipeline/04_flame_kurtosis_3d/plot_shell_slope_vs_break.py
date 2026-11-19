#!/usr/bin/env python3
"""Fitted sqrt(S2) slope vs where the fitted segment ends, for the synthetic shells.

Left:  fixed-window fits as in shell_projected_sqrtS2_slope_vs_width (seed-7 shell NPZs):
       slope of a single line from r_min to r_end, swept in r_end, one curve per width.
       Markers show the three r_end choices used before (0.25, 0.4, S2 peak).
Right: controlled segmented-fit experiment (mock_shell_segmented_breakpoints.py output):
       first-segment slope m vs first break, per seed (small) and median per width (big),
       with the left-panel curves recomputed from r_min = 0.1 overplotted (dashed).
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

from plot_shell_projected_vsf2_slope_vs_thickness import SHELL_DIR, load_s2, shell_paths_in_dir  # noqa: E402

STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"


def slope_to_end(r, y, r_min, r_end):
    sel = np.isfinite(y) & (r >= r_min) & (r <= r_end)
    if sel.sum() < 3:
        return np.nan
    return np.polyfit(np.log10(r[sel]), y[sel], 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock-csv", type=Path,
                   default=SHELL_DIR / "mock_shell_segmented_breaks_dBIC6_kmin2.csv")
    args = p.parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    curves = []
    for path in shell_paths_in_dir(SHELL_DIR):
        w, r, s2, _ = load_s2(path)
        y = 0.5 * np.log10(s2)
        ok = np.isfinite(y)
        m = ok & (r > 0.2) & (r < 1.0)
        r_peak = float(r[m][np.argmax(s2[m])])
        curves.append((w, r, y, float(r[ok].min()), r_peak))
    widths = np.array([c[0] for c in curves])

    cmap = plt.colormaps["viridis"]
    norm = matplotlib.colors.LogNorm(widths.min(), widths.max())
    ends = np.logspace(np.log10(0.12), np.log10(1.5), 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)

    # (a) fixed-window fits from the smallest lag, as in the sqrtS2-vs-width figure
    ax = axes[0]
    for w, r, y, rmin, rpk in curves:
        c = cmap(norm(w))
        ax.plot(ends, [slope_to_end(r, y, rmin, e) for e in ends], color=c, lw=1.5)
        for e, mk in [(0.25, "s"), (0.4, "D"), (rpk, "o")]:
            ax.plot(e, slope_to_end(r, y, rmin, e), mk, color=c, mec="k", ms=6)
    for mk, lab in [("s", r"$r_{\rm end}=0.25$"), ("D", r"$r_{\rm end}=0.4$"), ("o", r"$r_{\rm end}=$ S2 peak")]:
        ax.plot([], [], mk, color="w", mec="k", label=lab)
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"end of fitted range $r_{\rm end}/R_{\rm out}$")
    ax.set_ylabel(r"fitted slope of $\sqrt{S_2}$")
    ax.set_title(rf"Fixed window from $r_{{\min}}\approx{curves[0][3]:.3f}$ (seed 7)")
    ax.legend(fontsize=8, loc="upper right")

    # (b) segmented-fit experiment
    ax = axes[1]
    mock = np.genfromtxt(args.mock_csv, delimiter=",", names=True)
    det = mock["break_is_lower_limit"] == 0
    ax.scatter(mock["lbreak_over_Rout"][det], mock["m"][det], c=cmap(norm(mock["width"][det])),
               s=16, alpha=0.5, lw=0, zorder=2)
    ax.scatter(mock["lbreak_over_Rout"][~det], mock["m"][~det], c=cmap(norm(mock["width"][~det])),
               s=26, marker=">", alpha=0.5, lw=0, zorder=2)
    for w in np.unique(mock["width"]):
        s = det & np.isclose(mock["width"], w)
        if s.sum() >= 3:
            ax.errorbar(np.median(mock["lbreak_over_Rout"][s]), np.median(mock["m"][s]),
                        yerr=np.std(mock["m"][s]), fmt="o", ms=9, mfc=cmap(norm(w)), mec="k",
                        ecolor=cmap(norm(w)), capsize=2, zorder=4)
    for w, r, y, _, _ in curves:
        ax.plot(ends, [slope_to_end(r, y, 0.1, e) for e in ends], "--", color=cmap(norm(w)), lw=1, zorder=1)
    ax.plot([], [], "k--", lw=1, label=r"fixed window from $r_{\min}=0.1$ (seed 7)")
    ax.plot([], [], "o", color="gray", mec="k", ms=8, label="segmented fit: median per width")
    ax.plot([], [], ">", color="gray", alpha=0.5, label="no break selected (end of range)")
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell_{\rm break}/R_{\rm out}$  (segmented)  or  $r_{\rm end}$  (dashed)")
    ax.set_title(r"Controlled experiment: segmented fit, $\Delta$BIC=6, $r_{\min}=0.1$")
    ax.legend(fontsize=8, loc="upper right")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label=r"shell width $w/R_{\rm out}$", pad=0.01)

    out = SHELL_DIR / "shell_sqrtS2_slope_vs_break.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
