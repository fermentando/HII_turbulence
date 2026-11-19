#!/usr/bin/env python3
"""Pass synthetic projected shells through the observational segmented-VSF fit.

For several turbulence seeds and the standard shell-width grid (fixed R_out = 1):
  1. build the x-projected <v_x> map of the shell,
  2. compute S2(l) exactly over all pixel pairs with FFT correlations,
  3. bin in log l (0.05 dex, like the observed VSFs) from --r-min to the disk diameter,
  4. fit log10 sqrt(S2) with the continuous segmented power law of
     03_slopes_turnover/fit_segmented_vsf_breakpoints.py (max 3 segments,
     >= 5 points/segment, Delta-BIC threshold 6), using the seed-to-seed scatter
     of each bin as its error,
  5. record slope of the first segment (m) and first break (l_break / R_out);
     single-segment fits are reported as lower limits at the largest lag.

Optionally overplots the digitized observed (l_break/R_HII, m) points.
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
sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "03_slopes_turnover"))

from fit_segmented_vsf_breakpoints import best_fit_by_segment_count, select_fit  # noqa: E402
from generate_shell_thickness_grid import N_WIDTHS, WIDTH_MAX, WIDTH_MIN, width_grid  # noqa: E402
from projected_shell_kurtosis import shell_projected_velocity  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402

OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions" / "synthetic_projected_shell"
STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
N_GRID = 160
BOX_SIZE = 2.4


def masked_s2_by_lag(v: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact pair-summed S2 and pair counts for every 2-D lag, via zero-padded FFTs."""
    n = v.shape[0]
    m = mask.astype(float)
    vm = np.where(mask, v, 0.0)
    shape = (2 * n, 2 * n)
    F = lambda a: np.fft.rfft2(a, s=shape)  # noqa: E731
    corr = lambda a, b: np.fft.irfft2(np.conj(F(a)) * F(b), s=shape)  # noqa: E731
    pairs = corr(m, m)
    sum_sq = corr(m, vm**2) + corr(vm**2, m) - 2.0 * corr(vm, vm)
    return sum_sq, pairs


def vx_field(n_grid: int, box_size: float, rng: np.random.Generator, kmin_factor: float,
             kmax_fraction: float = 0.35) -> np.ndarray:
    """v_x only, identical to the first component of kolmogorov_velocity_grid for the same rng."""
    # Real FFTs in single precision keep a ~670^3 cube at a few GB.
    import scipy.fft as sfft

    kmin = kmin_factor * 2.0 * np.pi / box_size
    kmax = kmax_fraction * np.pi * n_grid / box_size
    noise = rng.normal(size=(n_grid,) * 3).astype(np.float32)
    modes = sfft.rfftn(noise, workers=-1); del noise
    f_full = (np.fft.fftfreq(n_grid, d=box_size / n_grid) * 2.0 * np.pi).astype(np.float32)
    f_half = (np.fft.rfftfreq(n_grid, d=box_size / n_grid) * 2.0 * np.pi).astype(np.float32)
    for i in range(n_grid):  # slab-wise to avoid a full-size k array
        k = np.sqrt(f_full[i] ** 2 + f_full[:, None] ** 2 + f_half[None, :] ** 2)
        amp = np.where((k >= kmin) & (k <= kmax), np.maximum(k, 0.5 * kmin) ** (-11.0 / 6.0), 0.0)
        modes[i] *= amp.astype(np.float32)
    field = sfft.irfftn(modes, s=(n_grid,) * 3, workers=-1); del modes
    field -= field.mean(dtype=np.float64)
    field /= field.std(dtype=np.float64)
    return field


class ShellProjector:
    """Project v_x along x through spherical shells (r_inner <= r <= r_out), cropped to |y|,|z| <= r_out."""

    def __init__(self, n_grid: int, box_size: float, r_out: float = 1.0):
        coords = np.linspace(-box_size / 2, box_size / 2, n_grid)
        keep = np.abs(coords) <= r_out + 1e-9
        self.sl = slice(np.argmax(keep), len(keep) - np.argmax(keep[::-1]))
        c = coords[self.sl]
        self.radius = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
        self.r_out = r_out

    def __call__(self, vx: np.ndarray, r_inner: float) -> tuple[np.ndarray, np.ndarray]:
        sub = vx[self.sl, self.sl, self.sl]
        shell = (self.radius >= r_inner) & (self.radius <= self.r_out)
        counts = shell.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            proj = np.where(shell, sub, 0.0).sum(axis=0) / counts
        return np.nan_to_num(proj), counts > 0


def binned_s2(v, mask, pixel, edges):
    sum_sq, pairs = masked_s2_by_lag(v, mask)
    n = v.shape[0]
    idx = np.fft.fftfreq(2 * n, d=1.0 / (2 * n))
    lx, ly = np.meshgrid(idx, idx, indexing="ij")
    lag = np.sqrt(lx**2 + ly**2) * pixel
    good = pairs > 0.5
    which = np.digitize(lag[good], edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    num = np.bincount(which[ok], weights=sum_sq[good][ok], minlength=len(edges) - 1)
    den = np.bincount(which[ok], weights=pairs[good][ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, den / 2.0  # S2, number of unordered pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seeds", type=int, default=16)
    p.add_argument("--r-min", type=float, default=0.1, help="smallest lag fitted [R_out]")
    p.add_argument("--dlog", type=float, default=0.05)
    p.add_argument("--min-delta-bic", type=float, default=6.0)
    p.add_argument("--max-segments", type=int, default=3)
    p.add_argument("--min-points-per-segment", type=int, default=5)
    p.add_argument("--min-pairs", type=float, default=2000)
    p.add_argument("--observed-csv", type=Path, default=None,
                   help="CSV with columns line,object,lbreak_over_RHII,m,break_is_lower_limit")
    p.add_argument("--kmin-factor", type=float, default=2.0,
                   help="outer driving scale L_out = box / kmin_factor")
    p.add_argument("--box-size", type=float, default=BOX_SIZE, help="box side in units of R_out")
    p.add_argument("--n-grid", type=int, default=N_GRID,
                   help="cells per side; keep (n_grid-1)/box fixed to keep the pixel/cutoff scale")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    pixel = args.box_size / (args.n_grid - 1)
    projector = ShellProjector(args.n_grid, args.box_size)
    l_out = args.box_size / args.kmin_factor
    print(f"box={args.box_size} R_out, n={args.n_grid}, pixel={pixel:.4f}, L_out={l_out:.2f} R_out", flush=True)
    edges = 10 ** np.arange(np.log10(args.r_min), np.log10(2.0) + 1e-9, args.dlog)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = width_grid(WIDTH_MIN, WIDTH_MAX, N_WIDTHS)

    # S2 curves: [seed, width, bin]
    s2 = np.full((args.n_seeds, len(widths), len(centers)), np.nan)
    npairs = np.zeros_like(s2)
    for s in range(args.n_seeds):
        vx = vx_field(args.n_grid, args.box_size, np.random.default_rng(1000 + s), args.kmin_factor)
        for j, w in enumerate(widths):
            proj, mask = projector(vx, 1.0 - w)
            s2[s, j], npairs[s, j] = binned_s2(proj, mask, pixel, edges)
        print(f"seed {s + 1}/{args.n_seeds} done", flush=True)

    y_all = 0.5 * np.log10(s2)  # log10 sqrt(S2)
    y_err = np.nanstd(y_all, axis=0, ddof=1)  # seed-to-seed scatter per (width, bin)

    rows = []
    for j, w in enumerate(widths):
        for s in range(args.n_seeds):
            y = y_all[s, j]
            e = y_err[j]
            ok = np.isfinite(y) & np.isfinite(e) & (e > 0) & (npairs[s, j] >= args.min_pairs)
            x = np.log10(centers[ok])
            fits = best_fit_by_segment_count(x, y[ok], e[ok], args.max_segments, args.min_points_per_segment)
            sel = select_fit(fits, args.min_delta_bic)
            if sel.n_segments > 1:
                lb, limit = 10 ** sel.break_log_r[0], 0
            else:
                lb, limit = 10 ** x.max(), 1
            rows.append((w, s, sel.n_segments, lb, sel.slopes[0], sel.slope_errs[0], limit))
    rows = np.array(rows)

    tag = f"_{args.tag}" if args.tag else ""
    out_csv = OUT_DIR / f"mock_shell_segmented_breaks_dBIC{args.min_delta_bic:g}{tag}.csv"
    np.savetxt(out_csv, rows, delimiter=",", fmt="%.5f",
               header="width,seed,n_segments,lbreak_over_Rout,m,m_err,break_is_lower_limit", comments="")

    # ---- plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    obs_col = {"NII": "#8B2E57", "SII": "#D81B60", "SIII": "#F39C32"}
    if args.observed_csv and args.observed_csv.exists():
        obs = np.genfromtxt(args.observed_csv, delimiter=",", names=True, dtype=None, encoding=None)
        for L, c in obs_col.items():
            sl = obs["line"] == L
            det, lim = sl & (obs["break_is_lower_limit"] == 0), sl & (obs["break_is_lower_limit"] == 1)
            ax.scatter(obs["lbreak_over_RHII"][det], obs["m"][det], c=c, edgecolor="k", s=55,
                       marker="s", zorder=4, label=f"observed [{L}]")
            ax.scatter(obs["lbreak_over_RHII"][lim], obs["m"][lim], c=c, edgecolor="k", s=55,
                       marker=">", alpha=0.6, zorder=4)

    cmap = plt.colormaps["viridis"]
    norm = matplotlib.colors.LogNorm(widths.min(), widths.max())
    det = rows[:, 6] == 0
    ax.scatter(rows[det, 3], rows[det, 4], c=cmap(norm(rows[det, 0])), s=14, alpha=0.55, lw=0, zorder=2)
    ax.scatter(rows[~det, 3], rows[~det, 4], c=cmap(norm(rows[~det, 0])), s=22, marker=">",
               alpha=0.55, lw=0, zorder=2)
    for w in widths:  # median per width (detections only)
        sel = det & np.isclose(rows[:, 0], w)
        if sel.sum() >= 3:
            ax.errorbar(np.median(rows[sel, 3]), np.median(rows[sel, 4]),
                        yerr=np.std(rows[sel, 4]), fmt="o", ms=9, mfc=cmap(norm(w)), mec="k",
                        ecolor=cmap(norm(w)), capsize=2, zorder=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r"model shell width $w/R_{\rm out}$")
    ax.axhline(1 / 3, color="gray", ls="--", lw=1)
    ax.axvline(1, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell_{\rm break}/R$  (model: $R_{\rm out}$; data: $R_{\rm HII}$)")
    ax.set_ylabel(r"$m$ = slope of $\sqrt{S_2}$ below the break")
    ax.set_title(rf"Mock shells, $\Delta$BIC={args.min_delta_bic:g}, $r_{{\min}}={args.r_min:g}$, "
                 rf"$L_{{\rm out}}={l_out:.2f}\,R_{{\rm out}}$", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for j, w in enumerate(widths):
        med = np.nanmedian(y_all[:, j], axis=0)
        ax.fill_between(centers, 10 ** (med - y_err[j]), 10 ** (med + y_err[j]), color=cmap(norm(w)), alpha=0.2, lw=0)
        ax.plot(centers, 10 ** med, color=cmap(norm(w)), lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell / R_{\rm out}$")
    ax.set_ylabel(r"$\sqrt{S_2}$  (median over seeds, $\pm1\sigma$)")
    ax.set_title(f"Model curves ({args.n_seeds} seeds)")

    out_fig = OUT_DIR / f"mock_shell_segmented_breaks_dBIC{args.min_delta_bic:g}{tag}.png"
    fig.savefig(out_fig, dpi=180)
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}\nWrote {out_csv.relative_to(REPO_ROOT)}")

    print("width  frac_with_break  median_lbreak  median_m  std_m  n_seg_counts(1/2/3)")
    for w in widths:
        sel = np.isclose(rows[:, 0], w)
        d = sel & det
        segs = [int(np.sum(rows[sel, 2] == k)) for k in (1, 2, 3)]
        print(f"{w:.3f}  {d.sum() / sel.sum():.2f}  "
              f"{np.median(rows[d, 3]) if d.any() else np.nan:.3f}  "
              f"{np.median(rows[d, 4]) if d.any() else np.nan:.3f}  "
              f"{np.std(rows[d, 4]) if d.any() else np.nan:.3f}  {segs}")


if __name__ == "__main__":
    main()
