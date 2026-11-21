"""Fit the observed K(l) of each region/line with the closed-form POINT-SAMPLE K(l; Delta, lambda) (comparison).

Model: analytic K(l; lambda) of manuscript_notes/kurtosis_K_lambda_derivation.md and
04_flame_kurtosis_3d/analytic_bend_slope.py (thin shell, Gaussian Kolmogorov turbulence, no inner cutoff):

    K_pt(l) = P [3 N2 + 24 chi q2 lam^2 + 4 q4 lam^4] / [N1 + 2 q2 lam^2]^2
    P, q2, q4   closed-form pair-count geometry of the shell (depend on l, Delta)
    N1, N2      D- and D^2-weighted pair counts, chi = int_0^1 mu D(sqrt(4 mu^2 + l^2)) dmu
    D(s)        k_max -> inf structure function with k_min = 2 pi / L_out, L_out = FOV (as the centroid fit)

This is a POINT-SAMPLE model: it ignores that one LVM fibre is a line-of-sight emission-weighted centroid
(fit_centroid_joint_grid.py models that). It is a comparison, not a replacement.

Fit: K only, per line independently, on a (Delta, lambda) grid; lambda = v_bulk / v_turb (point model, no
projection factor). Delta <= 0.6: the thin-shell closed forms fail for Delta >~ 2/3 (derivation note).
Covariance = K block of the shrunk jackknife covariance + model error (as in the joint fits).
Noise: fixed white noise N2 = NOISE_C^2 x catalogue N2 (same calibration as the centroid fit):
    S2_sig = v_turb^2 s2_pt,  s2_pt = (N1 + 2 q2 lam^2) / P,
    K_obs = (K_pt S2_sig^2 + 6 S2_sig N2 + 3 N2^2) / (S2_sig + N2)^2,
with v_turb set at each (Delta, lambda) so that the model's mean ln S2 over the bins equals the observed one
(the S2 amplitude only: the point model cannot reproduce the observed S2(l) slope, so S2 is not fitted).
Physical map: v_bulk = lambda x v_turb(Delta, lambda), with v_turb from the S2 amplitude as above.

Outputs (Figures/manuscript/08_shell_inference/analytic_point_K/):
  analytic_point_K_summary.csv, <region>_<LINE>_map.npz, analytic_point_K_curves.png,
  analytic_point_K_maps_physical.png (top: centroid joint fit, bottom: this fit; selected regions)

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_analytic_point_K.py [--region R] [--centroid-run centroid_joint_v_a0_lammin1]
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
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core"), str(HERE.parent / "04_flame_kurtosis_3d")]
import analytic_bend_slope as ab  # noqa: E402
from fit_kurtosis_formula import DATA_DIR  # noqa: E402
from fit_kurtosis_s2_joint_grid import NOISE_C, load, prange  # noqa: E402
from gridfree_kolmogorov import I_INF, structure_function  # noqa: E402
from plot_centroid_joint_selected_physical import REGIONS, load_adopted_radii  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

OUT_BASE = Path("Figures/manuscript/08_shell_inference")
OUT_DIR = OUT_BASE / "analytic_point_K"
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}
D_GRID = np.geomspace(0.005, 0.6, 26)  # thin-shell formula valid for Delta <~ 2/3
LAM_GRID = np.geomspace(0.01, 30.0, 90)
LNV2_GRID = np.log(np.geomspace(0.2, 40.0, 140) ** 2)  # ln v_turb^2, v in km/s (as the centroid fit)


# ------------------------------------------------------------------ analytic terms for a given outer scale
def terms(ell, delta, kmin, n=3000):
    """Arrays (P, q2, q4, N1, N2, chi) on ell for one Delta (analytic_bend_slope with k_min as a parameter)."""
    A = (4 / 3) * I_INF * kmin ** (2 / 3)
    P = ab.pair_count(np.asarray(ell, float), delta)
    q2 = np.array([ab.q2(l) for l in ell])
    q4 = np.array([ab.q4(l) for l in ell])
    N = np.zeros((2, len(ell)))
    mu = np.linspace(0, 1, 2001)
    chi = np.empty(len(ell))
    for i, l in enumerate(ell):
        u0 = np.arccosh(max(delta / l, 1.0))
        u = np.linspace(u0, np.arccosh(2 / l), n)
        Du = structure_function(l * np.cosh(u), kmin)
        v = np.linspace(0, u0, n)
        s = l * np.cosh(v)
        for k in (1, 2):
            N[k - 1, i] = np.trapezoid(Du ** k, u)
            if l < delta:  # same-wall slab pairs, D = A s^(2/3)
                N[k - 1, i] += 2 / delta ** 2 * np.trapezoid((delta - s / 2) * (A * s ** (2 / 3)) ** k * s, v)
        chi[i] = np.trapezoid(mu * structure_function(np.sqrt(4 * mu ** 2 + l ** 2), kmin), mu)
    return P, q2, q4, N[0], N[1], chi


def model_grid(ell, kmin):
    """K_pt and s2_pt (units of v_turb^2) on (D_GRID, LAM_GRID, ell)."""
    lam2 = LAM_GRID[:, None] ** 2
    K = np.empty((len(D_GRID), len(LAM_GRID), len(ell)))
    s2 = np.empty_like(K)
    for j, d in enumerate(D_GRID):
        P, q2, q4, N1, N2, chi = terms(ell, d, kmin)
        den = N1 + 2 * q2 * lam2
        K[j] = P * (3 * N2 + 24 * chi * q2 * lam2 + 4 * q4 * lam2 ** 2) / den ** 2
        s2[j] = den / P
    return K, s2


# ------------------------------------------------------------------ fit
def fit_line(region, X):
    d = load(region, X, "overlap")
    if d is None:
        return None
    n = len(d["ell"])
    if d["ell"].max() >= 2:
        raise ValueError(f"{region} {X}: bins at l >= 2 r_HII are outside the shell model")
    kmin = 2 * np.pi / d["L_out"]
    Kpt, s2pt = model_grid(d["ell"], kmin)
    N2 = d["N2"]
    Linv = np.linalg.inv(np.linalg.cholesky(d["C0"][:n, :n]))
    # v_turb at each (Delta, lam): the model's mean ln S2 over the bins matches the observed one (amplitude only)
    target = np.mean(d["lnS2"])
    mis = np.stack([np.abs(np.mean(np.log(np.exp(u) * s2pt + N2), -1) - target) for u in LNV2_GRID])
    iv = mis.argmin(0)
    vturb = np.exp(LNV2_GRID[iv] / 2)
    S2s = vturb[..., None] ** 2 * s2pt
    Ko = (Kpt * S2s ** 2 + 6 * S2s * N2 + 3 * N2 ** 2) / (S2s + N2) ** 2
    chi2 = (((d["K"] - Ko) @ Linv.T) ** 2).sum(-1)
    iD, iL = np.unravel_index(np.argmin(chi2), chi2.shape)
    Kbest = Ko
    dmap = chi2 - chi2.min()
    d_lo, d_hi, d_ok = prange(dmap.min(1), D_GRID)
    l_lo, l_hi, l_ok = prange(dmap.min(0), LAM_GRID)
    flags = [f for f, ok in (("Delta 1-sigma hits grid edge", d_ok), ("lambda 1-sigma hits grid edge", l_ok)) if not ok]
    if chi2[iD, iL] > n + 3 * np.sqrt(2 * n):
        flags.append("poor fit")
    row = {"region": region, "line": X, "n_bins": n, "L_out_over_rhii": d["L_out"], "noise_N2": N2,
           "best_delta": D_GRID[iD], "delta_1sig_lo": d_lo, "delta_1sig_hi": d_hi,
           "best_lambda": LAM_GRID[iL], "lambda_1sig_lo": l_lo, "lambda_1sig_hi": l_hi,
           "v_turb_from_S2": vturb[iD, iL], "v_bulk_at_best": LAM_GRID[iL] * vturb[iD, iL],
           "chi2_K": chi2[iD, iL], "n_data": n, "flags": "; ".join(flags)}
    return row, dict(data=d, Kfit=Kbest[iD, iL], dmap=dmap, vturb=vturb)


# ------------------------------------------------------------------ figures
def plot_curves(store):
    regions = list(store)
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.4 * nr), squeeze=False)
    for ax, region in zip(axes.flat, regions):
        for X, (row, f) in store[region].items():
            c, d = LINE_COLORS[X], f["data"]
            ax.errorbar(d["ell"], d["K"], d["sdK"], fmt="o", ms=3.5, color=c, capsize=2)
            ax.plot(d["ell"], f["Kfit"], LINESTYLES[X], color=c, lw=1.5,
                    label=fr"{LINE_LABELS[X]} $\Delta$={row['best_delta']:.2g} $\lambda$={row['best_lambda']:.2g} "
                          fr"$\chi^2$={row['chi2_K']:.0f}/{row['n_data']}")
        ax.axhline(3, color="0.5", ls="--", lw=0.8)
        ax.set(xscale="log", title=region.replace("_", " "), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K$")
        ax.legend(fontsize=6.5)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    fig.suptitle(r"Observed $K(\ell)$ with the best analytic point-sample fit ($k_{\max}\to\infty$; no sightline averaging)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "analytic_point_K_curves.png", dpi=150)
    plt.close(fig)


def _panel(ax, xx, yy, dm, c, ls, xb, yb, xs, ys):
    inside = dm <= 6.18
    xs += [xx[inside].min(), xx[inside].max()]
    ys += [yy[inside].min(), yy[inside].max()]
    ax.contourf(xx, yy, dm, levels=[0, 2.30], colors=[c], alpha=0.25)
    ax.contour(xx, yy, dm, levels=[2.30, 6.18], colors=[c], linestyles=[ls], linewidths=[1.8, 0.9])
    ax.plot(xb, yb, "*", ms=14, color=c, mec="white", mew=0.7)


def _finish(ax, xs, ys, r_hii):
    ax.axvline(r_hii, color="0.3", ls="--", lw=1.2)
    ax.set(xscale="log", yscale="log")
    ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
    ylo, yhi = min(ys) / 1.1, max(ys) * 1.1
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(FixedLocator([t for t in (0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100) if ylo <= t <= yhi]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_physical(store, centroid_run):
    """Top row: centroid joint fit (fit_centroid_joint_grid.py); bottom row: analytic point-sample K fit."""
    run = OUT_BASE / centroid_run
    with open(run / "joint_summary.csv", encoding="utf-8") as f:
        csumm = {(r["region"], r["line"]): r for r in csv.DictReader(f)}
    adopted = load_adopted_radii()
    regions = [r for r in REGIONS if r in store]
    fig, axes = plt.subplots(2, len(regions), figsize=(4.6 * len(regions), 7.6), squeeze=False)
    for col, region in enumerate(regions):
        lines = list(store[region])
        r_hii = adopted.get(region) or float(np.load(DATA_DIR / f"{region}_{lines[0]}_formula_data.npz")["r_hii_pc"])
        # centroid
        ax, xs, ys = axes[0, col], [r_hii], []
        for X in [X for X in LINES if (run / f"{region}_{X}_map.npz").exists()]:
            m, s = np.load(run / f"{region}_{X}_map.npz"), csumm[(region, X)]
            v = float(s["v_turb"])
            xx, yy = np.meshgrid(m["delta_grid"] * r_hii, m["lambda_grid"] * v, indexing="ij")
            _panel(ax, xx, yy, m["dm2lnL"], LINE_COLORS[X], LINESTYLES[X], float(s["best_delta"]) * r_hii,
                   float(s["best_lambda_eff"]) * v, xs, ys)
        _finish(ax, xs, ys, r_hii)
        ax.set_title(f"{region.replace('_', ' ')}: centroid model, K + S$_2$", fontsize=11)
        # analytic point K
        ax2, xs2, ys2 = axes[1, col], [r_hii], []
        for X, (row, f) in store[region].items():
            xx = (D_GRID * r_hii)[:, None] * np.ones(len(LAM_GRID))[None]
            yy = LAM_GRID[None, :] * f["vturb"]  # v_bulk = lam x v_turb(Delta, lam) from the S2 amplitude
            _panel(ax2, xx, yy, f["dmap"], LINE_COLORS[X], LINESTYLES[X], row["best_delta"] * r_hii,
                   row["v_bulk_at_best"], xs2, ys2)
            ax2.plot([], [], color=LINE_COLORS[X], ls=LINESTYLES[X], lw=1.8, label=LINE_LABELS[X])
        _finish(ax2, xs2, ys2, r_hii)
        ax2.set_title(f"{region.replace('_', ' ')}: analytic point K only", fontsize=11)
        ax2.text(0.97, 0.04, r"$v_{\rm bulk}=\lambda v_{\rm turb}$, $v_{\rm turb}$ from mean $S_2$", transform=ax2.transAxes, ha="right", fontsize=9)
        ax2.set_xlabel(r"$\Delta r_{\rm shell}$ [pc]")
    for r in range(2):
        axes[r, 0].set_ylabel(r"$v_{\rm bulk}$ (km/s)")
    axes[1, 0].legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "analytic_point_K_maps_physical.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None)
    ap.add_argument("--centroid-run", default="centroid_joint_v_a0_lammin1")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    regions = sorted({p.name.rsplit("_", 3)[0] for p in DATA_DIR.glob("*_formula_data.npz")})
    if args.region:
        regions = [args.region]
    rows, store = [], {}
    for region in regions:
        for X in LINES:
            out = fit_line(region, X)
            if out is None:
                continue
            row, f = out
            rows.append(row)
            store.setdefault(region, {})[X] = (row, f)
            np.savez(OUT_DIR / f"{region}_{X}_map.npz", delta_grid=D_GRID, lambda_grid=LAM_GRID, dchi2=f["dmap"],
                     v_turb=f["vturb"])
            print(f"{region:15s} {X:4s} Delta={row['best_delta']:.3f} [{row['delta_1sig_lo']:.3f},{row['delta_1sig_hi']:.3f}] "
                  f"lam={row['best_lambda']:.2f} [{row['lambda_1sig_lo']:.2f},{row['lambda_1sig_hi']:.2f}] "
                  f"v_turb(S2)={row['v_turb_from_S2']:.2f} chi2={row['chi2_K']:.1f}/{row['n_data']} {row['flags']}", flush=True)
    suffix = "" if args.region is None else f"_{args.region}"
    with open(OUT_DIR / f"analytic_point_K_summary{suffix}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(rows[0]))
        for r in rows:
            w.writerow([f"{v:.5g}" if isinstance(v, (float, np.floating)) else v for v in r.values()])
    if args.region is None:
        plot_curves(store)
        plot_physical(store, args.centroid_run)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
