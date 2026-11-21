#!/usr/bin/env python3
"""Analytic K(l; lambda) at all l, and the bend and post-bend slope it predicts.

Extends the plateau formula (manuscript_notes/geometric_averages/00_plateau_kurtosis_summary.md) from
l = 0 to any sky separation l (thin shell, no inner cutoff, outer scale k_min):

    K(l; lam) = P(l) [3 N2(l) + 24 chi(l) q2(l) lam^2 + 4 q4(l) lam^4] / [N1(l) + 2 q2(l) lam^2]^2

  P(l)   pair count (= thin-shell bulk curve K_b), closed form, slab (s < Delta) + 1/s (s > Delta) pieces:
           P = arccosh(2/l) + [3/2 sqrt(1-y^2) - (1 + y^2/2) arccosh(1/y)] * (y < 1),   y = l/Delta
  q2, q4 large-l bulk corrections (disc cross-correlations of mu0^a), elementary closed forms; -> 1 at l -> 0
  N_k(l) D^k-weighted pair count: int_{arccosh(Delta/l)}^{arccosh(2/l)} D(l cosh u)^k du + slab term (l < Delta)
  chi(l) int_0^1 mu0 D(sqrt(4 mu0^2 + l^2)) dmu0  (~1; opposite-wall pairs)

At l = 0 it is exactly the plateau formula; at lam -> inf it is K_b = P q4 / q2^2.

Bend and slope: the same broken-line ("hinge") fit as fit_kurtosis_curve_hinge.py,
    K = K_plat + S max(0, x - x_b),  x = log10(l / r_HII),
applied to (i) the grid-free simulations (8 seeds, odd terms removed; single-seed scatter as the error bar),
(ii) the analytic K(l) on the same l bins. Also shown: the leading-order closed forms
    x_b - log10 Delta = log10(2 e^{-3/2}) = -0.350,   S = -ln10 * F_d(lam),
    F_d = [3 J2 + 4 lam^2 (lam^2 + 6)] / (J + 2 lam^2)^2.

Output (gridfree_turbulence_test/): analytic_bend_slope.png, analytic_bend_slope.csv, table on stdout.
Run (numpy + matplotlib):
    python manuscript_pipeline/04_flame_kurtosis_3d/analytic_bend_slope.py
"""
from __future__ import annotations

import csv
import sys
from math import gamma, pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import test_gridfree_kolmogorov_model as tg  # noqa: E402
from gridfree_kolmogorov import I_INF, structure_function  # noqa: E402

KMIN = tg.KMIN
A = (4 / 3) * I_INF * KMIN ** (2 / 3)
_s = np.geomspace(1e-9, 2, 200001)
_D = structure_function(_s, KMIN)
J = np.trapezoid(_D / _s, _s)
J2 = np.trapezoid(_D ** 2 / _s, _s)
# small-l expansion of N_k beyond the wall: N_k ~ J_k - c_k A^k l^(2k/3), c_k = -B(-k/3, 1/2)/2
C1 = -0.5 * gamma(-1 / 3) * sqrt(pi) / gamma(1 / 6)
C2 = -0.5 * gamma(-2 / 3) * sqrt(pi) / gamma(-1 / 6)
BEND_RATIO = 2 * np.exp(-1.5)

WIDTHS = [0.02, 0.09, 0.3]
SHAPE_WIDTH = 0.09
SHAPE_LAMS = [0.0, 1.0, 2.0, 10.0, np.inf]
SIM_LAMS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
LAM_DENSE = np.geomspace(0.2, 50.0, 40)
LAM_ZERO_PLOT, LAM_INF_PLOT = 0.1, 100.0
MIN_PAIRS = 200
XB_GRID = np.linspace(-3.0, np.log10(tg.L_MAX), 1201)
OUT_PNG = tg.OUT_DIR / "analytic_bend_slope.png"
OUT_CSV = tg.OUT_DIR / "analytic_bend_slope.csv"


# ----------------------------------------------------------------- geometry (closed forms)
def pair_count(ell, width):
    """P(l): thin-shell pair count at sky separation l, normalised so that P(0) = 3/2 + ln(2/Delta) = K_b(0)."""
    ell = np.asarray(ell, float)
    y = ell / width
    out = np.arccosh(2 / ell)
    m = y < 1
    out[m] += 1.5 * np.sqrt(1 - y[m] ** 2) - (1 + y[m] ** 2 / 2) * np.arccosh(1 / y[m])
    return out


def _proj_moments(ell):
    """M_n = int s^n dx along the line of sight over s = sqrt(l^2 + x^2) <= 2, n = 0..3."""
    L = np.sqrt(4 - ell ** 2)
    a = np.arccosh(2 / ell)
    return 2 * L, 2 * L + ell ** 2 * a, 2 * (ell ** 2 * L + L ** 3 / 3), L * (2 * L ** 2 + 5 * ell ** 2) / 2 + 0.75 * ell ** 4 * a


def q2(ell):
    """<dmu^2> P / 2: cross-correlation of the mu0 and 1/mu0 disc maps (ball x sphere), / its l = 0 value."""
    M0, M1, _, _ = _proj_moments(ell)
    return (M0 - M1 / 2) / 2


def q4(ell):
    """<dmu^4> P / 4: [mu0^3 x 1/mu0 + 3 mu0 x mu0] cross-correlations, / their l = 0 value."""
    M0, M1, M2, M3 = _proj_moments(ell)
    return 0.375 * (M1 - M2 + M3 / 4) + (M3 - 12 * M1 + 16 * M0) / 32


# ----------------------------------------------------------------- turbulence (1D integrals of D)
def n_k(ell, width, k, n=3000):
    """D^k-weighted pair count, same normalisation as P (N_k(0) = J_k - shell correction)."""
    u0 = np.arccosh(max(width / ell, 1.0))
    u = np.linspace(u0, np.arccosh(2 / ell), n)
    out = np.trapezoid(structure_function(ell * np.cosh(u), KMIN) ** k, u)
    if ell < width:  # same-wall slab pairs, D = A s^(2/3); s ds / sqrt(s^2 - l^2) = s dv with s = l cosh v
        v = np.linspace(0, u0, n)
        s = ell * np.cosh(v)
        out += 2 / width ** 2 * np.trapezoid((width - s / 2) * (A * s ** (2 / 3)) ** k * s, v)
    return out


def chi(ell):
    mu = np.linspace(0, 1, 2001)
    return np.trapezoid(mu * structure_function(np.sqrt(4 * mu ** 2 + ell ** 2), KMIN), mu)


def terms(ells, width):
    return [dict(P=pair_count(np.array([l]), width)[0], q2=q2(l), q4=q4(l), N1=n_k(l, width, 1),
                 N2=n_k(l, width, 2), chi=chi(l)) for l in ells]


def k_analytic(t, lam):
    if np.isinf(lam):
        return t["P"] * t["q4"] / t["q2"] ** 2
    return t["P"] * (3 * t["N2"] + 24 * t["chi"] * t["q2"] * lam ** 2 + 4 * t["q4"] * lam ** 4) / (
        t["N1"] + 2 * t["q2"] * lam ** 2) ** 2


def f_decline(lam):
    """Leading-order dilution of the decline: K ~ F_d(lam) ln(4/l) for Delta << l << outer scale."""
    if np.isinf(lam):
        return 1.0
    return (3 * J2 + 4 * lam ** 2 * (lam ** 2 + 6)) / (J + 2 * lam ** 2) ** 2


# ----------------------------------------------------------------- hinge fit
def hinge_fit(x, y):
    best = None
    for xb in XB_GRID:
        h = np.maximum(0.0, x - xb)
        if np.count_nonzero(h > 0) < 2 or np.count_nonzero(h == 0) < 1:
            continue
        X = np.column_stack([np.ones_like(x), h])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ coef - y) ** 2))
        if best is None or sse < best[1]:
            best = ((coef[0], xb, coef[1]), sse)
    return best[0]


def load(width):
    acc = tg.accumulate(width, 30000)
    ell_all = np.sqrt(tg.EDGES[:-1] * tg.EDGES[1:])
    good = (acc["n"] >= MIN_PAIRS) & (ell_all <= tg.L_MAX)
    ell = ell_all[good]
    return dict(ell=ell, x=np.log10(ell), acc={k: v[..., good] for k, v in acc.items()}, terms=terms(ell, width))


def sim_curve(d, lam, seed=None):
    if seed is None:
        return tg.K_sim(d["acc"], lam)  # 8 seeds, odd terms removed (= mirror pooling)
    return tg.K_sim(d["acc"], lam, seeds=slice(seed, seed + 1), odd=True)  # one realisation, as observed


def an_curve(d, lam):
    return np.array([k_analytic(t, lam) for t in d["terms"]])


def lam_plot(lam):
    if lam == 0:
        return LAM_ZERO_PLOT
    return LAM_INF_PLOT if np.isinf(lam) else lam


# ----------------------------------------------------------------- main
def main():
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    print(f"A = {A:.3f}, J = {J:.3f}, J2 = {J2:.3f}, c1 = {C1:.4f}, c2 = {C2:.4f}")
    data = {w: load(w) for w in WIDTHS}

    rows = []
    print(f"{'Delta':>5} {'lam':>5} | {'sim: x_b-logD':>13} {'S':>12} | {'analytic: x_b-logD':>18} {'S':>6} | "
          f"{'leading order S':>15} | rms(an-sim)")
    for w, d in data.items():
        lw = np.log10(w)
        for lam in SIM_LAMS:
            ys = sim_curve(d, lam)
            ya = an_curve(d, lam)
            ps_ = hinge_fit(d["x"], ys)
            pa = hinge_fit(d["x"], ya)
            single = np.array([hinge_fit(d["x"], sim_curve(d, lam, s)) for s in range(tg.N_SEEDS)])
            exb, es = single[:, 1].std(ddof=1), single[:, 2].std(ddof=1)
            s_lo = -np.log(10) * f_decline(lam)
            rms = np.sqrt(np.mean((ya - ys) ** 2))
            print(f"{w:5g} {lam:5g} | {ps_[1] - lw:+6.3f}±{exb:.3f} {ps_[2]:6.2f}±{es:.2f} | "
                  f"{pa[1] - lw:+18.3f} {pa[2]:6.2f} | {s_lo:15.2f} | {rms:.3f}")
            rows.append(dict(width=w, lam=lam, sim_Kplat=ps_[0], sim_xb_minus_logD=ps_[1] - lw, sim_S=ps_[2],
                             single_source_sigma_xb=exb, single_source_sigma_S=es,
                             an_Kplat=pa[0], an_xb_minus_logD=pa[1] - lw, an_S=pa[2],
                             leading_xb_minus_logD=np.log10(BEND_RATIO), leading_S=s_lo, rms_curve=rms))
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- figure
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={"wspace": 0.28})
    cm = plt.colormaps["viridis"]
    lam_cols = [plt.colormaps["magma"](v) for v in np.linspace(0.85, 0.1, len(SHAPE_LAMS))]
    wnorm = matplotlib.colors.LogNorm(vmin=0.02, vmax=1.0)

    ax = axs[0]
    d = data[SHAPE_WIDTH]
    for lam, col in zip(SHAPE_LAMS, lam_cols):
        lab = r"$\lambda=\infty$" if np.isinf(lam) else rf"$\lambda={lam:g}$"
        ax.plot(d["ell"], sim_curve(d, lam), "o", ms=4.5, color=col, alpha=0.8)
        ax.plot(d["ell"], an_curve(d, lam), "-", color=col, lw=2.2, label=lab)
        pa = hinge_fit(d["x"], an_curve(d, lam))
        xx = np.linspace(d["x"].min(), d["x"].max(), 200)
        ax.plot(10 ** xx, pa[0] + pa[2] * np.maximum(0, xx - pa[1]), ":", color=col, lw=1.4)
    ax.axvline(BEND_RATIO * SHAPE_WIDTH, color="0.5", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell/r_{\rm HII}$")
    ax.set_ylabel(r"$K(\ell)$")
    ax.set_title(rf"$\Delta={SHAPE_WIDTH:g}$: simulation (points), analytic (lines), hinge (dotted)", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")

    for ax, key, ylab in ((axs[1], "xb", r"$\log_{10}(\ell_b/\Delta)$"), (axs[2], "S", r"post-bend slope $S$ [per dex]")):
        for w, d in data.items():
            col = cm(wnorm(w))
            lw = np.log10(w)
            sub = [r for r in rows if r["width"] == w]
            xs = [lam_plot(r["lam"]) * 10 ** (0.03 * (WIDTHS.index(w) - 1)) for r in sub]
            if key == "xb":
                ys = [r["sim_xb_minus_logD"] for r in sub]
                es = [r["single_source_sigma_xb"] for r in sub]
            else:
                ys = [r["sim_S"] for r in sub]
                es = [r["single_source_sigma_S"] for r in sub]
            ax.errorbar(xs, ys, yerr=es, fmt="o", color=col, mec="k", ms=6, capsize=3, label=rf"$\Delta={w:g}$")
            dense = []
            for lam in LAM_DENSE:
                pa = hinge_fit(d["x"], an_curve(d, lam))
                dense.append(pa[1] - lw if key == "xb" else pa[2])
            ax.plot(LAM_DENSE, dense, "-", color=col, lw=2)
            for lam in (0.0, np.inf):
                pa = hinge_fit(d["x"], an_curve(d, lam))
                ax.plot(lam_plot(lam), pa[1] - lw if key == "xb" else pa[2], "_", color=col, ms=16, mew=2)
        if key == "xb":
            ax.axhline(np.log10(BEND_RATIO), color="0.3", ls="--", lw=1.2, label=r"leading order $2e^{-3/2}$")
        else:
            lo = [-np.log(10) * f_decline(l) for l in LAM_DENSE]
            ax.plot(LAM_DENSE, lo, "--", color="0.3", lw=1.2, label=r"leading order $-\ln 10\,F_d(\lambda)$")
        ax.set_xscale("log")
        ax.set_xlim(0.07, 150)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([LAM_ZERO_PLOT, 1, 10, LAM_INF_PLOT]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0", "1", "10", r"$\infty$"]))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"$\lambda=v_{\rm bulk}/v_{\rm turb}$")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.legend(fontsize=9, loc="lower left" if key == "xb" else "upper right")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
