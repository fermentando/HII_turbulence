#!/usr/bin/env python3
"""Check the Kolmogorov simplifications against the 8-seed low-lambda runs.

Per-pair-average form of the model:
    K = [3<D^2> + 6 lam^2 <D dmu^2> + lam^4 <dmu^4>] / [<D> + lam^2 <dmu^2>]^2
  <D^k>        exact: Abel projection of the shell autocorrelation C(s; Delta) with D(s)^k
  <dmu^2>      = 2 / K_b(l),  <dmu^4> = 4 / K_b(l)    (thin-shell bulk sums; lam -> inf gives K_b)
  <D dmu^2>    = (4 / K_b) int_0^1 mu0 D(sqrt(4 mu0^2 + l^2)) dmu0   (thin-shell cross term)
with three structure functions:
  generator  D(s) from the k^(-11/3) spectrum with k_min = 5.24, k_max = 73.3
  scale-free D = A s^(2/3), A = 5.855 (generator inertial-range amplitude; no fit)
  uncorrelated D = 2
plus the closed-form scale-free thin-shell plateau
    K_plat = K_b [1 + 3 c L^2 / (1.5 c + 2 L^2)^2],  c = 2^(2/3),  L = lambda / sqrt(A).
Data: kurtosis_vratio_seed_ensemble.py runs (8 field seeds pooled, N = 30,000, all pairs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
import kurtosis_shape_vs_vratio_fixed_width as kv  # noqa: E402
from derive_bulk_limit_kurtosis import k_theory  # noqa: E402
from derive_kurtosis_vratio_dependence import correlation_function  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3, 1.0]
LAMS = [0.5, 1.0, 2.0]
L_MAX = 0.6
A_KOL = 5.855
OUT = kv.BASE_DIR / "kolmogorov_limits_low_lambda.png"

_d, _xi = correlation_function()
_Dg = 2 * (1 - _xi)
D_MODELS = {
    "generator D(s)": lambda s: np.interp(s, _d, _Dg, right=_Dg[-1]),
    r"scale-free $A s^{2/3}$": lambda s: A_KOL * np.asarray(s) ** (2 / 3),
    "uncorrelated D = 2": lambda s: 2.0 * np.ones_like(np.asarray(s, float)),
}
STYLE = {"generator D(s)": ("C3", "-"), r"scale-free $A s^{2/3}$": ("C2", "--"), "uncorrelated D = 2": ("C1", ":")}


def lens(R1, R2, s):
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    out[s <= abs(R1 - R2)] = 4 / 3 * np.pi * min(R1, R2) ** 3
    m = (s > abs(R1 - R2)) & (s < R1 + R2)
    sm = s[m]
    out[m] = np.pi * (R1 + R2 - sm) ** 2 * (sm ** 2 + 2 * sm * (R1 + R2) - 3 * (R1 - R2) ** 2) / (12 * sm)
    return out


def C_shell(s, delta):
    a = 1 - delta
    return lens(1, 1, s) - 2 * lens(1, a, s) + lens(a, a, s)


X_GRID = np.concatenate(([0.0], np.geomspace(1e-8, 2.0, 6000)))
MU0 = np.linspace(0, 1, 801)


def model(ell, delta, lam, Dfun):
    kb = k_theory(ell, 1.0 - delta)
    out = np.empty_like(ell)
    for i, (l, k) in enumerate(zip(ell, kb)):
        s = np.sqrt(l ** 2 + X_GRID ** 2)
        c = C_shell(s, delta)
        Ds = Dfun(s)
        W = np.trapezoid(c, X_GRID)
        D1, D2 = np.trapezoid(c * Ds, X_GRID) / W, np.trapezoid(c * Ds ** 2, X_GRID) / W
        B2, B4 = 2 / k, 4 / k
        X = (4 / k) * np.trapezoid(MU0 * Dfun(np.sqrt(4 * MU0 ** 2 + l ** 2)), MU0)
        out[i] = (3 * D2 + 6 * lam ** 2 * X + lam ** 4 * B4) / (D1 + lam ** 2 * B2) ** 2
    return out, kb


def main():
    ell_all = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    fig, ax = plt.subplots(len(LAMS), len(WIDTHS), figsize=(18, 11), sharex=True)
    rows = []
    for j, w in enumerate(WIDTHS):
        tag = f"w{w:g}".replace(".", "p")
        c = np.load(kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds8.npz")
        pooled = {k: c[k].sum(0) for k in kv.KEYS}
        seeds = [{k: c[k][s] for k in kv.KEYS} for s in range(c["n"].shape[0])]
        good = (pooled["n"] >= 200) & (ell_all <= L_MAX)
        ell = ell_all[good]
        for i, lam in enumerate(LAMS):
            a = ax[i, j]
            kp = kv.k_of_lambda(pooled, lam)[good]
            ks = np.array([kv.k_of_lambda(s, lam)[good] for s in seeds])
            lo, hi = np.nanpercentile(ks, [16, 84], axis=0)
            a.fill_between(ell, lo, hi, color="C0", alpha=0.18, lw=0)
            a.plot(ell, kp, "o", ms=3, color="C0", label="8 seeds pooled")
            row = [w, lam]
            for name, Dfun in D_MODELS.items():
                km, kb = model(ell, w, lam, Dfun)
                colr, ls = STYLE[name]
                a.plot(ell, km, ls, color=colr, lw=1.8, label=name)
                row.append(np.sqrt(np.nanmean((kp - km) ** 2)))
            a.plot(ell, kb, color="0.5", lw=0.8, alpha=0.7, label=r"bulk limit $K_b$")
            cst = 2 ** (2 / 3)
            Lam2 = lam ** 2 / A_KOL
            kplat = kb[0] * (1 + 3 * cst * Lam2 / (1.5 * cst + 2 * Lam2) ** 2)
            a.hlines(kplat, ell[0], ell[0] * 4, color="C2", lw=4, alpha=0.5, label="scale-free plateau (closed form)")
            row.append(kplat)
            row.append(np.nanmean(kp[:5]))
            rows.append(row)
            a.axhline(3, color="0.6", lw=0.6, ls="--")
            a.set(xscale="log", ylim=(1.8, 7.5))
            a.grid(True, ls="--", alpha=0.4)
            if i == 0:
                a.set_title(rf"$\Delta r_{{\rm shell}}={w:g}$")
            if j == 0:
                a.set_ylabel(rf"$\lambda={lam:g}$" + "\n" + r"$K(\ell)$")
            if i == len(LAMS) - 1:
                a.set_xlabel(r"$\ell/r_{\rm HII}$")
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Structure-function limits vs 8-seed runs (band: single seeds 16–84%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("rms(pooled - model) over l <= 0.6   |  plateau: scale-free closed form vs pooled")
    print(" Delta lambda   generator  scale-free  uncorrelated  |  closed-form  data")
    for r in rows:
        print(f" {r[0]:<5g} {r[1]:<5g}    {r[2]:.3f}      {r[3]:.3f}       {r[4]:.3f}      |   {r[5]:.2f}       {r[6]:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
