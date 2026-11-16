#!/usr/bin/env python3
"""Compare the Kolmogorov-field model with the original v_bulk/v_turb = 10 shell runs.

Data: the 14-width N = 100,000 runs (field seed 7) behind fit_kurtosis_curve_shape.py,
K(l) from the pipeline pair estimator, r_min = 0.001.

Model:
  * K(l; lambda) from derive_kurtosis_shape_kolmogorov.py (same-wall / opposite-wall pairs,
    generator D(d), bulk curve K_b(l)), at lambda = 10; lambda = inf (= K_b) for reference.
  * Plateau formula (l -> 0), with exact chord moments:
        K_plat = W [3 G2 + 6 lam^2 X + lam^4 Q4] / [G1 + lam^2 Q2]^2
    W = int n^2 rho drho,  Q2 = int n^2 2 m2,  Q4 = int n^2 (2 m4 + 6 m2^2),
    G_k = int n^2 (1/2)(<D^k>_opp + <D^k>_same),  X = int n^2 2 m2 <D>_opp.
  * Free softplus fits of the data give the measured K_plat (kurtosis_curve_shape_fits.csv).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
from derive_bulk_limit_kurtosis import chord, k_theory  # noqa: E402
from derive_kurtosis_shape_kolmogorov import S as D_field, k_model, model_sums  # S(d) there = D(d) here  # noqa: E402
from fit_kurtosis_curve_shape import BASE_DIR, L_MAX_FIT, WIDTHS, load_or_compute  # noqa: E402

LAM = 10.0
OUT_PNG = BASE_DIR / "kolmogorov_model_vs_lambda10_runs.png"
OUT_CSV = BASE_DIR / "kolmogorov_model_vs_lambda10_runs.csv"


def plateau(width, lam):
    a = 1 - width
    eps = 1 - a * a
    mu = np.geomspace(1e-6, 1, 8000)
    rho = np.sqrt(1 - mu ** 2)
    n, m2, m4 = chord(rho, a)
    x2, x1 = mu, np.sqrt(np.clip(mu ** 2 - eps, 0, None))
    h = x2 - x1
    I = lambda f: np.trapezoid(n ** 2 * mu * f, mu)          # rho drho = mu0 dmu0
    zf = np.linspace(0, 1, 400)
    pz, dz = 2 * (1 - zf), np.gradient(zf)
    Ds = D_field(zf[None, :] * h[:, None])
    uf = np.linspace(-1, 1, 401)
    pu = 1 - np.abs(uf)
    pu /= pu.sum()
    Do = D_field((x1 + x2)[:, None] + uf[None, :] * h[:, None])
    W, Q2, Q4 = I(1), I(2 * m2), I(2 * m4 + 6 * m2 ** 2)
    G1 = I(0.5 * ((Do * pu).sum(1) + (Ds * pz * dz).sum(1)))
    G2 = I(0.5 * ((Do ** 2 * pu).sum(1) + (Ds ** 2 * pz * dz).sum(1)))
    X = I(2 * m2 * (Do * pu).sum(1))
    if np.isinf(lam):
        return W * Q4 / Q2 ** 2
    return W * (3 * G2 + 6 * lam ** 2 * X + lam ** 4 * Q4) / (G1 + lam ** 2 * Q2) ** 2


def fitted_kplat():
    out = {}
    with (BASE_DIR / "kurtosis_curve_shape_fits.csv").open() as f:
        for row in csv.DictReader(f):
            if row["fit"] == "free" and abs(float(row["r_min"]) - 0.001) < 1e-6:
                out[round(float(row["width"]), 5)] = (float(row["K_plat"]), float(row["K_plat_err"] or "nan"))
    return out


def main():
    curves, _ = load_or_compute()
    kfit = fitted_kplat()
    cmap = plt.get_cmap("viridis")
    lw = np.log10(WIDTHS)
    col = lambda i: cmap((lw[i] - lw.min()) / (lw.max() - lw.min()))

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axP = fig.add_subplot(gs[0, 2])
    axR = fig.add_subplot(gs[1, :2])
    axQ = fig.add_subplot(gs[1, 2])

    rows = []
    kp_model, kp_bulk, kp_data, kp_err = [], [], [], []
    for i, (w, (lc, kk)) in enumerate(zip(WIDTHS, curves)):
        ell = 10 ** lc
        ok = np.isfinite(kk) & (ell <= L_MAX_FIT)
        e, k = ell[ok], kk[ok]
        kb = k_theory(e, 1.0 - w)
        km = np.array([k_model(model_sums(l, w, kbl), LAM) for l, kbl in zip(e, kb)])
        ax = axA if w < 0.2 else axB
        ax.plot(e, k, "o", ms=2.8, color=col(i), alpha=0.65)
        ax.plot(e, km, "-", lw=1.7, color=col(i))
        ax.plot(e, kb, ":", lw=1.0, color=col(i))
        axR.plot(e, k - km, "-", lw=1.1, color=col(i))
        kp_model.append(plateau(w, LAM))
        kp_bulk.append(plateau(w, np.inf))
        kd, ke = kfit.get(round(float(w), 5), (np.nan, np.nan))
        kp_data.append(kd)
        kp_err.append(ke)
        small = e < min(0.3 * w, 0.02)
        rows.append((w, kp_model[-1], kp_bulk[-1], kd, ke,
                     np.nanmean((k - km)[small]) if small.any() else np.nan,
                     np.sqrt(np.nanmean((k - km) ** 2)), np.sqrt(np.nanmean((k - kb) ** 2))))

    for ax, ttl in ((axA, r"$\Delta r_{\rm shell}<0.2$"), (axB, r"$\Delta r_{\rm shell}\geq0.2$")):
        ax.plot([], [], "ko", ms=3, label=r"simulation, $\lambda=10$ (seed 7)")
        ax.plot([], [], "k-", label=r"model, $\lambda=10$")
        ax.plot([], [], "k:", label=r"bulk limit $K_b(\ell)$ ($\lambda=\infty$)")
        ax.set(xscale="log", ylim=(1.8, 7.0), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$", title=ttl)
        ax.legend(fontsize=8, loc="lower left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.LogNorm(WIDTHS.min(), WIDTHS.max()))
    fig.colorbar(sm, ax=axB, label=r"$\Delta r_{\rm shell}/r_{\rm HII}$")

    axP.errorbar(WIDTHS, kp_data, yerr=kp_err, fmt="o", color="k", ms=5, capsize=2,
                 label=r"simulation: fitted $K_{\rm plat}$")
    axP.plot(WIDTHS, kp_model, "-", color="C3", lw=2, label=r"plateau formula, $\lambda=10$")
    axP.plot(WIDTHS, kp_bulk, "--", color="C0", lw=1.4, label=r"$K_b(0)=C_p-\ln 2\Delta$ (exact), $\lambda=\infty$")
    axP.set(xscale="log", ylim=(2.4, 7.0), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$", ylabel=r"$K_{\rm plat}$",
            title="plateau level vs shell width")
    axP.legend(fontsize=8)

    axR.axhline(0, color="k", lw=0.8)
    axR.set(xscale="log", ylim=(-1, 1), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"data $-$ model",
            title=r"residuals, $\lambda=10$ (colour = width)")

    r = np.array(rows)
    axQ.plot(WIDTHS, r[:, 6], "-o", color="C3", label=r"rms(data $-$ model, $\lambda=10$)")
    axQ.plot(WIDTHS, r[:, 7], "--s", color="C0", label=r"rms(data $-$ $K_b$)")
    axQ.set(xscale="log", xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$", ylabel=r"rms over $\ell\leq0.6$",
            title="curve-level agreement")
    axQ.legend(fontsize=8)
    for a in (axA, axB, axP, axR, axQ):
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)

    with OUT_CSV.open("w") as f:
        f.write("width,K_plat_formula_lam10,K_b0,K_plat_fit_data,K_plat_fit_err,mean_resid_small_l,rms_model,rms_bulk\n")
        for row in rows:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")
    print(" width  K_plat: formula(10)  K_b(0)  data-fit      rms: model  bulk")
    for row in rows:
        print(f" {row[0]:.3f}         {row[1]:.2f}       {row[2]:.2f}    {row[3]:.2f}+-{row[4]:.2f}       {row[6]:.3f}  {row[7]:.3f}")
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
