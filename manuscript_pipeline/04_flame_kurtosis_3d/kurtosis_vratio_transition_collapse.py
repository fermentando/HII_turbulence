#!/usr/bin/env python3
"""Which variable controls the turbulence -> bulk transition of K(l)?

Uses the per-width pair-moment caches written by kurtosis_shape_vs_vratio_fixed_width.py
(run it first for each width). For each width and a dense lambda grid:

  R(lambda) = <(K_lambda - K_t)(K_b - K_t)> / <(K_b - K_t)^2>   over l in [l_min, 0.6]

i.e. how far the whole curve has moved from pure turbulence (R=0) to pure bulk (R=1).
Candidate collapse variables:
  X_var  = lambda^2 <dmu^2> / <du^2>     (bulk / turbulent velocity variance, pair-weighted over the window)
  X_limb = lambda sqrt(2 Delta)          (bulk velocity of the limb pairs / v_turb)
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
sys.argv = sys.argv[:1]  # the imported module reads an optional width from argv
import kurtosis_shape_vs_vratio_fixed_width as kv  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3, 1.0]
L_MAX = 0.6
LAM = np.geomspace(0.03, 1000, 600)
OUT_PNG = kv.BASE_DIR / "kurtosis_vratio_transition_collapse.png"
OUT_CSV = kv.BASE_DIR / "kurtosis_vratio_transition_collapse.csv"


def load(width):
    tag = f"w{width:g}".replace(".", "p")
    path = kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}.npz"
    c = np.load(path)
    return {k: c[k] for k in kv.KEYS}


def main():
    ell = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    out = {}
    for w in WIDTHS:
        M = load(w)
        sel = (M["n"] >= 200) & (ell <= L_MAX)
        kt, kb = kv.k_of_lambda(M, 0.0)[sel], kv.k_of_lambda(M, np.inf)[sel]
        d = kb - kt
        R = np.array([np.sum((kv.k_of_lambda(M, l)[sel] - kt) * d) / np.sum(d * d) for l in LAM])
        var_ratio = M["m2"][sel].sum() / M["u2"][sel].sum()           # <dmu^2>/<du^2> over window
        lam_half = np.interp(0.5, R, LAM)                                # R is monotone here
        out[w] = dict(R=R, var_ratio=var_ratio, lam_half=lam_half,
                      du2=M["u2"][sel].sum() / M["n"][sel].sum(), dmu2=M["m2"][sel].sum() / M["n"][sel].sum(),
                      contrast=np.sqrt(np.mean(d * d)))
        assert np.all(np.diff(R[(R > 0.05) & (R < 0.95)]) > 0), f"R not monotone for width {w}"

    print(" width   <du^2>  <dmu^2>  lam_1/2   X_var(1/2)   X_limb(1/2)   rms(K_b-K_t)")
    with OUT_CSV.open("w") as f:
        f.write("width,du2,dmu2,lambda_half,X_var_half,X_limb_half,rms_Kb_minus_Kt\n")
        for w, o in out.items():
            xv = o["lam_half"] ** 2 * o["var_ratio"]
            xl = o["lam_half"] * np.sqrt(2 * w)
            print(f" {w:<5g}   {o['du2']:.3f}   {o['dmu2']:.3f}    {o['lam_half']:.2f}     {xv:.2f}         {xl:.2f}          {o['contrast']:.2f}")
            f.write(f"{w},{o['du2']:.4f},{o['dmu2']:.4f},{o['lam_half']:.4f},{xv:.4f},{xl:.4f},{o['contrast']:.4f}\n")

    lh = np.array([out[w]["lam_half"] for w in WIDTHS])
    xv = np.array([out[w]["lam_half"] ** 2 * out[w]["var_ratio"] for w in WIDTHS])
    xl = lh * np.sqrt(2 * np.array(WIDTHS))
    for name, arr in (("lambda", lh), ("X_var", xv), ("X_limb", xl)):
        print(f"spread of midpoint in {name:>6}: max/min = {arr.max() / arr.min():.2f}")

    cmap = plt.get_cmap("viridis")
    col = {w: cmap(i / (len(WIDTHS) - 1)) for i, w in enumerate(WIDTHS)}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for w, o in out.items():
        lab = rf"$\Delta={w:g}$"
        ax[0].plot(LAM, o["R"], color=col[w], lw=2, label=lab)
        ax[1].plot(LAM ** 2 * o["var_ratio"], o["R"], color=col[w], lw=2, label=lab)
        ax[2].plot(LAM * np.sqrt(2 * w), o["R"], color=col[w], lw=2, label=lab)
    titles = [r"(a) vs $\lambda = v_{\rm bulk}/v_{\rm turb}$",
              r"(b) vs variance ratio $\lambda^2\langle\delta\mu^2\rangle/\langle\delta u^2\rangle$",
              r"(c) vs limb criterion $\lambda\sqrt{2\Delta}$"]
    xlabels = [r"$\lambda$", r"$X_{\rm var}$", r"$X_{\rm limb}$"]
    for a, t, xl_ in zip(ax, titles, xlabels):
        a.set(xscale="log", xlabel=xl_, title=t, ylim=(-0.1, 1.1))
        a.axhline(0.5, color="0.5", ls=":", lw=1)
        a.grid(True, ls="--", alpha=0.4)
    ax[1].axvline(1, color="0.5", ls=":", lw=1)
    ax[0].set_ylabel(r"$R$: fraction of the way from turbulent to bulk $K(\ell)$")
    ax[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
