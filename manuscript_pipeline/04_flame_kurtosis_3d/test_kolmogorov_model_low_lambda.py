#!/usr/bin/env python3
"""Test the Kolmogorov-field K(l; lambda) model where turbulence matters (lambda ~ 0.5-2).

Figure 1 (kolmogorov_model_low_lambda_seeds.png): 8-seed lambda-sweep runs
  (kurtosis_vratio_seed_ensemble.py; N = 30,000, all pairs), widths 0.02, 0.09, 0.3, 1.0,
  lambda = 0.5, 1, 2. Points: pooled over seeds; band: 16-84% range of the 8 single seeds;
  line: general-l model; short bar: exact plateau formula; dotted: bulk limit K_b.
Figure 2 (kolmogorov_model_lambda1_n5000_sweep.png): original N = 5,000 v_bulk/v_turb sweep
  at lambda = 1 (single field seed 7), all 14 widths, vs the model.
See manuscript_notes/kurtosis_kolmogorov_lambda_derivation.md.
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
from compare_kolmogorov_model_lambda10 import plateau  # noqa: E402
from derive_bulk_limit_kurtosis import k_theory  # noqa: E402
from derive_kurtosis_shape_kolmogorov import k_model, model_sums  # noqa: E402
import plot_vbulk_ratio_sweep_n5000_logistic as sweep  # noqa: E402

WIDTHS_SEEDS = [0.02, 0.09, 0.3, 1.0]
LAMS = [0.5, 1.0, 2.0]
L_MAX = 0.6
OUT1 = kv.BASE_DIR / "kolmogorov_model_low_lambda_seeds.png"
OUT2 = kv.BASE_DIR / "kolmogorov_model_lambda1_n5000_sweep.png"


def model_curve(ell, width, lam):
    kb = k_theory(ell, 1.0 - width)
    return np.array([k_model(model_sums(l, width, k), lam) for l, k in zip(ell, kb)]), kb


def figure_seeds():
    ell_all = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    fig, ax = plt.subplots(len(LAMS), len(WIDTHS_SEEDS), figsize=(18, 11), sharex=True)
    print("8-seed runs: rms(pooled - model) over l <= 0.6 | plateau formula vs pooled (first 5 bins) | seed spread (16-84%) on plateau")
    for j, w in enumerate(WIDTHS_SEEDS):
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
            km, kb = model_curve(ell, w, lam)
            kplat = plateau(w, lam)
            a.fill_between(ell, lo, hi, color="C0", alpha=0.2, lw=0, label="single seeds, 16–84%")
            a.plot(ell, kp, "o", ms=3, color="C0", label="8 seeds pooled")
            a.plot(ell, km, "-", color="C3", lw=1.8, label="model")
            a.plot(ell, kb, ":", color="0.3", lw=1.2, label=r"bulk limit $K_b$")
            a.hlines(kplat, ell[0], ell[0] * 4, color="C3", lw=4, alpha=0.6, label="plateau formula")
            a.axhline(3, color="0.6", lw=0.8, ls="--")
            a.set(xscale="log", ylim=(1.8, 7.2))
            a.grid(True, ls="--", alpha=0.4)
            if i == 0:
                a.set_title(rf"$\Delta r_{{\rm shell}}={w:g}$")
            if j == 0:
                a.set_ylabel(rf"$\lambda={lam:g}$" + "\n" + r"$K(\ell)$")
            if i == len(LAMS) - 1:
                a.set_xlabel(r"$\ell/r_{\rm HII}$")
            rms = np.sqrt(np.nanmean((kp - km) ** 2))
            spread = np.nanmean((hi - lo)[:5]) / 2
            print(f"  Delta={w:<5g} lambda={lam:g}: rms={rms:.3f} | plateau {kplat:.2f} vs {np.nanmean(kp[:5]):.2f}"
                  f" | +-{spread:.2f}")
    ax[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Model vs 8-seed runs (N = 30,000, all pairs) at low $v_{\\rm bulk}/v_{\\rm turb}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT1, dpi=150)
    print(f"wrote {OUT1}")


def figure_sweep():
    records, _ = sweep.load_shell_records(1.0)
    cmap = plt.get_cmap("viridis")
    lw = np.log10(sweep.WIDTHS)
    col = lambda i: cmap((lw[i] - lw.min()) / (lw.max() - lw.min()))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    print("\nN=5,000 sweep at lambda=1 (seed 7): rms(data - model), rms(data - K_b)")
    rms_m, rms_b = [], []
    for i, (w, rec) in enumerate(zip(sweep.WIDTHS, records)):
        ok = np.isfinite(rec["kk"]) & (rec["r"] <= L_MAX)
        ell, k = rec["r"][ok], rec["kk"][ok]
        km, kb = model_curve(ell, w, 1.0)
        a = ax[0] if w < 0.2 else ax[1]
        a.plot(ell, k, "o", ms=3, color=col(i), alpha=0.7)
        a.plot(ell, km, "-", lw=1.7, color=col(i))
        rms_m.append(np.sqrt(np.mean((k - km) ** 2)))
        rms_b.append(np.sqrt(np.mean((k - kb) ** 2)))
        print(f"  Delta={w:.3f}: {rms_m[-1]:.3f}  {rms_b[-1]:.3f}")
    for a, ttl in ((ax[0], r"$\Delta r_{\rm shell}<0.2$"), (ax[1], r"$\Delta r_{\rm shell}\geq0.2$")):
        a.plot([], [], "ko", ms=3, label=r"N = 5,000 sweep, $\lambda=1$ (seed 7)")
        a.plot([], [], "k-", label=r"model, $\lambda=1$")
        a.axhline(3, color="0.6", lw=0.8, ls="--")
        a.set(xscale="log", ylim=(1.8, 7.2), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$", title=ttl)
        a.grid(True, ls="--", alpha=0.4)
        a.legend(fontsize=8, loc="lower left")
    ax[2].plot(sweep.WIDTHS, rms_m, "-o", color="C3", label=r"rms(data $-$ model, $\lambda=1$)")
    ax[2].plot(sweep.WIDTHS, rms_b, "--s", color="0.4", label=r"rms(data $-$ bulk limit $K_b$)")
    ax[2].set(xscale="log", xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$", ylabel=r"rms over $\ell\leq0.6$",
              title="curve-level agreement")
    ax[2].grid(True, ls="--", alpha=0.4)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT2, dpi=150)
    print(f"wrote {OUT2}")


if __name__ == "__main__":
    figure_seeds()
    figure_sweep()
