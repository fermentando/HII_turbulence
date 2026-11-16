#!/usr/bin/env python3
"""Global fit of dK/dlog10 r vs shell width, v_bulk/v_turb and r_min/r_HII.

Reads the (v_bulk/v_turb, r_min/r_HII, width) slope grid cached by
plot_vbulk_rmin_2d_response_model.py (pair-kurtosis estimator) and fits ONE
formula to every slope at once, instead of fitting each curve separately and
then fitting the per-curve parameters (whose floor/x0/scale trade off
against each other curve by curve):

    dK/dlog10 r = F(v, rho) / (1 + exp[(log10 w - x0(rho)) / s(v, rho)])

    w   = Delta r_shell / r_HII
    v   = v_bulk / v_turb
    rho = log10[(r_min/r_HII) / 0.01]
    H(v; v_c, n) = v^n / (v^n + v_c^n)              (Hill switch = logistic in log v)

    x0(rho)   = x0_0 + a_x * rho
    F(v, rho) = F0(rho) + (Finf(rho) - F0(rho)) * H(v; v_F, n_F)
    s(v, rho) = [s0 + (sinf - s0) * H(v; v_s, n_s)] * 10^(beta * rho)

Model A: F0, Finf independent of rho.
Model B: F0 = F0_0 + f0_rho * rho, Finf = Finf_0 + finf_rho * rho.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "00_core"))

from generate_shell_thickness_grid import R_OUTER_FIXED  # noqa: E402
from plot_vbulk_rmin_2d_response_model import CACHE, KK_ESTIMATOR, RMIN_RATIOS, WIDTHS  # noqa: E402
from plot_vbulk_ratio_sweep_n5000_logistic import BASE_DIR, VBULK_RATIOS  # noqa: E402

RMIN_REF = 0.01

PARAMS_A = ["x0_0", "a_x", "F0", "Finf", "v_F", "n_F", "s0", "sinf", "v_s", "n_s", "beta"]
P0_A = [-1.34, 0.4, -0.76, -3.4, 1.4, 1.6, 1.1, 0.38, 0.65, 2.8, -0.2]
LO_A = [-4.0, -3.0, -5.0, -8.0, 1e-2, 0.2, 0.02, 0.02, 1e-2, 0.2, -3.0]
HI_A = [2.0, 3.0, 0.0, 0.0, 1e2, 10.0, 5.0, 5.0, 1e2, 10.0, 3.0]

PARAMS_B = PARAMS_A + ["f0_rho", "finf_rho"]
P0_B = P0_A + [0.0, 0.0]
LO_B = LO_A + [-5.0, -5.0]
HI_B = HI_A + [5.0, 5.0]


def hill(v: np.ndarray, v_c: float, n: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v**n / (v**n + v_c**n)


def model_terms(log_w, v, rho, p: dict):
    x0 = p["x0_0"] + p["a_x"] * rho
    f0 = p["F0"] + p.get("f0_rho", 0.0) * rho
    finf = p["Finf"] + p.get("finf_rho", 0.0) * rho
    floor = f0 + (finf - f0) * hill(v, p["v_F"], p["n_F"])
    scale = (p["s0"] + (p["sinf"] - p["s0"]) * hill(v, p["v_s"], p["n_s"])) * 10 ** (p["beta"] * rho)
    return floor, x0, scale


def model(xdata: np.ndarray, *theta, names: list[str]) -> np.ndarray:
    log_w, v, rho = xdata
    floor, x0, scale = model_terms(log_w, v, rho, dict(zip(names, theta)))
    return floor / (1.0 + np.exp((log_w - x0) / scale))


def load_data():
    with np.load(CACHE) as c:
        if str(c["kk_estimator"]) != KK_ESTIMATOR:
            raise RuntimeError("2D grid cache was not built with the pair-kurtosis estimator")
        slopes, fallback, eff = c["slopes"], c["fallback"], c["effective_rmin"]
    nv, nr, nw = slopes.shape
    v = np.broadcast_to(VBULK_RATIOS[:, None, None], slopes.shape)
    log_w = np.broadcast_to(np.log10(WIDTHS)[None, None, :], slopes.shape)
    # rho from the effective (achieved) r_min of each cell, median over widths
    cell_rmin = np.nanmedian(eff, axis=2) / R_OUTER_FIXED
    rho = np.broadcast_to(np.log10(cell_rmin / RMIN_REF)[:, :, None], slopes.shape)
    good = np.isfinite(slopes) & ~fallback
    xdata = np.vstack((log_w[good], v[good], rho[good]))
    return slopes, fallback, cell_rmin, xdata, slopes[good], good


def fit(names, p0, lo, hi, xdata, y):
    f = lambda x, *t: model(x, *t, names=names)  # noqa: E731
    params, cov = curve_fit(f, xdata, y, p0=p0, bounds=(lo, hi), maxfev=50000)
    resid = y - f(xdata, *params)
    rss = float(np.sum(resid**2))
    n, k = y.size, len(params)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return params, np.sqrt(np.diag(cov)), float(np.sqrt(rss / (n - k))), aic, bic


def main() -> None:
    slopes, fallback, cell_rmin, xdata, y, good = load_data()
    print(f"Fitting {y.size} slopes ({good.size - y.size} NaN/fallback excluded)")

    results = {}
    for label, names, p0, lo, hi in (("A", PARAMS_A, P0_A, LO_A, HI_A), ("B", PARAMS_B, P0_B, LO_B, HI_B)):
        params, perr, rms, aic, bic = fit(names, p0, lo, hi, xdata, y)
        results[label] = (names, params, perr, rms, aic, bic)
        print(f"\nModel {label}: RMS residual={rms:.4f}, AIC={aic:.1f}, BIC={bic:.1f}")
        for name, val, err in zip(names, params, perr):
            at_bound = np.isclose(val, lo[names.index(name)]) or np.isclose(val, hi[names.index(name)])
            print(f"  {name:9s} = {val:9.4f} +/- {err:.4f}{'  (AT BOUND)' if at_bound else ''}")

    best = min(results, key=lambda k: results[k][5])
    print(f"\nPreferred by BIC: model {best}")

    out_csv = BASE_DIR / "global_dk_dlogr_model_fit.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("model,parameter,value,error,rms_residual,aic,bic,preferred\n")
        for label, (names, params, perr, rms, aic, bic) in results.items():
            for name, val, err in zip(names, params, perr):
                f.write(f"{label},{name},{val:.8g},{err:.8g},{rms:.6g},{aic:.6g},{bic:.6g},{label == best}\n")

    names, params = results[best][0], results[best][1]
    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    nonzero = VBULK_RATIOS > 0
    norm = matplotlib.colors.LogNorm(vmin=VBULK_RATIOS[nonzero].min(), vmax=VBULK_RATIOS[nonzero].max())
    cmap = plt.colormaps["plasma"]
    ncol = 4
    nrow = int(np.ceil(len(RMIN_RATIOS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), sharex=True, sharey=True)
    dense = np.linspace(np.log10(WIDTHS).min(), np.log10(WIDTHS).max(), 200)
    for j, ax in enumerate(axes.flat):
        if j >= len(RMIN_RATIOS):
            ax.axis("off")
            continue
        for i, v in enumerate(VBULK_RATIOS):
            color = "0.2" if v == 0 else cmap(norm(v))
            mask = np.isfinite(slopes[i, j]) & ~fallback[i, j]
            ax.plot(WIDTHS[mask], slopes[i, j][mask], "o", ms=3.0, color=color, alpha=0.85)
            rho = np.log10(cell_rmin[i, j] / RMIN_REF)
            xd = np.vstack((dense, np.full_like(dense, v), np.full_like(dense, rho)))
            ax.plot(10**dense, model(xd, *params, names=names), "-", lw=1.2, color=color)
        ax.axhline(0, color="0.25", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(fr"$r_{{\rm min}}/r_{{\rm HII}}\approx{np.nanmedian(cell_rmin[:, j]):.3f}$", fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$dK/d\log_{10} r$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.8, label=r"$v_{\rm bulk}/v_{\rm turb}$ (black: 0)")
    fig.suptitle(f"Global fit, model {best} (points: N=5,000 slopes; lines: one formula)", fontsize=12)

    out_png = BASE_DIR / "global_dk_dlogr_model_fit.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
