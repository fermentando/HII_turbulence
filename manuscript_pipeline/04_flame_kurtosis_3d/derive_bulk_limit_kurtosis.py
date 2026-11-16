#!/usr/bin/env python3
"""Semi-analytic K(l) of a uniformly filled expanding shell in the bulk-dominated limit.

Limit v_bulk >> v_turb: the projected velocity of a point is v_bulk * mu, with
mu = x / r (x = line of sight). Kurtosis is scale-free, so v_bulk drops out and
K(l) is pure geometry of the shell [a, R], a = R - Delta, R = r_HII = 1.

Derivation
----------
At sky radius rho, points are uniform in x on the chord C(rho):
    a^2 <= x^2 + rho^2 <= R^2.
Column:          n(rho)   = |C(rho)|
Chord moments:   m_k(rho) = (1/n) int_C mu^k dx,   odd k vanish (x -> -x).
With x1 = sqrt(max(a^2-rho^2, 0)), x2 = sqrt(R^2-rho^2), and
    F0 = x
    F2 = x - rho atan(x/rho)
    F4 = x - (3/2) rho atan(x/rho) + rho^2 x / (2 (x^2+rho^2))
we get n = 2 (x2 - x1), m_k = 2 [F_k(x2) - F_k(x1)] / n.

Two points of a pair are independent given their sky positions, so for
dmu = mu_i - mu_j:
    E[dmu^2] = m2_i + m2_j
    E[dmu^4] = m4_i + m4_j + 6 m2_i m2_j.
The pair estimator averages over pairs at sky separation l with weight
n(rho_i) n(rho_j) (large-N limit). Writing rho_j = rho_i + l e_psi:
    W  = int d^2rho n n'
    Q2 = int d^2rho n n' (m2 + m2')
    Q4 = int d^2rho n n' (m4 + m4' + 6 m2 m2')
    K(l) = W Q4 / Q2^2.
l -> 0 gives the plateau as 1D integrals:
    K(0) = [int n^2] [int n^2 (m4 + 3 m2^2)] / (2 [int n^2 m2]^2)   (rho d rho measure)

Outputs: K_theory(l, Delta), a pure-bulk check with the real pipeline estimator,
and softplus joint fits of theory vs the v_bulk/v_turb = 10 simulation.
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
sys.path.insert(0, str(HERE.parents[0] / "00_core"))

from fit_kurtosis_curve_shape import (  # noqa: E402
    BASE_DIR, L_MAX_FIT, RMIN_SWEEP, WIDTHS, fit_joint, load_or_compute, softplus,
)
from generate_random_point_expanding_shell_grid import random_points_in_shell  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402

R = 1.0
N_SIM = 100_000
POINT_SEED = 1234
SIM_WIDTH_IDX = [0, 5, 9, 13]  # widths checked against the pipeline estimator
CACHE = BASE_DIR / "_bulk_limit_kurtosis_cache.npz"
OUT_PNG = BASE_DIR / "bulk_limit_kurtosis_derivation.png"
OUT_CSV = BASE_DIR / "bulk_limit_kurtosis_derivation.csv"


# ---------------- chord moments (closed form) ----------------
def _F(x, rho):
    at = np.arctan2(x, rho)
    s = x * x + rho * rho
    with np.errstate(invalid="ignore", divide="ignore"):
        f4_tail = np.where(s > 0, rho * rho * x / (2.0 * s), 0.0)
    return x, x - rho * at, x - 1.5 * rho * at + f4_tail


def chord(rho, a):
    rho = np.abs(rho)
    inside = rho < R
    x2 = np.sqrt(np.clip(R * R - rho * rho, 0, None))
    x1 = np.sqrt(np.clip(a * a - rho * rho, 0, None))
    f0b, f2b, f4b = _F(x2, rho)
    f0a, f2a, f4a = _F(x1, rho)
    n = 2.0 * (f0b - f0a)
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = np.where(n > 0, 2.0 * (f2b - f2a) / n, 0.0)
        m4 = np.where(n > 0, 2.0 * (f4b - f4a) / n, 0.0)
    n = np.where(inside, n, 0.0)
    return n, m2, m4


# ---------------- quadrature ----------------
def rho_nodes(a, n_seg=90, order=12):
    """Composite Gauss-Legendre on [0, R], segments clustered toward a and R."""
    d = R - a
    pts = set(np.linspace(0, a, 25))
    for k in np.geomspace(1e-4 * d, d, 25):
        pts.update([a - k, a + k, R - k])
    edges = np.array(sorted(p for p in pts if 0 <= p <= R))
    g, w = np.polynomial.legendre.leggauss(order)
    lo, hi = edges[:-1], edges[1:]
    x = (0.5 * (hi - lo)[:, None] * g[None, :] + 0.5 * (hi + lo)[:, None]).ravel()
    ww = (0.5 * (hi - lo)[:, None] * w[None, :]).ravel()
    return x, ww * x  # rho d rho measure (the theta integral is a constant factor)


def k_theory(ells, a, n_psi=256):
    rho, wr = rho_nodes(a)
    n_i, m2_i, m4_i = chord(rho, a)
    g, w = np.polynomial.legendre.leggauss(n_psi)
    psi = 0.5 * np.pi * (g + 1.0)          # [0, pi]; psi -> -psi symmetric
    wpsi = 0.5 * np.pi * w
    out = np.empty(len(ells))
    for k, ell in enumerate(ells):
        rj = np.sqrt(rho[:, None] ** 2 + ell ** 2 + 2 * rho[:, None] * ell * np.cos(psi[None, :]))
        n_j, m2_j, m4_j = chord(rj, a)
        wt = (wr * n_i)[:, None] * n_j * wpsi[None, :]
        W = wt.sum()
        Q2 = (wt * (m2_i[:, None] + m2_j)).sum()
        Q4 = (wt * (m4_i[:, None] + m4_j + 6 * m2_i[:, None] * m2_j)).sum()
        out[k] = W * Q4 / Q2 ** 2
    return out


def k_zero(a):
    rho, wr = rho_nodes(a, n_seg=200)
    n, m2, m4 = chord(rho, a)
    I0, I2, I4 = (wr * n * n).sum(), (wr * n * n * m2).sum(), (wr * n * n * (m4 + 3 * m2 * m2)).sum()
    return I0 * I4 / (2 * I2 ** 2)


# ---------------- pipeline check (pure bulk, real estimator) ----------------
def k_pipeline(width):
    rng = np.random.default_rng(POINT_SEED)
    p = random_points_in_shell(N_SIM, R - width, R, rng)
    y, z, v = p[:, 1], p[:, 2], p[:, 0] / np.linalg.norm(p, axis=1)
    lc, e, dvb, _ = compute_bins(y, z, v, bin_resolution=20, min_separation=RMIN_SWEEP[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        *_, kk = compute_pdf_and_statistics(y, z, v, np.zeros_like(v), lc, e, dvb, order=1, n_jobs=1)
    return np.asarray(lc, float), np.asarray(kk, float)


def load_or_compute_all(sim_curves):
    lc_ref = sim_curves[0][0]
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        if np.allclose(c["widths"], WIDTHS) and np.allclose(c["lc"], lc_ref):
            return c["k_th"], c["k0"], list(c["pipe"])
    k_th = np.array([k_theory(10 ** lc_ref, R - w) for w in WIDTHS])
    k0 = np.array([k_zero(R - w) for w in WIDTHS])
    pipe = []
    for i in SIM_WIDTH_IDX:
        print(f"pure-bulk pipeline check, width={WIDTHS[i]:.3f}")
        pipe.append(k_pipeline(WIDTHS[i]))
    np.savez(CACHE, widths=WIDTHS, lc=lc_ref, k_th=k_th, k0=k0, pipe=np.array(pipe, dtype=object))
    return k_th, k0, pipe


def main():
    sim = load_or_compute()[0]           # v_bulk/v_turb = 10 curves (N = 100,000)
    lc = sim[0][0]
    k_th, k0, pipe = load_or_compute_all(sim)

    th_curves = [(lc, k) for k in k_th]
    A_t, S_t, w_t, xb_t, rms_t = fit_joint(th_curves, RMIN_SWEEP[0])
    A_s, S_s, w_s, xb_s, rms_s = fit_joint(sim, RMIN_SWEEP[0])
    kp_t, kp_s = A_t + S_t * xb_t, A_s + S_s * xb_s

    print(f"theory  (lambda=inf): A={A_t:.3f} S={S_t:.3f} w={w_t:.3f} rms={rms_t:.3f}")
    print(f"sim     (lambda=10) : A={A_s:.3f} S={S_s:.3f} w={w_s:.3f} rms={rms_s:.3f}")
    print("width   K0_exact  Kplat_th  Kplat_sim   lb_th  lb_th/D   lb_sim  lb_sim/D")
    with OUT_CSV.open("w") as f:
        f.write("width,K0_exact,K_plat_theory,K_plat_sim10,l_b_theory,l_b_sim10\n")
        for i, wd in enumerate(WIDTHS):
            lbt, lbs = 10 ** xb_t[i], 10 ** xb_s[i]
            print(f"{wd:.3f}   {k0[i]:.3f}    {kp_t[i]:.3f}     {kp_s[i]:.3f}     "
                  f"{lbt:.4f}  {lbt / wd:.2f}     {lbs:.4f}  {lbs / wd:.2f}")
            f.write(f"{wd:.5f},{k0[i]:.4f},{kp_t[i]:.4f},{kp_s[i]:.4f},{lbt:.5f},{lbs:.5f}\n")

    # ---------------- figure ----------------
    cmap = plt.get_cmap("viridis")
    lw = np.log10(WIDTHS)
    col = lambda i: cmap((lw[i] - lw.min()) / (lw.max() - lw.min()))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    (a1, a2), (a3, a4) = ax
    ell = 10 ** lc

    for j, i in enumerate(SIM_WIDTH_IDX):
        plc, pk = pipe[j]
        a1.plot(10 ** plc, pk, "o", ms=3.5, color=col(i), alpha=0.6)
        a1.plot(ell, k_th[i], "-", lw=1.8, color=col(i), label=rf"$\Delta={WIDTHS[i]:.2f}$")
    a1.plot([], [], "ko", ms=3.5, label="pipeline, pure bulk")
    a1.plot([], [], "k-", label="semi-analytic")
    a1.set(xscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=r"(a) $\lambda\to\infty$: semi-analytic vs pipeline estimator")
    a1.legend(fontsize=8)

    for i in range(len(WIDTHS)):
        a2.plot(ell, k_th[i], "-", lw=1.6, color=col(i))
        a2.plot(10 ** sim[i][0], sim[i][1], ".", ms=3, color=col(i), alpha=0.5)
    a2.axvline(L_MAX_FIT, color="0.5", ls=":")
    a2.set(xscale="log", ylim=(1.5, 9), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=r"(b) theory $\lambda\to\infty$ (lines) vs simulation $\lambda=10$ (dots)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.LogNorm(WIDTHS.min(), WIDTHS.max()))
    fig.colorbar(sm, ax=a2, label=r"$\Delta r_{\rm shell}/r_{\rm HII}$")

    a3.plot(WIDTHS, kp_t, "-o", color="C3", label=r"theory $\lambda\to\infty$: fitted $K_{\rm plat}$")
    a3.plot(WIDTHS, k0, "--", color="C3", lw=1, label=r"theory: exact $K(\ell\to0)$")
    a3.plot(WIDTHS, kp_s, "-s", color="C0", label=r"simulation $\lambda=10$: fitted $K_{\rm plat}$")
    a3.set(xscale="log", xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$", ylabel=r"$K_{\rm plat}$",
           title="(c) plateau level")
    a3.legend(fontsize=8)

    a4.plot(WIDTHS, 10 ** xb_t / WIDTHS, "-o", color="C3", label=r"theory $\lambda\to\infty$")
    a4.plot(WIDTHS, 10 ** xb_s / WIDTHS, "-s", color="C0", label=r"simulation $\lambda=10$")
    a4.axhline(0.45, color="0.4", ls=":", lw=1)
    a4.set(xscale="log", ylim=(0, 1.2), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=r"$\ell_b/\Delta r_{\rm shell}$", title="(d) bend scale (joint softplus fits)")
    a4.legend(fontsize=8)

    for a in ax.flat:
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
