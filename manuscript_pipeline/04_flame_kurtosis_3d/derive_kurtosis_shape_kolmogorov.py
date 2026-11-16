#!/usr/bin/env python3
"""Thin-shell model of K(l; lambda) for the correlated Kolmogorov field.

Extends the bulk-limit pair counting to turbulence with the generator's own
structure function S(d) = 2[1 - rho(d)].

Pair classes. At sky separation l << r_HII, the front and back walls project to the
same column map, so pairs split evenly into
  same-wall:     dmu ~ 0,      3D separation sqrt(l^2 + z^2), z = depth offset in the wall
  opposite-wall: dmu = +-2 mu, 3D separation sqrt(4 mu^2 + l^2)
where mu = line-of-sight cosine of the sightline. Each class has pair weight
2 Delta^2 dmu/mu on mu in [mu_cut, 1], with mu_cut = exp(-kappa/2) chosen so that each
class holds Delta^2 kappa pairs, kappa(l) = bulk-limit curve (W = 2 Delta^2 kappa).
(In the decline regime this gives mu_cut = sqrt(l)/2.)

Same-wall depth offsets: two points uniform across a wall of path length
t = min(Delta/mu, t_max), t_max = 2 sqrt(2 Delta) (grazing chord): triangular density
2(t - z)/t^2 on [0, t].

Sums over pairs (common factor 2 Delta^2 dropped):
  N    = kappa
  Sig_S   = int (S_o + <S>_s) dmu/mu          Sig_S2 = int (S_o^2 + <S^2>_s) dmu/mu
  Sig_SB  = int S_o 4 mu^2 dmu/mu             Sig_B  = int 4 mu^2 dmu/mu,  Sig_B2 = int 16 mu^4 dmu/mu
  K = N [3 Sig_S2 + 6 l^2 Sig_SB + l^4 Sig_B2] / [Sig_S + l^2 Sig_B]^2     (l = lambda here)
The lambda -> inf limit returns kappa (to O(mu_cut^2)), as it must.

Inputs: kappa(l) from derive_bulk_limit_kurtosis.k_theory (semi-analytic bulk curve),
S(d) from the generator spectrum. Checks: exact pair-sum ingredients for the same
points (ensemble Gaussian field) and the 8-seed pooled simulation.
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

WIDTHS = [0.02, 0.09, 0.3]
LAMS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
OUT_PNG = kv.BASE_DIR / "kurtosis_shape_kolmogorov_model.png"

_d, _rho = correlation_function()
_S = 2 * (1 - _rho)


def S(d):
    return np.interp(d, _d, _S, right=_S[-1])


def model_sums(ell, width, kappa, n_mu=400, n_z=60):
    # exact chord geometry along the sightline: x_out = mu (R = 1), x_in = sqrt(mu^2 - eps)
    # (eps = 1 - a^2); grazing sightlines (mu^2 < eps) have one chord, split into two halves
    eps = 1 - (1 - width) ** 2
    mu = np.geomspace(1e-4, 1.0, n_mu)
    x_out = mu
    x_in = np.sqrt(np.clip(mu ** 2 - eps, 0, None))
    t = x_out - x_in                                   # wall (half-chord) path length
    # pair weight per class: column^2 = (2t)^2 per unit area, area element rho drho = mu dmu,
    # normalised by eps^2 so that the disc part is dmu/mu (as in the bulk derivation).
    w = 4 * t ** 2 * mu * np.gradient(mu) / eps ** 2
    # separation l removes the lowest-mu (limb) pairs: cut from below so each class holds kappa(l)/2
    # (both classes together = kappa, the normalised pair count)
    target = kappa / 2
    cum = np.cumsum(w[::-1])[::-1]                     # weight in [mu_i, 1]
    if cum[0] > target:
        i_cut = np.searchsorted(-cum, -target)
        w[:i_cut] = 0.0
    else:
        w *= target / cum[0]
    zf = np.linspace(0, 1, n_z)[None, :]               # z = zf * t
    pz = 2 * (1 - zf)                                  # triangular density in zf
    dz = np.gradient(zf[0])
    # foreshortening: a sky step l at angle phi to the radial direction is a step
    # l * sqrt(cos^2 phi / mu^2 + sin^2 phi) along the wall surface
    phi = np.linspace(0, np.pi / 2, 24)
    Ss1 = np.zeros_like(mu)
    Ss2 = np.zeros_like(mu)
    for ph in phi:
        surf2 = ell ** 2 * (np.cos(ph) ** 2 / mu ** 2 + np.sin(ph) ** 2)
        s_same = S(np.sqrt(surf2[:, None] + (zf * t[:, None]) ** 2))
        Ss1 += (s_same * pz * dz).sum(1) / len(phi)
        Ss2 += (s_same ** 2 * pz * dz).sum(1) / len(phi)
    # opposite walls: line-of-sight separation x1 + x2 with x1, x2 uniform on [x_in, x_out]
    # -> triangular on [2 x_in, 2 x_out], peaked at x_in + x_out
    uf = np.linspace(-1, 1, n_z)[None, :]
    pu = (1 - np.abs(uf))
    pu = pu / pu.sum()
    sep = (x_in + x_out)[:, None] + uf * t[:, None]
    s_opp = S(np.sqrt(sep ** 2 + ell ** 2))
    So = (s_opp * pu).sum(1)
    So2 = (s_opp ** 2 * pu).sum(1)
    # velocity-weighted (bulk) sums are disc-dominated: integrate over the full disc, mu in [0, 1],
    # so that lambda -> inf returns kappa exactly (B = 2, B2 = 4, K_b = N B2 / B^2 = kappa)
    mu_d = np.linspace(0, 1, 2001)
    SB = np.trapezoid(4 * mu_d * S(np.sqrt(4 * mu_d ** 2 + ell ** 2)), mu_d)
    return dict(
        N=kappa,
        S=np.sum(w * (So + Ss1)), S2=np.sum(w * (So2 + Ss2)),
        SB=SB, B=2.0, B2=4.0,
    )


def k_model(m, lam):
    if np.isinf(lam):
        return m["N"] * m["B2"] / m["B"] ** 2
    s2 = m["S"] + lam ** 2 * m["B"]
    s4 = 3 * m["S2"] + 6 * lam ** 2 * m["SB"] + lam ** 4 * m["B2"]
    return m["N"] * s4 / s2 ** 2


def k_exact_gaussian(G, lam):
    with np.errstate(invalid="ignore", divide="ignore"):
        if np.isinf(lam):
            return G["B2"] * G["n"] / G["B"] ** 2
        s2 = G["T"] + lam ** 2 * G["B"]
        return (3 * G["S2"] + 6 * lam ** 2 * G["SB"] + lam ** 4 * G["B2"]) * G["n"] / s2 ** 2


def main():
    ell_all = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    cmap = plt.get_cmap("viridis")
    col = {l: cmap(i / (len(LAMS) - 1)) for i, l in enumerate(LAMS)}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for j, w in enumerate(WIDTHS):
        tag = f"w{w:g}".replace(".", "p")
        G = dict(np.load(kv.BASE_DIR / f"_kurtosis_vratio_geom_{tag}_n{kv.N_POINTS}.npz"))
        c = np.load(kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds8.npz")
        M = {k: c[k].sum(0) for k in kv.KEYS}
        good = (G["n"] >= 200) & (ell_all <= 0.6)
        ell = ell_all[good]
        kappa = k_theory(ell, 1.0 - w)
        sums = [model_sums(l, w, k) for l, k in zip(ell, kappa)]
        print(f"Delta={w}:  max |model - exact Gaussian| for l<=0.3, per lambda")
        for lam in LAMS:
            km = np.array([k_model(s, lam) for s in sums])
            ke = k_exact_gaussian(G, lam)[good]
            sel = ell <= 0.3
            print(f"   lambda={lam:>4g}: {np.nanmax(np.abs(km - ke)[sel]):.2f}   "
                  f"(plateau model {km[:4].mean():.2f} vs exact {np.nanmean(ke[:4]):.2f})")
            lab = rf"$\lambda={lam:g}$" if np.isfinite(lam) else r"$\lambda=\infty$"
            ax[0, j].plot(ell, ke, "o", ms=2.5, color=col[lam], alpha=0.6)
            ax[0, j].plot(ell, km, "-", lw=1.6, color=col[lam], label=lab)
            ax[1, j].plot(ell, kv.k_of_lambda(M, lam)[good], "o", ms=2.5, color=col[lam], alpha=0.6)
            ax[1, j].plot(ell, km, "-", lw=1.6, color=col[lam])
        for a, ttl in ((ax[0, j], "points: exact Gaussian-ensemble pair sums"),
                       (ax[1, j], "points: simulation, 8 field seeds")):
            a.axvline(2 * np.exp(-1.5) * w, color="0.5", ls=":", lw=1)
            a.axhline(3, color="0.5", ls=":", lw=1)
            a.set(xscale="log", ylim=(1.5, 7.5), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
                  title=rf"$\Delta={w}$: {ttl}")
            a.grid(True, ls="--", alpha=0.4)
        ax[0, j].legend(fontsize=8, ncol=2)
    fig.suptitle("Lines: thin-shell model (same-wall / opposite-wall pairs, generator S(d), bulk curve "
                 r"$\kappa(\ell)$); dotted: $\ell_b=2e^{-3/2}\Delta$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
