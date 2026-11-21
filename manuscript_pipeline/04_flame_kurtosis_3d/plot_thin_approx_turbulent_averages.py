#!/usr/bin/env python3
"""How well the thin-shell closed form for the turbulent plateau averages works as a function of Delta.

thin_approx_turbulent_averages.png, 2 x 3 panels (columns <D>_0, <D^2>_0, K_t(0)):
  top    -- exact (Abel projection of the ball-overlap C(s)), simplified (slab + 1/s pair counts with the
           full D(s), numerical), closed form [2J - (27/20) A D^(2/3)] / [3 + 2 ln(2/D)] and its D^(4/3) analogue
  bottom -- difference from exact

k_max = inf (no inner cutoff), k_min = 5.24. See manuscript_notes/geometric_averages/01_turbulent_D.md.

Run (needs matplotlib):
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_thin_approx_turbulent_averages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from gridfree_kolmogorov import I_INF, structure_function  # noqa: E402
from test_gridfree_kolmogorov_model import C_shell  # noqa: E402

KMIN = 2 * 2 * np.pi / 2.4
A = (4 / 3) * I_INF * KMIN ** (2 / 3)
OUT = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions" / "gridfree_turbulence_test" / "thin_approx_turbulent_averages.png"
DELTAS = np.geomspace(0.005, 1.0, 40)
ELL0 = 1e-4


def D(s):
    return structure_function(np.asarray(s, float), KMIN)


_s = np.geomspace(1e-9, 2, 200001)
J = np.trapezoid(D(_s) / _s, _s)
J2 = np.trapezoid(D(_s) ** 2 / _s, _s)
X = np.concatenate(([0.0], np.geomspace(1e-9, 2.0, 40000)))


def exact(d, ell=ELL0):
    s = np.sqrt(ell ** 2 + X ** 2)
    c, Ds = C_shell(s, d), D(s)
    W = np.trapezoid(c, X)
    return np.trapezoid(c * Ds, X) / W, np.trapezoid(c * Ds ** 2, X) / W


def simplified(d, ell=ELL0):
    out = []
    for f in (lambda q: np.ones_like(q), D, lambda q: D(q) ** 2):
        t = np.geomspace(1e-12, np.sqrt(d * d - ell * ell), 20001)
        q = np.sqrt(ell ** 2 + t ** 2)
        tot = np.trapezoid(4 * np.pi * (d - q / 2) * f(q), t)
        u = np.linspace(np.arccosh(d / ell), np.arccosh(2 / ell), 20001)
        tot += 2 * np.pi * d ** 2 * np.trapezoid(f(ell * np.cosh(u)), u)
        out.append(tot)
    return out[1] / out[0], out[2] / out[0]


def closed(d):
    L = 3 + 2 * np.log(2 / d)
    return (2 * J - 27 / 20 * A * d ** (2 / 3)) / L, (2 * J2 - 27 / 70 * A ** 2 * d ** (4 / 3)) / L


def main():
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    ex = np.array([exact(d) for d in DELTAS])
    si = np.array([simplified(d) for d in DELTAS])
    cl = np.array([closed(d) for d in DELTAS])
    kt = lambda a: 3 * a[:, 1] / a[:, 0] ** 2
    cols = [(r"$\langle D\rangle_0$", ex[:, 0], si[:, 0], cl[:, 0]),
            (r"$\langle D^2\rangle_0$", ex[:, 1], si[:, 1], cl[:, 1]),
            (r"$K_t(0)=3\langle D^2\rangle_0/\langle D\rangle_0^2$", kt(ex), kt(si), kt(cl))]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06,
                                                                              "wspace": 0.28})
    for j, (lab, e, s, c) in enumerate(cols):
        a, r = ax[0, j], ax[1, j]
        a.plot(DELTAS, e, "-", color="k", lw=2.4, label="exact (ball-overlap $C$, Abel)")
        a.plot(DELTAS, s, "--", color="C0", lw=2, label="slab + $1/s$, numerical")
        a.plot(DELTAS, c, ":", color="C3", lw=2.6, label="closed form")
        r.plot(DELTAS, s - e, "--", color="C0", lw=2)
        r.plot(DELTAS, c - e, ":", color="C3", lw=2.6)
        r.axhline(0, color="k", lw=1)
        r.axhspan(-0.01, 0.01, color="0.85", zorder=0)
        for axx in (a, r):
            axx.axvline(2 / 3, color="0.5", ls="-.", lw=1)
            axx.set_xscale("log")
            axx.grid(True, ls="--", alpha=0.4)
        a.set_ylabel(lab)
        r.set_ylabel("approx. $-$ exact")
        r.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
        lim = 0.12 if j < 2 else 0.2
        r.set_ylim(-lim, lim)
    ax[0, 0].legend(loc="lower right", fontsize=10)
    ax[0, 2].text(0.62, 0.95, r"$\Delta=2/3$: $1/s$ range vanishes", transform=ax[0, 2].transAxes, fontsize=9,
                  ha="right", va="top", color="0.4")
    ax[1, 0].text(0.03, 0.08, r"grey band: $\pm0.01$", transform=ax[1, 0].transAxes, fontsize=9, color="0.4")
    fig.suptitle(rf"Thin-shell plateau averages, no inner cutoff ($k_{{\min}}={KMIN:.2f}$; $J={J:.2f}$, $J_2={J2:.2f}$, $A={A:.2f}$)",
                 fontsize=13, y=0.95)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")
    for d, e, c in zip(DELTAS[::4], ex[::4], cl[::4]):
        print(f"Delta={d:.3f}  <D> {e[0]:.3f}/{c[0]:.3f}  <D2> {e[1]:.3f}/{c[1]:.3f}")


if __name__ == "__main__":
    main()
