"""Precomputed geometric averages for the pair-kurtosis formula of an expanding shell with Kolmogorov turbulence.

Model (manuscript_notes/kurtosis_kolmogorov_lambda_derivation.md, sections 1-7), in units of v_turb:

    S2 / v_turb^2 = <D> + lam^2 <dmu^2>
    S4 / v_turb^4 = 3 <D^2> + 6 lam^2 <D dmu^2> + lam^4 <dmu^4>,     K = S4 / S2^2

with the note's recommended combination:
    <D>, <D^2>   exact Abel projection of the shell autocorrelation C(s; Delta)            (section 4)
    <dmu^2> = 2/K_b, <dmu^4> = 4/K_b, K_b(l; Delta) exact sky-plane integral                (section 5)
    <D dmu^2> = (4/K_b) J(l),  J = int_0^1 mu0 D(sqrt(4 mu0^2 + l^2)) dmu0                  (section 6)
and the continuous Kolmogorov structure function with NO inner cutoff (k_max -> inf; section 3 and
manuscript_notes/geometric_averages/01_turbulent_D.md)
    D(s) = (4/3) x^(2/3) [I - G(x)],  x = k_min s,  k_min = 2 pi / L_out,  I = (9/20) Gamma(1/3),
    G(x) = int_0^x t^(-5/3) (1 - sin t / t) dt          (gridfree_kolmogorov.structure_function).
A real H II region's dissipation scale is far below any shell wall, so k_max -> inf is the physical limit;
the grid-free simulations (k_max = 1e5) agree with it to <= 0.004 in K (manuscript_notes/gridfree_turbulence_test.md).

None of these depend on lam, so they are tabulated once on (Delta, L_out, l) and K(l; Delta, lam, L_out)
is then algebra. Lengths are in units of r_HII.

Also tabulated: the realisation (field-seed) scatter of K and ln S2 from the grid-free 8-seed ensemble
(test_gridfree_kolmogorov_model.py pair caches; N = 30,000, k_min = 5.24 i.e. L_out = 1.2, k_max = 1e5;
single realisations, odd terms included), as a function of (Delta, lam, l).
The formula is an ensemble mean; this scatter enters the fit covariance.

POINT SAMPLES ONLY: one LVM fibre is a line-of-sight emission-weighted centroid, not a point; the
pixelised version is still to do (note, section 10).

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/kurtosis_formula_tables.py build      # ~minutes
    python manuscript_pipeline/08_shell_inference/kurtosis_formula_tables.py validate   # vs grid-free 8-seed runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIM_DIR = HERE.parent / "04_flame_kurtosis_3d"
sys.path.insert(0, str(SIM_DIR))
from gridfree_kolmogorov import structure_function  # noqa: E402
TABLE = REPO / "Figures/manuscript/08_shell_inference/data/kurtosis_formula_tables.npz"

DELTA_GRID = np.geomspace(0.005, 1.0, 36)
LOUT_GRID = np.geomspace(0.05, 40.0, 20)
ELL_GRID = np.geomspace(0.001, 1.99, 70)
L_IN = 0.0  # no inner cutoff (k_max = inf); kept in the table file for provenance
X_GRID = np.concatenate(([0.0], np.geomspace(1e-8, 2.0, 6000)))  # line-of-sight depth difference d_x
MU0 = np.linspace(0.0, 1.0, 801)
S_GRID = np.concatenate(([0.0], np.geomspace(1e-5, 2.9, 3000)))


# ------------------------------------------------------------------ geometry
def lens(R1, R2, s):
    """Overlap volume of balls of radii R1, R2 with centres s apart."""
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    out[s <= abs(R1 - R2)] = 4 / 3 * np.pi * min(R1, R2) ** 3
    m = (s > abs(R1 - R2)) & (s < R1 + R2)
    sm = s[m]
    out[m] = np.pi * (R1 + R2 - sm) ** 2 * (sm ** 2 + 2 * sm * (R1 + R2) - 3 * (R1 - R2) ** 2) / (12 * sm)
    return out


def C_shell(s, delta):
    """3D autocorrelation of the shell 1 - delta <= r <= 1 (number of pairs at 3D separation s)."""
    a = 1.0 - delta
    return lens(1, 1, s) - 2 * lens(1, a, s) + lens(a, a, s)


def D_kolmogorov(s, l_out, l_in=L_IN):
    """Structure function of the unit-variance field with a k^-11/3 spectrum between 2pi/l_out and 2pi/l_in
    (closed form; l_in = 0 means no inner cutoff, k_max = inf)."""
    s = np.atleast_1d(np.asarray(s, float))
    kmax = np.inf if l_in <= 0 else 2 * np.pi / l_in
    return structure_function(s, 2 * np.pi / l_out, kmax)


def turb_averages(ell, delta, dfuns):
    """<D>, <D^2> at each l for each structure function in dfuns (exact Abel projection of C)."""
    D1 = np.empty((len(dfuns), len(ell)))
    D2 = np.empty_like(D1)
    for j, l in enumerate(ell):
        s = np.sqrt(l ** 2 + X_GRID ** 2)
        c = C_shell(s, delta)
        W = np.trapezoid(c, X_GRID)
        for i, f in enumerate(dfuns):
            Ds = f(s)
            D1[i, j] = np.trapezoid(c * Ds, X_GRID) / W
            D2[i, j] = np.trapezoid(c * Ds ** 2, X_GRID) / W
    return D1, D2


def cross_J(ell, dfun):
    return np.array([np.trapezoid(MU0 * dfun(np.sqrt(4 * MU0 ** 2 + l ** 2)), MU0) for l in ell])


def assemble(D1, D2, Kb, J, lam):
    """(S2 / v_turb^2, K) from the tabulated averages; arrays broadcast."""
    lam2 = np.asarray(lam, float) ** 2
    s2 = D1 + lam2 * 2.0 / Kb
    s4 = 3.0 * D2 + 6.0 * lam2 * 4.0 * J / Kb + lam2 ** 2 * 4.0 / Kb
    return s2, s4 / s2 ** 2


def _import_sim_modules():
    """(grid-free simulation module, exact bulk curve K_b(l))."""
    sys.path[:0] = [str(HERE.parent / "00_core")]
    import test_gridfree_kolmogorov_model as tg
    from derive_bulk_limit_kurtosis import k_theory
    return tg, k_theory


# ------------------------------------------------------------------ build
SEED_WIDTHS = (0.02, 0.09, 0.3, 1.0)
SEED_LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
SEED_N_POINTS = 30000


def realisation_scatter(tg):
    """Field-seed std of K and ln S2 on the grid-free seed-ensemble l bins (single realisations, odd terms
    included), smoothed over 3 bins in log l."""
    ell = np.sqrt(tg.EDGES[:-1] * tg.EDGES[1:])
    sK = np.full((len(SEED_WIDTHS), len(SEED_LAMS), len(ell)), np.nan)
    sS = np.full_like(sK, np.nan)
    for a, w in enumerate(SEED_WIDTHS):
        c = tg.accumulate(w, SEED_N_POINTS)  # loads the cache
        good = c["n"] >= 200
        n_seeds = c["g2"].shape[0]
        for b, lam in enumerate(SEED_LAMS):
            with np.errstate(invalid="ignore", divide="ignore"):
                ks = np.array([tg.K_sim(c, lam, seeds=slice(s, s + 1), odd=True) for s in range(n_seeds)])
                s2 = (c["g2"] + 2 * lam * c["g1m1"] + lam ** 2 * c["m2"]) / c["n"]  # pair sums
                sK[a, b, good] = np.nanstd(ks[:, good], axis=0, ddof=1)
                sS[a, b, good] = np.nanstd(np.log(s2[:, good]), axis=0, ddof=1)
    # fill bins with too few pairs from the nearest good bin, then smooth in log l
    for arr in (sK, sS):
        for a in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                v = arr[a, b]
                ok = np.isfinite(v)
                v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
                arr[a, b] = np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return ell, sK, sS


def build():
    tg, k_theory = _import_sim_modules()
    dfuns, Dtabs = [], []
    for L in LOUT_GRID:
        Dt = D_kolmogorov(S_GRID, L)
        Dtabs.append(Dt)
        dfuns.append(lambda s, Dt=Dt: np.interp(s, S_GRID, Dt))
    nD, nL, nl = len(DELTA_GRID), len(LOUT_GRID), len(ELL_GRID)
    D1 = np.empty((nD, nL, nl))
    D2 = np.empty_like(D1)
    Kb = np.empty((nD, nl))
    for a, delta in enumerate(DELTA_GRID):
        d1, d2 = turb_averages(ELL_GRID, delta, dfuns)
        D1[a], D2[a] = d1, d2
        Kb[a] = k_theory(ELL_GRID, 1.0 - delta)
        print(f"Delta={delta:.4f}: K_b(l_min)={Kb[a, 0]:.2f}  K_t(l_min, L_out=1.2)="
              f"{3 * d2[9, 0] / d1[9, 0] ** 2:.2f}", flush=True)
    J = np.array([cross_J(ELL_GRID, f) for f in dfuns])
    seed_ell, sK, sS = realisation_scatter(tg)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABLE, delta=DELTA_GRID, l_out=LOUT_GRID, ell=ELL_GRID, l_in=L_IN, D1=D1, D2=D2, Kb=Kb, J=J,
             s_grid=S_GRID, D_of_s=np.array(Dtabs), seed_ell=seed_ell, seed_widths=np.array(SEED_WIDTHS),
             seed_lams=np.array(SEED_LAMS), seed_sigK=sK, seed_sigLnS2=sS)
    print(f"wrote {TABLE}")


# ------------------------------------------------------------------ evaluation
class FormulaModel:
    """Interpolates the tables in log Delta, log L_out, log l (linear in the logs)."""

    def __init__(self, path=TABLE):
        t = np.load(path)
        self.lD, self.lL, self.ll = np.log10(t["delta"]), np.log10(t["l_out"]), np.log10(t["ell"])
        self._D1 = RegularGridInterpolator((self.lD, self.lL, self.ll), np.log(t["D1"]))
        self._D2 = RegularGridInterpolator((self.lD, self.lL, self.ll), np.log(t["D2"]))
        self._Kb = RegularGridInterpolator((self.lD, self.ll), np.log(t["Kb"]))
        self._J = RegularGridInterpolator((self.lL, self.ll), np.log(t["J"]))
        sl = np.log10(t["seed_ell"])
        grid = (np.log10(t["seed_widths"]), np.log10(t["seed_lams"]), sl)
        self._sK = RegularGridInterpolator(grid, t["seed_sigK"])
        self._sS = RegularGridInterpolator(grid, t["seed_sigLnS2"])
        self._sgrid = grid
        self.bounds = {"delta": t["delta"][[0, -1]], "l_out": t["l_out"][[0, -1]], "ell": t["ell"][[0, -1]]}

    def averages(self, ell, delta, l_out):
        """<D>, <D^2>, K_b, J for arrays broadcast to a common shape (l is clipped to the table)."""
        ell, delta, l_out = np.broadcast_arrays(*(np.asarray(v, float) for v in (ell, delta, l_out)))
        x = np.clip(np.log10(ell), self.ll[0], self.ll[-1])
        d = np.clip(np.log10(delta), self.lD[0], self.lD[-1])
        L = np.clip(np.log10(l_out), self.lL[0], self.lL[-1])
        p3 = np.stack((d, L, x), axis=-1)
        D1, D2 = np.exp(self._D1(p3)), np.exp(self._D2(p3))
        Kb = np.exp(self._Kb(np.stack((d, x), axis=-1)))
        J = np.exp(self._J(np.stack((L, x), axis=-1)))
        return D1, D2, Kb, J

    def predict(self, ell, delta, lam, l_out):
        """(S2 / v_turb^2, K) at sky separations ell (units of r_HII)."""
        return assemble(*self.averages(ell, delta, l_out), lam)

    def realisation_sigma(self, ell, delta, lam):
        """Field-seed std of (K, ln S2); clipped to the seed-ensemble ranges in Delta, lam and l."""
        ell, delta, lam = np.broadcast_arrays(*(np.asarray(v, float) for v in (ell, delta, lam)))
        g = self._sgrid
        p = np.stack([np.clip(np.log10(v), gg[0], gg[-1]) for v, gg in zip((delta, lam, ell), g)], axis=-1)
        return self._sK(p), self._sS(p)


# ------------------------------------------------------------------ validation
def validate():
    """(1) direct formula (k_max = inf, L_out = 1.2) vs the grid-free 8-seed runs (odd terms removed);
    (2) table interpolation vs direct evaluation at off-grid points."""
    tg, k_theory = _import_sim_modules()
    ell_all = np.sqrt(tg.EDGES[:-1] * tg.EDGES[1:])
    Dgen = D_kolmogorov(S_GRID, 2 * np.pi / tg.KMIN)
    fgen = lambda s: np.interp(s, S_GRID, Dgen)  # noqa: E731
    lams = (0.0, 0.5, 1, 2, 10)
    print("(1) k_max = inf formula vs grid-free 8 seeds pooled: rms over l <= 0.6")
    print(" Delta  " + "  ".join(f"lam={lam:<4g}" for lam in lams))
    for w in SEED_WIDTHS:
        c = tg.accumulate(w, SEED_N_POINTS)
        good = (c["n"] >= 200) & (ell_all <= 0.6)
        ell = ell_all[good]
        a = {k: v[..., good] for k, v in c.items()}
        (d1,), (d2,) = turb_averages(ell, w, [fgen])
        kb, J = k_theory(ell, 1.0 - w), cross_J(ell, fgen)
        out = []
        for lam in lams:
            _, km = assemble(d1, d2, kb, J, lam)
            out.append(np.sqrt(np.nanmean((tg.K_sim(a, lam) - km) ** 2)))
        print(f" {w:<5g}  " + "  ".join(f"{r:8.3f}" for r in out))

    model = FormulaModel()
    rng = np.random.default_rng(1)
    errK, errS = [], []
    for _ in range(12):
        delta = 10 ** rng.uniform(np.log10(0.006), np.log10(0.9))
        L = 10 ** rng.uniform(np.log10(0.06), np.log10(15))
        lam = 10 ** rng.uniform(-0.7, 1.3)
        ell = np.geomspace(0.003, 1.2, 9)
        Dt = D_kolmogorov(S_GRID, L)
        f = lambda s: np.interp(s, S_GRID, Dt)  # noqa: E731
        (d1,), (d2,) = turb_averages(ell, delta, [f])
        s2d, kd = assemble(d1, d2, k_theory(ell, 1.0 - delta), cross_J(ell, f), lam)
        s2t, kt = model.predict(ell, delta, lam, L)
        errK.append(np.max(np.abs(kt - kd)))
        errS.append(np.max(np.abs(np.log(s2t / s2d))))
    print(f"(2) table vs direct, 12 random (Delta, L_out, lam): max|dK| = {max(errK):.3f}, "
          f"max|d ln S2| = {max(errS):.3f}")


if __name__ == "__main__":
    {"build": build, "validate": validate}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
