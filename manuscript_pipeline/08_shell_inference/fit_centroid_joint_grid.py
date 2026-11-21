"""Centroid (one velocity per sightline) shell model, fitted jointly to K(l) and S2(l) of [NII], [SII], [SIII].

Why: an LVM fibre gives the emission-weighted line-of-sight mean velocity, not a point sample. The point-sample
formula (kurtosis_formula_tables.py) cannot produce S2(l) steeper than l^0.26 over l = 0.05-0.5, while the
data have 0.43-0.94 (formula_grid_joint_v/s2_slopes_by_line.png). Line-of-sight averaging steepens S2 below
the emitting depth, which is what makes S2 sensitive to Delta.

Model (lengths in r_HII(catalogue); velocities in v_turb):
  emissivity: uniform in the shell 1 - Delta <= r <= 1, front half (z > 0) weight 1, back half weight a
              (a = 1: symmetric shell, a = 0: only the near side visible, e.g. blister / extinction)
  sightline at sky position p, |p| = rho < 1: weights w_p(z) on a z grid (exact chord overlaps per cell)
  turbulence: unit-variance Gaussian LOS field, correlation xi(s) = 1 - D(s)/2, Kolmogorov D(s) between
              L_out (= FOV, as in the point model) and L_IN (kurtosis_formula_tables.D_kolmogorov; L_IN = 0:
              no inner cutoff, k_max -> inf, the closed-form D of manuscript_notes/kurtosis_K_lambda_derivation.md)
  centroid:   V_p = sum_z w_p(z) [g(p, z) + lam mu(p, z)],  mu = z / r  (radial expansion at v_bulk)
For a pair (p1, p2) at sky separation l, the turbulent increment is Gaussian with variance
    T = w1' X0 w1 + w2' X0 w2 - 2 w1' X_l w2,   X_l[i, j] = xi(sqrt(l^2 + (z_i - z_j)^2))
and the bulk increment is deterministic, m = lam_eff (G1 - G2),  G_p = sum w_p mu / ((1 - a)/(1 + a))
(normalised so that lam_eff = v_bulk (1 - a)/(1 + a) / v_turb is the net projected expansion at the centre
of a thin shell; a symmetric shell has no bulk signal in the centroids). Over pairs at l:
    S2 = <T> + lam^2 <m^2>,   S4 = 3<T^2> + 6 lam^2 <T m^2> + lam^4 <m^4>
so five pair moments per (l, Delta, a) give K and S2 for every lam (as in the point formula).
Pairs: N_PAIRS Monte-Carlo sightline pairs per l, uniform over the disc rho < 1 (fixed seed: common random
numbers across parameters). Not included: the plane subtracted from the data.
--beam: fibre footprint (35.3" top hat, = floor x 35.3/37 in r_HII): the turbulent correlation between two
fibres is averaged over N_OFF point offsets within the fibres; w(z) and the bulk centroid are taken at the
fibre centre. It steepens S2 below ~ the beam (so, for a given slope, it lowers the inferred Delta).
--fit-noise: N2_X = c_X^2 x catalogue N2 with c_X free per line on C_GRID and a log-normal prior
(width SIG_LN_C) around the overlap-pair calibration NOISE_C[X]; profiled out like a and M.
--noise-kurt: noise 4th moment K_N N2^2 instead of 3 N2^2; K_N = pooled kurtosis of duplicate-fibre velocity
differences (measure_overlap_noise.py) for Orion and Rosette, geometric mean of the two per line elsewhere
(suffix _noisekurt).
--mixing-meas: <T^2> -> M_meas <T^2> (turbulent part of S4 only; bulk is deterministic), M_meas = measured
width-mixing factor per line (compute_multifloor_slopes, in the formula data; suffix _mixmeas).
--no-noise: N2 = 0 (no noise floor in the model; output suffix _nonoise).
--mixing: S4 -> M S4, M free per line (excess K at all l from dispersion varying across the field).
--lam-min L: prior lambda_eff >= L (output suffix _lammin<L>); below ~1 the bulk term is not separable from
turbulence.
--fix-a A: fix the back-side weight a (one of A_GRID) instead of profiling it; with no --mixing/--fit-noise
the only free parameters are Delta and lam_eff per line plus the shared v_turb (S2 normalisation).

Fit: as fit_kurtosis_s2_joint_grid.py: y = [K, ln S2] per line, shrunk jackknife covariance + model error,
fixed noise floor N2 = NOISE_C^2 x catalogue N2, shared v_turb across lines (--share v: Delta, lam, a per
line; --share v_lam: lam shared). Parameters per line: Delta (grid), lam_eff (grid), a (grid, profiled).

Outputs: Figures/manuscript/08_shell_inference/centroid_joint_<share>/
  centroid_moments/<region>_<LINE>.npz (cache), joint_maps.png, joint_delta_profiles.png, joint_curves.png,
  joint_summary.csv

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_centroid_joint_grid.py [--share v|v_lam] [--mixing]
        [--beam] [--fit-noise] [--region R]
(output folder suffixes: _mixing, _beam, _noisefit)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_kurtosis_s2_joint_grid import NOISE_C, load, prange  # noqa: E402
from kurtosis_formula_tables import L_IN, D_kolmogorov  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

OUT_BASE = Path("Figures/manuscript/08_shell_inference")
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}
NZ = 240
N_PAIRS = 1500
D_GRID = np.geomspace(0.005, 1.0, 28)
A_GRID = np.array([0.0, 0.4, 0.7])
LAM_GRID = np.geomspace(0.01, 30.0, 90)
LNV2_GRID = np.log(np.geomspace(0.2, 40.0, 140) ** 2)
M_GRID = np.geomspace(1.0, 5.0, 10)  # --mixing: S4 -> M S4 (dispersion varying across the field), free per line
C_GRID = np.geomspace(0.1, 1.0, 12)  # --fit-noise: N2 = c^2 x catalogue N2, c free per line
SIG_LN_C = 0.35  # log-normal prior width on c around the overlap calibration NOISE_C[line]
FIBRE_DIAM_ARCSEC, FLOOR_ARCSEC = 35.3, 37.0
N_OFF = 64  # beam-offset samples for the fibre-footprint average
ZE = np.linspace(-1.0, 1.0, NZ + 1)
ZC = 0.5 * (ZE[:-1] + ZE[1:])
S_TAB = np.concatenate(([0.0], np.geomspace(1e-5, 3.0, 2500)))


NOISE_KURT_CSV = OUT_BASE / "overlap_noise" / "overlap_noise_sep0.25.csv"


def noise_kurtosis(region, line):
    """(K_N, source): own duplicate-pair kurtosis if measured, else geometric mean over measured regions."""
    with open(NOISE_KURT_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["line"] == line and r.get("K_dv")]
    own = [float(r["K_dv"]) for r in rows if r["region"] == region]
    if own:
        return own[0], "measured"
    return float(np.exp(np.mean(np.log([float(r["K_dv"]) for r in rows])))), "borrowed"


# ------------------------------------------------------------------ geometry
def weights(rho, delta, a):
    """Normalised emission weights (n, NZ) and bulk centroid G (n,) for sightlines at sky radii rho."""
    zo = np.sqrt(np.clip(1 - rho ** 2, 0, None))[:, None]
    zi = np.sqrt(np.clip((1 - delta) ** 2 - rho ** 2, 0, None))[:, None]
    lo, hi = ZE[None, :-1], ZE[None, 1:]
    front = np.clip(np.minimum(hi, zo) - np.maximum(lo, zi), 0, None)
    back = np.clip(np.minimum(hi, -zi) - np.maximum(lo, -zo), 0, None)
    w = front + a * back
    w /= w.sum(1, keepdims=True)
    mu = ZC[None, :] / np.sqrt(rho[:, None] ** 2 + ZC[None, :] ** 2)
    G = (w * mu).sum(1) / ((1 - a) / (1 + a))
    return w, G


def sample_pairs(ell, rng):
    """N_PAIRS sightline pairs at sky separation ell, both inside the unit disc (rejection sampling)."""
    out1, out2 = [], []
    n = 0
    while n < N_PAIRS:
        m = 4 * N_PAIRS
        r = np.sqrt(rng.random(m))
        t = 2 * np.pi * rng.random(m)
        p1 = np.column_stack((r * np.cos(t), r * np.sin(t)))
        th = 2 * np.pi * rng.random(m)
        p2 = p1 + ell * np.column_stack((np.cos(th), np.sin(th)))
        ok = np.hypot(*p2.T) < 1
        out1.append(p1[ok])
        out2.append(p2[ok])
        n += ok.sum()
    return np.hypot(*np.vstack(out1)[:N_PAIRS].T), np.hypot(*np.vstack(out2)[:N_PAIRS].T)


def beam_offsets(beam, seed=7):
    """N_OFF vector differences b1 - b2 of two points uniform in a fibre of diameter `beam` (r_HII units)."""
    if beam <= 0:
        return np.zeros((1, 2))
    rng = np.random.default_rng(seed)
    r = 0.5 * beam * np.sqrt(rng.random((2, N_OFF)))
    t = 2 * np.pi * rng.random((2, N_OFF))
    return np.column_stack((r[0] * np.cos(t[0]) - r[1] * np.cos(t[1]), r[0] * np.sin(t[0]) - r[1] * np.sin(t[1])))


def centroid_moments(ell, l_out, beam=0.0, seed=1):
    """Pair moments <T>, <T^2>, <m2>, <T m2>, <m4> (m2 = (G1-G2)^2) on (ell, D_GRID, A_GRID).
    Fibre footprint: the turbulent correlation between two fibres is averaged over points uniform in each
    fibre (top hat of diameter `beam`); emission weights w(z) and the bulk centroid are taken at the
    fibre centre (they vary on scales >> beam except at the limb)."""
    Dtab = D_kolmogorov(S_TAB, l_out, L_IN)
    xi = lambda s: 1 - 0.5 * np.interp(s, S_TAB, Dtab)  # noqa: E731
    dz2 = (ZC[:, None] - ZC[None, :]) ** 2
    off = beam_offsets(beam)
    xib = lambda l: np.mean([xi(np.sqrt((l + o[0]) ** 2 + o[1] ** 2 + dz2)) for o in off], axis=0)  # noqa: E731
    X0 = xib(0.0)
    rng = np.random.default_rng(seed)
    mom = np.zeros((5, len(ell), len(D_GRID), len(A_GRID)))
    for i, l in enumerate(ell):
        r1, r2 = sample_pairs(l, rng)
        Xl = xib(l)
        for j, d in enumerate(D_GRID):
            for k, a in enumerate(A_GRID):
                w1, G1 = weights(r1, d, a)
                w2, G2 = weights(r2, d, a)
                T = (np.einsum("pi,ij,pj->p", w1, X0, w1, optimize=True)
                     + np.einsum("pi,ij,pj->p", w2, X0, w2, optimize=True)
                     - 2 * np.einsum("pi,ij,pj->p", w1, Xl, w2, optimize=True))
                T = np.clip(T, 0, None)
                m2 = (G1 - G2) ** 2
                mom[:, i, j, k] = T.mean(), (T ** 2).mean(), m2.mean(), (T * m2).mean(), (m2 ** 2).mean()
    return mom


def model_s2_K(mom, lam):
    """S2 / v_turb^2 and K on (..., lam): mom has leading axis 5."""
    T1, T2, M2, TM, M4 = (m[..., None] for m in mom)
    l2 = lam ** 2
    s2 = T1 + l2 * M2
    s4 = 3 * T2 + 6 * l2 * TM + l2 ** 2 * M4
    return s2, s4 / s2 ** 2


# ------------------------------------------------------------------ likelihood
def m2lnL(data, mom, m_grid, c_grid, c0):
    """chi2 (+ noise prior) profiled over (a, M, c): returns min on (Delta, lam, lnv2) and the flat argmin
    index into (A_GRID, m_grid, c_grid). mom: (5, n_ell, nD, nA). Covariance fixed (no realisation term)."""
    Linv = np.linalg.inv(np.linalg.cholesky(data["C0"]))
    y = np.r_[data["K"], data["lnS2"]]
    s2u, Ku = model_s2_K(np.moveaxis(mom, 1, -1), LAM_GRID)  # (nD, nA, n_ell, nLam)
    s2u, Ku = np.swapaxes(s2u, -1, -2)[:, :, None], np.swapaxes(Ku, -1, -2)[:, :, None]  # (nD, nA, 1, nLam, n_ell)
    Mg = m_grid[None, None, :, None, None]
    nD, nA, nL = s2u.shape[0], s2u.shape[1], s2u.shape[3]
    best = np.full((nD, nL, len(LNV2_GRID)), np.inf)
    arg = np.zeros(best.shape, int)
    for ic, c in enumerate(c_grid):
        N2 = c ** 2 * data["N2_cat"]
        prior = (np.log(c / c0) / SIG_LN_C) ** 2 if len(c_grid) > 1 else 0.0
        for k, u in enumerate(LNV2_GRID):
            S2s = np.exp(u) * s2u
            S2o = S2s + N2
            Ko = (Mg * Ku * S2s ** 2 + 6 * S2s * N2 + data.get("K_N", 3.0) * N2 ** 2) / S2o ** 2
            r = y - np.concatenate((Ko, np.broadcast_to(np.log(S2o), Ko.shape)), axis=-1)
            chi = ((r @ Linv.T) ** 2).sum(-1) + prior  # (nD, nA, nM, nLam)
            flat = chi.transpose(0, 3, 1, 2).reshape(nD, nL, -1)
            j = flat.argmin(-1)
            v = np.take_along_axis(flat, j[..., None], -1)[..., 0]
            upd = v < best[..., k]
            best[..., k] = np.where(upd, v, best[..., k])
            arg[..., k] = np.where(upd, j + ic * nA * len(m_grid), arg[..., k])
    return best, arg


def get_moments(region, line, data, cache_dir, beam):
    p = cache_dir / f"{region}_{line}{'_beam' if beam > 0 else ''}.npz"
    if p.exists():
        c = np.load(p)
        if (np.allclose(c["ell"], data["ell"]) and np.isclose(float(c["l_out"]), data["L_out"])
                and np.allclose(c["delta"], D_GRID) and np.allclose(c["a"], A_GRID)
                and np.isclose(float(c["beam"]) if "beam" in c else 0.0, beam)
                and np.isclose(float(c["l_in"]) if "l_in" in c else 0.005, L_IN)):  # old caches: l_in = 0.005
            return c["mom"]
    t0 = time.time()
    mom = centroid_moments(data["ell"], data["L_out"], beam)
    np.savez(p, mom=mom, ell=data["ell"], l_out=data["L_out"], delta=D_GRID, a=A_GRID, nz=NZ, n_pairs=N_PAIRS,
             beam=beam, l_in=L_IN)
    print(f"  moments {region} {line}: {time.time() - t0:.0f} s", flush=True)
    return mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", choices=("v", "v_lam"), default="v")
    ap.add_argument("--region", default=None)
    ap.add_argument("--mixing", action="store_true", help="free S4 factor M per line (absorbs excess K)")
    ap.add_argument("--beam", action="store_true", help="average over the 35.3\" fibre footprint")
    ap.add_argument("--fit-noise", action="store_true",
                    help="noise factor c free per line with a log-normal prior around the overlap calibration")
    ap.add_argument("--fix-a", type=float, default=None, choices=[float(x) for x in A_GRID],
                    help="fix the back-side weight a (default: profiled over A_GRID)")
    ap.add_argument("--noise-kurt", action="store_true",
                    help="noise 4th moment K_N N2^2 with K_N from tile-overlap duplicates (measure_overlap_noise.py)")
    ap.add_argument("--mixing-meas", action="store_true",
                    help="<T^2> -> M_meas <T^2> (turbulent S4 only), M_meas measured (compute_multifloor_slopes)")
    ap.add_argument("--no-noise", action="store_true", help="no noise floor in the model (N2 = 0)")
    ap.add_argument("--lam-min", type=float, default=None,
                    help="prior lambda_eff >= LAM_MIN (lambda < 1: bulk not separable from turbulence)")
    args = ap.parse_args()
    global LAM_GRID
    if args.lam_min is not None:
        LAM_GRID = LAM_GRID[LAM_GRID >= args.lam_min]
    a_idx = np.arange(len(A_GRID)) if args.fix_a is None else np.flatnonzero(np.isclose(A_GRID, args.fix_a))
    a_grid = A_GRID[a_idx]
    m_grid = M_GRID if args.mixing else np.array([1.0])
    m_grids = {}
    out_dir = OUT_BASE / (f"centroid_joint_{args.share}" + ("_mixing" if args.mixing else "")
                          + ("_mixmeas" if args.mixing_meas else "") + ("_noisekurt" if args.noise_kurt else "")
                          + ("_beam" if args.beam else "") + ("_noisefit" if args.fit_noise else "")
                          + ("_nonoise" if args.no_noise else "")
                          + (f"_a{args.fix_a:g}" if args.fix_a is not None else "")
                          + (f"_lammin{args.lam_min:g}" if args.lam_min is not None else ""))
    cache = OUT_BASE / "centroid_moments"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    from fit_kurtosis_formula import DATA_DIR

    regions = sorted({p.name.rsplit("_", 3)[0] for p in DATA_DIR.glob("*_formula_data.npz")})
    if args.region:
        regions = [args.region]
    results, store = [], {}
    for region in regions:
        datas = {X: load(region, X, "overlap") for X in LINES}
        datas = {X: d for X, d in datas.items() if d is not None}
        if len(datas) < 2:
            print(f"{region}: fewer than 2 lines usable; skipped")
            continue
        prof, args_idx, moms, c_grids = {}, {}, {}, {}
        for X, d in datas.items():
            raw = np.load(DATA_DIR / f"{region}_{X}_formula_data.npz")
            beam = float(raw["floor_over_rhii"]) * FIBRE_DIAM_ARCSEC / FLOOR_ARCSEC if args.beam else 0.0
            d["beam"] = beam
            d["N2_cat"] = float(raw["N2"])
            c_grids[X] = (np.array([0.0]) if args.no_noise else
                          C_GRID if args.fit_noise else np.array([NOISE_C[X]]))
            moms[X] = get_moments(region, X, d, cache, beam)[..., a_idx]
            m_grids[X] = m_grid
            if args.mixing_meas:  # width mixing acts on the turbulent term only: <T^2> -> M_meas <T^2>
                moms[X] = moms[X].copy()
                moms[X][1] *= float(raw["mixing_factor"])
                d["M_meas"] = float(raw["mixing_factor"])
            if args.noise_kurt:
                d["K_N"], d["K_N_src"] = noise_kurtosis(region, X)
            prof[X], args_idx[X] = m2lnL(d, moms[X], m_grids[X], c_grids[X], NOISE_C[X])  # (nD, nLam, nV)
        if args.share == "v":
            own = {X: p.min(axis=(0, 1)) for X, p in prof.items()}
            tot = {X: prof[X] + sum(own[Y] for Y in prof if Y != X)[None, None] for X in prof}
        else:
            own = {X: p.min(axis=0) for X, p in prof.items()}
            tot = {X: prof[X] + sum(own[Y] for Y in prof if Y != X)[None] for X in prof}
        gmin = min(t.min() for t in tot.values())
        maps = {X: tot[X].min(axis=2) - gmin for X in tot}
        vprof = sum(own.values()) if args.share == "v" else sum(own.values()).min(0)
        v_lo, v_hi, _ = prange(vprof - vprof.min(), np.exp(LNV2_GRID / 2))
        best = {}
        for X, d in datas.items():
            iD, iL, iv = np.unravel_index(np.argmin(tot[X]), tot[X].shape)
            iC, iA, iM = np.unravel_index(args_idx[X][iD, iL, iv], (len(c_grids[X]), len(a_grid), len(m_grids[X])))
            n = len(d["ell"])
            c_best = c_grids[X][iC]
            c2 = prof[X][iD, iL, iv] - ((np.log(c_best / NOISE_C[X]) / SIG_LN_C) ** 2
                                         if args.fit_noise and not args.no_noise else 0.0)
            d_lo, d_hi, d_ok = prange(maps[X].min(1), D_GRID)
            l_lo, l_hi, l_ok = prange(maps[X].min(0), LAM_GRID)
            flags = [f for f, ok in (("Delta 1-sigma hits grid edge", d_ok), ("lambda 1-sigma hits grid edge", l_ok))
                     if not ok]
            if c2 > 2 * n + 3 * np.sqrt(4 * n):
                flags.append("poor fit")
            best[X] = dict(iD=iD, iA=iA, a=a_grid[iA], iL=iL, iv=iv, M=m_grids[X][iM], N2=c_best ** 2 * d["N2_cat"])
            r = {"region": region, "line": X, "share": args.share, "n_bins": n, "L_out_over_rhii": d["L_out"],
                 "beam_over_rhii": d["beam"], "noise_c": c_best, "noise_N2": c_best ** 2 * d["N2_cat"],
                 "noise_K_N": d.get("K_N", 3.0), "noise_K_N_source": d.get("K_N_src", "gaussian"), "v_turb": np.exp(LNV2_GRID[iv] / 2), "v_turb_1sig_lo": v_lo,
                 "v_turb_1sig_hi": v_hi, "best_delta": D_GRID[iD], "delta_1sig_lo": d_lo, "delta_1sig_hi": d_hi,
                 "best_lambda_eff": LAM_GRID[iL], "lambda_1sig_lo": l_lo, "lambda_1sig_hi": l_hi,
                 "best_back_weight_a": a_grid[iA], "best_mixing_M": d.get("M_meas", m_grids[X][iM]), "chi2_line": c2, "n_data_line": 2 * n, "flags": "; ".join(flags)}
            results.append(r)
            print(f"{region:15s} {X:4s} v_turb={r['v_turb']:5.2f} [{v_lo:.2f},{v_hi:.2f}] Delta={D_GRID[iD]:.3f} "
                  f"[{d_lo:.3f},{d_hi:.3f}] lam_eff={LAM_GRID[iL]:.2f} [{l_lo:.2f},{l_hi:.2f}] a={a_grid[iA]:.1f} M={m_grids[X][iM]:.2f} c={c_best:.2f} "
                  f"chi2={c2:.1f}/{2 * n} {r['flags']}", flush=True)
            np.savez(out_dir / f"{region}_{X}_map.npz", delta_grid=D_GRID, lambda_grid=LAM_GRID, dm2lnL=maps[X])
        store[region] = (datas, maps, best, moms)

    cols = list(results[0])
    with open(out_dir / "joint_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])
    plot(store, out_dir, args)
    print(f"wrote {out_dir}")


def plot(store, out_dir, args):
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    regions = list(store)
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    tag = (r"centroid model, shared $v_{\rm turb}$" + (r", $\lambda$ per line" if args.share == "v" else r" and $\lambda$")
           + (r", free $M$" if args.mixing else "") + (r", $M=M_{\rm meas}$" if args.mixing_meas else "")
           + (r", $K_N$ meas." if args.noise_kurt else "") + (", fibre footprint" if args.beam else "")
           + (", noise fitted" if args.fit_noise else "") + (", no noise" if args.no_noise else "")
           + (fr", $a$={args.fix_a:g} fixed" if args.fix_a is not None else "")
           + (fr", $\lambda\geq${args.lam_min:g}" if args.lam_min is not None else ""))
    DDg, LLg = np.meshgrid(D_GRID, LAM_GRID, indexing="ij")

    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.9 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, region in zip(axes.flat, regions):
        datas, maps, best, _ = store[region]
        for X in datas:
            c, b = LINE_COLORS[X], best[X]
            ax.contourf(DDg, LLg, maps[X], levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(DDg, LLg, maps[X], levels=[2.30, 6.18], colors=[c], linestyles=[LINESTYLES[X]],
                       linewidths=[1.5, 0.8])
            ax.plot(D_GRID[b["iD"]], LAM_GRID[b["iL"]], "*", ms=11, color=c, mec="white", mew=0.6,
                    label=fr"{LINE_LABELS[X]}: $\Delta$={D_GRID[b['iD']]:.2g}, $a$={b['a']:.1f}")
        v = np.exp(LNV2_GRID[best[next(iter(datas))]["iv"]] / 2)
        ax.set(xscale="log", yscale="log")
        ax.set_title(fr"{region.replace('_', ' ')}  ($v_{{\rm turb}}$={v:.1f} km/s)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\lambda_{\rm eff}$ = net $v_{\rm bulk}/v_{\rm turb}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle(fr"Joint $K(\ell)$ + $S_2(\ell)$ fit ({tag}): filled = 1$\sigma$, thin = 2$\sigma$", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_maps.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.6 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, region in zip(axes.flat, regions):
        datas, maps, _, _ = store[region]
        for X in datas:
            ax.plot(D_GRID, maps[X].min(1), LINESTYLES[X], color=LINE_COLORS[X], lw=1.8, label=LINE_LABELS[X])
        ax.axhline(1, color="0.5", lw=0.7, ls="--")
        ax.set(xscale="log", ylim=(0, 12))
        ax.set_title(region.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7.5)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta(-2\ln L)$ profiled")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle(fr"Profile likelihood of $\Delta$ per line ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_delta_profiles.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2 * nr, nc, figsize=(4.4 * nc, 3.0 * 2 * nr), squeeze=False)
    for i, region in enumerate(regions):
        aK, aS = axes[2 * (i // nc), i % nc], axes[2 * (i // nc) + 1, i % nc]
        datas, _, best, moms = store[region]
        for X, d in datas.items():
            c, b = LINE_COLORS[X], best[X]
            s2u, Ku = model_s2_K(moms[X][:, :, b["iD"], b["iA"]], np.array([LAM_GRID[b["iL"]]]))
            S2s = np.exp(LNV2_GRID[b["iv"]]) * s2u[:, 0]
            S2o = S2s + b["N2"]
            Ko = (b["M"] * Ku[:, 0] * S2s ** 2 + 6 * S2s * b["N2"] + d.get("K_N", 3.0) * b["N2"] ** 2) / S2o ** 2
            aK.errorbar(d["ell"], d["K"], d["sdK"], fmt="o", ms=3.5, color=c, capsize=2)
            aK.plot(d["ell"], Ko, LINESTYLES[X], color=c, lw=1.4)
            aS.errorbar(d["ell"], d["S2"], d["S2"] * d["sdlS"], fmt="o", ms=3.5, color=c, capsize=2,
                        label=fr"{LINE_LABELS[X]} $\Delta$={D_GRID[b['iD']]:.2g} $\lambda$={LAM_GRID[b['iL']]:.2g}")
            aS.plot(d["ell"], S2o, LINESTYLES[X], color=c, lw=1.4)
        aK.set(xscale="log", title=region.replace("_", " "), ylabel=r"$K$")
        aK.axhline(3, color="0.5", ls="--", lw=0.8)
        aS.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$S_2$ [km$^2$ s$^{-2}$]")
        aS.legend(fontsize=6.5)
    for j in range(len(regions), nr * nc):
        axes[2 * (j // nc), j % nc].axis("off")
        axes[2 * (j // nc) + 1, j % nc].axis("off")
    fig.suptitle(fr"Observed $K$ and $S_2$ with the joint best fit ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
