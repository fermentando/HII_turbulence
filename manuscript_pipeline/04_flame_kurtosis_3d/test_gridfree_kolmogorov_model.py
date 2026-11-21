"""Test the finite-lambda kurtosis model against grid-free Kolmogorov shells (no inner cutoff).

Prototype, numpy only. Two tests:

1. `dcheck`: D(s) measured from realisations of `GridFreeKolmogorov` vs the analytic continuum
   D(s) (k_max = 1e5) and the k_max -> inf form.
2. `kurtosis`: for each shell width, all pairs of N random shell points are binned by sky separation
   and accumulated for 8 field seeds. Three estimates of K(ell; lambda) are compared:
     - simulation : realised fields, odd terms removed (equivalent to pooling each seed with g -> -g)
     - pair sums  : exact Gaussian-ensemble average on the same points (D at each pair's 3D separation)
     - model      : semi-analytic formula of `kurtosis_kolmogorov_lambda_derivation.md`
                    (Abel projection of C for <D>, <D^2>; thin-shell 2/K_b, 4/K_b; approximate cross term)
   plus the model with D for k_max -> inf, and term-by-term checks of each geometric average.

Usage:  python3 test_gridfree_kolmogorov_model.py dcheck
        python3 test_gridfree_kolmogorov_model.py kurtosis [--n-points 30000] [--widths 0.02 0.09 0.3 1.0]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from gridfree_kolmogorov import GridFreeKolmogorov, structure_function

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[1] / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions" / "gridfree_turbulence_test"

KMIN = 2.0 * 2 * np.pi / 2.4      # same outer scale as the grid generator (5.24)
KMAX = 1.0e5                      # inner scale 6e-5 r_HII: far below every shell width
N_SEEDS = 8
POINT_SEED = 1234
EDGES = 10 ** np.arange(np.log10(2e-3), np.log10(2.0) + 1e-9, 1 / 20)
L_MAX = 0.6
LAMS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
R = 1.0

S_TAB = np.geomspace(1e-5, 2.0, 4000)
D_TAB = structure_function(S_TAB, KMIN, KMAX)
D_TAB_INF = structure_function(S_TAB, KMIN)


def D_of(s, inf=False):
    return np.interp(np.log(np.maximum(s, 1e-5)), np.log(S_TAB), D_TAB_INF if inf else D_TAB)


def random_points_in_shell(n, r_in, r_out, rng):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    r = (r_in ** 3 + rng.random(n) * (r_out ** 3 - r_in ** 3)) ** (1 / 3)
    return d * r[:, None]


# ---------------------------------------------------------------- test 1: D(s)
def dcheck(n_pairs=3000, n_real=6):
    rng = np.random.default_rng(11)
    s_vals = np.geomspace(1e-3, 1.5, 16)
    acc = np.zeros(len(s_vals))
    acc_sq = np.zeros(len(s_vals))
    for r in range(n_real):
        f = GridFreeKolmogorov(KMIN, KMAX, rng)
        for i, s in enumerate(s_vals):
            p1 = rng.uniform(-1, 1, size=(n_pairs, 3))
            u = rng.normal(size=(n_pairs, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            dv2 = (f(p1) - f(p1 + s * u)) ** 2
            acc[i] += dv2.mean()
            acc_sq[i] += dv2.mean() ** 2
    m = acc / n_real
    err = np.sqrt(np.maximum(acc_sq / n_real - m ** 2, 0) / (n_real - 1))
    print(f"{'s':>8} {'D measured':>12} {'D analytic':>11} {'D k_max=inf':>12} {'D grid k_max=73':>16}")
    rows = []
    for s, mm, ee in zip(s_vals, m, err):
        a, ai, ag = D_of(np.array([s]))[0], D_of(np.array([s]), True)[0], structure_function([s], KMIN, 73.3)[0]
        print(f"{s:8.4f} {mm:8.3f}±{ee:.3f} {a:11.3f} {ai:12.3f} {ag:16.3f}")
        rows.append(dict(s=s, D_measured=mm, D_err=ee, D_analytic=a, D_inf=ai, D_kmax73=ag))
    write_csv(OUT_DIR / "gridfree_D_check.csv", rows)


# ---------------------------------------------------------------- pair accumulation
KEYS_GEOM = ["n", "m2", "m4", "D", "D2", "Dm2", "Di", "Di2", "Dim2"]
KEYS_SEED = ["g2", "g4", "g2m2", "g1m1", "g3m1", "g1m3"]


def accumulate(width, n_points):
    tag = f"w{width:g}".replace(".", "p")
    cache = OUT_DIR / f"_pairs_{tag}_n{n_points}_s{N_SEEDS}_kmax{KMAX:g}.npz"
    if cache.exists():
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    rng = np.random.default_rng(POINT_SEED)
    p = random_points_in_shell(n_points, R - width, R, rng)
    mu = p[:, 0] / np.linalg.norm(p, axis=1)
    t0 = time.time()
    g = np.array([GridFreeKolmogorov(KMIN, KMAX, np.random.default_rng(1000 + s))(p) for s in range(N_SEEDS)])
    print(f"  fields: {time.time() - t0:.0f} s", flush=True)
    x, y, z = p.T
    nb = len(EDGES) - 1
    e2 = EDGES ** 2
    acc = {k: np.zeros(nb) for k in KEYS_GEOM}
    acc.update({k: np.zeros((N_SEEDS, nb)) for k in KEYS_SEED})
    seed_off = (np.arange(N_SEEDS) * nb)[:, None]
    t0 = time.time()
    for i in range(n_points - 1):
        dy, dz = y[i + 1:] - y[i], z[i + 1:] - z[i]
        d2 = dy * dy + dz * dz
        ok = (d2 >= e2[0]) & (d2 < L_MAX ** 2)
        if not ok.any():
            continue
        j = np.nonzero(ok)[0] + i + 1
        b = np.searchsorted(e2, d2[ok], side="right") - 1
        s3 = np.sqrt(d2[ok] + (x[j] - x[i]) ** 2)
        dm = mu[j] - mu[i]
        dm2 = dm * dm
        Ds, Di = D_of(s3), D_of(s3, True)
        for k, v in (("n", None), ("m2", dm2), ("m4", dm2 * dm2), ("D", Ds), ("D2", Ds * Ds), ("Dm2", Ds * dm2),
                     ("Di", Di), ("Di2", Di * Di), ("Dim2", Di * dm2)):
            acc[k] += np.bincount(b, weights=v, minlength=nb)
        dg = g[:, j] - g[:, i][:, None]
        dg2 = dg * dg
        idx = (b[None, :] + seed_off).ravel()
        for k, v in (("g2", dg2), ("g4", dg2 * dg2), ("g2m2", dg2 * dm2), ("g1m1", dg * dm),
                     ("g3m1", dg2 * dg * dm), ("g1m3", dg * dm2 * dm)):
            acc[k] += np.bincount(idx, weights=v.ravel(), minlength=N_SEEDS * nb).reshape(N_SEEDS, nb)
        if i % 5000 == 0 and i:
            print(f"  {i}/{n_points}  {time.time() - t0:.0f} s", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **acc)
    return acc


# ---------------------------------------------------------------- model pieces
def _F(xx, rho):
    at = np.arctan2(xx, rho)
    s = xx * xx + rho * rho
    with np.errstate(invalid="ignore", divide="ignore"):
        f4_tail = np.where(s > 0, rho * rho * xx / (2.0 * s), 0.0)
    return xx, xx - rho * at, xx - 1.5 * rho * at + f4_tail


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
    return np.where(inside, n, 0.0), m2, m4


def rho_nodes(a, order=12):
    d = R - a
    pts = set(np.linspace(0, a, 25))
    for k in np.geomspace(1e-4 * d, d, 25):
        pts.update([a - k, a + k, R - k])
    edges = np.array(sorted(q for q in pts if 0 <= q <= R))
    gl, w = np.polynomial.legendre.leggauss(order)
    lo, hi = edges[:-1], edges[1:]
    xx = (0.5 * (hi - lo)[:, None] * gl[None, :] + 0.5 * (hi + lo)[:, None]).ravel()
    ww = (0.5 * (hi - lo)[:, None] * w[None, :]).ravel()
    return xx, ww * xx


def bulk_moments(ells, a, n_psi=256):
    """Exact <dmu^2>_l, <dmu^4>_l from sky-plane integrals of the chord moments."""
    rho, wr = rho_nodes(a)
    n_i, m2_i, m4_i = chord(rho, a)
    gl, w = np.polynomial.legendre.leggauss(n_psi)
    psi, wpsi = 0.5 * np.pi * (gl + 1.0), 0.5 * np.pi * w
    out = np.empty((len(ells), 2))
    for k, ell in enumerate(ells):
        rj = np.sqrt(rho[:, None] ** 2 + ell ** 2 + 2 * rho[:, None] * ell * np.cos(psi[None, :]))
        n_j, m2_j, m4_j = chord(rj, a)
        wt = (wr * n_i)[:, None] * n_j * wpsi[None, :]
        W = wt.sum()
        out[k] = [(wt * (m2_i[:, None] + m2_j)).sum() / W,
                  (wt * (m4_i[:, None] + m4_j + 6 * m2_i[:, None] * m2_j)).sum() / W]
    return out


def lens(R1, R2, s):
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    out[s <= abs(R1 - R2)] = 4 / 3 * np.pi * min(R1, R2) ** 3
    m = (s > abs(R1 - R2)) & (s < R1 + R2)
    sm = s[m]
    out[m] = np.pi * (R1 + R2 - sm) ** 2 * (sm ** 2 + 2 * sm * (R1 + R2) - 3 * (R1 - R2) ** 2) / (12 * sm)
    return out


def C_shell(s, width):
    a = R - width
    return lens(R, R, s) - 2 * lens(R, a, s) + lens(a, a, s)


X_GRID = np.concatenate(([0.0], np.geomspace(1e-8, 2.0, 8000)))
MU0 = np.linspace(0, 1, 2001)


def model_terms(ells, width, inf=False):
    bm = bulk_moments(ells, R - width)
    out = []
    for ell, (b2, b4) in zip(ells, bm):
        s = np.sqrt(ell ** 2 + X_GRID ** 2)
        c = C_shell(s, width)
        Ds = D_of(s, inf)
        W = np.trapezoid(c, X_GRID)
        kb = b4 / b2 ** 2
        cross = (4 / kb) * np.trapezoid(MU0 * D_of(np.sqrt(4 * MU0 ** 2 + ell ** 2), inf), MU0)
        out.append(dict(D=np.trapezoid(c * Ds, X_GRID) / W, D2=np.trapezoid(c * Ds ** 2, X_GRID) / W,
                        m2=b2, m4=b4, m2_thin=2 / kb, m4_thin=4 / kb, Dm2=cross, Kb=kb))
    return out


def K_from(D1, D2, Dm2, m2, m4, lam):
    if np.isinf(lam):
        return m4 / m2 ** 2
    return (3 * D2 + 6 * lam ** 2 * Dm2 + lam ** 4 * m4) / (D1 + lam ** 2 * m2) ** 2


def K_sim(acc, lam, seeds=slice(None), odd=False):
    n = acc["n"] * (acc["g2"][seeds].shape[0] if acc["g2"][seeds].ndim == 2 else 1)
    g2, g4, g2m2 = (np.atleast_2d(acc[k][seeds]).sum(0) for k in ("g2", "g4", "g2m2"))
    ns = n / acc["n"]
    if np.isinf(lam):
        return acc["n"] * acc["m4"] / acc["m2"] ** 2
    num = g4 + 6 * lam ** 2 * g2m2 + lam ** 4 * ns * acc["m4"]
    den = g2 + lam ** 2 * ns * acc["m2"]
    if odd:
        g1m1, g3m1, g1m3 = (np.atleast_2d(acc[k][seeds]).sum(0) for k in ("g1m1", "g3m1", "g1m3"))
        num = num + 4 * lam * g3m1 + 4 * lam ** 3 * g1m3
        den = den + 2 * lam * g1m1
    return n * num / den ** 2


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def kurtosis_test(widths, n_points):
    ell_c = np.sqrt(EDGES[:-1] * EDGES[1:])
    rows, term_rows, summary = [], [], []
    for w in widths:
        print(f"width {w}", flush=True)
        acc = accumulate(w, n_points)
        good = (acc["n"] >= 200) & (ell_c <= L_MAX)
        ell = ell_c[good]
        mt = model_terms(ell, w)
        mti = model_terms(ell, w, inf=True)
        a = {k: v[..., good] for k, v in acc.items()}
        n = a["n"]
        pair = dict(D=a["D"] / n, D2=a["D2"] / n, Dm2=a["Dm2"] / n, m2=a["m2"] / n, m4=a["m4"] / n)
        pair_i = dict(D=a["Di"] / n, D2=a["Di2"] / n, Dm2=a["Dim2"] / n)
        # measured turbulent averages from the fields (all seeds)
        g2m = a["g2"].sum(0) / (N_SEEDS * n)
        for i, l in enumerate(ell):
            term_rows.append(dict(width=w, ell=l, n_pairs=n[i],
                                  D_fields=g2m[i], D_pairs=pair["D"][i], D_model=mt[i]["D"], D_model_inf=mti[i]["D"],
                                  D2_pairs=pair["D2"][i], D2_model=mt[i]["D2"],
                                  Kt_pairs=3 * pair["D2"][i] / pair["D"][i] ** 2, Kt_model=3 * mt[i]["D2"] / mt[i]["D"] ** 2,
                                  m2_pairs=pair["m2"][i], m2_exact=mt[i]["m2"], m2_thin=mt[i]["m2_thin"],
                                  m4_pairs=pair["m4"][i], m4_exact=mt[i]["m4"], m4_thin=mt[i]["m4_thin"],
                                  Kb_pairs=pair["m4"][i] / pair["m2"][i] ** 2, Kb_exact=mt[i]["Kb"],
                                  Dm2_pairs=pair["Dm2"][i], Dm2_model=mt[i]["Dm2"],
                                  chi_pairs=pair["Dm2"][i] / (pair["D"][i] * pair["m2"][i]),
                                  chi_model_thin=mt[i]["Dm2"] / (mt[i]["D"] * mt[i]["m2_thin"])))
        for lam in LAMS:
            ks = K_sim(a, lam)
            ks_odd = K_sim(a, lam, odd=True)
            per_seed = np.array([K_sim(a, lam, seeds=slice(s, s + 1)) for s in range(N_SEEDS)])
            kp = K_from(pair["D"], pair["D2"], pair["Dm2"], pair["m2"], pair["m4"], lam)
            km = np.array([K_from(t["D"], t["D2"], t["Dm2"], t["m2_thin"], t["m4_thin"], lam) for t in mt])
            km_exact_bulk = np.array([K_from(t["D"], t["D2"], t["Dm2"], t["m2"], t["m4"], lam) for t in mt])
            km_exact_cross = np.array([K_from(t["D"], t["D2"], pc, t["m2"], t["m4"], lam)
                                       for t, pc in zip(mt, pair["Dm2"])])
            kmi = np.array([K_from(t["D"], t["D2"], t["Dm2"], t["m2_thin"], t["m4_thin"], lam) for t in mti])
            for i, l in enumerate(ell):
                rows.append(dict(width=w, lam=lam, ell=l, n_pairs=n[i], K_sim=ks[i], K_sim_with_odd=ks_odd[i],
                                 K_seed_std=per_seed[:, i].std(ddof=1), K_pairs=kp[i], K_model=km[i],
                                 K_model_exact_bulk=km_exact_bulk[i], K_model_exact_cross=km_exact_cross[i],
                                 K_model_kmax_inf=kmi[i]))
            rms = lambda q: float(np.sqrt(np.mean((ks - q) ** 2)))
            plat = slice(0, min(5, len(ell)))
            summary.append(dict(width=w, lam=lam,
                                rms_sim_pairs=rms(kp), rms_sim_model=rms(km), rms_sim_model_exact_bulk=rms(km_exact_bulk),
                                rms_sim_model_exact_cross=rms(km_exact_cross), rms_sim_model_kmax_inf=rms(kmi),
                                plateau_sim=float(ks[plat].mean()), plateau_pairs=float(kp[plat].mean()),
                                plateau_model=float(km[plat].mean()), plateau_seed_std=float(per_seed[:, plat].mean(1).std(ddof=1))))
            s_ = summary[-1]
            print(f"  lam={lam:<5} rms(sim-pairs)={s_['rms_sim_pairs']:.3f}  rms(sim-model)={s_['rms_sim_model']:.3f}  "
                  f"exact-bulk={s_['rms_sim_model_exact_bulk']:.3f}  exact-cross={s_['rms_sim_model_exact_cross']:.3f}  "
                  f"kmax=inf={s_['rms_sim_model_kmax_inf']:.3f} | plateau sim {s_['plateau_sim']:.2f} "
                  f"pairs {s_['plateau_pairs']:.2f} model {s_['plateau_model']:.2f} (seed std {s_['plateau_seed_std']:.2f})",
                  flush=True)
    write_csv(OUT_DIR / "gridfree_kurtosis_curves.csv", rows)
    write_csv(OUT_DIR / "gridfree_geometric_terms.csv", term_rows)
    write_csv(OUT_DIR / "gridfree_kurtosis_summary.csv", summary)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["dcheck", "kurtosis"])
    ap.add_argument("--n-points", type=int, default=30000)
    ap.add_argument("--widths", type=float, nargs="+", default=[0.02, 0.09, 0.3, 1.0])
    args = ap.parse_args()
    if args.test == "dcheck":
        dcheck()
    else:
        kurtosis_test(args.widths, args.n_points)
