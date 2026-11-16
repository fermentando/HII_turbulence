"""Velocity-noise calibration from tile-overlap duplicate fibres, all regions and lines.

Overlapping pointings put fibres from two tiles at (nearly) the same sky position. For a pair at
separation s << a (a = lattice spacing), the velocity difference is measurement noise (plus a tiny
turbulent S2(s) and any inter-pointing offset). Measured per region/line:
  c      = sqrt( <dv^2> / <sig1^2 + sig2^2> )       catalogue-error rescaling (NOISE_C in the fits)
  c_mad  = 1.4826 MAD(dv) / sqrt(median(sig1^2 + sig2^2))   robust version (c >> c_mad: heavy tails)
  K_dv   = <dv^4> / <dv^2>^2                         pooled noise kurtosis, what enters S4 (Gaussian: 3,
                                                     plus spread from pair-to-pair sigma variation)
  K_norm = kurtosis of dv / sqrt(sig1^2 + sig2^2)    shape test with sigma variation divided out
  N2_meas = <dv^2>                                   direct noise floor for S2 (cf. c^2 x catalogue N2)
Cuts as load_line_data: flux S/N > SNR_MIN for both fibres, 10-sigma MAD velocity clip on each fibre
(region-wide), no gradient subtraction (irrelevant at s << a). Duplicates: s < SEP_FRAC x a.
Bootstrap (pairs) 1-sigma errors on c and K.

Run from the repo root:  python manuscript_pipeline/08_shell_inference/measure_overlap_noise.py [--sep-frac 0.25]
Output: Figures/manuscript/08_shell_inference/overlap_noise/overlap_noise.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from vsf_module import VSF2D, get_heliocentric_distance  # noqa: E402

OUT = Path("Figures/manuscript/08_shell_inference/overlap_noise")
LINES = ("nii", "sii", "siii")
SNR_MIN, CLIP = 5.0, 10.0
N_BOOT = 500


def stats(dv, s2):
    m2 = np.mean(dv ** 2)
    z = dv / np.sqrt(s2)
    return (np.sqrt(m2 / s2.mean()), 1.4826 * np.median(np.abs(dv - np.median(dv))) / np.sqrt(np.median(s2)),
            np.mean(dv ** 4) / m2 ** 2, np.mean(z ** 4) / np.mean(z ** 2) ** 2, m2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sep-frac", type=float, default=0.25, help="duplicate if s < SEP_FRAC x lattice spacing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    for fn in sorted(Path(".").glob("LVM_*.fits")):
        region = fn.stem[4:]
        t = fits.open(fn)[1].data
        ra, dec = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
        v = VSF2D(ra, dec, np.zeros(len(ra)), dist=get_heliocentric_distance(fn.name))
        xy = np.column_stack((v.x_physical, v.y_physical))
        tree = cKDTree(xy)
        a = np.median(tree.query(xy, 2)[0][:, 1])
        pairs = tree.query_pairs(args.sep_frac * a, output_type="ndarray")
        for ln in LINES:
            vel, err = np.asarray(t[f"vel_{ln}"], float), np.asarray(t[f"vel_{ln}_err"], float)
            fl, fe = np.asarray(t[f"flux_{ln}"], float), np.asarray(t[f"flux_{ln}_err"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = np.isfinite(vel) & np.isfinite(err) & (err > 0) & (fl / fe > SNR_MIN)
            med = np.median(vel[ok])
            ok &= np.abs(vel - med) < CLIP * 1.4826 * np.median(np.abs(vel[ok] - med))
            p = pairs[ok[pairs[:, 0]] & ok[pairs[:, 1]]]
            row = {"region": region, "line": ln.upper(), "n_pairs": len(p), "sep_frac": args.sep_frac,
                   "catalogue_N2": 2 * np.mean(err[ok] ** 2)}
            if len(p) >= 20:
                dv = vel[p[:, 0]] - vel[p[:, 1]]
                s2 = err[p[:, 0]] ** 2 + err[p[:, 1]] ** 2
                c, cm, K, Kn, N2m = stats(dv, s2)
                b = np.array([stats(dv[i], s2[i]) for i in rng.integers(0, len(p), (N_BOOT, len(p)))])
                row.update(c=c, c_err=b[:, 0].std(), c_mad=cm, K_dv=K, K_dv_err=b[:, 2].std(), K_norm=Kn,
                           K_norm_err=b[:, 3].std(), N2_meas=N2m)
            rows.append(row)
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                           if k != "sep_frac"), flush=True)
    cols = ["region", "line", "n_pairs", "sep_frac", "catalogue_N2", "c", "c_err", "c_mad", "K_dv", "K_dv_err",
            "K_norm", "K_norm_err", "N2_meas"]
    with open(OUT / f"overlap_noise_sep{args.sep_frac:g}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
