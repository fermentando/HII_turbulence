#!/usr/bin/env python3
"""Pool sampled-field pair moments over several turbulent-field seeds.

Tests whether the single-realisation turbulent kurtosis converges to the
Gaussian-field ensemble prediction of derive_kurtosis_vratio_dependence.py.
Same points (POINT_SEED) and binning as kurtosis_shape_vs_vratio_fixed_width.py;
only FIELD_SEED changes. Usage: python kurtosis_vratio_seed_ensemble.py WIDTH [N_SEEDS]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WIDTH = float(sys.argv[1])
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
sys.argv = sys.argv[:1]
import kurtosis_shape_vs_vratio_fixed_width as kv  # noqa: E402
from generate_random_point_expanding_shell_grid import random_points_in_shell, sample_periodic_grid  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402

SEEDS = [kv.FIELD_SEED] + [100 + s for s in range(N_SEEDS - 1)]


def moments_for_seed(seed, p, mu):
    grids = kolmogorov_velocity_grid(kv.N_GRID, kv.BOX, np.random.default_rng(seed))
    u = sample_periodic_grid(grids, p, kv.BOX)[:, 0]
    u = (u - u.mean()) / u.std()
    y, z = p[:, 1], p[:, 2]
    nb = len(kv.EDGES) - 1
    e2 = kv.EDGES ** 2
    acc = {k: np.zeros(nb) for k in kv.KEYS}
    for ii in range(len(u) - 1):
        j = slice(ii + 1, None)
        d2 = (y[j] - y[ii]) ** 2 + (z[j] - z[ii]) ** 2
        b = np.searchsorted(e2, d2) - 1
        ok = (b >= 0) & (b < nb)
        b = b[ok]
        du = (u[j] - u[ii])[ok]
        dm = (mu[j] - mu[ii])[ok]
        du2, dm2 = du * du, dm * dm
        for k, val in (("n", np.ones_like(du)), ("u2", du2), ("u4", du2 * du2), ("m2", dm2),
                       ("m4", dm2 * dm2), ("u1m1", du * dm), ("u3m1", du2 * du * dm),
                       ("u2m2", du2 * dm2), ("u1m3", du * dm2 * dm)):
            acc[k] += np.bincount(b, weights=val, minlength=nb)
    return acc


def main():
    tag = f"w{WIDTH:g}".replace(".", "p")
    out = kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds{N_SEEDS}.npz"
    if out.exists():
        print(f"exists: {out}")
        return
    p = random_points_in_shell(kv.N_POINTS, 1.0 - WIDTH, 1.0, np.random.default_rng(kv.POINT_SEED))
    mu = p[:, 0] / np.linalg.norm(p, axis=1)
    per_seed = []
    for s in SEEDS:
        print(f"  width={WIDTH} field seed {s}")
        per_seed.append(moments_for_seed(s, p, mu))
    np.savez(out, seeds=np.array(SEEDS),
             **{k: np.array([m[k] for m in per_seed]) for k in kv.KEYS})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
