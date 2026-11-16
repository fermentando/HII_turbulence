#!/usr/bin/env python3
"""Does v_turb by itself change dK/dlogr, or only through v_bulk/v_turb?

Same N=5,000 no-pixelized random-point shells as
plot_vbulk_ratio_sweep_n5000_logistic.py (same field seed, same point seed,
same widths), but the turbulent grid is rescaled by v_turb instead of fixing
v_turb = 1 and varying v_bulk.

Panels:
  (a) v_bulk = 0, v_turb varied            -> pure-turbulence curves
  (b) v_bulk = 1 fixed, v_turb varied      -> changes the ratio
  (c) v_bulk/v_turb = 1 fixed, both scaled -> same ratio, different v_turb
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "00_core"))

from generate_random_point_expanding_shell_grid import write_random_shell_npz  # noqa: E402
from generate_shell_thickness_grid import shell_output_path  # noqa: E402
from plot_vbulk_ratio_sweep_n5000_logistic import (  # noqa: E402
    BIN_RESOLUTION, BOX_SIZE, FIELD_SEED, KK_KEY, N_GRID, N_POINTS, POINT_SEED,
    R_OUTER, WIDTHS, slope_for_record,
)
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
OUT_PNG = BASE_DIR / "vturb_scaling_test_dK_dlogr.png"
CACHE = BASE_DIR / "_vturb_scaling_test_cache.npz"

CASES = {
    "a": [(0.0, vt) for vt in (0.1, 1.0, 10.0)],
    "b": [(1.0, vt) for vt in (0.1, 0.3, 1.0, 3.0, 10.0)],
    "c": [(vt, vt) for vt in (0.1, 1.0, 10.0)],
}


def slopes_for(grids, v_bulk: float, v_turb: float, tmp: Path) -> np.ndarray:
    scaled = tuple(g * v_turb for g in grids)
    rng_points = np.random.default_rng(POINT_SEED)
    out = []
    for width in WIDTHS:
        r_inner = R_OUTER - float(width)
        path = shell_output_path(tmp, r_inner, R_OUTER)
        write_random_shell_npz(
            scaled, BOX_SIZE, r_inner, R_OUTER, v_bulk, N_POINTS, n_proj_grid=51,
            out_path=path, rng=rng_points, bin_resolution=BIN_RESOLUTION,
            n_jobs=1, no_pixelize=True,
        )
        with np.load(path) as d:
            rec = {"r": 10 ** np.asarray(d["log_centers"], float), "kk": np.asarray(d[KK_KEY], float)}
        out.append(slope_for_record(rec)[0])
    return np.asarray(out)


def load_or_compute() -> dict[tuple[float, float], np.ndarray]:
    pairs = sorted({p for v in CASES.values() for p in v})
    if CACHE.exists():
        c = np.load(CACHE)
        if np.array_equal(c["pairs"], np.asarray(pairs)):
            return {tuple(p): s for p, s in zip(c["pairs"], c["slopes"])}
    grids = kolmogorov_velocity_grid(N_GRID, BOX_SIZE, np.random.default_rng(FIELD_SEED))
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for vb, vt in pairs:
            print(f"v_bulk={vb:g} v_turb={vt:g}")
            res[(vb, vt)] = slopes_for(grids, vb, vt, Path(td))
    np.savez(CACHE, pairs=np.asarray(pairs), slopes=np.asarray([res[p] for p in pairs]))
    return res


def main() -> None:
    res = load_or_compute()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    titles = {
        "a": r"(a) $v_{\rm bulk}=0$, vary $v_{\rm turb}$",
        "b": r"(b) $v_{\rm bulk}=1$ fixed, vary $v_{\rm turb}$",
        "c": r"(c) $v_{\rm bulk}/v_{\rm turb}=1$, vary $v_{\rm turb}$",
    }
    markers = ["o", "s", "^", "D", "v"]
    for ax, (key, pairs) in zip(axes, CASES.items()):
        cmap = plt.get_cmap("viridis")
        for i, (vb, vt) in enumerate(pairs):
            col = cmap(i / max(len(pairs) - 1, 1))
            lab = rf"$v_{{\rm turb}}={vt:g}$"
            if key == "b":
                lab += rf"  ($v_{{\rm bulk}}/v_{{\rm turb}}={vb / vt:.3g}$)"
            ax.plot(WIDTHS, res[(vb, vt)], marker=markers[i % 5], color=col, lw=1.4,
                    ms=6 - i if key != "b" else 5, mfc="none" if key != "b" else col,
                    label=lab, alpha=0.9)
        ax.set_xscale("log")
        ax.axhline(0, color="0.3", ls="--", lw=1)
        ax.set_title(titles[key])
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel(r"$dK/d\log_{10} r$")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PNG}")

    for key, pairs in CASES.items():
        if key == "b":
            continue
        ref = res[pairs[0]]
        dmax = max(np.nanmax(np.abs(res[p] - ref)) for p in pairs[1:])
        print(f"panel {key}: max |slope - slope(v_turb={pairs[0][1]:g})| = {dmax:.2e}")


if __name__ == "__main__":
    main()
