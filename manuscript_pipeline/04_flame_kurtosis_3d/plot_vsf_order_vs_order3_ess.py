#!/usr/bin/env python3
"""Extended self-similarity (ESS): VSF of order p (2-5) vs VSF of order 3.

Reuses the exact same raw point data as plot_fixed_rmin_comparison.py (the
N=100,000, v_bulk=10, no-pixelize shell sweep across WIDTHS) -- no new
points are drawn. For each shell width, the velocity-difference PDF is
computed once per width (same compute_bins/compute_pdf_and_statistics call
as the dK/dlogr diagnostics), and every order's structure function
S_p(r) = <|Δv(r)|^p> is read off that single PDF rather than recomputed
per order, since the PDF already captures the full Δv distribution in each
r-bin.

Produces one 3-panel figure (p=2, p=4, p=5), each showing log10(S_p) vs
log10(S_3) -- the classic ESS plot -- with one curve per shell width
(colored by width/r_HII) and a power-law (ESS exponent zeta_p/zeta_3) fit
line through the pooled points from every width.

Caching: own cache file (_vsf_order_vs_order3_cache.npz).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402
from generate_shell_thickness_grid import width_grid, shell_output_path, R_OUTER_FIXED  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
N_POINTS = 100_000
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
ORDERS = [2, 3, 4, 5]
REFERENCE_ORDER = 3

CACHE = BASE_DIR / "_vsf_order_vs_order3_cache.npz"


def run_dir(n_points: int) -> Path:
    return BASE_DIR / f"synthetic_projected_shell_random_points_nopixel_n{n_points}"


def load_shell_points(n_points: int, widths: np.ndarray):
    """(y, z, v) raw arrays, one tuple per width -- identical loader to
    plot_fixed_rmin_comparison.py's load_shell_points."""
    d = run_dir(n_points)
    points = []
    for width in widths:
        r_inner = R_OUTER_FIXED - float(width)
        path = shell_output_path(d, r_inner, R_OUTER_FIXED)
        with np.load(path) as data:
            points.append((
                np.asarray(data["y"], dtype=float),
                np.asarray(data["z"], dtype=float),
                np.asarray(data["projected_v"], dtype=float),
            ))
    return points


def moment_from_pdf(pdf: np.ndarray, dv_centers: np.ndarray, order: int) -> np.ndarray:
    """S_p(r) = <|Δv|^p> per r-bin, read directly off an already-computed
    PDF -- mirrors kurtosis_from_pdf's normalization pattern, but for the
    uncentered absolute moment (the standard VSF definition) instead of a
    centered one.
    """
    out = np.full(pdf.shape[0], np.nan)
    for i, row in enumerate(pdf):
        good = np.isfinite(row) & np.isfinite(dv_centers) & (row > 0)
        if not np.any(good):
            continue
        p = row[good].astype(float)
        p_sum = np.sum(p)
        if p_sum <= 0:
            continue
        p /= p_sum
        out[i] = np.sum(p * np.abs(dv_centers[good]) ** order)
    return out


def structure_functions_for_width(y: np.ndarray, z: np.ndarray, v: np.ndarray) -> dict[int, np.ndarray]:
    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y, z, v, bin_resolution=BIN_RESOLUTION, min_separation=None
    )
    n_jobs = 1  # forced serial -- see plot_fixed_rmin_comparison.py for why
    vel_err = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y, z, v, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=n_jobs,
        )
    return {p: moment_from_pdf(pdf, dv_centers, p) for p in ORDERS}


def load_or_compute() -> dict[int, list[np.ndarray]]:
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as c:
            if (
                np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
                and int(c["n_points"]) == N_POINTS
                and list(c["orders"]) == ORDERS
            ):
                print(f"Loading cached structure functions from {CACHE.relative_to(REPO_ROOT)}...")
                return {p: [c[f"s{p}"][i] for i in range(len(WIDTHS))] for p in ORDERS}
        print("Cache doesn't match current WIDTHS/BIN_RESOLUTION/N_POINTS/ORDERS -- recomputing...")

    print(f"Computing S_p(r) for p={ORDERS}, N={N_POINTS:,}, across {len(WIDTHS)} widths...")
    points = load_shell_points(N_POINTS, WIDTHS)
    sfs = {p: [] for p in ORDERS}
    for i, width in enumerate(WIDTHS):
        y, z, v = points[i]
        result = structure_functions_for_width(y, z, v)
        for p in ORDERS:
            sfs[p].append(result[p])
        print(f"  width={width:.4f}: S_3 range [{np.nanmin(result[3]):.3g}, {np.nanmax(result[3]):.3g}]")

    np.savez_compressed(
        CACHE,
        widths=WIDTHS, bin_resolution=BIN_RESOLUTION, n_points=N_POINTS, orders=np.array(ORDERS),
        **{f"s{p}": np.array([np.asarray(x) for x in sfs[p]], dtype=object) for p in ORDERS},
    )
    print(f"Cached to {CACHE.relative_to(REPO_ROOT)}.")
    return sfs


sfs = load_or_compute()

style_path = REPO_ROOT / "custom_plot.mplstyle"
try:
    plt.style.use(str(style_path))
except OSError:
    pass

plot_orders = [p for p in ORDERS if p != REFERENCE_ORDER]
fig, axes = plt.subplots(1, len(plot_orders), figsize=(6.5 * len(plot_orders), 5.5))

norm = matplotlib.colors.LogNorm(vmin=WIDTHS.min(), vmax=WIDTHS.max())
cmap = plt.colormaps["viridis"]

for ax, p in zip(axes, plot_orders):
    pooled_x, pooled_y = [], []
    for i, width in enumerate(WIDTHS):
        s_ref = np.asarray(sfs[REFERENCE_ORDER][i])
        s_p = np.asarray(sfs[p][i])
        good = np.isfinite(s_ref) & np.isfinite(s_p) & (s_ref > 0) & (s_p > 0)
        if not np.any(good):
            continue
        log_ref = np.log10(s_ref[good])
        log_p = np.log10(s_p[good])
        ax.plot(log_ref, log_p, "-", lw=1.2, color=cmap(norm(width)), alpha=0.8)
        pooled_x.append(log_ref)
        pooled_y.append(log_p)

    if pooled_x:
        pooled_x = np.concatenate(pooled_x)
        pooled_y = np.concatenate(pooled_y)
        zeta, intercept = np.polyfit(pooled_x, pooled_y, 1)
        dense_x = np.linspace(pooled_x.min(), pooled_x.max(), 100)
        ax.plot(
            dense_x, intercept + zeta * dense_x, "--", lw=2.0, color="crimson", zorder=5,
            label=fr"$\zeta_{{{p}}}/\zeta_3 = {zeta:.2f}$",
        )
        ax.legend(fontsize=10)

    ax.set_xlabel(fr"$\log_{{10}} S_3(r)$")
    ax.set_ylabel(fr"$\log_{{10}} S_{{{p}}}(r)$")
    ax.set_title(fr"order {p} vs order 3", fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)

sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, pad=0.01)
cbar.set_label(r"$\Delta r_{\rm shell}/r_{\rm HII}$")

out = BASE_DIR / "random_point_count_vsf_order_vs_order3_ess.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out.relative_to(REPO_ROOT)}")
