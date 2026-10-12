#!/usr/bin/env python3
"""Slab-control test for the shell-thickness kurtosis turnover.

For each thickness, project a slab |x| <= thickness/2 through the same
Kolmogorov-like cube, restricted to the same projected disk r_proj <= 1 used
by the fixed-outer-radius shell grid. Every valid sightline therefore has the
same path length for a given slab thickness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    KURTOSIS_FLOOR,
    MAX_BOOTSTRAP_COUNT,
    MIN_FIT_POINTS,
    N_WIDTHS,
    RNG_SEED,
    R_OUTER_FIXED,
    WIDTH_MAX,
    WIDTH_MIN,
    bootstrap_kurtosis_from_pdf,
    width_grid,
)


SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
SLAB_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_slab"
)


def slab_projected_velocity(
    vx: np.ndarray,
    box_size: float,
    thickness: float,
    projected_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_grid = vx.shape[0]
    half = box_size / 2
    coords = np.linspace(-half, half, n_grid)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    slab_x = np.abs(x) <= thickness / 2
    disk_yz = (y**2 + z**2) <= projected_radius**2
    slab = slab_x & disk_yz

    slab_counts = np.sum(slab, axis=0)
    weighted_sum = np.sum(np.where(slab, vx, 0.0), axis=0)
    projected_v = np.full_like(weighted_sum, np.nan, dtype=float)
    valid = slab_counts > 0
    projected_v[valid] = weighted_sum[valid] / slab_counts[valid]
    return coords, coords, projected_v, slab_counts, slab


def slab_path_for_width(out_dir: Path, width: float) -> Path:
    tag = f"width{width:.4f}".replace(".", "p")
    return out_dir / f"Synthetic_x_projected_slab_{tag}_order1_pdfkk.npz"


def write_slab_npz(
    vx: np.ndarray,
    box_size: float,
    width: float,
    projected_radius: float,
    out_path: Path,
    bin_resolution: int,
    n_jobs: int,
) -> dict[str, float | int]:
    y, z, projected_v, slab_counts, slab_mask = slab_projected_velocity(
        vx, box_size, width, projected_radius
    )

    yy, zz = np.meshgrid(y, z, indexing="ij")
    valid = np.isfinite(projected_v)
    y_flat = yy[valid].ravel()
    z_flat = zz[valid].ravel()
    v_flat = projected_v[valid].ravel()
    vel_err = np.zeros_like(v_flat)

    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y_flat, z_flat, v_flat, bin_resolution=bin_resolution
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y_flat,
            z_flat,
            v_flat,
            vel_err,
            log_centers,
            bin_edges,
            dv_bins,
            order=1,
            n_jobs=n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)

    np.savez_compressed(
        out_path,
        y=y_flat,
        z=z_flat,
        projected_v=v_flat,
        projected_map=projected_v,
        slab_counts=slab_counts,
        slab_mask=slab_mask,
        thickness=width,
        r_outer=projected_radius,
        pdf=pdf,
        dv_bins=dv_bins,
        log_centers=log_centers,
        kk=kk_pdf,
        kk_pdf=kk_pdf,
        kk_pairs=kk_pairs,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
    )

    valid_counts = slab_counts[slab_counts > 0]
    finite = kk_pdf[np.isfinite(kk_pdf) & (kk_pdf > 0)]
    return {
        "n_proj_pixels": int(v_flat.size),
        "path_length_min": int(np.min(valid_counts)),
        "path_length_max": int(np.max(valid_counts)),
        "k_median": float(np.nanmedian(finite)) if finite.size else np.nan,
        "k_max": float(np.nanmax(finite)) if finite.size else np.nan,
    }


def load_slab(path: Path, rng: np.random.Generator) -> dict:
    data = np.load(path)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    kk = np.asarray(data["kk"], dtype=float)
    pdf = np.asarray(data["pdf"], dtype=float)
    cnt = np.asarray(data["cnt"], dtype=float)
    dv_centers = np.asarray(data["dv_centers"], dtype=float)
    r = 10**log_centers

    valid = np.isfinite(r) & np.isfinite(kk) & (r > 0)
    order = np.argsort(r[valid])
    idx = np.flatnonzero(valid)[order]
    kk_lo_all, kk_hi_all = bootstrap_kurtosis_from_pdf(pdf, dv_centers, cnt, rng)

    slab_counts = np.asarray(data["slab_counts"])
    valid_counts = slab_counts[slab_counts > 0]
    return {
        "path": path,
        "r": r[idx],
        "log_r": np.log10(r[idx]),
        "kk": kk[idx],
        "kk_lo": kk_lo_all[idx],
        "kk_hi": kk_hi_all[idx],
        "width": float(data["thickness"]),
        "thickness": float(data["thickness"]),
        "r_outer": float(data["r_outer"]),
        "n_proj_pixels": int(np.asarray(data["y"]).size),
        "path_length_min": int(np.min(valid_counts)),
        "path_length_max": int(np.max(valid_counts)),
    }


def fit_radius_linear_above_floor(record: dict, floor: float = KURTOSIS_FLOOR) -> dict:
    """Fit K = intercept + slope r to the non-Gaussian part of K(r)."""
    mask = (
        np.isfinite(record["r"])
        & np.isfinite(record["kk"])
        & (record["kk"] >= floor)
    )
    if np.count_nonzero(mask) < MIN_FIT_POINTS:
        return {
            "slope": np.nan,
            "slope_err": np.nan,
            "intercept": np.nan,
            "intercept_err": np.nan,
            "n_points": int(np.count_nonzero(mask)),
            "fit_mask": mask,
            "r_fit": record["r"][mask],
            "kk_fit": record["kk"][mask],
        }

    r = record["r"][mask]
    kk = record["kk"][mask]
    try:
        coeffs, cov = np.polyfit(r, kk, 1, cov=True)
        slope_err, intercept_err = np.sqrt(np.diag(cov))
    except (ValueError, np.linalg.LinAlgError):
        coeffs = np.polyfit(r, kk, 1)
        slope_err = np.nan
        intercept_err = np.nan

    slope, intercept = coeffs
    return {
        "slope": float(slope),
        "slope_err": float(slope_err),
        "intercept": float(intercept),
        "intercept_err": float(intercept_err),
        "n_points": int(np.count_nonzero(mask)),
        "fit_mask": mask,
        "r_fit": record["r"][mask],
        "kk_fit": record["kk"][mask],
    }


def load_shell_records(rng: np.random.Generator) -> list[dict]:
    records = []
    for width in width_grid():
        r_inner = R_OUTER_FIXED - width
        tag = f"rin{r_inner:.4f}_rout{R_OUTER_FIXED:.4f}".replace(".", "p")
        path = SHELL_DIR / f"Synthetic_x_projected_shell_{tag}_order1_pdfkk.npz"
        if not path.exists():
            continue
        data = np.load(path)
        log_centers = np.asarray(data["log_centers"], dtype=float)
        kk = np.asarray(data["kk"], dtype=float)
        pdf = np.asarray(data["pdf"], dtype=float)
        cnt = np.asarray(data["cnt"], dtype=float)
        dv_centers = np.asarray(data["dv_centers"], dtype=float)
        r = 10**log_centers
        valid = np.isfinite(r) & np.isfinite(kk) & (r > 0)
        order = np.argsort(r[valid])
        idx = np.flatnonzero(valid)[order]
        kk_lo_all, kk_hi_all = bootstrap_kurtosis_from_pdf(pdf, dv_centers, cnt, rng)
        shell = {
            "r": r[idx],
            "log_r": np.log10(r[idx]),
            "kk": kk[idx],
            "kk_lo": kk_lo_all[idx],
            "kk_hi": kk_hi_all[idx],
            "width": float(data["r_outer"]) - float(data["r_inner"]),
        }
        fit = fit_radius_linear_above_floor(shell)
        if fit["n_points"] >= MIN_FIT_POINTS and np.isfinite(fit["slope"]):
            records.append({**shell, **fit})
    return records


def plot_slab_examples(records: list[dict], out: Path) -> None:
    usable = [record for record in records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    example_idx = [0, len(usable) // 2, len(usable) - 1]
    example_idx = sorted(set(min(max(i, 0), len(usable) - 1) for i in example_idx))

    fig, axes = plt.subplots(1, len(example_idx), figsize=(4.6 * len(example_idx), 4.2), sharey=True)
    if len(example_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, example_idx):
        slab = usable[idx]
        yerr = np.vstack([
            np.clip(slab["kk"] - slab["kk_lo"], 0, None),
            np.clip(slab["kk_hi"] - slab["kk"], 0, None),
        ])
        ax.errorbar(slab["r"], slab["kk"], yerr=yerr, fmt="o", ms=4, color="0.6", ecolor="0.75", capsize=2)
        mask = slab["fit_mask"]
        yerr_fit = np.vstack([
            np.clip(slab["kk"][mask] - slab["kk_lo"][mask], 0, None),
            np.clip(slab["kk_hi"][mask] - slab["kk"][mask], 0, None),
        ])
        ax.errorbar(slab["r_fit"], slab["kk_fit"], yerr=yerr_fit, fmt="o", ms=4, color="C0", ecolor="C0", capsize=2.5)
        r_line = np.geomspace(slab["r_fit"].min(), slab["r_fit"].max(), 200)
        model = slab["intercept"] + slab["slope"] * r_line
        ax.plot(r_line, model, "-", color="C3", lw=2, label=fr"$m={slab['slope']:.3f}$")
        ax.axhline(KURTOSIS_FLOOR, color="0.4", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(r"Projected radius, $r_{\rm proj}$")
        ax.set_title(fr"slab $\Delta x={slab['width']:.3f}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Kurtosis, $K$")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_slab_vs_shell(slab_records: list[dict], shell_records: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for records, label, color, marker in [
        (shell_records, "spherical shell", "C0", "o"),
        (slab_records, "constant-depth slab", "C3", "s"),
    ]:
        widths = np.array([record["width"] for record in records])
        slope = np.array([record["slope"] for record in records])
        slope_err = np.array([record["slope_err"] for record in records])
        order = np.argsort(widths)
        ax.errorbar(
            widths[order],
            slope[order],
            yerr=slope_err[order],
            fmt=f"{marker}-",
            ms=5,
            lw=1.5,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("LOS thickness")
    ax.axhline(0, color="0.35", lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel(r"Linear slope, $dK / dr$")
    ax.set_title("Shell vs slab control with linear K(r) fits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_slab_voxels(ax, slab: dict, stride: int = 4) -> None:
    mask = slab["slab_mask"][::stride, ::stride, ::stride]
    n_grid = slab["slab_mask"].shape[0]
    coords = np.linspace(-slab["box_size"] / 2.0, slab["box_size"] / 2.0, n_grid)[::stride]
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=x[mask],
        cmap="viridis",
        s=0.65,
        linewidths=0,
        alpha=0.25,
        depthshade=False,
    )
    half = slab["box_size"] / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=-48)
    ax.set_title(fr"$\Delta x={slab['width']:.3f}$", fontsize=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False


def load_slab_geometries(widths: np.ndarray, box_size: float) -> list[dict]:
    slabs = []
    for width in widths:
        path = slab_path_for_width(SLAB_DIR, float(width))
        if not path.exists():
            continue
        with np.load(path) as data:
            slabs.append(
                {
                    "width": float(data["thickness"]),
                    "box_size": box_size,
                    "slab_mask": np.asarray(data["slab_mask"], dtype=bool),
                    "slab_counts": np.asarray(data["slab_counts"], dtype=float),
                    "projected_map": np.asarray(data["projected_map"], dtype=float),
                }
            )
    return sorted(slabs, key=lambda item: item["width"])


def plot_slab_geometry_grid(slabs: list[dict], out: Path) -> None:
    if not slabs:
        return
    n_show = min(6, len(slabs))
    indices = np.linspace(0, len(slabs) - 1, n_show).round().astype(int)
    selected = [slabs[i] for i in indices]

    fig = plt.figure(figsize=(3.2 * n_show, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(3, n_show, height_ratios=[1.15, 1.0, 1.0])

    vmax_count = max(float(np.nanmax(slab["slab_counts"])) for slab in selected)
    finite_v = np.concatenate(
        [slab["projected_map"][np.isfinite(slab["projected_map"])] for slab in selected]
    )
    vlim = max(abs(np.nanpercentile(finite_v, 1)), abs(np.nanpercentile(finite_v, 99)))

    count_axes = []
    vel_axes = []
    last_count = None
    last_vel = None
    for col, slab in enumerate(selected):
        ax3d = fig.add_subplot(grid[0, col], projection="3d")
        plot_slab_voxels(ax3d, slab)

        ax_count = fig.add_subplot(grid[1, col])
        count_axes.append(ax_count)
        last_count = ax_count.imshow(
            slab["slab_counts"].T,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax_count,
            interpolation="nearest",
            aspect="equal",
        )
        ax_count.set_xticks([])
        ax_count.set_yticks([])
        if col == 0:
            ax_count.set_ylabel("path length")

        ax_vel = fig.add_subplot(grid[2, col])
        vel_axes.append(ax_vel)
        last_vel = ax_vel.imshow(
            slab["projected_map"].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
            aspect="equal",
        )
        ax_vel.set_xticks([])
        ax_vel.set_yticks([])
        if col == 0:
            ax_vel.set_ylabel(r"$\langle v_x\rangle_x$")

    fig.colorbar(last_count, ax=count_axes, shrink=0.8, label="slab cells")
    fig.colorbar(last_vel, ax=vel_axes, shrink=0.8, label=r"$\langle v_x\rangle_x$")
    fig.suptitle("Constant-depth slab control: same path length for every sightline", fontsize=13)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--r-outer", type=float, default=R_OUTER_FIXED)
    parser.add_argument("--width-min", type=float, default=WIDTH_MIN)
    parser.add_argument("--width-max", type=float, default=WIDTH_MAX)
    parser.add_argument("--n-widths", type=int, default=N_WIDTHS)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Regenerate slab NPZ files even if they already exist.")
    args = parser.parse_args()

    SLAB_DIR.mkdir(parents=True, exist_ok=True)

    widths = width_grid()
    missing = [width for width in widths if not slab_path_for_width(SLAB_DIR, float(width)).exists()]
    if args.force or missing:
        rng_field = np.random.default_rng(args.seed)
        vx, _, _ = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng_field)
    else:
        vx = None

    for width in widths:
        out_path = slab_path_for_width(SLAB_DIR, float(width))
        if out_path.exists() and not args.force:
            print(f"Reusing {out_path.relative_to(REPO_ROOT)}")
            continue
        stats = write_slab_npz(
            vx,
            args.box_size,
            float(width),
            args.r_outer,
            out_path,
            args.bin_resolution,
            args.n_jobs,
        )
        print(
            f"Wrote {out_path.relative_to(REPO_ROOT)}  width={width:.4f} "
            f"pixels={stats['n_proj_pixels']} path={stats['path_length_min']}..{stats['path_length_max']} "
            f"Kmed={stats['k_median']:.2f} Kmax={stats['k_max']:.2f}"
        )

    rng = np.random.default_rng(RNG_SEED)
    slab_records = []
    for width in widths:
        slab = load_slab(slab_path_for_width(SLAB_DIR, float(width)), rng)
        fit = fit_radius_linear_above_floor(slab)
        slab_records.append({**slab, **fit})
        print(
            f"slab width={slab['width']:.4f} path={slab['path_length_min']}..{slab['path_length_max']} "
            f"n_above_floor={fit['n_points']:3d} slope={fit['slope']:.3f} +/- {fit['slope_err']:.3f}"
        )

    usable_slabs = [record for record in slab_records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    shell_records = load_shell_records(np.random.default_rng(RNG_SEED))

    control_out = SLAB_DIR / "slab_vs_shell_linear_radius_slope_control.png"
    plot_slab_vs_shell(usable_slabs, shell_records, control_out)
    examples_out = SLAB_DIR / "slab_radius_linear_fit_examples.png"
    plot_slab_examples(usable_slabs, examples_out)
    geometry_out = SLAB_DIR / "slab_thickness_geometry_grid.png"
    plot_slab_geometry_grid(load_slab_geometries(widths, args.box_size), geometry_out)
    print(f"Wrote {control_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {examples_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {geometry_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
