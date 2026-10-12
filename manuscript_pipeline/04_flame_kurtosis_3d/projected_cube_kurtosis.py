#!/usr/bin/env python3
"""Run the kurtosis/PDF analysis on an x-projected Kolmogorov cube."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from scipy.fft import irfftn, rfftn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_cube"
)


def kolmogorov_scalar_velocity_grid(
    n_grid: int,
    box_size: float,
    rng: np.random.Generator,
    kmin_factor: float = 2.0,
    kmax_fraction: float = 0.35,
) -> np.ndarray:
    """Create one Kolmogorov-like Gaussian velocity component using real FFTs."""
    freq = np.fft.fftfreq(n_grid, d=box_size / n_grid).astype(np.float32) * np.float32(2.0 * np.pi)
    freq_r = np.fft.rfftfreq(n_grid, d=box_size / n_grid).astype(np.float32) * np.float32(2.0 * np.pi)

    k_fund = np.float32(2.0 * np.pi / box_size)
    k_nyq = np.float32(np.pi * n_grid / box_size)
    kmin = np.float32(kmin_factor) * k_fund
    kmax = np.float32(kmax_fraction) * k_nyq

    amplitude = np.empty((n_grid, n_grid, n_grid // 2 + 1), dtype=np.float32)
    kx2 = freq[:, None, None] ** 2
    ky2 = freq[None, :, None] ** 2
    kz2 = freq_r[None, None, :] ** 2
    np.sqrt(kx2 + ky2 + kz2, out=amplitude)
    inertial = (amplitude >= kmin) & (amplitude <= kmax)
    np.power(amplitude, np.float32(-11.0 / 6.0), out=amplitude, where=inertial)
    amplitude[~inertial] = 0.0

    white_noise = rng.normal(size=(n_grid, n_grid, n_grid)).astype(np.float32)
    modes = rfftn(white_noise, workers=-1)
    del white_noise
    modes *= amplitude
    del amplitude
    field = irfftn(modes, s=(n_grid, n_grid, n_grid), workers=-1).real.astype(np.float32)
    field -= np.mean(field, dtype=np.float64)
    field /= np.std(field, dtype=np.float64)
    return field


def finite_kurtosis_norm(kk: np.ndarray, default_min: float = 3.0, default_max: float = 6.0):
    finite = kk[np.isfinite(kk) & (kk > 0)]
    if finite.size == 0:
        return plt.Normalize(default_min, default_max)
    vmin = min(default_min, float(np.nanpercentile(finite, 5)))
    vmax = max(default_max, float(np.nanpercentile(finite, 95)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return plt.Normalize(default_min, default_max)
    return plt.Normalize(vmin, vmax)


def kurtosis_from_pdf(pdf: np.ndarray, dv_centers: np.ndarray) -> np.ndarray:
    """Compute centered kurtosis directly from each P(delta-v) curve."""
    kk = np.full(pdf.shape[0], np.nan)
    for i, row in enumerate(pdf):
        good = np.isfinite(row) & np.isfinite(dv_centers) & (row > 0)
        if not np.any(good):
            continue
        p = row[good].astype(float)
        p_sum = np.sum(p)
        if p_sum <= 0:
            continue
        p /= p_sum
        dv = dv_centers[good]
        mean = np.sum(p * dv)
        centered = dv - mean
        m2 = np.sum(p * centered**2)
        m4 = np.sum(p * centered**4)
        if m2 > 0:
            kk[i] = m4 / m2**2
    return kk


def gaussian_from_pdf(row: np.ndarray, dv_centers: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return the moment-matched Gaussian for a discrete P(delta-v) curve."""
    good = np.isfinite(row) & np.isfinite(dv_centers) & (row > 0)
    if not np.any(good):
        return np.nan, np.nan, np.full_like(dv_centers, np.nan, dtype=float)

    p = row[good].astype(float)
    p /= np.sum(p)
    dv = dv_centers[good]
    mean = np.sum(p * dv)
    sigma = np.sqrt(np.sum(p * (dv - mean) ** 2))

    model = np.full_like(dv_centers, np.nan, dtype=float)
    if sigma > 0:
        model_good = np.exp(-0.5 * ((dv - mean) / sigma) ** 2)
        model_good *= np.sum(row[good]) / np.sum(model_good)
        model[good] = model_good
    return mean, sigma, model


def make_projected_map(y: np.ndarray, z: np.ndarray, projected_v: np.ndarray, out: Path) -> None:
    lo, hi = np.nanpercentile(projected_v, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    im = ax.imshow(
        projected_v.T,
        origin="lower",
        extent=(y.min(), y.max(), z.min(), z.max()),
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(r"x-projected cube: $\langle v_x\rangle_x$")
    fig.colorbar(im, ax=ax, label=r"$\langle v_x\rangle_x$")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def selected_slice_indices(pdf: np.ndarray, log_centers: np.ndarray, n_slices: int) -> np.ndarray:
    valid = np.flatnonzero(np.sum(pdf > 0, axis=1) > 8)
    if valid.size <= n_slices:
        return valid
    targets = np.linspace(log_centers[valid[1]], log_centers[valid[-2]], n_slices)
    return np.array([valid[np.argmin(np.abs(log_centers[valid] - target))] for target in targets])


def make_kurtosis_plot(
    pdf: np.ndarray,
    dv_bins: np.ndarray,
    log_centers: np.ndarray,
    kk: np.ndarray,
    out: Path,
    x_min: float | None,
    x_max: float | None,
) -> None:
    r = 10**log_centers
    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    if x_min is None:
        x_min = float(np.nanpercentile(dv_centers, 2))
    if x_max is None:
        x_max = float(np.nanpercentile(dv_centers, 98))
    mask = (dv_centers >= x_min) & (dv_centers <= x_max)
    if not np.any(mask):
        mask = np.isfinite(dv_centers)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.03, 1.0], hspace=0.18)
    cax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], projection="3d")

    cmap = cm.viridis
    norm = finite_kurtosis_norm(kk)
    for i in range(len(r)):
        y = pdf[i][mask]
        good = np.isfinite(y) & (y > 0)
        if not np.any(good):
            continue
        ax.plot(
            dv_centers[mask][good],
            np.log10(np.full(np.sum(good), r[i])),
            y[good],
            color=cmap(norm(kk[i])),
            linewidth=1.0,
        )

    ax.set_xlabel(r"$(\Delta \langle v_x\rangle_x)$", fontsize=14, labelpad=10)
    ax.set_ylabel(r"$\log r_{\rm proj}$", fontsize=14, labelpad=10)
    ax.set_zlabel(r"$\mathcal{P}(\Delta v)$", fontsize=14, labelpad=10)
    ax.set_xlim(float(np.nanmin(dv_centers[mask])), float(np.nanmax(dv_centers[mask])))
    ax.set_ylim(np.log10(r[0]), np.log10(r[-1]))
    ax.view_init(elev=25, azim=45)
    try:
        ax.dist = 9
    except Exception:
        pass

    positive = pdf[:, mask][np.isfinite(pdf[:, mask]) & (pdf[:, mask] > 0)]
    if positive.size:
        ax.set_zlim(max(float(np.nanpercentile(positive, 1)), 1e-4), min(float(np.nanmax(positive)), 1.0))
    else:
        ax.set_zlim(1e-4, 1.0)
    ax.set_zscale("log")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=12)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("w")

    ax.set_title(r"x-projected cube, kurtosis from $P(\Delta v)$", pad=18)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("Kurtosis", labelpad=3, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_kurtosis_with_gaussian_slices_plot(
    pdf: np.ndarray,
    dv_bins: np.ndarray,
    log_centers: np.ndarray,
    kk: np.ndarray,
    out: Path,
    x_min: float | None,
    x_max: float | None,
    n_slices: int = 4,
) -> None:
    r = 10**log_centers
    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    if x_min is None:
        x_min = float(np.nanpercentile(dv_centers, 2))
    if x_max is None:
        x_max = float(np.nanpercentile(dv_centers, 98))
    mask = (dv_centers >= x_min) & (dv_centers <= x_max)
    if not np.any(mask):
        mask = np.isfinite(dv_centers)

    slice_idx = selected_slice_indices(pdf, log_centers, n_slices)
    cmap = cm.viridis
    norm = finite_kurtosis_norm(kk)

    fig = plt.figure(figsize=(13.5, 7.3), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    left = outer[0].subgridspec(2, 1, height_ratios=[0.04, 1.0], hspace=0.10)
    cax = fig.add_subplot(left[0])
    ax = fig.add_subplot(left[1], projection="3d")

    for i in range(len(r)):
        y = pdf[i][mask]
        good = np.isfinite(y) & (y > 0)
        if not np.any(good):
            continue
        linewidth = 2.2 if i in slice_idx else 0.9
        alpha = 1.0 if i in slice_idx else 0.72
        ax.plot(
            dv_centers[mask][good],
            np.log10(np.full(np.sum(good), r[i])),
            y[good],
            color=cmap(norm(kk[i])),
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_xlabel(r"$(\Delta \langle v_x\rangle_x)$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$\log r_{\rm proj}$", fontsize=12, labelpad=8)
    ax.set_zlabel(r"$\mathcal{P}(\Delta v)$", fontsize=12, labelpad=8)
    ax.set_xlim(float(np.nanmin(dv_centers[mask])), float(np.nanmax(dv_centers[mask])))
    ax.set_ylim(np.log10(r[0]), np.log10(r[-1]))
    ax.view_init(elev=25, azim=45)
    positive = pdf[:, mask][np.isfinite(pdf[:, mask]) & (pdf[:, mask] > 0)]
    ax.set_zlim(max(float(np.nanpercentile(positive, 1)), 1e-4), min(float(np.nanmax(positive)), 1.0))
    ax.set_zscale("log")
    ax.set_title(r"PDF surface colored by $K[P(\Delta v)]$", pad=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("w")

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("Kurtosis", labelpad=3, fontsize=12)

    right = outer[1].subgridspec(2, 2, wspace=0.22, hspace=0.30)
    for panel, idx in enumerate(slice_idx):
        ax_slice = fig.add_subplot(right[panel // 2, panel % 2])
        mean, sigma, model = gaussian_from_pdf(pdf[idx], dv_centers)
        good_pdf = mask & np.isfinite(pdf[idx]) & (pdf[idx] > 0)
        good_model = mask & np.isfinite(model) & (model > 0)
        color = cmap(norm(kk[idx]))
        ax_slice.semilogy(
            dv_centers[good_pdf],
            pdf[idx][good_pdf],
            "o",
            ms=4,
            color=color,
            label=r"$P(\Delta v)$",
        )
        ax_slice.semilogy(
            dv_centers[good_model],
            model[good_model],
            color="0.1",
            lw=1.8,
            label="Gaussian",
        )
        ax_slice.set_title(
            fr"$r_{{proj}}={r[idx]:.3f}$, $K={kk[idx]:.2f}$",
            fontsize=11,
        )
        ax_slice.set_xlabel(r"$\Delta \langle v_x\rangle_x$")
        ax_slice.set_ylabel(r"$P(\Delta v)$")
        ax_slice.set_xlim(float(np.nanmin(dv_centers[mask])), float(np.nanmax(dv_centers[mask])))
        if np.isfinite(mean) and np.isfinite(sigma):
            ax_slice.text(
                0.03,
                0.92,
                fr"$\mu={mean:.2f}$" + "\n" + fr"$\sigma={sigma:.2f}$",
                transform=ax_slice.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
        if panel == 0:
            ax_slice.legend(frameon=False, fontsize=9)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=800)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--n-slices", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    rng = np.random.default_rng(args.seed)
    vx = kolmogorov_scalar_velocity_grid(args.n_grid, args.box_size, rng)
    projected_v = np.nanmean(vx, axis=0)

    half = args.box_size / 2
    y = np.linspace(-half, half, args.n_grid)
    z = np.linspace(-half, half, args.n_grid)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    y_flat = yy.ravel()
    z_flat = zz.ravel()
    v_flat = projected_v.ravel()
    vel_err = np.zeros_like(v_flat)

    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y_flat, z_flat, v_flat, bin_resolution=args.bin_resolution
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
            n_jobs=args.n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_out = args.output_dir / "Synthetic_x_projected_cube_order1_pdfkk.npz"
    map_out = args.output_dir / "Synthetic_x_projected_cube_map.png"
    kurt_out = args.output_dir / "Synthetic_x_projected_cube_order1_3d_kurtosis.png"
    slices_out = args.output_dir / "Synthetic_x_projected_cube_order1_3d_kurtosis_gaussian_slices.png"

    np.savez_compressed(
        npz_out,
        y=y_flat,
        z=z_flat,
        projected_v=v_flat,
        projected_map=projected_v,
        vx_cube=vx,
        pdf=pdf,
        dv_bins=dv_bins,
        log_centers=log_centers,
        kk=kk_pdf,
        kk_pairs=kk_pairs,
        kk_pdf=kk_pdf,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
    )
    make_projected_map(y, z, projected_v, map_out)
    make_kurtosis_plot(pdf, dv_bins, log_centers, kk_pdf, kurt_out, args.x_min, args.x_max)
    make_kurtosis_with_gaussian_slices_plot(
        pdf,
        dv_bins,
        log_centers,
        kk_pdf,
        slices_out,
        args.x_min,
        args.x_max,
        args.n_slices,
    )

    finite = kk_pdf[np.isfinite(kk_pdf) & (kk_pdf > 0)]
    print(f"Wrote {npz_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {map_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {kurt_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {slices_out.relative_to(REPO_ROOT)}")
    if finite.size:
        print(
            "Kurtosis range: "
            f"min={np.nanmin(finite):.2f}, median={np.nanmedian(finite):.2f}, "
            f"max={np.nanmax(finite):.2f}"
        )


if __name__ == "__main__":
    main()
