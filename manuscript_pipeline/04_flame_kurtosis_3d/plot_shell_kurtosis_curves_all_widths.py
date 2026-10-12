#!/usr/bin/env python3
"""Plot kurtosis K(r) curves for all shell widths in one figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_random_points_vbulk0p0"
)
FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")


def parse_shell_radii(path: Path) -> tuple[float, float]:
    match = FNAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected shell filename: {path.name}")
    return float(match.group(1).replace("p", ".")), float(match.group(2).replace("p", "."))


def load_records(shell_dir: Path) -> list[dict]:
    records = []
    for path in sorted(shell_dir.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz")):
        if FNAME_RE.match(path.name) is None:
            continue
        r_inner, r_outer = parse_shell_radii(path)
        with np.load(path) as data:
            log_centers = np.asarray(data["log_centers"], dtype=float)
            kk = np.asarray(data["kk"], dtype=float)
            cnt = np.asarray(data["cnt"], dtype=float) if "cnt" in data else np.full_like(kk, np.nan)
            n_pixels = int(np.asarray(data["y"]).size)
        records.append(
            {
                "path": path,
                "r_inner": r_inner,
                "r_outer": r_outer,
                "width": r_outer - r_inner,
                "width_norm": (r_outer - r_inner) / r_outer,
                "r": 10**log_centers,
                "kk": kk,
                "cnt": cnt,
                "n_pixels": n_pixels,
            }
        )
    return sorted(records, key=lambda item: item["width_norm"])


def plot_curves(records: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    cmap = plt.colormaps["viridis"]
    widths = np.array([record["width_norm"] for record in records])
    wmin, wmax = float(widths.min()), float(widths.max())
    for record in records:
        color = cmap((record["width_norm"] - wmin) / (wmax - wmin))
        good = np.isfinite(record["r"]) & np.isfinite(record["kk"]) & (record["kk"] > 0)
        ax.plot(
            record["r"][good],
            record["kk"][good],
            "o-",
            ms=2.9,
            lw=1.1,
            alpha=0.9,
            color=color,
            label=fr"{record['width_norm']:.3f}",
        )
    ax.axhline(3.0, color="0.25", ls="--", lw=1.1, label="K=3")
    ax.set_xscale("log")
    ax.set_xlabel(r"Projected separation, $r_{\rm proj}$")
    ax.set_ylabel("Kurtosis, $K$")
    ax.set_title("Kurtosis as a function of lag for all shell widths")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title=r"$\Delta R/r_{out}$", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_csv(records: list[dict], out: Path) -> None:
    with out.open("w", encoding="utf-8") as fobj:
        fobj.write("width_over_rout,width,r_inner,r_outer,r_proj,kurtosis,pair_count,n_projected_pixels\n")
        for record in records:
            for r, kk, cnt in zip(record["r"], record["kk"], record["cnt"]):
                fobj.write(
                    f"{record['width_norm']:.8f},{record['width']:.8f},"
                    f"{record['r_inner']:.8f},{record['r_outer']:.8f},"
                    f"{r:.8e},{kk:.8e},{cnt:.8e},{record['n_pixels']}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    args = parser.parse_args()

    records = load_records(args.shell_dir)
    if not records:
        raise FileNotFoundError(f"No shell NPZ files found in {args.shell_dir}")
    out_png = args.shell_dir / "kurtosis_vs_lag_all_shell_widths.png"
    out_csv = args.shell_dir / "kurtosis_vs_lag_all_shell_widths.csv"
    plot_curves(records, out_png)
    write_csv(records, out_csv)
    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")
    for record in records:
        finite = record["kk"][np.isfinite(record["kk"])]
        print(
            f"width/rout={record['width_norm']:.4f} "
            f"Kmed={np.nanmedian(finite):.3f} Kmax={np.nanmax(finite):.3f}"
        )


if __name__ == "__main__":
    main()
