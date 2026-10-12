"""Compare segmented VSF breakpoints with adopted HII-region radii.

The x-axis radii are read from ``manuscript_notes/adopted_hii_region_radii.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

LINE_ORDER = ["NII", "SII", "SIII"]
LINE_COLORS = {"NII": "tab:blue", "SII": "tab:red", "SIII": "tab:green"}
TARGET_SLOPE = 2.0 / 3.0

INPUT_CSV = Path("Figures/manuscript/03_slopes_turnover/segmented_vsf_breakpoint_fits_total_fullfield.csv")
RADIUS_CSV = Path("manuscript_notes/adopted_hii_region_radii.csv")
OUT_CSV = Path("Figures/manuscript/03_slopes_turnover/breakpoints_vs_adopted_hii_radius_total_fullfov.csv")
OUT_FIG = Path("Figures/manuscript/03_slopes_turnover/breakpoints_vs_adopted_hii_radius_total_fullfov.png")
OUT_FIT_CSV = Path("Figures/manuscript/03_slopes_turnover/breakpoints_vs_adopted_hii_radius_tracer_fits_total_fullfov.csv")


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def select_breakpoint_for_target_slope(row: dict[str, str], target_slope: float = TARGET_SLOPE) -> dict[str, object] | None:
    """Choose the right-hand breakpoint for the segment whose slope is closest to target_slope."""
    selected_segments = int(to_float(row.get("selected_segments", "0")))
    if selected_segments <= 1:
        return None

    breaks = [to_float(row.get(f"break{i}_pc", "")) for i in range(1, selected_segments)]
    finite_breaks = [break_pc for break_pc in breaks if np.isfinite(break_pc) and break_pc > 0]
    if not finite_breaks:
        return None

    candidates = []
    for segment_index in range(1, selected_segments + 1):
        slope = to_float(row.get(f"slope{segment_index}", ""))
        slope_err = to_float(row.get(f"slope{segment_index}_err", ""))
        if not np.isfinite(slope):
            continue

        if segment_index < selected_segments:
            break_pc = breaks[segment_index - 1]
            boundary_kind = "right"
        else:
            break_pc = max(finite_breaks)
            boundary_kind = "largest_finite_breakpoint"

        if not np.isfinite(break_pc) or break_pc <= 0:
            continue

        candidates.append(
            {
                "selected_segment": segment_index,
                "selected_slope": slope,
                "selected_slope_err": slope_err,
                "break_pc": break_pc,
                "boundary_kind": boundary_kind,
                "target_slope": target_slope,
                "target_slope_distance": abs(slope - target_slope),
            }
        )

    if not candidates:
        return None
    return min(candidates, key=lambda item: item["target_slope_distance"])


def load_adopted_radii() -> dict[str, dict[str, object]]:
    if not RADIUS_CSV.exists():
        raise FileNotFoundError(RADIUS_CSV)

    radii = {}
    with RADIUS_CSV.open("r", encoding="utf-8", newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            radius_pc = to_float(row.get("radius_pc", ""))
            if not np.isfinite(radius_pc) or radius_pc <= 0:
                continue
            radii[row["object"]] = {
                "radius_pc": radius_pc,
                "source": row.get("source", ""),
                "source_url": row.get("source_url", ""),
                "notes": row.get("notes", ""),
            }
    return radii


def load_breakpoint_rows() -> list[dict[str, object]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    adopted_radii = load_adopted_radii()
    rows = []

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            object_name = row["object"]
            line = row["line"]
            selected_segments = int(to_float(row.get("selected_segments", "0")))
            if selected_segments <= 1:
                continue

            if object_name not in adopted_radii:
                raise KeyError(f"No adopted radius found for {object_name} in {RADIUS_CSV}")

            selected_break = select_breakpoint_for_target_slope(row)
            if selected_break is None:
                continue

            radius_info = adopted_radii[object_name]
            break_pc = float(selected_break["break_pc"])
            rows.append(
                {
                    "object": object_name,
                    "line": line,
                    "selected_segment": selected_break["selected_segment"],
                    "selected_slope": selected_break["selected_slope"],
                    "selected_slope_err": selected_break["selected_slope_err"],
                    "target_slope": selected_break["target_slope"],
                    "target_slope_distance": selected_break["target_slope_distance"],
                    "boundary_kind": selected_break["boundary_kind"],
                    "break_pc": break_pc,
                    "radius_pc": radius_info["radius_pc"],
                    "break_over_radius": break_pc / radius_info["radius_pc"] if radius_info["radius_pc"] > 0 else np.nan,
                    "radius_source": radius_info["source"],
                    "radius_source_url": radius_info["source_url"],
                    "radius_notes": radius_info["notes"],
                }
            )

    return rows


def write_rows(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_line_by_tracer(rows: list[dict[str, object]]) -> list[dict[str, float | str | int]]:
    """Fit log10(break_pc) = intercept + slope log10(radius_pc) for each tracer."""
    fit_rows = []
    for line in LINE_ORDER:
        line_rows = [row for row in rows if row["line"] == line]
        radius = np.asarray([row["radius_pc"] for row in line_rows], dtype=float)
        breaks = np.asarray([row["break_pc"] for row in line_rows], dtype=float)
        finite = np.isfinite(radius) & np.isfinite(breaks) & (radius > 0) & (breaks > 0)
        if np.count_nonzero(finite) < 3:
            continue

        x = np.log10(radius[finite])
        y = np.log10(breaks[finite])
        result = linregress(x, y)
        fit_rows.append(
            {
                "line": line,
                "n_points": int(np.count_nonzero(finite)),
                "log10_intercept": float(result.intercept),
                "slope": float(result.slope),
                "r_value": float(result.rvalue),
                "p_value": float(result.pvalue),
                "stderr": float(result.stderr),
                "intercept_stderr": float(result.intercept_stderr),
            }
        )
    return fit_rows


def write_fit_rows(fit_rows: list[dict[str, float | str | int]]) -> None:
    if not fit_rows:
        return

    OUT_FIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FIT_CSV.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(fit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fit_rows)


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = nan"
    if p_value < 1e-3:
        return f"p = {p_value:.1e}"
    return f"p = {p_value:.3f}"


def make_plot(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No breakpoint rows to plot.")

    fig, ax = plt.subplots(figsize=(6, 5))

    label_seen = set()
    for row in rows:
        line = row["line"]
        label = line
        if label in label_seen:
            label = None
        else:
            label_seen.add(label)

        ax.scatter(
            row["radius_pc"],
            row["break_pc"],
            color=LINE_COLORS.get(line, "0.2"),
            marker="o",
            s=58,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.45,
            #label=label,
        )

    finite_sizes = np.asarray([row["radius_pc"] for row in rows], dtype=float)
    finite_breaks = np.asarray([row["break_pc"] for row in rows], dtype=float)
    finite = np.isfinite(finite_sizes) & np.isfinite(finite_breaks) & (finite_sizes > 0) & (finite_breaks > 0)
    lo = min(np.nanmin(finite_sizes[finite]), np.nanmin(finite_breaks[finite]))
    hi = max(np.nanmax(finite_sizes[finite]), np.nanmax(finite_breaks[finite]))
    grid = np.logspace(np.log10(lo * 0.8), np.log10(hi * 1.25), 200)
    ax.plot(grid, grid, color="0.35", linestyle="--", linewidth=1.0, label="break = radius")
    ax.plot(grid, 2.0 * grid, color="0.55", linestyle=":", linewidth=1.0, label="break = diameter")

    fit_rows = fit_line_by_tracer(rows)
    for fit in fit_rows:
        line = str(fit["line"])
        color = LINE_COLORS.get(line, "0.2")
        fitted = 10 ** (float(fit["log10_intercept"]) + float(fit["slope"]) * np.log10(grid))
        ax.plot(
            grid,
            fitted,
            color=color,
            linewidth=1.8,
            alpha=0.95,
            #label=f"{line} fit ({format_p_value(float(fit['p_value']))})",
        )

    annotation_lines = [
        f"{fit['line']}: slope = {float(fit['slope']):.2f}, {format_p_value(float(fit['p_value']))}"
        for fit in fit_rows
    ]
    if annotation_lines:
        ax.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.82},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$D_\mathrm{HII}$ (pc)")
    ax.set_ylabel(r"$r_\mathrm{turnover}$ (pc)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass

    rows = load_breakpoint_rows()
    write_rows(rows)
    fit_rows = fit_line_by_tracer(rows)
    write_fit_rows(fit_rows)
    make_plot(rows)
    print(f"Wrote table: {OUT_CSV}")
    print(f"Wrote tracer fits: {OUT_FIT_CSV}")
    print(f"Wrote plot: {OUT_FIG}")
    print(f"Plotted {len(rows)} breakpoints from {len(set(row['object'] for row in rows))} regions")
    print(f"Using adopted radii from: {RADIUS_CSV}")


if __name__ == "__main__":
    main()
