import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


LINE_ORDER = ["NII", "SII", "SIII"]
LINE_COLORS = {"NII": "C0", "SII": "C1", "SIII": "C2"}
TARGET_SLOPE = 2/3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the VSF segment slope closest to 1.66 against its breakpoint scale."
    )
    parser.add_argument(
        "--input-csv",
        default="Figures/manuscript/03_slopes_turnover/segmented_vsf_breakpoint_fits_selected_subregions_logflux2.8.csv",
        help="Path to segmented VSF breakpoint fit CSV.",
    )
    parser.add_argument(
        "--output",
        default="Figures/manuscript/03_slopes_turnover/slope_closest_1p66_vs_breakpoint_scale_logflux2.8.png",
        help="Output plot path",
    )
    return parser.parse_args()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def select_segment_closest_to_target(row, target_slope=TARGET_SLOPE):
    selected_segments = int(to_float(row.get("selected_segments", 0)))
    if selected_segments <= 1:
        return None

    breaks = [to_float(row.get(f"break{i}_pc", np.nan)) for i in range(1, selected_segments)]
    finite_breaks = [b for b in breaks if np.isfinite(b) and b > 0]
    if not finite_breaks:
        return None

    candidates = []
    for segment_index in range(1, selected_segments + 1):
        slope = to_float(row.get(f"slope{segment_index}", np.nan))
        slope_err = to_float(row.get(f"slope{segment_index}_err", np.nan))
        if not np.isfinite(slope):
            continue

        if segment_index < selected_segments:
            break_pc = breaks[segment_index - 1]
            boundary_kind = "right"
        else:
            # Last segment has no finite right edge; keep the finite breakpoint
            # that bounds this segment within the fitted radial range.
            break_pc = max(finite_breaks)
            boundary_kind = "last_segment_finite_boundary"

        if not np.isfinite(break_pc) or break_pc <= 0:
            continue

        candidates.append(
            {
                "segment": segment_index,
                "slope": slope,
                "slope_err": slope_err,
                "break_pc": break_pc,
                "boundary_kind": boundary_kind,
                "target_distance": abs(slope - target_slope),
            }
        )

    if not candidates:
        return None
    return min(candidates, key=lambda item: item["target_distance"])


def load_points(csv_path):
    points = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row.get("object", "")
            line = row.get("line", "")
            if line not in LINE_ORDER:
                continue

            if "selected_segments" in row:
                selected = select_segment_closest_to_target(row)
                split_used = selected is not None
                slope_low = selected["slope"] if selected else np.nan
                slope_low_err = selected["slope_err"] if selected else np.nan
                break_pc = selected["break_pc"] if selected else np.nan
                selected_segment = selected["segment"] if selected else np.nan
                boundary_kind = selected["boundary_kind"] if selected else ""
                break_p16 = np.nan
                break_p84 = np.nan
            else:
                split_used = to_bool(row.get("split_used", False))
                slope_low = to_float(row.get("slope_low_median", np.nan))
                slope_low_err = to_float(row.get("slope_low_err_half68", np.nan))
                break_pc = to_float(row.get("break_r_pc", np.nan))
                selected_segment = np.nan
                boundary_kind = "legacy"
                break_p16 = to_float(row.get("break_bootstrap_p16", np.nan))
                break_p84 = to_float(row.get("break_bootstrap_p84", np.nan))

            # Keep only valid piecewise fits.
            if (not split_used) or (not np.isfinite(slope_low)) or (not np.isfinite(break_pc)):
                continue
            if break_pc <= 0:
                continue

            xerr = np.nan
            if np.isfinite(break_p16) and np.isfinite(break_p84):
                xerr = 0.5 * (break_p84 - break_p16)

            points.append(
                {
                    "object": obj,
                    "line": line,
                    "x": break_pc,
                    "y": slope_low,
                    "xerr": xerr,
                    "yerr": slope_low_err if np.isfinite(slope_low_err) else np.nan,
                    "selected_segment": selected_segment,
                    "boundary_kind": boundary_kind,
                }
            )
    return points


def build_marker_map(objects):
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "h", "<", ">", "X", "P"]
    marker_map = {}
    for i, name in enumerate(sorted(objects)):
        marker_map[name] = marker_list[i % len(marker_list)]
    return marker_map


def make_plot(points, out_path):
    if not points:
        raise RuntimeError("No valid points found to plot.")

    objects = {p["object"] for p in points}
    marker_map = build_marker_map(objects)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.05,
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_kde_y = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_kde_x = fig.add_subplot(gs[0, 0], sharex=ax_main)

    x_all = []
    y_all = []
    x_phase = {k: [] for k in LINE_ORDER}
    y_phase = {k: [] for k in LINE_ORDER}

    shown_phase_label = {k: False for k in LINE_ORDER}

    for p in points:
        phase = p["line"]
        label = phase if not shown_phase_label[phase] else None
        shown_phase_label[phase] = True

        ax_main.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"] if np.isfinite(p["xerr"]) else None,
            yerr=p["yerr"] if np.isfinite(p["yerr"]) else None,
            fmt=marker_map[p["object"]],
            color=LINE_COLORS[phase],
            markersize=7,
            alpha=0.8,
            capsize=2,
            label=label,
        )

        x_all.append(p["x"])
        y_all.append(p["y"])
        x_phase[phase].append(p["x"])
        y_phase[phase].append(p["y"])

    ax_main.set_xscale("log")
    ax_main.set_xlabel("$r_\mathrm{turnover}$ (pc)")
    ax_main.set_ylabel(r"$ m $ (from $S_2 \sim \ell^m$)")
    ax_main.axhline(TARGET_SLOPE, color="0.3", linestyle="--", linewidth=1.1, alpha=0.7)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="best")

    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)

    if np.sum(np.isfinite(y_all)) >= 2:
        sns.kdeplot(y=y_all[np.isfinite(y_all)], fill=True, ax=ax_kde_y, color="black", alpha=0.25)
        for phase in LINE_ORDER:
            arr = np.asarray(y_phase[phase], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2:
                sns.kdeplot(y=arr, fill=True, ax=ax_kde_y, color=LINE_COLORS[phase], alpha=0.45)

    if np.sum(np.isfinite(x_all)) >= 2:
        sns.kdeplot(x=x_all[np.isfinite(x_all)], fill=True, ax=ax_kde_x, color="black", alpha=0.25)
        for phase in LINE_ORDER:
            arr = np.asarray(x_phase[phase], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2:
                sns.kdeplot(x=arr, fill=True, ax=ax_kde_x, color=LINE_COLORS[phase], alpha=0.45)

    ax_kde_y.set_xlabel("Density")
    ax_kde_x.set_ylabel("Density")
    ax_kde_y.grid(True, alpha=0.2)
    ax_kde_x.grid(True, alpha=0.2)

    plt.setp(ax_kde_x.get_xticklabels(), visible=False)
    plt.setp(ax_kde_y.get_yticklabels(), visible=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    style_path = "custom_plot.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)

    points = load_points(args.input_csv)
    make_plot(points, args.output)
    print(f"Wrote plot: {args.output}")
    print(f"Plotted {len(points)} points from: {args.input_csv}")


if __name__ == "__main__":
    main()
