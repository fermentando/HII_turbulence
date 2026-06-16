import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('./custom_plot.mplstyle')

SUMMARY_CSV = 'SlopeResults/median_vsf_slope_ks_summary.csv'
OUT_FIG = 'SlopeResults/slope_vs_sqrtS2_at_10pc.png'
R_TARGET_PC = 10.0

LINE_ORDER = ['NII', 'SII', 'SIII']
LINE_COLORS = {'NII': 'C0', 'SII': 'C1', 'SIII': 'C2'}
MARKER_LIST = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']


def parse_line_and_npz(row):
    line = row.get('line', '').upper()
    npz_file = row.get('npz_file', '')
    if not line or not npz_file:
        return None, None
    return line, npz_file


def load_s2_at_target(npz_path, r_target):
    """
    Estimate S2 and its uncertainty at target distance from median NPZ.

    Interpolation is done in log-r vs linear-S2 space to respect radial spacing.
    """
    with np.load(npz_path) as d:
        vsf = d['vsf'].astype(float)
        vsf_err = d['vsf_err'].astype(float)
        log_r = d['log_centers'].astype(float)

    mask = np.isfinite(log_r) & np.isfinite(vsf) & np.isfinite(vsf_err)
    mask &= (vsf > 0) & (vsf_err > 0)
    log_r = log_r[mask]
    vsf = vsf[mask]
    vsf_err = vsf_err[mask]

    if len(log_r) < 2:
        return np.nan, np.nan

    order = np.argsort(log_r)
    log_r = log_r[order]
    vsf = vsf[order]
    vsf_err = vsf_err[order]

    log_target = np.log10(r_target)
    if (log_target < log_r.min()) or (log_target > log_r.max()):
        return np.nan, np.nan

    s2_target = np.interp(log_target, log_r, vsf)
    s2_err_target = np.interp(log_target, log_r, vsf_err)
    return float(s2_target), float(s2_err_target)


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f'Missing summary file: {SUMMARY_CSV}')

    rows = []
    with open(SUMMARY_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build region marker map from available objects in summary.
    regions = sorted({r['object'] for r in rows if r.get('object')})
    markers = {region: MARKER_LIST[i % len(MARKER_LIST)] for i, region in enumerate(regions)}

    # Figure layout matching slope_turnover style.
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_kde_y = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_kde_x = fig.add_subplot(gs[0, 0], sharex=ax_main)

    # For KDE uncertainty propagation.
    x_all, y_all = [], []
    x_phase = {line: [] for line in LINE_ORDER}
    y_phase = {line: [] for line in LINE_ORDER}

    rng = np.random.default_rng(20260504)

    for row in rows:
        region = row.get('object', '')
        if not region:
            continue

        line, npz_file = parse_line_and_npz(row)
        if line not in LINE_ORDER or not os.path.exists(npz_file):
            continue

        try:
            slope = float(row.get('slope_median', 'nan'))
            slope_err = float(row.get('slope_err_half68', 'nan'))
        except ValueError:
            continue

        if not np.isfinite(slope) or not np.isfinite(slope_err) or slope_err <= 0:
            continue

        s2_10, s2_10_err = load_s2_at_target(npz_file, R_TARGET_PC)
        if not np.isfinite(s2_10) or not np.isfinite(s2_10_err) or s2_10 <= 0:
            continue

        # x-axis requested: sqrt(total VSF value at 10 pc)
        x_val = np.sqrt(s2_10)
        # Propagate error: sigma_sqrt = 0.5 * sigma_S2 / sqrt(S2)
        x_err = 0.5 * s2_10_err / np.sqrt(s2_10)

        if not np.isfinite(x_err) or x_err <= 0:
            continue

        ax_main.errorbar(
            x_val,
            slope,
            xerr=x_err,
            yerr=slope_err,
            fmt=markers[region],
            color=LINE_COLORS[line],
            markersize=8,
            alpha=0.75,
            label=line if region == regions[0] else ''
        )

        # Uncertainty-aware samples for KDEs.
        y_samples = rng.normal(loc=slope, scale=slope_err, size=120)
        x_samples = rng.normal(loc=x_val, scale=x_err, size=120)
        y_samples = y_samples[np.isfinite(y_samples)]
        x_samples = x_samples[np.isfinite(x_samples) & (x_samples > 0)]

        y_all.extend(y_samples.tolist())
        x_all.extend(x_samples.tolist())
        y_phase[line].extend(y_samples.tolist())
        x_phase[line].extend(x_samples.tolist())

    # Main panel formatting.
    ax_main.set_xlabel(r'$\sqrt{\mathcal{S}_2(10\,\mathrm{pc})}$ [km/s]')
    ax_main.set_ylabel(r'$m$ (where $\mathcal{S}_2 \propto r^m$)')
    ax_main.axhline(2.0 / 3.0, color='grey', linestyle='--', alpha=0.3)
    ax_main.grid(True)
    ax_main.set_xscale('log')
    ax_main.set_xlim(left=1.0)

    # Right KDE: distribution in slope.
    if len(y_all) > 10:
        sns.kdeplot(y=y_all, fill=True, ax=ax_kde_y, color='black', alpha=0.25)
    for line in LINE_ORDER:
        ys = np.asarray(y_phase[line], dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size > 10:
            sns.kdeplot(y=ys, fill=True, ax=ax_kde_y, color=LINE_COLORS[line], alpha=0.45, label=line)
    ax_kde_y.set_xlabel(r'$\mathcal{P}(y)$')
    ax_kde_y.axhline(2.0 / 3.0, color='grey', linestyle='--', alpha=0.3)
    ax_kde_y.grid(True)

    # Top KDE: distribution in sqrt(S2(10 pc)).
    x_all_arr = np.asarray(x_all, dtype=float)
    x_all_arr = x_all_arr[np.isfinite(x_all_arr) & (x_all_arr > 0)]
    if x_all_arr.size > 10:
        sns.kdeplot(x=x_all_arr, fill=True, ax=ax_kde_x, color='black', alpha=0.25)
    for line in LINE_ORDER:
        xs = np.asarray(x_phase[line], dtype=float)
        xs = xs[np.isfinite(xs) & (xs > 0)]
        if xs.size > 10:
            sns.kdeplot(x=xs, fill=True, ax=ax_kde_x, color=LINE_COLORS[line], alpha=0.45, label=line)
    ax_kde_x.set_ylabel(r'$\mathcal{P}(x)$')
    ax_kde_x.set_xscale('log')
    ax_kde_x.grid(True)
    ax_kde_x.legend(loc='upper left')

    plt.setp(ax_kde_x.get_xticklabels(), visible=False)
    plt.setp(ax_kde_y.get_yticklabels(), visible=False)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)

    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
