import os
import glob
import numpy as np
import matplotlib.pyplot as plt

SLOPE_TARGET = 0.66666
MIN_POINTS = 4
SLOPE_SIMILARITY = 0.1  # 10% difference for "similar" slopes

def fit_powerlaw(log_r, log_vsf, log_vsf_err=None):
    """Fit single power law with optional error weighting and return BIC, KS, slope, intercept, slope uncertainty"""
    n_points = len(log_r)
    if n_points < MIN_POINTS:
        return np.inf, np.nan, np.nan, np.nan, np.nan

    # Use polyfit for both slope and error
    if log_vsf_err is not None and np.all(np.isfinite(log_vsf_err)) and np.any(log_vsf_err > 0):
        weights = 1.0 / (log_vsf_err ** 2)
        (alpha, logC), cov = np.polyfit(log_r, log_vsf, 1, w=weights, cov=True)
    else:
        (alpha, logC), cov = np.polyfit(log_r, log_vsf, 1, cov=True)
    alpha_err = np.sqrt(cov[0,0])
    fit = logC + alpha * log_r
    residuals = log_vsf - fit
    if log_vsf_err is not None and np.all(np.isfinite(log_vsf_err)) and np.any(log_vsf_err > 0):
        chi2 = np.sum((residuals / log_vsf_err) ** 2)
        rss = chi2
    else:
        rss = np.sum(residuals ** 2)
    rss = max(rss, 1e-16)
    k = 2
    bic = n_points * np.log(rss / n_points) + k * np.log(n_points)
    ks = np.max(np.abs(residuals))
    return bic, ks, alpha, logC, alpha_err


def fit_double_powerlaw(log_r, log_vsf, log_vsf_err=None):
    """Fit broken power law with optional error weighting and include slope uncertainties (polyfit only)"""
    n = len(log_r)
    if n < 2 * MIN_POINTS:
        return np.inf, None
    best_bic = np.inf
    best_params = None
    for i in range(MIN_POINTS, n - MIN_POINTS):
        r1, r2 = log_r[:i], log_r[i:]
        y1, y2 = log_vsf[:i], log_vsf[i:]
        e1, e2 = None, None
        if log_vsf_err is not None:
            e1, e2 = log_vsf_err[:i], log_vsf_err[i:]
        # Polyfit for segment 1
        if e1 is not None and np.all(np.isfinite(e1)) and np.any(e1 > 0):
            weights1 = 1.0 / (e1 ** 2)
            (alpha1, logC1), cov1 = np.polyfit(r1, y1, 1, w=weights1, cov=True)
        else:
            (alpha1, logC1), cov1 = np.polyfit(r1, y1, 1, cov=True)
        alpha1_err = np.sqrt(cov1[0,0])
        # Polyfit for segment 2
        if e2 is not None and np.all(np.isfinite(e2)) and np.any(e2 > 0):
            weights2 = 1.0 / (e2 ** 2)
            (alpha2, logC2), cov2 = np.polyfit(r2, y2, 1, w=weights2, cov=True)
        else:
            (alpha2, logC2), cov2 = np.polyfit(r2, y2, 1, cov=True)
        alpha2_err = np.sqrt(cov2[0,0])
        fit1 = logC1 + alpha1 * r1
        fit2 = logC2 + alpha2 * r2
        if e1 is not None and np.all(np.isfinite(e1)) and np.any(e1 > 0):
            chi2_1 = np.sum(((y1 - fit1) / e1) ** 2)
        else:
            chi2_1 = np.sum((y1 - fit1) ** 2)
        if e2 is not None and np.all(np.isfinite(e2)) and np.any(e2 > 0):
            chi2_2 = np.sum(((y2 - fit2) / e2) ** 2)
        else:
            chi2_2 = np.sum((y2 - fit2) ** 2)
        rss = chi2_1 + chi2_2
        rss = max(rss, 1e-16)
        k = 5
        bic = n * np.log(rss / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_params = {
                'alpha1': alpha1, 'alpha1_err': alpha1_err, 'logC1': logC1,
                'alpha2': alpha2, 'alpha2_err': alpha2_err, 'logC2': logC2,
                'break_index': i, 'break_log_r': log_r[i],
                'bic': bic
            }
    return best_bic, best_params


def scan_xrange(log_r, log_vsf, log_vsf_err=None):
    n = len(log_r)
    log_xmin_vals = np.linspace(log_r.min(), log_r.max() - 0.1, 20)
    log_xmax_vals = np.linspace(log_r.min() + 0.1, log_r.max(), 20)
    dx = log_xmax_vals[1] - log_xmax_vals[0]  # step size in log_x

    best_single = {'bic': np.inf}
    best_double = {'bic': np.inf}

    for xmin in log_xmin_vals:
        for xmax in log_xmax_vals:
            if xmax <= xmin:
                continue

            mask = (log_r >= xmin) & (log_r <= xmax)
            if np.sum(mask) < MIN_POINTS:
                continue

            log_r_sub = log_r[mask]
            log_vsf_sub = log_vsf[mask]
            log_vsf_err_sub = log_vsf_err[mask] if log_vsf_err is not None else None

            # --- Single power law ---
            bic_s, ks_s, alpha_s, logC_s, alpha_err_s = fit_powerlaw(log_r_sub, log_vsf_sub, log_vsf_err_sub)
            if bic_s < best_single['bic']:
                best_single = {
                    'bic': bic_s,
                    'ks': ks_s,
                    'alpha': alpha_s,
                    'alpha_err': alpha_err_s,
                    'logC': logC_s,
                    'xmin': xmin,
                    'xmax': xmax,
                    'log_r': log_r_sub,
                    'log_vsf': log_vsf_sub,
                    'dx': dx  # store dx here
                }

            # --- Double power law ---
            bic_d, params_d = fit_double_powerlaw(log_r_sub, log_vsf_sub, log_vsf_err_sub)
            if params_d is not None and bic_d < best_double['bic']:
                best_double = {
                    'params': params_d,
                    'bic': bic_d,
                    'xmin': xmin,
                    'xmax': xmax,
                    'log_r': log_r_sub,
                    'log_vsf': log_vsf_sub,
                    'alpha1_err': params_d['alpha1_err'],
                    'alpha2_err': params_d['alpha2_err'],
                    'dx': dx  # store dx here too
                }

    return best_single, best_double


def plot_bestfit_grid(results_dict, plot_data, lines_order=('NII', 'SII', 'SIII'), output_file='vsf_powerlaw_bestfits_grid.png'):
    objects = sorted(results_dict.keys())
    n_total = len(objects)
    half = (n_total + 1) // 2
    object_splits = [objects[:half], objects[half:]]
    for idx, obj_list in enumerate(object_splits):
        nrows = len(obj_list)
        ncols = len(lines_order)
        figsize = (4 * ncols, 3 * max(nrows, 1))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        for i, object_name in enumerate(obj_list):
            for j, line in enumerate(lines_order):
                ax = axes[i, j]
                if i == 0:
                    ax.set_title(line, fontsize=12)
                if j == 0:
                    ax.set_ylabel(
                        r'$\bar{S}_\mathrm{2} = \langle | v(r + \ell) - v(r) | \rangle ^{2}\ \mathrm{(km/s)^2}$',
                        rotation=90
                    )
                else:
                    ax.set_ylabel("")
                ax.grid(True, which='both', ls=':', alpha=0.4)
                key = (object_name, line)
                if key not in plot_data:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    continue
                entry = plot_data[key]
                log_r = entry['log_r']
                log_vsf = entry['log_vsf']
                best_single = entry['best_single']
                best_double = entry['best_double']
                r = 10**log_r
                vsf = 10**log_vsf
                ax.loglog(r, vsf, 'o', ms=4, alpha=0.7, label='data')
                if best_double['bic'] < best_single['bic'] and best_double['params'] is not None:
                    params = best_double['params']
                    i_break = params['break_index']
                    r1 = 10**best_double['log_r'][:i_break]
                    r2 = 10**best_double['log_r'][i_break:]
                    y1_fit = 10**params['logC1'] * (10**best_double['log_r'][:i_break])**params['alpha1']
                    y2_fit = 10**params['logC2'] * (10**best_double['log_r'][i_break:])**params['alpha2']
                    ax.loglog(r1, y1_fit, '-', color='red', lw=1.8)
                    ax.loglog(r2, y2_fit, '-', color='orange', lw=1.8)
                    ax.axvline(10**best_double['xmin'], color='gray', ls='--', lw=0.8)
                    ax.axvline(10**best_double['xmax'], color='gray', ls='--', lw=0.8)
                    ax.axvline(10**params['break_log_r'], color='black', ls=':', label='breakpoint')
                    label_text = f"α1={params['alpha1']:.2f}, α2={params['alpha2']:.2f}"
                else:
                    alpha = best_single['alpha']
                    logC = best_single['logC']
                    x_fit = best_single['log_r']
                    y_fit = 10**logC * (10**x_fit)**alpha
                    ax.loglog(10**x_fit, y_fit, '-', color='red', lw=1.8)
                    ax.axvline(10**best_single['xmin'], color='gray', ls='--', lw=0.8)
                    ax.axvline(10**best_single['xmax'], color='gray', ls='--', lw=0.8)
                    label_text = f"α={alpha:.2f} (curve_fit)"
                ax.text(0.05, 0.92, f"Region: {object_name}", transform=ax.transAxes, fontsize=8, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                ax.set_xlabel('r (pc)')
                ax.text(0.05, 0.05, label_text, transform=ax.transAxes,
                        fontsize=7, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.tight_layout(h_pad=1.5, w_pad=1.2)
        figures_dir = os.path.join('Figures', 'manuscript', '03_slopes_turnover', 'legacy_bic_powerlaw_fits')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        output_path = os.path.join(figures_dir, f"vsf_powerlaw_bestfits_grid_part{idx+1}.png")
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined best-fit grid figure to {output_path}")


def analyze_vsfs_nested():
    # Search for VSF_*.npz in both current directory and figures/
    files = glob.glob('VSF_*.npz')
    files += glob.glob(os.path.join('figures', 'VSF_*.npz'))
    # Ensure output directories exist
    figures_dir = os.path.join('Figures', 'manuscript', '03_slopes_turnover', 'legacy_bic_powerlaw_fits')
    results_dir = 'results'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_dict = {}
    plot_data = {}

    for file in files:
        print(f"Processing {file}")
        # Parse filename: VSF_{object}_{line}.npz
        parts = file.replace('VSF_', '').replace('.npz', '').rsplit('_', 1)
        if len(parts) != 2:
            print(f"Skipping {file}: unexpected filename format")
            continue
        object_name, line = parts

        data = np.load(file)
        log_r = data['log_centers']
        vsf = data['vsf']
        vsf_err = data['vsf_err']
        counts = data['counts']

        # Convert to log space for fitting
        log_vsf = np.log10(vsf)
        
        # Convert VSF errors to log-space errors using error propagation
        # If y = log10(x), then dy = dx / (x * ln(10))
        log_vsf_err = vsf_err / (vsf * np.log(10))

        # Only use points with valid data and sufficient counts
        valid_mask = np.isfinite(log_vsf) & np.isfinite(log_r) & (counts > 0) & np.isfinite(log_vsf_err)
        if np.sum(valid_mask) < MIN_POINTS:
            print(f"Skipping {file}: insufficient valid data points ({np.sum(valid_mask)})")
            continue

        log_r = log_r[valid_mask]
        log_vsf = log_vsf[valid_mask]
        log_vsf_err = log_vsf_err[valid_mask]

        best_single, best_double = scan_xrange(log_r, log_vsf, log_vsf_err)

        # ---------- SELECT SLOPE/POINT BASED ON CRITERIA ----------
        if best_double['bic'] < best_single['bic'] and best_double['params'] is not None:
            params = best_double['params']
            alpha1, alpha2 = params['alpha1'], params['alpha2']
            if abs(alpha1 - alpha2) / max(abs(alpha1), abs(alpha2)) < SLOPE_SIMILARITY:
                slope = alpha2  # rightmost slope
                point = 10**best_double['xmax']
                slope_err = best_double['alpha2_err']
                turnover_type = 'double_right'
            else:
                slope = alpha1 if abs(alpha1 - SLOPE_TARGET) < abs(alpha2 - SLOPE_TARGET) else alpha2
                point = 10**params['break_log_r'] if abs(alpha1 - SLOPE_TARGET) < abs(alpha2 - SLOPE_TARGET) else 10**best_double['xmax']
                slope_err = best_double['alpha1_err'] if abs(alpha1 - SLOPE_TARGET) < abs(alpha2 - SLOPE_TARGET) else best_double['alpha2_err']
                turnover_type = 'double_break' if abs(alpha1 - SLOPE_TARGET) < abs(alpha2 - SLOPE_TARGET) else 'double_right'
        else:
            slope = best_single['alpha']
            point = 10**best_single['xmax']
            slope_err = best_single['alpha_err']
            turnover_type = 'single'

        dx = best_double['dx'] if best_double['bic'] < best_single['bic'] else best_single['dx']

        # ---------- STORE IN NESTED DICTIONARY ----------
        if object_name not in results_dict:
            results_dict[object_name] = {}

        # Check if turnover point is at the edge of the data range
        r_max = 10**log_r.max()
        lowerlim = 1 if 4 * point >= r_max else 0

        results_dict[object_name][line] = {
            'slope': slope,
            'slope_err': slope_err,
            'turnover_point': point,
            'turnover_type': turnover_type,
            'dx': dx,
            'lowerlim': lowerlim,
            'bic_single': best_single['bic'],
            'bic_double': best_double['bic']
        }

        plot_data[(object_name, line)] = {
            'log_r': log_r,
            'log_vsf': log_vsf,
            'best_single': best_single,
            'best_double': best_double
        }

        print(f"Stored for {object_name} - {line}: slope={slope:.2f}±{slope_err:.2f}, "
              f"turnover={point:.2f} pc ({turnover_type}), dx={dx:.3f}")

        # ---------- PLOTTING ----------
        plt.figure(figsize=(8,6))
        plt.loglog(10**log_r, 10**log_vsf, 'o', alpha=0.7, label='VSF data')

        if best_double['bic'] < best_single['bic'] and best_double['params'] is not None:
            params = best_double['params']
            i = params['break_index']
            r1, r2 = best_double['log_r'][:i], best_double['log_r'][i:]
            y1_fit = 10**params['logC1'] * (10**r1)**params['alpha1']
            y2_fit = 10**params['logC2'] * (10**r2)**params['alpha2']
            plt.loglog(10**r1, y1_fit, '-', color='red', lw=2, label=f'Double α1={params["alpha1"]:.2f}')
            plt.loglog(10**r2, y2_fit, '-', color='orange', lw=2, label=f'Double α2={params["alpha2"]:.2f}')
            plt.axvline(10**best_double['xmin'], color='gray', ls='--')
            plt.axvline(10**best_double['xmax'], color='gray', ls='--')
            plt.axvline(10**params['break_log_r'], color='black', ls=':', label='breakpoint')
        else:
            alpha = best_single['alpha']
            logC = best_single['logC']
            x_fit = best_single['log_r']
            y_fit = 10**logC * (10**x_fit)**alpha
            plt.loglog(10**x_fit, y_fit, '-', color='red', lw=2, label=f'Single α={alpha:.2f}')
            plt.axvline(10**best_single['xmin'], color='gray', ls='--')
            plt.axvline(10**best_single['xmax'], color='gray', ls='--')

        plt.xlabel('r (pc)')
        plt.ylabel(r'$\bar{S}_\mathrm{2} = \langle | v(r + \ell) - v(r) | \rangle ^{2}\ \mathrm{(km/s)^2}$')
        # Add region name to the title
        plt.title(f"{object_name} (Region: {object_name}) - {line}")
        plt.legend()
        fig_path = os.path.join(figures_dir, os.path.basename(file.replace('.npz', '_best_fit.png')))
        plt.savefig(fig_path)
        plt.close()

    # Save nested dictionary
    np.save(os.path.join(results_dir, 'vsf_powerlaw_fits.npy'), results_dict)
    print(f"Saved results to {os.path.join(results_dir, 'vsf_powerlaw_fits.npy')}")

    # Save combined multi-panel best fit figure
    plot_bestfit_grid(results_dict, plot_data, output_file=os.path.join(figures_dir, 'vsf_powerlaw_bestfits_grid.png'))

    return results_dict


if __name__ == "__main__":
    results = analyze_vsfs_nested()

    print("\n" + "="*60)
    print("POWER LAW FIT RESULTS SUMMARY")
    print("="*60)

    for object_name in sorted(results.keys()):
        print(f"\n{object_name}:")
        for line in sorted(results[object_name].keys()):
            data = results[object_name][line]
            print(f"  {line}: slope = {data['slope']:.2f} ± {data['slope_err']:.2f}, "
                  f"turnover = {data['turnover_point']:.2f} pc ({data['turnover_type']})")

    print(f"\nTotal objects processed: {len(results)}")
    total_fits = sum(len(lines) for lines in results.values())
    print(f"Total fits performed: {total_fits}")
    print(f"Results saved to: {os.path.join('results', 'vsf_powerlaw_fits.npy')}")
