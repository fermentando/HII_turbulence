import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import kstest

MIN_POINTS = 4

npz_files = sorted(glob.glob('VSF_*.npz'))
results = {}

for file in npz_files:
    data = np.load(file)
    vsf = data['vsf']
    log_centers = data['log_centers']
    counts = data['counts']

    # Only use bins with sufficient counts
    valid = np.isfinite(vsf) & np.isfinite(log_centers) & (counts > 0)
    log_r = log_centers[valid]
    log_vsf = np.log10(vsf[valid])

    n = len(log_r)
    if n < MIN_POINTS:
        print(f"Skipping {file}: not enough valid points")
        continue

    best_ks = np.inf
    best_fit = None
    best_range = None
    best_params = None

    # Try all possible fitting windows
    for i in range(n - MIN_POINTS + 1):
        for j in range(i + MIN_POINTS, n + 1):
            x = log_r[i:j]
            y = log_vsf[i:j]
            if len(x) < MIN_POINTS:
                continue
            # Fit power law
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            # KS test
            ks_stat, _ = kstest(y, lambda z: np.interp(z, y_fit, y_fit))
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_fit = (x.copy(), y.copy(), y_fit.copy())
                best_range = (i, j)
                best_params = (slope, intercept)

    if best_fit is None:
        print(f"No valid fit for {file}")
        continue

    slope, intercept = best_params
    x, y, y_fit = best_fit
    results[file] = {
        'slope': slope,
        'intercept': intercept,
        'ks': best_ks,
        'range': best_range,
        'log_r': x,
        'log_vsf': y,
        'fit': y_fit
    }

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(10**log_r, 10**log_vsf, 'o', label='VSF data')
    plt.plot(10**x, 10**y_fit, '-', label=f'Best fit (slope={slope:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('r (pc)')
    plt.ylabel('VSF')
    plt.title(f'{file} (KS={best_ks:.3g})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file.replace('.npz', '_ksfit.png'))
    plt.close()
    print(f"{file}: slope={slope:.2f}, KS={best_ks:.3g}, range={best_range}")

np.save('vsf_ksfit_summary.npy', results)
print('Saved results to vsf_ksfit_summary.npy')
