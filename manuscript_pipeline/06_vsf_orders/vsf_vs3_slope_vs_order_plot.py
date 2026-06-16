import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('./kurtosis_plot.mplstyle')

csv_path = 'Figures/vsf_vs3_slopes_by_region.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

ps = np.arange(1, 6) 
print(ps)
k41 = ps / 3
print(k41)
sl94 = ps / 9 + 2 * (1 - (1 - 2 / (3 * 2))** (ps / 3))
bol02 = ps / 9 + 1 * (1 - (1 - 2 / (3 * 1))** (ps / 3))

df = pd.read_csv(csv_path)
orders = sorted(df['order'].unique())
regions = sorted(df['region'].unique())

if 'slope_err' not in df.columns:
    df['slope_err'] = np.nan

plt.figure(figsize=(10, 6))
for region in regions:
    region_df = df[df['region'] == region]
    region_map = {
        int(r['order']): (float(r['slope']), float(r['slope_err']) if np.isfinite(r['slope_err']) else np.nan)
        for _, r in region_df.iterrows()
    }

    region_orders = []
    region_slopes = []
    region_slope_errs = []
    for p in ps:
        if p == 3:
            # By construction, zeta_3 / zeta_3 = 1 with no fit uncertainty.
            region_orders.append(3)
            region_slopes.append(1.0)
            region_slope_errs.append(0.0)
            continue

        if p in region_map:
            m, m_err = region_map[p]
            region_orders.append(p)
            region_slopes.append(m)
            region_slope_errs.append(m_err)

    region_orders = np.asarray(region_orders, dtype=float)
    region_slopes = np.asarray(region_slopes, dtype=float)
    region_slope_errs = np.asarray(region_slope_errs, dtype=float)

    # Normalize both slope values and uncertainties by K41 expectation.
    k41_region = region_orders / 3.0
    yvals = region_slopes / k41_region
    yerrs = region_slope_errs / k41_region

    plt.errorbar(
        region_orders,
        yvals,
        yerr=yerrs,
        fmt='-o',
        capsize=3,
        alpha=0.8,
        elinewidth=1.0,
        capthick=1.0,
        label=region.replace('LVM_', '').replace('.fits', '').replace('_', ' ')
    )
plt.plot(ps, sl94/k41, linestyle = '--', color = 'k', alpha = 0.6, label = "She & Levesque '94")
plt.plot(ps, bol02/k41, linestyle = ':', color = 'k', alpha = 0.6, label = "Boldyrev '02")

plt.xlabel('p', fontsize=13)
plt.ylabel(r'$\tilde{\zeta} / \zeta_{K41}$', fontsize=13)
plt.grid(True, ls='--', alpha=0.5)
plt.legend(fontsize=9, loc='best', ncol=2)
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
out_dir = 'Figures/manuscript/07_final_slope_vs_order'
os.makedirs(out_dir, exist_ok=True)
out_plot = os.path.join(out_dir, 'vsf_vs3_slope_vs_order_by_region.png')
plt.savefig(out_plot, dpi=150)
plt.show()
print(f"✓ Slope vs order plot saved as {out_plot}")
