# Manuscript Pipeline

Run scripts from the project root:

```bash
cd "/Users/fernando/PhDResearch/LVM stuff"
```

## Paper-Order Workflow

1. Summary-region figures
   - `manuscript_pipeline/01_summary_regions/orion_subwindow_fast.py`
   - `manuscript_pipeline/01_summary_regions/orion_fullfov_fast.py`
   - `manuscript_pipeline/01_summary_regions/detect_brightest_nii_sources.py`
   - `manuscript_pipeline/01_summary_regions/select_hii_occupied_subregion.py`
   - `manuscript_pipeline/01_summary_regions/plot_nii_flux_maps_all_subregions.py`
   - Main plot outputs: `Figures/manuscript/01_summary_regions/`
   - Helper catalogs remain in `Figures/bright_sources/` because later scripts read them there.

2. Compute second-order VSFs for all regions
   - `manuscript_pipeline/02_vsf2_multifit/multi_fits_vsf_analysis.py`
   - Main data outputs: `NPZ/VSF_*_median_allsubwindows.npz` and `NPZ/VSF_*_mean_allsubwindows.npz`
   - Main plot outputs: `Figures/manuscript/02_vsf2_multifit/`
   - Selected-subregion variant: `manuscript_pipeline/02_vsf2_multifit/multi_fits_vsf_analysis_selected_subregions.py`

3. Slopes and turnover scales
   - `manuscript_pipeline/03_slopes_turnover/analyze_median_vsf_slopes_and_inflections.py`
   - Clean segmented fitter: `manuscript_pipeline/03_slopes_turnover/fit_segmented_vsf_breakpoints.py`
   - Main data output: `SlopeResults/median_vsf_slope_ks_summary.csv`
   - Main plot outputs: `Figures/manuscript/03_slopes_turnover/`
   - Full-FOV segmented-fit diagnostics: `Figures/manuscript/03_slopes_turnover/segmented_vsf_breakpoint_fits_total_fullfov.csv` and `Figures/manuscript/03_slopes_turnover/segmented_vsf_bestfits_total_fullfov_allregions.png`
   - Turnover-scale summary figure: `manuscript_pipeline/03_slopes_turnover/slope_turnover.py`
   - Legacy BIC single-vs-broken-line comparison retained: `manuscript_pipeline/03_slopes_turnover/ks_powerlaw_fits.py`

4. Flame NII 3D PDF/kurtosis figure
   - `manuscript_pipeline/04_flame_kurtosis_3d/kurtosis_plot.py`
   - Main plot output: `Figures/manuscript/04_flame_kurtosis_3d/Flame_NII_refactored_3D.png`
   - Supporting outputs: `Figures/Flame_NII_refactored_order1.npz` and `Figures/Flame_NII_refactored_stats.npz`

5. Kurtosis as a function of scale
   - `manuscript_pipeline/05_kurtosis_scale/plot_kurtosis_vs_separation_all_regions.py`
   - `manuscript_pipeline/05_kurtosis_scale/plot_kurtosis_direct_from_pdf_all_regions.py`
   - Main plot outputs: `Figures/manuscript/05_kurtosis_scale/kurtosis_vs_separation_all_regions.png` and `Figures/manuscript/05_kurtosis_scale/kurtosis_direct_vs_separation_all_regions.png`
   - Per-region outputs are written to `Figures/manuscript/05_kurtosis_scale/per_region/`

6. VSF orders against VSF3
   - `manuscript_pipeline/06_vsf_orders/vsf_order_vs3_allregions.py`
   - Main plot output: `Figures/manuscript/06_vsf_orders/VSF_orders1to5_vs_order3_allregions.png`
   - Main data output: `Figures/vsf_vs3_slopes_by_region.csv`

7. Final slope-vs-order figure
   - `manuscript_pipeline/06_vsf_orders/vsf_vs3_slope_vs_order_plot.py`
   - Main plot output: `Figures/manuscript/07_final_slope_vs_order/vsf_vs3_slope_vs_order_by_region.png`

## Core Code

- `manuscript_pipeline/00_core/vsf_module.py` contains the shared VSF computation classes and helpers.
- The active scripts insert this core directory into `sys.path`.

## Method Notes

- `manuscript_notes/vsf_breakpoint_model_selection.md` sketches the cleaner line/breakpoint fitting criterion to develop next.

## Archived Material

Old notebooks, one-off scripts, bplus-noise fits, kurtosis/tail-fit work, old KS-min grids, caches, and exploratory generated figures were moved to `archive_old/`.

Nothing was deleted.
