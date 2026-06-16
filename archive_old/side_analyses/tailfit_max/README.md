# tailfit_binned

Pipeline for fitting binned tail distributions from `counts__*` arrays in an NPZ file.

## Run

```bash
/Users/maxbg/miniforge3/bin/python -m tailfit_binned.run_npz input.npz output_dir
```

## Core CLI parameters

- `--nbins`  
  Number of fixed log bins over `[1, 1e8]` (default: `50`).

### xmin inference controls

- `--xmin-mode {map,smallest_good,ks_min}`  
  `smallest_good` (default) chooses the smallest xmin whose PL bootstrap p-value exceeds threshold.  
  `map` chooses maximum posterior weight.  
  `ks_min` chooses xmin minimizing KS distance between empirical and fitted PL CDF on the tail (Clauset/powerlaw behavior).  
  `ks_min` ignores priors and p-values and can choose larger xmin on curved/broken tails.
  In `ks_min`, xmin uncertainty is derived from the KS curve around its minimum and propagated into reported BPL/LN parameter intervals.

- `--xmin-softcap`  
  Soft-cap prior scale in `logprior(xmin) = - (xmin/softcap)^gamma` (default: `200`).  
  If `<= 0`, soft-cap prior is disabled.

- `--xmin-gamma`  
  Soft-cap prior exponent (default: `2.0`).

- `--xmin-max`  
  Optional hard cap on xmin candidates. If set, candidates satisfy `xmin <= xmin_max`.

- `--xmin-pthresh`  
  Goodness threshold for `smallest_good` (default: `0.1`).

- `--xmin-candidates`  
  Number of candidate xmin values sampled from bin edges (default: `50`).

- `--xmin-min-tail-count`  
  Minimum counts above xmin for a candidate to be valid (default: `200`).

- `--xmin-lambda-discard`  
  Fallback selector strength (used if no candidate passes `pthresh`). Larger values favor keeping more tail data.

- `--xmin-bootstrap-sims`  
  Number of parametric bootstrap realizations for PL goodness p-values.

- `--propagate-xmin-uncertainty {on,off}`  
  Whether to propagate xmin uncertainty into BPL/LN parameter intervals (default: `on`).  
  Set `off` to report the smaller conditional parameter errors at the selected xmin only.

### BPL xB prior controls

- `--bpl-xb-prior-min-ratio`  
  Minimum preferred ratio `xB/xmin` for BPL prior (default: `1.0`).

- `--bpl-xb-prior-strength`  
  Strength of soft prior penalty (default: `0.0`, off).  
  Penalty is quadratic in log-space when `xB/xmin` is below `--bpl-xb-prior-min-ratio`.

### Other

- `--quiet`  
  Suppress per-dataset fit logs.

- `--continue-on-error`  
  Continue processing remaining snapshots if one `counts__*` dataset fails.  
  When enabled, failures are collected in `fit_summary.json` under `failures` and `n_failed`.

## Outputs

- Per dataset:
  - `dataset_<key>.json`
  - `<key>/pdf_x2p.png`
  - `<key>/delta_aic_bic.png`
- Aggregate:
  - `fit_summary.json`
  - `overview_bestfit_vs_snapshot.png`
  - `overview_delta_aic_vs_pl_vs_snapshot.png`

## Notes

- Fitting and AIC/BIC use a multinomial binned likelihood in tail bins.
- `xmin.selected` is the value actually used for PL/BPL/LN fits and plots.
- `xmin.map` is the posterior MAP over candidates; it may differ from `xmin.selected` in `smallest_good` mode.
