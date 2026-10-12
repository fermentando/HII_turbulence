"""
VSF2D Module - 2D Velocity Structure Function Analysis
======================================================
Contains the VSF2D class for computing 2D velocity structure functions
and related statistics (PDF, C2, C4, intermittency).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Rectangle, Circle


# ============================================================================
# UTILITY FUNCTIONS FOR INDEPENDENT COMPUTATION
# ============================================================================

def fkp_vsf_error(r, Dp, n_bar, area_survey):
    """Feldman-Kaiser-Peacock (1994) style error estimate for the VSF.

    Adapts the FKP power-spectrum variance formula
    ``sigma_P/P = sqrt(2/N_k) * (1 + 1/(n_bar P))`` to a real-space
    structure function by mapping k-bin -> separation bin r, N_k (number
    of independent Fourier modes in the k-shell) -> N_eff(r) (number of
    independent spatial cells of size r tiling the survey footprint,
    estimated geometrically as area_survey / (pi r^2)), and P -> Dp(r).

    This is a cheap, closed-form cosmic-variance-style error -- no
    resampling required -- but it rests on the same disk-packing
    approximation FKP uses for mode-counting, so treat it as a
    cross-check against (or substitute for) a resampling-based estimate
    such as the block jackknife, not a replacement for one derived from
    the actual data.

    Parameters
    ----------
    r : array
        Separations at which Dp is evaluated (same units as area_survey**0.5).
    Dp : array
        Measured structure function values at each r (the "signal").
    n_bar : float
        Mean areal point density (points per unit area) of the survey.
    area_survey : float
        Survey footprint area (same area units as n_bar is defined in).

    Returns
    -------
    sigma : array
        Estimated 1-sigma error on Dp at each r.
    n_eff : array
        The geometric N_eff(r) used, for diagnostics.
    """
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)
    n_eff = area_survey / (np.pi * r**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.sqrt(2.0 / n_eff) * (1.0 + 1.0 / (n_bar * Dp))
        sigma = Dp * rel_err
    return sigma, n_eff


def fkp_vsf_error_calibrated(Dp, n_bar, jackknife_err):
    """FKP-style VSF error with N_eff(r) calibrated from a block jackknife.

    Rather than guessing N_eff(r) from an idealized disk-packing argument
    (see `fkp_vsf_error`), invert the FKP sample-variance term against the
    *measured* block-jackknife error -- treating the full jackknife spread
    as the pure sample/cosmic-variance piece:

        sigma_jack(r)/Dp(r) = sqrt(2/N_eff(r))  =>  N_eff(r) = 2/(sigma_jack(r)/Dp(r))^2

    then re-applies the FKP shot-noise correction on top of that:

        sigma_FKP(r) = Dp(r) * sqrt(2/N_eff(r)) * (1 + 1/(n_bar*Dp(r)))
                     = sigma_jack(r) * (1 + 1/(n_bar*Dp(r)))

    This avoids the geometric estimator's divergence once fewer than one
    independent cell fits in the survey footprint (N_eff -> data-driven,
    always well-defined wherever the jackknife is), while still adding the
    FKP shot-noise inflation for lag bins where the signal is weak and/or
    the survey is sparse relative to that scale.

    Parameters
    ----------
    Dp : array
        Measured structure function values (the "signal").
    n_bar : float
        Mean areal point density of the survey.
    jackknife_err : array
        Block-jackknife standard error on Dp, same shape as Dp.

    Returns
    -------
    sigma : array
        Calibrated FKP-style error on Dp.
    n_eff : array
        The jackknife-implied N_eff(r), for diagnostics.
    """
    Dp = np.asarray(Dp, dtype=float)
    jackknife_err = np.asarray(jackknife_err, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_jack = jackknife_err / Dp
        n_eff = 2.0 / rel_jack**2
        sigma = jackknife_err * (1.0 + 1.0 / (n_bar * Dp))
    return sigma, n_eff


def fkp_vsf_error_powerlaw(r, Dp, n_bar, jackknife_err):
    """FKP-style VSF error with a smooth power-law N_eff(r), anchored to jackknife.

    `fkp_vsf_error_calibrated` inverts N_eff(r) exactly per-bin from the
    jackknife error, which just reproduces the jackknife's own bin-to-bin
    noise (it's a tautological identity, not an independent model). This
    instead treats the per-bin inverted values
    ``N_eff(r) = 2/(sigma_jack(r)/Dp(r))^2`` as noisy *calibration data* for
    a single smooth trend, fit as a power law

        N_eff(r) = N0 * (r/r0)^(-alpha)

    (a straight line in log-log space) via least squares. This keeps the
    monotonic, smoothly-growing-with-lag error shape the original FKP
    geometric estimator has, but anchors its normalization and slope to
    the actual data instead of an idealized disk-packing argument that
    diverges once r approaches the survey size.

    Parameters
    ----------
    r : array
        Separations (must be > 0).
    Dp : array
        Measured structure function values.
    n_bar : float
        Mean areal point density of the survey.
    jackknife_err : array
        Block-jackknife standard error on Dp, same shape as Dp -- used only
        to fit the smooth N_eff(r) trend, not applied bin-by-bin.

    Returns
    -------
    sigma : array
        Smooth-N_eff FKP-style error on Dp.
    n_eff_smooth : array
        The fitted power-law N_eff(r).
    (n0, r0, alpha) : tuple
        Fit parameters: N_eff(r) = n0 * (r/r0)^(-alpha), r0 = geometric
        mean of the fitted r range.
    """
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)
    jackknife_err = np.asarray(jackknife_err, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_jack = jackknife_err / Dp
        n_eff_raw = 2.0 / rel_jack**2

    good = np.isfinite(r) & np.isfinite(n_eff_raw) & (r > 0) & (n_eff_raw > 0)
    log_r = np.log10(r[good])
    log_n = np.log10(n_eff_raw[good])
    r0 = 10**np.mean(log_r)
    slope, intercept = np.polyfit(log_r - np.log10(r0), log_n, 1)
    alpha = -slope
    n0 = 10**intercept

    n_eff_smooth = n0 * (r / r0)**(-alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = Dp * np.sqrt(2.0 / n_eff_smooth) * (1.0 + 1.0 / (n_bar * Dp))
    return sigma, n_eff_smooth, (n0, r0, alpha)


def fkp_vsf_error_geometric_calibrated(r, Dp, n_bar, area_survey, jackknife_err, n_eff_floor=1.0):
    """FKP-style VSF error: original 1/r^2 geometric N_eff(r) shape, amplitude
    calibrated against the jackknife, with a floor to prevent divergence.

    `fkp_vsf_error_powerlaw` fits both the slope and amplitude of N_eff(r)
    to the noisy per-bin jackknife inversion, which lets a handful of noisy
    bins distort the whole trend (e.g. a near-zero jackknife error at one r
    inflates N_eff there, dragging a free-slope fit toward near-flat). This
    instead keeps the FKP mode-counting argument's own shape --
    N_eff(r) = A_survey/(pi r^2), the same -2 power law as the original
    (uncalibrated) geometric estimator -- and only rescales its amplitude
    by a single robust (median-based) factor so the overall normalization
    matches the jackknife. A floor at `n_eff_floor` keeps sqrt(2/N_eff)
    finite instead of diverging once the disk-packing argument would
    otherwise claim fewer than one independent cell.

    Returns
    -------
    sigma : array
    n_eff_smooth : array
    kappa : float
        Calibration factor: N_eff(r) = max(kappa * area_survey/(pi r^2), n_eff_floor).
    """
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)
    jackknife_err = np.asarray(jackknife_err, dtype=float)

    geometric_shape = area_survey / (np.pi * r**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_jack = jackknife_err / Dp
        n_eff_raw = 2.0 / rel_jack**2
        ratio = n_eff_raw / geometric_shape

    good = np.isfinite(ratio) & (ratio > 0)
    kappa = 10**np.median(np.log10(ratio[good]))

    n_eff_smooth = np.maximum(kappa * geometric_shape, n_eff_floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = Dp * np.sqrt(2.0 / n_eff_smooth) * (1.0 + 1.0 / (n_bar * Dp))
    return sigma, n_eff_smooth, kappa


def fkp_vsf_error_mode_counting(r, Dp, n_bar, area_survey):
    """FKP-style VSF error with N_eff(r) from literal Fourier mode-counting.

    No fitting/calibration against the jackknife or anything else -- this
    derives N_eff(r) the way the original FKP paper actually derives N_k:
    by counting how many independent Fourier modes fit in a k-shell of a
    given survey area.

    For a 2D survey of area `area_survey`, Fourier modes are spaced by
    (2pi)^2/area_survey in k-space, so the number of modes in an annulus
    of radius k and width dk is

        N_k = (2 pi k dk) * area_survey / (2 pi)^2 = k dk area_survey / (2 pi)

    Real-space separation r and wavenumber are conjugate, k = 2pi/r, and
    the actual (possibly non-uniform) log-spacing of the r bins gives the
    corresponding fractional k-bin width directly: dk/k = d(ln r) (since
    k ~ 1/r). Substituting k = 2pi/r and dk = k * d(ln r):

        N_eff(r) = k^2 * d(ln r) * area_survey / (2 pi)
                 = 2 pi * area_survey * d(ln r) / r^2

    `d(ln r)` is computed directly from the actual r array (local gradient
    in log-space), so this is exact for whatever binning scheme produced
    `r` -- equal-width, shell, adaptive, anything -- not an assumed
    bin_resolution.

    Parameters
    ----------
    r : array
        Separations (must be > 0, ideally sorted).
    Dp : array
        Measured structure function values.
    n_bar : float
        Mean areal point density of the survey.
    area_survey : float
        Survey footprint area.

    Returns
    -------
    sigma : array
    n_eff : array
    """
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)

    d_ln_r = np.gradient(np.log(r))
    n_eff = 2.0 * np.pi * area_survey * d_ln_r / r**2

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = Dp * np.sqrt(2.0 / n_eff) * (1.0 + 1.0 / (n_bar * Dp))
    return sigma, n_eff


def fkp_vsf_error_effective_volume(x, y, r, Dp, n_grid=20):
    """FKP (1994)-style VSF error using the actual effective-volume result,
    not the uniform-density (Kaiser 1986) limit `fkp_vsf_error_mode_counting`
    uses.

    sigma_P/P = sqrt(2/N_k) * (1 + 1/(n_bar P)) assumes the tracer density
    n_bar is uniform across the survey. FKP's real 1994 result does not
    assume this: it weights each volume element by its *local*
    signal-to-noise squared and replaces the survey volume with an
    effective volume,

        V_eff = integral [ n(x) P / (1 + n(x) P) ]^2 dV

    A region/mode counts as "independent" only in proportion to its local
    signal-to-noise squared -- low-density or low-coverage regions
    contribute almost nothing (w -> 0 as n(x) -> 0), while high-density,
    signal-dominated regions count fully (w -> 1 as n(x) P -> infinity).
    Once this weighting is applied, the separate shot-noise correction
    (1 + 1/(n_bar P)) is no longer needed on top of sqrt(2/N_k) -- it's
    already absorbed into V_eff by construction.

    This computes the 2D analog for the VSF: grid the survey footprint
    into n_grid x n_grid cells, estimate the LOCAL point density
    n(x,y) = count/cell_area per cell (so spatially non-uniform coverage,
    e.g. a lower-S/N edge of the field, is explicitly down-weighted), and
    compute the effective area

        A_eff(r) = sum_cells [ n_cell * Dp(r) / (1 + n_cell * Dp(r)) ]^2 * cell_area

    separately for every r (since Dp(r) varies with r, so does the
    locally-optimal weighting), then reuse the same mode-counting relation
    as `fkp_vsf_error_mode_counting` with A_eff(r) in place of the naive
    (uniform-density) survey area.

    Parameters
    ----------
    x, y : arrays
        Point positions (same units as you want A_eff reported in).
    r : array
        Separations (must be > 0, ideally sorted -- used for d(ln r)).
    Dp : array
        Measured structure function values, same shape as r.
    n_grid : int
        Density-map grid side length (n_grid^2 cells). Finer grids track
        coverage variations more precisely but need enough points/cell for
        `n_cell` to be a meaningful local density estimate.

    Returns
    -------
    sigma : array
    n_eff : array
    a_eff : array
        Effective area per r (compare to the naive bounding-box area to see
        how much low-coverage regions are being discounted).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)

    x_edges = np.linspace(x.min(), x.max(), n_grid + 1)
    y_edges = np.linspace(y.min(), y.max(), n_grid + 1)
    cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    n_local = counts / cell_area  # local areal density per cell, n_grid x n_grid

    d_ln_r = np.gradient(np.log(r))

    a_eff = np.empty_like(r)
    for i, Dp_i in enumerate(Dp):
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (n_local * Dp_i) / (1.0 + n_local * Dp_i)
        a_eff[i] = np.sum(w**2) * cell_area

    n_eff = 2.0 * np.pi * a_eff * d_ln_r / r**2
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = Dp * np.sqrt(2.0 / n_eff)
    return sigma, n_eff, a_eff


def pair_covariance_vsf_error(xy, i_pairs, j_pairs, bin_idx, Dfun, sig_n=0.0,
                               n_mc=150_000, seed=0, n_bins=None):
    """Exact pair-covariance error on the VSF, in real space, via Monte Carlo.

    Unlike every FKP-style estimator above (which all approximate "how many
    independent samples does this bin have" through some geometric or
    mode-counting proxy), this computes the actual covariance of the
    structure-function estimator under Gaussian, zero-mean increments, given
    only a smooth model D(r) for the true structure function and the
    survey's actual pair geometry (so it automatically handles the mask --
    gaps, edges, non-uniform coverage -- with no separate area/density
    bookkeeping).

    For two pairs a=(i,j), b=(k,l), the covariance of their velocity
    increments under a Gaussian field is

        C_ab = 1/2 [D(x_i-x_l) + D(x_j-x_k) - D(x_i-x_k) - D(x_j-x_l)]
               + sig_n^2 (delta_ik - delta_il - delta_jk + delta_jl)

    and for Gaussian increments Cov(Delta_a^2, Delta_b^2) = 2 C_ab^2, so the
    variance of the bin-averaged structure function D_hat = mean(Delta_a^2)
    over N_p pairs is Var[D_hat] = (2/N_p^2) sum_ab C_ab^2. N_p^2 pair-pairs
    is too many to sum directly, so this estimates the sum by Monte Carlo:
    draw M random pair-pairs (a, b) uniformly from the bin's own pair list
    (with replacement, so a=b -- the diagonal term -- is included with the
    correct 1/N_p probability automatically) and average C_ab^2.

    N_eff(r) = D(r)^2 / mean(C_ab^2) is the Kish/Bretherton effective
    sample size implied by that variance -- directly comparable to the
    N_eff(r) the FKP-style functions above only approximate.

    Caveats carried over directly from the derivation (not handled here):
    - `Dfun` should be the PSF-smoothed structure function (flattened below
      the beam FWHM) if the data are beam-smoothed, and `sig_n` should be
      replaced by the actual noise covariance if neighboring-spaxel noise
      is correlated. Skipping this is "the most common reason N_eff comes
      out too high at small r" -- this implementation does neither, so
      treat the small-r end of its output with that caveat in mind.
    - `Dfun` here is built from the single observed D(r) curve, not
      iteratively refit with the resulting errors; 2-3 iterations
      (refit Dfun with the new weights, recompute) would sharpen it.

    Parameters
    ----------
    xy : (N, 2) array
        Point positions.
    i_pairs, j_pairs : arrays
        Pair index arrays (all pairs across all bins -- same length).
    bin_idx : array
        Bin index for each pair, same length as i_pairs/j_pairs (as
        produced by the same pair/bin-assignment logic used elsewhere,
        e.g. `_vsf_pairs_all`).
    Dfun : callable
        Vectorized model for the true structure function, `Dfun(r) -> D`,
        with `Dfun(0) = 0`.
    sig_n : float
        Representative per-point noise std (same convention as `vel_err`).
        0 disables the noise term.
    n_mc : int
        Monte Carlo draws of (pair, pair) per bin.
    seed : int
        RNG seed.
    n_bins : int, optional
        Total number of bins. Defaults to `bin_idx.max() + 1`, but pass this
        explicitly (e.g. `len(bin_edges) - 1`) whenever the highest-r bin(s)
        might have zero pairs -- otherwise those trailing empty bins get
        silently dropped instead of returned as NaN, truncating the output
        and desynchronizing it from whatever `r` array you're aligning to.

    Returns
    -------
    sigma : array
        sqrt(Var[D_hat]) per bin (NaN where a bin has <2 pairs).
    n_eff : array
        D(r)^2 / mean(C_ab^2) per bin.
    r_used : array
        Median pair separation actually used per bin (for comparison
        against whatever bin-center convention `r` elsewhere uses).
    """
    xy = np.asarray(xy, dtype=float)
    i_pairs = np.asarray(i_pairs)
    j_pairs = np.asarray(j_pairs)
    bin_idx = np.asarray(bin_idx)
    rng = np.random.default_rng(seed)

    if n_bins is not None:
        nbins = int(n_bins)
    else:
        nbins = int(bin_idx.max()) + 1 if bin_idx.size else 0
    sigma = np.full(nbins, np.nan)
    n_eff = np.full(nbins, np.nan)
    r_used = np.full(nbins, np.nan)

    for b in range(nbins):
        sel = np.flatnonzero(bin_idx == b)
        n_p = sel.size
        if n_p < 2:
            continue
        pi, pj = i_pairs[sel], j_pairs[sel]

        sep = np.linalg.norm(xy[pi] - xy[pj], axis=1)
        r_b = np.median(sep)
        d_b = float(Dfun(np.array([r_b]))[0])
        r_used[b] = r_b

        ia = rng.integers(n_p, size=n_mc)
        ib = rng.integers(n_p, size=n_mc)
        i, j = pi[ia], pj[ia]
        k, l = pi[ib], pj[ib]

        d_il = Dfun(np.linalg.norm(xy[i] - xy[l], axis=1))
        d_jk = Dfun(np.linalg.norm(xy[j] - xy[k], axis=1))
        d_ik = Dfun(np.linalg.norm(xy[i] - xy[k], axis=1))
        d_jl = Dfun(np.linalg.norm(xy[j] - xy[l], axis=1))
        c_ab = 0.5 * (d_il + d_jk - d_ik - d_jl)
        if sig_n > 0:
            c_ab = c_ab + sig_n**2 * (
                (i == k).astype(float) - (i == l).astype(float)
                - (j == k).astype(float) + (j == l).astype(float)
            )

        mean_c2 = np.mean(c_ab**2)
        var_d = 2.0 * mean_c2
        sigma[b] = np.sqrt(var_d)
        n_eff[b] = (d_b**2 / mean_c2) if mean_c2 > 0 else np.nan

    return sigma, n_eff, r_used


def fkp_vsf_error_capped_geometric_calibrated(r, Dp, n_bar, area_survey, jackknife_err,
                                               r_cap, n_eff_floor=1.0):
    """FKP-style VSF error: geometric N_eff(r), but with the disk-tiling
    radius capped at a fixed physical scale `r_cap` instead of growing with
    r indefinitely (contrast `fkp_vsf_error_geometric_calibrated`).

    The uncapped disk-tiling argument, N_eff(r) = Area/(pi r^2), implicitly
    assumes the turbulence's own coherence/correlation length scales with
    whatever lag r you happen to be measuring. That isn't physical: a
    turbulent field has one fixed correlation length set by its energy
    injection/driving scale, independent of which separation is being
    queried. Using r as the tiling-cell radius at every r means that once
    you probe lags comparable to the survey footprint, N_eff collapses
    below 1 (e.g. it already does for Flame by r~13 pc, well inside the
    probed range) and the error blows up past the signal itself -- not
    because the data actually got that much less constraining, but because
    the formula keeps inflating the assumed coherence cell past the point
    where the physics says pairs keep decorrelating.

    This keeps the same disk-tiling argument and the same jackknife
    amplitude calibration, but caps the radius used in the cell-size term:

        N_eff(r) = Area_survey / (pi * min(r, r_cap)^2)

    so N_eff stops shrinking once r exceeds r_cap, bounding the error
    instead of letting it diverge across the rest of the probed range.
    `r_cap` should be an independent estimate of the field's true coherence
    length -- a natural, already-available default is the spatial block
    size used by the block jackknife itself, sqrt(area_survey)/
    n_blocks_per_side, since that's already the "one coherent patch" scale
    the jackknife treats as independent.

    Returns
    -------
    sigma : array
    n_eff_smooth : array
    kappa : float
    """
    r = np.asarray(r, dtype=float)
    Dp = np.asarray(Dp, dtype=float)
    jackknife_err = np.asarray(jackknife_err, dtype=float)

    r_eff = np.minimum(r, r_cap)
    geometric_shape = area_survey / (np.pi * r_eff**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_jack = jackknife_err / Dp
        n_eff_raw = 2.0 / rel_jack**2
        ratio = n_eff_raw / geometric_shape

    good = np.isfinite(ratio) & (ratio > 0)
    kappa = 10**np.median(np.log10(ratio[good]))

    n_eff_smooth = np.maximum(kappa * geometric_shape, n_eff_floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = Dp * np.sqrt(2.0 / n_eff_smooth) * (1.0 + 1.0 / (n_bar * Dp))
    return sigma, n_eff_smooth, kappa


def _vsf_pairs_all(x, y, vel, bin_edges, order, min_separation=None):
    """Sum(|dv|^order) and pair counts per r-bin using ALL pairs (no subsample cap).

    Only safe for point counts where the full all-pairs cKDTree query is
    affordable (a few x 1e4 points at most); see compute_vsf_block_jackknife_2d.
    """
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=bin_edges[-1], output_type='ndarray')

    nbins = len(bin_edges) - 1
    vsf = np.zeros(nbins)
    counts = np.zeros(nbins)
    if pairs.size == 0:
        return vsf, counts

    i, j = pairs[:, 0], pairs[:, 1]
    dr2 = np.sum((coords[i] - coords[j])**2, axis=1)
    dv_p = np.abs(vel[i] - vel[j])**order

    bin_idx = np.searchsorted(bin_edges**2, dr2) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    if min_separation is not None:
        valid &= dr2 >= float(min_separation)**2

    r_i = bin_idx[valid]
    np.add.at(vsf, r_i, dv_p[valid])
    np.add.at(counts, r_i, 1)
    return vsf, counts


def noise_injection_vsf_2d(x, y, vel, vel_err, bin_edges, order,
                            n_realizations=150, seed=0, min_separation=None,
                            resample_errors=None):
    """Empirical noise-bias and noise-scatter on the VSF via noise injection.

    Repeatedly adds a random draw of the per-point measurement noise to the
    observed velocities and recomputes the structure function, instead of
    relying on an analytic noise-bias formula (only clean for order=2) or
    an independence assumption for the propagated error (wrong whenever a
    point is reused across many pairs, which is always, for a VSF). This
    directly measures:

    - bias(r) = mean(S_sim(r)) - S_observed(r): how much the noise pushes
      the estimator away from the true (noise-free) value -- subtract this.
    - scatter(r) = std(S_sim(r)): the actual noise-driven uncertainty on
      the mean, correctly reflecting that each point's noise enters many
      pairs (no independent-pairs assumption needed).

    Cost control: the (x, y) pair list and their r-bin assignment are built
    ONCE via `_vsf_pairs_all`'s cKDTree query (the only O(N^2)-ish step),
    since only the velocities change between realizations. Each realization
    then costs just `|dv|^order` + a bincount over the fixed pair list --
    no repeated tree queries. Affordable up to the same few x 1e4 points
    `_vsf_pairs_all` is safe for.

    Parameters
    ----------
    x, y, vel, vel_err : arrays
        Point positions, velocities, and per-point 1-sigma velocity errors.
    bin_edges : array
        Spatial bin edges (must match the bins S_observed was computed on).
    order : int
        Structure function order p.
    n_realizations : int
        Number of noise draws (150-200 is usually enough for the scatter to
        stabilize to within a few percent; bump it up if sigma itself looks
        noisy bin-to-bin).
    seed : int
        RNG seed.
    min_separation : float, optional
        Same meaning as elsewhere; must match how S_observed was computed.
    resample_errors : callable, optional
        ``resample_errors(rng, vel_err) -> noise_draw`` for a non-Gaussian
        noise model. Defaults to ``rng.normal(0, vel_err)``.

    Returns
    -------
    bias : array
        S_sim_mean(r) - S_observed(r) at the bins S_observed was computed on
        (S_observed is recomputed here on the unperturbed data for an exact
        bin-for-bin match).
    scatter : array
        std(S_sim(r), ddof=1) across realizations.
    s_observed : array
        The noise-free-data structure function on the same pair list, for
        convenience (should match your existing S(r) up to the usual
        min_pairs masking).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vel = np.asarray(vel, dtype=float)
    vel_err = np.asarray(vel_err, dtype=float)

    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=bin_edges[-1], output_type='ndarray')

    nbins = len(bin_edges) - 1
    if pairs.size == 0:
        nan = np.full(nbins, np.nan)
        return nan, nan, nan

    i, j = pairs[:, 0], pairs[:, 1]
    dr2 = np.sum((coords[i] - coords[j])**2, axis=1)
    bin_idx = np.searchsorted(bin_edges**2, dr2) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    if min_separation is not None:
        valid &= dr2 >= float(min_separation)**2
    i, j, bin_idx = i[valid], j[valid], bin_idx[valid]

    counts = np.bincount(bin_idx, minlength=nbins).astype(float)

    def _s_of(v):
        dv_p = np.abs(v[i] - v[j])**order
        vsf_sum = np.bincount(bin_idx, weights=dv_p, minlength=nbins)
        with np.errstate(divide='ignore', invalid='ignore'):
            return vsf_sum / counts

    s_observed = _s_of(vel)

    rng = np.random.default_rng(seed)
    s_sim = np.empty((n_realizations, nbins))
    for k in range(n_realizations):
        if resample_errors is not None:
            noise = resample_errors(rng, vel_err)
        else:
            noise = rng.normal(0.0, vel_err)
        s_sim[k] = _s_of(vel + noise)

    bias = np.nanmean(s_sim, axis=0) - s_observed
    scatter = np.nanstd(s_sim, axis=0, ddof=1)
    return bias, scatter, s_observed

def _sample_pairwise_log_separations(coords, r_max, max_pairs, seed):
    """Pairwise log10(separation) for a point subsample, bounding O(n^2) cost."""
    rng = np.random.default_rng(seed)
    n_points = len(coords)
    target_n = int(min(n_points, max(50, np.sqrt(2 * max_pairs))))
    if target_n < n_points:
        sub_idx = rng.choice(n_points, size=target_n, replace=False)
        sub_coords = coords[sub_idx]
    else:
        sub_coords = coords
    sub_tree = cKDTree(sub_coords)
    pairs = sub_tree.query_pairs(r=r_max, output_type='ndarray')
    if pairs.size == 0:
        return None
    sep = np.linalg.norm(sub_coords[pairs[:, 0]] - sub_coords[pairs[:, 1]], axis=1)
    sep = sep[sep > 0]
    if sep.size == 0:
        return None
    return np.log10(sep)


def _shell_binning_edges(coords, r_min, r_max, bin_resolution, transition_factor,
                          fine_per_bin, smooth_sigma, prominence_frac,
                          max_pairs, seed):
    """Bin edges for binning="shell": placed at density VALLEYS of the actual
    pairwise-separation distribution near r_min, rather than at arbitrary
    equal-log-width positions that can bisect a discrete pixel/fiber-lattice
    separation shell.

    On gridded/lattice data (regular pixel grids, hex fiber bundles, etc.)
    separations can only take a finite set of discrete values. A blind
    equal-width log grid places edges without any regard for where those
    values actually are, so some bins land squarely on a populated shell
    (thousands of pairs) while the very next bin lands in the gap between
    shells (a handful of stray pairs from positional jitter) -- causing both
    the VSF and its error bars to swing wildly bin-to-bin for reasons that
    are purely about bin-edge placement, not real small-scale structure.

    This finds the shells directly from the data: a finely-sampled histogram
    of pairwise separations is lightly smoothed, and edges are placed at the
    smoothed curve's local minima (valleys) with prominence above
    `prominence_frac` of the peak density -- i.e. only at *real* gaps between
    clusters of pairs, not at every small wiggle from Poisson noise. No
    minimum-pair-count floor is enforced here (unlike binning=
    "adaptive_min_count"): a shell with genuinely few pairs (e.g. a rarer
    dither sub-lattice spacing) simply becomes its own low-count bin rather
    than being merged into a neighbor -- the point is correct edge
    *placement*, not equalizing counts.

    Only applied up to `transition_factor * r_min` -- beyond that, shells are
    dense enough (many lattice combinations per decade) that an ordinary
    equal-width log grid already behaves fine, and continuing shell-detection
    out to r_max would mean comparing valley prominence against a global
    histogram max set by the huge, densely-populated small-r shells, which
    would suppress genuine (much smaller in absolute terms) structure out
    there. Beyond the transition radius this falls back to plain equal-width
    log bins, concatenated onto the shell-detected edges.
    """
    transition_r = min(r_min * transition_factor, r_max)

    log_sep = _sample_pairwise_log_separations(coords, r_max, max_pairs, seed)
    if log_sep is None:
        length = max(int(np.log10(r_max / r_min) * bin_resolution), 10)
        return np.linspace(np.log10(r_min), np.log10(r_max), length + 1)
    sep = 10**log_sep

    shell_part = sep[(sep >= r_min) & (sep <= transition_r)]
    if shell_part.size == 0 or transition_r <= r_min:
        shell_edges_log10 = np.array([np.log10(r_min)])
    else:
        n_fine = max(int(np.log10(transition_r / r_min) * bin_resolution * fine_per_bin), 20)
        fine_edges = np.logspace(np.log10(r_min), np.log10(transition_r), n_fine + 1)
        counts, _ = np.histogram(shell_part, bins=fine_edges)
        smoothed = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma, mode='nearest')
        if smoothed.max() > 0:
            prominence = prominence_frac * smoothed.max()
            valley_idx, _ = find_peaks(-smoothed, prominence=prominence)
        else:
            valley_idx = np.array([], dtype=int)
        edge_idx = np.unique(np.concatenate(([0], valley_idx, [n_fine])))
        shell_edges_log10 = np.log10(fine_edges[edge_idx])

    if transition_r < r_max:
        n_remaining = max(int(np.log10(r_max / transition_r) * bin_resolution), 3)
        remaining_edges_log10 = np.linspace(np.log10(transition_r), np.log10(r_max), n_remaining + 1)
        edges_log10 = np.concatenate([shell_edges_log10, remaining_edges_log10[1:]])
    else:
        edges_log10 = shell_edges_log10

    return np.unique(edges_log10)


def spatial_bin_edges(x_data, y_data, bin_resolution=80, binning="equal_width",
                       min_separation=None, min_pairs_per_bin=2000,
                       max_pairs_for_sampling=500_000, seed=0,
                       shell_transition_factor=10.0, shell_fine_per_bin=25,
                       shell_smooth_sigma=3.0, shell_prominence_frac=0.05):
    """Shared spatial bin-edge logic, used by both compute_bins (PDF/kurtosis
    pipeline) and VSF2D.generate_vsf_2d (production VSF pipeline) so a fix
    here benefits both instead of living in two duplicated implementations.

    See compute_bins for the meaning of "equal_width"/"equal_count"/
    "adaptive_min_count"; "shell" is documented in _shell_binning_edges.

    Returns
    -------
    log_centers, bin_edges, r_min, r_max
    """
    coords = np.column_stack((x_data, y_data))

    tree = cKDTree(coords)
    sample_size = min(1000000, len(coords))
    sample_idx = np.random.choice(len(coords), sample_size, replace=False)
    dists, _ = tree.query(coords[sample_idx], k=2)
    r_min = np.median(dists[:, 1])

    dx = x_data.max() - x_data.min()
    dy = y_data.max() - y_data.min()
    r_max = np.sqrt(dx**2 + dy**2)

    if min_separation is not None:
        r_min = max(r_min, float(min_separation))

    if r_min <= 0 or r_min >= r_max:
        r_min = r_max / 1000

    log_range = np.log10(r_max / r_min)
    length = max(int(log_range * bin_resolution), 10)

    if binning == "equal_count":
        log_sep = _sample_pairwise_log_separations(coords, r_max, max_pairs_for_sampling, seed)
        if log_sep is None:
            raise ValueError("binning='equal_count' found no pairs to estimate bin edges from")
        quantiles = np.linspace(0.0, 1.0, length + 1)
        edges = np.unique(np.quantile(log_sep, quantiles))
        if edges.size < 3:
            raise ValueError("binning='equal_count' degenerate: too few distinct separations")
        log_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges = 10**edges
    elif binning == "adaptive_min_count":
        fine_centers = np.linspace(np.log10(r_min), np.log10(r_max), length)
        bw = np.diff(fine_centers) / 2
        fine_edges = np.empty(len(fine_centers) + 1)
        fine_edges[1:-1] = fine_centers[:-1] + bw
        fine_edges[0] = fine_centers[0] - bw[0]
        fine_edges[-1] = fine_centers[-1] + bw[-1]

        log_sep = _sample_pairwise_log_separations(coords, r_max, max_pairs_for_sampling, seed)
        if log_sep is None:
            raise ValueError("binning='adaptive_min_count' found no pairs to estimate bin counts from")
        fine_counts, _ = np.histogram(log_sep, bins=fine_edges)

        merged = [fine_edges[0]]
        acc = 0
        for i, count in enumerate(fine_counts):
            acc += count
            if acc >= min_pairs_per_bin:
                merged.append(fine_edges[i + 1])
                acc = 0
        if acc > 0:
            if len(merged) > 1:
                merged[-1] = fine_edges[-1]
            else:
                merged.append(fine_edges[-1])
        edges = np.array(merged)
        if edges.size < 3:
            raise ValueError("binning='adaptive_min_count' degenerate: min_pairs_per_bin too high for the data")
        log_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges = 10**edges
    elif binning == "shell":
        edges = _shell_binning_edges(
            coords, r_min, r_max, bin_resolution,
            shell_transition_factor, shell_fine_per_bin,
            shell_smooth_sigma, shell_prominence_frac,
            max_pairs_for_sampling, seed,
        )
        if edges.size < 3:
            raise ValueError("binning='shell' degenerate: too few distinct separations")
        log_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges = 10**edges
    else:
        log_centers = np.linspace(np.log10(r_min), np.log10(r_max), length)
        bw = np.diff(log_centers) / 2
        edges = np.empty(len(log_centers) + 1)
        edges[1:-1] = log_centers[:-1] + bw
        edges[0] = log_centers[0] - bw[0]
        edges[-1] = log_centers[-1] + bw[-1]
        bin_edges = 10**edges

    return log_centers, bin_edges, r_min, r_max


def compute_bins(x_data, y_data, vel_data, bin_resolution=80, binning="equal_width",
                  min_pairs_per_bin=2000, max_pairs_for_equal_count=500_000, seed=0,
                  min_separation=None):
    """
    Compute spatial and velocity bins independently.

    This function can be called once and its results reused for multiple
    VSF computations, avoiding redundant computation.

    Parameters
    ----------
    x_data : array
        X coordinates
    y_data : array
        Y coordinates
    vel_data : array
        Velocity data
    bin_resolution : int
        Number of bins per decade in log-space separation (binning="equal_width"
        and "adaptive_min_count", where it sets the starting fine-grid
        resolution before merging), or bins per decade of the full separation
        range used only to size the number of bins (binning="equal_count").
    binning : {"equal_width", "equal_count", "adaptive_min_count", "shell"}
        "equal_width" (default, original behavior): bins of equal width in
        log10(r). Because pair counts in 2D grow as ~r^2 for fixed fractional
        bin width, the smallest-r bins are intrinsically the most pair-starved
        regardless of how much data you have.
        "equal_count": bin edges are placed at quantiles of the actual
        pairwise-separation distribution, so every bin gets ~exactly the same
        number of pairs. This equalizes counts perfectly but over-corrects:
        the first bin can end up spanning most of the dynamic range just to
        collect enough pairs, smearing out genuine small-r structure (e.g. a
        sharp kurtosis peak) rather than resolving it.
        "adaptive_min_count": starts from the same fine equal-width grid, then
        greedily merges consecutive bins starting from r_min until each
        merged bin clears `min_pairs_per_bin` pairs. Bins that already have
        enough pairs (typically everything away from r_min) are left at full
        resolution untouched. This is the usual middle ground: it removes the
        pair-count starvation right at r_min without flattening the whole
        curve's resolution the way "equal_count" does.
        "shell": for gridded/lattice data (regular pixel grids, hex fiber
        bundles), separations can only take a finite set of discrete values
        near r_min. Bin edges are placed at density valleys of the actual
        pairwise-separation distribution (see _shell_binning_edges) so each
        edge falls in a genuine gap between lattice-separation shells instead
        of randomly bisecting one -- this is a *placement* fix, not a count
        floor (see shell_* kwargs on spatial_bin_edges for the transition
        radius / smoothing / valley-prominence knobs); no `min_pairs_per_bin`
        merging is applied under this mode.
    min_pairs_per_bin : int
        Target floor used by binning="adaptive_min_count".
    max_pairs_for_equal_count : int
        Cap on points used to estimate the pairwise-separation distribution
        for binning="equal_count"/"adaptive_min_count" (bounds the O(n^2)
        pair enumeration cost).
    seed : int
        RNG seed for the point subsample used by binning="equal_count"/
        "adaptive_min_count".
    min_separation : float, optional
        Floor on r_min -- pass a fixed value here (e.g. from a reference
        dataset) to force the same bin grid floor across runs with
        different point counts/densities, instead of each one
        auto-detecting its own r_min from its own nearest-neighbor
        spacing. Only raises r_min (never lowers it below what would be
        auto-detected).

    Returns
    -------
    log_centers : array
        Log-space bin centers for spatial separation
    bin_edges : array
        Spatial bin edges (linear space)
    dv_bins : array
        Velocity difference bin edges
    dict : dict
        Dictionary containing all computed bins
    """
    # Compute spatial bins (shared with VSF2D.generate_vsf_2d via spatial_bin_edges)
    log_centers, bin_edges, r_min, r_max = spatial_bin_edges(
        x_data, y_data,
        bin_resolution=bin_resolution,
        binning=binning,
        min_pairs_per_bin=min_pairs_per_bin,
        max_pairs_for_sampling=max_pairs_for_equal_count,
        seed=seed,
        min_separation=min_separation,
    )

    # Compute velocity bins
    dv_min = np.nanmin(vel_data) - np.mean(vel_data)
    dv_max = np.nanmax(vel_data) - np.mean(vel_data)
    dv_bins = np.linspace(dv_min, dv_max, 80)

    bins_dict = {
        'log_centers': log_centers,
        'bin_edges': bin_edges,
        'dv_bins': dv_bins,
        'r_min': r_min,
        'r_max': r_max,
        'binning': binning,
    }

    return log_centers, bin_edges, dv_bins, bins_dict


def compute_pdf_and_statistics_chunk(x, y, vel, vel_err, bin_edges, dv_bins, 
                                     seed, order):
    """
    Compute PDF, C2, C4, and counts for a chunk of data.
    
    This function can be parallelized and results merged independently
    from the main VSF computation.
    
    Parameters
    ----------
    x, y : arrays
        Spatial coordinates
    vel : array
        Velocity
    vel_err : array
        Velocity errors
    bin_edges : array
        Spatial bin edges
    dv_bins : array
        Velocity bin edges
    seed : int
        Random seed for reproducibility
    order : int
        Order of structure function (1, 2, 4, etc.)
    
    Returns
    -------
    vsf, vsf_sq, vsf_properr, counts : arrays
        Structure function statistics
    pdf : 2D array
        Probability density function (separation x velocity)
    c2_r, c4_r, cnt_r : arrays
        Intermittency statistics
    """
    rng = np.random.default_rng(seed)
    sample_size = min(10000, len(x))
    idx = rng.choice(len(x), size=sample_size, replace=False)
    
    x_, y_, vel_, verr_ = x[idx], y[idx], vel[idx], vel_err[idx]
    coords = np.column_stack((x_, y_))
    
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=bin_edges[-1], output_type='ndarray')
    
    nbins = len(bin_edges) - 1
    ndv = len(dv_bins) - 1
    
    vsf = np.zeros(nbins)
    vsf_sq = np.zeros(nbins)
    vsf_properr = np.zeros(nbins)
    counts = np.zeros(nbins)
    pdf = np.zeros((nbins, ndv))
    
    # r-only intermittency
    c2_r = np.zeros(nbins)
    c4_r = np.zeros(nbins)
    cnt_r = np.zeros(nbins)
    
    if pairs.size == 0:
        return vsf, vsf_sq, vsf_properr, counts, pdf, c2_r, c4_r, cnt_r
    
    i, j = pairs[:, 0], pairs[:, 1]
    
    dr2 = np.sum((coords[i] - coords[j])**2, axis=1)
    dv = vel_[i] - vel_[j]
    
    dv_p = np.abs(dv)**order
    pair_err = order * dv**(order - 1) * np.sqrt(verr_[i]**2 + verr_[j]**2)
    
    # --- r bins ---
    bin_idx = np.searchsorted(bin_edges**2, dr2) - 1
    valid_r = (bin_idx >= 0) & (bin_idx < nbins)
    
    r_i = bin_idx[valid_r]
    
    np.add.at(vsf, r_i, dv_p[valid_r])
    np.add.at(vsf_sq, r_i, dv_p[valid_r]**2)
    np.add.at(vsf_properr, r_i, pair_err[valid_r]**2)
    np.add.at(counts, r_i, 1)
    
    # --- intermittency (r ONLY) ---
    c2 = np.abs(dv)**2
    c4 = np.abs(dv)**4
    np.add.at(c2_r, r_i, c2[valid_r])
    np.add.at(c4_r, r_i, c4[valid_r])
    np.add.at(cnt_r, r_i, 1)
    
    # --- PDF (KEEP dv_bins) ---
    dv_idx = np.digitize(dv, dv_bins) - 1
    valid_dv = (dv_idx >= 0) & (dv_idx < ndv)
    
    valid = valid_r & valid_dv
    r_i_pdf = bin_idx[valid]
    dv_i = dv_idx[valid]
    
    np.add.at(pdf, (r_i_pdf, dv_i), 1)
    
    return vsf, vsf_sq, vsf_properr, counts, pdf, c2_r, c4_r, cnt_r


def compute_pdf_and_statistics(x_data, y_data, vel_data, vel_err_data, 
                               log_centers, bin_edges, dv_bins, order=2, 
                               n_jobs=8):
    """
    Compute PDF, C2, C4, and counts independently.
    
    This function computes all the PDF and intermittency statistics
    without computing the full VSF. Results can be saved and reused.
    
    Parameters
    ----------
    x_data, y_data : arrays
        Spatial coordinates
    vel_data : array
        Velocity data
    vel_err_data : array
        Velocity errors
    log_centers : array
        Log-space bin centers
    bin_edges : array
        Spatial bin edges
    dv_bins : array
        Velocity bin edges
    order : int
        Order of computation
    n_jobs : int
        Number of parallel jobs
    
    Returns
    -------
    pdf_all : 2D array
        PDF (separation x velocity)
    c2_all, c4_all : arrays
        Intermittency statistics
    cnt_all : array
        Counts for intermittency
    dv_centers : array
        Velocity bin centers
    kk : array
        Kurtosis (intermittency parameter)
    """
    nbins = len(bin_edges) - 1
    
    pdf_all = np.zeros((nbins, len(dv_bins) - 1))
    c2_all = np.zeros(nbins)
    c4_all = np.zeros(nbins)
    cnt_all = np.zeros(nbins)
    
    seeds = np.arange(n_jobs)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pdf_and_statistics_chunk)(
            x_data, y_data, vel_data, vel_err_data,
            bin_edges, dv_bins, seed, order
        ) for seed in seeds
    )
    
    # --- merge ---
    for vsf_c, vsf_sq_c, err_c, cnt_c, pdf_c, c2_c, c4_c, cnt_r_c in results:
        pdf_all += pdf_c
        c2_all += c2_c
        c4_all += c4_c
        cnt_all += cnt_r_c
    
    # --- INTERMITTENCY ---
    kk = np.divide(
        c4_all / cnt_all,
        (c2_all / cnt_all)**2,
        out=np.zeros_like(c4_all),
        where=c2_all > 0
    )
    
    # --- normalize PDF ---
    pdf_sum = np.sum(pdf_all, axis=1, keepdims=True)
    pdf_all = np.divide(pdf_all, pdf_sum, out=np.zeros_like(pdf_all), 
                       where=pdf_sum > 0)
    
    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    
    return pdf_all, c2_all, c4_all, cnt_all, dv_centers, kk


# ============================================================================
# MAIN VSF2D CLASS
# ============================================================================

class VSF2D:
    """
    2D Velocity Structure Function analyzer.
    
    Computes velocity structure functions from spatial and velocity data.
    Supports angular (RA/Dec) or physical coordinates.
    """
    
    def __init__(self, ra, dec, vel, dist, coords='angular', vel_err=None):
        """
        Initialize VSF2D object.
        
        Parameters
        ----------
        ra, dec : arrays
            Right ascension and declination
        vel : array
            Velocity data
        dist : float
            Distance in parsecs
        coords : str
            'angular' or 'physical'
        vel_err : array, optional
            Velocity uncertainties
        """
        if coords == "angular":
            self.ra = ra
            self.dec = dec
            self.vel = vel
            self.vel_err = vel_err if vel_err is not None else np.zeros_like(vel)
            if self.vel_err.size == 0:
                self.vel_err = np.zeros_like(vel)
            coords_obj = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            ref_coord = SkyCoord(ra=coords_obj.ra.mean(), dec=coords_obj.dec.mean(), 
                               frame='icrs')
            
            delta_ra = (coords_obj.ra - ref_coord.ra) * np.cos(ref_coord.dec)
            delta_dec = coords_obj.dec - ref_coord.dec
            
            x = delta_ra.to(u.arcsec).value
            y = delta_dec.to(u.arcsec).value
            
            distance = dist * u.pc
            
            self.x_physical = (x * u.arcsec * distance).to(
                u.pc, equivalencies=u.dimensionless_angles()).value
            self.y_physical = (y * u.arcsec * distance).to(
                u.pc, equivalencies=u.dimensionless_angles()).value
        else:  # coords == 'physical'
            self.x_physical = ra
            self.y_physical = dec
            self.vel = vel
            self.vel_err = vel_err if vel_err is not None else np.zeros_like(vel)
            if self.vel_err.size == 0:
                self.vel_err = np.zeros_like(vel)
    
    def determine_length(self, r_min, r_max, bin_resolution=5):
        """Determine number of bins."""
        if r_min <= 0 or r_min >= r_max:
            raise ValueError("Invalid r_min/r_max")
        log_range = np.log10(r_max / r_min)
        return max(int(log_range * bin_resolution), 10)
    
    def generate_bins_2d(self, data, bin_resolution=80, length=None,
                         min_separation=None):
        """Generate logarithmic spatial bins."""
        coords = np.column_stack((data['x'], data['y']))
        tree = cKDTree(coords)
        
        sample_size = min(1000000, len(coords))
        sample_idx = np.random.choice(len(coords), sample_size, replace=False)
        dists, _ = tree.query(coords[sample_idx], k=2)
        r_min = np.median(dists[:, 1])
        
        dx = data['x'].max() - data['x'].min()
        dy = data['y'].max() - data['y'].min()
        r_max = np.sqrt(dx**2 + dy**2)

        if min_separation is not None:
            r_min = max(r_min, float(min_separation))
        
        if r_min <= 0 or r_min >= r_max:
            r_min = r_max / 1000

        if min_separation is not None:
            r_min = max(r_min, float(min_separation))

        if r_min <= 0 or r_min >= r_max:
            raise ValueError("Invalid r_min/r_max after applying min_separation")
        
        if length is None:
            length = self.determine_length(r_min, r_max, bin_resolution)
        
        log_centers = np.linspace(np.log10(r_min), np.log10(r_max), length)
        return log_centers
    
    def generate_bins_edges(self, bin_centers):
        """Convert bin centers to edges."""
        bw = np.diff(bin_centers) / 2
        edges = np.empty(len(bin_centers) + 1)
        edges[1:-1] = bin_centers[:-1] + bw
        edges[0] = bin_centers[0] - bw[0]
        edges[-1] = bin_centers[-1] + bw[-1]
        return edges
    
    def compute_vsf_chunk_2d(self, x, y, vel, vel_err, bin_edges,
                            seed, order, min_separation=None):
        """Compute VSF for a chunk of data (VSF ONLY, no PDF/C2/C4)."""
        rng = np.random.default_rng(seed)
        sample_size = min(10000, len(x))
        idx = rng.choice(len(x), size=sample_size, replace=False)
        
        x_, y_, vel_, verr_ = x[idx], y[idx], vel[idx], vel_err[idx]
        coords = np.column_stack((x_, y_))
        
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=bin_edges[-1], output_type='ndarray')
        
        nbins = len(bin_edges) - 1
        
        vsf = np.zeros(nbins)
        vsf_sq = np.zeros(nbins)
        vsf_properr = np.zeros(nbins)
        counts = np.zeros(nbins)
        
        if pairs.size == 0:
            return vsf, vsf_sq, vsf_properr, counts
        
        i, j = pairs[:, 0], pairs[:, 1]
        
        dr2 = np.sum((coords[i] - coords[j])**2, axis=1)
        dv = vel_[i] - vel_[j]
        
        dv_p = np.abs(dv)**order
        pair_err = order * dv**(order - 1) * np.sqrt(verr_[i]**2 + verr_[j]**2)
        
        # --- r bins ---
        bin_idx = np.searchsorted(bin_edges**2, dr2) - 1
        valid_r = (bin_idx >= 0) & (bin_idx < nbins)
        if min_separation is not None:
            valid_r &= dr2 >= float(min_separation)**2
        
        r_i = bin_idx[valid_r]
        
        np.add.at(vsf, r_i, dv_p[valid_r])
        np.add.at(vsf_sq, r_i, dv_p[valid_r]**2)
        np.add.at(vsf_properr, r_i, pair_err[valid_r]**2)
        np.add.at(counts, r_i, 1)
        
        return vsf, vsf_sq, vsf_properr, counts

    def compute_vsf_bootstrap_2d(
        self,
        x,
        y,
        vel,
        bin_edges,
        order,
        n_bootstrap=100,
        seed=12345,
        sample_size=10000,
        min_separation=None,
    ):
        """Bootstrap VSF uncertainty by resampling spaxels, not pairs."""
        rng = np.random.default_rng(seed)
        sample_size = min(sample_size, len(x))
        idx = rng.choice(len(x), size=sample_size, replace=False)

        x_, y_, vel_ = x[idx], y[idx], vel[idx]
        coords = np.column_stack((x_, y_))

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=bin_edges[-1], output_type='ndarray')

        nbins = len(bin_edges) - 1
        if pairs.size == 0:
            return np.full(nbins, np.nan), np.zeros(nbins)

        i, j = pairs[:, 0], pairs[:, 1]
        dr2 = np.sum((coords[i] - coords[j])**2, axis=1)
        dv_p = np.abs(vel_[i] - vel_[j])**order

        bin_idx = np.searchsorted(bin_edges**2, dr2) - 1
        valid_r = (bin_idx >= 0) & (bin_idx < nbins)
        if min_separation is not None:
            valid_r &= dr2 >= float(min_separation)**2
        i = i[valid_r]
        j = j[valid_r]
        bin_idx = bin_idx[valid_r]
        dv_p = dv_p[valid_r]

        bootstrap_vsf = np.full((n_bootstrap, nbins), np.nan)
        bootstrap_counts = np.zeros((n_bootstrap, nbins))

        for b in range(n_bootstrap):
            sampled = rng.integers(0, sample_size, size=sample_size)
            spaxel_counts = np.bincount(sampled, minlength=sample_size)
            pair_weights = spaxel_counts[i] * spaxel_counts[j]
            used = pair_weights > 0
            if not np.any(used):
                continue

            weighted_sum = np.bincount(
                bin_idx[used],
                weights=dv_p[used] * pair_weights[used],
                minlength=nbins,
            )
            weighted_counts = np.bincount(
                bin_idx[used],
                weights=pair_weights[used],
                minlength=nbins,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                bootstrap_vsf[b] = weighted_sum / weighted_counts
            bootstrap_counts[b] = weighted_counts

        return np.nanstd(bootstrap_vsf, axis=0, ddof=1), np.nanmedian(bootstrap_counts, axis=0)

    def compute_vsf_block_jackknife_2d(
        self,
        x,
        y,
        vel,
        bin_edges,
        order,
        n_blocks_per_side=5,
        min_block_points=20,
        min_valid_blocks=4,
        min_separation=None,
        sample_size_per_chunk=10000,
        max_n_jobs_per_replicate=8,
        n_jobs=3,
    ):
        """Spatial block-jackknife VSF error ('cosmic variance' style).

        Tiles the FOV into an n_blocks_per_side x n_blocks_per_side grid of
        spatial blocks and recomputes the VSF with each block held out in
        turn (leave-one-block-out). The jackknife spread across these
        replicates reflects how much the VSF would change under a
        different realization of the field's spatial sampling -- unlike
        pair-level std/sem, it does not assume pairs (or even nearby
        points) are independent draws, and it naturally grows for lag bins
        whose pairs depend on only a few, sparsely populated blocks.

        Each replicate is computed via the SAME chunked/capped subsampling
        as the main point estimate (`compute_vsf_chunk_2d`, pooled over
        several independent draws of up to `sample_size_per_chunk` points
        each -- the number of draws scales with the replicate's point
        count, capped at `max_n_jobs_per_replicate`), not an all-pairs
        KDTree query. This is deliberate, not just a performance
        workaround: it keeps the jackknife error consistent with whatever
        sampling cap the point estimate itself is subject to (relevant for
        fields with more than ~1e4 points, where `generate_vsf_2d` with
        `n_jobs=1` already only sees one 10k-point subsample -- the
        jackknife should reflect that same limitation, not pretend to
        exact all-pairs precision the point estimate doesn't have), and it
        scales to any N instead of being capped at a few x 1e4.

        Returns
        -------
        err : array
            Jackknife standard error per r-bin (NaN where fewer than
            `min_valid_blocks` leave-one-out replicates had data).
        n_valid : array
            Number of valid leave-one-out replicates contributing to each
            bin (diagnostic: low values mean that bin's error estimate is
            itself poorly constrained).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        vel = np.asarray(vel)
        zeros = np.zeros_like(vel)

        x_edges = np.linspace(x.min(), x.max(), n_blocks_per_side + 1)
        y_edges = np.linspace(y.min(), y.max(), n_blocks_per_side + 1)
        xi = np.clip(np.searchsorted(x_edges, x, side='right') - 1, 0, n_blocks_per_side - 1)
        yi = np.clip(np.searchsorted(y_edges, y, side='right') - 1, 0, n_blocks_per_side - 1)
        block_idx = xi * n_blocks_per_side + yi

        nbins = len(bin_edges) - 1

        # Flatten (block, chunk-seed) into one work list so every chunk across
        # every replicate is a single independent unit -- lets joblib spread
        # ALL of it across cores at once, rather than only parallelizing
        # within one replicate at a time (leave-one-out blocks have no
        # dependency on each other, so there's no reason not to).
        work_items = []  # (block_id, xk, yk, velk, zk, seed)
        for block in np.unique(block_idx):
            held_out = block_idx == block
            if np.count_nonzero(held_out) < min_block_points:
                # Block too small to move the statistic; skip leaving it out
                # rather than wasting a near-duplicate replicate.
                continue
            keep = ~held_out
            n_keep = np.count_nonzero(keep)
            if n_keep < 2:
                continue
            xk, yk, velk, zk = x[keep], y[keep], vel[keep], zeros[keep]
            n_jobs_rep = int(np.clip(round(n_keep / sample_size_per_chunk), 1, max_n_jobs_per_replicate))
            for seed in range(n_jobs_rep):
                work_items.append((block, xk, yk, velk, zk, seed))

        if not work_items:
            return np.full(nbins, np.nan), np.zeros(nbins, dtype=int)

        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(self.compute_vsf_chunk_2d)(xk, yk, velk, zk, bin_edges, seed, order, min_separation)
            for (_, xk, yk, velk, zk, seed) in work_items
        )

        per_block = {}
        for (block, *_), (vsf_c, _, _, cnt_c) in zip(work_items, results):
            vsf_sum, counts = per_block.setdefault(block, (np.zeros(nbins), np.zeros(nbins)))
            vsf_sum += vsf_c
            counts += cnt_c

        replicate_means = []
        for vsf_sum, counts in per_block.values():
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.where(counts > 0, vsf_sum / counts, np.nan)
            replicate_means.append(mean)

        if len(replicate_means) < min_valid_blocks:
            return np.full(nbins, np.nan), np.zeros(nbins, dtype=int)

        replicate_means = np.array(replicate_means)
        n_valid = np.sum(np.isfinite(replicate_means), axis=0)
        jack_mean = np.nanmean(replicate_means, axis=0)
        sq_dev = (replicate_means - jack_mean)**2
        with np.errstate(invalid='ignore'):
            var = np.nansum(sq_dev, axis=0) * (n_valid - 1) / np.maximum(n_valid, 1)
        err = np.sqrt(var)
        err = np.where(n_valid >= min_valid_blocks, err, np.nan)
        return err, n_valid

    def generate_vsf_2d(self, order, outname, n_jobs=8, min_pairs=3,
                       bin_resolution=80, error_mode='std', n_bootstrap=100,
                       bootstrap_seed=12345, bootstrap_sample_size=10000,
                       jackknife_n_blocks_per_side=5, jackknife_min_block_points=20,
                       jackknife_min_valid_blocks=4, jackknife_sample_size_per_chunk=10000,
                       jackknife_max_n_jobs_per_replicate=8, jackknife_n_jobs=3,
                       min_separation=None, binning='equal_width',
                       shell_transition_factor=10.0, shell_fine_per_bin=25,
                       shell_smooth_sigma=3.0, shell_prominence_frac=0.05,
                       min_pairs_per_bin=2000, max_pairs_for_binning=2_000_000,
                       binning_seed=0):
        """
        Generate 2D velocity structure function (VSF ONLY).
        
        NOTE: This method computes ONLY the VSF, not PDF/C2/C4 statistics.
        For independent PDF/C2/C4 computation, use compute_pdf_and_statistics().
        
        Parameters
        ----------
        order : int
            Order of structure function (1, 2, 4, etc.)
        outname : str
            Output filename prefix (without extension)
        n_jobs : int
            Number of parallel jobs
        min_pairs : int
            Minimum pairs per bin
        bin_resolution : int
            Number of bins per decade in log-space separation.
        error_mode : {'std', 'sem', 'propagated', 'combined_sem', 'bootstrap', 'block_jackknife'}
            Error returned as ``vsf_err``. ``std`` is the default and uses the
            standard deviation of pair values around the mean, combined in
            quadrature with the propagated mean velocity-error term.
            ``bootstrap`` resamples individual spaxels with replacement.
            ``block_jackknife`` leaves out whole spatial blocks in turn (see
            ``compute_vsf_block_jackknife_2d``) -- a 'cosmic variance' style
            error that reflects spatial correlation and lag-dependent data
            availability, not just pair-counting shot noise. Only use it for
            point counts the all-pairs KDTree can afford (a few x 1e4).
        n_bootstrap : int
            Number of spaxel bootstrap resamples when ``error_mode='bootstrap'``.
        bootstrap_seed : int
            Random seed for bootstrap resampling.
        bootstrap_sample_size : int
            Maximum number of spaxels used in each bootstrap realization.
        jackknife_n_blocks_per_side : int
            Spatial grid side length (blocks = side^2) for ``error_mode='block_jackknife'``.
        jackknife_min_block_points : int
            Minimum points a block must hold to be worth leaving out.
        jackknife_min_valid_blocks : int
            Minimum leave-one-out replicates required per bin, else NaN.
        min_separation : float, optional
            Minimum physical pair separation to include, in the same units as
            ``self.x_physical`` and ``self.y_physical``.
        binning : {'equal_width', 'equal_count', 'adaptive_min_count', 'shell'}
            Spatial binning scheme (see ``spatial_bin_edges``). Default
            ``'equal_width'`` reproduces the original behavior exactly.
            ``'shell'`` is for gridded/pixel-lattice data: it places bin
            edges at density valleys of the real pairwise-separation
            distribution near r_min, so edges fall between discrete
            lattice-separation shells instead of randomly bisecting one --
            fixes bin-to-bin VSF/error swings caused purely by bin-edge
            placement on regularly-gridded (e.g. IFU fiber lattice) data.
        shell_transition_factor, shell_fine_per_bin, shell_smooth_sigma,
        shell_prominence_frac : float
            Tuning knobs for ``binning='shell'``; see ``_shell_binning_edges``.
        min_pairs_per_bin, max_pairs_for_binning, binning_seed :
            Passed through to ``spatial_bin_edges`` for
            ``binning='adaptive_min_count'``/``'equal_count'``/``'shell'``.

        Returns
        -------
        vsf, vsf_err, log_centers, counts : arrays
            VSF data
        """
        if binning == 'equal_width':
            data = {'x': self.x_physical, 'y': self.y_physical}
            log_centers = self.generate_bins_2d(
                data,
                bin_resolution,
                min_separation=min_separation,
            )
            edges = self.generate_bins_edges(log_centers)
            bin_edges = 10**edges
        else:
            log_centers, bin_edges, _, _ = spatial_bin_edges(
                self.x_physical, self.y_physical,
                bin_resolution=bin_resolution,
                binning=binning,
                min_separation=min_separation,
                min_pairs_per_bin=min_pairs_per_bin,
                max_pairs_for_sampling=max_pairs_for_binning,
                seed=binning_seed,
                shell_transition_factor=shell_transition_factor,
                shell_fine_per_bin=shell_fine_per_bin,
                shell_smooth_sigma=shell_smooth_sigma,
                shell_prominence_frac=shell_prominence_frac,
            )

        nbins = len(bin_edges) - 1
        
        vsf = np.zeros(nbins)
        vsf_sq = np.zeros(nbins)
        counts = np.zeros(nbins)
        vsf_properr = np.zeros(nbins)
        
        seeds = np.arange(n_jobs)
        
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(self.compute_vsf_chunk_2d)(
                self.x_physical, self.y_physical,
                self.vel, self.vel_err,
                bin_edges, seed, order,
                min_separation
            ) for seed in seeds
        )
        
        # --- merge ---
        for vsf_c, vsf_sq_c, err_c, cnt_c in results:
            vsf += vsf_c
            vsf_sq += vsf_sq_c
            vsf_properr += err_c
            counts += cnt_c
        
        # --- VSF ---
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = vsf / counts
            mean_sq = vsf_sq / counts
            var = np.maximum(mean_sq - mean**2, 0.0)

            std = np.sqrt(var)
            sem = std / np.sqrt(counts)
            propagated_mean_err = np.sqrt(vsf_properr) / counts
            bootstrap_err = np.full(nbins, np.nan)
            bootstrap_counts = np.zeros(nbins)
            jackknife_err = np.full(nbins, np.nan)
            jackknife_n_valid = np.zeros(nbins, dtype=int)

            if error_mode == 'std':
                err = np.sqrt(std**2 + propagated_mean_err**2)
            elif error_mode == 'sem':
                err = sem
            elif error_mode == 'propagated':
                err = propagated_mean_err
            elif error_mode == 'combined_sem':
                err = np.sqrt(sem**2 + propagated_mean_err**2)
            elif error_mode == 'bootstrap':
                bootstrap_err, bootstrap_counts = self.compute_vsf_bootstrap_2d(
                    self.x_physical,
                    self.y_physical,
                    self.vel,
                    bin_edges,
                    order,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                    sample_size=bootstrap_sample_size,
                    min_separation=min_separation,
                )
                err = bootstrap_err
            elif error_mode == 'block_jackknife':
                jackknife_err, jackknife_n_valid = self.compute_vsf_block_jackknife_2d(
                    self.x_physical,
                    self.y_physical,
                    self.vel,
                    bin_edges,
                    order,
                    n_blocks_per_side=jackknife_n_blocks_per_side,
                    min_block_points=jackknife_min_block_points,
                    min_valid_blocks=jackknife_min_valid_blocks,
                    min_separation=min_separation,
                    sample_size_per_chunk=jackknife_sample_size_per_chunk,
                    max_n_jobs_per_replicate=jackknife_max_n_jobs_per_replicate,
                    n_jobs=jackknife_n_jobs,
                )
                err = jackknife_err
            else:
                raise ValueError(
                    "error_mode must be one of 'std', 'sem', 'propagated', "
                    "'combined_sem', 'bootstrap', or 'block_jackknife'"
                )
            
            vsf = np.where(counts >= min_pairs, mean, np.nan)
            vsf_err = np.where(counts >= min_pairs, err, np.nan)
            vsf_std = np.where(counts >= min_pairs, std, np.nan)
            vsf_sem = np.where(counts >= min_pairs, sem, np.nan)
            vsf_propagated_mean_err = np.where(counts >= min_pairs, propagated_mean_err, np.nan)
            vsf_bootstrap_err = np.where(counts >= min_pairs, bootstrap_err, np.nan)
            vsf_jackknife_err = np.where(counts >= min_pairs, jackknife_err, np.nan)

        np.savez_compressed(
            outname + f'_order{order}.npz',
            vsf=vsf,
            vsf_err=vsf_err,
            vsf_std=vsf_std,
            vsf_sem=vsf_sem,
            vsf_propagated_mean_err=vsf_propagated_mean_err,
            vsf_bootstrap_err=vsf_bootstrap_err,
            bootstrap_counts=bootstrap_counts,
            n_bootstrap=n_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            vsf_jackknife_err=vsf_jackknife_err,
            jackknife_n_valid_blocks=jackknife_n_valid,
            jackknife_n_blocks_per_side=jackknife_n_blocks_per_side,
            error_mode=error_mode,
            min_separation=min_separation,
            binning=binning,
            bin_edges=bin_edges,
            log_centers=log_centers,
            counts=counts
        )
        
        return vsf, vsf_err, log_centers, counts
    
    def generate_vsf_ratio_2d(self, outname, n_jobs=8, min_pairs=3, 
                             bin_resolution=80):
        """
        Generate and plot the ratio S4 / (S2)^2.
        
        Parameters
        ----------
        outname : str
            Output filename prefix
        n_jobs : int
            Number of parallel jobs
        min_pairs : int
            Minimum pairs per bin
        bin_resolution : int
            Number of bins per decade in log-space separation.
        
        Returns
        -------
        ratio, log_centers, valid_counts : arrays
            Ratio data
        """
        # --- Compute second and fourth order VSFs ---
        vsf2, vsf2err, log_centers, counts2 = self.generate_vsf_2d(
            order=2,
            outname=outname,
            n_jobs=n_jobs,
            min_pairs=min_pairs,
            bin_resolution=bin_resolution
        )
        
        vsf4, vsf4err, _, counts4 = self.generate_vsf_2d(
            order=4,
            outname=outname,
            n_jobs=n_jobs,
            min_pairs=min_pairs,
            bin_resolution=bin_resolution
        )
        
        # --- Compute ratio safely ---
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(
                (vsf2 > 0) & (~np.isnan(vsf2)) & (~np.isnan(vsf4)),
                np.sqrt(vsf4) / vsf2 - 1,
                np.nan
            )
        
        # Mask bins with insufficient statistics
        valid_counts = np.minimum(counts2, counts4)
        ratio = np.where(valid_counts >= min_pairs, ratio, np.nan)
        
        # --- Plot ---
        bin_centers_physical = 10**log_centers
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers_physical, ratio, 'o-', markersize=4, 
                label=r'$S_4 / S_2^2$')
        
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel('Separation Distance (pc)', fontsize=12)
        plt.ylabel(r'$S_4 / S_2^2$', fontsize=12)
        plt.title(f'{outname} VSF Ratio (Intermittency Diagnostic)', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        print(f"Saving ratio plot to {outname}_ratio.png")
        plt.savefig(outname + '_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ratio plot saved to {outname}_ratio.png")
        
        return ratio, log_centers, valid_counts
    
    def spatial_map(self, outname):
        """Create a spatial map of the velocity field."""
        plt.figure(figsize=(8, 6))
        
        vmin, vmax = np.percentile(self.vel, [20, 80])
        sc = plt.scatter(self.x_physical, self.y_physical, c=self.vel, 
                        cmap='RdBu_r', s=8, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, label='Velocity (km/s)')
        plt.xlabel('X (pc)')
        plt.ylabel('Y (pc)')
        plt.title(f'{outname} Velocity Map')
        plt.grid(True, alpha=0.3)
        plt.savefig(outname + '_velocity_map.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def vel_hist(self, outname):
        """Create a histogram of velocity values."""
        plt.figure(figsize=(8, 6))
        plt.hist(self.vel, bins=80, color='gray', edgecolor='black')
        plt.xlabel('Velocity (km/s)', fontsize=12)
        plt.ylabel('Number of Points', fontsize=12)
        plt.title(f'{outname} Velocity Histogram', fontsize=14)
        plt.yscale('log')
        
        plt.grid(True, alpha=0.3)
        
        plt.savefig(outname + '_velocity_histogram.png', dpi=300, 
                   bbox_inches='tight')
        plt.close()


# ============================================================================
# VSF2D SUBWINDOWS CLASS
# ============================================================================

class VSF2DSubwindows(VSF2D):
    """
    VSF2D analyzer with subwindow support.
    
    Divides the spatial domain into subwindows and computes VSFs for each.
    """
    
    def __init__(self, ra, dec, vel, dist):
        """Initialize VSF2DSubwindows."""
        super().__init__(ra, dec, vel, dist)
    
    def generate_subwindows(self, window_size, min_points=50,
                           fill_fraction=0.7, n_sub=5):
        """
        Divide spatial domain into subwindows.
        
        Parameters
        ----------
        window_size : float
            Size of each subwindow
        min_points : int
            Minimum number of points per window
        fill_fraction : float
            Fraction of subcells that must contain data (0-1)
        n_sub : int
            Number of subdivisions per axis
        
        Returns
        -------
        windows : list
            List of (i, j, mask) tuples
        x_edges, y_edges : arrays
            Grid edges
        """
        x = self.x_physical.value if hasattr(self.x_physical, 'value') \
            else self.x_physical
        y = self.y_physical.value if hasattr(self.y_physical, 'value') \
            else self.y_physical
        
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        
        x_edges = np.arange(xmin, xmax + window_size, window_size)
        y_edges = np.arange(ymin, ymax + window_size, window_size)
        
        windows = []
        
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                
                mask = (
                    (x >= x_edges[i]) & (x < x_edges[i+1]) &
                    (y >= y_edges[j]) & (y < y_edges[j+1])
                )
                
                if np.sum(mask) < min_points:
                    continue
                
                # --- check spatial filling ---
                xw = x[mask]
                yw = y[mask]
                
                # create subgrid inside window
                sub_x = np.linspace(x_edges[i], x_edges[i+1], n_sub+1)
                sub_y = np.linspace(y_edges[j], y_edges[j+1], n_sub+1)
                
                occupied = 0
                total_cells = n_sub * n_sub
                
                for ii in range(n_sub):
                    for jj in range(n_sub):
                        
                        submask = (
                            (xw >= sub_x[ii]) & (xw < sub_x[ii+1]) &
                            (yw >= sub_y[jj]) & (yw < sub_y[jj+1])
                        )
                        
                        if np.any(submask):
                            occupied += 1
                
                if occupied / total_cells >= fill_fraction:
                    windows.append((i, j, mask))
        
        return windows, x_edges, y_edges
    
    def compute_subwindow_vsfs(self, window_size, order=2, n_jobs=4,
                              bin_resolution=60, min_separation=None):
        """Compute VSF for each subwindow.

        Parameters
        ----------
        window_size : float
            Physical size of each subwindow.
        order : int
            Order of the structure function.
        n_jobs : int
            Number of parallel jobs.
        bin_resolution : int
            Number of bins per decade in log-space separation.
        """
        windows, x_edges, y_edges = self.generate_subwindows(window_size)
        
        results = []
        
        for i, j, mask in windows:
            
            x = self.x_physical.value if hasattr(self.x_physical, 'value') \
                else self.x_physical
            y = self.y_physical.value if hasattr(self.y_physical, 'value') \
                else self.y_physical
            
            x = x[mask]
            y = y[mask]
            vel = self.vel[mask]
            
            data = {'x': x, 'y': y}
            
            log_centers = self.generate_bins_2d(
                data,
                bin_resolution=bin_resolution,
                min_separation=min_separation,
            )
            edges = self.generate_bins_edges(log_centers)
            bin_edges = 10**edges
            
            vsf = np.zeros(len(bin_edges)-1)
            counts = np.zeros_like(vsf)
            
            seeds = np.arange(n_jobs)
            
            chunks = Parallel(n_jobs=n_jobs)(
                delayed(self.compute_vsf_chunk_2d)(x, y, vel, 
                                                  np.zeros_like(vel),
                                                  bin_edges, 
                                                  seed, order,
                                                  min_separation)
                for seed in seeds
            )
            
            for v, vsq, verr, c in chunks:
                vsf += v
                counts += c
            
            with np.errstate(divide='ignore'):
                vsf = np.where(counts > 0, vsf/counts, np.nan)
            
            results.append((i, j, vsf, log_centers))
        
        return results, x_edges, y_edges
    
    def plot_subwindow_vsfs(self, window_size, outname, order=2,
                           alpha=0.15, show_mean=True, n_common=60):
        """Plot VSFs from all subwindows."""
        results, x_edges, y_edges = self.compute_subwindow_vsfs(window_size, order)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        all_interp_vsfs = []
        
        # --- define common grid (log space) ---
        r_min = np.inf
        r_max = 0
        
        for _, _, _, log_centers in results:
            r = 10**log_centers
            r_min = min(r_min, r.min())
            r_max = max(r_max, r.max())
        
        common_r = np.logspace(np.log10(r_min), np.log10(r_max), n_common)
        
        # --- plot individual curves ---
        for i, j, vsf, log_centers in results:
            
            r = 10**log_centers
            
            ax.plot(r, vsf, '-', color='C0', alpha=alpha, linewidth=1)
            
            # interpolate onto common grid
            interp = np.interp(common_r, r, vsf, left=np.nan, right=np.nan)
            all_interp_vsfs.append(interp)
        
        # --- mean curve ---
        if show_mean and len(all_interp_vsfs) > 0:
            
            all_interp_vsfs = np.array(all_interp_vsfs)
            
            mean_vsf = np.nanmean(all_interp_vsfs, axis=0)
            
            ax.plot(common_r, mean_vsf, color='black', linewidth=2.5,
                   label='Mean VSF')
            
            slope = order / 3.0
            
            # choose a pivot point for normalization (middle of range)
            mid = len(common_r) // 2
            
            ref = common_r**slope
            
            # normalize to match mean VSF at midpoint
            ref *= mean_vsf[mid] / ref[mid]
            
            ax.plot(common_r, ref, '--', color='green', linewidth=2,
                   label=f'r^{order}/3')
            
            ax.legend()
        
        # --- formatting ---
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        ax.set_xlabel("Separation (pc)")
        ax.set_ylabel("VSF")
        ax.set_title("Subwindow VSFs (overlaid)")
        
        plt.tight_layout()
        plt.savefig(outname + "_subwindow_vsfs_overlay.png", dpi=300)
        plt.show()
        plt.close()
    
    def plot_subwindows_fov(self, window_size, shape="square", show_all=True):
        """
        Visualize the full FOV and selected subwindows.
        
        Parameters
        ----------
        window_size : float
            Size of each subwindow
        shape : str
            "square" or "circle"
        show_all : bool
            If True, draw all grid cells faintly
        """
        x = self.x_physical.value if hasattr(self.x_physical, 'value') \
            else self.x_physical
        y = self.y_physical.value if hasattr(self.y_physical, 'value') \
            else self.y_physical
        
        windows, x_edges, y_edges = self.generate_subwindows(window_size)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # --- plot all points ---
        ax.scatter(x, y, s=2, alpha=0.3, color='black')
        
        # --- optionally draw full grid ---
        if show_all and shape == "square":
            for i in range(len(x_edges)-1):
                for j in range(len(y_edges)-1):
                    rect = Rectangle(
                        (x_edges[i], y_edges[j]),
                        window_size,
                        window_size,
                        linewidth=0.5,
                        edgecolor='gray',
                        facecolor='none',
                        alpha=0.3
                    )
                    ax.add_patch(rect)
        
        # --- highlight selected windows ---
        for i, j, mask in windows:
            
            if shape == "square":
                rect = Rectangle(
                    (x_edges[i], y_edges[j]),
                    window_size,
                    window_size,
                    linewidth=1.5,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
            
            elif shape == "circle":
                cx = 0.5 * (x_edges[i] + x_edges[i+1])
                cy = 0.5 * (y_edges[j] + y_edges[j+1])
                
                circ = Circle(
                    (cx, cy),
                    radius=window_size / 2,
                    linewidth=1.5,
                    edgecolor='blue',
                    facecolor='none'
                )
                ax.add_patch(circ)
        
        ax.set_xlabel("x (pc)")
        ax.set_ylabel("y (pc)")
        ax.set_title("FOV with Subwindows")
        
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
