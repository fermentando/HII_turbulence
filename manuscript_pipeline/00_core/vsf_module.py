"""
VSF2D Module - 2D Velocity Structure Function Analysis
======================================================
Contains the VSF2D class for computing 2D velocity structure functions
and related statistics (PDF, C2, C4, intermittency).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Rectangle, Circle


# ============================================================================
# UTILITY FUNCTIONS FOR INDEPENDENT COMPUTATION
# ============================================================================

def compute_bins(x_data, y_data, vel_data, bin_resolution=80):
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
        Number of bins per decade in log-space separation.
    
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
    # Compute spatial bins
    data = {'x': x_data, 'y': y_data}
    coords = np.column_stack((data['x'], data['y']))
    
    tree = cKDTree(coords)
    sample_size = min(1000000, len(coords))
    sample_idx = np.random.choice(len(coords), sample_size, replace=False)
    dists, _ = tree.query(coords[sample_idx], k=2)
    r_min = np.median(dists[:, 1])
    
    dx = data['x'].max() - data['x'].min()
    dy = data['y'].max() - data['y'].min()
    r_max = np.sqrt(dx**2 + dy**2)
    
    # Determine bin length
    if r_min <= 0 or r_min >= r_max:
        r_min = r_max / 1000
    log_range = np.log10(r_max / r_min)
    length = max(int(log_range * bin_resolution), 10)
    
    log_centers = np.linspace(np.log10(r_min), np.log10(r_max), length)
    
    # Compute bin edges
    bw = np.diff(log_centers) / 2
    edges = np.empty(len(log_centers) + 1)
    edges[1:-1] = log_centers[:-1] + bw
    edges[0] = log_centers[0] - bw[0]
    edges[-1] = log_centers[-1] + bw[-1]
    bin_edges = 10**edges
    
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
    
    def generate_bins_2d(self, data, bin_resolution=80, length=None):
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
        
        if length is None:
            length = self.determine_length(r_min, r_max, bin_resolution)
        
        if r_min <= 0 or r_min >= r_max:
            r_min = r_max / 1000
        
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
                            seed, order):
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

    def generate_vsf_2d(self, order, outname, n_jobs=8, min_pairs=3, 
                       bin_resolution=80, error_mode='std', n_bootstrap=100,
                       bootstrap_seed=12345):
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
        error_mode : {'std', 'sem', 'propagated', 'combined_sem', 'bootstrap'}
            Error returned as ``vsf_err``. ``std`` is the default and uses the
            standard deviation of pair values around the mean, combined in
            quadrature with the propagated mean velocity-error term.
        n_bootstrap : int
            Number of spaxel bootstrap resamples when ``error_mode='bootstrap'``.
        bootstrap_seed : int
            Random seed for bootstrap resampling.
        
        Returns
        -------
        vsf, vsf_err, log_centers, counts : arrays
            VSF data
        """
        data = {'x': self.x_physical, 'y': self.y_physical}
        
        log_centers = self.generate_bins_2d(data, bin_resolution)
        edges = self.generate_bins_edges(log_centers)
        bin_edges = 10**edges
        
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
                bin_edges, seed, order
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
                )
                err = bootstrap_err
            else:
                raise ValueError(
                    "error_mode must be one of 'std', 'sem', 'propagated', "
                    "'combined_sem', or 'bootstrap'"
                )
            
            vsf = np.where(counts >= min_pairs, mean, np.nan)
            vsf_err = np.where(counts >= min_pairs, err, np.nan)
            vsf_std = np.where(counts >= min_pairs, std, np.nan)
            vsf_sem = np.where(counts >= min_pairs, sem, np.nan)
            vsf_propagated_mean_err = np.where(counts >= min_pairs, propagated_mean_err, np.nan)
            vsf_bootstrap_err = np.where(counts >= min_pairs, bootstrap_err, np.nan)
        
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
            error_mode=error_mode,
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
                              bin_resolution=60):
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
            
            log_centers = self.generate_bins_2d(data, bin_resolution=bin_resolution)
            edges = self.generate_bins_edges(log_centers)
            bin_edges = 10**edges
            
            vsf = np.zeros(len(bin_edges)-1)
            counts = np.zeros_like(vsf)
            
            seeds = np.arange(n_jobs)
            
            chunks = Parallel(n_jobs=n_jobs)(
                delayed(self.compute_vsf_chunk_2d)(x, y, vel, 
                                                  np.zeros_like(vel),
                                                  bin_edges, 
                                                  seed, order)
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
