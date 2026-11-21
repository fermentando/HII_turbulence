"""Grid-free Gaussian Kolmogorov turbulence and its analytic structure function.

Prototype (numpy only). The line-of-sight velocity is an isotropic scalar Gaussian field with 3D power
P(k) ~ k^(-11/3) between k_min and k_max, evaluated directly at the point positions as a sum of
random Fourier modes. There is no grid, so there is no Nyquist limit and no interpolation smoothing,
and k_max can be made effectively infinite (the inner scale far below any shell width).

    g(r) = sum_m sqrt(w_m) [A_m cos(k_m . r) + B_m sin(k_m . r)],   A_m, B_m ~ N(0, 1)

Given the wavevectors, g is exactly Gaussian with <g^2> = sum w_m = 1 and correlation
sum_m w_m cos(k_m . s). Averaged over the random directions and magnitudes of k_m this is the
continuum correlation

    xi(s) = int k^(-5/3) sinc(k s) dk / int k^(-5/3) dk,      D(s) = 2 [1 - xi(s)]

with the closed form D(s) = (2 s^(2/3) / Z) [G(k_max s) - G(k_min s)], Z = 1.5 (k_min^(-2/3) - k_max^(-2/3)),
G(x) = int_0^x t^(-5/3) (1 - sin t / t) dt (series for x < 10, asymptotic expansion above).
For k_max -> inf: D(s) = (4/3) x^(2/3) [I - G(x)], x = k_min s, I = (9/20) Gamma(1/3).
"""
from __future__ import annotations

from math import factorial, gamma

import numpy as np

I_INF = 0.45 * gamma(1.0 / 3.0)  # G(inf) = (9/20) Gamma(1/3) = 1.2055
_SERIES_N = np.arange(1, 45)
_SERIES_C = np.array([(-1.0) ** (n + 1) / (factorial(2 * n + 1) * (2 * n - 2.0 / 3.0)) for n in _SERIES_N])
_SERIES_P = 2.0 * _SERIES_N - 2.0 / 3.0


def G(x):
    """G(x) = int_0^x t^(-5/3) (1 - sin t / t) dt."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < 10.0
    xs = x[small]
    out[small] = (_SERIES_C[None, :] * xs[:, None] ** _SERIES_P[None, :]).sum(1) if xs.size else xs
    xl = x[~small]
    out[~small] = (I_INF - 1.5 * xl ** (-2 / 3) + np.cos(xl) * xl ** (-8 / 3)
                   + (8 / 3) * np.sin(xl) * xl ** (-11 / 3) - (88 / 9) * np.cos(xl) * xl ** (-14 / 3))
    return out


def structure_function(s, kmin, kmax=np.inf):
    """Continuum D(s) for a unit-variance field with P(k) ~ k^(-11/3) on [kmin, kmax]."""
    s = np.asarray(s, float)
    x = kmin * s
    if np.isinf(kmax):
        return (4.0 / 3.0) * x ** (2 / 3) * (I_INF - G(x))
    Z = 1.5 * (kmin ** (-2 / 3) - kmax ** (-2 / 3))
    return 2.0 * s ** (2 / 3) * (G(kmax * s) - G(x)) / Z


class GridFreeKolmogorov:
    """Random-mode realisation of the field; call it on (n, 3) positions."""

    def __init__(self, kmin, kmax, rng, n_shells=300, n_dir=32):
        edges = np.geomspace(kmin, kmax, n_shells + 1)
        # magnitude drawn within each shell from p(k) ~ k^(-5/3) (the shell's share of the variance)
        lo, hi = edges[:-1], edges[1:]
        u = rng.random((n_shells, n_dir))
        a = lo[:, None] ** (-2 / 3) - u * (lo[:, None] ** (-2 / 3) - hi[:, None] ** (-2 / 3))
        k = a ** (-1.5)
        w_shell = (lo ** (-2 / 3) - hi ** (-2 / 3)) / (kmin ** (-2 / 3) - kmax ** (-2 / 3))
        w = np.repeat(w_shell / n_dir, n_dir)
        d = rng.normal(size=(n_shells * n_dir, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        self.kvec = d * k.ravel()[:, None]
        amp = np.sqrt(w)
        self.A = amp * rng.normal(size=w.size)
        self.B = amp * rng.normal(size=w.size)

    def __call__(self, points, chunk=2000):
        out = np.empty(len(points))
        for i in range(0, len(points), chunk):
            ph = points[i:i + chunk] @ self.kvec.T
            out[i:i + chunk] = np.cos(ph) @ self.A + np.sin(ph) @ self.B
        return out
