# Why `K(r) > 3` fails to appear once `r_min` is too coarse

Context: the no-pixelization random-point shell pipeline
(`generate_random_point_expanding_shell_grid.py --no-pixelize`) samples 3D
points inside a spherical shell of finite radial depth, projects them onto
the sky plane `(y, z)`, and uses the line-of-sight velocity component to
compute `dK/dlog10(r)` as a function of projected separation `r`. The
diagnostic in `plot_fixed_rmin_comparison.py` showed that for `N=100,000`,
sweeping the bin floor `r_min` from `0.001` up to `0.1` (in units of
`r_HII`) causes the fitted slope to flatten to exactly zero everywhere once
`r_min` reaches `0.1` -- not because the fit degrades gracefully, but
because `K(r)` never climbs above the Gaussian floor (`K=3`) at all once the
grid starts that coarse.

## Not intermittency

The first instinct is to reach for turbulence intermittency: kurtosis
excess concentrated at small scales, decaying toward Gaussian statistics at
larger scales, the way it does for a real cascade. That is the wrong
mechanism here. This is synthetic, projected shell data, and the actual
cause is purely geometric.

## The real mechanism: near/far-side projection contamination

A spherical shell has finite depth along the line of sight. Two points can
have a **tiny projected separation** `r` while sitting on **opposite sides
of the shell in 3D** -- one on the near side, one on the far side, along
the same (or nearly the same) sightline. The generator script's own
docstring already flags this directly:

> "the radial bulk term therefore does not cancel exactly between front and
> back points in each sightline"

A near/far pair like that has a velocity difference set by the shell's
**depth** (roughly the line-of-sight chord length through the shell at that
sky position) -- a scale that is essentially **fixed**, set by the shell's
own geometry, and **not** small just because the pair's *projected*
separation happens to be small.

**Important correction (see "Isolating the source: a `v_bulk=0` test"
below): this is not primarily about the bulk/systematic velocity field.**
It was initially assumed that this fixed offset came from the coherent
radial outflow (`v = v_turb + v_bulk·r̂`), since at sky-center that gives a
clean `Δv_shell(ρ=0) = 2·v_bulk`. A direct test with `v_bulk=0` shows the
effect is reduced but **not eliminated** -- the true driver is more
general: a near-side and far-side point are separated by a large chunk of
the shell's full 3D depth, easily exceeding the turbulent field's own
correlation length. Once two points are farther apart than that
correlation length, their turbulent velocities are close to **independent
draws** rather than correlated the way two genuinely nearby points' are.
That variance mismatch -- independent vs. correlated -- is enough on its
own to produce a heavy-tailed `Δv` mixture and `K>3`, with no bulk flow
required. `v_bulk` amplifies the effect (adds a large, systematic,
same-sign contribution on top of the decorrelation) but is not its root
cause.

Meanwhile, **genuine** nearby pairs (points that are actually close
together in 3D, not just in projection) have a velocity difference that
grows with `r` in the ordinary structure-function way: tiny `Δv` at tiny
`r`, larger `Δv` at larger `r`.

### Why this inflates kurtosis specifically at small `r`

- At very small `r`, genuine pairs have very small `Δv`. The rare near/far
  pairs -- with their large, roughly `r`-independent `Δv` -- stand out as
  extreme outliers against that tight bulk distribution.
- Kurtosis is a 4th-moment statistic, so it is extremely sensitive to
  exactly this kind of rare-but-extreme outlier: a handful of near/far
  pairs, a tiny fraction of the total pair count, can dominate `⟨Δv⁴⟩` and
  push `K` well above 3 even though almost every pair in the bin is a
  normal, small-`Δv` pair.
- This was confirmed directly for `N=100,000` at its natural resolution
  (`r_min ≈ 0.002-0.003`): `K ≈ 4.5-6.2`, well above Gaussian.

### Why it fades as `r` grows

As `r` increases, the genuine pairs' `Δv` keeps growing while the near/far
pairs' `Δv` stays roughly where it was (set by the fixed shell depth, not
by `r`). The contrast between the two populations shrinks. Once the
genuine structure-function `Δv(r)` has grown enough to be comparable to, or
exceed, the near/far-pair `Δv` scale, the near/far pairs no longer look
like extreme outliers next to the bulk -- they've been absorbed into an
already-broader distribution, and the 4th moment is no longer dominated by
them. `K(r)` relaxes back toward (and in this data, even below) 3.

## Why `r_min = 0.1 r_HII` specifically fails

`r_min = 0.1` is not "slightly too coarse" in some abstract binning sense.
It is coarse enough that, by the very first bin, the genuine
structure-function amplitude has *already* caught up to the near/far-pair
outlier amplitude -- the dilution described above has already happened.
Measured directly (`N=100,000`, smallest shell width): `K` at `r=0.1` is
`3.077`, only marginally above the floor, and is **already declining** bin
to bin from there. Essentially no width has 3+ bins clearing `K ≥ 3`
(`MIN_FIT_POINTS = 3` in `fit_k_vs_logr_above_floor`), so the fit falls
back to its zero default almost everywhere -- which is exactly the flat
`r_min = 0.1` curve seen in the comparison plot.

## Isolating the source: a `v_bulk=0` test

The geometric argument above predicts a specific velocity scale: at
sky-center sightlines (`ρ=0`), the near and far shell crossings sit at
`x ≈ ∓R̄` (the shell's mean radius), so with `v = v_turb + v_bulk·r̂`,
`Δv_shell(ρ=0) = 2·v_bulk`. With the production value `v_bulk=10` against
`σ_turb=1`, that is a `20σ` outlier -- and since kurtosis weights by
`Δv⁴`, a single such pair contributes to `⟨Δv⁴⟩` as much as `20⁴=160,000`
ordinary-scale pairs would. The natural prediction from this picture is
that setting `v_bulk=0` should remove the systematic near/far offset
entirely and strongly suppress or eliminate the artifact.

This was tested directly: `generate_random_point_expanding_shell_grid.py
--v-bulk 0.0` regenerated all four `N` datasets
(`synthetic_projected_shell_random_points_nopixel_nobulk_n{N}`), and
`plot_fixed_rmin_comparison_nobulk.py` reran the identical left/middle/
right diagnostic on them
(`random_point_count_dK_dlogr_fixed_rmin_comparison_nobulk.png`).

**Result: the artifact survives almost fully.** The fitted floor at the
smallest widths drops from `≈-2.0` (`v_bulk=10`) to only `≈-1.4` to `-1.5`
(`v_bulk=0`) -- a ~25-30% reduction, not the near-total suppression the
`v_bulk`-only picture predicted. The sigmoid shape, the `r_min`-dependence
of `x0` (if anything steeper with `v_bulk=0`: slope `≈0.75` vs `≈0.42`
with `v_bulk=10`), and the all-fallback flat line at `r_min/r_HII=0.1` are
all still present.

This directly falsifies "bulk flow is the source" and supports the
decorrelation picture above instead: the dominant ingredient is that
near/far pairs are **3D-decorrelated regardless of bulk motion** (they are
simply too far apart in 3D for the turbulent field's correlation length to
connect them), and that decorrelation alone is sufficient to source most
of the effect. `v_bulk` makes it considerably worse (the `20σ` estimate
above is real, just not the only contribution), but removing it only
partially dilutes the artifact rather than eliminating it.

## Connection to the pixelized pipeline

This also explains, after the fact, why the *pixelized* version of this
pipeline averages together every raw point landing in the same sky pixel
before computing anything: that averaging is specifically what merges a
sightline's near-side and far-side contributions into one value, the way a
real telescope beam/fiber does. Skipping that averaging (the no-pixelize
mode) is precisely what leaves this near/far contamination fully exposed in
the raw pair statistics, undiluted, which is why it shows up so clearly in
this diagnostic.
