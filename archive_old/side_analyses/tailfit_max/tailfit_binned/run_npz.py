from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Fit binned tail models for counts__* arrays in an NPZ.")
    p.add_argument("input_npz", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--nbins", type=int, default=50)
    p.add_argument(
        "--xmin-mode",
        choices=["map", "smallest_good", "ks_min"],
        default="smallest_good",
        help=(
            "xmin selection mode. smallest_good: smallest xmin with PL bootstrap p>=threshold; "
            "map: xmin with max posterior weight; "
            "ks_min: Clauset/powerlaw KS-min mode on tail CDFs (ignores priors and p-values)."
        ),
    )
    p.add_argument("--xmin-softcap", type=float, default=200.0)
    p.add_argument("--xmin-gamma", type=float, default=2.0)
    p.add_argument("--xmin-max", type=float, default=None)
    p.add_argument("--xmin-pthresh", type=float, default=0.1)
    p.add_argument("--xmin-candidates", type=int, default=50)
    p.add_argument("--xmin-min-tail-count", type=int, default=200)
    p.add_argument("--xmin-lambda-discard", type=float, default=0.5)
    p.add_argument("--xmin-bootstrap-sims", type=int, default=40)
    p.add_argument(
        "--propagate-xmin-uncertainty",
        choices=["on", "off"],
        default="on",
        help="Propagate xmin uncertainty into BPL/LN parameter intervals (default: on).",
    )
    p.add_argument("--bpl-xb-prior-min-ratio", type=float, default=1.0)
    p.add_argument("--bpl-xb-prior-strength", type=float, default=0.0)
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next snapshot if one dataset key fails (default: off).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress fitting progress output.")
    args = p.parse_args()
    run_pipeline(
        args.input_npz,
        args.output_dir,
        nbins=args.nbins,
        xmin_mode=args.xmin_mode,
        xmin_softcap=args.xmin_softcap,
        xmin_gamma=args.xmin_gamma,
        xmin_max=args.xmin_max,
        xmin_pthresh=args.xmin_pthresh,
        xmin_candidates=args.xmin_candidates,
        xmin_min_tail_count=args.xmin_min_tail_count,
        xmin_lambda_discard=args.xmin_lambda_discard,
        xmin_bootstrap_sims=args.xmin_bootstrap_sims,
        propagate_xmin_uncertainty=(args.propagate_xmin_uncertainty == "on"),
        bpl_xb_prior_min_ratio=args.bpl_xb_prior_min_ratio,
        bpl_xb_prior_strength=args.bpl_xb_prior_strength,
        continue_on_error=args.continue_on_error,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
