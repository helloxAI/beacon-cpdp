"""Command-line entry point for the experimental matrix.

Examples
--------
python run_experiments.py --scenario rq1.1 --jobs 8
python run_experiments.py --scenario all --jobs 8
python run_experiments.py --scenario rq4 --jobs 8
python run_experiments.py --scenario smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blackbox.api import MODEL_NAMES
from src.experiments.pools import SCENARIOS
from src.experiments import runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS) + ["rq4", "rq5", "main", "all", "smoke"],
        help="rq1.1/rq1.2/rq2.1/rq2.2/rq2.3, rq4, rq5; 'main' = the five RQ1/RQ2 scenarios; 'all' = main + rq4 + rq5",
    )
    parser.add_argument("--models", default="all", help="comma-separated subset of the 18 backbones")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--data-root", default="Datasets")
    args = parser.parse_args()

    models = MODEL_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",")]

    if args.scenario == "smoke":
        runner.run_scenario("rq1.1", ["TNB"], n_jobs=1, data_root=args.data_root)
        return
    if args.scenario in SCENARIOS:
        runner.run_scenario(args.scenario, models, n_jobs=args.jobs, data_root=args.data_root)
        return

    if args.scenario in ("main", "all"):
        for name in SCENARIOS:
            runner.run_scenario(name, models, n_jobs=args.jobs, data_root=args.data_root)
    if args.scenario in ("rq4", "all"):
        runner.run_rq4(n_jobs=args.jobs, data_root=args.data_root)
    if args.scenario in ("rq5", "all"):
        runner.run_rq5(n_jobs=args.jobs, data_root=args.data_root)


if __name__ == "__main__":
    main()
