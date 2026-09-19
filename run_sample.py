"""Sampled end-to-end run: a small representative slice of the full matrix.

Covers all five RQ1/RQ2 scenarios, all three architecture groups, and all
four dataset suites, plus small RQ4/RQ5 slices. Sampled rows use the same
keys as the full matrix, so a later full run reuses them automatically.

Usage: python run_sample.py [--jobs N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import cache as cm
from src.experiments import runner

MODELS = ["TNB", "CPDP_IFS", "MLP_Mean", "SupCon_DP"]  # Classic x2, Shallow, Deep

MAIN_SAMPLE = {
    # scenario: (pair, [(suite, project), ...])
    "rq1.1": ("rq1", [("PROMISE", "velocity-1.6"), ("PROMISE", "camel-1.4")]),
    "rq1.2": ("rq1", [("JIRA", "jruby-1.1"), ("NASA", "PC1"), ("AEEEM", "EQ")]),
    "rq2.1": ("rq2", [("JIRA", "jruby-1.1"), ("JIRA", "hive-0.9.0")]),
    "rq2.2": ("rq2", [("PROMISE", "camel-1.4"), ("PROMISE", "jedit-4.0")]),
    "rq2.3": ("rq2", [("NASA", "PC1"), ("AEEEM", "Lucene")]),
}

RQ4_SAMPLE = {  # variant -> (seeds, targets)
    "BlackBox": ([-1], [("PROMISE", "ivy-2.0"), ("PROMISE", "camel-1.4")]),
    "CGAD-only": ([42, 43], [("PROMISE", "ivy-2.0"), ("PROMISE", "camel-1.4")]),
    "EGRW-only": ([42, 43], [("PROMISE", "ivy-2.0"), ("PROMISE", "camel-1.4")]),
    "Full": ([42, 43], [("PROMISE", "ivy-2.0"), ("PROMISE", "camel-1.4")]),
}
RQ4_CFG = {"CGAD-only": {"epsilon": 1.0}, "EGRW-only": {"kd_mode": "soft"}, "Full": {}}

RQ5_SAMPLE = {  # (param, value) overrides on two targets
    "epsilon=0.0": {"epsilon": 0.0},
    "tau0=0.9": {"tau0": 0.9},
}
RQ5_TARGETS = [("PROMISE", "ivy-2.0"), ("PROMISE", "camel-1.4")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    t_start = time.perf_counter()

    # ---- warm caches ------------------------------------------------------
    print("warming caches ...")
    needed = set()
    for pair, targets in MAIN_SAMPLE.values():
        for suite, proj in targets:
            needed.add((suite, proj))
    needed.add(("PROMISE", "jedit-4.0"))
    for suite, proj in needed:
        cm.ensure_features(suite, proj)
    for pair, targets in MAIN_SAMPLE.values():
        for suite, proj in targets:
            for model in MODELS:
                cm.ensure_pblack(pair, model, suite, proj)

    # ---- main scenarios ----------------------------------------------------
    for scenario, (pair, targets) in MAIN_SAMPLE.items():
        tasks = [
            {
                "scenario": scenario,
                "pair": pair,
                "model": model,
                "suite": suite,
                "project": proj,
                "variant": "Full",
                "seed": 42,
                "config": {},
            }
            for model in MODELS
            for suite, proj in targets
        ]
        runner.run_tasks(scenario, tasks, args.jobs)

    # ---- rq4 slice ----------------------------------------------------------
    rq4_tasks = []
    for variant, (seeds, targets) in RQ4_SAMPLE.items():
        for seed in seeds:
            for suite, proj in targets:
                rq4_tasks.append(
                    {
                        "scenario": "rq4",
                        "pair": "rq1",
                        "model": "SupCon_DP",
                        "suite": suite,
                        "project": proj,
                        "variant": variant,
                        "seed": seed,
                        "config": dict(RQ4_CFG.get(variant) or {}),
                    }
                )
    runner.run_tasks("rq4", rq4_tasks, args.jobs)

    # ---- rq5 slice ----------------------------------------------------------
    rq5_tasks = [
        {
            "scenario": "rq5",
            "pair": "rq1",
            "model": "SupCon_DP",
            "suite": suite,
            "project": proj,
            "variant": variant,
            "seed": 42,
            "config": dict(cfg),
        }
        for variant, cfg in RQ5_SAMPLE.items()
        for suite, proj in RQ5_TARGETS
    ]
    runner.run_tasks("rq5", rq5_tasks, args.jobs)

    print(f"sample run finished in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
