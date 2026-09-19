"""Experiment runner: one adaptation run, scenario orchestration, resumable CSV output.

Every run consumes the two disk caches (target features, frozen black-box
outputs), trains a fresh adapter, and evaluates both the frozen black-box
probabilities and the adapter probabilities on the same held-out 20% of the
target project. Result rows are appended to one CSV per scenario; rows
already present are skipped, so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import asdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.beacon.trainer import BeaconConfig, BeaconTrainer
from src.eval.metrics import METRIC_NAMES, evaluate
from src.experiments import cache as cache_mod
from src.experiments.pools import SCENARIOS

RESULTS_DIR = os.path.join("results", "raw")

ROW_FIELDS = (
    ["scenario", "pair", "model", "suite", "project", "variant", "seed"]
    + [f"base_{m}" for m in METRIC_NAMES]
    + [f"adap_{m}" for m in METRIC_NAMES]
    + ["seconds"]
)


def results_path(scenario: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{scenario}.csv")


def _row_key(row: dict) -> tuple:
    return (row["model"], row["suite"], row["project"], row["variant"], row["seed"])


def load_done_keys(scenario: str) -> set:
    path = results_path(scenario)
    done = set()
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(_row_key(row))
    return done


def append_rows(scenario: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = results_path(scenario)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ROW_FIELDS})


def run_task(task: dict) -> dict:
    """One adaptation run. Executed in worker processes."""
    import torch

    torch.set_num_threads(1)

    scenario = task["scenario"]
    pair = task["pair"]
    model = task["model"]
    suite, project = task["suite"], task["project"]
    variant = task["variant"]
    seed = int(task["seed"])

    X_all, y_all, test_idx, _ = cache_mod.load_features(suite, project)
    p_black = cache_mod.load_pblack(pair, model, suite, project)
    y_test = y_all[test_idx]

    row = {
        "scenario": scenario,
        "pair": pair,
        "model": model,
        "suite": suite,
        "project": project,
        "variant": variant,
        "seed": seed,
    }
    base = evaluate(y_test, p_black[test_idx])
    for m in METRIC_NAMES:
        row[f"base_{m}"] = f"{base[m]:.6f}"

    if variant == "BlackBox":
        for m in METRIC_NAMES:
            row[f"adap_{m}"] = row[f"base_{m}"]
        row["seconds"] = "0.0"
        return row

    cfg_dict = dict(task.get("config") or {})
    cfg_dict["seed"] = seed
    cfg = BeaconConfig(**cfg_dict)

    t0 = time.perf_counter()
    trainer = BeaconTrainer(cfg).fit(X_all, p_black)
    p_adap = trainer.predict_proba(X_all)
    elapsed = time.perf_counter() - t0

    adap = evaluate(y_test, p_adap[test_idx])
    for m in METRIC_NAMES:
        row[f"adap_{m}"] = f"{adap[m]:.6f}"
    row["seconds"] = f"{elapsed:.2f}"
    return row


def build_tasks(scenario: str, models: list[str]) -> list[dict]:
    spec = SCENARIOS[scenario]
    pair = spec["pair"]
    tasks = []
    for model in models:
        for suite, project in spec["targets"]:
            tasks.append(
                {
                    "scenario": scenario,
                    "pair": pair,
                    "model": model,
                    "suite": suite,
                    "project": project,
                    "variant": "Full",
                    "seed": 42,
                    "config": {},
                }
            )
    return tasks


def build_rq4_tasks() -> tuple[str, list[dict]]:
    from src.experiments.pools import RQ4_MODEL, RQ4_SEEDS, RQ4_TARGETS, RQ4_VARIANTS

    tasks = []
    for variant, cfg in RQ4_VARIANTS.items():
        for suite, project in RQ4_TARGETS:
            if variant == "BlackBox":
                tasks.append(
                    {
                        "scenario": "rq4",
                        "pair": "rq1",
                        "model": RQ4_MODEL,
                        "suite": suite,
                        "project": project,
                        "variant": variant,
                        "seed": -1,
                        "config": {},
                    }
                )
            else:
                for seed in RQ4_SEEDS:
                    tasks.append(
                        {
                            "scenario": "rq4",
                            "pair": "rq1",
                            "model": RQ4_MODEL,
                            "suite": suite,
                            "project": project,
                            "variant": variant,
                            "seed": seed,
                            "config": dict(cfg),
                        }
                    )
    return "rq4", tasks


def build_rq5_tasks() -> tuple[str, list[dict]]:
    from src.experiments.pools import RQ5_MODEL, RQ5_SEED, RQ5_SWEEPS, RQ5_TARGETS

    tasks = []
    for param, values in RQ5_SWEEPS.items():
        for value in values:
            for suite, project in RQ5_TARGETS:
                tasks.append(
                    {
                        "scenario": "rq5",
                        "pair": "rq1",
                        "model": RQ5_MODEL,
                        "suite": suite,
                        "project": project,
                        "variant": f"{param}={value}",
                        "seed": RQ5_SEED,
                        "config": {param: value},
                    }
                )
    return "rq5", tasks


def warm_caches(scenario: str, models: list[str], data_root: str = "Datasets") -> None:
    spec = SCENARIOS[scenario]
    for suite, project in spec["targets"]:
        cache_mod.ensure_features(suite, project, data_root)
    for model in models:
        for suite, project in spec["targets"]:
            cache_mod.ensure_pblack(spec["pair"], model, suite, project, data_root)


def run_tasks(scenario: str, tasks: list[dict], n_jobs: int = 1) -> None:
    done = load_done_keys(scenario)
    todo = [t for t in tasks if (t["model"], t["suite"], t["project"], t["variant"], str(t["seed"])) not in done]
    if not todo:
        print(f"[{scenario}] nothing to do ({len(done)} rows already present)")
        return
    print(f"[{scenario}] running {len(todo)} adaptation tasks ({len(done)} already done)")
    if n_jobs == 1:
        for i, task in enumerate(todo, 1):
            row = run_task(task)
            append_rows(scenario, [row])
            if i % 10 == 0 or i == len(todo):
                print(f"[{scenario}] {i}/{len(todo)}")
    else:
        from joblib import Parallel, delayed

        try:
            it = Parallel(n_jobs=n_jobs, return_as="generator_unordered", verbose=0)(
                delayed(run_task)(t) for t in todo
            )
        except TypeError:  # older joblib without generator support
            it = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_task)(t) for t in todo)
        buf = []
        for i, row in enumerate(it, 1):
            buf.append(row)
            if len(buf) >= 20:
                append_rows(scenario, buf)
                buf = []
            if i % 25 == 0 or i == len(todo):
                print(f"[{scenario}] {i}/{len(todo)}")
        append_rows(scenario, buf)


def run_scenario(scenario: str, models: list[str], n_jobs: int = 1, data_root: str = "Datasets") -> None:
    warm_caches(scenario, models, data_root)
    run_tasks(scenario, build_tasks(scenario, models), n_jobs)


def run_rq4(n_jobs: int = 1, data_root: str = "Datasets") -> None:
    from src.experiments.pools import RQ4_MODEL, RQ4_TARGETS

    for suite, project in RQ4_TARGETS:
        cache_mod.ensure_features(suite, project, data_root)
        cache_mod.ensure_pblack("rq1", RQ4_MODEL, suite, project, data_root)
    scenario, tasks = build_rq4_tasks()
    run_tasks(scenario, tasks, n_jobs)


def run_rq5(n_jobs: int = 1, data_root: str = "Datasets") -> None:
    from src.experiments.pools import RQ5_MODEL, RQ5_TARGETS

    for suite, project in RQ5_TARGETS:
        cache_mod.ensure_features(suite, project, data_root)
        cache_mod.ensure_pblack("rq1", RQ5_MODEL, suite, project, data_root)
    scenario, tasks = build_rq5_tasks()
    run_tasks(scenario, tasks, n_jobs)
