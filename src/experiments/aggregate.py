"""Aggregation of raw result rows into paper-format summary tables.

For each scenario the per-pair deltas (adapter vs frozen black box) are
summarized by mean, relative improvement, Wilcoxon signed-rank p-values
with Holm-Bonferroni correction across the five metrics, and Cliff's delta
effect sizes. RQ3 regroups the same rows by architecture paradigm; RQ4
averages the 10 seeds per target before the paired test (n = 9); RQ5
tabulates the one-factor sweeps.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

from src.eval.metrics import METRIC_NAMES
from src.eval.stats import cliffs_delta, cliffs_magnitude, holm_bonferroni, wilcoxon_p
from src.experiments.pools import ARCH_GROUPS, RQ5_SWEEPS, SCENARIOS
from src.experiments.runner import results_path

TABLES_DIR = os.path.join("results", "tables")


def load_rows(scenario: str) -> list[dict]:
    path = results_path(scenario)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for m in METRIC_NAMES:
                row[f"base_{m}"] = float(row[f"base_{m}"])
                row[f"adap_{m}"] = float(row[f"adap_{m}"])
            rows.append(row)
    return rows


def rel_improvement(base: float, adap: float) -> float:
    if abs(base) < 1e-9:
        return 0.0 if abs(adap) < 1e-9 else float("inf")
    return (adap - base) / abs(base) * 100.0


def _metric_block(base: np.ndarray, adap: np.ndarray) -> dict:
    return {
        "base_mean": float(np.mean(base)),
        "base_std": float(np.std(base)),
        "adap_mean": float(np.mean(adap)),
        "adap_std": float(np.std(adap)),
        "rel": rel_improvement(float(np.mean(base)), float(np.mean(adap))),
        "n": len(base),
    }


def summarize_pairs(rows: list[dict]) -> dict:
    """Paired adapter-vs-blackbox statistics over per-pair rows."""
    out = {}
    pvals = []
    for m in METRIC_NAMES:
        base = np.asarray([r[f"base_{m}"] for r in rows])
        adap = np.asarray([r[f"adap_{m}"] for r in rows])
        blk = _metric_block(base, adap)
        blk["p"] = wilcoxon_p(adap, base)
        blk["delta"] = cliffs_delta(adap, base)
        pvals.append(blk["p"])
        out[m] = blk
    adj = holm_bonferroni(pvals)
    for i, m in enumerate(METRIC_NAMES):
        out[m]["p_adj"] = float(adj[i])
        out[m]["mag"] = cliffs_magnitude(out[m]["delta"])
    return out


def summarize_scenario(scenario: str) -> dict:
    return summarize_pairs(load_rows(scenario))


def summarize_rq3() -> dict:
    rows = []
    for scenario in SCENARIOS:
        rows.extend(load_rows(scenario))
    out = {}
    for group, models in ARCH_GROUPS.items():
        grows = [r for r in rows if r["model"] in models]
        out[group] = {
            m: _metric_block(
                np.asarray([r[f"base_{m}"] for r in grows]),
                np.asarray([r[f"adap_{m}"] for r in grows]),
            )
            for m in METRIC_NAMES
        }
    return out


def summarize_rq4() -> dict:
    rows = load_rows("rq4")
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    variant_stats = {}
    for variant, vrows in by_variant.items():
        variant_stats[variant] = {
            m: {
                "mean": float(np.mean([r[f"adap_{m}"] for r in vrows])),
                "std": float(np.std([r[f"adap_{m}"] for r in vrows])),
            }
            for m in METRIC_NAMES
        }

    # paired test on per-target means over seeds (BlackBox is seed-free)
    def per_target(variant: str) -> dict[str, dict[str, float]]:
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in by_variant.get(variant, []):
            for m in METRIC_NAMES:
                acc[r["project"]][m].append(r[f"adap_{m}"])
        return {p: {m: float(np.mean(v[m])) for m in METRIC_NAMES} for p, v in acc.items()}

    full, black = per_target("Full"), per_target("BlackBox")
    projects = sorted(set(full) & set(black))
    paired = {}
    pvals = []
    for m in METRIC_NAMES:
        f = np.asarray([full[p][m] for p in projects])
        b = np.asarray([black[p][m] for p in projects])
        p = wilcoxon_p(f, b)
        paired[m] = {"p": p, "delta": cliffs_delta(f, b), "n": len(projects)}
        pvals.append(p)
    for i, m in enumerate(METRIC_NAMES):
        paired[m]["p_adj"] = float(holm_bonferroni(pvals)[i])
        paired[m]["mag"] = cliffs_magnitude(paired[m]["delta"])
    return {"variants": variant_stats, "full_vs_blackbox": paired}


def summarize_rq5() -> dict:
    rows = load_rows("rq5")
    out: dict[str, list[dict]] = {}
    for param, values in RQ5_SWEEPS.items():
        entries = []
        for v in values:
            vrows = [r for r in rows if r["variant"] == f"{param}={v}"]
            if not vrows:
                continue
            entries.append(
                {
                    "value": v,
                    "n": len(vrows),
                    **{
                        m: {
                            "mean": float(np.mean([r[f"adap_{m}"] for r in vrows])),
                            "std": float(np.std([r[f"adap_{m}"] for r in vrows])),
                        }
                        for m in METRIC_NAMES
                    },
                }
            )
        out[param] = entries
    return out


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def write_main_tables() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    lines = ["# RQ1/RQ2 aggregate results (adapter vs frozen black box)\n"]
    for scenario in SCENARIOS:
        rows = load_rows(scenario)
        if not rows:
            lines.append(f"\n## {scenario}\n\n(no rows yet)\n")
            continue
        stats = summarize_pairs(rows)
        lines.append(f"\n## {scenario}  (n = {len(rows)} pairs)\n")
        lines.append("| Metric | Baseline | +Adapter | Impr. | Adj. p | Cliff's d |")
        lines.append("|---|---|---|---|---|---|")
        for m in METRIC_NAMES:
            s = stats[m]
            lines.append(
                f"| {m} | {s['base_mean']:.3f} +/- {s['base_std']:.3f} "
                f"| {s['adap_mean']:.3f} +/- {s['adap_std']:.3f} "
                f"| {s['rel']:+.1f}%{_stars(s['p_adj'])} | {s['p_adj']:.3f} "
                f"| {s['delta']:+.3f} ({s['mag'][0].upper()}) |"
            )
    with open(os.path.join(TABLES_DIR, "rq1_rq2_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_rq3_table() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    stats = summarize_rq3()
    lines = ["# RQ3 architecture-group aggregation\n"]
    for group, gstats in stats.items():
        n = gstats["AUC"]["n"]
        lines.append(f"\n## {group} (n = {n})\n")
        lines.append("| Metric | Baseline | +Adapter | Rel. impr. |")
        lines.append("|---|---|---|---|")
        for m in METRIC_NAMES:
            s = gstats[m]
            lines.append(f"| {m} | {s['base_mean']:.3f} | {s['adap_mean']:.3f} | {s['rel']:+.1f}% |")
    with open(os.path.join(TABLES_DIR, "rq3_architecture.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_rq4_table() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    stats = summarize_rq4()
    order = ["BlackBox", "CGAD-only", "EGRW-only", "Full"]
    lines = ["# RQ4 component ablation (SupCon_DP, PROMISE, 9 targets x 10 seeds)\n"]
    lines.append("| Metric | BlackBox | CGAD-only | EGRW-only | Full | Full vs BB p (adj) | Cliff's d |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in METRIC_NAMES:
        cells = []
        for v in order:
            vs = stats["variants"].get(v, {}).get(m)
            cells.append(f"{vs['mean']:.3f}" if vs else "-")
        pv = stats["full_vs_blackbox"][m]
        lines.append(
            f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} "
            f"| {pv['p_adj']:.3f}{_stars(pv['p_adj'])} | {pv['delta']:+.3f} ({pv['mag'][0].upper()}) |"
        )
    with open(os.path.join(TABLES_DIR, "rq4_ablation.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_rq5_tables() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    stats = summarize_rq5()
    defaults = {"gamma_ema": 0.6, "theta": 1.2, "epsilon": 0.05, "tau0": 0.6}
    for param, entries in stats.items():
        lines = [f"# RQ5 sweep: {param} (default {defaults[param]})\n"]
        header = "| value | n |" + "".join(f" {m} |" for m in METRIC_NAMES)
        lines.append(header)
        lines.append("|---|---|---|" + "---|---|" * (len(METRIC_NAMES) - 1))
        for e in entries:
            mark = " (default)" if e["value"] == defaults[param] else ""
            cells = "".join(f" {e[m]['mean']:.3f} +/- {e[m]['std']:.3f} |" for m in METRIC_NAMES)
            lines.append(f"| {e['value']}{mark} | {e['n']} |{cells}")
        with open(os.path.join(TABLES_DIR, f"rq5_{param}.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def write_all() -> None:
    write_main_tables()
    write_rq3_table()
    write_rq4_table()
    write_rq5_tables()
