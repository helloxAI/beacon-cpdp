"""ARFF dataset loading for the four public defect-prediction suites.

Suites (paper Section 4.3 / Table: 27 projects):
  - AEEEM   (5 Java projects, 61 features, label `class`        in {clean, buggy})
  - JIRA    (7 Java projects, 65 features, label `RealBugCount` numeric)
  - NASA    (5 C projects,    37 features, label `Defective`    in {Y, N})
  - PROMISE (10 Java projects, 20 features, label `defects`     numeric)

Numeric labels are binarized as `> 0 -> 1 (defective)`; nominal labels map
{Y, buggy, true, yes} -> 1, everything else -> 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES: dict[str, dict] = {
    "PROMISE": {
        "subdir": "PROMISE",
        "label": "defects",
        "projects": [
            "ant-1.7", "camel-1.4", "ivy-2.0", "jedit-4.0", "log4j-1.0",
            "poi-2.0", "tomcat", "velocity-1.6", "xalan-2.4", "xerces-1.3",
        ],
    },
    "NASA": {
        "subdir": "NASA",
        "label": "Defective",
        "projects": ["CM1", "MW1", "PC1", "PC3", "PC4"],
    },
    "AEEEM": {
        "subdir": "AEEEM",
        "label": "class",
        "projects": ["EQ", "JDT", "Lucene", "Mylyn", "PDE"],
    },
    "JIRA": {
        "subdir": "JIRA",
        "label": "RealBugCount",
        "projects": [
            "activemq-5.0.0", "derby-10.5.1.1", "groovy-1_6_BETA_1",
            "hbase-0.94.0", "hive-0.9.0", "jruby-1.1", "wicket-1.3.0-beta2",
        ],
    },
}

_POSITIVE_TOKENS = {"y", "yes", "true", "buggy", "defective", "1"}


@dataclass
class ProjectData:
    """One project (dataset) from a suite."""

    suite: str
    project: str
    feature_names: list[str]
    X: np.ndarray  # (n_samples, n_features) float64
    y: np.ndarray  # (n_samples,) int64 in {0, 1}

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def defect_rate(self) -> float:
        return float(self.y.mean())


def _parse_arff(path: str) -> tuple[list[str], list[str], list[list[str]]]:
    """Minimal ARFF parser sufficient for the benchmark files.

    Returns (attribute_names, attribute_kinds, rows) where kind is
    'numeric' or 'nominal'. The dense @data section is comma separated.
    """
    names: list[str] = []
    kinds: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@data"):
                    in_data = True
                elif low.startswith("@attribute"):
                    rest = line.split(None, 1)[1].strip()
                    # attribute name may be quoted
                    if rest[0] in "'\"":
                        quote = rest[0]
                        end = rest.index(quote, 1)
                        name = rest[1:end]
                        kind = rest[end + 1:].strip()
                    else:
                        parts = rest.split(None, 1)
                        name = parts[0]
                        kind = parts[1] if len(parts) > 1 else "numeric"
                    names.append(name)
                    kinds.append(
                        "nominal" if kind.lstrip().startswith("{") else "numeric"
                    )
            else:
                rows.append([cell.strip() for cell in line.split(",")])
    return names, kinds, rows


def _binarize(values: list[str], kind: str) -> np.ndarray:
    if kind == "numeric":
        return (np.asarray([float(v) for v in values]) > 0).astype(np.int64)
    return np.asarray(
        [1 if v.strip().strip("'\"").lower() in _POSITIVE_TOKENS else 0 for v in values],
        dtype=np.int64,
    )


def load_project(
    suite: str, project: str, data_root: str = "Datasets"
) -> ProjectData:
    """Load one project as (X, y) with a binarized defect label."""
    suite = suite.upper()
    info = SUITES[suite]
    path = os.path.join(data_root, info["subdir"], f"{project}.arff")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset file not found: {path}")

    names, kinds, rows = _parse_arff(path)
    label_attr = info["label"]
    if names[-1] != label_attr:
        raise ValueError(
            f"{path}: expected last attribute '{label_attr}', got '{names[-1]}'"
        )

    feature_names = names[:-1]
    n_features = len(feature_names)
    X = np.empty((len(rows), n_features), dtype=np.float64)
    label_col: list[str] = []
    for i, row in enumerate(rows):
        if len(row) != len(names):
            raise ValueError(f"{path}: row {i} has {len(row)} fields, expected {len(names)}")
        for j in range(n_features):
            v = row[j]
            X[i, j] = float("nan") if v in ("?", "") else float(v)
        label_col.append(row[-1])

    # Median imputation for the (rare) missing numeric entries.
    if np.isnan(X).any():
        col_median = np.nanmedian(X, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_median, idx[1])

    y = _binarize(label_col, kinds[-1])
    return ProjectData(suite=suite, project=project, feature_names=feature_names, X=X, y=y)


def suite_of_project(project: str) -> str:
    for suite, info in SUITES.items():
        if project in info["projects"]:
            return suite
    raise KeyError(f"unknown project: {project}")


def list_projects(suite: str) -> list[str]:
    return list(SUITES[suite.upper()]["projects"])
