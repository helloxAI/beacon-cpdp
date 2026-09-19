"""Cross-suite feature correspondences for the relaxed black-box H-CPDP setting.

Paper Section 3.1: for heterogeneous feature spaces (d_s != d_t) a legal query
representation meeting the black box's input-feature specification is built
with available meta-information / metric conversion. Both RQ training sources
are PROMISE projects (20 CK metrics), so the required direction is
{JIRA, NASA, AEEEM} -> PROMISE-20. The correspondences below are the
concept-level metric matches between the suites (same CK/McCabe concept,
different tool naming); unmapped target-side slots are zero-filled (or filled
with the source mean, depending on the black box's imputation mode).

Each mapping is given as {target_feature_name: source_feature_name} for the
ordered pair (target_suite -> source_suite), i.e. "fill the source-spec slot
`source_feature_name` with the target project's `target_feature_name` column".
"""

from __future__ import annotations

# PROMISE-20 canonical order (as stored in the ARFF files)
PROMISE_FEATURES = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
]

# target feature -> source (PROMISE) feature
_JIRA_TO_PROMISE = {
    "CountDeclMethod": "wmc",
    "MaxInheritanceTree": "dit",
    "CountClassCoupled": "cbo",
    "CountLineCode": "loc",
    "MaxCyclomatic": "max_cc",
    "AvgCyclomatic": "avg_cc",
}

_NASA_TO_PROMISE = {
    "CYCLOMATIC_COMPLEXITY": "max_cc",
    "CYCLOMATIC_DENSITY": "amc",
    "LOC_TOTAL": "loc",
}

_AEEEM_TO_PROMISE = {
    "ck_oo_wmc": "wmc",
    "ck_oo_dit": "dit",
    "ck_oo_noc": "noc",
    "ck_oo_cbo": "cbo",
    "ck_oo_rfc": "rfc",
    "ck_oo_lcom": "lcom",
    "ck_oo_fanIn": "ca",
    "ck_oo_fanOut": "ce",
    "ck_oo_numberOfLinesOfCode": "loc",
}

# Reverse directions (kept symmetric where a concept match exists).
def _reverse(mapping: dict[str, str]) -> dict[str, str]:
    rev: dict[str, str] = {}
    for tgt, src in mapping.items():
        rev.setdefault(src, tgt)
    return rev


_MAPPINGS: dict[tuple[str, str], dict[str, str]] = {
    ("JIRA", "PROMISE"): _JIRA_TO_PROMISE,
    ("NASA", "PROMISE"): _NASA_TO_PROMISE,
    ("AEEEM", "PROMISE"): _AEEEM_TO_PROMISE,
    ("PROMISE", "JIRA"): _reverse(_JIRA_TO_PROMISE),
    ("PROMISE", "NASA"): _reverse(_NASA_TO_PROMISE),
    ("PROMISE", "AEEEM"): _reverse(_AEEEM_TO_PROMISE),
    # The RQ protocol never queries PROMISE/JIRA black boxes on the remaining
    # ordered pairs; an empty mapping yields an all-imputed representation.
    ("NASA", "JIRA"): {},
    ("JIRA", "NASA"): {},
    ("NASA", "AEEEM"): {},
    ("AEEEM", "NASA"): {},
    ("JIRA", "AEEEM"): {},
    ("AEEEM", "JIRA"): {},
}


def get_mapping(target_suite: str, source_suite: str) -> dict[str, str]:
    """Feature-name mapping target_suite -> source_suite ({} if same suite)."""
    target_suite, source_suite = target_suite.upper(), source_suite.upper()
    if target_suite == source_suite:
        return {}
    return _MAPPINGS.get((target_suite, source_suite), {})
