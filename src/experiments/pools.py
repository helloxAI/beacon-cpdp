"""Scenario definitions for the experimental matrix.

Two black-box groups exist (pair keys "rq1" and "rq2"). Each scenario's
evaluation pool excludes whatever that scenario's black boxes touched at
training time -- the source project (labels) and the alignment target
(features) -- so no evaluated project was seen, at feature or label level,
by the black box under test:

  rq1.1  in-domain        PROMISE 8 projects: the rq1 pair's source
                          ant-1.7 and alignment target ivy-2.0 excluded
                          (velocity-1.6, unseen by the rq1 group, stays)
  rq1.2  out-of-domain    JIRA 7 + NASA 5 + AEEEM 5 = 17 (all unseen by
                          the rq1 group; jruby-1.1 belongs to the rq2
                          pair, not rq1)
  rq2.1  target-domain    JIRA 7. Only 7 JIRA projects exist, so
                          jruby-1.1 -- the rq2 pair's alignment target,
                          seen at feature level by transfer-type
                          backbones -- is necessarily included to reach
                          the scenario size 18 x 7.
  rq2.2  source-retest    PROMISE 8: the rq2 pair's source velocity-1.6
                          excluded (ant-1.7, the other pair's source,
                          also excluded to fix the pool at 8); ivy-2.0,
                          unseen by the rq2 group, stays
  rq2.3  unseen-domain    NASA 5 + AEEEM 5 = 10

This gives 18 x 8 = 144, 18 x 17 = 306, 18 x 7 = 126, 18 x 8 = 144 and
18 x 10 = 180 model-target pairs per scenario, and 50 evaluation targets
per backbone overall (Classic 8 x 50 = 400, Shallow/Deep 5 x 50 = 250).

The ablation scenario rq4 uses the rq1 group backbone SupCon_DP on the 9
non-source PROMISE projects with four variants (BlackBox, CGAD-only,
EGRW-only, Full) and 10 training seeds; the data split and the black-box
outputs are fixed across variants and seeds. SupCon_DP's recovered
network (encoder / head / classifier, no domain component) is trained on
source labels only, so ivy-2.0 is unseen for this backbone.

The sensitivity scenario rq5 uses the same backbone and target pool with a
one-factor-at-a-time sweep over gamma_ema, theta, epsilon and tau0.
"""

from __future__ import annotations

from src.data.arff_data import list_projects

RQ1_TOUCHED = {"ant-1.7", "ivy-2.0"}       # rq1 pair: source + alignment target
RQ2_TOUCHED = {"velocity-1.6", "ant-1.7"}  # rq2 source + other pair's source

PROMISE_RQ11_8 = [p for p in list_projects("PROMISE") if p not in RQ1_TOUCHED]
PROMISE_RQ22_8 = [p for p in list_projects("PROMISE") if p not in RQ2_TOUCHED]
PROMISE_RQ4_9 = [p for p in list_projects("PROMISE") if p != "ant-1.7"]
JIRA_ALL = list_projects("JIRA")
NASA_ALL = list_projects("NASA")
AEEEM_ALL = list_projects("AEEEM")


def _tag(suite: str, projects: list[str]) -> list[tuple[str, str]]:
    return [(suite, p) for p in projects]


SCENARIOS: dict[str, dict] = {
    "rq1.1": {"pair": "rq1", "targets": _tag("PROMISE", PROMISE_RQ11_8)},
    "rq1.2": {
        "pair": "rq1",
        "targets": _tag("JIRA", JIRA_ALL) + _tag("NASA", NASA_ALL) + _tag("AEEEM", AEEEM_ALL),
    },
    "rq2.1": {"pair": "rq2", "targets": _tag("JIRA", JIRA_ALL)},
    "rq2.2": {"pair": "rq2", "targets": _tag("PROMISE", PROMISE_RQ22_8)},
    "rq2.3": {"pair": "rq2", "targets": _tag("NASA", NASA_ALL) + _tag("AEEEM", AEEEM_ALL)},
}

ARCH_GROUPS = {
    "Classic": ["TNB", "HDP_KS", "CPDP_IFS", "CCA_Plus", "TCA", "CORAL", "SubspaceAlignment", "TrAdaBoost"],
    "Shallow NN": ["MLP_Zero", "MLP_Mean", "MLP_Dropout", "DAE", "DBN"],
    "Deep NN": ["FT_Transformer", "CNN_SDP", "SupCon_DP", "DANN", "DeepOT"],
}

RQ4_MODEL = "SupCon_DP"
RQ4_TARGETS = _tag("PROMISE", PROMISE_RQ4_9)
RQ4_SEEDS = [42 + i for i in range(10)]
RQ4_VARIANTS = {
    "BlackBox": None,
    "CGAD-only": {"epsilon": 1.0},
    "EGRW-only": {"kd_mode": "soft"},
    "Full": {},
}

RQ5_MODEL = "SupCon_DP"
RQ5_TARGETS = _tag("PROMISE", PROMISE_RQ4_9)
RQ5_SEED = 42
RQ5_SWEEPS = {
    "gamma_ema": [0.1, 0.3, 0.5, 0.6, 0.7, 0.9],
    "theta": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "epsilon": [0.0, 0.01, 0.05, 0.1, 0.2, 0.4],
    "tau0": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
