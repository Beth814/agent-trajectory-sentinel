"""The protocol decision must not drift from the code or from its evidence.

A decision record is only worth something if it stays true. These pin it to
the module that defines it, to the sample-size floors it claims to respect,
and to the committed studies it cites.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from derail.monitor.deployment_calibration import (_RECOMMENDED_CALIBRATION,
                                                   _RECOMMENDED_FIT)
from derail.monitor.fusion_protocol import PROTOCOL, RECORD_PATH, REPO_ROOT
from derail.monitor.normalized_fusion import STATISTICS


def test_the_committed_record_matches_the_module():
    assert RECORD_PATH.exists(), "the protocol decision was never written out"
    assert json.loads(RECORD_PATH.read_text("utf-8")) == PROTOCOL.as_record()


def test_the_protocol_is_buildable():
    assert PROTOCOL.statistic in STATISTICS
    assert PROTOCOL.channels == ("e", "m")
    assert PROTOCOL.threshold_on == "fused_statistic"
    assert 0.0 < PROTOCOL.fa_budget < 1.0


def test_the_fold_sizes_respect_the_measured_floors():
    """The learning curve is the authority on these, not this file."""
    assert PROTOCOL.min_fit_episodes >= _RECOMMENDED_FIT
    assert PROTOCOL.min_threshold_episodes >= _RECOMMENDED_CALIBRATION


def test_the_untouched_test_size_can_carry_the_false_alarm_claim():
    """At the chosen N, a realistic 2% observed rate must still bound below
    the 5% budget -- otherwise the evaluation cannot say what it is for."""
    precision = pd.read_csv(REPO_ROOT / "results" / "tables" /
                            "calibration_fa_precision.csv")
    row = precision[(precision["n_untouched_healthy"] ==
                     PROTOCOL.untouched_test_episodes) &
                    (precision["observed_fa"].round(3) == 0.020)]
    assert not row.empty, (
        f"no precision row for N={PROTOCOL.untouched_test_episodes} at 2% "
        f"observed; the chosen size is not backed by the committed table")
    assert bool(row["supports_5pct_claim"].iloc[0])
    assert float(row["ci_hi"].iloc[0]) < PROTOCOL.fa_budget


def test_cross_fitting_is_recorded_as_rejected():
    assert PROTOCOL.protocol == "three_fold_holdout"
    source = (REPO_ROOT / "derail" / "monitor" /
              "fusion_protocol.py").read_text("utf-8")
    assert "Cross-fitting was tried and rejected" in source, (
        "the record must carry why cross-fitting was not used, not just "
        "which protocol won")


def test_every_cited_evidence_artifact_exists():
    missing = [name for name in PROTOCOL.evidence
               if not (REPO_ROOT / name).exists()]
    assert not missing, f"protocol cites artifacts that do not exist: {missing}"


def test_the_admission_rule_is_the_frozen_one():
    assert PROTOCOL.healthy_admission == "task_verifier_must_pass"
    assert PROTOCOL.exclude_unscoreable_from_calibration is True
    assert PROTOCOL.report_both_fa_denominators is True


@pytest.mark.parametrize("column", ["detection_rate_holm_p",
                                    "scoreable_fa_rate_holm_p"])
def test_the_arm_comparison_reports_corrected_p_values(column):
    """Selection may not rest on an uncorrected p-value: several metrics were
    tested per deployment."""
    paired = pd.read_csv(REPO_ROOT / "results" / "tables" /
                         "fusion_arm_paired.csv")
    assert column in paired.columns
    assert paired["replicates"].min() >= 30, (
        "below this project's replicate floor, where a gain at p = 0.004 has "
        "previously come back at p = 0.948 on a later block")
