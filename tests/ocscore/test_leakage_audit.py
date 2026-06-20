#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for staged OCScore leakage audit utility.'''

# Imports
###############################################################################
import pytest

from OCDocker.OCScore.Utils.LeakageAudit import run_leakage_audit

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


@pytest.mark.order(431)
def test_receptor_overlap_detected_for_affinity_split():
    audit = run_leakage_audit(
        pdbbind_split_diagnostics={
            "strategy": "affinity_quantile_stratified",
            "receptor_overlap": {"train∩validation": 2, "train∩test": 1, "validation∩test": 0},
        },
        strict=False,
    )
    assert any(finding.code == "pdbbind_receptor_overlap" for finding in audit.findings)


@pytest.mark.order(432)
def test_heldout_overlap_is_critical_in_strict_mode():
    audit = run_leakage_audit(
        dudez_split_diagnostics={
            "strategy": "receptor_heldout_complete",
            "receptor_overlap": {"train∩validation": 1, "train∩test": 0, "validation∩test": 0},
        },
        strict=True,
    )
    assert not audit.passed
