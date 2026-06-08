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
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
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
