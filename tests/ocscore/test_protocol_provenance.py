#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for production-grade protocol provenance helpers."""

# Imports
###############################################################################
import pytest

from OCDocker.OCScore.Utils.ProtocolProvenance import build_split_assignments_payload


# License
###############################################################################
"""
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
"""


# Functions
###############################################################################
## Public ##


@pytest.mark.order(295)
def test_build_split_assignments_payload_includes_indices_and_receptors():
    payload = build_split_assignments_payload(
        pdbbind_stage={
            "split_indices": {"train": [0, 1], "validation": [2], "test": [3]},
            "split_diagnostics": {
                "strategy": "receptor_heldout",
                "receptor_counts": {"train": 2, "validation": 1, "test": 1},
                "receptor_overlap": {"train∩validation": 0, "train∩test": 0, "validation∩test": 0},
            },
        },
        dudez_stage={
            "split_indices": {"train": [10, 11], "validation": [12], "test": [13]},
            "split_diagnostics": {
                "strategy": "receptor_heldout_complete",
                "train_receptors": ["r1", "r2"],
                "validation_receptors": ["r3"],
                "test_receptors": ["r4"],
            },
        },
    )
    assert payload["pdbbind"]["split_indices"]["train"] == [0, 1]
    assert payload["dudez"]["split_indices"]["test"] == [13]
    assert payload["pdbbind"]["split_diagnostics"]["receptor_overlap"]["train∩validation"] == 0
    assert payload["dudez"]["split_diagnostics"]["test_receptors"] == ["r4"]
