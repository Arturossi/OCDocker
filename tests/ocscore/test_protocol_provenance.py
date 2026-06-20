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
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
