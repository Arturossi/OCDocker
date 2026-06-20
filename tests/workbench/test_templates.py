#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench starter spec templates.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import available_template_names
from OCDocker.Workbench import build_template_payload
from OCDocker.Workbench import build_template_spec
from OCDocker.Workbench import plan_command

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def test_available_template_names_are_stable() -> None:
    '''Registered template names are deterministic.'''

    assert available_template_names() == (
        "ocscore_ablation",
        "ocscore_study",
        "vs_campaign",
    )


def test_templates_are_valid_and_plannable() -> None:
    '''Starter templates build validated specs that can be planned.'''

    for name in available_template_names():
        spec = build_template_spec(name)
        plan = plan_command(spec)
        payload = build_template_payload(name)

        assert payload["type"] == spec.type
        assert payload["schema_version"] == 1
        assert plan.command


def test_unknown_template_name_fails() -> None:
    '''Unknown template names raise a clear validation error.'''

    with pytest.raises(ValueError, match="Unknown Workbench template"):
        build_template_spec("unknown")
