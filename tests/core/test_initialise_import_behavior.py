#!/usr/bin/env python3

# Description
###############################################################################
'''
Regression tests for Initialise import behavior in test contexts.
'''

# Imports
###############################################################################
import importlib

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


# Functions
###############################################################################
## Public ##

def test_initialise_import_does_not_bootstrap_during_pytest(monkeypatch):
    monkeypatch.delenv("OC_BUILD_DOCS", raising=False)

    import OCDocker.Initialise as ocinit

    reloaded = importlib.reload(ocinit)
    assert reloaded.bootstrapped is False
