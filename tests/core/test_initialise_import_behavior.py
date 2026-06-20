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
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''


# Functions
###############################################################################
## Public ##

def test_initialise_import_does_not_bootstrap_during_pytest(monkeypatch):
    monkeypatch.delenv("OC_BUILD_DOCS", raising=False)

    import OCDocker.Initialise as ocinit

    reloaded = importlib.reload(ocinit)
    assert reloaded.bootstrapped is False
