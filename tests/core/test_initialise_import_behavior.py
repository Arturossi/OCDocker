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


# Functions
###############################################################################
## Public ##

def test_initialise_import_does_not_bootstrap_during_pytest(monkeypatch):
    monkeypatch.delenv("OCDOCKER_AUTO_BOOTSTRAP", raising=False)
    monkeypatch.delenv("OCDOCKER_NO_AUTO_BOOTSTRAP", raising=False)
    monkeypatch.delenv("OC_BUILD_DOCS", raising=False)

    import OCDocker.Initialise as ocinit

    reloaded = importlib.reload(ocinit)
    assert reloaded.bootstrapped is False
