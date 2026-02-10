#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for DB package export behavior.
'''

# Imports
###############################################################################
import builtins
import importlib

import pytest

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##


@pytest.mark.order(350)
def test_db_package_handles_optional_submodule_import_error(monkeypatch):
    db_pkg = importlib.import_module("OCDocker.DB")
    real_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "" and tuple(fromlist) == ("DB",) and level == 1:
            raise ImportError("forced test failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    reloaded = importlib.reload(db_pkg)
    assert reloaded.DB is None
    assert isinstance(reloaded.DB_IMPORT_ERROR, ImportError)

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(reloaded)
