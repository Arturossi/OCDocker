#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Toolbox.Basetools helpers.
'''

# Imports
###############################################################################
import builtins

import pytest

import OCDocker.Toolbox.Basetools as ocbasetools

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

@pytest.mark.order(262)
def test_redirect_to_tqdm_falls_back_to_builtin_print(monkeypatch):
    captured = []

    def _fake_builtin_print(*args, **kwargs):
        _ = kwargs
        captured.append(args)

    def _raise_from_tqdm(*args, **kwargs):
        _ = (args, kwargs)
        raise OSError("forced write failure")

    monkeypatch.setattr(ocbasetools.tqdm, "write", _raise_from_tqdm)
    monkeypatch.setattr(builtins, "print", _fake_builtin_print)

    with ocbasetools.redirect_to_tqdm():
        print("fallback-path")  # noqa: T201

    assert captured
    assert captured[0][0] == "fallback-path"
