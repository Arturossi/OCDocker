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
