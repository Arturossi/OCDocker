#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.Plotting.Colouring helpers.
'''

# Imports
###############################################################################
import types

import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.Plotting.Colouring as occolour

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

@pytest.mark.order(318)
def test_set_color_mapping_supports_builtin_and_rejects_invalid_palette():
    df = pd.DataFrame({"Methodology": ["NN", "XGB", "NN", "TRANS"]})

    mapping = occolour.set_color_mapping(df, palette_colour="Set2")
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {"NN", "XGB", "TRANS"}

    with pytest.raises(ValueError, match="Unsupported palette"):
        occolour.set_color_mapping(df, palette_colour="invalid_palette")


@pytest.mark.order(319)
def test_set_color_mapping_handles_glasbey_with_and_without_colorcet(monkeypatch):
    df = pd.DataFrame({"Methodology": ["NN", "XGB", "TRANS"]})

    monkeypatch.setattr(occolour, "cc", None, raising=False)
    fallback_mapping = occolour.set_color_mapping(df, palette_colour="glasbey")
    assert set(fallback_mapping.keys()) == {"NN", "XGB", "TRANS"}

    fake_cc = types.SimpleNamespace(glasbey=["#000000", "#111111", "#222222", "#333333"])
    monkeypatch.setattr(occolour, "cc", fake_cc, raising=False)
    direct_mapping = occolour.set_color_mapping(df, palette_colour="glasbey")
    assert set(direct_mapping.keys()) == {"NN", "XGB", "TRANS"}
