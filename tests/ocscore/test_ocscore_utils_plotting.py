#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Utils.Plotting module.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Utils.Plotting as ocscoreplot

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

@pytest.mark.order(287)
def test_plot_correlation_similarity_writes_both_standard_and_sorted_heatmaps(monkeypatch):
    saved = []
    monkeypatch.setattr(ocscoreplot.plt, "savefig", lambda path, **_k: saved.append(path))

    df1 = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 1.0, 4.0, 3.0],
            "c": [3.0, 3.5, 2.5, 2.0],
        }
    )
    df2 = pd.DataFrame(
        {
            "a": [1.5, 2.5, 3.5, 4.5],
            "b": [1.8, 1.2, 3.8, 3.2],
            "c": [2.8, 3.2, 2.2, 1.8],
        }
    )

    ocscoreplot.plot_correlation_similarity(df1, df2, columns=["a", "b", "c"], annot=True, fontsize=8, normalize=True)
    ocscoreplot.plot_correlation_similarity(df1, df2, columns=["a", "b", "c"], annot=False, fontsize=None, normalize=False)

    assert "correlation_similarity.png" in saved
    assert "correlation_similarity_sorted.png" in saved


@pytest.mark.order(288)
def test_plot_roc_curves_supports_single_and_multiple_db_names(monkeypatch):
    saved = []
    monkeypatch.setattr(ocscoreplot.plt, "savefig", lambda path, **_k: saved.append(path))

    labels = pd.Series([0, 1, 0, 1], dtype=int)
    df_single = pd.DataFrame(
        {
            "db": ["DUDEz", "DUDEz", "DUDEz", "DUDEz"],
            "m1": [0.1, 0.9, 0.2, 0.8],
            "m2": [0.3, 0.7, 0.4, 0.6],
        }
    )
    df_multi = pd.DataFrame(
        {
            "db": ["DUDEz", "PDBbind", "DUDEz", "PDBbind"],
            "m1": [0.1, 0.9, 0.2, 0.8],
            "m2": [0.3, 0.7, 0.4, 0.6],
        }
    )

    ocscoreplot.plot_roc_curves(df_single, feature_cols=["m1", "m2"], labels=labels, title="roc_single")
    ocscoreplot.plot_roc_curves(df_multi, feature_cols=["m1", "m2"], labels=labels, title="roc_multi")

    assert "roc_single.png" in saved
    assert "roc_multi.png" in saved
