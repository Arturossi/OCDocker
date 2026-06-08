#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Utils.SimpleConsensus helpers.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Utils.legacy.SimpleConsensus as ocsimple

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

def _mock_preprocess_df(_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    score_columns = ["SMINA_1", "VINA_1", "ODDT_1", "PLANTS_1"]

    dudez_data = pd.DataFrame(
        {
            "type": ["ligand", "decoy", "ligand", "decoy"],
            "SMINA_1": [0.9, 0.1, 0.8, 0.2],
            "VINA_1": [0.85, 0.15, 0.75, 0.25],
            "ODDT_1": [0.92, 0.12, 0.82, 0.22],
            "PLANTS_1": [0.88, 0.18, 0.78, 0.28],
        }
    )

    pdbbind_data = pd.DataFrame(
        {
            "experimental": [1.0, 2.0, 3.0, 4.0],
            "SMINA_1": [1.1, 2.1, 2.9, 4.2],
            "VINA_1": [0.9, 1.9, 3.1, 3.8],
            "ODDT_1": [1.0, 2.2, 2.8, 4.1],
            "PLANTS_1": [1.2, 1.8, 3.2, 3.9],
        }
    )

    return dudez_data, pdbbind_data, score_columns


## Public ##

@pytest.mark.order(696)
def test_simple_consensus_computes_expected_columns_with_optional_fields():
    full_df = pd.DataFrame(
        {
            "experimental": [1.0, 2.0, 3.0],
            "type": ["ligand", "decoy", "ligand"],
            "s1": [0.5, 0.2, 0.9],
            "s2": [0.6, 0.3, 0.8],
        }
    )
    out_full = ocsimple.simple_consensus(full_df, ["s1", "s2"])
    assert "mean" in out_full.columns
    assert "kurtosis" in out_full.columns
    assert "experimental" in out_full.columns
    assert "type" in out_full.columns

    partial_df = pd.DataFrame({"s1": [0.1, 0.2], "s2": [0.3, 0.4]})
    out_partial = ocsimple.simple_consensus(partial_df, ["s1", "s2"])
    assert "experimental" not in out_partial.columns
    assert "type" not in out_partial.columns


@pytest.mark.order(697)
def test_perform_simple_consensus_happy_path_with_selected_metrics(monkeypatch):
    monkeypatch.setattr(ocsimple.ocscoredata, "preprocess_df", _mock_preprocess_df)

    out = ocsimple.perform_simple_consensus(
        df_path="dummy.csv",
        metrics=["mean", "max"],
        verbose=False,
    )

    assert list(out.columns) == ["AUC", "RMSE"]
    assert set(out.index.tolist()) == {"mean", "max"}
    assert out["AUC"].between(0, 1).all()
    assert (out["RMSE"] >= 0).all()


@pytest.mark.order(698)
def test_perform_simple_consensus_uses_default_metrics_when_empty(monkeypatch):
    monkeypatch.setattr(ocsimple.ocscoredata, "preprocess_df", _mock_preprocess_df)

    out = ocsimple.perform_simple_consensus(
        df_path="dummy.csv",
        metrics=[],
        verbose=False,
    )

    assert "mean" in out.index
    assert "median" in out.index
    assert "kurtosis" in out.index
    assert len(out.index) == 13


@pytest.mark.order(699)
def test_perform_simple_consensus_rejects_unknown_metrics(monkeypatch):
    monkeypatch.setattr(ocsimple.ocscoredata, "preprocess_df", _mock_preprocess_df)

    with pytest.raises(ValueError, match="Unknown consensus metrics"):
        ocsimple.perform_simple_consensus(
            df_path="dummy.csv",
            metrics=["mean", "not_a_metric"],
            verbose=False,
        )


@pytest.mark.order(700)
def test_perform_simple_consensus_verbose_uses_rmse_threshold(monkeypatch):
    messages = []
    monkeypatch.setattr(ocsimple.ocscoredata, "preprocess_df", _mock_preprocess_df)
    monkeypatch.setattr(ocsimple.LOGGER, "info", lambda msg, *args: messages.append(msg % args if args else msg))

    out = ocsimple.perform_simple_consensus(
        df_path="dummy.csv",
        threshold=10.0,
        metrics=["mean", "max"],
        verbose=True,
    )

    assert not out.empty
    assert len(messages) == 1
    assert "threshold" in messages[0]
    assert "10.0" in messages[0]
    assert "RMSE" in messages[0]
