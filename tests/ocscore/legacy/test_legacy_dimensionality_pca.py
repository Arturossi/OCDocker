#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Dimensionality.PCA.run_pca.
'''

# Imports
###############################################################################
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.legacy.models.dimensionality.PCA as ocpca

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

def _build_datasets():
    dudez = pd.DataFrame(
        {
            "receptor": ["r1", "r2", "r3", "r4"],
            "ligand": ["l1", "l2", "l3", "l4"],
            "name": ["n1", "n2", "n3", "n4"],
            "type": ["active", "inactive", "active", "inactive"],
            "db": ["dudez", "dudez", "dudez", "dudez"],
            "scoreA": [1.2, 0.3, 0.8, 1.0],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 1.2, 1.4, 1.6],
            "f3": [2.0, 2.2, 2.4, 2.6],
        }
    )
    pdbbind = pd.DataFrame(
        {
            "receptor": ["pr1", "pr2", "pr3", "pr4"],
            "ligand": ["pl1", "pl2", "pl3", "pl4"],
            "name": ["pn1", "pn2", "pn3", "pn4"],
            "type": ["x", "x", "y", "y"],
            "db": ["pdbbind", "pdbbind", "pdbbind", "pdbbind"],
            "experimental": [7.1, 6.8, 8.0, 7.7],
            "scoreA": [0.5, 0.6, 0.7, 0.8],
            "f1": [0.5, 0.7, 0.9, 1.1],
            "f2": [1.5, 1.7, 1.9, 2.1],
            "f3": [2.5, 2.7, 2.9, 3.1],
        }
    )
    return dudez, pdbbind, ["scoreA"]


## Public ##

@pytest.mark.order(614)
def test_run_pca_rejects_invalid_variance(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ocpca.ocerror.Error, "value_error", lambda msg: calls.append(msg))

    with pytest.raises(ValueError, match="between 0 and 1"):
        ocpca.run_pca(
            df_path="unused.csv",
            variance=0.0,
            pca_path=str(tmp_path),
            verbose=False,
        )

    assert calls
    assert "between 0 and 1" in calls[0]


@pytest.mark.order(615)
def test_run_pca_saves_model_and_returns_expected_path(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(ocpca.ocscoredata, "preprocess_df", lambda *_a, **_k: _build_datasets())
    monkeypatch.setattr(
        ocpca.ocscoreio,
        "save_object",
        lambda model, path: saved.update({"model": model, "path": path}),
    )

    out_path = ocpca.run_pca(
        df_path="input.csv",
        variance=0.95,
        pca_path=str(tmp_path),
        verbose=False,
    )

    assert out_path == str(tmp_path / "pca95.pkl")
    assert saved["path"] == out_path
    assert hasattr(saved["model"], "transform")


@pytest.mark.order(616)
def test_run_pca_verbose_branch_prints_dataset_diagnostics(monkeypatch, tmp_path):
    prints = []
    monkeypatch.setattr(ocpca.ocscoredata, "preprocess_df", lambda *_a, **_k: _build_datasets())
    monkeypatch.setattr(ocpca.ocscoreio, "save_object", lambda *_a, **_k: None)
    monkeypatch.setattr(ocpca.LOGGER, "info", lambda msg, *args: prints.append(msg % args if args else str(msg)))

    out_path = ocpca.run_pca(
        df_path="input.csv",
        variance=0.8,
        pca_path=str(tmp_path),
        verbose=True,
    )

    assert out_path == str(tmp_path / "pca80.pkl")
    assert any("NaNs in PCA datasets" in msg for msg in prints)
    assert any("Dataset sizes" in msg for msg in prints)
    assert any("Before PCA" in msg for msg in prints)
    assert any("After PCA" in msg for msg in prints)


@pytest.mark.order(617)
def test_run_pca_creates_missing_output_directory(monkeypatch, tmp_path):
    output_dir = tmp_path / "new_pca_dir" / "nested"
    saved = {}

    monkeypatch.setattr(ocpca.ocscoredata, "preprocess_df", lambda *_a, **_k: _build_datasets())
    monkeypatch.setattr(
        ocpca.ocscoreio,
        "save_object",
        lambda model, path: saved.update({"model": model, "path": path}),
    )

    out_path = ocpca.run_pca(
        df_path="input.csv",
        variance=0.95,
        pca_path=str(output_dir),
        verbose=False,
    )

    assert output_dir.is_dir()
    assert out_path == str(output_dir / "pca95.pkl")
    assert saved["path"] == out_path


@pytest.mark.order(618)
def test_run_pca_defaults_to_cwd_when_path_is_empty(monkeypatch, tmp_path):
    saved = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ocpca.ocscoredata, "preprocess_df", lambda *_a, **_k: _build_datasets())
    monkeypatch.setattr(
        ocpca.ocscoreio,
        "save_object",
        lambda model, path: saved.update({"model": model, "path": path}),
    )

    out_path = ocpca.run_pca(
        df_path="input.csv",
        variance=0.9,
        pca_path="",
        verbose=False,
    )

    assert out_path == str(tmp_path / "pca90.pkl")
    assert saved["path"] == out_path
