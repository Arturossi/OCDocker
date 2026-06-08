#!/usr/bin/env python3

# Description
###############################################################################
'''
Security tests for OCScore.Analysis.StatTests pickle model loading.
'''

# Imports
###############################################################################
import pickle

import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Analysis.legacy.StatTests as ocstattests

from sklearn.decomposition import PCA

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

@pytest.mark.order(746)
def test_load_pca_model_requires_trust_by_default(tmp_path, monkeypatch):
    model_path = tmp_path / "pca.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"dummy": 1}, f)

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)

    with pytest.raises(PermissionError):
        _ = ocstattests.load_pca_model(str(model_path))


@pytest.mark.order(747)
def test_load_pca_model_allows_explicit_trust(tmp_path, monkeypatch):
    model_path = tmp_path / "pca.pkl"
    payload = {"dummy": 1}
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.delenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", raising=False)

    loaded = ocstattests.load_pca_model(str(model_path), trusted=True)
    assert loaded == payload


@pytest.mark.order(748)
def test_load_pca_model_allows_env_opt_in(tmp_path, monkeypatch):
    model_path = tmp_path / "pca.pkl"
    payload = {"dummy": 1}
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    monkeypatch.setenv("OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION", "1")

    loaded = ocstattests.load_pca_model(str(model_path))
    assert loaded == payload


@pytest.mark.order(749)
def test_compute_pca_feature_importance_returns_sorted_dataframe():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.1, 3.1],
            [0.9, 1.9, 2.8],
            [1.1, 2.2, 3.2],
        ]
    )
    pca = PCA(n_components=2).fit(X)

    out = ocstattests.compute_pca_feature_importance(pca, ["f1", "f2", "f3"])
    assert list(out.columns) == ["Feature", "Importance"]
    assert out.shape[0] == 3
    assert out["Importance"].is_monotonic_decreasing


@pytest.mark.order(750)
def test_run_pca_analysis_handles_present_and_missing_models(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "out"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create only one PCA model so loop covers both success and missing-file paths
    X = np.array(
        [
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [1.2, 2.2],
        ]
    )
    pca = PCA(n_components=2).fit(X)
    with open(models_dir / "pca80.pkl", "wb") as f:
        pickle.dump(pca, f)

    calls = {"bar": 0, "hist": 0, "groups": 0, "bins": 0}
    monkeypatch.setattr(ocstattests.ocstatplot, "plot_pca_importance_barplot", lambda *a, **k: calls.__setitem__("bar", calls["bar"] + 1))
    monkeypatch.setattr(ocstattests.ocstatplot, "plot_pca_importance_histogram", lambda *a, **k: calls.__setitem__("hist", calls["hist"] + 1))
    monkeypatch.setattr(ocstattests.ocstatplot, "save_pca_importance_groups", lambda *a, **k: calls.__setitem__("groups", calls["groups"] + 1))
    monkeypatch.setattr(ocstattests.ocstatplot, "save_pca_importance_bins", lambda *a, **k: calls.__setitem__("bins", calls["bins"] + 1))

    df = pd.DataFrame({"f1": [1.0, 1.1], "f2": [2.0, 2.1]})
    ocstattests.run_pca_analysis(df, str(models_dir), str(output_dir), n_trials=1, n_features=2)

    assert calls["bar"] == 1
    assert calls["hist"] == 1
    assert calls["groups"] == 1
    assert calls["bins"] == 1


@pytest.mark.order(751)
def test_run_statistical_tests_uses_expected_outputs(tmp_path, monkeypatch):
    # Build minimal valid input dataframe
    df = pd.DataFrame(
        {
            "Methodology": ["A", "A", "B", "B"],
            "AUC": [0.8, 0.82, 0.75, 0.77],
            "RMSE": [1.2, 1.1, 1.4, 1.35],
        }
    )

    # Stub pingouin outputs as simple DataFrames with to_csv
    welch_df = pd.DataFrame({"p-unc": [0.05]})
    gh_df = pd.DataFrame({"A": ["A"], "B": ["B"], "pval": [0.04], "diff": [0.1]})
    monkeypatch.setattr(ocstattests.pg, "welch_anova", lambda **kwargs: welch_df.copy())
    monkeypatch.setattr(ocstattests.pg, "pairwise_gameshowell", lambda **kwargs: gh_df.copy())

    calls = {"bar": 0, "heat": 0, "diag": 0}
    monkeypatch.setattr(ocstattests.ocstatplot, "plot_bar_with_significance", lambda *a, **k: calls.__setitem__("bar", calls["bar"] + 1))
    monkeypatch.setattr(ocstattests.ocstatplot, "plot_heatmap", lambda *a, **k: calls.__setitem__("heat", calls["heat"] + 1))
    monkeypatch.setattr(ocstattests.ocstatplot, "plot_normality_and_variance_diagnostics", lambda *a, **k: calls.__setitem__("diag", calls["diag"] + 1))

    # Run in temp cwd so relative csvs/ output is isolated
    monkeypatch.chdir(tmp_path)
    (tmp_path / "csvs").mkdir(exist_ok=True)

    ocstattests.run_statistical_tests(
        df=df,
        n_trials=3,
        colour_mapping={"A": (0.1, 0.2, 0.3), "B": (0.3, 0.2, 0.1)},
        output_dir=str(tmp_path / "plots"),
    )

    assert (tmp_path / "csvs" / "welch_anova_auc_3.csv").is_file()
    assert (tmp_path / "csvs" / "welch_anova_rmse_3.csv").is_file()
    assert (tmp_path / "csvs" / "games_howell_posthoc_AUC_3.csv").is_file()
    assert (tmp_path / "csvs" / "games_howell_posthoc_RMSE_3.csv").is_file()
    assert calls["bar"] == 2
    assert calls["heat"] == 2
    assert calls["diag"] == 2
