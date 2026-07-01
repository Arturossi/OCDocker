#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for reusable SHAP plotting utilities.
'''

# Imports
###############################################################################
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Analysis.SHAP.Plots as shapplots

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##


def _shap_values() -> pd.DataFrame:
    '''Return a deterministic SHAP table.

    Returns
    -------
    pd.DataFrame
        SHAP values.
    '''

    return pd.DataFrame(
        {
            "ligand_PMI1": [1.0, 2.0, -3.0, 0.5],
            "ligand_AUTOCORR2D_1": [0.1, 0.2, -0.3, 0.4],
            "plants_plp": [0.5, -0.6, 0.7, -0.8],
            "receptor_TotalAALength": [0.05, 0.02, -0.03, 0.01],
        }
    )


def _feature_matrix() -> pd.DataFrame:
    '''Return a deterministic feature matrix.

    Returns
    -------
    pd.DataFrame
        Feature matrix.
    '''

    return pd.DataFrame(
        {
            "ligand_PMI1": [10.0, 20.0, 30.0, 40.0],
            "ligand_AUTOCORR2D_1": [1.0, 2.0, 3.0, 4.0],
            "plants_plp": [-1.0, -2.0, -3.0, -4.0],
            "receptor_TotalAALength": [100.0, 110.0, 120.0, 130.0],
        }
    )


def _patch_shap_plotters(monkeypatch):
    '''Patch SHAP plotters with deterministic matplotlib plots.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    '''

    def fake_summary_plot(*_args, **_kwargs):
        plt.figure()
        plt.plot([0, 1], [0, 1])

    def fake_dependence_plot(*_args, **_kwargs):
        plt.scatter([0, 1], [1, 0])

    monkeypatch.setattr(shapplots.shap, "summary_plot", fake_summary_plot)
    monkeypatch.setattr(shapplots.shap, "dependence_plot", fake_dependence_plot)


## Public ##


@pytest.mark.order(434)
def test_assign_feature_families_uses_default_suggestions():
    '''Feature-family assignment should use ordered default patterns.'''

    table = shapplots.assign_feature_families(
        [
            "ligand_PMI2",
            "ligand_AUTOCORR2D_44",
            "plants_plp",
            "receptor_countA",
            "ligand_CustomDescriptor",
            "custom_score",
        ]
    )

    assert table["family"].tolist() == [
        "PMI",
        "AUTOCORR2D",
        "PLANTS",
        "receptor",
        "other ligand",
        "other",
    ]


@pytest.mark.order(435)
def test_family_spec_can_be_loaded_from_json(tmp_path):
    '''Family specifications should be loadable from JSON.'''

    spec_path = tmp_path / "families.json"
    spec_path.write_text(
        json.dumps({"families": {"custom": ["ligand_PMI*"], "other": ["*"]}}),
        encoding="utf-8",
    )

    table = shapplots.assign_feature_families(["ligand_PMI1", "plants_plp"], spec_path)

    assert table["family"].tolist() == ["custom", "other"]


@pytest.mark.order(436)
def test_global_and_family_plots_write_csv_and_png(tmp_path):
    '''Global and family SHAP plots should write image and summary CSV files.'''

    artifacts = shapplots.save_shap_plot_suite(
        _shap_values(),
        None,
        tmp_path,
        policy="full/model",
        top_n=3,
    )

    assert (tmp_path / "full_model_shap_feature_importance.png").exists()
    assert (tmp_path / "full_model_shap_feature_importance_logx.png").exists()
    assert (tmp_path / "full_model_shap_top_features.csv").exists()
    assert (tmp_path / "full_model_shap_family_importance.png").exists()
    assert (tmp_path / "full_model_shap_family_importance_logx.png").exists()
    assert (tmp_path / "full_model_shap_family_importance.csv").exists()
    assert (tmp_path / "shap_family_importance_all_policies.csv").exists()
    assert "feature_importance_png" in artifacts
    top = pd.read_csv(tmp_path / "full_model_shap_top_features.csv")
    assert top.shape[0] == 3


@pytest.mark.order(436)
def test_log_importance_plots_filter_or_keep_zero_rows(tmp_path):
    '''Log-scale SHAP plots should filter zero rows by default.'''

    shap_values = pd.DataFrame(
        {
            "ligand_PMI1": [1.0, -1.0],
            "plants_zero": [0.0, 0.0],
        }
    )
    artifacts = shapplots.save_global_feature_importance_plot(
        shap_values,
        None,
        tmp_path,
        policy="zero_default",
        include_log_plot=True,
    )
    keep_artifacts = shapplots.save_global_feature_importance_plot(
        shap_values,
        None,
        tmp_path,
        policy="zero_keep",
        include_log_plot=True,
        filter_zero_rows_log=False,
    )

    assert (tmp_path / "zero_default_shap_feature_importance_logx.png").exists()
    assert (tmp_path / "zero_keep_shap_feature_importance_logx.png").exists()
    assert "feature_importance_logx_png" in artifacts
    assert "feature_importance_logx_png" in keep_artifacts


@pytest.mark.order(437)
def test_beeswarm_and_dependence_skip_missing_features(monkeypatch, tmp_path, caplog):
    '''Dependence plotting should skip missing feature names without failing.'''

    _patch_shap_plotters(monkeypatch)

    artifacts = shapplots.save_shap_plot_suite(
        _shap_values(),
        None,
        tmp_path,
        policy="ligand",
        feature_matrix=_feature_matrix(),
        dependence_features=["ligand_PMI1", "missing_feature"],
    )

    assert (tmp_path / "ligand_shap_beeswarm.png").exists()
    assert (tmp_path / "ligand_shap_dependence_ligand_PMI1.png").exists()
    assert artifacts["skipped_features"] == ["missing_feature"]
    assert "Skipping missing SHAP dependence feature" in caplog.text


@pytest.mark.order(438)
def test_target_heatmap_and_label_distribution_write_outputs(tmp_path):
    '''Target heatmap and label distribution should write plot and CSV outputs.'''

    metadata = pd.DataFrame({"target_id": ["A", "A", "B", "B"]})
    labels = pd.DataFrame({"class": ["active", "decoy", "active", "decoy"]})

    artifacts = shapplots.save_shap_plot_suite(
        _shap_values(),
        None,
        tmp_path,
        policy="dudez",
        sample_metadata=metadata,
        target_column="target_id",
        labels=labels,
        label_column="class",
    )

    assert (tmp_path / "dudez_target_family_shap_heatmap.png").exists()
    assert (tmp_path / "dudez_target_family_shap_heatmap_logcolor.png").exists()
    assert (tmp_path / "dudez_target_family_shap_heatmap.csv").exists()
    assert (tmp_path / "dudez_active_decoy_shap_family_distribution.png").exists()
    assert (tmp_path / "dudez_active_decoy_shap_family_distribution.csv").exists()
    assert "target_family_heatmap_png" in artifacts
    assert "target_family_heatmap_logcolor_png" in artifacts


@pytest.mark.order(439)
def test_path_based_suite_loads_shap_and_features(monkeypatch, tmp_path):
    '''Path-based SHAP plotting should accept explicit SHAP and feature CSVs.'''

    _patch_shap_plotters(monkeypatch)
    shap_path = tmp_path / "shap_values.csv"
    feature_path = tmp_path / "features.csv"
    _shap_values().to_csv(shap_path, index=False)
    _feature_matrix().to_csv(feature_path, index=False)

    artifacts = shapplots.save_shap_plot_suite_from_paths(
        shap_path,
        tmp_path / "plots",
        policy="from_paths",
        feature_matrix_path=feature_path,
        dependence_features=["plants_plp"],
        top_n=2,
    )

    assert (tmp_path / "plots" / "from_paths_shap_feature_importance.png").exists()
    assert (tmp_path / "plots" / "from_paths_shap_beeswarm.png").exists()
    assert (tmp_path / "plots" / "from_paths_shap_dependence_plants_plp.png").exists()
    assert artifacts["dependence_pngs"]["plants_plp"].endswith("plants_plp.png")
