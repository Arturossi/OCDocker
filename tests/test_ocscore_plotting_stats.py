#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.Plotting.Stats helpers.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot

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

def _toy_method_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Methodology": [
                "NN_Method",
                "NN_Method",
                "XGB_Method",
                "XGB_Method",
                "Transformer_Method",
                "Transformer_Method",
                "Mean consensus",
            ],
            "RMSE": [1.0, 1.2, 1.1, 1.3, 1.4, 1.2, 0.9],
            "AUC": [0.70, 0.45, 0.80, 0.60, 0.55, 0.40, 0.75],
        }
    )


## Public ##

@pytest.mark.order(277)
def test_plot_bar_with_significance_and_heatmap_raise_without_pvalue_column():
    bad = pd.DataFrame({"A": ["m1"], "B": ["m2"], "diff": [0.1]})

    with pytest.raises(ValueError, match="p-value"):
        ocstatplot.plot_bar_with_significance(bad, metric="AUC")

    with pytest.raises(ValueError, match="p-value"):
        ocstatplot.plot_heatmap(bad, title="T", metric="AUC")


@pytest.mark.order(278)
def test_plot_bar_with_significance_and_heatmap_happy_paths(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(ocstatplot.sns, "barplot", lambda **_k: ocstatplot.plt.gca())
    monkeypatch.setattr(ocstatplot.np, "fill_diagonal", lambda *_a, **_k: None)

    gh = pd.DataFrame(
        {
            "A": ["m1", "m1", "m2"],
            "B": ["m2", "m3", "m3"],
            "diff": [0.2, -0.1, 0.15],
            "pval": [0.01, 0.20, 0.001],
        }
    )

    ocstatplot.plot_bar_with_significance(gh, metric="AUC", output_dir=str(tmp_path), top_n=2)
    ocstatplot.plot_heatmap(gh, title="GH", metric="AUC", output_dir=str(tmp_path))

    assert any("games_howell_bar_AUC.png" in p for p in saved)
    assert any("games_howell_heatmap_AUC.png" in p for p in saved)


@pytest.mark.order(279)
def test_plot_scatter_and_summary_plots_happy_paths(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))

    df = _toy_method_df()
    colours = {
        "NN_Method": "tab:blue",
        "XGB_Method": "tab:green",
        "Transformer_Method": "tab:red",
        "Mean consensus": "tab:purple",
    }

    ocstatplot.plot_barplots(df, n_trials=3, colour_mapping=colours, output_dir=str(tmp_path))
    ocstatplot.plot_boxplots(df, n_trials=3, colour_mapping=colours, output_dir=str(tmp_path), show_simple_consensus=False)
    ocstatplot.plot_combined_metric_scatter(df, n_trials=3, colour_mapping=colours, output_dir=str(tmp_path), alpha=0.7)

    ocstatplot.plot_scatterplot(
        df_rmse=df[df["RMSE"] <= 1.3],
        df_auc=df[df["AUC"] >= 0.5],
        df_all=df,
        n_trials=3,
        colour_mapping=colours,
        output_dir=str(tmp_path),
        orientation="horizontal",
    )
    ocstatplot.plot_scatterplot(
        df_rmse=df[df["RMSE"] <= 1.3],
        df_auc=df[df["AUC"] >= 0.5],
        df_all=df,
        n_trials=3,
        colour_mapping=colours,
        output_dir=str(tmp_path),
        orientation="vertical",
    )

    assert any("barplot_rmse_auc_3.png" in p for p in saved)
    assert any("boxplots_rmse_auc_3.png" in p for p in saved)
    assert any("scatter_combined_metric_3.png" in p for p in saved)
    assert any("scatter_rmse_auc_panels_3.png" in p for p in saved)


@pytest.mark.order(280)
def test_plot_scatterplot_rejects_invalid_orientation():
    df = _toy_method_df()
    colours = {m: "tab:blue" for m in df["Methodology"].unique()}

    with pytest.raises(ValueError, match="Orientation must be"):
        ocstatplot.plot_scatterplot(
            df_rmse=df,
            df_auc=df,
            df_all=df,
            n_trials=1,
            colour_mapping=colours,
            output_dir=".",
            orientation="diagonal",
        )


@pytest.mark.order(281)
def test_diagnostics_and_pca_helpers(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))

    df = _toy_method_df()
    ocstatplot.plot_normality_and_variance_diagnostics(df, metric="RMSE", n_trials=5, output_dir=str(tmp_path))

    importance_equal = pd.DataFrame({"Feature": ["a", "b", "c"], "Importance": [1.0, 1.0, 1.0]})
    importance_var = pd.DataFrame({"Feature": ["f1", "f2", "f3", "f4"], "Importance": [0.1, 0.5, 0.2, 0.9]})

    ocstatplot.save_pca_importance_bins(importance_equal, pca_type="1", n_trials=5, output_dir=str(tmp_path), n_bins=10)
    ocstatplot.save_pca_importance_groups(importance_var, pca_type="1", n_trials=5, output_dir=str(tmp_path))
    ocstatplot.plot_pca_importance_barplot(importance_var, pca_type="1", n_features=3, n_trials=5, output_dir=str(tmp_path))
    ocstatplot.plot_pca_importance_histogram(importance_var, pca_type="1", n_trials=5, output_dir=str(tmp_path))

    bins_csv = tmp_path / "pca1_importance_bins_5.csv"
    groups_csv = tmp_path / "pca1_importance_groups_5.csv"
    assert bins_csv.exists()
    assert groups_csv.exists()
    assert "bin" in pd.read_csv(bins_csv).columns
    assert "Group" in pd.read_csv(groups_csv).columns
    assert any("diagnostics_RMSE_5.png" in p for p in saved)
    assert any("pca1_importance_top3_5.png" in p for p in saved)
    assert any("pca1_importance_hist_5.png" in p for p in saved)
