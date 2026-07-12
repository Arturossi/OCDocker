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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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

def _toy_significance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy": ["shape_only", "no_pmi", "ligand_plus_scoring_function"],
            "reference_mean": [0.420, 0.420, 0.420],
            "policy_mean": [0.164, 0.432, 0.491],
            "mean_diff": [-0.256, 0.012, 0.071],
            "pvalue": [1.5e-8, 0.22, 0.005],
            "pvalue_corrected": [3.2e-7, 1.0, 0.09],
            "reject_null": [True, False, False],
        }
    )


@pytest.mark.order(512)
def test_plot_ablation_bedroc_significance_bars_happy_path(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))

    ocstatplot.plot_ablation_bedroc_significance_bars(
        _toy_significance_df(),
        reference_policy="full_ocscore",
        metric_label="BEDROC",
        output_dir=str(tmp_path),
    )

    assert any("ablation_bedroc_significance_bars.png" in p for p in saved)


@pytest.mark.order(513)
def test_plot_ablation_bedroc_significance_bars_handles_nan_pvalue(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))

    df = _toy_significance_df()
    df.loc[1, "pvalue_corrected"] = float("nan")
    ocstatplot.plot_ablation_bedroc_significance_bars(df, output_dir=str(tmp_path))

    assert any("ablation_bedroc_significance_bars.png" in p for p in saved)


@pytest.mark.order(514)
def test_plot_bedroc_vs_shortcut_risk_scatter_happy_path(tmp_path):
    plot_df = pd.DataFrame({
        "policy": ["full_ocscore", "no_pmi", "shape_only"],
        "bedroc_mean": [0.420, 0.452, 0.164],
        "shortcut_risk_max_pct": [72.8, 20.0, 91.2],
    })

    ocstatplot.plot_bedroc_vs_shortcut_risk_scatter(plot_df, output_dir=str(tmp_path))

    assert (tmp_path / "ablation_bedroc_vs_shortcut_risk_scatter.png").is_file()


@pytest.mark.order(515)
def test_plot_bedroc_vs_shortcut_risk_scatter_accepts_localized_text(tmp_path):
    plot_df = pd.DataFrame({
        "policy": ["full_ocscore", "no_pmi", "shape_only"],
        "bedroc_mean": [0.420, 0.452, 0.164],
        "shortcut_risk_max_pct": [72.8, 20.0, 91.2],
    })

    ocstatplot.plot_bedroc_vs_shortcut_risk_scatter(
        plot_df,
        title="Titulo",
        xlabel="Eixo x",
        legend_labels={"reference": "Referencia"},
        zone_note="Zona",
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "ablation_bedroc_vs_shortcut_risk_scatter.png").is_file()


@pytest.mark.order(516)
def test_classify_policies_by_shortcut_rule_applies_both_conditions():
    plot_df = pd.DataFrame({
        # reference, then: beats it at low risk, beats it at high risk, and two
        # that do not beat it (one of which is very high risk, so it would be
        # flagged were the risk condition applied on its own).
        "policy": ["full_ocscore", "keep_me", "drop_me", "below_ref", "below_ref_risky"],
        "bedroc_mean": [0.420, 0.462, 0.491, 0.405, 0.365],
        "shortcut_risk_max_pct": [72.8, 10.4, 61.7, 17.0, 81.4],
    })

    retained, discarded = ocstatplot.classify_policies_by_shortcut_rule(plot_df)

    assert retained == ["keep_me"]
    assert discarded == ["drop_me"]


@pytest.mark.order(517)
def test_classify_policies_by_shortcut_rule_raises_without_reference():
    plot_df = pd.DataFrame({
        "policy": ["no_pmi"],
        "bedroc_mean": [0.432],
        "shortcut_risk_max_pct": [18.6],
    })

    with pytest.raises(ValueError, match="reference policy"):
        ocstatplot.classify_policies_by_shortcut_rule(plot_df)


@pytest.mark.order(518)
def test_detect_x_break_splits_only_on_a_wide_empty_region():
    # one far-out control plus a dense cluster: worth breaking the axis
    left, right = ocstatplot._detect_x_break([0.164, 0.42, 0.45, 0.46, 0.49])
    assert left[1] < right[0]
    assert left[0] < 0.164 < left[1]
    assert right[0] < 0.42 and 0.49 < right[1]

    # evenly spread values: no gap wide enough to justify a break
    assert ocstatplot._detect_x_break([0.10, 0.20, 0.30, 0.40, 0.50]) is None


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


@pytest.mark.order(354)
def test_stats_pval_alias_and_consensus_include_branch(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(ocstatplot.sns, "barplot", lambda **_k: ocstatplot.plt.gca())

    gh = pd.DataFrame(
        {
            "A": ["m1", "m2"],
            "B": ["m2", "m1"],
            "diff": [0.2, -0.1],
            "pval_corr": [0.01, 0.20],
        }
    )

    # Covers p-value alias branch plus top_n=None path.
    ocstatplot.plot_bar_with_significance(gh, metric="AUC", output_dir=str(tmp_path), top_n=None)

    df = _toy_method_df()
    colours = {
        "NN_Method": "tab:blue",
        "XGB_Method": "tab:green",
        "Transformer_Method": "tab:red",
        "Mean consensus": "tab:purple",
    }

    # Covers show_simple_consensus=True path.
    ocstatplot.plot_boxplots(df, n_trials=2, colour_mapping=colours, output_dir=str(tmp_path), show_simple_consensus=True)
    assert any("games_howell_bar_AUC.png" in p for p in saved)
    assert any("boxplots_rmse_auc_2.png" in p for p in saved)


@pytest.mark.order(355)
def test_stats_diagnostics_and_qcut_error_fallbacks(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(
        ocstatplot.sstats,
        "shapiro",
        lambda _x: (_ for _ in ()).throw(TypeError("forced shapiro failure")),
    )
    monkeypatch.setattr(
        ocstatplot.sstats,
        "levene",
        lambda *_x: (_ for _ in ()).throw(TypeError("forced levene failure")),
    )

    df = _toy_method_df()
    ocstatplot.plot_normality_and_variance_diagnostics(df, metric="RMSE", n_trials=7, output_dir=str(tmp_path))
    assert any("diagnostics_RMSE_7.png" in p for p in saved)

    monkeypatch.setattr(
        ocstatplot.pd,
        "qcut",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("forced qcut failure")),
    )
    importance_var = pd.DataFrame({"Feature": ["f1", "f2", "f3", "f4"], "Importance": [0.1, 0.5, 0.2, 0.9]})
    ocstatplot.save_pca_importance_bins(importance_var, pca_type="2", n_trials=7, output_dir=str(tmp_path), n_bins=5)

    out_csv = tmp_path / "pca2_importance_bins_7.csv"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert "bin" in out_df.columns


@pytest.mark.order(356)
def test_stats_diagnostics_shapiro_exception_branch_with_valid_group_size(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ocstatplot.plt, "savefig", lambda path, **_k: saved.append(path))
    monkeypatch.setattr(
        ocstatplot.sstats,
        "shapiro",
        lambda _x: (_ for _ in ()).throw(ValueError("forced shapiro failure")),
    )

    # Ensure each methodology has >= 3 rows so the shapiro try/except block is executed.
    df = pd.DataFrame(
        {
            "Methodology": ["M1", "M1", "M1", "M2", "M2", "M2"],
            "RMSE": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "AUC": [0.6, 0.7, 0.8, 0.65, 0.75, 0.85],
        }
    )
    ocstatplot.plot_normality_and_variance_diagnostics(df, metric="RMSE", n_trials=8, output_dir=str(tmp_path))
    assert any("diagnostics_RMSE_8.png" in p for p in saved)
