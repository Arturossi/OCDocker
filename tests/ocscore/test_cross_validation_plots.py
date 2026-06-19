#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for OCScore cross-validation plotting helpers.'''

# Imports
###############################################################################
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import OCDocker.OCScore.Analysis.Plotting.CrossValidationPlots as occvplot
import OCDocker.OCScore.Optimization.ModelCrossValidation as occv

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


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(277)
def test_save_cross_validation_figures_writes_pngs(tmp_path):
    fold_results = [
        occv.CrossValidationFoldResult(
            fold_index=0,
            n_train=10,
            n_validation=5,
            train_indices=[0, 1],
            validation_indices=[2, 3],
            validation_metrics={"BEDROC": 0.9, "ROC-AUC": 0.8},
            scoring_function_metrics={"vina_vina": {"BEDROC": 0.2, "ROC-AUC": 0.6}},
        ),
        occv.CrossValidationFoldResult(
            fold_index=1,
            n_train=10,
            n_validation=5,
            train_indices=[4, 5],
            validation_indices=[6, 7],
            validation_metrics={"BEDROC": 0.7, "ROC-AUC": 0.85},
            scoring_function_metrics={"vina_vina": {"BEDROC": 0.8, "ROC-AUC": 0.5}},
        ),
    ]
    result = occv.CrossValidationResult(
        export_dir=str(tmp_path / "export"),
        task="dudez_screening",
        n_folds=2,
        effective_folds=2,
        strategy="receptor_grouped",
        epochs=1,
        random_seed=0,
        objective_metric="BEDROC",
        fold_results=fold_results,
        aggregate_validation_metrics={},
        model_config={},
        scoring_function_columns=["vina_vina"],
    )
    result.scorer_comparison_summary = occv.build_scorer_comparison_summary(
        result,
        comparison_metrics=["BEDROC", "ROC-AUC"],
    )
    cv_dir = tmp_path / "cv"
    occv.save_cross_validation_result(result, cv_dir)

    figures_dir = tmp_path / "figures"
    written = occvplot.save_cross_validation_figures(cv_dir, figures_dir=figures_dir, metrics=["BEDROC"])
    assert written
    for path in written.values():
        assert path.endswith(".png")
        from pathlib import Path

        assert Path(path).is_file()

    payload = json.loads((cv_dir / occvplot.RESULTS_JSON_NAME).read_text(encoding="utf-8"))
    assert payload["task"] == "dudez_screening"
    plt.close("all")


@pytest.mark.order(278)
def test_save_cross_validation_figures_skips_all_nan_fold_metric(tmp_path):
    cv_dir = tmp_path / "cv"
    cv_dir.mkdir()
    (cv_dir / occvplot.RESULTS_JSON_NAME).write_text(
        json.dumps({
            "objective_metric": "BEDROC",
            "scorer_comparison_summary": {"comparison_metrics": ["BEDROC"]},
        }),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"scorer": "OCScore", "fold_index": 0, "validation_BEDROC": float("nan")}],
    ).to_csv(cv_dir / occvplot.FOLD_COMPARISON_CSV_NAME, index=False)

    written = occvplot.save_cross_validation_figures(
        cv_dir,
        figures_dir=tmp_path / "figures",
        metrics=["BEDROC"],
    )

    assert written == {}
    plt.close("all")


@pytest.mark.order(279)
def test_plot_mean_std_bars_returns_axes():
    mean_std = pd.DataFrame(
        [
            {"scorer": "OCScore", "metric": "BEDROC", "mean": 0.8, "std": 0.05},
            {"scorer": "vina_vina", "metric": "BEDROC", "mean": 0.5, "std": 0.1},
        ]
    )
    fig, ax = occvplot.plot_mean_std_bars(mean_std, "BEDROC", top_n=10)
    assert ax.get_title().startswith("Cross-validation")
    assert ax.get_xlim()[0] == 0.0
    plt.close(fig)


@pytest.mark.order(280)
def test_resolve_cross_validation_dir_from_export_parent(tmp_path):
    cv_dir = tmp_path / "cross_validation"
    cv_dir.mkdir()
    (cv_dir / occvplot.RESULTS_JSON_NAME).write_text("{}", encoding="utf-8")
    assert occvplot.resolve_cross_validation_dir(cv_dir) == cv_dir.resolve()
    assert occvplot.resolve_cross_validation_dir(tmp_path) == cv_dir.resolve()


@pytest.mark.order(281)
def test_save_baseline_comparison_figures_writes_png(tmp_path):
    csv_path = tmp_path / "dudez_sf_baseline_comparison.csv"
    pd.DataFrame(
        [
            {"scorer": "OCScore", "scorer_type": "model", "split": "test", "BEDROC": 0.8},
            {"scorer": "vina_vina", "scorer_type": "sf", "split": "test", "BEDROC": 0.3},
        ]
    ).to_csv(csv_path, index=False)
    written = occvplot.save_baseline_comparison_figures(
        csv_path,
        figures_dir=tmp_path / "figures",
        metrics=["BEDROC"],
    )
    assert written
    assert any(Path(path).is_file() for path in written.values())
    plt.close("all")


@pytest.mark.order(282)
def test_save_per_target_figures_writes_png(tmp_path):
    per_target = pd.DataFrame(
        [
            {"split": "test", "group": "r1", "scorer": "OCScore", "scorer_type": "model", "BEDROC": 0.9},
            {"split": "test", "group": "r1", "scorer": "vina_vina", "scorer_type": "sf", "BEDROC": 0.2},
            {"split": "test", "group": "r2", "scorer": "OCScore", "scorer_type": "model", "BEDROC": 0.7},
            {"split": "test", "group": "r2", "scorer": "vina_vina", "scorer_type": "sf", "BEDROC": 0.5},
        ]
    )
    written = occvplot.save_per_target_figures(
        per_target,
        tmp_path / "figures",
        split="test",
        metrics=["BEDROC"],
    )
    assert written
    assert any("heatmap" in key for key in written)
    plt.close("all")


@pytest.mark.order(283)
def test_large_per_target_heatmap_disables_annotations_and_expands_canvas():
    rows = []
    groups = [f"r{i:02d}" for i in range(30)]
    scorers = ["OCScore", *[f"sf_{i:02d}" for i in range(19)]]
    for scorer_idx, scorer in enumerate(scorers):
        for group_idx, group in enumerate(groups):
            rows.append(
                {
                    "split": "validation",
                    "group": group,
                    "scorer": scorer,
                    "scorer_type": "model" if scorer == "OCScore" else "sf",
                    "ROC-AUC": 0.5 + 0.01 * ((scorer_idx + group_idx) % 20),
                }
            )
    per_target = pd.DataFrame(rows)

    fig, ax = occvplot.plot_per_target_heatmap(
        per_target,
        "ROC-AUC",
        split="validation",
        top_n=25,
        max_groups=None,
    )

    assert fig.get_size_inches()[0] > 12
    assert len(ax.texts) == 0
    assert "color scale only" in ax.get_title()
    plt.close(fig)


@pytest.mark.order(284)
def test_per_target_heatmap_transposes_when_fewer_receptors_than_scorers():
    rows = []
    groups = [f"r{i:02d}" for i in range(8)]
    scorers = ["OCScore", *[f"sf_{i:02d}" for i in range(11)]]
    for scorer in scorers:
        for group in groups:
            rows.append(
                {
                    "split": "validation",
                    "group": group,
                    "scorer": scorer,
                    "scorer_type": "model" if scorer == "OCScore" else "sf",
                    "BEDROC": 0.5,
                }
            )
    per_target = pd.DataFrame(rows)

    fig, ax = occvplot.plot_per_target_heatmap(
        per_target,
        "BEDROC",
        split="validation",
        top_n=12,
        max_groups=None,
    )

    assert ax.get_ylabel() == "Receptor"
    assert ax.get_xlabel() == "Scorer"
    plt.close(fig)


@pytest.mark.order(285)
def test_save_per_target_figures_writes_single_heatmap_for_many_receptors(tmp_path):
    rows = []
    groups = [f"r{i:02d}" for i in range(34)]
    scorers = ["OCScore", *[f"sf_{i:02d}" for i in range(8)]]
    for scorer_idx, scorer in enumerate(scorers):
        for group_idx, group in enumerate(groups):
            rows.append(
                {
                    "split": "validation",
                    "group": group,
                    "scorer": scorer,
                    "scorer_type": "model" if scorer == "OCScore" else "sf",
                    "NDCG@1%": 0.1 + 0.02 * ((scorer_idx + group_idx) % 25),
                }
            )
    per_target = pd.DataFrame(rows)

    written = occvplot.save_per_target_figures(
        per_target,
        tmp_path / "figures",
        split="validation",
        metrics=["NDCG@1%"],
        top_n=12,
    )

    assert "per_target_heatmap_validation_NDCG@1%" in written
    assert Path(written["per_target_heatmap_validation_NDCG@1%"]).is_file()
    assert not any("part" in key for key in written if "heatmap" in key)
    assert list((tmp_path / "figures").glob("*_heatmap_part*.png")) == []
    plt.close("all")


@pytest.mark.order(286)
def test_per_target_boxplot_is_horizontal_with_point_overlay():
    rows = []
    scorers = ["OCScore", "sf_a", "sf_b"]
    for scorer_idx, scorer in enumerate(scorers):
        for group_idx in range(8):
            rows.append(
                {
                    "split": "validation",
                    "group": f"r{group_idx}",
                    "scorer": scorer,
                    "scorer_type": "model" if scorer == "OCScore" else "sf",
                    "EF1%": float((group_idx + scorer_idx) % 4),
                }
            )
    per_target = pd.DataFrame(rows)

    fig, ax = occvplot.plot_per_target_boxplot(
        per_target,
        "EF1%",
        split="validation",
        top_n=3,
    )

    assert ax.get_xlabel() == "EF1%"
    assert ax.get_ylabel() == "Scorer"
    assert [label.get_text() for label in ax.get_yticklabels()] == scorers
    assert ax.patches
    plt.close(fig)


@pytest.mark.order(287)
def test_resolve_plot_metrics_excludes_confusion_counts():
    results = {
        "objective_metric": "BEDROC",
        "scorer_comparison_summary": {
            "comparison_metrics": ["BEDROC", "TN", "FP"],
        },
    }
    mean_std = pd.DataFrame(
        [
            {"scorer": "OCScore", "metric": "BEDROC", "mean": 0.5, "std": 0.1},
            {"scorer": "OCScore", "metric": "TN", "mean": 100.0, "std": 10.0},
        ]
    )
    metrics = occvplot._resolve_plot_metrics(None, results, mean_std)
    assert metrics == ["BEDROC"]
    assert occvplot._resolve_plot_metrics(["TN"], results, mean_std) == ["TN"]


@pytest.mark.order(290)
def test_large_per_target_heatmap_uses_scaled_typography():
    rows = []
    groups = [f"r{i:02d}" for i in range(43)]
    scorers = ["OCScore", *[f"sf_{i:02d}" for i in range(24)]]
    for scorer in scorers:
        for group in groups:
            rows.append(
                {
                    "split": "validation",
                    "group": group,
                    "scorer": scorer,
                    "scorer_type": "model" if scorer == "OCScore" else "sf",
                    "EF1%": 0.5,
                }
            )
    per_target = pd.DataFrame(rows)

    fig, ax = occvplot.plot_per_target_heatmap(
        per_target,
        "EF1%",
        split="validation",
        top_n=25,
        max_groups=None,
    )

    assert fig.get_size_inches()[0] > 20.0
    y_font = ax.get_yticklabels()[0].get_fontsize()
    x_font = ax.get_xticklabels()[0].get_fontsize()
    assert y_font >= 16
    assert x_font >= 15
    n_rows = 25
    n_cols = 43
    axes_width = ax.get_position().width * fig.get_figwidth()
    axes_height = ax.get_position().height * fig.get_figheight()
    cell_width = axes_width / n_cols
    cell_height = axes_height / n_rows
    assert abs(cell_width - cell_height) / max(cell_width, cell_height) < 0.08
    plt.close(fig)


@pytest.mark.order(289)
def test_plot_fold_metric_bars_groups_folds_and_summaries_with_external_legend():
    fold_comparison = pd.DataFrame(
        [
            {"scorer": "OCScore", "fold_index": 0, "validation_BEDROC": 0.5},
            {"scorer": "OCScore", "fold_index": 1, "validation_BEDROC": 0.6},
            {"scorer": "OCScore", "fold_index": 2, "validation_BEDROC": 0.55},
            {"scorer": "vina_vina", "fold_index": 0, "validation_BEDROC": 0.2},
            {"scorer": "vina_vina", "fold_index": 1, "validation_BEDROC": 0.25},
            {"scorer": "vina_vina", "fold_index": 2, "validation_BEDROC": 0.22},
        ]
    )
    fig, ax = occvplot.plot_fold_metric_bars(fold_comparison, "BEDROC", top_n=2)
    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert tick_labels == ["0", "1", "2", "Mean"]
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_bbox_to_anchor().x0 > 1.0
    assert len(ax.patches) == 2 * 4
    assert len(ax.lines) >= 2
    plt.close(fig)


@pytest.mark.order(290)
def test_fold_comparison_scorer_selection_excludes_desc_and_pins_sf_max():
    fold_comparison = pd.DataFrame(
        [
            {"scorer": "OCScore", "scorer_type": "model", "fold_index": 0, "validation_BEDROC": 0.9},
            {"scorer": "OCScore", "scorer_type": "model", "fold_index": 1, "validation_BEDROC": 0.8},
            {"scorer": "desc_min", "scorer_type": "descriptor_aggregate", "fold_index": 0, "validation_BEDROC": 0.85},
            {"scorer": "desc_min", "scorer_type": "descriptor_aggregate", "fold_index": 1, "validation_BEDROC": 0.84},
            {"scorer": "desc_max", "scorer_type": "descriptor_aggregate", "fold_index": 0, "validation_BEDROC": 0.83},
            {"scorer": "desc_max", "scorer_type": "descriptor_aggregate", "fold_index": 1, "validation_BEDROC": 0.82},
            {"scorer": "vina_vina", "scorer_type": "sf", "fold_index": 0, "validation_BEDROC": 0.4},
            {"scorer": "vina_vina", "scorer_type": "sf", "fold_index": 1, "validation_BEDROC": 0.35},
            {"scorer": "plants_plp", "scorer_type": "sf", "fold_index": 0, "validation_BEDROC": 0.3},
            {"scorer": "plants_plp", "scorer_type": "sf", "fold_index": 1, "validation_BEDROC": 0.25},
            {"scorer": "sf_max", "scorer_type": "sf_consensus", "fold_index": 0, "validation_BEDROC": 0.05},
            {"scorer": "sf_max", "scorer_type": "sf_consensus", "fold_index": 1, "validation_BEDROC": 0.06},
        ]
    )
    fig, ax = occvplot.plot_fold_metric_bars(fold_comparison, "BEDROC", top_n=3)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "desc_min" not in legend_labels
    assert "desc_max" not in legend_labels
    assert "sf_max" in legend_labels
    plt.close(fig)


@pytest.mark.order(292)
def test_save_cross_validation_figures_writes_fold_comparison_plot_names(tmp_path):
    fold_comparison = pd.DataFrame(
        [
            {"scorer": "OCScore", "scorer_type": "model", "fold_index": 0, "validation_BEDROC": 0.5},
            {"scorer": "OCScore", "scorer_type": "model", "fold_index": 1, "validation_BEDROC": 0.6},
            {"scorer": "vina_vina", "scorer_type": "sf", "fold_index": 0, "validation_BEDROC": 0.2},
            {"scorer": "vina_vina", "scorer_type": "sf", "fold_index": 1, "validation_BEDROC": 0.25},
        ]
    )
    cv_dir = tmp_path / "cv"
    cv_dir.mkdir()
    fold_comparison.to_csv(cv_dir / occvplot.FOLD_COMPARISON_CSV_NAME, index=False)
    (cv_dir / occvplot.RESULTS_JSON_NAME).write_text(
        json.dumps({"objective_metric": "BEDROC", "scorer_comparison_summary": {"comparison_metrics": ["BEDROC"]}}),
        encoding="utf-8",
    )

    figures_dir = tmp_path / "figures"
    legacy_path = figures_dir / "cv_fold_lines_BEDROC.png"
    figures_dir.mkdir()
    legacy_path.write_bytes(b"legacy")

    written = occvplot.save_cross_validation_figures(cv_dir, figures_dir=figures_dir, metrics=["BEDROC"])
    assert "fold_comparison_BEDROC" in written
    assert written["fold_comparison_BEDROC"].endswith("cv_fold_comparison_BEDROC.png")
    assert not legacy_path.exists()
    plt.close("all")


@pytest.mark.order(291)
def test_plot_ocscore_wins_uses_external_legend():
    ocscore_wins = pd.DataFrame(
        {
            "metric": ["BEDROC", "ROC-AUC", "TN", "FN"],
            "n_folds_won": [5, 5, 3, 0],
            "n_folds_compared": [5, 5, 5, 5],
        }
    )
    fig, ax = occvplot.plot_ocscore_wins(ocscore_wins)
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_bbox_to_anchor().x0 > 1.0
    plt.close(fig)


@pytest.mark.order(288)
def test_aggregate_cv_per_target_metrics_collapses_folds():
    per_target = pd.DataFrame(
        [
            {"fold_index": 0, "group": "r1", "scorer": "OCScore", "scorer_type": "model", "BEDROC": 0.8},
            {"fold_index": 1, "group": "r1", "scorer": "OCScore", "scorer_type": "model", "BEDROC": 0.6},
        ]
    )
    aggregated = occvplot.aggregate_cv_per_target_metrics(per_target)
    assert len(aggregated) == 1
    assert aggregated.iloc[0]["BEDROC"] == pytest.approx(0.7)
    assert aggregated.iloc[0]["split"] == "validation"
