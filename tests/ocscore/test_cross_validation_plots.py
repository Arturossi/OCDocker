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
    assert "values in CSV" in ax.get_title()
    plt.close(fig)


@pytest.mark.order(284)
def test_save_per_target_figures_writes_chunked_heatmaps_for_many_receptors(tmp_path):
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
        max_groups=34,
        heatmap_chunk_size=12,
    )

    chunk_keys = [key for key in written if "heatmap" in key and "part" in key]
    assert len(chunk_keys) == 3
    assert all(Path(written[key]).is_file() for key in chunk_keys)
    plt.close("all")


@pytest.mark.order(285)
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
    assert ax.collections
    plt.close(fig)


@pytest.mark.order(286)
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
