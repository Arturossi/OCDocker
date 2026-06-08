#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for production-grade DUDEz baseline evaluation."""

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

from OCDocker.OCScore.Analysis.ProductionBaselines import (
    ProductionBaselineConfig,
    TrainOnlyFitError,
    aggregate_baseline_rows,
    evaluate_learned_sf_baselines,
    validate_fit_uses_train_only,
    write_production_baseline_reports,
)


# License
###############################################################################
"""
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
"""


# Functions
###############################################################################
## Public ##


def _synthetic_dudez_frame(n_rows: int = 120, n_receptors: int = 6) -> tuple[pd.DataFrame, list[str], dict[str, list[int]]]:
    rng = np.random.default_rng(0)
    receptors = np.array([f"r{i % n_receptors}" for i in range(n_rows)])
    labels = np.array([1 if i % 2 == 0 else 0 for i in range(n_rows)], dtype=int)
    split_indices = {
        "train": list(range(0, 80)),
        "validation": list(range(80, 100)),
        "test": list(range(100, n_rows)),
    }
    selected_features = ["vina_vina", "gnina_score", "f0", "f1"]
    df = pd.DataFrame(
        {
            "receptor": receptors,
            "label": labels,
            "kind": np.where(labels == 1, "ligands", "decoys"),
            "vina_vina": rng.normal(size=n_rows),
            "gnina_score": rng.normal(size=n_rows),
            "f0": rng.normal(size=n_rows),
            "f1": rng.normal(size=n_rows),
        }
    )
    return df, selected_features, split_indices


@pytest.mark.order(289)
def test_validate_fit_uses_train_only_rejects_val_leakage():
    train_idx = np.arange(80)
    fit_idx = np.array([0, 1, 80, 81])
    with pytest.raises(TrainOnlyFitError):
        validate_fit_uses_train_only(train_idx, fit_idx)


@pytest.mark.order(290)
def test_learned_sf_baselines_use_train_indices_and_return_finite_metrics():
    df, selected_features, split_indices = _synthetic_dudez_frame()
    config = ProductionBaselineConfig(include_xgb=False, include_lgbm=False, random_seed=0)
    rows, _skip_notes = evaluate_learned_sf_baselines(
        df,
        selected_features,
        split_indices,
        label_column="label",
        group_column="receptor",
        config=config,
    )
    assert rows
    lr_rows = [row for row in rows if row["baseline"] == "lr_sf"]
    assert lr_rows
    for row in lr_rows:
        assert np.isfinite(row["ROC-AUC"])
        assert row["split"] in {"validation", "test"}


@pytest.mark.order(291)
def test_learned_sf_baselines_fail_when_fit_indices_include_validation():
    df, selected_features, split_indices = _synthetic_dudez_frame()
    config = ProductionBaselineConfig(include_xgb=False, include_lgbm=False, random_seed=0)
    leaked_fit = np.array(split_indices["train"] + split_indices["validation"], dtype=int)
    with pytest.raises(TrainOnlyFitError):
        evaluate_learned_sf_baselines(
            df,
            selected_features,
            split_indices,
            label_column="label",
            group_column="receptor",
            config=config,
            fit_indices=leaked_fit,
        )


@pytest.mark.order(292)
def test_shuffled_label_control_yields_chance_roc_auc():
    df, selected_features, split_indices = _synthetic_dudez_frame(n_rows=200, n_receptors=10)
    config = ProductionBaselineConfig(
        include_xgb=False,
        include_lgbm=False,
        include_shuffle_control=True,
        random_seed=0,
    )
    rows, _skip_notes = evaluate_learned_sf_baselines(
        df,
        selected_features,
        split_indices,
        label_column="label",
        group_column="receptor",
        config=config,
        shuffle_train_labels=True,
    )
    test_row = next(row for row in rows if row["baseline"] == "shuffled_lr_sf" and row["split"] == "test")
    assert 0.35 <= test_row["ROC-AUC"] <= 0.65


@pytest.mark.order(293)
def test_optional_xgb_skipped_without_import(monkeypatch):
    df, selected_features, split_indices = _synthetic_dudez_frame()
    config = ProductionBaselineConfig(include_xgb=True, include_lgbm=False, random_seed=0)

    def _raise_import(_name, _seed):
        return None, "xgboost not installed"

    monkeypatch.setattr(
        "OCDocker.OCScore.Analysis.ProductionBaselines._optional_estimator",
        _raise_import,
    )
    _rows, skip_notes = evaluate_learned_sf_baselines(
        df,
        selected_features,
        split_indices,
        label_column="label",
        group_column="receptor",
        config=config,
    )
    assert any(note.get("baseline") == "xgb_sf" for note in skip_notes)


@pytest.mark.order(294)
def test_aggregate_baseline_rows_reports_replica_count():
    rows = [
        {"replica": "replica_000", "baseline": "lr_sf", "baseline_family": "learned_sf", "split": "test", "BEDROC": 0.4},
        {"replica": "replica_001", "baseline": "lr_sf", "baseline_family": "learned_sf", "split": "test", "BEDROC": 0.6},
    ]
    summary = aggregate_baseline_rows(rows)
    assert int(summary.loc[0, "n_replicas"]) == 2
    assert summary.loc[0, "BEDROC"] == pytest.approx(0.5)

@pytest.mark.order(295)
def test_write_production_baseline_reports_writes_csvs(tmp_path):
    payloads = [
        {
            "rows": [
                {
                    "replica": "replica_000",
                    "baseline": "lr_sf",
                    "baseline_family": "learned_sf",
                    "split": "test",
                    "BEDROC": 0.5,
                }
            ],
            "skip_notes": [],
        }
    ]

    paths = write_production_baseline_reports(tmp_path, payloads)

    assert (tmp_path / "baselines_per_fold.csv").is_file()
    assert (tmp_path / "baselines_summary.csv").is_file()
    assert (tmp_path / "baselines_rank_table.csv").is_file()
    assert paths["baselines_per_fold_csv"].endswith("baselines_per_fold.csv")

