#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for descriptor and SF-consensus baseline scorers."""

# Imports
###############################################################################
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import (
    DESCRIPTOR_AGGREGATE_SCORER_TYPE,
    SF_CONSENSUS_SCORER_TYPE,
    evaluate_descriptor_aggregate_baselines_on_fold,
    evaluate_sf_consensus_baselines_on_fold,
    format_descriptor_aggregate_scorer,
    format_sf_consensus_scorer,
    row_aggregate_feature_scores,
    row_aggregate_sf_scores,
    scorer_type_for_baseline_name,
)

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Public ##


@pytest.mark.order(277)
def test_row_aggregate_feature_scores_computes_reducers():
    matrix = np.array([[1.0, 3.0], [2.0, 8.0]], dtype=np.float32)
    scores = row_aggregate_feature_scores(matrix)
    assert scores["mean"].tolist() == pytest.approx([2.0, 5.0])
    assert scores["median"].tolist() == pytest.approx([2.0, 5.0])
    assert scores["max"].tolist() == pytest.approx([3.0, 8.0])
    assert scores["min"].tolist() == pytest.approx([1.0, 2.0])


@pytest.mark.order(278)
def test_row_aggregate_sf_scores_uses_only_sf_columns():
    df = pd.DataFrame(
        {
            "f0": [0.0, 100.0],
            "f1": [0.0, 100.0],
            "vina_vina": [1.0, 3.0],
            "gnina_score": [2.0, 4.0],
        }
    )
    sf_only = row_aggregate_sf_scores(df, ["vina_vina", "gnina_score"])
    assert sf_only["mean"].tolist() == pytest.approx([1.5, 3.5])


@pytest.mark.order(279)
def test_row_aggregate_sf_scores_uses_positional_row_indices():
    df = pd.DataFrame(
        {
            "vina_vina": [1.0, 10.0, 100.0],
            "gnina_score": [3.0, 30.0, 300.0],
        },
        index=[3053, 155, 653],
    )
    subset = row_aggregate_sf_scores(df, ["vina_vina", "gnina_score"], np.array([0, 2], dtype=np.int64))
    assert subset["mean"].tolist() == pytest.approx([2.0, 200.0])


@pytest.mark.order(280)
def test_scorer_type_for_baseline_name():
    assert scorer_type_for_baseline_name("desc_mean") == DESCRIPTOR_AGGREGATE_SCORER_TYPE
    assert scorer_type_for_baseline_name("sf_median") == SF_CONSENSUS_SCORER_TYPE
    assert scorer_type_for_baseline_name("vina_vina") is None


@pytest.mark.order(281)
def test_evaluate_baseline_families_return_prefixed_scorer_keys():
    rng = np.random.default_rng(0)
    n_rows = 40
    feature_matrix = rng.normal(size=(n_rows, 6)).astype(np.float32)
    labels = np.array([1, 0] * (n_rows // 2), dtype=int)
    groups = np.array([f"r{i // 10}" for i in range(n_rows)])
    val_idx = np.arange(20, n_rows)
    dudez_df = pd.DataFrame(
        {
            **{f"f{i}": feature_matrix[:, i] for i in range(6)},
            "vina_vina": rng.normal(size=n_rows),
            "gnina_score": rng.normal(size=n_rows),
        }
    )

    def _always_higher(scores, label_array):
        return True

    desc_metrics = evaluate_descriptor_aggregate_baselines_on_fold(
        feature_matrix,
        val_idx,
        labels,
        groups,
        metric_names=("BEDROC", "ROC-AUC"),
        infer_higher_is_better=_always_higher,
    )
    sf_metrics = evaluate_sf_consensus_baselines_on_fold(
        dudez_df,
        val_idx,
        ["vina_vina", "gnina_score"],
        labels,
        groups,
        metric_names=("BEDROC", "ROC-AUC"),
        infer_higher_is_better=_always_higher,
    )
    assert format_descriptor_aggregate_scorer("mean") in desc_metrics
    assert format_sf_consensus_scorer("mean") in sf_metrics
    assert format_descriptor_aggregate_scorer("mean") != format_sf_consensus_scorer("mean")


@pytest.mark.order(282)
def test_build_comparison_table_includes_both_baseline_families(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    module_path = (
        Path(__file__).resolve().parents[2] / "examples" / "17_ocscore_dudez_sf_baseline_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("ex19_baseline", module_path)
    ex19 = importlib.util.module_from_spec(spec)
    sys.modules["ex19_baseline"] = ex19
    spec.loader.exec_module(ex19)

    n_rows = 12
    features = ["f0", "f1", "vina_vina"]
    rng = np.random.default_rng(1)
    dudez_df = pd.DataFrame(
        {
            "f0": rng.normal(size=n_rows),
            "f1": rng.normal(size=n_rows),
            "vina_vina": rng.normal(size=n_rows),
            "gnina_score": rng.normal(size=n_rows),
            "receptor": [f"r{i // 3}" for i in range(n_rows)],
            "kind": ["ligands" if i % 2 == 0 else "decoys" for i in range(n_rows)],
            "dataset": ["dudez"] * n_rows,
        }
    )
    bundle = {
        "model": SimpleNamespace(),
        "device": "cpu",
        "scaler": None,
        "split_indices": {
            "validation_indices": np.arange(6),
            "test_indices": np.arange(6, 12),
        },
        "summary": {"trial_number": 0},
    }

    monkeypatch.setattr(ex19.ocexport, "load_exported_model", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        ex19,
        "predict_ocscore_logits",
        lambda model, device, features: np.zeros(len(features)),
    )

    table, _calibrator, _logits, _labels = ex19.build_comparison_table(
        dudez_df=dudez_df,
        selected_features=features,
        export_dir=tmp_path,
        group_column="receptor",
        kind_column="kind",
        device="cpu",
        include_export_summary=False,
        calibration_method="none",
    )
    desc_rows = table[table["scorer_type"] == DESCRIPTOR_AGGREGATE_SCORER_TYPE]
    sf_rows = table[table["scorer_type"] == SF_CONSENSUS_SCORER_TYPE]
    assert set(desc_rows["scorer"]) >= {"desc_mean", "desc_median", "desc_max", "desc_min"}
    assert set(sf_rows["scorer"]) >= {"sf_mean", "sf_median", "sf_max", "sf_min"}
