#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for exported-model cross-validation."""

# Imports
###############################################################################
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import OCDocker.OCScore.Optimization.ModelCrossValidation as occv
import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

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

@pytest.mark.order(270)
def test_validate_fold_indices_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        occv.validate_fold_indices(np.array([0, 1, 2]), np.array([2, 3]), fold_index=0)


@pytest.mark.order(270)
def test_validate_receptor_group_split_rejects_shared_receptor():
    groups = np.array(["r1", "r1", "r2", "r2"])
    with pytest.raises(ValueError, match="share receptor"):
        occv.validate_receptor_group_split(
            groups,
            np.array([0, 1]),
            np.array([0, 2]),
            fold_index=1,
        )


@pytest.mark.order(270)
def test_diagnose_entity_overlap_reports_shared_name():
    df = pd.DataFrame(
        {
            "name": ["lig_a", "lig_b", "lig_a", "lig_c"],
            "receptor": ["r1", "r1", "r2", "r2"],
        }
    )
    overlap = occv.diagnose_entity_overlap(
        df,
        np.array([0, 1]),
        np.array([2, 3]),
        ("name",),
    )
    assert overlap["name"]["count"] == 1
    assert overlap["name"]["examples"] == ["lig_a"]


@pytest.mark.order(271)
@pytest.mark.order(270)
def test_cv_calibration_metrics_skip_one_class_fit_split():
    val_metrics: dict[str, float] = {}

    diagnostics = occv._merge_cv_calibration_metrics(
        val_metrics,
        train_true=np.array([0, 0, 0, 0], dtype=int),
        train_score=np.array([-2.0, -1.0, -0.5, -0.2], dtype=float),
        val_true=np.array([0, 1, 0, 1], dtype=int),
        val_score=np.array([-1.0, 1.0, -0.5, 0.5], dtype=float),
        method="platt",
    )

    assert diagnostics["calibration_status"] == "skipped"
    assert diagnostics["calibration_skip_reason"] == "fit_split_one_class"
    assert val_metrics["calibration_fit_valid"] == 0.0
    assert "Brier" in val_metrics
    assert "Brier_calibrated" not in val_metrics


def test_iter_receptor_group_kfold_indices_hold_out_whole_receptors():
    groups = np.array(["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"])
    folds = occv.iter_receptor_group_kfold_indices(groups, n_folds=2, random_seed=0, shuffle=True)
    assert len(folds) == 2
    for train_idx, val_idx in folds:
        train_receptors = set(groups[train_idx])
        val_receptors = set(groups[val_idx])
        assert train_receptors.isdisjoint(val_receptors)
        assert len(train_idx) + len(val_idx) == len(groups)


@pytest.mark.order(272)
def test_run_cross_validation_from_export_pdbbind(tmp_path):
    input_size = 6
    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [8, 4],
        "encoder_latent_dim": 3,
        "encoder_depth": 2,
        "encoder_is_monotonic": True,
        "projection_dim": 0,
        "encoder_activation": "GELU",
        "encoder_dropout": 0.0,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 4,
        "decoder_hidden_sizes": [],
        "decoder_depth": 0,
        "decoder_lambda_rec": 0.0,
        "dae_noise_type": "none",
        "dae_mask_prob": 0.0,
        "dae_gaussian_std": 0.0,
        "pdbbind_regression_loss": "mse",
        "pdbbind_huber_delta": 1.0,
    }
    model = ocstaged.build_pdbbind_model(input_size=input_size, params=params)
    features = [f"f{i}" for i in range(input_size)]
    rng = np.random.default_rng(0)
    n_rows = 24
    df = pd.DataFrame(rng.normal(size=(n_rows, input_size)), columns=features)
    df["experimental"] = rng.normal(size=n_rows)
    splits = {
        "X_train": rng.normal(size=(12, input_size)).astype(np.float32),
        "y_train": rng.normal(size=12).astype(np.float32),
        "X_val": rng.normal(size=(6, input_size)).astype(np.float32),
        "y_val": rng.normal(size=6).astype(np.float32),
        "X_test": rng.normal(size=(6, input_size)).astype(np.float32),
        "y_test": rng.normal(size=6).astype(np.float32),
        "train_indices": np.arange(12),
        "validation_indices": np.arange(12, 18),
        "test_indices": np.arange(18, 24),
        "split_config": {"target_column": "experimental"},
        "split_diagnostics": {},
    }
    export_dir = tmp_path / "best_model"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="pdbbind_regression",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=0,
        best_objective_value=1.0,
        validation_metrics={"RMSE": 1.0},
        test_metrics={"RMSE": 1.1},
        stage_config={},
        splits=splits,
        objective_metric="RMSE",
        direction="minimize",
        best_params=params,
        validate=False,
    )

    result = occv.run_cross_validation_from_export(
        export_dir,
        df,
        config=occv.CrossValidationConfig(n_folds=3, epochs=2, random_seed=0),
        device="cpu",
        output_dir=tmp_path / "cv",
    )
    assert result.task == "pdbbind_regression"
    assert result.effective_folds == 3
    assert len(result.fold_results) == 3
    assert "RMSE" in result.aggregate_validation_metrics
    assert (tmp_path / "cv" / "cross_validation_results.json").exists()
    payload = json.loads((tmp_path / "cv" / "cross_validation_results.json").read_text(encoding="utf-8"))
    assert payload["effective_folds"] == 3


@pytest.mark.order(273)
def test_run_cross_validation_from_export_dudez_receptor_grouped(tmp_path):
    base_input_size = 8
    params = {
        "dudez_use_transfer": False,
        "dudez_fine_tuning_mode": "full",
        "dudez_num_unfrozen_layers": 1,
        "dudez_classifier_hidden_size": 4,
        "dudez_classifier_dropout": 0.0,
        "dudez_classifier_activation": "GELU",
        "dudez_use_class_weighting": False,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 8,
    }
    features = [f"f{i}" for i in range(base_input_size)] + ["vina_vina"]
    input_size = len(features)
    architecture = {
        "hidden_sizes": [6, 4],
        "latent_dim": 3,
        "projection_dim": 0,
    }
    model = ocstaged.build_dudez_model(
        input_size=input_size,
        params=params,
        feature_extractor_architecture=architecture,
    )
    receptors = ["r1", "r2", "r3", "r4"]
    rows = []
    rng = np.random.default_rng(1)
    for receptor in receptors:
        for label, kind in ((1, "ligands"), (0, "decoys")):
            for _ in range(6):
                row = {feature: float(rng.normal()) for feature in features if feature != "vina_vina"}
                row["vina_vina"] = float(rng.normal()) - float(label)
                row["receptor"] = receptor
                row["kind"] = kind
                rows.append(row)
    df = pd.DataFrame(rows)

    splits = {
        "X_train": rng.normal(size=(24, input_size)).astype(np.float32),
        "y_train": np.array([1, 0] * 12, dtype=np.float32),
        "X_val": rng.normal(size=(12, input_size)).astype(np.float32),
        "y_val": np.array([1, 0] * 6, dtype=np.float32),
        "X_test": rng.normal(size=(12, input_size)).astype(np.float32),
        "y_test": np.array([1, 0] * 6, dtype=np.float32),
        "train_indices": np.arange(24),
        "validation_indices": np.arange(24, 36),
        "test_indices": np.arange(36, 48),
        "split_config": {"strategy": "receptor_heldout_complete"},
        "split_diagnostics": {},
    }
    export_dir = tmp_path / "dudez_best_model"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="dudez_screening",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=0,
        best_objective_value=0.5,
        validation_metrics={"BEDROC": 0.5, "ranking_metrics_valid": 1.0},
        test_metrics={"BEDROC": 0.4, "ranking_metrics_valid": 1.0},
        stage_config={"kind_column": "kind", "target_group_column": "receptor"},
        splits=splits,
        objective_metric="BEDROC",
        direction="maximize",
        best_params=params,
        validate=False,
    )

    result = occv.run_cross_validation_from_export(
        export_dir,
        df,
        config=occv.CrossValidationConfig(
            n_folds=4,
            epochs=2,
            random_seed=0,
            strategy="receptor_grouped",
        ),
        device="cpu",
    )
    assert result.strategy == "receptor_grouped"
    assert result.effective_folds == 4
    assert len(result.fold_results) == 4
    assert "vina_vina" in result.scoring_function_columns
    for fold in result.fold_results:
        assert fold.validation_metrics.get("ranking_metrics_valid") == 1.0
        assert "vina_vina" in fold.scoring_function_metrics
    assert "vina_vina" in result.aggregate_scoring_function_metrics

    assert result.scorer_comparison_summary
    bedroc_wins = next(
        row for row in result.scorer_comparison_summary["ocscore_wins"] if row["metric"] == "BEDROC"
    )
    assert bedroc_wins["n_folds_compared"] == 4
    assert len(result.scorer_comparison_summary["fold_rankings"]) >= 4

    groups_all = df["receptor"].to_numpy()
    for fold in result.fold_results:
        train_idx = np.asarray(fold.train_indices, dtype=np.int64)
        val_idx = np.asarray(fold.validation_indices, dtype=np.int64)
        assert set(train_idx).isdisjoint(set(val_idx))
        train_receptors = set(groups_all[train_idx])
        val_receptors = set(groups_all[val_idx])
        assert train_receptors.isdisjoint(val_receptors)
        assert fold.per_target_metrics
        oc_rows = [
            row for row in fold.per_target_metrics
            if row["scorer"] == occv.OCSCORE_MODEL_SCORER_NAME
        ]
        assert oc_rows
        assert {row["group"] for row in oc_rows} == val_receptors

    paths = occv.save_cross_validation_result(result, tmp_path / "cv_out")
    assert (tmp_path / "cv_out" / "cross_validation_fold_comparison.csv").exists()
    assert (tmp_path / "cv_out" / "cross_validation_scorer_mean_std.csv").exists()
    assert (tmp_path / "cv_out" / "cross_validation_ocscore_wins.csv").exists()
    assert (tmp_path / "cv_out" / "cross_validation_fold_rankings.csv").exists()
    assert paths["per_target_csv"] == str((tmp_path / "cv_out" / "cross_validation_per_target_metrics.csv").resolve())
    per_target = pd.read_csv(paths["per_target_csv"])
    assert {"fold_index", "group", "scorer", "scorer_type", "BEDROC"}.issubset(per_target.columns)
    assert (per_target["scorer"] == "vina_vina").any()
    assert "desc_mean" in result.scoring_function_columns
    assert "sf_mean" in result.scoring_function_columns
    assert "desc_mean" in result.fold_results[0].scoring_function_metrics
    assert "sf_mean" in result.fold_results[0].scoring_function_metrics


@pytest.mark.order(274)
def test_build_scorer_comparison_summary_reports_mean_std_and_wins():
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
        export_dir="/tmp/export",
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
    summary = occv.build_scorer_comparison_summary(
        result,
        comparison_metrics=["BEDROC", "ROC-AUC"],
    )
    ocscore_bedroc = next(
        row for row in summary["mean_std"] if row["scorer"] == "OCScore" and row["metric"] == "BEDROC"
    )
    assert ocscore_bedroc["mean"] == pytest.approx(0.8)
    assert ocscore_bedroc["std"] == pytest.approx(0.141421, rel=1e-3)
    bedroc_wins = next(row for row in summary["ocscore_wins"] if row["metric"] == "BEDROC")
    assert bedroc_wins["n_folds_won"] == 1
    assert bedroc_wins["n_folds_compared"] == 2
    fold0_bedroc_ranks = [
        row for row in summary["fold_rankings"] if row["fold_index"] == 0 and row["metric"] == "BEDROC"
    ]
    assert fold0_bedroc_ranks[0]["scorer"] == "OCScore"
    assert fold0_bedroc_ranks[0]["rank"] == 1


@pytest.mark.order(275)
def test_run_cross_validation_rejects_overlapping_fold_indices(tmp_path, monkeypatch):
    base_input_size = 8
    params = {
        "dudez_use_transfer": False,
        "dudez_fine_tuning_mode": "full",
        "dudez_num_unfrozen_layers": 1,
        "dudez_classifier_hidden_size": 4,
        "dudez_classifier_dropout": 0.0,
        "dudez_classifier_activation": "GELU",
        "dudez_use_class_weighting": False,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 8,
    }
    features = [f"f{i}" for i in range(base_input_size)]
    input_size = len(features)
    architecture = {"hidden_sizes": [6, 4], "latent_dim": 3, "projection_dim": 0}
    model = ocstaged.build_dudez_model(
        input_size=input_size,
        params=params,
        feature_extractor_architecture=architecture,
    )
    rng = np.random.default_rng(3)
    rows = []
    for receptor in ("r1", "r2"):
        for kind in ("ligands", "decoys"):
            for _ in range(4):
                row = {feature: float(rng.normal()) for feature in features}
                row["receptor"] = receptor
                row["kind"] = kind
                rows.append(row)
    df = pd.DataFrame(rows)
    export_dir = tmp_path / "dudez_export"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="dudez_screening",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=0,
        best_objective_value=0.5,
        validation_metrics={"BEDROC": 0.5},
        test_metrics={"BEDROC": 0.4},
        stage_config={"kind_column": "kind"},
        splits={
            "X_train": rng.normal(size=(8, input_size)).astype(np.float32),
            "y_train": np.array([1, 0] * 4, dtype=np.float32),
            "X_val": rng.normal(size=(4, input_size)).astype(np.float32),
            "y_val": np.array([1, 0] * 2, dtype=np.float32),
            "X_test": rng.normal(size=(4, input_size)).astype(np.float32),
            "y_test": np.array([1, 0] * 2, dtype=np.float32),
            "train_indices": np.arange(8),
            "validation_indices": np.arange(8, 12),
            "test_indices": np.arange(12, 16),
            "split_config": {},
            "split_diagnostics": {},
        },
        objective_metric="BEDROC",
        direction="maximize",
        best_params=params,
        validate=False,
    )

    def _bad_folds(groups, n_folds, **kwargs):
        return [(np.array([0, 1, 2]), np.array([2, 3, 4]))]

    monkeypatch.setattr(occv, "iter_receptor_group_kfold_indices", _bad_folds)
    with pytest.raises(ValueError, match="overlap"):
        occv.run_cross_validation_from_export(
            export_dir,
            df,
            config=occv.CrossValidationConfig(n_folds=2, epochs=1, strategy="receptor_grouped"),
            device="cpu",
        )


@pytest.mark.order(276)
def test_pdbbind_cv_scaler_fit_uses_training_rows_only(tmp_path, monkeypatch):
    fit_shapes: list[tuple[int, ...]] = []
    transform_shapes: list[tuple[int, ...]] = []

    class RecordingScaler(occv.StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            fit_shapes.append(np.asarray(X).shape)
            return super().fit(X, y=y, sample_weight=sample_weight)

        def fit_transform(self, X, y=None, sample_weight=None):
            fit_shapes.append(np.asarray(X).shape)
            return super().fit_transform(X, y=y, sample_weight=sample_weight)

        def transform(self, X):
            transform_shapes.append(np.asarray(X).shape)
            return super().transform(X)

    monkeypatch.setattr(occv, "StandardScaler", RecordingScaler)

    input_size = 6
    params = {
        "encoder_architecture_index": 0,
        "encoder_hidden_sizes": [8, 4],
        "encoder_latent_dim": 3,
        "encoder_depth": 2,
        "encoder_is_monotonic": True,
        "projection_dim": 0,
        "encoder_activation": "GELU",
        "encoder_dropout": 0.0,
        "optimizer_learning_rate": 1e-3,
        "optimizer_weight_decay": 1e-4,
        "optimizer_batch_size": 4,
        "decoder_hidden_sizes": [],
        "decoder_depth": 0,
        "decoder_lambda_rec": 0.0,
        "dae_noise_type": "none",
        "dae_mask_prob": 0.0,
        "dae_gaussian_std": 0.0,
        "pdbbind_regression_loss": "mse",
        "pdbbind_huber_delta": 1.0,
    }
    model = ocstaged.build_pdbbind_model(input_size=input_size, params=params)
    features = [f"f{i}" for i in range(input_size)]
    rng = np.random.default_rng(4)
    n_rows = 18
    df = pd.DataFrame(rng.normal(size=(n_rows, input_size)), columns=features)
    df["experimental"] = rng.normal(size=n_rows)
    export_dir = tmp_path / "pdbbind_export"
    ocexport.export_best_model_bundle(
        export_dir=export_dir,
        task="pdbbind_regression",
        model=model,
        model_config=params,
        selected_features=features,
        best_trial_number=0,
        best_objective_value=1.0,
        validation_metrics={"RMSE": 1.0},
        test_metrics={"RMSE": 1.1},
        stage_config={},
        splits={
            "X_train": rng.normal(size=(9, input_size)).astype(np.float32),
            "y_train": rng.normal(size=9).astype(np.float32),
            "X_val": rng.normal(size=(4, input_size)).astype(np.float32),
            "y_val": rng.normal(size=4).astype(np.float32),
            "X_test": rng.normal(size=(5, input_size)).astype(np.float32),
            "y_test": rng.normal(size=5).astype(np.float32),
            "train_indices": np.arange(9),
            "validation_indices": np.arange(9, 13),
            "test_indices": np.arange(13, 18),
            "split_config": {"target_column": "experimental"},
            "split_diagnostics": {},
        },
        objective_metric="RMSE",
        direction="minimize",
        best_params=params,
        validate=False,
    )

    result = occv.run_cross_validation_from_export(
        export_dir,
        df,
        config=occv.CrossValidationConfig(n_folds=3, epochs=1, random_seed=0),
        device="cpu",
        output_dir=tmp_path / "cv",
    )
    assert result.effective_folds == 3
    assert fit_shapes
    assert transform_shapes
    for fold in result.fold_results:
        n_train = fold.n_train
        n_val = fold.n_validation
        assert any(shape[0] == n_train for shape in fit_shapes)
        assert any(shape[0] == n_val for shape in transform_shapes)
        assert not any(shape[0] == n_train + n_val for shape in fit_shapes)
