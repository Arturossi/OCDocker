#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for strict OCScore Workbench layout discovery.
"""

# Imports
###############################################################################
from __future__ import annotations

import json

from OCDocker.Workbench import build_ocscore_workspace
from OCDocker.Workbench.OCScoreLayout import resolve_optuna_dashboard_slot_count

# License
###############################################################################
"""OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Functions
###############################################################################
## Private ##


def _write_strict_ocscore_root(root) -> None:
    '''Write a synthetic strict OCScore output root.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.
    '''

    replica_1 = root / "replica_1"
    replica_1.mkdir()
    (replica_1 / "metrics.csv").write_text("BEDROC,random_metric\n0.41,99\n0.43,100\n", encoding="utf-8")
    (replica_1 / "run.log").write_text("completed\n", encoding="utf-8")
    shap_dir = replica_1 / "dudez" / "shap"
    shap_dir.mkdir(parents=True)
    (shap_dir / "shap_feature_importance.png").write_bytes(b"png")

    baseline_export = root / "export" / "dudez" / "shap"
    baseline_export.mkdir(parents=True)
    (baseline_export / "shap_beeswarm_plot.png").write_bytes(b"png")

    replica_2 = root / "replica_2"
    replica_2.mkdir()
    (replica_2 / "summary.json").write_text(
        '{"aggregate_summary": {"metrics": {"dudez_test_bedroc": {"mean": 0.51}}}}',
        encoding="utf-8",
    )

    failed = root / "replica_3"
    failed.mkdir()
    (failed / "run.log").write_text("Traceback: failed\n", encoding="utf-8")

    ablation = root / "ablation" / "no_shape"
    replica = ablation / "replica_001"
    replica.mkdir(parents=True)
    (replica / "metrics.csv").write_text("metric,value\nBEDROC,0.33\nRMSE,1.4\n", encoding="utf-8")
    ablation_export = root / "export" / "ablations" / "no_shape" / "pdbbind" / "figures"
    ablation_export.mkdir(parents=True)
    (ablation_export / "cv_mean_std_RMSE.png").write_bytes(b"png")


## Public ##


def test_build_ocscore_workspace_detects_baseline_and_ablation_layout(tmp_path) -> None:
    '''Strict workspace discovery follows baseline and ablation replica folders.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_strict_ocscore_root(tmp_path)

    workspace = build_ocscore_workspace(tmp_path, max_depth=4)

    assert workspace.study_count == 2
    assert workspace.baseline_study.expected_replica_count == 3
    assert workspace.baseline_study.detected_replica_count == 3
    assert workspace.baseline_study.completed_count == 2
    assert workspace.baseline_study.failed_count == 1
    assert workspace.baseline_study.missing_count == 0
    assert workspace.baseline_study.metric_summary["test_bedroc"]["count"] == 2
    assert round(workspace.baseline_study.metric_summary["test_bedroc"]["mean"], 3) == 0.465
    assert workspace.baseline_study.replicas[0].metrics[0].name == "test_bedroc"
    assert workspace.baseline_study.replicas[0].figures[0].role == "shap_importance"
    assert workspace.baseline_study.figures[0].role == "shap_beeswarm"
    assert workspace.ablation_studies[0].study_name == "no_shape"
    assert workspace.ablation_studies[0].figures[0].dataset == "pdbbind"
    assert workspace.ablation_studies[0].figures[0].role == "cv_mean_std"
    assert workspace.ablation_studies[0].figures[0].metric_name == "rmse"
    assert workspace.ablation_studies[0].replicas[0].replica_name == "replica_001"
    assert set(workspace.ablation_studies[0].metric_summary) == {"test_bedroc", "rmse"}


def test_build_ocscore_workspace_preserves_validation_and_test_metric_scopes(tmp_path) -> None:
    '''Strict discovery keeps DUDEz validation and test metrics in separate scopes.'''

    root = tmp_path / "scoped"
    root.mkdir()
    replica = root / "replica_1"
    replica.mkdir()
    (replica / "summary.json").write_text(
        '{"aggregate_summary": {"metrics": {'
        '"dudez_test_bedroc": {"mean": 0.61}, '
        '"dudez_validation_primary_metric": {"mean": 0.55}'
        "}}}",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(root, max_depth=4)

    assert "test_bedroc" in workspace.baseline_study.metric_summary
    assert "validation_metric" in workspace.baseline_study.metric_summary
    assert round(workspace.baseline_study.metric_summary["test_bedroc"]["mean"], 2) == 0.61
    assert round(workspace.baseline_study.metric_summary["validation_metric"]["mean"], 2) == 0.55


def test_build_ocscore_workspace_loads_cross_validation_summary(tmp_path) -> None:
    '''Strict discovery loads exported cross-validation scorer summaries.'''

    root = tmp_path / "cv_root"
    root.mkdir()
    replica = root / "replica_1"
    replica.mkdir()
    cv_dir = root / "export" / "cross_validation"
    cv_dir.mkdir(parents=True)
    (cv_dir / "cross_validation_scorer_mean_std.csv").write_text(
        "scorer,metric,mean,std,n_folds\n"
        "OCScore,MAE,1.2,0.1,5\n"
        "vina_vina,MAE,1.5,0.2,5\n",
        encoding="utf-8",
    )
    (cv_dir / "cross_validation_folds.csv").write_text(
        "fold_index,n_train,n_validation,validation_MAE\n"
        "0,100,25,1.1\n"
        "1,100,25,1.3\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(root, max_depth=2)

    assert workspace.baseline_study.cross_validation is not None
    assert workspace.baseline_study.cross_validation.fold_count == 2
    assert len(workspace.baseline_study.cross_validation.metrics) == 2
    assert workspace.baseline_study.cross_validation.metrics[0].scorer == "OCScore"


def test_build_ocscore_workspace_loads_external_baselines(tmp_path) -> None:
    '''Strict workspace discovery loads baselines_summary.csv as external references.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "sf_mean,sf_consensus,test,0.31,0.71,0.04,10.0,4.5,5\n"
        "sf_median,sf_consensus,test,0.29,0.70,0.03,9.5,4.2,5\n"
        "sf_max,sf_consensus,test,0.30,0.705,0.035,9.8,4.3,5\n"
        "sf_min,sf_consensus,test,0.28,0.69,0.028,9.2,4.0,5\n"
        "desc_mean,descriptor_aggregate,test,0.27,0.68,0.025,8.8,3.9,5\n"
        "desc_median,descriptor_aggregate,test,0.26,0.67,0.024,8.5,3.7,5\n"
        "desc_max,descriptor_aggregate,test,0.265,0.675,0.0245,8.6,3.75,5\n"
        "desc_min,descriptor_aggregate,test,0.255,0.665,0.023,8.3,3.6,5\n"
        "vina_vina,scoring_function,test,0.25,0.68,0.02,8.0,3.8,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert len(workspace.external_baselines) == 9
    sf_mean = next(item for item in workspace.external_baselines if item.baseline_name == "sf_mean")
    assert sf_mean.baseline_family == "sf_consensus"
    assert round(sf_mean.metric_summary["test_bedroc"]["mean"], 2) == 0.31
    assert sf_mean.metric_summary["test_bedroc"]["direction"] == "max"
    desc_min = next(item for item in workspace.external_baselines if item.baseline_name == "desc_min")
    assert desc_min.baseline_family == "descriptor_aggregate"


def test_build_ocscore_workspace_loads_baselines_from_output_root_with_train_child(tmp_path) -> None:
    '''External baselines resolve from the OCScore output root when layout lives under train/.'''

    train_root = tmp_path / "train"
    train_root.mkdir()
    _write_strict_ocscore_root(train_root)
    (tmp_path / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "sf_mean,sf_consensus,test,0.31,0.71,0.04,10.0,4.5,5\n"
        "desc_mean,descriptor_aggregate,test,0.27,0.68,0.025,8.8,3.9,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert {item.baseline_name for item in workspace.external_baselines} == {"sf_mean", "desc_mean"}


def test_build_ocscore_workspace_loads_full_production_baseline_set(tmp_path) -> None:
    '''Production baseline CSV rows include SF, learned, and row aggregates.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "vina_vina,scoring_function,test,0.25,0.68,0.02,8.0,3.8,5\n"
        "sf_mean,sf_consensus,test,0.31,0.71,0.04,10.0,4.5,5\n"
        "sf_median,sf_consensus,test,0.29,0.70,0.03,9.5,4.2,5\n"
        "sf_max,sf_consensus,test,0.30,0.705,0.035,9.8,4.3,5\n"
        "sf_min,sf_consensus,test,0.28,0.69,0.028,9.2,4.0,5\n"
        "lr_sf,learned_sf,test,0.24,0.67,0.019,7.8,3.6,5\n"
        "rf_sf,learned_sf,test,0.245,0.672,0.0195,7.9,3.65,5\n"
        "xgb_sf,learned_sf,test,0.247,0.674,0.020,8.1,3.7,5\n"
        "shuffled_lr_sf,learned_sf,test,0.20,0.55,0.01,6.0,3.0,5\n"
        "desc_mean,descriptor_aggregate,test,0.27,0.68,0.025,8.8,3.9,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)
    names = {item.baseline_name for item in workspace.external_baselines}

    assert names == {
        "vina_vina",
        "sf_mean",
        "sf_median",
        "sf_max",
        "sf_min",
        "lr_sf",
        "rf_sf",
        "xgb_sf",
        "shuffled_lr_sf",
        "desc_mean",
    }
    assert "test_bedroc" in workspace.metric_names
    assert next(item for item in workspace.external_baselines if item.baseline_name == "xgb_sf").baseline_family == "learned_sf"


def test_build_ocscore_workspace_discovers_nested_baseline_csv(tmp_path) -> None:
    '''Nested export baseline CSV files are discovered under the output root.'''

    _write_strict_ocscore_root(tmp_path)
    nested = tmp_path / "export" / "full_ocscore"
    nested.mkdir(parents=True)
    (nested / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "sf_mean,sf_consensus,test,0.31,0.71,0.04,10.0,4.5,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert any(item.baseline_name == "sf_mean" for item in workspace.external_baselines)


def test_build_ocscore_workspace_synthesizes_sf_consensus_from_individual_sfs(tmp_path) -> None:
    '''SF row aggregates are derived when only individual scoring functions are exported.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "dudez_sf_baseline_comparison.csv").write_text(
        "scorer,scorer_type,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%\n"
        "vina_vina,sf,test,0.25,0.68,0.02,8.0,3.8\n"
        "gnina_gnina,sf,test,0.31,0.71,0.04,10.0,4.5\n"
        "plants_plants,sf,test,0.27,0.69,0.03,9.0,4.1\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)
    names = {item.baseline_name for item in workspace.external_baselines if item.split == "test"}

    assert {"vina_vina", "gnina_gnina", "plants_plants"}.issubset(names)
    assert {"sf_mean", "sf_median", "sf_max", "sf_min"}.issubset(names)
    sf_mean = next(item for item in workspace.external_baselines if item.baseline_name == "sf_mean")
    assert sf_mean.baseline_family == "sf_consensus"
    assert round(sf_mean.metric_summary["test_bedroc"]["mean"], 2) == 0.28
    assert sf_mean.synthesized is True
    sf_max = next(item for item in workspace.external_baselines if item.baseline_name == "sf_max")
    assert sf_max.synthesized is True
    assert round(sf_max.metric_summary["test_bedroc"]["mean"], 2) == 0.31


def test_build_ocscore_workspace_keeps_csv_sf_consensus_over_synthesis(tmp_path) -> None:
    '''Precomputed sf_* rows from CSV are not replaced by synthesized aggregates.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "vina_vina,scoring_function,test,0.25,0.68,0.02,8.0,3.8,5\n"
        "sf_mean,sf_consensus,test,0.99,0.99,0.99,99.0,99.0,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)
    sf_mean = next(item for item in workspace.external_baselines if item.baseline_name == "sf_mean")

    assert round(sf_mean.metric_summary["test_bedroc"]["mean"], 2) == 0.99
    assert sf_mean.synthesized is False


def test_build_ocscore_workspace_loads_dudez_baseline_comparison_csv(tmp_path) -> None:
    '''Example-17 comparison CSV rows are accepted as external baselines.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "dudez_sf_baseline_comparison.csv").write_text(
        "scorer,scorer_type,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%\n"
        "sf_mean,sf_consensus,test,0.31,0.71,0.04,10.0,4.5\n"
        "desc_median,descriptor_aggregate,test,0.26,0.67,0.024,8.5,3.7\n"
        "vina_vina,sf,test,0.25,0.68,0.02,8.0,3.8\n"
        "OCScore (trial 1),model,test,0.52,0.82,0.10,12.0,5.0\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert {item.baseline_name for item in workspace.external_baselines} == {
        "sf_mean",
        "sf_median",
        "sf_max",
        "sf_min",
        "desc_median",
        "vina_vina",
    }
    assert next(item for item in workspace.external_baselines if item.baseline_name == "vina_vina").baseline_family == "scoring_function"


def test_build_ocscore_workspace_loads_protocol_summary(tmp_path) -> None:
    '''Strict workspace discovery exposes curated protocol metadata for the dashboard.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "replicas_protocol.json").write_text(
        json.dumps(
            {
                "n_replicas": 3,
                "base_seed": 42,
                "replica_jobs": 2,
                "replica_names": ["replica_000", "replica_001", "replica_002"],
                "stage_list": [
                    {
                        "name": "pdbbind_optuna",
                        "config": {
                            "objective_metric": "RMSE",
                            "n_trials": 100,
                            "epochs": 100,
                            "split_config": {
                                "strategy": "receptor_heldout",
                                "train_size": 0.6,
                                "validation_size": 0.2,
                                "test_size": 0.2,
                            },
                        },
                    },
                    {
                        "name": "dudez_optuna",
                        "config": {
                            "primary_metric": "BEDROC",
                            "bedroc_alpha": 20.0,
                            "dudez_scaling_config": {"strategy": "pdbbind_scaler"},
                            "n_trials": 100,
                            "epochs": 100,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ablation_container = tmp_path / "ablations"
    ablation_container.mkdir()
    (ablation_container / "ablation_summary.json").write_text(
        json.dumps(
            {
                "protocol": "test-ablation-campaign",
                "variants": [
                    {"feature_policy_name": "full_ocscore"},
                    {"feature_policy_name": "no_shape"},
                ],
            }
        ),
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, max_depth=4)

    assert workspace.protocol is not None
    assert workspace.protocol.protocol_name == "test-ablation-campaign"
    assert workspace.protocol.n_replicas == 3
    assert workspace.protocol.pdbbind_split_strategy == "receptor_heldout"
    assert workspace.protocol.dudez_bedroc_alpha == 20.0
    assert workspace.protocol.ablation_variants == ("full_ocscore", "no_shape")
    assert workspace.baseline_study.protocol is not None
    assert workspace.baseline_study.protocol.stage_names == ("pdbbind_optuna", "dudez_optuna")
    assert workspace.ablation_studies[0].protocol is None


def test_build_ocscore_workspace_exposes_run_context(tmp_path) -> None:
    '''Strict workspace discovery exposes always-visible run context metadata.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "replicas_protocol.json").write_text(
        json.dumps(
            {
                "n_replicas": 3,
                "stage_list": [
                    {
                        "name": "pdbbind_optuna",
                        "config": {
                            "split_config": {
                                "strategy": "receptor_heldout",
                                "train_size": 0.6,
                                "validation_size": 0.2,
                                "test_size": 0.2,
                            },
                        },
                    },
                    {"name": "dudez_optuna", "config": {"bedroc_alpha": 20.0}},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "baselines_summary.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,n_replicas\n"
        "vina_vina,scoring_function,test,0.25,0.68,0.02,8.0,3.8,5\n",
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, max_depth=4)

    assert workspace.run_context is not None
    assert workspace.run_context.planned_replica_count == 3
    assert workspace.run_context.detected_replica_count == 3
    assert workspace.run_context.pdbbind_split_strategy == "receptor_heldout"
    assert workspace.run_context.dudez_bedroc_alpha == 20.0
    assert len(workspace.run_context.baseline_sources) == 1
    assert workspace.run_context.baseline_sources[0].path.name == "baselines_summary.csv"


def test_build_ocscore_workspace_infers_replica_count_from_protocol(tmp_path) -> None:
    '''Strict workspace discovery uses protocol metadata for planned replica slots.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "replicas_protocol.json").write_text(
        '{"n_replicas": 5, "replica_name_prefix": "replica_"}',
        encoding="utf-8",
    )

    workspace = build_ocscore_workspace(tmp_path, max_depth=4)

    assert workspace.baseline_study.expected_replica_count == 5
    assert workspace.baseline_study.detected_replica_count == 3
    assert workspace.baseline_study.missing_count == 2


def test_build_ocscore_workspace_reports_missing_root(tmp_path) -> None:
    '''Strict workspace discovery reports a missing OCScore root without crashing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    workspace = build_ocscore_workspace(tmp_path / "missing", expected_replica_count=2)

    assert workspace.issue_count == 1
    assert workspace.missing_count == 2
    assert workspace.baseline_study.replicas[0].status == "missing"


def test_build_ocscore_workspace_resolves_train_child_layout(tmp_path) -> None:
    '''Strict discovery accepts normal OCScore output roots with train children.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    train_root = tmp_path / "train"
    train_root.mkdir()
    _write_strict_ocscore_root(train_root)

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert workspace.root == train_root
    assert workspace.baseline_study.detected_replica_count == 3
    assert workspace.ablation_studies[0].study_name == "no_shape"

def test_build_ocscore_workspace_reports_unsupported_manifest_layout(tmp_path) -> None:
    '''Strict discovery rejects the removed generic Workbench manifest layout.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.yml").write_text("run_id: old-run\n", encoding="utf-8")
    (run_dir / "result_manifest.yml").write_text("metrics: {}\n", encoding="utf-8")

    workspace = build_ocscore_workspace(tmp_path, expected_replica_count=5, max_depth=4)

    assert workspace.issue_count == 1
    assert "Unsupported generic Workbench manifest layout" in workspace.issues[0].message
    assert workspace.baseline_study.detected_replica_count == 0
    assert workspace.ablation_studies == ()


def test_resolve_optuna_dashboard_slot_count_uses_replica_directories(tmp_path) -> None:
    '''Optuna slot count follows detected replica directories.'''

    _write_strict_ocscore_root(tmp_path)

    assert resolve_optuna_dashboard_slot_count(tmp_path) == 3


def test_resolve_optuna_dashboard_slot_count_uses_protocol_n_replicas(tmp_path) -> None:
    '''Optuna slot count follows protocol n_replicas when larger than detected dirs.'''

    _write_strict_ocscore_root(tmp_path)
    (tmp_path / "replicas_protocol.json").write_text(
        json.dumps({"n_replicas": 8}),
        encoding="utf-8",
    )

    assert resolve_optuna_dashboard_slot_count(tmp_path) == 8


def test_resolve_optuna_dashboard_slot_count_is_clamped(tmp_path) -> None:
    '''Optuna slot count stays within 1 and 50.'''

    assert resolve_optuna_dashboard_slot_count(tmp_path) == 1
    (tmp_path / "replicas_protocol.json").write_text(
        json.dumps({"n_replicas": 100}),
        encoding="utf-8",
    )

    assert resolve_optuna_dashboard_slot_count(tmp_path) == 50
    assert resolve_optuna_dashboard_slot_count(tmp_path, override=12) == 12
    assert resolve_optuna_dashboard_slot_count(tmp_path, override=80) == 50

