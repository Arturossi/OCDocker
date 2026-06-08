#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for replicated staged OCScore optimization protocol execution.
'''

# Imports
###############################################################################
from types import SimpleNamespace
from pathlib import Path

import json
import pandas as pd
import optuna
import pytest

from OCDocker.OCScore.Optimization.Protocol import ProtocolContext
from OCDocker.OCScore.Optimization.Protocol import ReplicatedStagedProtocol

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

class _FakePDBbindStage:
    name = "pdbbind_optuna"

    def __init__(self, fail_on_replica=None, storage="auto"):
        self.fail_on_replica = fail_on_replica
        self.config = SimpleNamespace(
            study_name="PDBbind_Regression_Optimization",
            storage=storage,
            random_seed=None,
            sampler_seed=None,
            n_trials=3,
            epochs=1,
            objective_metric="RMSE",
            direction="minimize",
        )

    def run(self, context):
        if context.metadata["replica_index"] == self.fail_on_replica:
            raise RuntimeError("planned pdbbind failure")

        replica_index = context.metadata["replica_index"]
        out = Path(context.output_dir) / "pdbbind"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / "pdbbind_best.pt"
        checkpoint_path.write_text("fake pdbbind checkpoint", encoding="utf-8")
        context.stage_results[self.name] = {
            "task": "affinity_regression",
            "objective_metric": "RMSE",
            "direction": "minimize",
            "study_name": self.config.study_name,
            "config_storage": self.config.storage,
            "config_random_seed": self.config.random_seed,
            "config_sampler_seed": self.config.sampler_seed,
            "best_trial": replica_index + 10,
            "best_value": 1.0 + replica_index,
            "validation_metrics": {
                "RMSE": 1.0 + replica_index,
                "MAE": 0.5 + replica_index,
                "Pearson r": 0.2 + replica_index,
                "Spearman rho": 0.3 + replica_index,
                "R2": 0.4 + replica_index,
            },
            "test_metrics": {
                "RMSE": 2.0 + replica_index,
                "MAE": 0.7 + replica_index,
                "Pearson r": 0.6 + replica_index,
                "Spearman rho": 0.65 + replica_index,
                "R2": 0.1 + replica_index,
            },
            "checkpoint_path": str(checkpoint_path),
            "optuna_storage_path": str(out / "pdbbind_optuna.db"),
        }
        return context


class _FakeTransferStage:
    name = "transfer_feature_extractor"

    def run(self, context):
        context.stage_results[self.name] = {
            "transferred_components": ["feature_extractor"],
            "excluded_components": ["regression_head", "decoder"],
        }
        return context


class _FakeDUDEzStage:
    name = "dudez_optuna"

    def __init__(self, fail_on_replica=None, storage="auto"):
        self.fail_on_replica = fail_on_replica
        self.config = SimpleNamespace(
            study_name="DUDEz_Screening_Optimization",
            storage=storage,
            random_seed=None,
            sampler_seed=None,
            n_trials=4,
            epochs=1,
            primary_metric="BEDROC",
            direction="maximize",
        )

    def run(self, context):
        if context.metadata["replica_index"] == self.fail_on_replica:
            raise RuntimeError("planned dudez failure")

        replica_index = context.metadata["replica_index"]
        out = Path(context.output_dir) / "dudez"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / "dudez_best.pt"
        checkpoint_path.write_text("fake dudez checkpoint", encoding="utf-8")
        context.stage_results[self.name] = {
            "task": "active_decoy_screening",
            "objective_metric": "BEDROC",
            "direction": "maximize",
            "study_name": self.config.study_name,
            "config_storage": self.config.storage,
            "config_random_seed": self.config.random_seed,
            "config_sampler_seed": self.config.sampler_seed,
            "best_trial": replica_index + 20,
            "best_value": 0.8 - (0.1 * replica_index),
            "validation_metrics": {
                "BEDROC": 0.8 - (0.1 * replica_index),
                "PR-AUC": 0.7 - (0.1 * replica_index),
            },
            "test_metrics": {
                "ROC-AUC": 0.9 - (0.1 * replica_index),
                "PR-AUC": 0.8 - (0.1 * replica_index),
                "BEDROC": 0.7 - (0.1 * replica_index),
                "EF1%": 5.0 + replica_index,
                "EF5%": 3.0 + replica_index,
                "NDCG@1%": 0.6 - (0.1 * replica_index),
                "NDCG@5%": 0.65 - (0.1 * replica_index),
            },
            "checkpoint_path": str(checkpoint_path),
            "optuna_storage_path": str(out / "dudez_optuna.db"),
        }
        return context


# Functions
###############################################################################
## Private ##

def _context(tmp_path):
    selected = ["f0", "f1"]
    return ProtocolContext(
        pdbbind_df=pd.DataFrame({"f0": [0.1, 0.2], "f1": [1.1, 1.2], "experimental": [7.0, 6.5]}),
        dudez_df=pd.DataFrame({"f0": [0.3, 0.4], "f1": [1.3, 1.4], "kind": ["ligands", "decoys"]}),
        selected_features=selected,
        output_dir=str(tmp_path),
        random_seed=42,
        metadata={"feature_reduction_protocol": "feature_reduction_protocol.json"},
    )


def _fake_stages(**kwargs):
    return [
        _FakePDBbindStage(
            fail_on_replica=kwargs.get("pdbbind_fail"),
            storage=kwargs.get("pdbbind_storage", "auto"),
        ),
        _FakeTransferStage(),
        _FakeDUDEzStage(
            fail_on_replica=kwargs.get("dudez_fail"),
            storage=kwargs.get("dudez_storage", "auto"),
        ),
    ]


## Public ##

@pytest.mark.order(496)
def test_replicated_protocol_runs_from_python_api_and_writes_reports(tmp_path):
    context = _context(tmp_path)
    protocol = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=3,
        base_seed=100,
        replica_name_prefix="replica",
    )

    result = protocol.run(context)

    assert len(result.replica_results) == 3
    assert result.summary_df["replica_name"].tolist() == ["replica_000", "replica_001", "replica_002"]
    assert result.summary_df["seed"].tolist() == [100, 101, 102]
    assert (tmp_path / "replica_000").is_dir()
    assert (tmp_path / "replica_001").is_dir()
    assert (tmp_path / "replica_002").is_dir()
    assert (tmp_path / "replicas_summary.csv").is_file()
    assert (tmp_path / "replicas_summary.json").is_file()
    assert (tmp_path / "replicas_protocol.json").is_file()

    protocol_payload = json.loads((tmp_path / "replicas_protocol.json").read_text(encoding="utf-8"))
    assert protocol_payload["n_replicas"] == 3
    assert protocol_payload["base_seed"] == 100
    assert "selected_features" not in protocol_payload
    assert "input_metadata" not in protocol_payload
    selected_features_path = protocol_payload["selected_features_json"]
    static_context = json.loads((tmp_path / "static_context.json").read_text(encoding="utf-8"))
    assert json.loads((tmp_path / "selected_features.json").read_text(encoding="utf-8")) == context.selected_features
    assert selected_features_path == str(tmp_path / "selected_features.json")
    assert static_context["n_selected_features"] == len(context.selected_features)
    assert static_context["input_metadata"]["feature_reduction_protocol"] == "feature_reduction_protocol.json"
    assert "run_feature_reduction_protocol" not in json.dumps(protocol_payload)


@pytest.mark.order(497)
def test_replica_outputs_study_names_seeds_and_paths_are_unique(tmp_path):
    result = ReplicatedStagedProtocol(
        stages=_fake_stages(
            pdbbind_storage=f"sqlite:///{tmp_path}/shared_pdbbind.db",
            dudez_storage=f"sqlite:///{tmp_path}/shared_dudez.db",
        ),
        n_replicas=2,
        base_seed=9,
    ).run(_context(tmp_path))

    first, second = result.replica_results
    first_pdb = first.context.stage_results["pdbbind_optuna"]
    second_pdb = second.context.stage_results["pdbbind_optuna"]
    first_dudez = first.context.stage_results["dudez_optuna"]
    second_dudez = second.context.stage_results["dudez_optuna"]

    assert first_pdb["study_name"].endswith("_replica_000")
    assert second_pdb["study_name"].endswith("_replica_001")
    assert first_dudez["study_name"].endswith("_replica_000")
    assert second_dudez["study_name"].endswith("_replica_001")
    assert first_pdb["config_storage"] == second_pdb["config_storage"]
    assert first_dudez["config_storage"] == second_dudez["config_storage"]
    assert first_pdb["config_storage"].endswith("shared_pdbbind.db")
    assert first_dudez["config_storage"].endswith("shared_dudez.db")
    assert first_pdb["config_random_seed"] == 9
    assert second_pdb["config_random_seed"] == 10
    assert first_dudez["config_sampler_seed"] == 9
    assert second_dudez["config_sampler_seed"] == 10
    assert result.summary_df["pdbbind_checkpoint_path"].nunique() == 2
    assert result.summary_df["dudez_checkpoint_path"].nunique() == 2
    assert result.summary_df["protocol_log_path"].nunique() == 2


@pytest.mark.order(498)
def test_replicated_protocol_aggregates_metrics_separately_by_task(tmp_path):
    result = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=2,
        base_seed=42,
    ).run(_context(tmp_path))

    metrics = result.aggregate_summary["metrics"]
    assert metrics["pdbbind_validation_rmse"]["mean"] == pytest.approx(1.5)
    assert metrics["pdbbind_test_rmse"]["mean"] == pytest.approx(2.5)
    assert metrics["pdbbind_test_mae"]["mean"] == pytest.approx(1.2)
    assert metrics["pdbbind_test_pearson_r"]["mean"] == pytest.approx(1.1)
    assert metrics["pdbbind_test_spearman_rho"]["mean"] == pytest.approx(1.15)
    assert metrics["pdbbind_test_r2"]["mean"] == pytest.approx(0.6)
    assert metrics["dudez_validation_primary_metric"]["mean"] == pytest.approx(0.75)
    assert metrics["dudez_test_roc_auc"]["mean"] == pytest.approx(0.85)
    assert metrics["dudez_test_pr_auc"]["mean"] == pytest.approx(0.75)
    assert metrics["dudez_test_bedroc"]["mean"] == pytest.approx(0.65)
    assert metrics["dudez_test_ef1"]["mean"] == pytest.approx(5.5)
    assert metrics["dudez_test_ef5"]["mean"] == pytest.approx(3.5)
    assert metrics["dudez_test_ndcg_1"]["mean"] == pytest.approx(0.55)
    assert metrics["dudez_test_ndcg_5"]["mean"] == pytest.approx(0.6)
    assert result.aggregate_summary["best_pdbbind_replica"]["replica_name"] == "replica_000"
    assert result.aggregate_summary["best_dudez_replica"]["replica_name"] == "replica_000"
    assert "RMSE - AUC" not in json.dumps(result.aggregate_summary)


@pytest.mark.order(499)
def test_replicated_protocol_failure_behavior_is_explicit(tmp_path):
    context = _context(tmp_path / "fail_loudly")
    with pytest.raises(RuntimeError, match="replica_001"):
        ReplicatedStagedProtocol(
            stages=_fake_stages(dudez_fail=1),
            n_replicas=3,
            base_seed=1,
        ).run(context)

    result = ReplicatedStagedProtocol(
        stages=_fake_stages(dudez_fail=1),
        n_replicas=3,
        base_seed=1,
        continue_on_replica_failure=True,
    ).run(_context(tmp_path / "continue"))

    assert len(result.replica_results) == 3
    assert len(result.failed_replicas) == 1
    assert result.failed_replicas[0].replica_name == "replica_001"
    assert result.failed_replicas[0].failed_stage == "dudez_optuna"
    assert result.summary_df.loc[1, "status"] == "failed"
    payload = json.loads((tmp_path / "continue" / "replicas_summary.json").read_text(encoding="utf-8"))
    assert payload["failed_replicas"][0]["replica_name"] == "replica_001"


@pytest.mark.order(500)
def test_replicated_protocol_single_replica_preserves_single_run_shape(tmp_path):
    result = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=1,
        base_seed=42,
    ).run(_context(tmp_path))

    assert len(result.replica_results) == 1
    assert result.summary_df.shape[0] == 1
    assert result.summary_df.loc[0, "replica_name"] == "replica_000"
    assert result.summary_df.loc[0, "seed"] == 42
    assert result.failed_replicas == []


@pytest.mark.order(501)
def test_replicated_protocol_parallel_replica_jobs_preserve_report_order(tmp_path):
    result = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=3,
        base_seed=20,
        replica_jobs=2,
    ).run(_context(tmp_path))

    assert result.summary_df["replica_name"].tolist() == ["replica_000", "replica_001", "replica_002"]
    assert result.summary_df["seed"].tolist() == [20, 21, 22]
    payload = json.loads((tmp_path / "replicas_protocol.json").read_text(encoding="utf-8"))
    assert payload["replica_jobs"] == 2
    assert payload["replica_names"] == ["replica_000", "replica_001", "replica_002"]


@pytest.mark.order(502)
def test_replicated_protocol_resume_completed_reuses_existing_replicas(tmp_path):
    ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=2,
        base_seed=30,
    ).run(_context(tmp_path))

    result = ReplicatedStagedProtocol(
        stages=_fake_stages(dudez_fail=0),
        n_replicas=2,
        base_seed=30,
        resume_completed=True,
    ).run(_context(tmp_path))

    assert result.failed_replicas == []
    assert result.summary_df["replica_name"].tolist() == ["replica_000", "replica_001"]
    assert result.summary_df["seed"].tolist() == [30, 31]
    payload = json.loads((tmp_path / "replicas_protocol.json").read_text(encoding="utf-8"))
    assert payload["resume_completed"] is True


@pytest.mark.order(503)
def test_replicated_protocol_resume_ignores_incomplete_replica(tmp_path):
    replica_dir = tmp_path / "replica_000"
    replica_dir.mkdir(parents=True)
    (replica_dir / "protocol_log.json").write_text(
        json.dumps({"stages": [{"name": "pdbbind_optuna", "result": {}}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="replica_000"):
        ReplicatedStagedProtocol(
            stages=_fake_stages(dudez_fail=0),
            n_replicas=1,
            base_seed=30,
            resume_completed=True,
        ).run(_context(tmp_path))


@pytest.mark.order(503)
def test_replicated_protocol_resume_cleans_incomplete_replica_output(tmp_path):
    replica_dir = tmp_path / "replica_000"
    replica_dir.mkdir(parents=True)
    stale_file = replica_dir / "optuna.db"
    stale_file.write_text("stale failed study", encoding="utf-8")

    result = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=1,
        base_seed=30,
        resume_completed=True,
    ).run(_context(tmp_path))

    assert result.failed_replicas == []
    assert not stale_file.exists()
    assert (replica_dir / "protocol_log.json").is_file()
    assert (replica_dir / "pdbbind" / "pdbbind_best.pt").is_file()


@pytest.mark.order(503)
def test_replicated_protocol_resume_deletes_stale_shared_optuna_study(tmp_path):
    storage = f"sqlite:///{tmp_path}/optuna.db"
    optuna.create_study(
        direction="minimize",
        study_name="PDBbind_Regression_Optimization_replica_000",
        storage=storage,
    )

    result = ReplicatedStagedProtocol(
        stages=_fake_stages(),
        n_replicas=1,
        base_seed=30,
        resume_completed=True,
    ).run(_context(tmp_path))

    assert result.failed_replicas == []
    study_names = [study.study_name for study in optuna.get_all_study_summaries(storage=storage)]
    assert "PDBbind_Regression_Optimization_replica_000" not in study_names
