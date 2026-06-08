#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for staged OCScore protocol YAML files.'''

# Imports
###############################################################################
import pytest

from OCDocker.OCScore.Optimization.StagedTrainProtocol import DEFAULT_ABLATION_VARIANTS
from OCDocker.OCScore.Optimization.StagedTrainProtocol import bundled_protocol_names
from OCDocker.OCScore.Optimization.StagedTrainProtocol import load_staged_train_protocol
from OCDocker.OCScore.Optimization.StagedTrainProtocol import normalize_ablation_variant_name
from OCDocker.OCScore.Optimization.StagedTrainProtocol import resolve_protocol_path

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


@pytest.mark.order(411)
def test_bundled_protocol_names_include_production_and_smoke_test():
    names = bundled_protocol_names()
    assert "production" in names
    assert "smoke-test" in names
    assert "development" in names
    assert "example" in names


@pytest.mark.order(412)
def test_smoke_test_protocol_is_fast():
    protocol = load_staged_train_protocol(resolve_protocol_path("smoke-test"))
    assert protocol.name == "smoke-test"
    assert protocol.replicas == 1
    assert protocol.pdbbind.trials == 3
    assert protocol.dudez.scaling_strategy == "pdbbind_scaler"


@pytest.mark.order(413)
def test_production_protocol_budgets_and_split():
    protocol = load_staged_train_protocol(resolve_protocol_path("production"))
    assert protocol.name == "production"
    assert protocol.pdbbind.trials >= 50
    assert protocol.dudez.trials >= 50
    assert protocol.replicas >= 3
    split = protocol.pdbbind_split_config()
    assert split.strategy == "receptor_heldout"
    assert protocol.dudez.scaling_strategy == "pdbbind_scaler"
    assert protocol.reporting.generate_final_report is True
    assert protocol.ablation.enabled is False
    assert protocol.ablation.variants == DEFAULT_ABLATION_VARIANTS


@pytest.mark.order(414)
def test_example_protocol_loads():
    protocol = load_staged_train_protocol(resolve_protocol_path("example"))
    assert protocol.name == "my-custom-run"
    assert protocol.replicas == 2


@pytest.mark.order(415)
def test_production_protocol_budget_validation_fails_when_edited(tmp_path):
    production_path = resolve_protocol_path("production")
    edited = tmp_path / "production-low.yml"
    edited.write_text(
        production_path.read_text(encoding="utf-8").replace("trials: 50", "trials: 5", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="production-claim budget requirements"):
        load_staged_train_protocol(edited)


@pytest.mark.order(416)
def test_ablation_variant_aliases_normalize():
    assert normalize_ablation_variant_name("Ligand-only") == "ligand_only"
    assert normalize_ablation_variant_name("SF") == "sf_only"
    assert normalize_ablation_variant_name("receptor-SF") == "receptor_sf"


@pytest.mark.order(417)
def test_protocol_parallelism_fields_load_from_yaml(tmp_path):
    protocol_path = tmp_path / "parallel.yml"
    protocol_path.write_text(
        """
name: parallel-test
description: test parallelism fields
replicas: 3
seed: 42
pdbbind:
  target_column: experimental
  trials: 5
  epochs: 10
  n_jobs: 2
  search_phase: full
  enable_pruning: true
  split:
    strategy: receptor_heldout
    train_size: 0.6
    validation_size: 0.2
    test_size: 0.2
dudez:
  kind_column: kind
  positive_kind: ligands
  negative_kind: decoys
  trials: 5
  epochs: 10
  n_jobs: 3
  primary_metric: BEDROC
  scaling_strategy: pdbbind_scaler
  ignore_unknown_kind: false
runtime:
  use_gpu: false
  pdbbind_only: false
  replica_jobs: 2
  resume_completed: true
ablation:
  enabled: false
  variants:
    - ligand_only
reporting:
  generate_final_report: false
  run_leakage_audit: false
  run_baselines: false
  calibration_report_mode: ranking_only
""",
        encoding="utf-8",
    )

    protocol = load_staged_train_protocol(protocol_path)

    assert protocol.pdbbind.n_jobs == 2
    assert protocol.dudez.n_jobs == 3
    assert protocol.runtime.replica_jobs == 2
    assert protocol.runtime.resume_completed is True
    assert protocol.budget_dict()["pdbbind_n_jobs"] == 2
    assert protocol.budget_dict()["dudez_n_jobs"] == 3
    assert protocol.budget_dict()["replica_jobs"] == 2
    assert protocol.budget_dict()["resume_completed"] is True


@pytest.mark.order(418)
def test_protocol_parallelism_fields_must_be_positive(tmp_path):
    protocol_path = tmp_path / "bad-parallel.yml"
    protocol_path.write_text(
        """
name: bad-parallel
replicas: 1
seed: 42
pdbbind:
  trials: 5
  epochs: 10
  n_jobs: 0
  split:
    strategy: receptor_heldout
dudez:
  trials: 5
  epochs: 10
  n_jobs: 1
runtime:
  use_gpu: false
  pdbbind_only: false
  replica_jobs: 1
reporting:
  generate_final_report: false
  run_leakage_audit: false
  run_baselines: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pdbbind.n_jobs"):
        load_staged_train_protocol(protocol_path)
