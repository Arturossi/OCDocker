#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for the OCDocker workbench schema and planning layer.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest
from pydantic import ValidationError

from OCDocker.Workbench import FeaturePolicySelection
from OCDocker.Workbench import OCScoreAblationSpec
from OCDocker.Workbench import OCScoreInputSpec
from OCDocker.Workbench import OCScoreStudySpec
from OCDocker.Workbench import ResourceSpec
from OCDocker.Workbench import SnakemakeWorkflowSpec
from OCDocker.Workbench import VSInputSpec
from OCDocker.Workbench import VSCampaignSpec
from OCDocker.Workbench import build_run_manifest
from OCDocker.Workbench import plan_ocscore_train_command
from OCDocker.Workbench import plan_snakemake_command
from OCDocker.Workbench import plan_vs_campaign_command
from OCDocker.Workbench import read_spec
from OCDocker.Workbench import write_model

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def test_ocscore_input_spec_requires_one_input_mode() -> None:
    '''OCScore training specs fail closed on ambiguous raw-input selection.'''

    with pytest.raises(ValidationError, match="Select exactly one"):
        OCScoreInputSpec(raw_input_dir="raw", merged_input="merged.csv")
    with pytest.raises(ValidationError, match="supplied together"):
        OCScoreInputSpec(pdbbind_input="pdbbind.csv")

    spec = OCScoreInputSpec(pdbbind_input="pdbbind.csv", dudez_input="dudez.csv")
    assert str(spec.pdbbind_input) == "pdbbind.csv"


def test_feature_policy_selection_matches_cli_exclusivity() -> None:
    '''The workbench mirrors OCScore's feature-policy selection rules.'''

    with pytest.raises(ValidationError, match="run_all"):
        FeaturePolicySelection(names=("no_pmi",), run_all=True)
    with pytest.raises(ValidationError, match="unique"):
        FeaturePolicySelection(names=("no_pmi", "no_pmi"))

    selection = FeaturePolicySelection(names="no_pmi,no_shape_core")
    assert selection.names == ("no_pmi", "no_shape_core")


def test_ocscore_train_command_plan_for_named_ablation() -> None:
    '''Ablation specs produce a deterministic ``ocscore train`` command.'''

    spec = OCScoreAblationSpec(
        name="shape-ablation",
        protocol="production",
        inputs=OCScoreInputSpec(raw_input_dir="/data/raw_prepare"),
        output_dir="/data/out/ablations",
        feature_policies=FeaturePolicySelection(names=("no_pmi", "shape_only")),
    )

    plan = plan_ocscore_train_command(spec)

    assert plan.command == (
        "ocdocker",
        "ocscore",
        "train",
        "--protocol",
        "production",
        "--output-dir",
        "/data/out/ablations",
        "--raw-input-dir",
        "/data/raw_prepare",
        "--feature-policy",
        "no_pmi",
        "--feature-policy",
        "shape_only",
    )
    assert plan.writes == (spec.output_dir,)


def test_ocscore_study_command_plan_supports_policy_lookup_paths() -> None:
    '''Study planning preserves custom feature-policy lookup paths.'''

    spec = OCScoreStudySpec(
        name="focused-study",
        protocol="development",
        inputs=OCScoreInputSpec(merged_input="merged_input_dataset.csv"),
        output_dir="out/focused",
        feature_policies=FeaturePolicySelection(
            names=("my_policy",),
            policy_dirs=("policies",),
            policy_ymls=("one_off.yml",),
        ),
    )

    command = plan_ocscore_train_command(spec).command

    assert "--feature-policy-dir" in command
    assert "--feature-policy-yml" in command
    assert command[-2:] == ("--feature-policy", "my_policy")


def test_snakemake_plan_uses_stable_flags_and_config_order() -> None:
    '''Snakemake command planning is stable and does not execute anything.'''

    workflow = SnakemakeWorkflowSpec(
        snakefile="examples/20_Snakefile_ocdocker_granular_pipeline.smk",
        workdir="campaigns/4cfe",
        resources=ResourceSpec(cores=12),
        config={"threads": 4, "sample": "4cfe"},
        dry_run=True,
    )

    plan = plan_snakemake_command(workflow)

    assert plan.command[:5] == (
        "snakemake",
        "-s",
        "examples/20_Snakefile_ocdocker_granular_pipeline.smk",
        "--cores",
        "12",
    )
    assert "--dry-run" in plan.command
    assert plan.command[plan.command.index("--config") + 1 :] == (
        "sample=4cfe",
        "threads=4",
    )


def test_vs_campaign_plan_infers_single_sample_config() -> None:
    '''A single-input VS campaign can seed existing Snakemake config keys.'''

    spec = VSCampaignSpec(
        name="4cfe-screen",
        workspace="campaigns/4cfe",
        workflow=SnakemakeWorkflowSpec(
            snakefile="examples/20_Snakefile_ocdocker_granular_pipeline.smk",
            resources=ResourceSpec(cores=8),
        ),
        inputs=(
            VSInputSpec(
                sample="4cfe",
                receptor="input/4cfe/receptor.pdbqt",
                ligand="input/4cfe/ligand.pdbqt",
                box="input/4cfe/box.txt",
                engines=("vina", "smina"),
            ),
        ),
    )

    plan = plan_vs_campaign_command(spec)

    assert plan.label == "vs_campaign:4cfe-screen"
    assert "sample=4cfe" in plan.command
    assert "engines=vina,smina" in plan.command


def test_workbench_spec_round_trips_through_yaml(tmp_path) -> None:
    '''Specs can be serialized for GUI/CLI handoff and loaded again.

    Parameters
    ----------
    tmp_path : Any
        Input value.
    '''

    spec = OCScoreStudySpec(
        name="production-study",
        protocol="production",
        inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
        output_dir="out/production",
    )

    path = write_model(tmp_path / "study.yml", spec)
    loaded = read_spec(path)

    assert isinstance(loaded, OCScoreStudySpec)
    assert loaded == spec


def test_run_manifest_uses_planned_command() -> None:
    '''Run manifests capture the command that a GUI would later launch.'''

    spec = OCScoreStudySpec(
        name="manifest-study",
        protocol="smoke-test",
        inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
        output_dir="out/smoke",
    )
    plan = plan_ocscore_train_command(spec)

    manifest = build_run_manifest(spec, plan, run_id="run-001")

    assert manifest.status == "defined"
    assert manifest.workspace == spec.output_dir
    assert manifest.command == plan.command
