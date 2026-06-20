#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench ablation-design API helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

import pytest

from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_METADATA_JSON
from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_from_mapping
from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_to_yaml_text
from OCDocker.Workbench import ablation_container_paths
from OCDocker.Workbench import build_ablation_design_context
from OCDocker.Workbench import build_workbench_api_payload
from OCDocker.Workbench import discover_ablation_input_features
from OCDocker.Workbench import plan_ablation_design
from OCDocker.Workbench import preview_ablation_design
from OCDocker.Workbench import write_ablation_design_policy
from OCDocker.Workbench import resolve_ocscore_layout_root

# License
###############################################################################
'''OCDocker
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
'''

# Functions
###############################################################################
## Public ##


def test_feature_policy_draft_helpers_round_trip() -> None:
    '''Draft feature-policy helpers round-trip name and YAML fields.'''

    payload = {
        "name": "custom_no_pmi",
        "description": "Remove PMI descriptors.",
        "include_patterns": ["*"],
        "exclude_features": ["ligand_PMI1", "ligand_PMI2"],
    }
    policy = feature_policy_from_mapping(payload)
    assert policy.name == "custom_no_pmi"
    yaml_text = feature_policy_to_yaml_text(payload)
    assert "custom_no_pmi" in yaml_text
    assert "ligand_PMI1" in yaml_text


def test_feature_policy_draft_helpers_include_features() -> None:
    '''Draft feature-policy helpers preserve explicit include_features lists.'''

    payload = {
        "name": "shape_only",
        "description": "Shape descriptors only.",
        "include_features": ["ligand_PMI1", "ligand_PMI2"],
    }
    policy = feature_policy_from_mapping(payload)
    yaml_text = feature_policy_to_yaml_text(payload)

    assert list(policy.include_features) == ["ligand_PMI1", "ligand_PMI2"]
    assert "include_features:" in yaml_text
    assert "ligand_PMI1" in yaml_text


def test_preview_ablation_design_uses_candidate_features(tmp_path) -> None:
    '''Preview applies draft policies against workspace candidate features.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    replica = tmp_path / "replica_000"
    replica.mkdir()
    metadata = {
        "candidate_features_before_policy": [
            "ligand_PMI1",
            "ligand_PMI2",
            "ligand_MW",
            "vina_vina",
        ]
    }
    (replica / FEATURE_POLICY_METADATA_JSON).write_text(json.dumps(metadata), encoding="utf-8")

    payload = preview_ablation_design(
        tmp_path,
        {
            "policy": {
                "name": "custom_no_pmi",
                "include_patterns": ["*"],
                "exclude_features": ["ligand_PMI1", "ligand_PMI2"],
            }
        },
    )

    assert payload["preview_available"] is True
    assert payload["kept_feature_count"] == 2
    assert payload["excluded_feature_count"] == 2
    assert "ligand_MW" in payload["kept_features_sample"]


def test_plan_ablation_design_emits_command(tmp_path) -> None:
    '''Plan emits a feature-policy train command and preflight report.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    protocol = tmp_path / "production.yml"
    protocol.write_text("name: production\n", encoding="utf-8")
    raw_input = tmp_path / "raw"
    raw_input.mkdir()

    payload = plan_ablation_design(
        tmp_path,
        {
            "policy": {
                "name": "custom_shape_only",
                "include_patterns": ["ligand_shape_*"],
                "description": "Shape only test.",
            },
            "protocol": str(protocol),
            "raw_input_dir": str(raw_input),
            "output_dir": str(tmp_path / "ablations" / "custom_shape_only"),
            "policy_yml_path": str(tmp_path / "Ablations" / "custom_shape_only.yml"),
        },
    )

    assert payload["ok"] is True
    assert "--feature-policy" in payload["planned_command"]
    assert "custom_shape_only" in payload["planned_command"]
    assert "--feature-policy-yml" in payload["planned_command"]
    assert payload["preflight"]["ready"] is False


def test_build_ablation_design_context_includes_catalog(tmp_path) -> None:
    '''Design context exposes bundled feature-policy catalog entries.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    context = build_ablation_design_context(tmp_path)
    assert context["ok"] is True
    assert context["catalog"]
    assert any(item["name"] == "no_pmi" for item in context["catalog"])


def test_discover_ablation_input_features_from_pdbbind_csv(tmp_path) -> None:
    '''Input discovery strips metadata and returns grouped descriptor columns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pdbbind_path = tmp_path / "raw_pdbbind.csv"
    pdbbind_path.write_text(
        "receptor,ligand,name,type,db,experimental,ligand_MW,vina_vina,receptor_SASA\n"
        "r1,l1,n1,t,pdb,7.1,180.0,8.2,1200.0\n",
        encoding="utf-8",
    )

    payload = discover_ablation_input_features({"pdbbind_input": str(pdbbind_path)})

    assert payload["ok"] is True
    assert payload["columns_only"] is True
    assert payload["feature_source"] == "pdbbind"
    assert "experimental" in payload["target_columns"]
    assert "receptor" in payload["metadata_columns"]
    assert "ligand_MW" in payload["feature_groups"]["ligand"]
    assert "vina_vina" in payload["feature_groups"]["scoring"]
    assert "ligand_MW" in payload["candidate_features"]
    assert "experimental" not in payload["candidate_features"]


def test_discover_ablation_input_features_from_dudez_csv(tmp_path) -> None:
    '''DUDEz-only input discovery uses DUDEz descriptor columns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    dudez_path = tmp_path / "raw_dudez.csv"
    dudez_path.write_text(
        "receptor,ligand,name,type,db,kind,ligand_MW,plants_plp\n"
        "r1,l1,n1,t,dude,l,180.0,7.5\n",
        encoding="utf-8",
    )

    payload = discover_ablation_input_features(
        {"dudez_input": str(dudez_path), "feature_source": "dudez"},
    )

    assert payload["columns_only"] is True
    assert payload["feature_source"] == "dudez"
    assert "kind" in payload["metadata_columns"]
    assert "plants_plp" in payload["candidate_features"]


def test_discover_ablation_input_features_from_served_raw_prepare(tmp_path) -> None:
    '''Feature discovery uses raw_prepare/ under the served Workbench root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    (raw_prepare / "raw_pdbbind.csv").write_text(
        "receptor,ligand,name,type,db,experimental,ligand_MW,vina_vina\n"
        "r1,l1,n1,t,pdb,7.1,180.0,8.2\n",
        encoding="utf-8",
    )
    (raw_prepare / "raw_dudez.csv").write_text(
        "receptor,ligand,name,type,db,kind,ligand_MW,vina_vina\n"
        "r2,l2,n2,t,dude,l,180.0,7.5\n",
        encoding="utf-8",
    )

    payload = discover_ablation_input_features({}, root=tmp_path)

    assert payload["ok"] is True
    assert payload["columns_only"] is True
    assert payload["auto_discovered"] is True
    assert payload["resolved_inputs"]["raw_input_dir"] == str(raw_prepare.resolve())
    assert "ligand_MW" in payload["candidate_features"]


def test_build_ablation_design_context_reports_discovered_inputs(tmp_path) -> None:
    '''Design context exposes raw_prepare defaults for the UI.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    (raw_prepare / "raw_pdbbind.csv").write_text(
        "receptor,ligand,name,type,db,experimental,ligand_MW\n"
        "r1,l1,n1,t,pdb,7.1,180.0\n",
        encoding="utf-8",
    )
    (raw_prepare / "raw_dudez.csv").write_text(
        "receptor,ligand,name,type,db,kind,ligand_MW\n"
        "r2,l2,n2,t,dude,l,180.0\n",
        encoding="utf-8",
    )

    context = build_ablation_design_context(tmp_path)

    assert context["discovered_inputs"]["ok"] is True
    assert context["discovered_inputs"]["raw_input_dir"] == str(raw_prepare.resolve())
    assert context["discovered_inputs"]["pdbbind_input"] == str((raw_prepare / "raw_pdbbind.csv").resolve())
    assert context["discovered_inputs"]["dudez_input"] == str((raw_prepare / "raw_dudez.csv").resolve())


def test_api_index_lists_ablation_design_endpoints(tmp_path) -> None:
    '''Workbench API index exposes ablation-design endpoints.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    payload = build_workbench_api_payload(tmp_path, "/api")
    assert "/api/ablation-design" in payload["endpoints"]
    assert "/api/ablation-design/preview" in payload["endpoints"]
    assert "/api/ablation-design/plan" in payload["endpoints"]
    assert "/api/ablation-design/features" in payload["endpoints"]
    assert "/api/ablation-design/write" in payload["endpoints"]


def test_write_ablation_design_policy_writes_under_layout_root(tmp_path) -> None:
    '''Write endpoint saves YAML under the served workspace layout root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    target = tmp_path / "Ablations" / "custom_test.yml"
    payload = write_ablation_design_policy(
        tmp_path,
        {
            "confirm": True,
            "overwrite": True,
            "policy_yml_path": str(target),
            "policy": {
                "name": "custom_test",
                "include_patterns": ["*"],
                "exclude_features": ["ligand_PMI1"],
            },
        },
    )

    assert payload["ok"] is True
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "custom_test" in text
    assert "ligand_PMI1" in text


def test_write_ablation_design_policy_requires_confirm(tmp_path) -> None:
    '''Write endpoint rejects requests without explicit confirmation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="confirm"):
        write_ablation_design_policy(
            tmp_path,
            {
                "policy_yml_path": str(tmp_path / "Ablations" / "custom_test.yml"),
                "policy": {"name": "custom_test", "include_patterns": ["*"]},
            },
        )


def test_resolve_ocscore_layout_root_prefers_train_child(tmp_path) -> None:
    '''Layout-root resolution follows strict train/ child conventions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    train_root = tmp_path / "train"
    train_root.mkdir()
    (train_root / "replica_000").mkdir()
    assert resolve_ocscore_layout_root(tmp_path) == train_root
    assert ablation_container_paths(tmp_path) == ()
