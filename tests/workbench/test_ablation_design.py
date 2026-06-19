#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for Workbench ablation-design API helpers."""

from __future__ import annotations

import json

from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_METADATA_JSON
from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_from_mapping
from OCDocker.OCScore.Utils.FeaturePolicy import feature_policy_to_yaml_text
from OCDocker.Workbench import build_workbench_api_payload
from OCDocker.Workbench.AblationDesign import build_ablation_design_context
from OCDocker.Workbench.AblationDesign import plan_ablation_design
from OCDocker.Workbench.AblationDesign import preview_ablation_design

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


def test_feature_policy_draft_helpers_round_trip() -> None:
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


def test_preview_ablation_design_uses_candidate_features(tmp_path) -> None:
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
    context = build_ablation_design_context(tmp_path)
    assert context["ok"] is True
    assert context["catalog"]
    assert any(item["name"] == "no_pmi" for item in context["catalog"])


def test_api_index_lists_ablation_design_endpoints(tmp_path) -> None:
    payload = build_workbench_api_payload(tmp_path, "/api")
    assert "/api/ablation-design" in payload["endpoints"]
