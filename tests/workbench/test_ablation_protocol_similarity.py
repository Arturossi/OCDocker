#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for ablation protocol similarity analysis.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

import pytest

from OCDocker.OCScore.Utils.FeaturePolicy import FeaturePolicy
from OCDocker.OCScore.Utils.FeaturePolicy import apply_feature_policy
from OCDocker.OCScore.Utils.FeaturePolicy import discover_feature_policies
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_ablation_protocol_similarity_analysis
from OCDocker.Workbench import build_workbench_api_payload
from OCDocker.Workbench import write_model
from OCDocker.Workbench.AblationProtocolSimilarity import _build_family_definitions
from OCDocker.Workbench.AblationProtocolSimilarity import _cluster_labels
from OCDocker.Workbench.AblationProtocolSimilarity import _jaccard
from OCDocker.Workbench.AblationProtocolSimilarity import _resolve_protocol
from OCDocker.Workbench.AblationProtocolSimilarity import _similarity_matrix

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

SHAPE_FEATURES = [
    "ligand_PMI1",
    "ligand_PMI2",
    "ligand_PMI3",
    "ligand_NPR1",
    "ligand_NPR2",
    "ligand_Asphericity",
    "ligand_Eccentricity",
    "ligand_InertialShapeFactor",
    "ligand_RadiusOfGyration",
    "ligand_SpherocityIndex",
]
SCORING_FEATURES = ["plants_plp", "vina_vina", "smina_vinardo"]
FEATURES = [
    "receptor_countA",
    "receptor_SASA",
    *SHAPE_FEATURES,
    "ligand_MW",
    "ligand_BertzCT",
    "ligand_AUTOCORR2D_1",
    *SCORING_FEATURES,
]

# Functions
###############################################################################
## Private ##


def _policy(name: str):
    return discover_feature_policies().policies[name]


def _write_result_pair(root, run_id: str, source_path: str, metrics: dict) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id=run_id,
            spec_type="ocscore_ablation",
            name=run_id,
            status="completed",
            workspace=source_path,
            metadata={"adopted": True, "source_path": source_path},
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(run_id=run_id, status="completed", metrics=metrics),
    )


def _ligand_only_explicit_policy() -> FeaturePolicy:
    ligand_features = [feature for feature in FEATURES if feature.startswith("ligand_")]
    return FeaturePolicy(
        name="ligand_only_explicit",
        description="Explicit ligand feature list.",
        include_features=tuple(ligand_features),
        include_patterns=(),
        exclude_features=(),
        exclude_patterns=(),
        allow_missing_exclude_features=True,
        allow_empty_policy=False,
        source_path=__file__,
        source_kind="explicit_yml",
        source_hash="test",
    )


## Public ##


def test_ligand_pattern_and_explicit_list_have_identical_expansion() -> None:
    definitions = _build_family_definitions(FEATURES)
    pattern = _resolve_protocol(_policy("ligand_only"), FEATURES, definitions)
    explicit = _resolve_protocol(_ligand_only_explicit_policy(), FEATURES, definitions)
    assert pattern.expanded_features == explicit.expanded_features
    matrix = _similarity_matrix([pattern.expanded_features, explicit.expanded_features])
    assert matrix[0][1] == pytest.approx(1.0)


def test_no_pmi_differs_from_full_ocscore_and_removes_pmi_features() -> None:
    definitions = _build_family_definitions(FEATURES)
    full = _resolve_protocol(_policy("full_ocscore"), FEATURES, definitions)
    no_pmi = _resolve_protocol(_policy("no_pmi"), FEATURES, definitions)
    assert _jaccard(full.expanded_features, no_pmi.expanded_features) < 1.0
    assert "ligand_PMI1" in full.expanded_features - no_pmi.expanded_features


def test_ligand_plus_scoring_adds_scoring_family_vs_ligand_only() -> None:
    definitions = _build_family_definitions(FEATURES)
    ligand_only = _resolve_protocol(_policy("ligand_only"), FEATURES, definitions)
    combo = _resolve_protocol(_policy("ligand_plus_scoring_function"), FEATURES, definitions)
    ligand_family = next(state for state in combo.family_states if state.family_id == "ligand")
    scoring_family = next(state for state in combo.family_states if state.family_id == "scoring")
    assert ligand_family.present
    assert scoring_family.present
    assert combo.expanded_features - ligand_only.expanded_features


def test_ligand_family_rollup_counts_expanded_members() -> None:
    definitions = _build_family_definitions(FEATURES)
    resolved = _resolve_protocol(_policy("ligand_only"), FEATURES, definitions)
    ligand_family = next(state for state in resolved.family_states if state.family_id == "ligand")
    assert ligand_family.present
    assert ligand_family.member_count == len([feature for feature in FEATURES if feature.startswith("ligand_")])


def test_cluster_labels_are_stable_for_fixed_distance_matrix() -> None:
    distance = [
        [0.0, 0.2, 0.8],
        [0.2, 0.0, 0.75],
        [0.8, 0.75, 0.0],
    ]
    labels_a, order_a = _cluster_labels(distance)
    labels_b, order_b = _cluster_labels(distance)
    assert labels_a == labels_b
    assert order_a == order_b
    assert len(labels_a) == 3


def test_build_analysis_without_candidates_is_preview_unavailable(tmp_path) -> None:
    analysis = build_ablation_protocol_similarity_analysis(tmp_path, include_catalog_only=True)
    assert analysis.preview_available is False
    assert analysis.protocol_count == 0
    assert analysis.message


def test_build_analysis_with_workspace_runs_and_metric(tmp_path) -> None:
    _write_result_pair(
        tmp_path,
        "train",
        "/source/output/train",
        {"auc": 0.88},
    )
    _write_result_pair(
        tmp_path,
        "shape_only",
        "/source/output/train/ablations/shape_only",
        {"auc": 0.91},
    )
    _write_result_pair(
        tmp_path,
        "no_shape_core",
        "/source/output/train/ablations/no_shape_core",
        {"auc": 0.82},
    )

    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        metric="auc:max",
        include_catalog_only=False,
        max_depth=2,
    )

    assert analysis.preview_available is True
    assert analysis.protocol_count >= 2
    assert analysis.similarity_matrix
    assert len(analysis.protocol_order) == analysis.protocol_count
    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["shape_only"].metric_value == pytest.approx(0.91)
    assert analysis.cluster_summaries


def test_build_analysis_without_runs_still_shows_full_catalog(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    assert analysis.preview_available is True
    assert analysis.protocol_count > 1
    assert "full_ocscore" in analysis.protocol_order


def test_workspace_ablation_folders_mark_study_present_without_completed_run(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    ablations = tmp_path / "ablations"
    (ablations / "shape_only" / "replica_0").mkdir(parents=True)
    (ablations / "no_pmi" / "replica_0").mkdir(parents=True)

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=True,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert analysis.protocol_count > 3
    assert by_name["shape_only"].study_present is True
    assert by_name["no_pmi"].study_present is True
    assert by_name["shape_only"].run_id is None
    assert by_name["no_pmi"].run_id is None


def test_fully_completed_study_gets_run_id(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    replica = tmp_path / "ablations" / "shape_only" / "replica_1"
    replica.mkdir(parents=True)
    pdbbind = replica / "pdbbind"
    pdbbind.mkdir()
    (pdbbind / "pdbbind_best.pt").write_bytes(b"pt")
    dudez = replica / "dudez"
    dudez.mkdir()
    (dudez / "dudez_best.pt").write_bytes(b"pt")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["shape_only"].study_present is True
    assert by_name["shape_only"].run_id == "shape_only"


def test_build_analysis_catalog_only_includes_policies_without_runs(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    filtered = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )
    catalog = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=True,
        max_depth=2,
    )

    assert catalog.protocol_count >= filtered.protocol_count


def test_workbench_api_exposes_protocol_similarity_endpoint(tmp_path) -> None:
    payload = build_workbench_api_payload(tmp_path, "/api")
    assert "/api/ablation-protocol-similarity" in payload["endpoints"]


def test_workbench_api_protocol_similarity_payload(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    payload = build_workbench_api_payload(
        tmp_path,
        "/api/ablation-protocol-similarity",
        query={"include_catalog_only": ["true"]},
    )

    assert payload["preview_available"] is True
    assert payload["protocol_count"] > 0
    assert payload["similarity_matrix"]


def test_metadata_included_features_found_fallback(tmp_path) -> None:
    study = tmp_path / "ablations" / "no_pmi"
    replica = study / "replica_1"
    replica.mkdir(parents=True)
    metadata = {
        "feature_policy_name": "no_pmi",
        "included_features_found": ["ligand_MW", "vina_vina", "receptor_SASA"],
    }
    (replica / "feature_policy_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["no_pmi"].expanded_feature_count == 3


def test_study_metadata_is_found_in_later_replica(tmp_path) -> None:
    study = tmp_path / "ablations" / "no_pmi"
    (study / "replica_1").mkdir(parents=True)
    replica_two = study / "replica_2"
    replica_two.mkdir()
    metadata = {
        "feature_policy_name": "no_pmi",
        "final_candidate_features_before_reduction": ["ligand_MW", "vina_vina"],
    }
    (replica_two / "feature_policy_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["no_pmi"].expanded_feature_count == 2


def test_all_ablation_folders_are_selected_without_catalog(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    for name in ("shape_only", "no_pmi", "no_shape_core"):
        (tmp_path / "ablations" / name).mkdir(parents=True)

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    assert set(analysis.protocol_order) >= {"shape_only", "no_pmi", "no_shape_core"}


def test_executed_study_uses_feature_policy_metadata_expansion(tmp_path) -> None:
    study = tmp_path / "ablations" / "shape_only"
    replica = study / "replica_1"
    replica.mkdir(parents=True)
    metadata = {
        "feature_policy_name": "shape_only",
        "feature_policy_description": "Shape only.",
        "feature_policy_source_kind": "bundled",
        "feature_policy_source_path": "/policies/shape_only.yml",
        "final_candidate_features_before_reduction": [
            "ligand_PMI1",
            "ligand_PMI2",
            "ligand_NPR1",
        ],
    }
    (replica / "feature_policy_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        include_catalog_only=False,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["shape_only"].expanded_feature_count == 3


def test_fast_study_level_metric_overlay(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    study = tmp_path / "ablations" / "shape_only"
    (study / "replica_0").mkdir(parents=True)
    (study / "staged_optuna_protocol.json").write_text(
        json.dumps(
            {
                "aggregate_summary": {
                    "metrics": {
                        "auc": {"mean": 0.91, "std": 0.01},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        metric="auc:max",
        include_catalog_only=False,
        max_depth=2,
    )

    by_name = {entry.policy_name: entry for entry in analysis.protocols}
    assert by_name["shape_only"].metric_value == pytest.approx(0.91)


def test_unknown_metric_returns_issue_not_exception(tmp_path) -> None:
    raw_prepare = tmp_path / "raw_prepare"
    raw_prepare.mkdir()
    columns = ",".join(["receptor", "label", "experimental", *FEATURES])
    (raw_prepare / "raw_pdbbind.csv").write_text(f"{columns}\n", encoding="utf-8")
    (raw_prepare / "raw_dudez.csv").write_text(f"{columns}\n", encoding="utf-8")

    analysis = build_ablation_protocol_similarity_analysis(
        tmp_path,
        metric="not_a_real_metric:max",
        include_catalog_only=True,
        max_depth=2,
    )

    assert analysis.preview_available is True
    assert all(entry.metric_value is None for entry in analysis.protocols)
