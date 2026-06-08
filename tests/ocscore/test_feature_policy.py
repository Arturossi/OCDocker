#!/usr/bin/env python3

"""Tests for OCScore feature-ablation policies."""

from __future__ import annotations

import pytest

from OCDocker.OCScore.Utils.FeaturePolicy import FeaturePolicy
from OCDocker.OCScore.Utils.FeaturePolicy import apply_feature_policy
from OCDocker.OCScore.Utils.FeaturePolicy import discover_candidate_model_features
from OCDocker.OCScore.Utils.FeaturePolicy import discover_feature_policies
from OCDocker.OCScore.Utils.FeaturePolicy import resolve_requested_feature_policies


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
SCORING_FEATURES = ["plants_plp", "vina_vina", "smina_vinardo", "gnina_ad4_scoring", "oddt_rfscore_v1"]
FEATURES = [
    "receptor_countA",
    "receptor_countG",
    *SHAPE_FEATURES,
    "ligand_MolWt",
    "ligand_BertzCT",
    *SCORING_FEATURES,
]


def _policy(name: str):
    return discover_feature_policies().policies[name]


def test_bundled_ablation_policies_are_discovered():
    discovered = discover_feature_policies()
    assert "full_ocscore" in discovered.policies
    assert "no_pmi" in discovered.policies
    assert discovered.policies["no_pmi"].source_kind == "bundled"
    assert discovered.policies["no_pmi"].source_path.suffix == ".yml"


def test_user_policy_dir_is_added_and_yaml_is_not_required(tmp_path):
    custom = tmp_path / "Ablations"
    custom.mkdir()
    (custom / "my_custom.yml").write_text(
        "name: my_custom\ninclude_patterns:\n  - ligand_*\n",
        encoding="utf-8",
    )
    (custom / "ignored.yaml").write_text(
        "name: ignored\ninclude_patterns:\n  - '*'\n",
        encoding="utf-8",
    )

    discovered = discover_feature_policies(policy_dirs=[custom])

    assert "my_custom" in discovered.policies
    assert "ignored" not in discovered.policies
    assert discovered.policies["my_custom"].source_kind == "user_dir"


def test_duplicate_policy_names_fail_and_list_paths(tmp_path):
    custom = tmp_path / "Ablations"
    custom.mkdir()
    duplicate = custom / "no_pmi.yml"
    duplicate.write_text("name: no_pmi\ninclude_patterns:\n  - '*'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Feature policy name conflict") as err:
        discover_feature_policies(policy_dirs=[custom])

    message = str(err.value)
    assert "no_pmi.yml" in message
    assert str(duplicate) in message


def test_unknown_policy_fails_with_available_names():
    with pytest.raises(ValueError, match="Unknown feature policy") as err:
        resolve_requested_feature_policies(requested_names=["no_pmii"])

    message = str(err.value)
    assert "full_ocscore" in message
    assert "no_pmi" in message


def test_explicit_policy_yml_can_run_without_python_changes(tmp_path):
    policy_path = tmp_path / "one_off.yml"
    policy_path.write_text(
        "name: one_off\ninclude_patterns:\n  - receptor_*\n",
        encoding="utf-8",
    )

    policies, _ = resolve_requested_feature_policies(explicit_ymls=[policy_path])

    assert [policy.name for policy in policies] == ["one_off"]


def test_run_all_feature_policies_includes_custom_directory(tmp_path):
    custom = tmp_path / "Ablations"
    custom.mkdir()
    (custom / "my_custom.yml").write_text(
        "name: my_custom\ninclude_patterns:\n  - ligand_*\n",
        encoding="utf-8",
    )

    policies, _ = resolve_requested_feature_policies(policy_dirs=[custom], run_all=True)
    names = [policy.name for policy in policies]

    assert "full_ocscore" in names
    assert "my_custom" in names


def test_full_ocscore_includes_all_candidate_features_not_metadata():
    columns = ["receptor", "label", "kind", "experimental", *FEATURES, "unmatched_project_note"]
    discovery = discover_candidate_model_features(
        columns,
        non_feature_columns=["receptor", "label", "kind", "experimental"],
    )
    application = apply_feature_policy(_policy("full_ocscore"), discovery.candidate_features)

    assert application.final_candidate_features_before_reduction == FEATURES
    assert "receptor" not in application.final_candidate_features_before_reduction
    assert "label" not in application.final_candidate_features_before_reduction
    assert "experimental" not in application.final_candidate_features_before_reduction
    assert "unmatched_project_note" not in application.final_candidate_features_before_reduction


def test_no_pmi_excludes_exact_pmi_features_when_present():
    application = apply_feature_policy(_policy("no_pmi"), FEATURES)
    final = application.final_candidate_features_before_reduction

    assert "ligand_PMI1" not in final
    assert "ligand_PMI2" not in final
    assert "ligand_PMI3" not in final
    assert application.excluded_features_found == ["ligand_PMI1", "ligand_PMI2", "ligand_PMI3"]


def test_no_shape_core_excludes_all_shape_core_features():
    final = apply_feature_policy(_policy("no_shape_core"), FEATURES).final_candidate_features_before_reduction
    assert all(feature not in final for feature in SHAPE_FEATURES)


def test_shape_only_includes_only_shape_descriptors():
    final = apply_feature_policy(_policy("shape_only"), FEATURES).final_candidate_features_before_reduction
    assert final == SHAPE_FEATURES


def test_scoring_function_only_includes_only_scoring_patterns():
    final = apply_feature_policy(_policy("scoring_function_only"), FEATURES).final_candidate_features_before_reduction
    assert final == SCORING_FEATURES


def test_ligand_plus_scoring_function_excludes_receptor_columns():
    final = apply_feature_policy(_policy("ligand_plus_scoring_function"), FEATURES).final_candidate_features_before_reduction
    assert all(not feature.startswith("receptor_") for feature in final)
    assert any(feature.startswith("ligand_") for feature in final)
    assert any(feature in SCORING_FEATURES for feature in final)


def test_no_scoring_function_excludes_scoring_patterns():
    final = apply_feature_policy(_policy("no_scoring_function"), FEATURES).final_candidate_features_before_reduction
    assert all(feature not in final for feature in SCORING_FEATURES)


def test_missing_include_features_fail():
    policy = FeaturePolicy(name="bad", include_features=("missing_feature",))
    with pytest.raises(ValueError, match="missing include_features"):
        apply_feature_policy(policy, FEATURES)


def test_missing_exclude_features_warn_and_are_recorded_by_default():
    policy = FeaturePolicy(name="soft_missing", include_patterns=("*",), exclude_features=("missing_feature",))
    application = apply_feature_policy(policy, FEATURES)
    assert application.missing_exclude_features == ["missing_feature"]


def test_missing_exclude_features_can_fail():
    policy = FeaturePolicy(
        name="strict_missing",
        include_patterns=("*",),
        exclude_features=("missing_feature",),
        allow_missing_exclude_features=False,
    )
    with pytest.raises(ValueError, match="exclude_features not present"):
        apply_feature_policy(policy, FEATURES)


def test_patterns_with_no_matches_are_recorded():
    policy = FeaturePolicy(name="pattern", include_patterns=("ligand_*",), exclude_patterns=("absent_*",))
    application = apply_feature_policy(policy, FEATURES)
    assert application.patterns_with_no_matches == ["absent_*"]


def test_feature_order_is_deterministic():
    policy = FeaturePolicy(
        name="ordered",
        include_features=("ligand_MolWt", "receptor_countA"),
        include_patterns=("vina_*", "ligand_PMI*"),
    )
    first = apply_feature_policy(policy, FEATURES).final_candidate_features_before_reduction
    second = apply_feature_policy(policy, FEATURES).final_candidate_features_before_reduction
    assert first == second
    assert first == ["receptor_countA", "ligand_PMI1", "ligand_PMI2", "ligand_PMI3", "ligand_MolWt", "vina_vina"]
