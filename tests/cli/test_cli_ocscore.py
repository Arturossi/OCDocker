#!/usr/bin/env python3

"""Tests for ``ocdocker ocscore`` CLI wiring."""

from __future__ import annotations

import argparse
import json
import tarfile

import pytest
import pandas as pd

from pathlib import Path
from types import SimpleNamespace

import OCDocker.CLI as cli
from OCDocker.CLI import ocscore as ocscore_cli
from OCDocker.OCScore.CLI import train as ocscore_train


@pytest.mark.order(451)
def test_ocscore_subcommands_registered():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ocscore", "--help"])


@pytest.mark.order(452)
def test_ocscore_reduce_parser_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "reduce",
        "--pdbbind-archive",
        "/tmp/pdbbind.tar.gz",
        "--dudez-archive",
        "/tmp/dudez.tar.gz",
        "--output-dir",
        "/tmp/out",
    ])
    assert args.ocscore_command == "reduce"
    assert args.pdbbind_archive.endswith("pdbbind.tar.gz")
    assert args.func is not None


@pytest.mark.order(453)
def test_ocscore_train_parser_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "train",
        "--protocol",
        "smoke-test",
        "--raw-input-dir",
        "/tmp/raw_prepare",
        "--output-dir",
        "/tmp/optuna",
    ])
    assert args.ocscore_command == "train"
    assert args.protocol == "smoke-test"


@pytest.mark.order(453)
def test_ocscore_train_parser_feature_policy_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "train",
        "--protocol",
        "production",
        "--raw-input-dir",
        "/tmp/raw_prepare",
        "--output-dir",
        "/tmp/ablations",
        "--feature-policy-dir",
        "/tmp/Ablations",
        "--feature-policy",
        "no_pmi",
        "--feature-policy",
        "shape_only",
        "--feature-policy-yml",
        "/tmp/one_off.yml",
    ])
    assert args.feature_policy == ["no_pmi", "shape_only"]
    assert args.feature_policy_dir == ["/tmp/Ablations"]
    assert args.feature_policy_yml == ["/tmp/one_off.yml"]
    assert args.run_all_feature_policies is False


@pytest.mark.order(453)
def test_ocscore_train_parser_accepts_focused_ablation_policy_names():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "train",
        "--protocol",
        "production",
        "--raw-input-dir",
        "/tmp/raw_prepare",
        "--output-dir",
        "/tmp/ablations",
        "--feature-policy",
        "no_shape_core_no_receptor_length_pair",
        "--feature-policy",
        "ligand_plus_scoring_function_no_shape_core",
    ])
    assert args.feature_policy == [
        "no_shape_core_no_receptor_length_pair",
        "ligand_plus_scoring_function_no_shape_core",
    ]


def test_ocscore_train_parser_run_all_feature_policies_flag():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "train",
        "--protocol",
        "production",
        "--raw-input-dir",
        "/tmp/raw_prepare",
        "--output-dir",
        "/tmp/ablations",
        "--run-all-feature-policies",
    ])
    assert args.run_all_feature_policies is True


@pytest.mark.order(454)
def test_ocscore_score_parser_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "ocscore",
        "score",
        "--export-dir",
        "/tmp/best_model",
        "--raw-archive",
        "/tmp/raw",
        "--output-csv",
        "/tmp/out.csv",
    ])
    assert args.ocscore_command == "score"
    assert args.output_csv.endswith("out.csv")


@pytest.mark.order(455)
def test_cmd_ocscore_dispatches(monkeypatch):
    seen = {}

    def _handler(args: argparse.Namespace) -> int:
        seen["called"] = True
        return 0

    args = argparse.Namespace(func=_handler)
    assert ocscore_cli.cmd_ocscore(args) == 0
    assert seen["called"] is True


@pytest.mark.order(456)
def test_cmd_ocscore_missing_dependency_hint(monkeypatch):
    def _handler(_args: argparse.Namespace) -> int:
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(
        "OCDocker.CLI.common._print_optional_dependency_hint",
        lambda **kwargs: 9,
    )
    args = argparse.Namespace(func=_handler)
    assert ocscore_cli.cmd_ocscore(args) == 9


@pytest.mark.order(459)
def test_ocscore_subcommands_have_no_docking_parent_flags(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ocscore", "reduce", "--help"])
    captured = capsys.readouterr()
    assert "  --conf" not in captured.out
    assert "  --multiprocess" not in captured.out


@pytest.mark.order(460)
def test_top_level_shap_command_removed():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["shap", "--help"])


@pytest.mark.order(461)
def test_ocscore_has_no_archived_shap_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ocscore", "shap-archived", "--help"])

@pytest.mark.order(462)
def test_ocscore_train_selected_features_load_from_directory_and_tar(tmp_path):
    source_dir = tmp_path / "reduction"
    nested = source_dir / "nested"
    nested.mkdir(parents=True)
    (nested / ocscore_train.SELECTED_FEATURES_JSON).write_text(
        json.dumps({"selected_features": ["f1", "f2"]}),
        encoding="utf-8",
    )

    selected, source = ocscore_train.load_selected_features(source_dir)
    assert selected == ["f1", "f2"]
    assert str(source).endswith(ocscore_train.SELECTED_FEATURES_JSON)

    tar_path = tmp_path / "reduction.tar.gz"
    txt_path = tmp_path / ocscore_train.SELECTED_FEATURES_TXT
    txt_path.write_text("f3\nf4\n", encoding="utf-8")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(txt_path, arcname=f"artifacts/{ocscore_train.SELECTED_FEATURES_TXT}")

    selected_tar, source_tar = ocscore_train.load_selected_features(tar_path)
    assert selected_tar == ["f3", "f4"]
    assert str(source_tar).endswith(f"::{ 'artifacts/' + ocscore_train.SELECTED_FEATURES_TXT}")

    with pytest.raises(ValueError, match="selected_features"):
        ocscore_train._parse_selected_features_payload({"wrong": []}, "bad.json")


@pytest.mark.order(463)
def test_ocscore_train_reduced_dataset_split_and_target_cleanup(tmp_path):
    reduced = pd.DataFrame({
        "dataset": ["pdbbind", "DUDEz", "pdbbind", "dude-z"],
        "experimental": [7.1, None, "bad", None],
        "kind": [None, "ligands", None, "decoys"],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    paths: dict[str, str] = {}

    pdbbind, dudez, updated_paths = ocscore_train._split_reduced_dataset(reduced, paths)
    assert len(pdbbind) == 2
    assert len(dudez) == 2
    assert updated_paths["source_column"] == "dataset"

    pdbbind_clean, dropped = ocscore_train.prepare_pdbbind_for_optuna(pdbbind, target_column="experimental")
    assert dropped == 1
    assert pdbbind_clean["experimental"].tolist() == [7.1]

    dudez_clean, counts, dropped_unknown = ocscore_train.prepare_dudez_for_optuna(
        dudez,
        kind_column="kind",
        positive_kind="ligands",
        negative_kind="decoys",
    )
    assert counts == {0: 1, 1: 1}
    assert dropped_unknown == 0
    assert dudez_clean["kind"].tolist() == ["ligands", "decoys"]

    with pytest.raises(ValueError, match="metadata/target columns"):
        ocscore_train.validate_selected_features(pdbbind_clean, dudez_clean, ["experimental"])


@pytest.mark.order(464)
def test_ocscore_train_summary_row_helpers(tmp_path):
    aggregate = {
        "n_successful_replicas": 2,
        "n_failed_replicas": 0,
        "metrics": {
            "dudez_test_bedroc": {"mean": 0.7, "std": 0.1, "n": 2},
            "pdbbind_test_rmse": {"mean": 1.2, "std": 0.2},
        },
    }
    result = SimpleNamespace(
        aggregate_summary=aggregate,
        replica_results=[object(), object()],
        output_paths={"replicas_summary_csv": "replicas.csv"},
    )

    ablation_row = ocscore_train._build_ablation_summary_row(
        variant="ligand_sf",
        feature_blocks=["ligand", "scoring"],
        output_dir=tmp_path / "ligand_sf",
        selected_features=["f1", "f2"],
        result=result,
        written={"staged_optuna_protocol_json": "protocol.json"},
    )
    assert ablation_row["n_selected_features"] == 2
    assert ablation_row["dudez_test_bedroc_mean"] == 0.7
    assert ocscore_train._ablation_csv_rows([ablation_row])[0]["feature_blocks"] == "ligand+scoring"

    policy = SimpleNamespace(
        name="no_pmi",
        description="drop PMI",
        source_path=Path("policy.yml"),
        source_kind="bundled",
        source_hash="abc",
    )
    policy_row = ocscore_train._build_feature_policy_summary_row(
        policy=policy,
        output_dir=tmp_path / "no_pmi",
        selected_features=["f1", "f2"],
        result=result,
        written={"staged_optuna_protocol_json": "protocol.json"},
        feature_policy_metadata={"kept": 2},
    )
    assert policy_row["feature_policy_name"] == "no_pmi"
    assert policy_row["dudez_test_BEDROC_mean"] == 0.7
    assert policy_row["pdbbind_test_RMSE_std"] == 0.2
    assert "aggregate_summary" not in ocscore_train._feature_policy_csv_rows([policy_row])[0]

    assert ocscore_train._policy_output_dir(tmp_path, policy).name == "no_pmi"
    unsafe = SimpleNamespace(name="../bad")
    with pytest.raises(ValueError, match="not safe"):
        ocscore_train._policy_output_dir(tmp_path, unsafe)

