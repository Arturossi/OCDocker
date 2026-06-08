#!/usr/bin/env python3

"""Tests for ``ocdocker ocscore`` CLI wiring."""

from __future__ import annotations

import argparse

import pytest

import OCDocker.CLI as cli
from OCDocker.CLI import ocscore as ocscore_cli


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
