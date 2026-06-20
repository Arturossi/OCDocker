#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for CLI reproducibility manifest functionality.
'''

# Imports
###############################################################################
import argparse
import json

from types import SimpleNamespace

import pytest

import OCDocker.CLI as cli
import OCDocker.CLI.manifest as cli_manifest

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(464)
def test_build_parser_manifest_subcommand_parse():
    parser = cli.build_parser()
    ns = parser.parse_args(["manifest", "--output", "manifest.json", "--no-packages"])
    assert callable(getattr(ns, "func", None))
    assert ns.output == "manifest.json"
    assert ns.no_packages is True


@pytest.mark.order(465)
def test_generate_reproducibility_manifest_structure(monkeypatch):
    monkeypatch.setattr(cli_manifest, "_collect_ocdocker_version", lambda: "0.0.test")
    monkeypatch.setattr(
        cli_manifest,
        "_collect_external_tool_manifest",
        lambda: {"vina": {"configured": "vina", "resolved": None, "available": False, "version": "unknown"}},
    )
    monkeypatch.setattr(cli_manifest, "_collect_git_manifest", lambda: {"commit": "abc", "branch": "main", "dirty": False})

    manifest = cli_manifest.generate_reproducibility_manifest(include_python_packages=False)

    assert manifest["schema_version"] == 1
    assert manifest["ocdocker"]["version"] == "0.0.test"
    assert "generated_at_utc" in manifest
    assert "python" in manifest
    assert "platform" in manifest
    assert "external_tools" in manifest
    assert "python_packages" not in manifest


@pytest.mark.order(466)
def test_cmd_manifest_writes_output_and_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_manifest, "_preparse_global_args", lambda _argv: argparse.Namespace(config_file=None))
    monkeypatch.setattr(cli_manifest, "_bootstrap_ocdocker_env", lambda _ns: None)
    monkeypatch.setattr(
        cli_manifest,
        "generate_reproducibility_manifest",
        lambda include_python_packages=True: {
            "schema_version": 1,
            "ocdocker": {"version": "0.0.test"},
            "python_package_count": 1 if include_python_packages else 0,
        },
    )

    output = tmp_path / "repro_manifest.json"
    rc = cli_manifest.cmd_manifest(SimpleNamespace(output=str(output), no_packages=False))
    assert rc == 0

    payload_stdout = json.loads(capsys.readouterr().out)
    payload_file = json.loads(output.read_text(encoding="utf-8"))

    assert payload_stdout == payload_file
    assert payload_stdout["ocdocker"]["version"] == "0.0.test"
    assert payload_stdout["bootstrap"]["status"] == "ok"


@pytest.mark.order(467)
def test_cmd_manifest_bootstrap_failure_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(cli_manifest, "_preparse_global_args", lambda _argv: argparse.Namespace(config_file=None))
    monkeypatch.setattr(
        cli_manifest,
        "_bootstrap_ocdocker_env",
        lambda _ns: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    monkeypatch.setattr(cli_manifest, "generate_reproducibility_manifest", lambda include_python_packages=True: {"schema_version": 1})

    rc = cli_manifest.cmd_manifest(SimpleNamespace(output=None, no_packages=True))
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["bootstrap"]["status"] == "error"
    assert "bootstrap failed" in payload["bootstrap"]["error"]
