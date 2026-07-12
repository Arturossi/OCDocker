#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for CLI doctor diagnostics branches.
'''

# Imports
###############################################################################
import importlib.metadata
import json
import os
import shutil
import sys
import types

from types import SimpleNamespace

import pytest

import OCDocker.CLI.doctor as cli_doctor
import OCDocker.Error as ocerror

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


class _Conn:
    def __init__(self, tracker):
        self._tracker = tracker

    class _ScalarResult:
        def __init__(self, value):
            self._value = value

        def scalar(self):
            return self._value

    def exec_driver_sql(self, sql):
        self._tracker["sql"] = self._tracker.get("sql", []) + [sql]
        if "SHOW server_version" in sql:
            return _Conn._ScalarResult("16.4")
        if "SELECT VERSION()" in sql:
            return _Conn._ScalarResult("8.4.0")
        if "SELECT sqlite_version()" in sql:
            return _Conn._ScalarResult("3.45.1")
        if "SELECT current_user" in sql:
            return _Conn._ScalarResult("ocdocker")
        if "SELECT current_database()" in sql:
            return _Conn._ScalarResult("optimization")
        if "SELECT CURRENT_USER()" in sql:
            return _Conn._ScalarResult("ocdocker@localhost")
        if "SELECT DATABASE()" in sql:
            return _Conn._ScalarResult("optimization")
        return _Conn._ScalarResult(None)

    def close(self):
        self._tracker["closed"] = self._tracker["closed"] + 1


class _EngineOK:
    def __init__(self, tracker, drivername="postgresql+psycopg"):
        self._tracker = tracker
        self.url = SimpleNamespace(drivername=drivername)

    def connect(self):
        self._tracker["connected"] = self._tracker["connected"] + 1
        return _Conn(self._tracker)


# Functions
###############################################################################
## Private ##

def _install_cli_doctor_modules(monkeypatch, initialise_mod, config_mod, logging_raise=False):
    error_mod = types.ModuleType("OCDocker.Error")
    error_mod.Error = SimpleNamespace(get_output_level=lambda: ocerror.ReportLevel.INFO)

    logging_mod = types.ModuleType("OCDocker.Toolbox.Logging")
    if logging_raise:
        logging_mod.configure = lambda *a, **k: (_ for _ in ()).throw(OSError("logging failure"))  # type: ignore[attr-defined]
    else:
        logging_mod.configure = lambda *a, **k: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Error", error_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Logging", logging_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", initialise_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Config", config_mod)
    import OCDocker
    import OCDocker.Toolbox as octoolbox

    monkeypatch.setattr(OCDocker, "Error", error_mod, raising=False)
    monkeypatch.setattr(OCDocker, "Initialise", initialise_mod, raising=False)
    monkeypatch.setattr(OCDocker, "Config", config_mod, raising=False)
    monkeypatch.setattr(octoolbox, "Logging", logging_mod, raising=False)


## Public ##

@pytest.mark.order(175)
def test_cmd_doctor_reports_config_unavailable_and_missing_engine(monkeypatch, capsys):
    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.config_file = "/tmp/fake.cfg"
    init_mod.engine = None

    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: (_ for _ in ()).throw(RuntimeError("missing config"))  # type: ignore[attr-defined]

    _install_cli_doctor_modules(monkeypatch, init_mod, config_mod, logging_raise=False)
    monkeypatch.setattr("OCDocker.CLI.doctor._preparse_global_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr("OCDocker.CLI.doctor._bootstrap_ocdocker_env", lambda _ns: None)
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    monkeypatch.setattr("OCDocker.CLI.manifest._probe_executable_version", lambda _exe: "unknown")

    args = SimpleNamespace(log_file="", no_stdout_log=True)
    rc = cli_doctor.cmd_doctor(args)
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    assert report["config"]["path"] == "/tmp/fake.cfg"
    assert report["binaries"]["vina"] == "MISSING"
    assert report["binaries"]["smina"] == "MISSING"
    assert report["binaries"]["plants"] == "MISSING"
    assert report["external_tools"]["vina"]["version"] == "unknown"
    assert report["external_tools"]["gnina"]["available"] is False
    assert report["binaries_error"].startswith("CONFIG_UNAVAILABLE")
    assert report["database"]["status"] == "MISSING ENGINE"
    assert report["database"]["access"] is False
    assert report["database"]["backend"] == "unknown"
    assert "sqlalchemy_version" in report["database"]


@pytest.mark.order(176)
def test_cmd_doctor_reports_binary_and_database_ok(monkeypatch, tmp_path, capsys):
    tracker = {"connected": 0, "closed": 0}
    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.config_file = "/tmp/ok.cfg"
    init_mod.engine = _EngineOK(tracker)

    vina_exe = tmp_path / "vina_exec"
    vina_exe.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(vina_exe, 0o755)

    cfg_obj = SimpleNamespace(
        vina=SimpleNamespace(executable=str(vina_exe)),
        smina=SimpleNamespace(executable="smina_cmd"),
        plants=SimpleNamespace(executable=str(tmp_path / "missing_plants")),
        database=SimpleNamespace(backend="postgresql", user="ocdocker", database="optimization"),
    )
    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: cfg_obj  # type: ignore[attr-defined]

    _install_cli_doctor_modules(monkeypatch, init_mod, config_mod, logging_raise=False)
    monkeypatch.setattr("OCDocker.CLI.doctor._preparse_global_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr("OCDocker.CLI.doctor._bootstrap_ocdocker_env", lambda _ns: None)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/smina_cmd" if cmd == "smina_cmd" else None)
    monkeypatch.setattr(
        "OCDocker.CLI.manifest._probe_executable_version",
        lambda exe: f"v@{os.path.basename(exe)}" if exe else "unknown",
    )

    args = SimpleNamespace(log_file="", no_stdout_log=False)
    rc = cli_doctor.cmd_doctor(args)
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    assert report["binaries"]["vina"] == "OK"
    assert report["binaries"]["smina"] == "OK"
    assert report["binaries"]["plants"] == "MISSING"
    assert report["external_tools"]["vina"]["version"] == f"v@{os.path.basename(str(vina_exe))}"
    assert report["external_tools"]["smina"]["version"] == "v@smina_cmd"
    assert report["external_tools"]["plants"]["version"] == "unknown"
    assert report["database"]["status"] == "OK"
    assert report["database"]["access"] is True
    assert report["database"]["backend"] == "postgresql"
    assert report["database"]["driver"] == "postgresql+psycopg"
    assert report["database"]["server_version"] == "16.4"
    assert report["database"]["current_user"] == "ocdocker"
    assert report["database"]["current_database"] == "optimization"
    assert report["database"]["expected_user"] == "ocdocker"
    assert report["database"]["expected_database"] == "optimization"
    assert report["database"]["user_check"] == "ok"
    assert report["database"]["database_check"] == "ok"
    assert "client_version" in report["database"]
    assert "sqlalchemy_version" in report["database"]
    assert tracker["sql"] == [
        "SHOW server_version",
        "SELECT current_user",
        "SELECT current_database()",
    ]
    assert tracker["connected"] == 1
    assert tracker["closed"] == 1


@pytest.mark.order(177)
def test_cmd_doctor_ignores_logging_configuration_errors(monkeypatch, capsys):
    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.config_file = "/tmp/ok.cfg"
    init_mod.engine = None

    cfg_obj = SimpleNamespace(
        vina=SimpleNamespace(executable=""),
        smina=SimpleNamespace(executable=""),
        plants=SimpleNamespace(executable=""),
    )
    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: cfg_obj  # type: ignore[attr-defined]

    _install_cli_doctor_modules(monkeypatch, init_mod, config_mod, logging_raise=True)
    monkeypatch.setattr("OCDocker.CLI.doctor._preparse_global_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr("OCDocker.CLI.doctor._bootstrap_ocdocker_env", lambda _ns: None)
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    rc = cli_doctor.cmd_doctor(SimpleNamespace(log_file="/tmp/x.log", no_stdout_log=False))
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert "binaries" in report
    assert report["database"]["status"] == "MISSING ENGINE"


class _FakeOddtMetadata(dict):
    '''Minimal stand-in for the ``email.message.Message`` importlib.metadata returns.'''

    def get(self, key, default=None):  # type: ignore[override]
        return dict.get(self, key, default)


@pytest.mark.order(178)
def test_oddt_dependency_status_reports_missing_when_not_importable(monkeypatch):
    # A None entry in sys.modules makes `import oddt` raise ImportError deterministically.
    monkeypatch.setitem(sys.modules, "oddt", None)

    status = cli_doctor._oddt_dependency_status()
    assert status.startswith("MISSING")


@pytest.mark.order(179)
def test_oddt_dependency_status_reports_ok_when_vendored(monkeypatch):
    monkeypatch.setitem(sys.modules, "oddt", types.ModuleType("oddt"))

    def _raise_not_found(_name):
        raise importlib.metadata.PackageNotFoundError("oddt")

    monkeypatch.setattr(cli_doctor.importlib.metadata, "metadata", _raise_not_found)

    assert cli_doctor._oddt_dependency_status() == "OK (vendored)"


@pytest.mark.order(180)
def test_oddt_dependency_status_flags_vanilla_upstream_home_page(monkeypatch):
    monkeypatch.setitem(sys.modules, "oddt", types.ModuleType("oddt"))
    monkeypatch.setattr(
        cli_doctor.importlib.metadata,
        "metadata",
        lambda _name: _FakeOddtMetadata({"Home-page": "https://github.com/oddt/oddt", "Author-email": ""}),
    )

    status = cli_doctor._oddt_dependency_status()
    assert status.startswith("WRONG")
    assert "vendor_oddt.sh" in status


@pytest.mark.order(181)
def test_oddt_dependency_status_flags_vanilla_upstream_author_email(monkeypatch):
    # Home-page absent/different, but the upstream maintainer's email is still a giveaway.
    monkeypatch.setitem(sys.modules, "oddt", types.ModuleType("oddt"))
    monkeypatch.setattr(
        cli_doctor.importlib.metadata,
        "metadata",
        lambda _name: _FakeOddtMetadata({"Home-page": "", "Author-email": "mwojcikowski@ibb.waw.pl"}),
    )

    assert cli_doctor._oddt_dependency_status().startswith("WRONG")


@pytest.mark.order(182)
def test_oddt_dependency_status_accepts_non_upstream_standalone_install(monkeypatch):
    # e.g., someone pip-installed the fork directly instead of running the vendor script.
    monkeypatch.setitem(sys.modules, "oddt", types.ModuleType("oddt"))
    monkeypatch.setattr(
        cli_doctor.importlib.metadata,
        "metadata",
        lambda _name: _FakeOddtMetadata({"Home-page": "https://github.com/Arturossi/oddt", "Author-email": ""}),
    )

    status = cli_doctor._oddt_dependency_status()
    assert status == "OK (standalone: https://github.com/Arturossi/oddt)"


@pytest.mark.order(183)
def test_oddt_dependency_status_tolerates_unexpected_metadata_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "oddt", types.ModuleType("oddt"))

    def _raise_unexpected(_name):
        raise RuntimeError("corrupted dist-info")

    monkeypatch.setattr(cli_doctor.importlib.metadata, "metadata", _raise_unexpected)

    assert cli_doctor._oddt_dependency_status() == "OK (standalone: unknown origin)"
