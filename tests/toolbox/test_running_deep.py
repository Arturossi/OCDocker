#!/usr/bin/env python3

# Description
###############################################################################
'''
Deep branch coverage tests for Toolbox.Running helpers.
'''

# Imports
###############################################################################
import os
import subprocess
import sys
import types

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Running as ocrun

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

@pytest.mark.order(99)
def test_env_helpers_and_formatting(monkeypatch):
    monkeypatch.setenv("OCR_FLAG", "yes")
    assert ocrun._env_flag("OCR_FLAG") is True

    monkeypatch.setenv("OCR_FLAG", "0")
    assert ocrun._env_flag("OCR_FLAG") is False

    monkeypatch.setenv("OCR_INT", "12")
    assert ocrun._env_int("OCR_INT", 3) == 12

    monkeypatch.setenv("OCR_INT", "-5")
    assert ocrun._env_int("OCR_INT", 3) == 3

    monkeypatch.setenv("OCR_INT", "abc")
    assert ocrun._env_int("OCR_INT", 3) == 3

    monkeypatch.setenv("OCR_A", "A")
    monkeypatch.delenv("OCR_B", raising=False)
    assert ocrun._env_snapshot(["OCR_A", "OCR_B"]) == {"OCR_A": "A"}

    assert ocrun._format_cmd(["cmd", "a b"]) == "cmd 'a b'"


@pytest.mark.order(100)
def test_tail_text_and_tail_file(tmp_path, monkeypatch):
    assert ocrun._tail_text("", 2) == ""
    assert ocrun._tail_text("a\nb\nc\n", 2) == "b\nc"
    assert ocrun._tail_text("a\nb\n", 0) == "a\nb"

    sample = tmp_path / "sample.log"
    sample.write_text("l1\nl2\nl3\n", encoding="utf-8")
    assert ocrun._tail_file(str(sample), 2) == "l2\nl3"
    assert ocrun._tail_file(str(tmp_path / "missing.log"), 3) == ""
    assert ocrun._tail_file(os.devnull, 3) == ""

    real_open = open

    def broken_open(path, mode="r", *args, **kwargs):
        if path == str(sample) and "rb" in mode:
            raise OSError("read failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", broken_open)
    assert ocrun._tail_file(str(sample), 2) == ""


@pytest.mark.order(101)
def test_failure_report_path_and_write(monkeypatch, tmp_path):
    fake_cfg = types.ModuleType("OCDocker.Config")

    class _Cfg:
        logdir = str(tmp_path / "cfg_logs")

    fake_cfg.get_config = lambda: _Cfg()
    monkeypatch.setitem(sys.modules, "OCDocker.Config", fake_cfg)

    report_path = ocrun._failure_report_path()
    assert report_path.endswith("subprocess_failures.log")
    assert str(tmp_path / "cfg_logs") in report_path

    written = ocrun._write_failure_report("hello")
    assert written == report_path
    assert "hello" in (tmp_path / "cfg_logs" / "subprocess_failures.log").read_text(encoding="utf-8")

    monkeypatch.setattr(ocrun, "_failure_report_path", lambda: "")
    assert ocrun._write_failure_report("ignored") == ""

    monkeypatch.setattr(ocrun, "_failure_report_path", lambda: str(tmp_path / "x.log"))

    def boom_open(*_args, **_kwargs):
        raise OSError("write failure")

    monkeypatch.setattr("builtins.open", boom_open)
    assert ocrun._write_failure_report("x") == ""


@pytest.mark.order(102)
def test_failure_report_path_fallback_and_makedirs_error(monkeypatch, tmp_path):
    broken_cfg = types.ModuleType("OCDocker.Config")

    def _raise_get_config():
        raise RuntimeError("boom")

    broken_cfg.get_config = _raise_get_config
    monkeypatch.setitem(sys.modules, "OCDocker.Config", broken_cfg)

    path = ocrun._failure_report_path()
    assert path.endswith(os.path.join("logs", "subprocess_failures.log"))

    cfg_mod = types.ModuleType("OCDocker.Config")

    class _Cfg:
        logdir = str(tmp_path / "denied")

    cfg_mod.get_config = lambda: _Cfg()
    monkeypatch.setitem(sys.modules, "OCDocker.Config", cfg_mod)
    monkeypatch.setattr(ocrun.os, "makedirs", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")))
    assert ocrun._failure_report_path() == ""


@pytest.mark.order(103)
def test_is_tool_available_with_absolute_and_which(tmp_path, monkeypatch):
    assert ocrun.is_tool_available("") is False

    script = tmp_path / "tool.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o644)
    assert ocrun.is_tool_available(str(script)) is False

    script.chmod(0o755)
    assert ocrun.is_tool_available(str(script)) is True

    monkeypatch.setattr(ocrun.shutil, "which", lambda _exe: "/usr/bin/fake")
    assert ocrun.is_tool_available("fakecmd") is True


@pytest.mark.order(104)
@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("no such file"),
        subprocess.TimeoutExpired(cmd=["x"], timeout=1),
        RuntimeError("generic failure"),
    ],
)
def test_run_subprocess_exception_branches(monkeypatch, tmp_path, exc):
    monkeypatch.setattr(ocrun.shutil, "which", lambda _exe: "/usr/bin/fake")

    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(ocrun.subprocess, "run", _raise)

    code = ocrun.run(["fakecmd"], logFile=str(tmp_path / "run.log"))
    assert isinstance(code, int)
    assert code == ocerror.ErrorCode.SUBPROCESS


@pytest.mark.order(105)
def test_run_success_with_cwd_and_invalid_timeout_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OCDOCKER_TIMEOUT", "not-an-int")
    log_file = tmp_path / "success.log"
    code = ocrun.run(
        [sys.executable, "-c", "print('ok')"],
        logFile=str(log_file),
        cwd=str(tmp_path),
    )
    assert code == ocerror.ErrorCode.OK
    assert "ok" in log_file.read_text(encoding="utf-8")


@pytest.mark.order(106)
def test_run_empty_and_invalid_executable_paths(tmp_path):
    empty_exe_code = ocrun.run([""], logFile=str(tmp_path / "empty.log"))
    assert empty_exe_code == ocerror.ErrorCode.SUBPROCESS

    bad_path = tmp_path / "not_exec"
    bad_path.write_text("x", encoding="utf-8")
    bad_exe_code = ocrun.run([str(bad_path)], logFile=str(tmp_path / "bad.log"))
    assert bad_exe_code == ocerror.ErrorCode.SUBPROCESS


@pytest.mark.order(107)
def test_run_nonzero_exit_returns_tuple_and_report(monkeypatch, tmp_path):
    monkeypatch.setenv("OCDOCKER_SUBPROCESS_TAIL_LINES", "2")
    monkeypatch.setenv("OCDOCKER_DEBUG_SUBPROCESS", "1")
    monkeypatch.delenv("OCDOCKER_RAISE_SUBPROCESS", raising=False)
    monkeypatch.setattr(ocrun.shutil, "which", lambda _exe: "/usr/bin/fake")

    class _Proc:
        returncode = 5
        stderr = b"e1\ne2\ne3\n"

    def _fake_run(_cmd, stdout=None, stderr=None, timeout=None, cwd=None):
        assert stderr == subprocess.PIPE
        assert timeout is None
        if stdout is not None:
            stdout.write("o1\no2\no3\n")
        return _Proc()

    captured = {}

    def _capture_report(report: str) -> str:
        captured["report"] = report
        return str(tmp_path / "failure-report.log")

    monkeypatch.setattr(ocrun.subprocess, "run", _fake_run)
    monkeypatch.setattr(ocrun, "_write_failure_report", _capture_report)

    result = ocrun.run(["fakecmd", "--x"], logFile=str(tmp_path / "cmd.log"))
    assert isinstance(result, tuple)
    code, stderr_text = result
    assert code == ocerror.ErrorCode.SUBPROCESS
    assert "e3" in stderr_text

    report = captured["report"]
    assert "Stderr:" in report
    assert "o2\no3" in report
    assert "Environment snapshot:" in report


@pytest.mark.order(108)
def test_run_nonzero_exit_raises_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OCDOCKER_RAISE_SUBPROCESS", "1")
    monkeypatch.delenv("OCDOCKER_DEBUG_SUBPROCESS", raising=False)
    monkeypatch.setattr(ocrun.shutil, "which", lambda _exe: "/usr/bin/fake")

    class _Proc:
        returncode = 8
        stderr = b"fatal\n"

    monkeypatch.setattr(ocrun.subprocess, "run", lambda *_args, **_kwargs: _Proc())
    monkeypatch.setattr(ocrun, "_write_failure_report", lambda _report: str(tmp_path / "subproc.log"))

    with pytest.raises(ocrun.SubprocessError) as exc_info:
        ocrun.run(["fakecmd"], logFile=str(tmp_path / "cmd.log"))

    err = exc_info.value
    assert err.returncode == 8
    assert err.cmd == ["fakecmd"]
    assert err.stdout_log == str(tmp_path / "cmd.log")
    assert err.report_path.endswith("subproc.log")
