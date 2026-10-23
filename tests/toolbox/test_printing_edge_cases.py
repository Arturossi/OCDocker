#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Printing helpers.
'''

# Imports
###############################################################################
import logging

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(84)
def test_print_to_log_files(tmp_path):
    log = tmp_path / "out.log"
    ocprint.print_info_log("alpha", str(log))
    ocprint.print_warning_log("beta", str(log))
    ocprint.print_error_log("gamma", str(log))
    txt = log.read_text()
    assert "INFO: alpha" in txt
    assert "WARNING: beta" in txt
    assert "ERROR: gamma" in txt


@pytest.mark.order(84)
def test_print_to_log_creates_missing_log_dir(tmp_path):
    log = tmp_path / "logs" / "out.log"
    ocprint.print_error_log("gamma", str(log))
    assert "ERROR: gamma" in log.read_text()


@pytest.mark.order(84)
def test_print_to_log_unwritable_path_does_not_raise(tmp_path, caplog):
    # A regular file as the parent directory makes the log unwritable even for root
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    log = blocker / "out.log"
    ocerror.Error.set_output_level(ocerror.ReportLevel.WARNING)
    with caplog.at_level(logging.WARNING, logger="ocdocker"):
        ocprint.print_info_log("alpha", str(log))
        ocprint.print_warning_log("beta", str(log))
        ocprint.print_error_log("gamma", str(log))
        ocprint.print_success_log("delta", str(log))
    assert not log.exists()
    for message in ("alpha", "beta", "gamma", "delta"):
        assert message in caplog.text


@pytest.mark.order(84)
def test_print_section_and_subsection_write_to_log_name(tmp_path):
    log = tmp_path / "progress.out"
    ocprint.print_section(1, "Runtime Arguments", logName = str(log))
    ocprint.print_subsection(1, "Docking", logName = str(log))
    txt = log.read_text()
    assert "Starting new OCDocker run" in txt
    assert "Docking..." in txt


@pytest.mark.order(84)
def test_print_section_unwritable_log_does_not_raise(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    ocprint.print_section(1, "Runtime Arguments", logName = str(blocker / "progress.out"))
    ocprint.print_subsection(1, "Docking", logName = str(blocker / "progress.out"))


@pytest.mark.order(83)
def test_printv_gated_by_level(capsys, caplog):
    # Ensure DEBUG prints
    ocerror.Error.set_output_level(ocerror.ReportLevel.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="ocdocker"):
        ocprint.printv("visible")
    captured = capsys.readouterr()
    out = f"{captured.out}\n{captured.err}\n{caplog.text}"
    assert "visible" in out

    # Ensure non-DEBUG suppresses
    ocerror.Error.set_output_level(ocerror.ReportLevel.INFO)
    ocprint.printv("hidden")
    captured2 = capsys.readouterr()
    out2 = f"{captured2.out}\n{captured2.err}\n{caplog.text}"
    assert "hidden" not in out2


@pytest.mark.order(84)
def test_print_to_log_rotates_by_size(monkeypatch, tmp_path):
    import OCDocker.Toolbox.Logging as oclogging

    monkeypatch.setattr(oclogging, "_log_limits", lambda: (500, 2, 10), raising=False)
    log = tmp_path / "report.log"
    for i in range(100):
        ocprint.print_error_log(f"failure {i:03d}", str(log))

    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == ["report.log", "report.log.1", "report.log.2"]
    # A file only rolls over once it reaches the limit, so it holds at most one line more
    longest_line = max(len(line) + 1 for path in tmp_path.iterdir() for line in path.read_text(encoding="utf-8").splitlines())
    assert all(path.stat().st_size < 500 + longest_line for path in tmp_path.iterdir())
    assert "failure 099" in log.read_text(encoding="utf-8")
