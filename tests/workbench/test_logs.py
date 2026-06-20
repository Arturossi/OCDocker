#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench bounded log previews.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import preview_run_logs
from OCDocker.Workbench import write_model

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def test_preview_run_logs_returns_bounded_tail_and_missing_logs(tmp_path) -> None:
    '''Log previews tail existing logs and report missing logs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "run.log").write_text(
        "line-1\nline-2\nline-3\nline-4\nline-5\n",
        encoding="utf-8",
    )
    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-logs",
            spec_type="ocscore_study",
            name="logs-study",
            workspace=tmp_path,
            log_files=("logs/run.log", "logs/missing.log"),
        ),
    )

    preview = preview_run_logs(tmp_path, lines=2, max_bytes=1024)

    assert preview.run_id == "run-logs"
    assert preview.line_limit == 2
    assert preview.logs[0].exists is True
    assert preview.logs[0].lines == ("line-4", "line-5")
    assert preview.logs[0].text == "line-4\nline-5"
    assert preview.logs[0].truncated is True
    assert preview.logs[1].exists is False
    assert preview.logs[1].error == "Log file does not exist."


def test_preview_run_logs_reports_byte_truncation(tmp_path) -> None:
    '''Log previews report when max_bytes truncates the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    (tmp_path / "run.log").write_text("abcdef\nuvwxyz\n", encoding="utf-8")
    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-byte-tail",
            spec_type="ocscore_study",
            name="byte-tail",
            workspace=tmp_path,
            log_files=("run.log",),
        ),
    )

    preview = preview_run_logs(tmp_path / "run_manifest.yml", lines=5, max_bytes=4)

    assert preview.logs[0].truncated is True
    assert preview.logs[0].read_bytes == 4
    assert preview.logs[0].text == "xyz"


def test_preview_run_logs_rejects_non_positive_limits(tmp_path) -> None:
    '''Log previews require positive line and byte limits.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="lines"):
        preview_run_logs(tmp_path, lines=0)
    with pytest.raises(ValueError, match="max_bytes"):
        preview_run_logs(tmp_path, max_bytes=0)
