#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the Workbench job execution and tracking layer.
"""

# Imports
###############################################################################
from __future__ import annotations

import stat
import time

import pytest

from OCDocker.Workbench.Jobs import JobError
from OCDocker.Workbench.Jobs import JobManager
from OCDocker.Workbench.Jobs import WORKBENCH_JOBS_DIRNAME

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

_POLL_INTERVAL_SECONDS = 0.05
_POLL_TIMEOUT_SECONDS = 5.0

# Functions
###############################################################################
## Private ##


def _wait_for_status(manager: JobManager, job_id: str, status: str) -> None:
    '''Poll a tracked job until it reaches a terminal status or the timeout elapses.

    Parameters
    ----------
    manager : JobManager
        Job manager tracking the job.
    job_id : str
        Job identifier to poll.
    status : str
        Expected terminal status.
    '''

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record.status == status:
            return
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Job {job_id} did not reach status {status!r} in time (last: {record.status!r}).")


def _write_ignoring_sleeper(path) -> None:
    '''Write a script that sleeps regardless of the arguments it is given.

    Parameters
    ----------
    path : pathlib.Path
        Script path to write.
    '''

    path.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


## Public ##


def test_job_manager_tracks_a_successful_job(tmp_path) -> None:
    '''A launched job that exits zero is reported as completed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    record = manager.launch("vs", ["--foo", "bar"])

    assert record.status == "running"
    assert record.pid is not None
    assert record.command == ("true", "vs", "--foo", "bar")

    _wait_for_status(manager, record.job_id, "completed")
    refreshed = manager.get(record.job_id)
    assert refreshed.return_code == 0
    assert refreshed.finished_at is not None


def test_job_manager_tracks_a_failed_job(tmp_path) -> None:
    '''A launched job that exits non-zero is reported as failed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="false")
    record = manager.launch("ocscore_train", [])

    _wait_for_status(manager, record.job_id, "failed")
    refreshed = manager.get(record.job_id)
    assert refreshed.return_code == 1


def test_job_manager_rejects_unsupported_kind(tmp_path) -> None:
    '''Launching an unsupported job kind raises a structured JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    with pytest.raises(JobError, match="Unsupported job kind"):
        manager.launch("bogus", [])


def test_job_manager_get_rejects_unknown_job_id(tmp_path) -> None:
    '''Looking up an unknown job id raises a 404-flavored JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    with pytest.raises(JobError, match="Unknown job id") as excinfo:
        manager.get("does-not-exist")
    assert excinfo.value.status_code == 404


def test_job_manager_cancel_terminates_a_running_job(tmp_path) -> None:
    '''Cancelling a running job terminates it and marks it cancelled.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    sleeper = tmp_path / "sleeper.sh"
    _write_ignoring_sleeper(sleeper)

    manager = JobManager(tmp_path, executable=str(sleeper))
    record = manager.launch("pipeline", ["ignored"])
    assert manager.get(record.job_id).status == "running"

    cancelled = manager.cancel(record.job_id)
    assert cancelled.status == "cancelled"
    assert manager.get(record.job_id).status == "cancelled"


def test_job_manager_cancel_is_a_no_op_for_finished_jobs(tmp_path) -> None:
    '''Cancelling an already-finished job returns it unchanged.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    record = manager.launch("vs", [])
    _wait_for_status(manager, record.job_id, "completed")

    cancelled = manager.cancel(record.job_id)
    assert cancelled.status == "completed"


def test_job_manager_reconciles_state_across_restarts(tmp_path) -> None:
    '''A fresh JobManager instance recovers job status from disk.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    first_manager = JobManager(tmp_path, executable="true")
    completed = first_manager.launch("vs", [])
    _wait_for_status(first_manager, completed.job_id, "completed")

    sleeper = tmp_path / "sleeper.sh"
    _write_ignoring_sleeper(sleeper)
    sleeper_manager = JobManager(tmp_path, executable=str(sleeper))
    still_running = sleeper_manager.launch("pipeline", [])
    assert sleeper_manager.get(still_running.job_id).status == "running"

    restarted_manager = JobManager(tmp_path, executable="true")
    listed = {record.job_id: record.status for record in restarted_manager.list()}
    assert listed[completed.job_id] == "completed"
    assert listed[still_running.job_id] == "running"

    restarted_manager.cancel(still_running.job_id)


def test_job_manager_persists_jobs_under_a_hidden_directory(tmp_path) -> None:
    '''Job artifacts are written under a hidden directory below the served root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    record = manager.launch("vs", [])
    _wait_for_status(manager, record.job_id, "completed")

    jobs_dir = tmp_path / WORKBENCH_JOBS_DIRNAME
    assert jobs_dir.is_dir()
    assert (jobs_dir / record.job_id / "job.json").is_file()


def test_job_manager_logs_returns_stdout_and_stderr_previews(tmp_path) -> None:
    '''Job log previews expose bounded stdout and stderr content.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    record = manager.launch("vs", [])
    _wait_for_status(manager, record.job_id, "completed")

    stdout_preview, stderr_preview = manager.logs(record.job_id)
    assert stdout_preview.exists
    assert stderr_preview.exists
