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
import sys
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


def _write_ligand_gated_fake(path) -> None:
    '''Write a fake ocdocker executable that fails only for a "bad" ligand.

    Parameters
    ----------
    path : pathlib.Path
        Script path to write.
    '''

    path.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *bad_ligand*) echo 'simulated docking failure' >&2; exit 1 ;;\n"
        "  *) echo \"ran: $*\"; exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
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


def test_vs_campaign_runs_every_row_and_continues_past_failure(tmp_path) -> None:
    '''A campaign with one failing row still runs every row and reports an aggregate.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    fake = tmp_path / "fake_ocdocker.sh"
    _write_ligand_gated_fake(fake)
    manager = JobManager(tmp_path, executable=str(fake))
    manifest = [
        {"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l1.smi", "box": "b.pdb", "engines": ["vina"]},
        {"sample": "s2", "row_kind": "vs", "receptor": "r.pdb", "ligand": "bad_ligand.smi", "box": "b.pdb", "engines": ["vina"]},
        {"sample": "s3", "row_kind": "pipeline", "receptor": "r.pdb", "ligand": "l3.smi", "box": "b.pdb", "engines": ["vina", "smina"]},
    ]

    record = manager.launch("vs_campaign", [], manifest=manifest)
    _wait_for_status(manager, record.job_id, "failed")

    stdout_preview, stderr_preview = manager.logs(record.job_id)
    assert "[sample 1/3] s1" in stdout_preview.text
    assert "[sample 2/3] s2" in stdout_preview.text
    assert "[sample 3/3] s3" in stdout_preview.text
    assert "2/3 succeeded, 1 failed" in stdout_preview.text
    assert "simulated docking failure" in stderr_preview.text


def test_vs_campaign_all_rows_succeed_reports_completed(tmp_path) -> None:
    '''A campaign with no failing rows is reported as completed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    manifest = [
        {"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l1.smi", "box": "b.pdb", "engines": ["vina"]},
        {"sample": "s2", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l2.smi", "box": "b.pdb", "engines": ["vina"]},
    ]

    record = manager.launch("vs_campaign", [], manifest=manifest)
    _wait_for_status(manager, record.job_id, "completed")

    stdout_preview, _stderr_preview = manager.logs(record.job_id)
    assert "2/2 succeeded, 0 failed" in stdout_preview.text


def test_vs_campaign_rejects_empty_manifest(tmp_path) -> None:
    '''Launching a vs_campaign job without a manifest raises a structured JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    with pytest.raises(JobError, match="non-empty manifest"):
        manager.launch("vs_campaign", [])


def test_vs_campaign_rejects_row_missing_required_field(tmp_path) -> None:
    '''A manifest row missing a required field raises a structured JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    with pytest.raises(JobError, match="missing 'box'"):
        manager.plan("vs_campaign", [], manifest=[{"sample": "s1", "receptor": "r.pdb", "ligand": "l.smi", "engines": ["vina"]}])


def test_vs_campaign_rejects_unknown_row_kind(tmp_path) -> None:
    '''A manifest row with an unsupported row_kind raises a structured JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    bad_row = {"sample": "s1", "row_kind": "ocscore_train", "receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}
    with pytest.raises(JobError, match="unsupported row_kind"):
        manager.plan("vs_campaign", [], manifest=[bad_row])


def test_vs_campaign_common_args_applied_to_every_row(tmp_path) -> None:
    '''Common args passed to launch() are appended to every row command.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    manifest = [{"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}]

    plan = manager.plan("vs_campaign", ["--store-db"], manifest=manifest)

    assert "--store-db" in plan["command"][2]


def test_vs_campaign_snakemake_engine_plan_shape(tmp_path) -> None:
    '''plan() with engine="snakemake" returns a real snakemake argv, not a shell script.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    manager = JobManager(tmp_path, executable="true")
    manifest = [{"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}]

    plan = manager.plan("vs_campaign", [], manifest=manifest, engine="snakemake", cores=3)

    assert plan["command"][:3] == [sys.executable, "-m", "snakemake"]
    assert "-s" in plan["command"]
    assert "--cores" in plan["command"] and plan["command"][plan["command"].index("--cores") + 1] == "3"
    assert any(part.startswith("samples=") for part in plan["command"])


def test_vs_campaign_snakemake_engine_rejects_empty_manifest(tmp_path) -> None:
    '''engine="snakemake" also refuses to build a command from an empty manifest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    manager = JobManager(tmp_path, executable="true")
    with pytest.raises(JobError, match="non-empty manifest"):
        manager.plan("vs_campaign", [], manifest=[], engine="snakemake")


def test_vs_campaign_rejects_unsupported_engine(tmp_path) -> None:
    '''An unrecognized engine name raises a structured JobError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    manifest = [{"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}]
    with pytest.raises(JobError, match="Unsupported vs_campaign engine"):
        manager.plan("vs_campaign", [], manifest=manifest, engine="bogus")


def test_vs_campaign_snakemake_engine_runs_and_continues_past_failure(tmp_path) -> None:
    '''engine="snakemake" runs every sample and continues past one failing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    fake = tmp_path / "fake_ocdocker.sh"
    _write_ligand_gated_fake(fake)
    manager = JobManager(tmp_path, executable=str(fake))
    for name in ("r.pdb", "l1.smi", "bad_ligand.smi", "b.pdb"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    manifest = [
        {"sample": "s1", "row_kind": "vs", "receptor": str(tmp_path / "r.pdb"), "ligand": str(tmp_path / "l1.smi"), "box": str(tmp_path / "b.pdb"), "engines": ["vina"]},
        {"sample": "s2", "row_kind": "vs", "receptor": str(tmp_path / "r.pdb"), "ligand": str(tmp_path / "bad_ligand.smi"), "box": str(tmp_path / "b.pdb"), "engines": ["vina"]},
    ]

    record = manager.launch("vs_campaign", [], manifest=manifest, engine="snakemake", cores=2)
    _wait_for_status(manager, record.job_id, "failed")

    assert (tmp_path / "results" / "s1" / ".campaign_done").is_file()
    assert not (tmp_path / "results" / "s2" / ".campaign_done").is_file()


def test_vs_campaign_snakemake_engine_all_succeed_reports_completed(tmp_path) -> None:
    '''engine="snakemake" reports completed when every sample succeeds.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    for name in ("r.pdb", "l1.smi", "l2.smi", "b.pdb"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    manager = JobManager(tmp_path, executable="true")
    manifest = [
        {"sample": "s1", "row_kind": "vs", "receptor": str(tmp_path / "r.pdb"), "ligand": str(tmp_path / "l1.smi"), "box": str(tmp_path / "b.pdb"), "engines": ["vina"]},
        {"sample": "s2", "row_kind": "vs", "receptor": str(tmp_path / "r.pdb"), "ligand": str(tmp_path / "l2.smi"), "box": str(tmp_path / "b.pdb"), "engines": ["vina"]},
    ]

    record = manager.launch("vs_campaign", [], manifest=manifest, engine="snakemake", cores=2)
    _wait_for_status(manager, record.job_id, "completed")

    assert (tmp_path / "results" / "s1" / ".campaign_done").is_file()
    assert (tmp_path / "results" / "s2" / ".campaign_done").is_file()


def test_vs_campaign_results_dir_nests_per_sample_shell_engine(tmp_path) -> None:
    '''results_dir gets a per-sample subdirectory, not one shared --outdir.

    Regression test: ocdocker vs/pipeline write straight under --outdir with
    no nesting of their own, so a literal shared --outdir across every row
    would make every sample overwrite the same directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    manager = JobManager(tmp_path, executable="true")
    manifest = [
        {"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l1.smi", "box": "b.pdb", "engines": ["vina"]},
        {"sample": "s2", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l2.smi", "box": "b.pdb", "engines": ["vina"]},
    ]

    plan = manager.plan("vs_campaign", [], manifest=manifest, results_dir="my_runs")

    assert "--outdir my_runs/s1" in plan["command"][2]
    assert "--outdir my_runs/s2" in plan["command"][2]


def test_vs_campaign_results_dir_passed_to_snakemake_config(tmp_path) -> None:
    '''results_dir is threaded into the Snakefile's results_dir config value.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    pytest.importorskip("snakemake")

    manager = JobManager(tmp_path, executable="true")
    manifest = [{"sample": "s1", "row_kind": "vs", "receptor": "r.pdb", "ligand": "l.smi", "box": "b.pdb", "engines": ["vina"]}]

    plan = manager.plan("vs_campaign", [], manifest=manifest, engine="snakemake", results_dir="my_runs")

    assert "results_dir=my_runs" in plan["command"]
