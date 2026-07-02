#!/usr/bin/env python3

# Description
###############################################################################
"""
Local subprocess launcher and tracker for Workbench API jobs.
"""

# Imports
###############################################################################
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import uuid

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Sequence

from OCDocker.Workbench.IO import read_data
from OCDocker.Workbench.IO import write_model
from OCDocker.Workbench.Logs import DEFAULT_LOG_BYTE_LIMIT
from OCDocker.Workbench.Logs import DEFAULT_LOG_LINE_LIMIT
from OCDocker.Workbench.Logs import build_log_file_preview
from OCDocker.Workbench.Models import RunLogFilePreview
from OCDocker.Workbench.Models import WorkbenchJobKind
from OCDocker.Workbench.Models import WorkbenchJobRecord

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

WORKBENCH_JOBS_DIRNAME = ".ocdocker-jobs"
JOB_KIND_COMMAND_PREFIX: dict[WorkbenchJobKind, tuple[str, ...]] = {
    "vs": ("vs",),
    "pipeline": ("pipeline",),
    "ocscore_train": ("ocscore", "train"),
    "ocscore_reduce": ("ocscore", "reduce"),
}
CAMPAIGN_ROW_KINDS = ("vs", "pipeline")

# Classes
###############################################################################


class JobError(Exception):
    """HTTP-aware error raised by Workbench job execution helpers."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        '''Create a Workbench job error.

        Parameters
        ----------
        message : str
            Error message.
        status_code : int
            HTTP status code returned by the Workbench API.
        '''

        super().__init__(message)
        self.status_code = status_code


class JobManager:
    """Launch and track ``ocdocker`` CLI jobs as local subprocesses."""

    def __init__(self, root: str | Path, *, executable: str = "ocdocker") -> None:
        '''Bind a job manager to one served Workbench root.

        Parameters
        ----------
        root : str or pathlib.Path
            Served Workbench root. Job artifacts are written under
            ``root / ".ocdocker-jobs"``.
        executable : str
            ``ocdocker`` executable used to launch jobs.
        '''

        self.root = Path(root).resolve()
        self.executable = executable
        self.jobs_dir = self.root / WORKBENCH_JOBS_DIRNAME
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def plan(
        self,
        kind: WorkbenchJobKind,
        args: Sequence[str],
        *,
        cwd: str | Path | None = None,
        manifest: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        '''Compute the command a job would run, without launching it.

        Parameters
        ----------
        kind : WorkbenchJobKind
            Job kind. Selects the ``ocdocker`` subcommand prefix, or (for
            ``"vs_campaign"``) the per-row subcommand used inside the
            generated batch script.
        args : Sequence[str]
            Extra command-line arguments appended after the subcommand
            prefix (or, for ``"vs_campaign"``, appended to every row).
        cwd : str, pathlib.Path, or None
            Working directory the job would run in. Defaults to the served root.
        manifest : Sequence[dict[str, Any]] or None
            Required for ``kind="vs_campaign"``: one entry per sample
            (``sample``, ``receptor``, ``ligand``, ``box``, ``engines``,
            optional ``rescoring_engines``). Ignored for other kinds.

        Returns
        -------
        dict[str, Any]
            JSON-safe preview: ``kind``, ``command``, ``cwd``.

        Raises
        ------
        JobError
            If ``kind`` is unsupported, ``cwd`` does not exist, or (for
            ``"vs_campaign"``) ``manifest`` is missing or malformed.
        '''

        command, resolved_cwd = self._resolve(kind, args, cwd=cwd, manifest=manifest)
        return {"kind": kind, "command": list(command), "cwd": str(resolved_cwd)}

    def launch(
        self,
        kind: WorkbenchJobKind,
        args: Sequence[str],
        *,
        cwd: str | Path | None = None,
        manifest: Sequence[dict[str, Any]] | None = None,
    ) -> WorkbenchJobRecord:
        '''Launch one tracked job as a background subprocess.

        Parameters
        ----------
        kind : WorkbenchJobKind
            Job kind. Selects the ``ocdocker`` subcommand prefix, or (for
            ``"vs_campaign"``) the per-row subcommand used inside the
            generated batch script.
        args : Sequence[str]
            Extra command-line arguments appended after the subcommand
            prefix (or, for ``"vs_campaign"``, appended to every row).
        cwd : str, pathlib.Path, or None
            Working directory for the launched process. Defaults to the served root.
        manifest : Sequence[dict[str, Any]] or None
            Required for ``kind="vs_campaign"``: see :meth:`plan`.

        Returns
        -------
        WorkbenchJobRecord
            Tracked job record with ``status: "running"``.

        Raises
        ------
        JobError
            If ``kind`` is unsupported, ``cwd`` does not exist, ``manifest`` is
            missing/malformed for ``"vs_campaign"``, or the subprocess could
            not be launched.
        '''

        command, resolved_cwd = self._resolve(kind, args, cwd=cwd, manifest=manifest)
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        stdout_log = job_dir / "stdout.log"
        stderr_log = job_dir / "stderr.log"
        returncode_path = job_dir / "returncode"
        manifest_path = job_dir / "job.json"

        process = self._spawn(command, cwd=resolved_cwd, stdout_log=stdout_log, stderr_log=stderr_log, returncode_path=returncode_path)
        record = WorkbenchJobRecord(
            job_id=job_id,
            kind=kind,
            command=command,
            cwd=resolved_cwd,
            status="running",
            pid=process.pid,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            returncode_path=returncode_path,
            manifest_path=manifest_path,
        )
        self._processes[job_id] = process
        write_model(manifest_path, record)
        return record

    def list(self) -> tuple[WorkbenchJobRecord, ...]:
        '''Return every tracked job, most recently created first.

        Returns
        -------
        tuple[WorkbenchJobRecord, ...]
            Tracked job records with freshly reconciled status.
        '''

        manifest_paths = sorted(self.jobs_dir.glob("*/job.json"), key=lambda path: path.parent.name, reverse=True)
        return tuple(self._refresh(self._read(path)) for path in manifest_paths)

    def get(self, job_id: str) -> WorkbenchJobRecord:
        '''Return one tracked job with freshly reconciled status.

        Parameters
        ----------
        job_id : str
            Job identifier returned by :meth:`launch`.

        Returns
        -------
        WorkbenchJobRecord
            Tracked job record.

        Raises
        ------
        JobError
            If no job with ``job_id`` is tracked.
        '''

        manifest_path = self.jobs_dir / job_id / "job.json"
        if not manifest_path.is_file():
            raise JobError(f"Unknown job id: {job_id}", status_code=404)
        return self._refresh(self._read(manifest_path))

    def logs(
        self,
        job_id: str,
        *,
        lines: int = DEFAULT_LOG_LINE_LIMIT,
        max_bytes: int = DEFAULT_LOG_BYTE_LIMIT,
    ) -> tuple[RunLogFilePreview, RunLogFilePreview]:
        '''Return bounded stdout and stderr previews for one tracked job.

        Parameters
        ----------
        job_id : str
            Job identifier returned by :meth:`launch`.
        lines : int
            Maximum returned lines per log file.
        max_bytes : int
            Maximum bytes read from the end of each log file.

        Returns
        -------
        tuple[RunLogFilePreview, RunLogFilePreview]
            Stdout preview, then stderr preview.

        Raises
        ------
        JobError
            If no job with ``job_id`` is tracked.
        '''

        record = self.get(job_id)
        stdout_preview = build_log_file_preview(record.manifest_path.parent, record.stdout_log, lines=lines, max_bytes=max_bytes)
        stderr_preview = build_log_file_preview(record.manifest_path.parent, record.stderr_log, lines=lines, max_bytes=max_bytes)
        return stdout_preview, stderr_preview

    def cancel(self, job_id: str) -> WorkbenchJobRecord:
        '''Terminate one running tracked job.

        Parameters
        ----------
        job_id : str
            Job identifier returned by :meth:`launch`.

        Returns
        -------
        WorkbenchJobRecord
            Tracked job record after cancellation. Unchanged if the job was
            not running.

        Raises
        ------
        JobError
            If no job with ``job_id`` is tracked.
        '''

        record = self.get(job_id)
        if record.status != "running":
            return record

        process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        elif record.pid is not None and _pid_alive(record.pid):
            try:
                os.kill(record.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._processes.pop(job_id, None)
        return self._finalize(record, status="cancelled", return_code=None)

    def _resolve(
        self,
        kind: WorkbenchJobKind,
        args: Sequence[str],
        *,
        cwd: str | Path | None,
        manifest: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[tuple[str, ...], Path]:
        resolved_cwd = Path(cwd).resolve() if cwd is not None else self.root
        if not resolved_cwd.is_dir():
            raise JobError(f"Job working directory does not exist: {resolved_cwd}")

        if kind == "vs_campaign":
            script = build_campaign_script(manifest, args, executable=self.executable)
            return ("/bin/sh", "-c", script), resolved_cwd

        prefix = JOB_KIND_COMMAND_PREFIX.get(kind)
        if prefix is None:
            valid = ", ".join(sorted((*JOB_KIND_COMMAND_PREFIX, "vs_campaign")))
            raise JobError(f"Unsupported job kind {kind!r}. Expected one of: {valid}.")

        command = (self.executable, *prefix, *(str(item) for item in args))
        return command, resolved_cwd

    def _spawn(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        stdout_log: Path,
        stderr_log: Path,
        returncode_path: Path,
    ) -> subprocess.Popen[Any]:
        shell_command = " ".join(shlex.quote(part) for part in command)
        quoted_returncode_path = shlex.quote(str(returncode_path))
        # Capture the real exit code before `echo`/redirection can overwrite $?, then
        # re-exit with it so the wrapper process's own exit code matches the job's.
        wrapped = f"{shell_command}; rc=$?; echo $rc > {quoted_returncode_path}; exit $rc"
        try:
            with stdout_log.open("wb") as stdout_handle, stderr_log.open("wb") as stderr_handle:
                return subprocess.Popen(
                    ["/bin/sh", "-c", wrapped],
                    cwd=cwd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
        except OSError as exc:
            raise JobError(f"Could not launch job: {exc}", status_code=500) from exc

    def _read(self, manifest_path: Path) -> WorkbenchJobRecord:
        return WorkbenchJobRecord.model_validate(read_data(manifest_path))

    def _refresh(self, record: WorkbenchJobRecord) -> WorkbenchJobRecord:
        if record.status != "running":
            return record

        process = self._processes.get(record.job_id)
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return record
            self._processes.pop(record.job_id, None)
            return self._finalize(record, status="completed" if return_code == 0 else "failed", return_code=return_code)

        if record.returncode_path.is_file():
            try:
                return_code = int(record.returncode_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                return_code = None
            if return_code is not None:
                return self._finalize(record, status="completed" if return_code == 0 else "failed", return_code=return_code)

        if not _pid_alive(record.pid):
            return self._finalize(record, status="failed", return_code=None)
        return record

    def _finalize(self, record: WorkbenchJobRecord, *, status: str, return_code: int | None) -> WorkbenchJobRecord:
        now = datetime.now(timezone.utc)
        updated = record.model_copy(update={"status": status, "return_code": return_code, "finished_at": now, "updated_at": now})
        write_model(updated.manifest_path, updated)
        return updated


# Functions
###############################################################################
## Private ##


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True
    try:
        return (proc_root / str(pid)).exists()
    except OSError:
        return True


## Public ##


def build_campaign_script(manifest: Sequence[dict[str, Any]] | None, args: Sequence[str], *, executable: str = "ocdocker") -> str:
    '''Build a POSIX shell script that runs one ``ocdocker`` command per manifest row.

    Shared by :meth:`JobManager.launch`/:meth:`JobManager.plan` (to actually
    build/preview a ``"vs_campaign"`` job's command) and
    :func:`OCDocker.Workbench.VSDesign.plan_vs_campaign` (to preview the same
    script before a job is ever created) — one source of truth for what a
    campaign script looks like.

    Parameters
    ----------
    manifest : Sequence[dict[str, Any]] or None
        One entry per sample: ``sample``, ``row_kind`` (``"vs"`` or
        ``"pipeline"``), ``receptor``, ``ligand``, ``box``, ``engines``
        (non-empty), optional ``rescoring_engines``.
    args : Sequence[str]
        Flags appended to every row's command (common to the whole batch).
    executable : str
        ``ocdocker`` executable invoked for each row.

    Returns
    -------
    str
        Generated shell script. Continues past a failing row; exits
        non-zero only if any row failed.

    Raises
    ------
    JobError
        If ``manifest`` is empty or a row is missing a required field.
    '''

    if not manifest:
        raise JobError("vs_campaign jobs require a non-empty manifest.")

    common_args = [str(item) for item in args]
    lines = ["set -u", f"total={len(manifest)}", "failed=0"]
    for index, row in enumerate(manifest, start=1):
        row_kind = str(row.get("row_kind") or "vs")
        if row_kind not in CAMPAIGN_ROW_KINDS:
            raise JobError(f"Manifest row {index} has an unsupported row_kind: {row_kind!r}.")
        prefix = JOB_KIND_COMMAND_PREFIX[row_kind]
        sample = str(row.get("sample") or f"row-{index}")
        for key in ("receptor", "ligand", "box"):
            if not row.get(key):
                raise JobError(f"Manifest row {index} ({sample}) is missing {key!r}.")
        engines = [str(item) for item in (row.get("engines") or []) if str(item).strip()]
        if not engines:
            raise JobError(f"Manifest row {index} ({sample}) has no engines.")

        row_command = [executable, *prefix, "--receptor", str(row["receptor"]), "--ligand", str(row["ligand"]), "--box", str(row["box"])]
        if row_kind == "vs":
            row_command.extend(["--engine", engines[0]])
        else:
            row_command.extend(["--engines", ",".join(engines)])
            rescoring_engines = [str(item) for item in (row.get("rescoring_engines") or []) if str(item).strip()]
            if rescoring_engines:
                row_command.extend(["--rescoring-engines", ",".join(rescoring_engines)])
        row_command.extend(common_args)

        quoted = " ".join(shlex.quote(part) for part in row_command)
        lines.append(f"echo '[sample {index}/{len(manifest)}] {shlex.quote(sample)}'")
        lines.append(f"{quoted} || failed=$((failed+1))")

    lines.append('echo "[vs_campaign] completed: $((total-failed))/${total} succeeded, ${failed} failed"')
    lines.append('if [ "$failed" -gt 0 ]; then exit 1; fi')
    return "\n".join(lines)


__all__ = [
    "CAMPAIGN_ROW_KINDS",
    "JOB_KIND_COMMAND_PREFIX",
    "WORKBENCH_JOBS_DIRNAME",
    "JobError",
    "JobManager",
    "build_campaign_script",
]
