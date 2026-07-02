#!/usr/bin/env python3

# Description
###############################################################################
'''
Structured per-sample progress for ``vs_campaign`` jobs, parsed from the
job's own stdout — no Snakemake internals, no extra process. Supports both
execution engines (see :mod:`OCDocker.Workbench.Jobs`): the ``"snakemake"``
engine's own console output (job start/finish/error blocks, ``X of Y steps``
lines) and the ``"shell"`` engine's ``[sample i/N]`` markers.

Text-format parsing is inherently tied to the exact wording a given tool
version prints; a future Snakemake major version could change it. When
nothing recognizable is found, this degrades to ``engine: "unknown"`` with
an empty ``samples`` mapping — never an exception.
'''

# Imports
###############################################################################
from __future__ import annotations

import re

from typing import Any
from typing import Sequence

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

# Progress parsing needs the full picture (early job-start blocks map jobid -> sample
# for "Finished jobid: N" lines that may arrive much later), so callers should read a
# much larger tail than the default human-readable log preview.
CAMPAIGN_PROGRESS_LOG_LINE_LIMIT = 20000
CAMPAIGN_PROGRESS_LOG_BYTE_LIMIT = 20_000_000

_SNAKEMAKE_RULE_BLOCK_RE = re.compile(r"^(?:local)?rule (\w+):\s*$")
_SNAKEMAKE_ERROR_BLOCK_RE = re.compile(r"^Error in rule (\w+):\s*$")
_SNAKEMAKE_JOBID_RE = re.compile(r"^\s*jobid:\s*(\d+)\s*$")
_SNAKEMAKE_WILDCARDS_SAMPLE_RE = re.compile(r"^\s*wildcards:.*\bsample=([^,\s]+)")
_SNAKEMAKE_FINISHED_RE = re.compile(r"^Finished jobid:\s*(\d+)\s*\(Rule:\s*(\w+)\)")
_SNAKEMAKE_PROGRESS_RE = re.compile(r"^(\d+) of (\d+) steps \((\d+)%\) done")
_SHELL_SAMPLE_RE = re.compile(r"^\[sample (\d+)/(\d+)\]\s+(.+?)\s*$")
_SHELL_COMPLETE_RE = re.compile(r"^\[vs_campaign\] completed: (\d+)/(\d+) succeeded, (\d+) failed")

# Functions
###############################################################################
## Private ##


def _empty_samples(expected_samples: Sequence[str] | None) -> dict[str, dict[str, Any]]:
    '''Seed a samples mapping with "pending" status for every expected sample.

    Parameters
    ----------
    expected_samples : Sequence[str] or None
        Sample names known ahead of parsing (e.g. from the launched manifest).

    Returns
    -------
    dict[str, dict[str, Any]]
        One ``{"status": "pending", "jobid": None}`` entry per expected sample.
    '''

    return {name: {"status": "pending", "jobid": None} for name in (expected_samples or ())}


def _parse_snakemake_progress(log_text: str, expected_samples: Sequence[str] | None) -> dict[str, Any]:
    '''Parse structured per-sample progress from Snakemake's own console output.

    Parameters
    ----------
    log_text : str
        A ``vs_campaign`` (``engine="snakemake"``) job's stdout/stderr text.
    expected_samples : Sequence[str] or None
        Sample names known ahead of parsing.

    Returns
    -------
    dict[str, Any]
        ``{"engine": "snakemake", "overall": {...} | None, "samples": {...}}``.
    '''

    samples = _empty_samples(expected_samples)
    jobid_to_sample: dict[int, str] = {}
    current_jobid: int | None = None
    in_error_block = False
    overall: dict[str, Any] | None = None

    for line in log_text.splitlines():
        if _SNAKEMAKE_RULE_BLOCK_RE.match(line):
            current_jobid = None
            in_error_block = False
            continue
        error_match = _SNAKEMAKE_ERROR_BLOCK_RE.match(line)
        if error_match:
            in_error_block = True
            current_jobid = None
            continue
        jobid_match = _SNAKEMAKE_JOBID_RE.match(line)
        if jobid_match:
            current_jobid = int(jobid_match.group(1))
            if in_error_block:
                sample = jobid_to_sample.get(current_jobid)
                if sample:
                    samples[sample] = {"status": "failed", "jobid": current_jobid}
                in_error_block = False
            continue
        wildcards_match = _SNAKEMAKE_WILDCARDS_SAMPLE_RE.match(line)
        if wildcards_match and current_jobid is not None:
            sample = wildcards_match.group(1)
            jobid_to_sample[current_jobid] = sample
            samples[sample] = {"status": "running", "jobid": current_jobid}
            continue
        finished_match = _SNAKEMAKE_FINISHED_RE.match(line)
        if finished_match:
            jobid = int(finished_match.group(1))
            rule = finished_match.group(2)
            sample = jobid_to_sample.get(jobid)
            if sample and rule != "all":
                samples[sample]["status"] = "done"
            continue
        progress_match = _SNAKEMAKE_PROGRESS_RE.match(line)
        if progress_match:
            overall = {
                "completed_steps": int(progress_match.group(1)),
                "total_steps": int(progress_match.group(2)),
                "percent": int(progress_match.group(3)),
            }

    return {"engine": "snakemake", "overall": overall, "samples": samples}


def _parse_shell_progress(log_text: str, expected_samples: Sequence[str] | None) -> dict[str, Any]:
    '''Parse best-effort per-sample progress from the shell engine's ``[sample i/N]`` markers.

    The shell engine (:func:`OCDocker.Workbench.Jobs.build_campaign_script`)
    never echoes an individual row's pass/fail — only a final aggregate
    count — so a sample already marked "done" here means "this row's command
    finished running", not "it succeeded". Use the Snakemake engine for
    reliable per-sample outcomes.

    Parameters
    ----------
    log_text : str
        A ``vs_campaign`` (``engine="shell"``) job's stdout text.
    expected_samples : Sequence[str] or None
        Sample names known ahead of parsing.

    Returns
    -------
    dict[str, Any]
        ``{"engine": "shell", "overall": {...} | None, "samples": {...}}``.
    '''

    samples = _empty_samples(expected_samples)
    started_order: list[str] = []
    overall: dict[str, Any] | None = None

    for line in log_text.splitlines():
        sample_match = _SHELL_SAMPLE_RE.match(line)
        if sample_match:
            _index, _total, sample = sample_match.groups()
            if started_order:
                samples[started_order[-1]] = {"status": "done"}
            samples[sample] = {"status": "running"}
            started_order.append(sample)
            continue
        complete_match = _SHELL_COMPLETE_RE.match(line)
        if complete_match:
            succeeded, total, failed = (int(group) for group in complete_match.groups())
            if started_order:
                samples[started_order[-1]] = {"status": "done"}
            overall = {
                "completed_steps": succeeded + failed,
                "total_steps": total,
                "percent": round((succeeded + failed) / total * 100) if total else 100,
                "succeeded": succeeded,
                "failed": failed,
            }

    return {"engine": "shell", "overall": overall, "samples": samples}


## Public ##


def parse_campaign_progress(log_text: str, *, expected_samples: Sequence[str] | None = None) -> dict[str, Any]:
    '''Parse structured per-sample progress from a ``vs_campaign`` job's log.

    Tries the Snakemake console-output format first, then the shell engine's
    ``[sample i/N]`` markers, and degrades to ``engine: "unknown"`` with an
    empty ``samples`` mapping for anything else — never raises.

    Parameters
    ----------
    log_text : str
        A tracked job's stdout (and, for the error-block detection in the
        Snakemake format, ideally its stderr too — callers may concatenate
        both; see :func:`OCDocker.Workbench.Jobs.JobManager.logs`).
    expected_samples : Sequence[str] or None
        Sample names known ahead of parsing (e.g. from the launched
        manifest); samples not yet mentioned in the log are reported as
        ``"pending"`` instead of being omitted.

    Returns
    -------
    dict[str, Any]
        ``{"engine": "snakemake" | "shell" | "unknown", "overall": {...} | None, "samples": {name: {"status": ...}}}``.
    '''

    if log_text and re.search(_SNAKEMAKE_RULE_BLOCK_RE.pattern, log_text, re.MULTILINE):
        return _parse_snakemake_progress(log_text, expected_samples)
    if log_text and "[sample " in log_text:
        return _parse_shell_progress(log_text, expected_samples)
    return {"engine": "unknown", "overall": None, "samples": _empty_samples(expected_samples)}


__all__ = [
    "CAMPAIGN_PROGRESS_LOG_BYTE_LIMIT",
    "CAMPAIGN_PROGRESS_LOG_LINE_LIMIT",
    "parse_campaign_progress",
]
