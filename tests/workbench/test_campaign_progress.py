#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for structured vs_campaign progress parsing.
"""

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench.CampaignProgress import parse_campaign_progress

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

# Captured verbatim from a real `snakemake --cores 2` run against the bundled
# vs_campaign.smk-style rule (Snakemake 9.16.3), all samples succeeding.
_SNAKEMAKE_SUCCESS_LOG = """[Thu Jul  2 00:55:14 2026]
localrule work:
    output: results/sample_002/done.txt
    jobid: 2
    reason: Missing output files: results/sample_002/done.txt
    wildcards: sample=sample_002
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:14 2026]
localrule work:
    output: results/sample_003/done.txt
    jobid: 3
    reason: Missing output files: results/sample_003/done.txt
    wildcards: sample=sample_003
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:14 2026]
Finished jobid: 2 (Rule: work)
1 of 4 steps (25%) done
Select jobs to execute...
Execute 1 jobs...

[Thu Jul  2 00:55:14 2026]
localrule work:
    output: results/sample_001/done.txt
    jobid: 1
    reason: Missing output files: results/sample_001/done.txt
    wildcards: sample=sample_001
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:14 2026]
Finished jobid: 3 (Rule: work)
2 of 4 steps (50%) done
[Thu Jul  2 00:55:14 2026]
Finished jobid: 1 (Rule: work)
3 of 4 steps (75%) done
[Thu Jul  2 00:55:14 2026]
localrule all:
    input: results/sample_001/done.txt, results/sample_002/done.txt, results/sample_003/done.txt
    jobid: 0
    reason: Input files updated by another job
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:14 2026]
Finished jobid: 0 (Rule: all)
4 of 4 steps (100%) done
Complete log(s): /tmp/x/.snakemake/log/x.log
"""

# Captured verbatim from a real run where sample_002's row fails and
# --keep-going lets sample_001 finish anyway.
_SNAKEMAKE_FAILURE_LOG = """[Thu Jul  2 00:55:33 2026]
localrule work:
    output: results/sample_002/done.txt
    jobid: 2
    reason: Missing output files: results/sample_002/done.txt
    wildcards: sample=sample_002
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:33 2026]
localrule work:
    output: results/sample_001/done.txt
    jobid: 1
    reason: Missing output files: results/sample_001/done.txt
    wildcards: sample=sample_001
    resources: tmpdir=/tmp
[Thu Jul  2 00:55:33 2026]
Finished jobid: 1 (Rule: work)
1 of 3 steps (33%) done
RuleException:
CalledProcessError in file "x", line 13:
Command 'x' returned non-zero exit status 1.
[Thu Jul  2 00:55:33 2026]
Error in rule work:
    message: None
    jobid: 2
    output: results/sample_002/done.txt
    shell:
        x
        (command exited with non-zero exit code)
Exiting because a job execution failed. Look below for error messages
"""

_SHELL_LOG = """[sample 1/3] s1
ran: vs --receptor r.pdb
[sample 2/3] s2
simulated docking failure
[sample 3/3] s3
ran: pipeline --receptor r.pdb
[vs_campaign] completed: 2/3 succeeded, 1 failed
"""

# Functions
###############################################################################
## Public ##


def test_parse_campaign_progress_snakemake_success() -> None:
    '''A successful Snakemake run reports every sample done and 100% overall.'''

    result = parse_campaign_progress(_SNAKEMAKE_SUCCESS_LOG, expected_samples=["sample_001", "sample_002", "sample_003"])

    assert result["engine"] == "snakemake"
    assert result["overall"] == {"completed_steps": 4, "total_steps": 4, "percent": 100}
    assert result["samples"]["sample_001"]["status"] == "done"
    assert result["samples"]["sample_002"]["status"] == "done"
    assert result["samples"]["sample_003"]["status"] == "done"


def test_parse_campaign_progress_snakemake_failure() -> None:
    '''A failing sample is reported as failed; a succeeding one stays done.'''

    result = parse_campaign_progress(_SNAKEMAKE_FAILURE_LOG, expected_samples=["sample_001", "sample_002"])

    assert result["engine"] == "snakemake"
    assert result["samples"]["sample_001"]["status"] == "done"
    assert result["samples"]["sample_002"]["status"] == "failed"


def test_parse_campaign_progress_snakemake_pending_samples_not_yet_started() -> None:
    '''Expected samples not yet mentioned in the log stay pending, not omitted.'''

    result = parse_campaign_progress(_SNAKEMAKE_FAILURE_LOG, expected_samples=["sample_001", "sample_002", "sample_003"])

    assert result["samples"]["sample_003"]["status"] == "pending"


def test_parse_campaign_progress_shell_engine() -> None:
    '''The shell engine format reports started samples as done and aggregate counts.'''

    result = parse_campaign_progress(_SHELL_LOG, expected_samples=["s1", "s2", "s3"])

    assert result["engine"] == "shell"
    assert result["overall"] == {"completed_steps": 3, "total_steps": 3, "percent": 100, "succeeded": 2, "failed": 1}
    assert result["samples"]["s1"]["status"] == "done"
    assert result["samples"]["s2"]["status"] == "done"
    assert result["samples"]["s3"]["status"] == "done"


def test_parse_campaign_progress_unknown_format_degrades_gracefully() -> None:
    '''Arbitrary non-campaign log text degrades to engine "unknown", not an exception.'''

    result = parse_campaign_progress("just some random log text\nno markers here at all")

    assert result["engine"] == "unknown"
    assert result["overall"] is None
    assert result["samples"] == {}


def test_parse_campaign_progress_empty_log() -> None:
    '''An empty log string does not raise.'''

    result = parse_campaign_progress("")

    assert result["engine"] == "unknown"
    assert result["samples"] == {}


def test_parse_campaign_progress_without_expected_samples() -> None:
    '''expected_samples is optional; samples are still discovered from the log.'''

    result = parse_campaign_progress(_SHELL_LOG)

    assert set(result["samples"]) == {"s1", "s2", "s3"}
