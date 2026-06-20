#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench read-only preflight checks.
'''

# Imports
###############################################################################
from __future__ import annotations

import sys

from OCDocker.Workbench import FeaturePolicySelection
from OCDocker.Workbench import OCScoreInputSpec
from OCDocker.Workbench import OCScoreStudySpec
from OCDocker.Workbench import preflight_spec_file
from OCDocker.Workbench import write_model

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

# Functions
###############################################################################
## Public ##


def test_preflight_spec_file_reports_ready_when_inputs_exist(tmp_path) -> None:
    '''Preflight reports ready when required inputs and executable exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    raw_dir = tmp_path / "raw_prepare"
    raw_dir.mkdir()
    spec_path = write_model(
        tmp_path / "study.yml",
        OCScoreStudySpec(
            name="ready-study",
            protocol="smoke-test",
            inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
            output_dir="out/ready-study",
        ),
    )

    report = preflight_spec_file(spec_path, ocdocker_executable=sys.executable)

    assert report.ready is True
    assert report.error_count == 0
    assert report.planned_command[0] == sys.executable
    assert any(
        check.code == "input.raw_input_dir" and check.passed for check in report.checks
    )


def test_preflight_spec_file_reports_missing_inputs_and_executable(tmp_path) -> None:
    '''Preflight reports missing inputs and executables as errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    spec_path = write_model(
        tmp_path / "study.yml",
        OCScoreStudySpec(
            name="missing-study",
            protocol="smoke-test",
            inputs=OCScoreInputSpec(raw_input_dir="missing_raw"),
            output_dir="out/missing-study",
            feature_policies=FeaturePolicySelection(policy_dirs=("missing_policies",)),
        ),
    )

    report = preflight_spec_file(
        spec_path,
        ocdocker_executable="definitely-missing-ocdocker",
    )

    assert report.ready is False
    assert report.error_count >= 2
    failed_codes = {check.code for check in report.checks if not check.passed}
    assert "command.executable" in failed_codes
    assert "input.raw_input_dir" in failed_codes
    assert "feature_policy.dir" in failed_codes


def test_preflight_spec_file_reports_non_runnable_executable_path(tmp_path) -> None:
    '''Preflight rejects executable paths that are not runnable.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    raw_dir = tmp_path / "raw_prepare"
    executable_path = tmp_path / "not-executable"
    raw_dir.mkdir()
    executable_path.write_text("#!/bin/sh\n")
    spec_path = write_model(
        tmp_path / "study.yml",
        OCScoreStudySpec(
            name="not-runnable-study",
            protocol="smoke-test",
            inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
            output_dir="out/not-runnable-study",
        ),
    )

    report = preflight_spec_file(
        spec_path,
        ocdocker_executable=str(executable_path),
    )

    executable_check = next(
        check for check in report.checks if check.code == "command.executable"
    )
    assert report.ready is False
    assert report.error_count == 1
    assert executable_check.passed is False
    assert executable_check.path == executable_path
    assert "not runnable" in executable_check.message


def test_preflight_spec_file_reports_existing_output_as_warning(tmp_path) -> None:
    '''Existing planned output paths are warnings, not blocking errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    raw_dir = tmp_path / "raw_prepare"
    output_dir = tmp_path / "out" / "existing-study"
    raw_dir.mkdir()
    output_dir.mkdir(parents=True)
    spec_path = write_model(
        tmp_path / "study.yml",
        OCScoreStudySpec(
            name="existing-study",
            protocol="smoke-test",
            inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
            output_dir="out/existing-study",
        ),
    )

    report = preflight_spec_file(spec_path, ocdocker_executable=sys.executable)

    assert report.ready is True
    assert report.error_count == 0
    assert report.warning_count == 1
    assert any(check.code == "output.exists" for check in report.checks)
