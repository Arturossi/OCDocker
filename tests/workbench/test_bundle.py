#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for prepared Workbench run bundles.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

import pytest

from OCDocker.Workbench import OCScoreInputSpec
from OCDocker.Workbench import OCScoreStudySpec
from OCDocker.Workbench import build_run_bundle
from OCDocker.Workbench import read_run_manifest
from OCDocker.Workbench import read_spec

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
## Private ##


def _study_spec() -> OCScoreStudySpec:
    '''Return a minimal OCScore study spec for bundle tests.

    Returns
    -------
    OCScoreStudySpec
        Test study spec.
    '''

    return OCScoreStudySpec(
        name="bundle-study",
        protocol="smoke-test",
        inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
        output_dir="out/bundle",
    )


## Public ##


def test_build_run_bundle_writes_expected_files(tmp_path) -> None:
    '''Run bundle creation writes spec, plan, manifest, and bundle files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    bundle_dir = tmp_path / "run-001"
    bundle = build_run_bundle(
        _study_spec(),
        bundle_dir,
        run_id="run-001",
        ocdocker_executable="ocdocker-dev",
    )

    assert bundle.root == bundle_dir
    assert bundle.spec_path.is_file()
    assert bundle.plan_path.is_file()
    assert bundle.run_manifest_path.is_file()
    assert bundle.bundle_manifest_path.is_file()
    assert read_spec(bundle.spec_path).name == "bundle-study"
    assert read_run_manifest(bundle.run_manifest_path).workspace == bundle_dir
    plan_payload = json.loads(bundle.plan_path.read_text(encoding="utf-8"))
    assert plan_payload["plan"]["command"][:3] == [
        "ocdocker-dev",
        "ocscore",
        "train",
    ]
    assert plan_payload["shell_command"].startswith("ocdocker-dev ocscore train")


def test_build_run_bundle_refuses_existing_files_without_overwrite(tmp_path) -> None:
    '''Existing bundle files require explicit overwrite.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    bundle_dir = tmp_path / "run-001"
    build_run_bundle(_study_spec(), bundle_dir, run_id="run-001")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_run_bundle(_study_spec(), bundle_dir, run_id="run-001")

    bundle = build_run_bundle(
        _study_spec(), bundle_dir, run_id="run-001", overwrite=True
    )
    assert bundle.bundle_manifest_path.is_file()
