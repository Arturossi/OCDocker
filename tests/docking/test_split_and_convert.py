#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for split_and_convert error handling.
'''

# Imports
###############################################################################
from pathlib import Path

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Conversion as occonversion

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

@pytest.mark.order(1)
def test_split_and_convert_invalid_input(tmp_path):
    invalid_path = tmp_path / "molecule.bad"
    invalid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(invalid_path), str(tmp_path), "sdf")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(2)
def test_split_and_convert_invalid_output(tmp_path):
    valid_path = tmp_path / "molecule.smi"
    valid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(valid_path), str(tmp_path), "bad")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(3)
def test_split_and_convert_success_writes_expected_files(monkeypatch, tmp_path):
    input_path = tmp_path / "molecule.smi"
    input_path.write_text("CCO", encoding="utf-8")

    class _FakeMol:
        def __init__(self, title):
            self.title = title

        def write(self, ext, outfile, overwrite=False):
            _ = overwrite
            Path(outfile).write_text(f"{ext}\n", encoding="utf-8")

    monkeypatch.setattr(
        occonversion.pybel,
        "readfile",
        lambda _ext, _path: [_FakeMol(" ligand one "), _FakeMol("none ligand two")],
    )

    rc = occonversion.split_and_convert(str(input_path), str(tmp_path), "mol2", overwrite=True)
    assert rc == ocerror.Error.ok()
    assert (tmp_path / "ligand_one.mol2").is_file()
    assert (tmp_path / "ligand_two.mol2").is_file()


@pytest.mark.order(4)
def test_split_and_convert_returns_write_file_when_write_raises(monkeypatch, tmp_path):
    input_path = tmp_path / "molecule.smi"
    input_path.write_text("CCO", encoding="utf-8")

    class _FakeMol:
        title = "bad molecule"

        def write(self, _ext, _outfile, overwrite=False):
            _ = overwrite
            raise RuntimeError("write failed")

    monkeypatch.setattr(occonversion.pybel, "readfile", lambda _ext, _path: [_FakeMol()])

    rc = occonversion.split_and_convert(str(input_path), str(tmp_path), "mol2", overwrite=True)
    assert rc == ocerror.ErrorCode.WRITE_FILE
