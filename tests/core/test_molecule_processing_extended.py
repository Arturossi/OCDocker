#!/usr/bin/env python3

# Description
###############################################################################
'''
Additional coverage tests for Toolbox.MoleculeProcessing helpers.
'''

# Imports
###############################################################################
import os

from types import SimpleNamespace

import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc

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

def _atom_line(atom: str, res: str, chain: str = " ", element: str = "C", record: str = "ATOM  ") -> str:
    line = [" "] * 80
    line[0:6] = list(record[:6].ljust(6))
    line[12:16] = list(f"{atom:>4}"[:4])
    line[17:20] = list(f"{res:>3}"[:3])
    line[21] = chain
    line[76:78] = list(f"{element:>2}"[:2])
    return "".join(line) + "\n"


## Public ##

@pytest.mark.order(220)
@pytest.mark.parametrize(
    ("atom", "element", "expected"),
    [
        ("ABCDE", "C", "ABCD"),
        ("1HG1", "H", "1HG1"),
        ("CA", "C", " CA "),
        ("CA", "FE", "  CA"),
    ],
)
def test_format_atom_name_branches(atom, element, expected):
    assert ocmolproc._format_atom_name(atom, element) == expected


@pytest.mark.order(221)
def test_clean_for_dssp_nonexistent_file(tmp_path):
    rc = ocmolproc.clean_for_dssp(str(tmp_path / "missing.pdb"))
    assert rc == ocerror.ErrorCode.FILE_NOT_EXIST


@pytest.mark.order(222)
def test_clean_for_dssp_writes_header_cryst1_and_chain(tmp_path):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(_atom_line("CA", "ALA", chain=" "), encoding="utf-8")

    rc = ocmolproc.clean_for_dssp(str(pdb_path))
    assert rc == ocerror.ErrorCode.OK

    out = pdb_path.read_text(encoding="utf-8").splitlines()
    assert out[0].startswith("HEADER")
    assert out[1].startswith("CRYST1")
    assert out[2].startswith("ATOM")
    assert out[2][21] == "A"


@pytest.mark.order(223)
def test_clean_pdb_file_rejects_existing_output_without_overwrite(tmp_path):
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "output.pdb"
    input_path.write_text(_atom_line("CA", "ALA"), encoding="utf-8")
    output_path.write_text("existing", encoding="utf-8")

    rc = ocmolproc.clean_pdb_file(str(input_path), str(output_path), overwrite=False)
    assert rc == ocerror.ErrorCode.FILE_EXISTS


@pytest.mark.order(224)
def test_clean_pdb_file_keep_hetatm_controls_output(tmp_path):
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "output.pdb"
    input_path.write_text(
        _atom_line("CA", "ALA", chain=" ")
        + _atom_line("C1", "LIG", chain=" ", record="HETATM"),
        encoding="utf-8",
    )

    rc_no_het = ocmolproc.clean_pdb_file(str(input_path), str(output_path), overwrite=True, keep_hetatm=False)
    assert rc_no_het == ocerror.ErrorCode.OK
    out_no_het = output_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("ATOM") for line in out_no_het)
    assert not any(line.startswith("HETATM") for line in out_no_het)

    rc_keep_het = ocmolproc.clean_pdb_file(str(input_path), str(output_path), overwrite=True, keep_hetatm=True)
    assert rc_keep_het == ocerror.ErrorCode.OK
    out_keep_het = output_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("HETATM") for line in out_keep_het)
    atom_or_het = [line for line in out_keep_het if line.startswith("ATOM") or line.startswith("HETATM")]
    assert atom_or_het and all(line[21] == "A" for line in atom_or_het)


@pytest.mark.order(225)
def test_clean_pdb_file_returns_write_error_on_open_failure(monkeypatch, tmp_path):
    real_open = open
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "output.pdb"
    input_path.write_text(_atom_line("CA", "ALA"), encoding="utf-8")

    def fake_open(path, mode="r", *args, **kwargs):
        if os.fspath(path) == str(output_path) and "w" in mode:
            raise OSError("cannot write")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(ocmolproc, "open", fake_open, raising=False)

    rc = ocmolproc.clean_pdb_file(str(input_path), str(output_path), overwrite=True)
    assert rc == ocerror.ErrorCode.WRITE_FILE


@pytest.mark.order(226)
def test_convert_pdb_charmm_to_canonical_validates_inputs(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pdb"
    rc_missing = ocmolproc.convert_pdb_charmm_to_canonical(str(missing), str(tmp_path / "out.pdb"))
    assert rc_missing == ocerror.ErrorCode.FILE_NOT_EXIST

    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "out.pdb"
    input_path.write_text(_atom_line("CA", "ALA"), encoding="utf-8")
    output_path.write_text("existing", encoding="utf-8")

    rc_exists = ocmolproc.convert_pdb_charmm_to_canonical(str(input_path), str(output_path), overwrite=False)
    assert rc_exists == ocerror.ErrorCode.FILE_EXISTS

    monkeypatch.setattr(
        ocmolproc,
        "build_charmm_to_canonical_map",
        lambda: (_ for _ in ()).throw(ImportError("missing pdb2pqr")),
    )
    rc_import = ocmolproc.convert_pdb_charmm_to_canonical(str(input_path), str(tmp_path / "import_fail.pdb"), overwrite=True)
    assert rc_import == ocerror.ErrorCode.VALUE_ERROR

    monkeypatch.setattr(
        ocmolproc,
        "build_charmm_to_canonical_map",
        lambda: (_ for _ in ()).throw(RuntimeError("bad mapping")),
    )
    rc_unknown = ocmolproc.convert_pdb_charmm_to_canonical(str(input_path), str(tmp_path / "unknown_fail.pdb"), overwrite=True)
    assert rc_unknown == ocerror.ErrorCode.UNKNOWN


@pytest.mark.order(227)
def test_convert_pdb_charmm_to_canonical_success_and_in_place(monkeypatch, tmp_path):
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "converted" / "output.pdb"
    input_path.write_text(_atom_line("HD1", "HSD", element="H"), encoding="utf-8")

    monkeypatch.setattr(
        ocmolproc,
        "build_charmm_to_canonical_map",
        lambda: {("HSD", "HD1"): ("HSD", "HD1")},
    )

    info = []
    monkeypatch.setattr(ocmolproc.ocprint, "print_success", lambda message: info.append(message))

    rc = ocmolproc.convert_pdb_charmm_to_canonical(str(input_path), str(output_path), overwrite=True, collapse_resnames=True)
    assert rc == ocerror.ErrorCode.OK
    assert output_path.is_file()
    converted_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert converted_line[17:20].strip() == "HIS"
    assert converted_line[12:16] == " HD1"
    assert info

    in_place_input = tmp_path / "in_place.pdb"
    in_place_input.write_text(_atom_line("HD1", "HSD", element="H"), encoding="utf-8")
    rc_in_place = ocmolproc.convert_pdb_charmm_to_canonical(
        str(in_place_input),
        str(tmp_path / "ignored_output.pdb"),
        in_place=True,
        collapse_resnames=True,
    )
    assert rc_in_place == ocerror.ErrorCode.OK
    assert in_place_input.read_text(encoding="utf-8").splitlines()[0][17:20].strip() == "HIS"
    assert not os.path.exists(f"{in_place_input}.canonical_tmp")


@pytest.mark.order(228)
def test_convert_pdb_charmm_to_canonical_returns_write_error_on_write_failure(monkeypatch, tmp_path):
    real_open = open
    input_path = tmp_path / "input.pdb"
    output_path = tmp_path / "output.pdb"
    input_path.write_text(_atom_line("CA", "ALA"), encoding="utf-8")

    monkeypatch.setattr(ocmolproc, "build_charmm_to_canonical_map", lambda: {})

    def fake_open(path, mode="r", *args, **kwargs):
        if os.fspath(path) == str(output_path) and "w" in mode:
            raise OSError("cannot write output")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(ocmolproc, "open", fake_open, raising=False)

    rc = ocmolproc.convert_pdb_charmm_to_canonical(str(input_path), str(output_path), overwrite=True)
    assert rc == ocerror.ErrorCode.WRITE_FILE


@pytest.mark.order(229)
def test_needs_canonical_pdb_fix_handles_missing_mapping_and_read_errors(monkeypatch, tmp_path):
    warnings = []
    input_path = tmp_path / "input.pdb"
    input_path.write_text(_atom_line("CA", "ALA"), encoding="utf-8")

    monkeypatch.setattr(ocmolproc.ocprint, "print_warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        ocmolproc,
        "build_charmm_to_canonical_map",
        lambda: (_ for _ in ()).throw(ImportError("missing pdb2pqr")),
    )
    assert ocmolproc.needs_canonical_pdb_fix(str(input_path)) is False
    assert warnings

    warnings.clear()
    monkeypatch.setattr(ocmolproc, "build_charmm_to_canonical_map", lambda: {})
    monkeypatch.setattr(ocmolproc, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("read fail")), raising=False)
    assert ocmolproc.needs_canonical_pdb_fix(str(input_path)) is False
    assert warnings


@pytest.mark.order(230)
def test_needs_canonical_pdb_fix_detects_and_ignores_changes(monkeypatch, tmp_path):
    changed_path = tmp_path / "changed.pdb"
    unchanged_path = tmp_path / "unchanged.pdb"
    changed_path.write_text(_atom_line("HD1", "HSD", element="H"), encoding="utf-8")
    unchanged_path.write_text(_atom_line("CA", "ALA", element="C"), encoding="utf-8")

    monkeypatch.setattr(
        ocmolproc,
        "build_charmm_to_canonical_map",
        lambda: {("HSD", "HD1"): ("HSD", "HD1")},
    )
    assert ocmolproc.needs_canonical_pdb_fix(str(changed_path), collapse_resnames=True) is True
    assert ocmolproc.needs_canonical_pdb_fix(str(unchanged_path), collapse_resnames=True) is False
    assert ocmolproc.needs_canonical_pdb_fix(str(tmp_path / "missing.pdb")) is False


@pytest.mark.order(231)
def test_split_poses_builds_expected_command(monkeypatch, tmp_path):
    ligand_path = tmp_path / "ligand.pdbqt"
    ligand_path.write_text("LIG", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(
        ocmolproc,
        "get_config",
        lambda: SimpleNamespace(vina=SimpleNamespace(split_executable="vina_split")),
    )
    monkeypatch.setattr(ocmolproc.ocff, "normalize_path", lambda p: os.path.abspath(os.fspath(p)))
    monkeypatch.setattr(
        ocmolproc.ocrun,
        "run",
        lambda cmd, logFile="": captured.update({"cmd": cmd, "logFile": logFile}) or ocerror.ErrorCode.OK,
    )

    output_dir = tmp_path / "poses"
    rc = ocmolproc.split_poses(str(ligand_path), "ligA", str(output_dir), suffix="_x", logFile="run.log")
    assert rc == ocerror.ErrorCode.OK
    assert output_dir.is_dir()
    assert captured["cmd"][0] == "vina_split"
    assert captured["cmd"][1:4] == ["--input", os.path.abspath(str(ligand_path)), "--flex"]
    assert captured["cmd"][4] == "''"
    assert captured["cmd"][5] == "--ligand"
    assert captured["cmd"][6].endswith("ligA_x")
    assert captured["logFile"] == "run.log"
