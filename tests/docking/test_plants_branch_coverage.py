#!/usr/bin/env python3

# Description
###############################################################################
'''
Targeted branch coverage tests for PLANTS gaps.
'''

# Imports
###############################################################################
import builtins
import json
import os

from pathlib import Path
from types import SimpleNamespace

import pytest

import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

def _dummy_ligand(path: str, name: str = "lig") -> ocl.Ligand:
    lig = object.__new__(ocl.Ligand)
    lig.path = path
    lig.name = name
    return lig


def _dummy_receptor(path: str, mol2_path: str = "", name: str = "rec") -> ocr.Receptor:
    rec = object.__new__(ocr.Receptor)
    rec.path = path
    rec.mol2_path = mol2_path
    rec.name = name
    return rec


def _cfg(tmp_path: Path):
    plants_cfg = SimpleNamespace(
        executable="/missing/plants",
        scoring_functions=["chemplp", "plp"],
        scoring="chemplp",
        cluster_structures=2,
        rescoring_mode="simplex",
        search_speed=1,
        cluster_rmsd=2.0,
    )
    return SimpleNamespace(plants=plants_cfg, logdir=str(tmp_path))


## Public ##

def test_plants_init_and_private_parser_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(ocplants, "get_binding_site", lambda *_a, **_k: ocerror.Error.file_not_exist())

    _ = ocplants.PLANTS(
        config_path=str(tmp_path / "p.conf"),
        box_file=str(tmp_path / "box0.pdb"),
        receptor=object(),
        prepared_receptor_path=str(tmp_path / "prep_rec.mol2"),
        ligand=object(),
        prepared_ligand_path=str(tmp_path / "prep_lig.mol2"),
        plants_log=str(tmp_path / "plants.log"),
        output_plants=str(tmp_path / "out"),
    )

    monkeypatch.setattr(ocplants, "get_binding_site", lambda *_a, **_k: ((1.0, 2.0, 3.0), 7.5))
    monkeypatch.setattr(ocplants.ocff, "safe_create_dir", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocplants, "get_config", lambda: _cfg(tmp_path))
    monkeypatch.setattr(ocplants.PLANTS, "write_config_file", lambda self: ocerror.Error.ok())

    rec = _dummy_receptor(str(tmp_path / "rec.pdb"))
    lig = _dummy_ligand(str(tmp_path / "lig.smi"), name="lig1")

    inst = ocplants.PLANTS(
        config_path=str(tmp_path / "cfg.txt"),
        box_file=str(tmp_path / "box0.pdb"),
        receptor=rec,
        prepared_receptor_path=str(tmp_path / "prep_rec.mol2"),
        ligand=lig,
        prepared_ligand_path=str(tmp_path / "prep_lig.pdbqt"),
        plants_log=str(tmp_path / "plants.log"),
        output_plants=str(tmp_path / "out"),
    )
    assert inst.prepared_ligand.endswith(".mol2")

    assert inst._PLANTS__parse_ligand_path("bad") == ""

    rec2 = _dummy_receptor(str(tmp_path / "rec2.pdb"), mol2_path="")
    monkeypatch.setattr(ocplants.occonversion, "convert_mols", lambda *_a, **_k: ocerror.Error.ok())
    monkeypatch.setattr(ocplants.os.path, "isfile", lambda path: not str(path).endswith(".mol2"))
    assert inst._PLANTS__parse_receptor_path(rec2, forceMol2=True) is None

    rec3 = _dummy_receptor("")
    assert inst._PLANTS__parse_receptor_path(rec3, forceMol2=False) is None

    monkeypatch.setattr(ocplants.os.path, "isfile", lambda _p: False)
    assert inst._PLANTS__parse_receptor_path("/tmp/missing.pdb", forceMol2=False) == ""
    assert inst._PLANTS__parse_receptor_path(12345, forceMol2=False) == ""


def test_plants_print_rescore_and_prepare_receptor_guards(monkeypatch, capsys, tmp_path):
    class Prep:
        def get_receptor_command(self, *_a, **_k):
            return ["prep", "rec"]

        def get_ligand_command(self, *_a, **_k):
            return ["prep", "lig"]

        def prepare_receptor(self, *_a, **_k):
            return ocerror.Error.ok()

    inst = object.__new__(ocplants.PLANTS)
    inst.name = "p"
    inst.box_file = "b"
    inst.config = "c"
    inst.input_receptor = None
    inst.input_receptor_path = ""
    inst.prepared_receptor = ""
    inst.input_ligand = None
    inst.input_ligand_path = ""
    inst.prepared_ligand = ""
    inst.plants_log = ""
    inst.output_plants = ""
    inst.output_csv = ""
    inst.plants_cmd = []
    inst.preparation_strategy = Prep()

    inst.print_attributes()
    printed = capsys.readouterr().out
    assert "Prepared receptor command:   '-'" in printed
    assert "Prepared ligand command:     '-'" in printed

    monkeypatch.setattr(
        ocplants.PLANTS,
        "get_rescore_log_paths",
        lambda self, onlyBest=False: [
            str(tmp_path / "abc" / "ranking.csv"),
            str(tmp_path / "run_chemplp" / "ranking.csv"),
        ],
    )

    def fake_read(path, onlyBest=False):
        if "abc" in path:
            return {1: {"S1": [1.0, 2.0], "S2": "bad", "S3": [3.0], "S4": 4.0}}
        return {}

    monkeypatch.setattr(ocplants, "read_log", fake_read)
    data = inst.read_rescore_logs(onlyBest=False)
    assert "plants_abc" in data
    assert data["plants_abc"]["S3"] == 3.0
    assert data["plants_abc"]["S4"] == 4.0
    assert data["plants_chemplp"] == {}

    inst.input_receptor_path = ""
    inst.prepared_receptor = "ok.mol2"
    assert inst.run_prepare_receptor() == ocerror.Error.file_not_exist()

    inst.input_receptor_path = "in.pdb"
    inst.prepared_receptor = ""
    assert inst.run_prepare_receptor() == ocerror.Error.file_not_exist()


def test_plants_run_plants_cleanup_paths(monkeypatch, tmp_path):
    out_dir = tmp_path / "plants_out"
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    inst = object.__new__(ocplants.PLANTS)
    inst.output_plants = str(out_dir)
    inst.input_ligand = _dummy_ligand("lig.smi", name="ligX")
    inst.config = str(tmp_path / "cfg.txt")
    inst.plants_log = str(tmp_path / "plants.log")
    inst.plants_cmd = ["plants", "--mode", "screen", inst.config]

    removed = {"rmtree": 0, "rmdir": 0}

    monkeypatch.setattr(ocplants, "get_config", lambda: _cfg(tmp_path))
    monkeypatch.setattr(ocplants.shutil, "rmtree", lambda *_a, **_k: removed.__setitem__("rmtree", removed["rmtree"] + 1))
    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    monkeypatch.setattr(ocplants.ocrun, "run", lambda *_a, **_k: (0, "stderr"))
    monkeypatch.setattr(
        ocplants,
        "glob",
        lambda pattern: [str(tmp_path / "a.pid")] if "PLANTS" in pattern else [str(tmp_path / "bad.mol2")],
    )
    monkeypatch.setattr(ocplants.os, "remove", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))

    rc_overwrite = inst.run_plants(overwrite=True)
    assert rc_overwrite == 0
    assert removed["rmtree"] >= 1

    monkeypatch.setattr(ocplants.os.path, "isdir", lambda p: p == str(run_dir))
    monkeypatch.setattr(ocplants.os, "listdir", lambda _p: [])
    monkeypatch.setattr(ocplants.os, "rmdir", lambda _p: removed.__setitem__("rmdir", removed["rmdir"] + 1))
    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: None)
    monkeypatch.setattr(ocplants, "glob", lambda _p: [])
    monkeypatch.setattr(ocplants.ocrun, "run", lambda *_a, **_k: 0)

    rc_no_overwrite = inst.run_plants(overwrite=False)
    assert rc_no_overwrite == 0
    assert removed["rmdir"] >= 1


def test_plants_box_binding_read_and_digest_paths(monkeypatch, tmp_path):
    real_get_binding_site = ocplants.get_binding_site

    monkeypatch.setattr(ocplants, "get_binding_site", lambda *_a, **_k: "bad")
    rc_bad_binding = ocplants.box_to_plants("box.pdb", "conf.txt", "rec.mol2", "lig.mol2", str(tmp_path))
    assert rc_bad_binding == ocerror.Error.read_file()

    monkeypatch.setattr(ocplants, "get_binding_site", lambda *_a, **_k: ((1.0, 2.0, 3.0), None))
    rc_none_radius = ocplants.box_to_plants("box.pdb", "conf.txt", "rec.mol2", "lig.mol2", str(tmp_path))
    assert rc_none_radius == ocerror.Error.read_file()

    monkeypatch.setattr(ocplants, "get_binding_site", lambda *_a, **_k: (None, 5.0))
    rc_none_center = ocplants.box_to_plants("box.pdb", "conf.txt", "rec.mol2", "lig.mol2", str(tmp_path))
    assert rc_none_center == ocerror.Error.read_file()
    monkeypatch.setattr(ocplants, "get_binding_site", real_get_binding_site)

    box_no_remark = tmp_path / "box_no_remark.pdb"
    box_no_remark.write_text("HEADER                       1.0     2.0     3.0     4.0     5.0     6.0\n", encoding="utf-8")
    assert isinstance(ocplants.get_binding_site(str(box_no_remark)), int)

    box_no_header = tmp_path / "box_no_header.pdb"
    box_no_header.write_text("REMARK                        1.0     2.0     3.0\n", encoding="utf-8")
    assert isinstance(ocplants.get_binding_site(str(box_no_header)), int)

    assert ocplants.get_pose_index_from_file_path("ligand_9") == 9

    rank = tmp_path / "ranking.csv"
    rank.write_text("dummy\n", encoding="utf-8")

    class EmptyDF:
        shape = (0, 0)

    monkeypatch.setattr(ocplants.pd, "read_csv", lambda *_a, **_k: EmptyDF())
    assert ocplants.read_log(str(rank)) == {}

    monkeypatch.setattr(ocplants.pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("csv fail")))
    monkeypatch.setattr(ocplants, "get_config", lambda: _cfg(tmp_path))
    assert ocplants.read_log(str(rank)) == {}

    digest = tmp_path / "digest.json"
    log = tmp_path / "ranking.csv"
    log.write_text("x", encoding="utf-8")

    monkeypatch.setattr(ocplants.ocvalidation, "validate_digest_extension", lambda *_a, **_k: False)
    assert ocplants.generate_digest(str(digest), str(log), overwrite=True, digestFormat="hdf5") == ocerror.Error.unsupported_extension()

    monkeypatch.setattr(ocplants.ocvalidation, "validate_digest_extension", lambda *_a, **_k: True)
    digest.write_text("{bad json", encoding="utf-8")
    assert ocplants.generate_digest(str(digest), str(log), overwrite=True, digestFormat="json") == ocerror.Error.file_not_exist()

    digest.write_text(json.dumps({}), encoding="utf-8")
    assert ocplants.generate_digest(str(digest), str(log), overwrite=True, digestFormat="txt") == ocerror.Error.wrong_type()


def test_plants_file_writers_and_run_plants_function(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ocplants, "get_config", lambda: cfg)

    conf = tmp_path / "plants_conf.txt"
    out = tmp_path / "out"

    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    rc_cfg = ocplants.write_config_file(str(conf), "rec.mol2", "lig.mol2", str(out), 1.0, 2.0, 3.0, 8.0)
    assert rc_cfg == ocerror.Error.ok()

    real_open = builtins.open

    def broken_open_cfg(file, mode="r", *args, **kwargs):
        if str(file) == str(conf) and "w" in mode:
            raise OSError("deny")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", broken_open_cfg)
    rc_cfg_write = ocplants.write_config_file(str(conf), "rec.mol2", "lig.mol2", str(out), 1.0, 2.0, 3.0, 8.0)
    assert rc_cfg_write == ocerror.Error.write_file()

    pose_list = tmp_path / "pose_list.txt"
    pose_out = ocplants.write_pose_list("single_pose.mol2", str(pose_list), overwrite=True)
    assert pose_out == str(pose_list)
    assert pose_list.read_text(encoding="utf-8") == "single_pose.mol2"

    resc_conf = tmp_path / "rescoring_conf.txt"

    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))
    rc_rescore_cfg = ocplants.write_rescoring_config_file(
        str(resc_conf),
        "rec.mol2",
        "pose_list.txt",
        str(tmp_path / "out_rescore"),
        1.0,
        2.0,
        3.0,
        7.0,
    )
    assert rc_rescore_cfg == ocerror.Error.ok()

    def broken_open_rescore(file, mode="r", *args, **kwargs):
        if str(file) == str(resc_conf) and "w" in mode:
            raise OSError("deny")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", broken_open_rescore)
    rc_rescore_cfg_write = ocplants.write_rescoring_config_file(
        str(resc_conf),
        "rec.mol2",
        "pose_list.txt",
        str(tmp_path / "out_rescore"),
        1.0,
        2.0,
        3.0,
        7.0,
    )
    assert rc_rescore_cfg_write == ocerror.Error.write_file()

    conf_run = tmp_path / "run_plants.conf"
    conf_run.write_text("dummy", encoding="utf-8")

    output_plants = tmp_path / "global_out"
    output_plants.mkdir(exist_ok=True)

    removed = {"rmtree": 0, "rmdir": 0}
    monkeypatch.setattr(ocplants.shutil, "rmtree", lambda *_a, **_k: removed.__setitem__("rmtree", removed["rmtree"] + 1))
    monkeypatch.setattr(ocplants.os.path, "isdir", lambda p: p == str(output_plants))
    monkeypatch.setattr(ocplants.os.path, "isabs", lambda _p: True)
    monkeypatch.setattr(ocplants.os.path, "isfile", lambda p: p == str(conf_run))
    monkeypatch.setattr(ocplants.shutil, "which", lambda _p: None)
    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("deny")))

    log_file = tmp_path / "plants_stub.log"
    rc_stub = ocplants.run_plants(str(conf_run), str(output_plants), overwrite=True, logFile=str(log_file))
    assert rc_stub == ocerror.Error.ok()
    assert removed["rmtree"] >= 1

    cfg2 = _cfg(tmp_path)
    cfg2.plants.executable = "plants"
    monkeypatch.setattr(ocplants, "get_config", lambda: cfg2)
    monkeypatch.setattr(ocplants.shutil, "which", lambda _p: "/usr/bin/plants")
    monkeypatch.setattr(ocplants.os, "listdir", lambda _p: [])
    monkeypatch.setattr(ocplants.os, "rmdir", lambda _p: removed.__setitem__("rmdir", removed["rmdir"] + 1))
    monkeypatch.setattr(ocplants.ocrun, "run", lambda *_a, **_k: 0)
    monkeypatch.setattr(ocplants.os, "makedirs", lambda *_a, **_k: None)

    rc_real = ocplants.run_plants(str(conf_run), str(output_plants), overwrite=False, logFile="")
    assert rc_real == 0
    assert removed["rmdir"] >= 1
