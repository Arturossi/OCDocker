#!/usr/bin/env python3

# Description
###############################################################################
'''
End-to-end test running the real ``ocdocker vs`` command with AutoDock Vina on
the bundled test_ptn1 complex (needs vina, obabel and dssp on PATH).
'''

# Imports
###############################################################################
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path

import pytest

import OCDocker.Docking.BaseVinaLike as ocvinalike

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##


def _write_config(config_path: Path, data_dir: Path) -> None:
    '''Write a config file based on OCDocker.cfg.example using the tools found on PATH.

    Parameters
    ----------
    config_path : Path
        Path of the config file to write.
    data_dir : Path
        Directory for the data paths required by the config.
    '''

    project_root = Path(__file__).resolve().parents[2]
    overrides = {
        "DB_BACKEND": "sqlite",
        "SQLITE_PATH": str(data_dir / "ocdocker.sqlite"),
        "ocdb": str(data_dir / "ocdb"),
        "pca": str(data_dir / "pca"),
        "vina": shutil.which("vina") or "vina",
        "vina_split": shutil.which("vina_split") or "vina_split",
        "obabel": shutil.which("obabel") or "obabel",
        "dssp": shutil.which("mkdssp") or shutil.which("dssp") or "dssp",
    }

    for directory in (data_dir, data_dir / "ocdb", data_dir / "pca"):
        directory.mkdir(parents=True, exist_ok=True)

    text = (project_root / "OCDocker.cfg.example").read_text()
    for key, value in overrides.items():
        text, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, count=1, flags=re.MULTILINE)
        assert count == 1, f"'{key}' not found in OCDocker.cfg.example"
    config_path.write_text(text)


## Public ##


@pytest.mark.external
@pytest.mark.slow
def test_vs_vina_docks_test_ptn1(tmp_path):
    '''Dock test_ptn1 with ``ocdocker vs --engine vina`` and check the poses and scores.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary working directory.
    '''

    missing = [exe for exe in ("vina", "obabel") if shutil.which(exe) is None]
    if shutil.which("mkdssp") is None and shutil.which("dssp") is None:
        missing.append("dssp")
    if missing:
        pytest.skip(f"Missing external tools: {', '.join(missing)}")

    # The vs command writes its outputs next to the inputs, so work on a copy
    project_root = Path(__file__).resolve().parents[2]
    shutil.copytree(project_root / "test_files" / "test_ptn1", tmp_path / "test_ptn1")
    receptor = tmp_path / "test_ptn1" / "receptor.pdb"
    ligand_dir = tmp_path / "test_ptn1" / "compounds" / "ligands" / "ligand"

    config_path = tmp_path / "OCDocker.cfg"
    _write_config(config_path, tmp_path / "data")

    env = dict(os.environ)
    env["OCDOCKER_CONFIG"] = str(config_path)
    env["OCDOCKER_TIMEOUT"] = "600"

    result = subprocess.run(
        [
            sys.executable, "-c", "import sys; from OCDocker.CLI import main; sys.exit(main())",
            "vs",
            "--engine", "vina",
            "--receptor", str(receptor),
            "--ligand", str(ligand_dir / "ligand.smi"),
            "--box", str(ligand_dir / "boxes" / "box0.pdb"),
            "--name", "e2e",
            "--skip-rescore",
        ],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, output

    vina_dir = ligand_dir / "vinaFiles"
    assert (vina_dir / "e2e.pdbqt").is_file(), output

    # Each pose maps {scoring function: affinity in kcal/mol}
    poses = ocvinalike.read_vina_log(str(vina_dir / "e2e.log"))
    assert poses, output
    affinities = [affinity for pose in poses.values() for affinity in pose.values()]
    # Vina's search is stochastic, so lower-ranked poses can score positive; the best pose must bind
    assert all(-30.0 < affinity < 30.0 for affinity in affinities), affinities
    assert min(affinities) < 0.0, affinities

    assert len(ocvinalike.get_vina_docked_poses(str(vina_dir))) == len(poses)
