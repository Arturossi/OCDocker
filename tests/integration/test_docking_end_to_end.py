#!/usr/bin/env python3

# Description
###############################################################################
'''
End-to-end tests running the real ``ocdocker vs`` command with each docking
engine on the bundled test_ptn1 complex (needs the engine, obabel and dssp;
engines not found are skipped). Executables are looked up on PATH, then in the
project's bin/docking folder, then in its OCDocker.cfg (or the file in
OCDOCKER_CONFIG).
'''

# Imports
###############################################################################
import math
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

import OCDocker.Docking.BaseVinaLike as ocvinalike
import OCDocker.Docking.PLANTS as ocplants

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##

# Config key: executable names to look for on PATH
EXECUTABLES = {
    "vina": ("vina",),
    "vina_split": ("vina_split",),
    "smina": ("smina",),
    "gnina": ("gnina",),
    "plants": ("plants", "PLANTS", "PLANTS1.2_64bit"),
    "spores": ("spores", "SPORES", "SPORES_64bit"),
    "obabel": ("obabel",),
    "dssp": ("mkdssp", "dssp"),
}


def _find_executable(config_key: str) -> Optional[str]:
    '''Find an executable on PATH, in the project's bin/docking folder or in the user's OCDocker config.

    Parameters
    ----------
    config_key : str
        Config key of the executable (e.g. ``gnina``).

    Returns
    -------
    str | None
        Path of the executable, or None if it was not found.
    '''

    for name in EXECUTABLES[config_key]:
        found = shutil.which(name)
        if found:
            return found

    project_root = Path(__file__).resolve().parents[2]
    for name in EXECUTABLES[config_key]:
        for bundled in sorted((project_root / "bin" / "docking").glob(f"*/{name}")):
            if os.access(bundled, os.X_OK):
                return str(bundled)

    config_file = Path(os.getenv("OCDOCKER_CONFIG", "") or project_root / "OCDocker.cfg")
    if config_file.is_file():
        match = re.search(rf"^{config_key} = (.+)$", config_file.read_text(), flags=re.MULTILINE)
        if match and os.access(match.group(1).strip(), os.X_OK):
            return match.group(1).strip()

    return None


def _write_config(config_path: Path, data_dir: Path, executables: Dict[str, str]) -> None:
    '''Write a config file based on OCDocker.cfg.example using the given executables.

    Parameters
    ----------
    config_path : Path
        Path of the config file to write.
    data_dir : Path
        Directory for the data paths required by the config.
    executables : Dict[str, str]
        Config key and path of each executable found.
    '''

    project_root = Path(__file__).resolve().parents[2]
    overrides = {
        "DB_BACKEND": "sqlite",
        "SQLITE_PATH": str(data_dir / "ocdocker.sqlite"),
        "ocdb": str(data_dir / "ocdb"),
        "pca": str(data_dir / "pca"),
        **executables,
    }

    for directory in (data_dir, data_dir / "ocdb", data_dir / "pca"):
        directory.mkdir(parents=True, exist_ok=True)

    text = (project_root / "OCDocker.cfg.example").read_text()
    for key, value in overrides.items():
        text, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, count=1, flags=re.MULTILINE)
        assert count == 1, f"'{key}' not found in OCDocker.cfg.example"
    config_path.write_text(text)


def _run_vs(tmp_path: Path, engine: str, required: Tuple[str, ...]) -> Tuple[Path, str]:
    '''Run ``ocdocker vs --engine <engine>`` on a copy of test_ptn1, skipping if a tool is missing.

    Parameters
    ----------
    tmp_path : Path
        Temporary working directory.
    engine : str
        Docking engine passed to ``--engine``.
    required : Tuple[str, ...]
        Config keys of the executables the engine needs.

    Returns
    -------
    Tuple[Path, str]
        The ligand directory of the copy and the command output.
    '''

    executables = {key: _find_executable(key) for key in (*required, "obabel", "dssp")}
    missing = [key for key, path in executables.items() if path is None]
    if missing:
        pytest.skip(f"Missing external tools: {', '.join(missing)}")

    # The vs command writes its outputs next to the inputs, so work on a copy
    project_root = Path(__file__).resolve().parents[2]
    shutil.copytree(project_root / "test_files" / "test_ptn1", tmp_path / "test_ptn1")
    receptor = tmp_path / "test_ptn1" / "receptor.pdb"
    ligand_dir = tmp_path / "test_ptn1" / "compounds" / "ligands" / "ligand"

    config_path = tmp_path / "OCDocker.cfg"
    _write_config(config_path, tmp_path / "data", {key: str(path) for key, path in executables.items()})

    env = dict(os.environ)
    env["OCDOCKER_CONFIG"] = str(config_path)
    env["OCDOCKER_TIMEOUT"] = "600"

    result = subprocess.run(
        [
            sys.executable, "-c", "import sys; from OCDocker.CLI import main; sys.exit(main())",
            "vs",
            "--engine", engine,
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

    return ligand_dir, output


## Public ##

# Engine: (executables needed, output directory, log reader, docked poses getter)
VINA_LIKE_ENGINES = {
    "vina": (("vina", "vina_split"), "vinaFiles", ocvinalike.read_vina_log, ocvinalike.get_vina_docked_poses),
    "smina": (("smina", "vina_split"), "sminaFiles", ocvinalike.read_smina_log, ocvinalike.get_smina_docked_poses),
    "gnina": (("gnina", "vina_split"), "gninaFiles", ocvinalike.read_gnina_log, ocvinalike.get_gnina_docked_poses),
}


@pytest.mark.external
@pytest.mark.slow
@pytest.mark.parametrize("engine", sorted(VINA_LIKE_ENGINES))
def test_vs_docks_test_ptn1(tmp_path, engine):
    '''Dock test_ptn1 with ``ocdocker vs --engine <engine>`` and check the poses and scores.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary working directory.
    engine : str
        Docking engine passed to ``--engine``.
    '''

    required, files_dir_name, read_log, get_docked_poses = VINA_LIKE_ENGINES[engine]
    ligand_dir, output = _run_vs(tmp_path, engine, required)

    files_dir = ligand_dir / files_dir_name
    assert (files_dir / "e2e.pdbqt").is_file(), output

    # Each pose maps {scoring function: affinity in kcal/mol}
    poses = read_log(str(files_dir / "e2e.log"))
    assert poses, output
    affinities = [affinity for pose in poses.values() for affinity in pose.values()]
    # The scale depends on the scoring function (e.g. ad4_scoring, the gnina default in
    # OCDocker.cfg.example, gives about -40 where vina gives -7.5 for test_ptn1), so only check
    # that every score was parsed and the best pose binds
    assert all(math.isfinite(affinity) for affinity in affinities), affinities
    assert min(affinities) < 0.0, affinities

    assert len(get_docked_poses(str(files_dir))) == len(poses)


@pytest.mark.external
@pytest.mark.slow
def test_vs_plants_docks_test_ptn1(tmp_path):
    '''Dock test_ptn1 with ``ocdocker vs --engine plants`` and check the poses and scores.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary working directory.
    '''

    ligand_dir, output = _run_vs(tmp_path, "plants", ("plants", "spores"))

    # PLANTS writes its ranking and the poses (mol2) in plantsFiles/run
    run_dir = ligand_dir / "plantsFiles" / "run"
    poses = ocplants.read_log(str(run_dir / "ranking.csv"))
    assert poses, output

    # ChemPLP scores are unitless and negative for a binding pose
    total_scores = [pose["PLANTS_TOTAL_SCORE"] for pose in poses.values()]
    assert min(total_scores) < 0.0, total_scores

    assert len(ocplants.get_docked_poses(str(run_dir))) == len(poses)
