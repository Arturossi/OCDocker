#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for the VS/pipeline design assistant.
"""

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench.VSDesign import discover_vs_design_candidates
from OCDocker.Workbench.VSDesign import plan_vs_design
from OCDocker.Workbench.VSDesign import preview_vs_design

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


def _write_sample_workspace(root) -> dict:
    '''Write a synthetic receptor/ligand/box workspace for tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.

    Returns
    -------
    dict
        Absolute paths for the written receptor, ligand, and box files.
    '''

    receptor = root / "receptor.pdb"
    receptor.write_text("ATOM", encoding="utf-8")

    ligand_dir = root / "compounds" / "ligands" / "ligand"
    ligand_dir.mkdir(parents=True)
    ligand = ligand_dir / "ligand.smi"
    ligand.write_text("CCO", encoding="utf-8")

    box_dir = root / "boxes"
    box_dir.mkdir()
    box = box_dir / "box.pdb"
    box.write_text("REMARK", encoding="utf-8")
    (box_dir / "box1.pdb").write_text("REMARK", encoding="utf-8")

    return {"receptor": str(receptor), "ligand": str(ligand), "box": str(box)}


## Public ##


def test_discover_finds_receptor_ligand_and_box_candidates(tmp_path) -> None:
    '''Discovery finds candidates by filename/extension heuristics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_sample_workspace(tmp_path)
    context = discover_vs_design_candidates(tmp_path)

    assert len(context["candidates"]["receptors"]) == 1
    assert len(context["candidates"]["ligands"]) == 1
    assert len(context["candidates"]["boxes"]) == 2
    assert context["issues"] == []


def test_discover_empty_workspace_reports_issue_without_raising(tmp_path) -> None:
    '''An empty workspace returns empty candidates with a clear issue, not an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    context = discover_vs_design_candidates(tmp_path)

    assert context["candidates"] == {"receptors": [], "ligands": [], "boxes": []}
    assert context["issues"]


def test_discover_missing_scan_root_reports_issue_without_raising(tmp_path) -> None:
    '''A non-existent scan root reports an issue instead of raising.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    context = discover_vs_design_candidates(tmp_path, input_dir=tmp_path / "does-not-exist")

    assert context["candidates"] == {"receptors": [], "ligands": [], "boxes": []}
    assert context["issues"]


def test_preview_vs_design_valid_draft(tmp_path) -> None:
    '''A valid vs draft previews as valid with resolved fields.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "vs", "engine": "smina", **paths}

    preview = preview_vs_design(tmp_path, draft)

    assert preview["valid"] is True
    assert preview["errors"] == []
    assert preview["resolved"]["engine"] == "smina"


def test_preview_vs_design_rejects_unknown_engine(tmp_path) -> None:
    '''An unknown docking engine name is rejected with a clear error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "vs", "engine": "bogus", **paths}

    preview = preview_vs_design(tmp_path, draft)

    assert preview["valid"] is False
    assert any("bogus" in error for error in preview["errors"])


def test_preview_vs_design_rejects_missing_receptor(tmp_path) -> None:
    '''A missing receptor path is rejected with a clear error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "vs", "receptor": str(tmp_path / "does-not-exist.pdb"), "ligand": paths["ligand"], "box": paths["box"]}

    preview = preview_vs_design(tmp_path, draft)

    assert preview["valid"] is False
    assert any("receptor" in error for error in preview["errors"])


def test_preview_vs_design_resolves_relative_paths_against_root(tmp_path) -> None:
    '''Relative receptor/ligand/box paths resolve against the served root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_sample_workspace(tmp_path)
    draft = {
        "kind": "vs",
        "receptor": "receptor.pdb",
        "ligand": "compounds/ligands/ligand/ligand.smi",
        "box": "boxes/box.pdb",
    }

    preview = preview_vs_design(tmp_path, draft)

    assert preview["valid"] is True, preview["errors"]
    assert preview["resolved"]["receptor"] == str(tmp_path / "receptor.pdb")


def test_preview_pipeline_design_rejects_cluster_min_gte_max(tmp_path) -> None:
    '''cluster_min must be strictly less than cluster_max for pipeline drafts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "pipeline", "cluster_min": 20.0, "cluster_max": 10.0, **paths}

    preview = preview_vs_design(tmp_path, draft)

    assert preview["valid"] is False
    assert any("cluster_min" in error for error in preview["errors"])


def test_plan_vs_design_matches_real_cli_flags(tmp_path) -> None:
    '''plan_vs_design produces the exact argv a human would type for `ocdocker vs`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "vs", "engine": "smina", "skip_rescore": True, **paths}

    plan = plan_vs_design(tmp_path, draft)

    assert plan["kind"] == "vs"
    assert plan["args"] == [
        "--receptor", paths["receptor"],
        "--ligand", paths["ligand"],
        "--box", paths["box"],
        "--engine", "smina",
        "--skip-rescore",
    ]


def test_plan_pipeline_design_matches_real_cli_flags(tmp_path) -> None:
    '''plan_vs_design produces the exact argv a human would type for `ocdocker pipeline`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {
        "kind": "pipeline",
        "engines": ["vina", "smina"],
        "rescoring_engines": ["oddt"],
        "all_boxes": True,
        "name": "myrun",
        **paths,
    }

    plan = plan_vs_design(tmp_path, draft)

    assert plan["kind"] == "pipeline"
    assert "--engines" in plan["args"]
    assert plan["args"][plan["args"].index("--engines") + 1] == "vina,smina"
    assert "--rescoring-engines" in plan["args"]
    assert plan["args"][plan["args"].index("--rescoring-engines") + 1] == "oddt"
    assert "--all-boxes" in plan["args"]
    assert "--name" in plan["args"]
    assert plan["args"][plan["args"].index("--name") + 1] == "myrun"


def test_plan_vs_design_raises_on_invalid_draft(tmp_path) -> None:
    '''plan_vs_design refuses to build a command from an invalid draft.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    paths = _write_sample_workspace(tmp_path)
    draft = {"kind": "vs", "engine": "bogus", **paths}

    with pytest.raises(ValueError, match="Cannot plan an invalid VS design"):
        plan_vs_design(tmp_path, draft)
