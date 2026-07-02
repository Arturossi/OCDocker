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

from OCDocker.Workbench.VSDesign import discover_vs_campaign_candidates
from OCDocker.Workbench.VSDesign import discover_vs_design_candidates
from OCDocker.Workbench.VSDesign import plan_vs_campaign
from OCDocker.Workbench.VSDesign import plan_vs_design
from OCDocker.Workbench.VSDesign import preview_vs_campaign
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


def _write_campaign_samples(root, names) -> Path:
    '''Write an ``input/{sample}/{receptor,ligand,box}`` layout for campaign tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary root.
    names : Sequence[str]
        Sample directory names to create, each with a complete receptor/ligand/box set.

    Returns
    -------
    pathlib.Path
        The created ``input/`` directory.
    '''

    input_dir = root / "input"
    for name in names:
        sample_dir = input_dir / name
        sample_dir.mkdir(parents=True)
        (sample_dir / "receptor.pdbqt").write_text("ATOM", encoding="utf-8")
        (sample_dir / "ligand.pdbqt").write_text("MOL", encoding="utf-8")
        (sample_dir / "box.txt").write_text("REMARK", encoding="utf-8")
    return input_dir


def test_discover_vs_campaign_candidates_finds_complete_samples(tmp_path) -> None:
    '''Discovery builds one manifest row per complete sample directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001", "sample_002"])

    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)

    assert len(context["manifest"]) == 2
    assert {row["sample"] for row in context["manifest"]} == {"sample_001", "sample_002"}
    assert context["issues"] == []


def test_discover_vs_campaign_candidates_auto_detects_input_dir_by_default(tmp_path) -> None:
    '''Without an explicit input_dir, discovery uses root/input, not root itself.

    Regression test: root's only immediate child directory is "input" (not a
    sample directory), so a naive scan of root's immediate children would
    treat "input" itself as one bogus sample and pick one receptor/ligand/box
    from across the *different* real samples nested inside it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_campaign_samples(tmp_path, ["sample_001", "sample_002"])

    context = discover_vs_campaign_candidates(tmp_path)

    assert context["scan_root"] == str(tmp_path / "input")
    assert len(context["manifest"]) == 2
    assert {row["sample"] for row in context["manifest"]} == {"sample_001", "sample_002"}


def test_discover_vs_campaign_candidates_falls_back_to_root_without_input_dir(tmp_path) -> None:
    '''When root/input does not exist, discovery scans root's immediate children directly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    for name in ("sample_001",):
        sample_dir = tmp_path / name
        sample_dir.mkdir()
        (sample_dir / "receptor.pdbqt").write_text("ATOM", encoding="utf-8")
        (sample_dir / "ligand.pdbqt").write_text("MOL", encoding="utf-8")
        (sample_dir / "box.txt").write_text("REMARK", encoding="utf-8")

    context = discover_vs_campaign_candidates(tmp_path)

    assert context["scan_root"] == str(tmp_path)
    assert len(context["manifest"]) == 1
    assert context["manifest"][0]["sample"] == "sample_001"


def test_discover_vs_campaign_candidates_skips_incomplete_samples(tmp_path) -> None:
    '''An incomplete sample directory is skipped with an explanatory issue.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001"])
    (input_dir / "sample_002").mkdir()
    (input_dir / "sample_002" / "receptor.pdbqt").write_text("ATOM", encoding="utf-8")

    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)

    assert len(context["manifest"]) == 1
    assert any("sample_002" in issue for issue in context["issues"])


def test_discover_vs_campaign_candidates_empty_root_reports_issue(tmp_path) -> None:
    '''An empty scan root returns an empty manifest with an issue, not an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    context = discover_vs_campaign_candidates(tmp_path)

    assert context["manifest"] == []
    assert context["issues"]


def test_preview_vs_campaign_valid_manifest(tmp_path) -> None:
    '''A valid manifest previews as valid with resolved rows.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001", "sample_002"])
    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)

    preview = preview_vs_campaign(tmp_path, {"manifest": context["manifest"]})

    assert preview["valid"] is True
    assert len(preview["resolved"]["rows"]) == 2


def test_preview_vs_campaign_rejects_empty_manifest(tmp_path) -> None:
    '''An empty manifest is rejected with a clear error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    preview = preview_vs_campaign(tmp_path, {"manifest": []})

    assert preview["valid"] is False
    assert any("at least one row" in error for error in preview["errors"])


def test_preview_vs_campaign_rejects_duplicate_sample_names(tmp_path) -> None:
    '''Two rows sharing a sample name are rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001"])
    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)
    manifest = context["manifest"] + [dict(context["manifest"][0])]

    preview = preview_vs_campaign(tmp_path, {"manifest": manifest})

    assert preview["valid"] is False
    assert any("Duplicate sample name" in error for error in preview["errors"])


def test_preview_vs_campaign_rejects_bad_engine_in_one_row(tmp_path) -> None:
    '''An unknown engine in one row is rejected with a row-prefixed error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001"])
    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)
    manifest = [dict(context["manifest"][0], engines=["bogus"])]

    preview = preview_vs_campaign(tmp_path, {"manifest": manifest})

    assert preview["valid"] is False
    assert any("Row 1 (sample_001)" in error and "bogus" in error for error in preview["errors"])


def test_plan_vs_campaign_builds_script_for_every_row(tmp_path) -> None:
    '''plan_vs_campaign builds a shell script covering every manifest row.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    input_dir = _write_campaign_samples(tmp_path, ["sample_001", "sample_002"])
    context = discover_vs_campaign_candidates(tmp_path, input_dir=input_dir)

    plan = plan_vs_campaign(tmp_path, {"manifest": context["manifest"], "store_db": True})

    assert plan["kind"] == "vs_campaign"
    assert len(plan["manifest"]) == 2
    assert "--store-db" in plan["args"]
    assert "sample_001" in plan["shell_command"]
    assert "sample_002" in plan["shell_command"]


def test_plan_vs_campaign_raises_on_invalid_manifest(tmp_path) -> None:
    '''plan_vs_campaign refuses to build a script from an invalid manifest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="Cannot plan an invalid VS campaign"):
        plan_vs_campaign(tmp_path, {"manifest": []})
