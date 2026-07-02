#!/usr/bin/env python3

# Description
###############################################################################
'''
Design assistant for virtual-screening runs (``ocdocker vs`` and
``ocdocker pipeline``): discover receptor/ligand/box candidates, validate a
draft selection, and plan the exact command to run — for one target
(``*_vs_design*``) or a multi-sample batch (``*_vs_campaign*``). Mirrors
:mod:`OCDocker.Workbench.AblationDesign`, but for docking runs instead of
OCScore training. Read-only: nothing here writes files or executes commands —
the plan output is handed to :meth:`OCDocker.Workbench.Jobs.JobManager.launch`
(via ``run_job``/``plan_job``) to actually run.
'''

# Imports
###############################################################################
from __future__ import annotations

import shlex

from pathlib import Path
from typing import Any

from OCDocker.Workbench.Jobs import build_campaign_script
from OCDocker.Workbench.Models import VALID_DOCKING_ENGINES
from OCDocker.Workbench.Models import VALID_RESCORING_ENGINES

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

VS_DESIGN_KINDS = ("vs", "pipeline")
DEFAULT_VS_DESIGN_SCAN_DEPTH = 6
DEFAULT_VS_CAMPAIGN_SAMPLE_SCAN_DEPTH = 3
DEFAULT_PIPELINE_ENGINES = ("vina", "smina", "plants")
DEFAULT_VS_ENGINE = "vina"

_RECEPTOR_EXTENSIONS = frozenset({".pdb", ".pdbqt"})
_LIGAND_EXTENSIONS = frozenset({".smi", ".sdf", ".mol2", ".pdbqt"})
_BOX_EXTENSIONS = frozenset({".pdb", ".txt"})

# Functions
###############################################################################
## Private ##


def _is_hidden(path: Path) -> bool:
    '''Return whether a path should be skipped while scanning for candidates.

    Parameters
    ----------
    path : pathlib.Path
        Path to inspect.

    Returns
    -------
    bool
        True when the path is a dotfile/dotdir or a ``__pycache__`` directory.
    '''

    return path.name.startswith(".") or path.name == "__pycache__"


def _iter_candidate_files(root: Path, max_depth: int):
    '''Yield files below a root without descending beyond a depth limit.

    Parameters
    ----------
    root : pathlib.Path
        Directory to scan.
    max_depth : int
        Maximum directory depth below root to descend.

    Yields
    ------
    pathlib.Path
        Discovered file paths, hidden directories skipped.
    '''

    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda child: child.name)
        except OSError:
            continue
        for child in children:
            if _is_hidden(child):
                continue
            if child.is_file():
                yield child
            elif depth < max_depth and child.is_dir() and not child.is_symlink():
                stack.append((child, depth + 1))


def _looks_like_box(path: Path) -> bool:
    '''Return whether a path plausibly names a binding-site box file.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file path.

    Returns
    -------
    bool
        True when the extension and filename match the ``box*.pdb``/``box*.txt`` convention.
    '''

    return path.suffix.lower() in _BOX_EXTENSIONS and path.stem.lower().startswith("box")


def _looks_like_receptor(path: Path) -> bool:
    '''Return whether a path plausibly names a receptor structure file.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file path.

    Returns
    -------
    bool
        True when the extension matches and the filename mentions "receptor".
    '''

    return path.suffix.lower() in _RECEPTOR_EXTENSIONS and "receptor" in path.stem.lower()


def _looks_like_ligand(path: Path) -> bool:
    '''Return whether a path plausibly names a ligand file.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file path.

    Returns
    -------
    bool
        True when the extension matches and the filename or parent directory mentions "ligand".
    '''

    if path.suffix.lower() not in _LIGAND_EXTENSIONS:
        return False
    return "ligand" in path.stem.lower() or "ligand" in path.parent.name.lower()


def _candidate_entry(path: Path, scan_root: Path) -> dict[str, str]:
    '''Build one discovered-candidate payload entry.

    Parameters
    ----------
    path : pathlib.Path
        Discovered file path.
    scan_root : pathlib.Path
        Root the scan started from, used to build a relative display name.

    Returns
    -------
    dict[str, str]
        JSON-safe candidate entry (``path``, ``name``).
    '''

    try:
        name = str(path.relative_to(scan_root))
    except ValueError:
        name = path.name
    return {"path": str(path), "name": name}


def _resolve_path(root: Path, value: str) -> Path:
    '''Resolve a receptor/ligand/box path, relative to root when not absolute.

    Parameters
    ----------
    root : pathlib.Path
        Served Workbench root.
    value : str
        Path string from a design request body.

    Returns
    -------
    pathlib.Path
        Resolved path (not required to exist).
    '''

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _required_path(root: Path, body: dict[str, Any], key: str, errors: list[str], warnings: list[str], extensions: frozenset[str]) -> Path | None:
    '''Validate one required receptor/ligand/box path field.

    Parameters
    ----------
    root : pathlib.Path
        Served Workbench root, used to resolve relative paths.
    body : dict[str, Any]
        Draft design request body.
    key : str
        Field name (``"receptor"``, ``"ligand"``, or ``"box"``).
    errors : list[str]
        Accumulator for validation errors.
    warnings : list[str]
        Accumulator for validation warnings.
    extensions : frozenset[str]
        Expected file extensions for this field.

    Returns
    -------
    pathlib.Path or None
        Resolved path when present, else None (an error is already recorded).
    '''

    raw = str(body.get(key) or "").strip()
    if not raw:
        errors.append(f"{key} is required.")
        return None
    resolved = _resolve_path(root, raw)
    if not resolved.is_file():
        errors.append(f"{key} path does not exist or is not a file: {resolved}")
        return None
    if resolved.suffix.lower() not in extensions:
        warnings.append(f"{key} has an unusual extension for this field: {resolved.suffix or '(none)'}")
    return resolved


def _resolve_docking_engines(raw: Any, errors: list[str]) -> tuple[str, ...]:
    '''Validate and normalize a list of docking engine names.

    Parameters
    ----------
    raw : Any
        Engine names from a draft request body (a list, or a single string).
    errors : list[str]
        Accumulator for validation errors.

    Returns
    -------
    tuple[str, ...]
        Validated, order-preserved engine names.
    '''

    if raw is None:
        return ()
    values = [raw] if isinstance(raw, str) else list(raw)
    engines = tuple(str(item).strip().lower() for item in values if str(item).strip())
    for engine in engines:
        if engine not in VALID_DOCKING_ENGINES:
            errors.append(f"Unknown docking engine {engine!r}. Expected one of: {sorted(VALID_DOCKING_ENGINES)}.")
    return engines


def _resolve_rescoring_engines(raw: Any, errors: list[str]) -> tuple[str, ...]:
    '''Validate and normalize a list of rescoring engine names.

    Parameters
    ----------
    raw : Any
        Rescoring engine names from a draft request body.
    errors : list[str]
        Accumulator for validation errors.

    Returns
    -------
    tuple[str, ...]
        Validated, order-preserved engine names.
    '''

    if raw is None:
        return ()
    values = [raw] if isinstance(raw, str) else list(raw)
    engines = tuple(str(item).strip().lower() for item in values if str(item).strip())
    for engine in engines:
        if engine not in VALID_RESCORING_ENGINES:
            errors.append(f"Unknown rescoring engine {engine!r}. Expected one of: {sorted(VALID_RESCORING_ENGINES)}.")
    return engines


def _validate_campaign_row(root_path: Path, row: dict[str, Any], index: int) -> tuple[list[str], list[str], dict[str, Any]]:
    '''Validate one manifest row of a VS campaign draft.

    Parameters
    ----------
    root_path : pathlib.Path
        Served Workbench root, used to resolve relative paths.
    row : dict[str, Any]
        Draft row: ``sample``, ``row_kind`` (``"vs"`` or ``"pipeline"``),
        ``receptor``, ``ligand``, ``box``, ``engines``, optional
        ``rescoring_engines``.
    index : int
        1-based row position, used to prefix error/warning messages.

    Returns
    -------
    tuple[list[str], list[str], dict[str, Any]]
        Row-prefixed errors, row-prefixed warnings, and the resolved row.
    '''

    row_errors: list[str] = []
    row_warnings: list[str] = []
    sample = str(row.get("sample") or f"row-{index}")
    row_kind = str(row.get("row_kind") or "vs")
    if row_kind not in VS_DESIGN_KINDS:
        row_errors.append(f"row_kind must be one of {VS_DESIGN_KINDS}, got {row_kind!r}.")

    receptor = _required_path(root_path, row, "receptor", row_errors, row_warnings, _RECEPTOR_EXTENSIONS)
    ligand = _required_path(root_path, row, "ligand", row_errors, row_warnings, _LIGAND_EXTENSIONS)
    box = _required_path(root_path, row, "box", row_errors, row_warnings, _BOX_EXTENSIONS)

    default_engines = DEFAULT_PIPELINE_ENGINES if row_kind == "pipeline" else (DEFAULT_VS_ENGINE,)
    engines = _resolve_docking_engines(row.get("engines"), row_errors) or default_engines
    if row_kind == "vs" and len(engines) > 1:
        row_warnings.append("vs rows only use the first engine listed; the rest are ignored.")
    rescoring_engines = _resolve_rescoring_engines(row.get("rescoring_engines"), row_errors)

    resolved_row = {
        "sample": sample,
        "row_kind": row_kind,
        "receptor": str(receptor) if receptor else None,
        "ligand": str(ligand) if ligand else None,
        "box": str(box) if box else None,
        "engines": list(engines),
        "rescoring_engines": list(rescoring_engines) if rescoring_engines else None,
    }
    prefixed_errors = [f"Row {index} ({sample}): {message}" for message in row_errors]
    prefixed_warnings = [f"Row {index} ({sample}): {message}" for message in row_warnings]
    return prefixed_errors, prefixed_warnings, resolved_row


## Public ##


def discover_vs_design_candidates(
    root: str | Path,
    *,
    input_dir: str | Path | None = None,
    max_depth: int = DEFAULT_VS_DESIGN_SCAN_DEPTH,
) -> dict[str, Any]:
    '''Discover receptor/ligand/box candidates under a workspace.

    Best-effort, depth-limited scan by filename/extension heuristics — there
    is no single fixed input layout in OCDocker, unlike OCScore's
    ``raw_prepare/``. Results are candidates for the caller to choose from,
    never a hard requirement; an empty or ambiguous scan is not an error.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root.
    input_dir : str, pathlib.Path, or None
        Optional subdirectory to scan instead of the whole served root.
    max_depth : int
        Maximum directory depth below the scan root to descend.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload: ``candidates`` (``receptors``, ``ligands``,
        ``boxes``), ``scan_root``, ``issues``.
    '''

    scan_root = Path(input_dir).expanduser().resolve() if input_dir else Path(root).expanduser().resolve()
    issues: list[str] = []
    receptors: list[dict[str, str]] = []
    ligands: list[dict[str, str]] = []
    boxes: list[dict[str, str]] = []

    if not scan_root.is_dir():
        issues.append(f"Scan root does not exist or is not a directory: {scan_root}")
        return {
            "candidates": {"receptors": [], "ligands": [], "boxes": []},
            "scan_root": str(scan_root),
            "issues": issues,
        }

    for path in _iter_candidate_files(scan_root, max_depth):
        if _looks_like_box(path):
            boxes.append(_candidate_entry(path, scan_root))
        elif _looks_like_receptor(path):
            receptors.append(_candidate_entry(path, scan_root))
        elif _looks_like_ligand(path):
            ligands.append(_candidate_entry(path, scan_root))

    if not receptors and not ligands and not boxes:
        issues.append("No receptor, ligand, or box candidates found under the scanned root.")

    return {
        "candidates": {"receptors": receptors, "ligands": ligands, "boxes": boxes},
        "scan_root": str(scan_root),
        "issues": issues,
    }


def preview_vs_design(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Validate one draft VS design without running anything.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root, used to resolve relative paths.
    body : dict[str, Any]
        Draft design: ``kind`` (``"vs"`` or ``"pipeline"``), ``receptor``,
        ``ligand``, ``box`` (paths), plus kind-specific fields — ``engine``
        (``vs``) or ``engines``/``rescoring_engines`` (``pipeline``).

    Returns
    -------
    dict[str, Any]
        ``{"valid", "errors", "warnings", "resolved"}``.
    '''

    root_path = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    kind = str(body.get("kind") or "").strip()
    if kind not in VS_DESIGN_KINDS:
        errors.append(f"kind must be one of {VS_DESIGN_KINDS}, got {kind!r}.")

    receptor = _required_path(root_path, body, "receptor", errors, warnings, _RECEPTOR_EXTENSIONS)
    ligand = _required_path(root_path, body, "ligand", errors, warnings, _LIGAND_EXTENSIONS)
    box = _required_path(root_path, body, "box", errors, warnings, _BOX_EXTENSIONS)

    resolved: dict[str, Any] = {
        "kind": kind or None,
        "receptor": str(receptor) if receptor else None,
        "ligand": str(ligand) if ligand else None,
        "box": str(box) if box else None,
    }

    if kind == "vs":
        engine = str(body.get("engine") or DEFAULT_VS_ENGINE).strip().lower()
        if engine not in VALID_DOCKING_ENGINES:
            errors.append(f"Unknown docking engine {engine!r}. Expected one of: {sorted(VALID_DOCKING_ENGINES)}.")
        resolved["engine"] = engine
    elif kind == "pipeline":
        engines = _resolve_docking_engines(body.get("engines"), errors) or DEFAULT_PIPELINE_ENGINES
        rescoring_engines = _resolve_rescoring_engines(body.get("rescoring_engines"), errors)
        resolved["engines"] = list(engines)
        resolved["rescoring_engines"] = list(rescoring_engines) if rescoring_engines else None
        cluster_min = body.get("cluster_min")
        cluster_max = body.get("cluster_max")
        if cluster_min is not None and cluster_max is not None:
            try:
                if float(cluster_min) >= float(cluster_max):
                    errors.append("cluster_min must be less than cluster_max.")
            except (TypeError, ValueError):
                errors.append("cluster_min and cluster_max must be numeric.")

    return {"valid": not errors, "errors": errors, "warnings": warnings, "resolved": resolved}


def plan_vs_design(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Build the exact ``ocdocker vs``/``pipeline`` argv for a valid draft.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root, used to resolve relative paths.
    body : dict[str, Any]
        Same draft shape as :func:`preview_vs_design`, plus optional
        ``all_boxes``, ``name``, ``outdir``, ``timeout``, ``store_db``, and
        (``vs`` only) ``skip_rescore``/``skip_split``, or (``pipeline`` only)
        ``cluster_min``/``cluster_max``/``cluster_step``/``strict_engines``.

    Returns
    -------
    dict[str, Any]
        ``{"kind", "args", "cwd", "shell_command"}`` — ``args`` and ``kind``
        are ready to pass directly to
        :meth:`OCDocker.Workbench.Jobs.JobManager.launch` (or the
        ``run_job``/``plan_job`` API and MCP tools).

    Raises
    ------
    ValueError
        If the draft is not valid (call :func:`preview_vs_design` first).
    '''

    preview = preview_vs_design(root, body)
    if not preview["valid"]:
        raise ValueError("Cannot plan an invalid VS design: " + "; ".join(preview["errors"]))

    resolved = preview["resolved"]
    kind = resolved["kind"]
    args: list[str] = ["--receptor", resolved["receptor"], "--ligand", resolved["ligand"], "--box", resolved["box"]]

    if kind == "vs":
        args.extend(["--engine", resolved["engine"]])
        if body.get("skip_rescore"):
            args.append("--skip-rescore")
        if body.get("skip_split"):
            args.append("--skip-split")
    else:
        args.extend(["--engines", ",".join(resolved["engines"])])
        if resolved["rescoring_engines"]:
            args.extend(["--rescoring-engines", ",".join(resolved["rescoring_engines"])])
        if body.get("cluster_min") is not None:
            args.extend(["--cluster-min", str(body["cluster_min"])])
        if body.get("cluster_max") is not None:
            args.extend(["--cluster-max", str(body["cluster_max"])])
        if body.get("cluster_step") is not None:
            args.extend(["--cluster-step", str(body["cluster_step"])])
        if body.get("strict_engines"):
            args.append("--strict-engines")

    if body.get("all_boxes"):
        args.append("--all-boxes")
    if body.get("name"):
        args.extend(["--name", str(body["name"])])
    if body.get("outdir"):
        args.extend(["--outdir", str(body["outdir"])])
    if body.get("timeout") is not None:
        args.extend(["--timeout", str(body["timeout"])])
    if body.get("store_db"):
        args.append("--store-db")

    cwd = str(body.get("cwd")) if body.get("cwd") else None
    shell_command = " ".join(shlex.quote(part) for part in ("ocdocker", kind, *args))
    return {"kind": kind, "args": args, "cwd": cwd, "shell_command": shell_command}


def discover_vs_campaign_candidates(
    root: str | Path,
    *,
    input_dir: str | Path | None = None,
    sample_scan_depth: int = DEFAULT_VS_CAMPAIGN_SAMPLE_SCAN_DEPTH,
) -> dict[str, Any]:
    '''Discover a draft multi-sample manifest from an ``input/{sample}/...`` layout.

    Matches the convention used by ``examples/19_Snakefile_ocdocker_pipeline.smk``
    and ``examples/20_Snakefile_ocdocker_granular_pipeline.smk``: one
    subdirectory per sample directly under the scan root, each containing a
    receptor/ligand/box file. Best-effort — a workspace not organized this
    way yields an empty manifest with an explanatory issue, not an error; the
    caller can still hand-author a manifest directly.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root.
    input_dir : str, pathlib.Path, or None
        Optional subdirectory to scan instead of the whole served root
        (typically an ``input/`` directory containing one folder per sample).
        When omitted, ``root/input`` is used automatically if present,
        otherwise ``root`` itself.
    sample_scan_depth : int
        Maximum directory depth below each sample directory to descend while
        looking for its receptor/ligand/box files.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload: ``manifest`` (list of draft rows), ``scan_root``,
        ``issues``.
    '''

    if input_dir:
        scan_root = Path(input_dir).expanduser().resolve()
    else:
        root_path = Path(root).expanduser().resolve()
        default_input_dir = root_path / "input"
        scan_root = default_input_dir if default_input_dir.is_dir() else root_path
    issues: list[str] = []
    if not scan_root.is_dir():
        issues.append(f"Scan root does not exist or is not a directory: {scan_root}")
        return {"manifest": [], "scan_root": str(scan_root), "issues": issues}

    manifest: list[dict[str, Any]] = []
    sample_dirs = sorted(
        (child for child in scan_root.iterdir() if child.is_dir() and not _is_hidden(child)),
        key=lambda child: child.name,
    )
    for sample_dir in sample_dirs:
        receptor = ligand = box = None
        for path in _iter_candidate_files(sample_dir, sample_scan_depth):
            if box is None and _looks_like_box(path):
                box = path
            elif receptor is None and _looks_like_receptor(path):
                receptor = path
            elif ligand is None and _looks_like_ligand(path):
                ligand = path
        if receptor and ligand and box:
            manifest.append({
                "sample": sample_dir.name,
                "row_kind": "vs",
                "receptor": str(receptor),
                "ligand": str(ligand),
                "box": str(box),
                "engines": [DEFAULT_VS_ENGINE],
            })
        else:
            missing = [name for name, value in (("receptor", receptor), ("ligand", ligand), ("box", box)) if value is None]
            issues.append(f"Sample {sample_dir.name!r} is missing {', '.join(missing)} — skipped.")

    if not manifest:
        issues.append("No complete sample directories (receptor + ligand + box) found under the scanned root.")

    return {"manifest": manifest, "scan_root": str(scan_root), "issues": issues}


def preview_vs_campaign(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Validate a draft multi-sample VS campaign manifest without running anything.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root, used to resolve relative paths.
    body : dict[str, Any]
        Draft campaign: ``manifest`` — a non-empty list of rows, each shaped
        like :func:`discover_vs_campaign_candidates`'s output rows
        (``sample``, ``row_kind``, ``receptor``, ``ligand``, ``box``,
        ``engines``, optional ``rescoring_engines``).

    Returns
    -------
    dict[str, Any]
        ``{"valid", "errors", "warnings", "resolved": {"rows": [...]}}``.
    '''

    root_path = Path(root).expanduser().resolve()
    manifest = body.get("manifest") or []
    errors: list[str] = []
    warnings: list[str] = []
    resolved_rows: list[dict[str, Any]] = []

    if not manifest:
        errors.append("manifest must contain at least one row.")

    seen_samples: set[str] = set()
    for index, row in enumerate(manifest, start=1):
        row_errors, row_warnings, resolved_row = _validate_campaign_row(root_path, row, index)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        resolved_rows.append(resolved_row)
        sample = resolved_row["sample"]
        if sample in seen_samples:
            errors.append(f"Duplicate sample name: {sample!r}.")
        seen_samples.add(sample)

    return {"valid": not errors, "errors": errors, "warnings": warnings, "resolved": {"rows": resolved_rows}}


def plan_vs_campaign(root: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    '''Build the ``vs_campaign`` job payload for a valid draft manifest.

    Parameters
    ----------
    root : str or pathlib.Path
        Served Workbench root, used to resolve relative paths.
    body : dict[str, Any]
        Same ``manifest`` shape as :func:`preview_vs_campaign`, plus optional
        common flags applied to every row: ``outdir``, ``timeout``, ``store_db``,
        and ``cwd`` (the campaign job's working directory).

    Returns
    -------
    dict[str, Any]
        ``{"kind": "vs_campaign", "manifest", "args", "cwd", "shell_command"}``
        — ready to pass directly to
        :meth:`OCDocker.Workbench.Jobs.JobManager.launch` (or the
        ``run_job``/``plan_job`` API and MCP tools).

    Raises
    ------
    ValueError
        If the draft is not valid (call :func:`preview_vs_campaign` first).
    '''

    preview = preview_vs_campaign(root, body)
    if not preview["valid"]:
        raise ValueError("Cannot plan an invalid VS campaign: " + "; ".join(preview["errors"]))

    rows = preview["resolved"]["rows"]
    args: list[str] = []
    if body.get("outdir"):
        args.extend(["--outdir", str(body["outdir"])])
    if body.get("timeout") is not None:
        args.extend(["--timeout", str(body["timeout"])])
    if body.get("store_db"):
        args.append("--store-db")

    cwd = str(body.get("cwd")) if body.get("cwd") else None
    shell_command = build_campaign_script(rows, args)
    return {"kind": "vs_campaign", "manifest": rows, "args": args, "cwd": cwd, "shell_command": shell_command}


__all__ = [
    "DEFAULT_PIPELINE_ENGINES",
    "DEFAULT_VS_CAMPAIGN_SAMPLE_SCAN_DEPTH",
    "DEFAULT_VS_DESIGN_SCAN_DEPTH",
    "DEFAULT_VS_ENGINE",
    "VS_DESIGN_KINDS",
    "discover_vs_campaign_candidates",
    "discover_vs_design_candidates",
    "plan_vs_campaign",
    "plan_vs_design",
    "preview_vs_campaign",
    "preview_vs_design",
]
