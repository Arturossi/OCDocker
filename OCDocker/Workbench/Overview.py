#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only workspace overview helpers for GUI dashboard entry points.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

from OCDocker.Workbench.Models import RunInventoryItem
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchSpecType
from OCDocker.Workbench.Models import WorkspaceOverview
from OCDocker.Workbench.Registry import scan_workspace

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

RUN_STATUS_ORDER: tuple[RunStatus, ...] = (
    "defined",
    "built",
    "dry_run",
    "running",
    "completed",
    "failed",
    "cancelled",
)
SPEC_TYPE_ORDER: tuple[WorkbenchSpecType, ...] = (
    "vs_campaign",
    "ocscore_study",
    "ocscore_ablation",
)

# Functions
###############################################################################
## Private ##


def _count_by_status(runs: tuple[RunInventoryItem, ...]) -> dict[str, int]:
    '''Count runs by Workbench run status.

    Parameters
    ----------
    runs : tuple[RunInventoryItem, ...]
        Inventory run items.

    Returns
    -------
    dict[str, int]
        Status counts with stable keys.
    '''

    counts: dict[str, int] = {status: 0 for status in RUN_STATUS_ORDER}
    for run in runs:
        counts[run.status] = counts.get(run.status, 0) + 1
    return counts


def _count_by_spec_type(runs: tuple[RunInventoryItem, ...]) -> dict[str, int]:
    '''Count runs by Workbench spec type.

    Parameters
    ----------
    runs : tuple[RunInventoryItem, ...]
        Inventory run items.

    Returns
    -------
    dict[str, int]
        Spec-type counts with stable keys.
    '''

    counts: dict[str, int] = {spec_type: 0 for spec_type in SPEC_TYPE_ORDER}
    for run in runs:
        counts[run.spec_type] = counts.get(run.spec_type, 0) + 1
    return counts


def _recent_runs(
    runs: tuple[RunInventoryItem, ...], *, recent_limit: int
) -> tuple[RunInventoryItem, ...]:
    '''Return the most recently updated run inventory items.

    Parameters
    ----------
    runs : tuple[RunInventoryItem, ...]
        Inventory run items.
    recent_limit : int
        Maximum number of runs to return.

    Returns
    -------
    tuple[RunInventoryItem, ...]
        Runs ordered newest first.
    '''

    return tuple(
        sorted(runs, key=lambda run: (run.updated_at, run.run_id), reverse=True)[
            :recent_limit
        ]
    )


## Public ##


def build_workspace_overview(
    root: str | Path, *, max_depth: int = 6, recent_limit: int = 20
) -> WorkspaceOverview:
    '''Build a read-only workspace overview for GUI dashboards.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or manifest file to inspect.
    max_depth : int
        Maximum directory depth below root to scan.
    recent_limit : int
        Maximum number of recently updated runs to include.

    Returns
    -------
    WorkspaceOverview
        Workspace overview payload.
    '''

    if recent_limit < 1:
        raise ValueError("recent_limit must be greater than or equal to one.")

    inventory = scan_workspace(root, max_depth=max_depth)
    missing_artifact_count = sum(len(run.missing_artifacts) for run in inventory.runs)
    return WorkspaceOverview(
        root=inventory.root,
        max_depth=inventory.max_depth,
        scanned_at=inventory.scanned_at,
        run_count=len(inventory.runs),
        result_manifest_count=len(inventory.result_manifests),
        issue_count=len(inventory.issues),
        missing_artifact_count=missing_artifact_count,
        status_counts=_count_by_status(inventory.runs),
        spec_type_counts=_count_by_spec_type(inventory.runs),
        recent_runs=_recent_runs(inventory.runs, recent_limit=recent_limit),
        issues=inventory.issues,
    )


__all__ = [
    "RUN_STATUS_ORDER",
    "SPEC_TYPE_ORDER",
    "build_workspace_overview",
]
