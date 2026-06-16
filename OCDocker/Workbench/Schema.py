#!/usr/bin/env python3

# Description
###############################################################################
'''
JSON Schema helpers for Workbench GUI and automation integrations.
'''

# Imports
###############################################################################
from __future__ import annotations

from typing import Any
from typing import Literal

from OCDocker.Workbench.Models import FeaturePolicySelection
from OCDocker.Workbench.Models import MetricLeaderboardEntry
from OCDocker.Workbench.Models import MetricLeaderboard
from OCDocker.Workbench.Models import ParetoObjective
from OCDocker.Workbench.Models import ParetoFront
from OCDocker.Workbench.Models import ParetoEntry
from OCDocker.Workbench.Models import MetricCatalogEntry
from OCDocker.Workbench.Models import MetricCatalog
from OCDocker.Workbench.Models import MetricMatrixRow
from OCDocker.Workbench.Models import MetricMatrix
from OCDocker.Workbench.Models import OCScoreAblationSpec
from OCDocker.Workbench.Models import OCScoreInputSpec
from OCDocker.Workbench.Models import OCScoreStudySpec
from OCDocker.Workbench.Models import PlannedCommand
from OCDocker.Workbench.Models import PreflightCheck
from OCDocker.Workbench.Models import PreflightReport
from OCDocker.Workbench.Models import PublicationExport
from OCDocker.Workbench.Models import ResourceSpec
from OCDocker.Workbench.Models import ResultArtifact
from OCDocker.Workbench.Models import ResultArtifactStatus
from OCDocker.Workbench.Models import ResultManifest
from OCDocker.Workbench.Models import ResultSummary
from OCDocker.Workbench.Models import RunBundle
from OCDocker.Workbench.Models import RunInventoryItem
from OCDocker.Workbench.Models import RunLaunchPlan
from OCDocker.Workbench.Models import RunLogFilePreview
from OCDocker.Workbench.Models import RunLogPreview
from OCDocker.Workbench.Models import RunManifest
from OCDocker.Workbench.Models import RunPathStatus
from OCDocker.Workbench.Models import RunStatusReport
from OCDocker.Workbench.Models import SnakemakeWorkflowSpec
from OCDocker.Workbench.Models import VSInputSpec
from OCDocker.Workbench.Models import VSCampaignSpec
from OCDocker.Workbench.Models import WorkbenchAnalysisReport
from OCDocker.Workbench.Models import WorkbenchArtifactEntry
from OCDocker.Workbench.Models import WorkbenchArtifactIndex
from OCDocker.Workbench.Models import WorkbenchModel
from OCDocker.Workbench.Models import WorkbenchPlot
from OCDocker.Workbench.Models import WorkbenchReportFinding
from OCDocker.Workbench.Models import WorkspaceInventory
from OCDocker.Workbench.Models import WorkspaceOverview

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

# Type aliases
###############################################################################

WorkbenchSchemaName = Literal[
    "feature_policy",
    "metric_catalog",
    "metric_catalog_entry",
    "metric_leaderboard",
    "metric_leaderboard_entry",
    "metric_matrix",
    "metric_matrix_row",
    "ocscore_ablation",
    "ocscore_input",
    "ocscore_study",
    "pareto_entry",
    "pareto_front",
    "pareto_objective",
    "planned_command",
    "preflight_check",
    "preflight_report",
    "publication_export",
    "resource_spec",
    "result_artifact",
    "result_artifact_status",
    "result_manifest",
    "result_summary",
    "run_bundle",
    "run_inventory_item",
    "run_launch_plan",
    "run_log_file_preview",
    "run_log_preview",
    "run_manifest",
    "run_path_status",
    "run_status_report",
    "snakemake_workflow",
    "vs_campaign",
    "vs_input",
    "workbench_analysis_report",
    "workbench_artifact_entry",
    "workbench_artifact_index",
    "workbench_plot",
    "workbench_report_finding",
    "workspace_inventory",
    "workspace_overview",
]

# Constants
###############################################################################

_SCHEMA_MODELS: dict[str, type[WorkbenchModel]] = {
    "feature_policy": FeaturePolicySelection,
    "metric_catalog": MetricCatalog,
    "metric_catalog_entry": MetricCatalogEntry,
    "metric_leaderboard": MetricLeaderboard,
    "metric_leaderboard_entry": MetricLeaderboardEntry,
    "metric_matrix": MetricMatrix,
    "metric_matrix_row": MetricMatrixRow,
    "ocscore_ablation": OCScoreAblationSpec,
    "ocscore_input": OCScoreInputSpec,
    "ocscore_study": OCScoreStudySpec,
    "pareto_entry": ParetoEntry,
    "pareto_front": ParetoFront,
    "pareto_objective": ParetoObjective,
    "planned_command": PlannedCommand,
    "preflight_check": PreflightCheck,
    "preflight_report": PreflightReport,
    "publication_export": PublicationExport,
    "resource_spec": ResourceSpec,
    "result_artifact": ResultArtifact,
    "result_artifact_status": ResultArtifactStatus,
    "result_manifest": ResultManifest,
    "result_summary": ResultSummary,
    "run_bundle": RunBundle,
    "run_inventory_item": RunInventoryItem,
    "run_launch_plan": RunLaunchPlan,
    "run_log_file_preview": RunLogFilePreview,
    "run_log_preview": RunLogPreview,
    "run_manifest": RunManifest,
    "run_path_status": RunPathStatus,
    "run_status_report": RunStatusReport,
    "snakemake_workflow": SnakemakeWorkflowSpec,
    "vs_campaign": VSCampaignSpec,
    "vs_input": VSInputSpec,
    "workbench_analysis_report": WorkbenchAnalysisReport,
    "workbench_artifact_entry": WorkbenchArtifactEntry,
    "workbench_artifact_index": WorkbenchArtifactIndex,
    "workbench_plot": WorkbenchPlot,
    "workbench_report_finding": WorkbenchReportFinding,
    "workspace_inventory": WorkspaceInventory,
    "workspace_overview": WorkspaceOverview,
}

# Functions
###############################################################################
## Private ##


def _schema_model(name: str) -> type[WorkbenchModel]:
    '''Return the model registered for a schema name.

    Parameters
    ----------
    name : str
        Schema name.

    Returns
    -------
    type[WorkbenchModel]
        Registered Pydantic model class.
    '''

    normalized = str(name).strip()
    try:
        return _SCHEMA_MODELS[normalized]
    except KeyError as exc:
        available = ", ".join(available_schema_names())
        raise ValueError(
            f"Unknown Workbench schema {normalized!r}. Expected one of: {available}."
        ) from exc


## Public ##


def available_schema_names() -> tuple[str, ...]:
    '''Return registered Workbench JSON Schema names.

    Returns
    -------
    tuple[str, ...]
        Registered schema names in deterministic order.
    '''

    return tuple(sorted(_SCHEMA_MODELS))


def build_json_schema(name: str) -> dict[str, Any]:
    '''Build a JSON Schema for one registered Workbench model.

    Parameters
    ----------
    name : str
        Registered schema name.

    Returns
    -------
    dict[str, Any]
        JSON-compatible schema payload.
    '''

    model = _schema_model(name)
    return model.model_json_schema()


def build_schema_catalog(names: tuple[str, ...] | None = None) -> dict[str, Any]:
    '''Build a JSON Schema catalog for Workbench models.

    Parameters
    ----------
    names : tuple[str, ...] or None
        Optional subset of schema names. If None, all registered schemas are included.

    Returns
    -------
    dict[str, Any]
        JSON-compatible schema catalog.
    '''

    selected_names = available_schema_names() if names is None else tuple(names)
    schemas = {name: build_json_schema(name) for name in selected_names}
    return {
        "schema_version": 1,
        "available_schemas": available_schema_names(),
        "schemas": schemas,
    }


__all__ = [
    "WorkbenchSchemaName",
    "available_schema_names",
    "build_json_schema",
    "build_schema_catalog",
]
