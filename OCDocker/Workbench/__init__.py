#!/usr/bin/env python3

# Description
###############################################################################
"""
Experiment workbench models and planners for OCDocker.
"""

# Imports
###############################################################################
from OCDocker.Workbench.AblationDesign import build_ablation_design_context
from OCDocker.Workbench.AblationDesign import discover_ablation_input_features
from OCDocker.Workbench.AblationDesign import handle_ablation_design_post
from OCDocker.Workbench.AblationDesign import plan_ablation_design
from OCDocker.Workbench.AblationDesign import preview_ablation_design
from OCDocker.Workbench.AblationDesign import write_ablation_design_policy
from OCDocker.Workbench.Adoption import build_adoption_plan
from OCDocker.Workbench.Adoption import write_adoption_workspace
from OCDocker.Workbench.Ablation import build_ablation_analysis
from OCDocker.Workbench.Ablation import parse_ablation_metric
from OCDocker.Workbench.AblationProtocolSimilarity import build_ablation_protocol_similarity_analysis
from OCDocker.Workbench.Artifacts import build_artifact_index
from OCDocker.Workbench.Bundle import build_run_bundle
from OCDocker.Workbench.Comparison import build_run_comparison
from OCDocker.Workbench.Comparison import parse_comparison_metric
from OCDocker.Workbench.Decision import build_metrics_catalog
from OCDocker.Workbench.Decision import build_pareto_front
from OCDocker.Workbench.Decision import parse_pareto_objective
from OCDocker.Workbench.Evidence import build_evidence_index
from OCDocker.Workbench.Evidence import resolve_evidence_asset
from OCDocker.Workbench.Export import build_publication_export
from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.IO import read_spec
from OCDocker.Workbench.IO import write_model
from OCDocker.Workbench.Launch import build_launch_script
from OCDocker.Workbench.Launch import build_run_launch_plan
from OCDocker.Workbench.Launch import write_launch_script
from OCDocker.Workbench.Leaderboard import build_metric_leaderboard
from OCDocker.Workbench.MetricsMatrix import build_metric_matrix
from OCDocker.Workbench.Logs import preview_run_logs
from OCDocker.Workbench.Models import ComparisonDirection
from OCDocker.Workbench.Models import FeaturePolicySelection
from OCDocker.Workbench.Models import EvidenceKind
from OCDocker.Workbench.Models import ExportedArtifact
from OCDocker.Workbench.Models import InventoryIssue
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
from OCDocker.Workbench.Models import WorkbenchOCScoreWorkspace
from OCDocker.Workbench.Models import WorkbenchOCScoreStudy
from OCDocker.Workbench.Models import WorkbenchOCScoreReplica
from OCDocker.Workbench.Models import WorkbenchOCScoreMetric
from OCDocker.Workbench.Models import WorkbenchOCScoreFigure
from OCDocker.Workbench.Models import OCScoreWorkspaceRole
from OCDocker.Workbench.Models import OCScoreReplicaStatus
from OCDocker.Workbench.Models import OCScoreMetricDirection
from OCDocker.Workbench.Models import OCScoreInputSpec
from OCDocker.Workbench.Models import OCScoreStudySpec
from OCDocker.Workbench.Models import MetricSortMode
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
from OCDocker.Workbench.Models import RunDetail
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
from OCDocker.Workbench.Models import WorkbenchAblationAnalysis
from OCDocker.Workbench.Models import WorkbenchAblationCandidate
from OCDocker.Workbench.Models import WorkbenchAdoptedRun
from OCDocker.Workbench.Models import WorkbenchAdoptionCandidate
from OCDocker.Workbench.Models import WorkbenchAdoptionPlan
from OCDocker.Workbench.Models import WorkbenchAdoptionResult
from OCDocker.Workbench.Models import WorkbenchAnalysisReport
from OCDocker.Workbench.Models import WorkbenchArtifactEntry
from OCDocker.Workbench.Models import WorkbenchArtifactIndex
from OCDocker.Workbench.Models import WorkbenchEvidenceEntry
from OCDocker.Workbench.Models import WorkbenchEvidenceIndex
from OCDocker.Workbench.Models import WorkbenchComparison
from OCDocker.Workbench.Models import WorkbenchComparisonCandidate
from OCDocker.Workbench.Models import WorkbenchComparisonMetric
from OCDocker.Workbench.Models import WorkbenchSpec
from OCDocker.Workbench.Models import WorkbenchReportFinding
from OCDocker.Workbench.Models import WorkbenchReportFindingKind
from OCDocker.Workbench.Models import WorkspaceInventory
from OCDocker.Workbench.Models import WorkspaceOverview
from OCDocker.Workbench.OCScoreLayout import ablation_container_paths
from OCDocker.Workbench.OCScoreLayout import build_ocscore_workspace
from OCDocker.Workbench.OCScoreLayout import resolve_ocscore_layout_root
from OCDocker.Workbench.Overview import build_workspace_overview
from OCDocker.Workbench.Planner import build_run_manifest
from OCDocker.Workbench.Plots import build_leaderboard_plot
from OCDocker.Workbench.Plots import build_metric_scatter_plot
from OCDocker.Workbench.Plots import build_parallel_coordinates_plot
from OCDocker.Workbench.Plots import build_pareto_scatter_plot
from OCDocker.Workbench.Planner import plan_command
from OCDocker.Workbench.Planner import plan_ocscore_train_command
from OCDocker.Workbench.Planner import plan_snakemake_command
from OCDocker.Workbench.Planner import plan_vs_campaign_command
from OCDocker.Workbench.Preflight import preflight_spec
from OCDocker.Workbench.Report import build_analysis_report
from OCDocker.Workbench.Report import parse_report_metric
from OCDocker.Workbench.Report import render_analysis_report_markdown
from OCDocker.Workbench.Preflight import preflight_spec_file
from OCDocker.Workbench.Registry import discover_result_manifest_paths
from OCDocker.Workbench.Registry import discover_run_manifest_paths
from OCDocker.Workbench.Registry import scan_workspace
from OCDocker.Workbench.Registry import summarize_run_manifest
from OCDocker.Workbench.RunDetail import build_run_detail
from OCDocker.Workbench.Results import summarize_results
from OCDocker.Workbench.Schema import available_schema_names
from OCDocker.Workbench.Schema import build_json_schema
from OCDocker.Workbench.Schema import build_schema_catalog
from OCDocker.Workbench.Server import WorkbenchAPIError
from OCDocker.Workbench.Server import build_workbench_api_handler
from OCDocker.Workbench.Server import build_workbench_api_payload
from OCDocker.Workbench.Server import serve_workbench_api
from OCDocker.Workbench.Status import inspect_run_status
from OCDocker.Workbench.Templates import available_template_names
from OCDocker.Workbench.Templates import build_template_payload
from OCDocker.Workbench.Templates import build_template_spec
from OCDocker.Workbench.Web import build_workbench_web_asset
from OCDocker.Workbench.Web import is_workbench_web_asset_path

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


__all__ = [
    "ComparisonDirection",
    "EvidenceKind",
    "FeaturePolicySelection",
    "ExportedArtifact",
    "InventoryIssue",
    "MetricSortMode",
    "MetricLeaderboardEntry",
    "MetricMatrixRow",
    "MetricMatrix",
    "MetricLeaderboard",
    "ParetoObjective",
    "ParetoFront",
    "ParetoEntry",
    "MetricCatalogEntry",
    "MetricCatalog",
    "OCScoreAblationSpec",
    "WorkbenchOCScoreWorkspace",
    "WorkbenchOCScoreStudy",
    "WorkbenchOCScoreReplica",
    "WorkbenchOCScoreMetric",
    "WorkbenchOCScoreFigure",
    "OCScoreWorkspaceRole",
    "OCScoreReplicaStatus",
    "OCScoreMetricDirection",
    "OCScoreInputSpec",
    "OCScoreStudySpec",
    "PreflightCheck",
    "PreflightReport",
    "PlannedCommand",
    "PublicationExport",
    "ResourceSpec",
    "ResultArtifact",
    "ResultArtifactStatus",
    "ResultManifest",
    "ResultSummary",
    "RunBundle",
    "RunDetail",
    "RunInventoryItem",
    "RunLogFilePreview",
    "RunLogPreview",
    "RunLaunchPlan",
    "RunManifest",
    "RunPathStatus",
    "RunStatusReport",
    "SnakemakeWorkflowSpec",
    "VSInputSpec",
    "VSCampaignSpec",
    "WorkbenchAPIError",
    "WorkbenchAblationCandidate",
    "WorkbenchAblationAnalysis",
    "WorkbenchAdoptionResult",
    "WorkbenchAdoptionPlan",
    "WorkbenchAdoptionCandidate",
    "WorkbenchAdoptedRun",
    "WorkbenchAnalysisReport",
    "WorkbenchArtifactEntry",
    "WorkbenchArtifactIndex",
    "WorkbenchEvidenceEntry",
    "WorkbenchEvidenceIndex",
    "WorkbenchComparison",
    "WorkbenchComparisonCandidate",
    "WorkbenchComparisonMetric",
    "WorkbenchPlot",
    "WorkbenchPlotKind",
    "WorkbenchSpec",
    "WorkbenchReportFinding",
    "WorkbenchReportFindingKind",
    "WorkspaceInventory",
    "WorkspaceOverview",
    "available_schema_names",
    "parse_ablation_metric",
    "build_ablation_analysis",
    "build_ablation_protocol_similarity_analysis",
    "available_template_names",
    "write_adoption_workspace",
    "build_adoption_plan",
    "ablation_container_paths",
    "build_ablation_design_context",
    "build_artifact_index",
    "discover_ablation_input_features",
    "handle_ablation_design_post",
    "build_evidence_index",
    "resolve_evidence_asset",
    "resolve_ocscore_layout_root",
    "build_run_comparison",
    "build_json_schema",
    "write_launch_script",
    "build_run_launch_plan",
    "build_launch_script",
    "build_metric_leaderboard",
    "build_metric_matrix",
    "parse_comparison_metric",
    "parse_pareto_objective",
    "parse_report_metric",
    "build_pareto_front",
    "build_ocscore_workspace",
    "build_metrics_catalog",
    "build_leaderboard_plot",
    "build_metric_scatter_plot",
    "build_parallel_coordinates_plot",
    "build_pareto_scatter_plot",
    "build_analysis_report",
    "build_workspace_overview",
    "build_schema_catalog",
    "build_workbench_api_handler",
    "build_workbench_api_payload",
    "build_workbench_web_asset",
    "build_template_payload",
    "build_template_spec",
    "build_run_manifest",
    "build_run_bundle",
    "build_run_detail",
    "build_publication_export",
    "discover_result_manifest_paths",
    "discover_run_manifest_paths",
    "inspect_run_status",
    "is_workbench_web_asset_path",
    "model_to_data",
    "plan_command",
    "plan_ablation_design",
    "preview_ablation_design",
    "write_ablation_design_policy",
    "plan_ocscore_train_command",
    "plan_snakemake_command",
    "plan_vs_campaign_command",
    "preflight_spec",
    "preflight_spec_file",
    "preview_run_logs",
    "read_result_manifest",
    "render_analysis_report_markdown",
    "read_run_manifest",
    "read_spec",
    "scan_workspace",
    "serve_workbench_api",
    "summarize_results",
    "summarize_run_manifest",
    "write_model",
]
