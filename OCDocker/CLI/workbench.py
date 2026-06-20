#!/usr/bin/env python3

# Description
###############################################################################
'''
Workbench CLI commands for validating specs and planning commands.
'''

from __future__ import annotations

# Imports
###############################################################################
import argparse
import json
import shlex

from pathlib import Path
from typing import Any

import yaml

from OCDocker.Workbench.Adoption import build_adoption_plan
from OCDocker.Workbench.Adoption import write_adoption_workspace
from OCDocker.Workbench.Ablation import build_ablation_analysis
from OCDocker.Workbench.Ablation import parse_ablation_metric
from OCDocker.Workbench.Artifacts import build_artifact_index
from OCDocker.Workbench.Bundle import build_run_bundle
from OCDocker.Workbench.Comparison import build_run_comparison
from OCDocker.Workbench.Comparison import parse_comparison_metric
from OCDocker.Workbench.Decision import build_metrics_catalog
from OCDocker.Workbench.Decision import build_pareto_front
from OCDocker.Workbench.Decision import parse_pareto_objective
from OCDocker.Workbench.Evidence import build_evidence_index
from OCDocker.Workbench.Export import build_publication_export
from OCDocker.Workbench.IO import model_to_data
from OCDocker.Workbench.IO import read_spec
from OCDocker.Workbench.IO import write_model
from OCDocker.Workbench.Launch import build_run_launch_plan
from OCDocker.Workbench.Launch import write_launch_script
from OCDocker.Workbench.Leaderboard import build_metric_leaderboard
from OCDocker.Workbench.MetricsMatrix import build_metric_matrix
from OCDocker.Workbench.Logs import preview_run_logs
from OCDocker.Workbench.Models import WorkbenchSpec
from OCDocker.Workbench.Overview import build_workspace_overview
from OCDocker.Workbench.Planner import build_run_manifest
from OCDocker.Workbench.Planner import plan_command
from OCDocker.Workbench.Plots import build_leaderboard_plot
from OCDocker.Workbench.Plots import build_metric_scatter_plot
from OCDocker.Workbench.Plots import build_parallel_coordinates_plot
from OCDocker.Workbench.Plots import build_pareto_scatter_plot
from OCDocker.Workbench.Preflight import preflight_spec_file
from OCDocker.Workbench.Report import build_analysis_report
from OCDocker.Workbench.Report import parse_report_metric
from OCDocker.Workbench.Report import render_analysis_report_markdown
from OCDocker.Workbench.Registry import scan_workspace
from OCDocker.Workbench.Results import summarize_results
from OCDocker.Workbench.Schema import available_schema_names
from OCDocker.Workbench.Schema import build_schema_catalog
from OCDocker.Workbench.Server import DEFAULT_WORKBENCH_API_HOST
from OCDocker.Workbench.Server import DEFAULT_WORKBENCH_API_PORT
from OCDocker.Workbench.OCScoreLayout import MAX_OPTUNA_DASHBOARD_SLOT_COUNT
from OCDocker.Workbench.Server import serve_workbench_api
from OCDocker.Workbench.Status import inspect_run_status
from OCDocker.Workbench.Templates import available_template_names
from OCDocker.Workbench.Templates import build_template_payload

# License
###############################################################################
"""OCDocker
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
"""

# Functions
###############################################################################
## Private ##


def _load_spec_for_cli(path: str | Path) -> WorkbenchSpec:
    '''Load a Workbench spec for CLI commands.

    Parameters
    ----------
    path : str or pathlib.Path
        Spec path to load.

    Returns
    -------
    WorkbenchSpec
        Validated Workbench spec.
    '''

    return read_spec(path)


def _write_json_payload(payload: dict[str, Any], output: str | Path | None) -> None:
    '''Write a JSON payload to a file or stdout.

    Parameters
    ----------
    payload : dict[str, Any]
        JSON-compatible payload.
    output : str or pathlib.Path or None
        Optional output path. If None, the payload is printed to stdout.
    '''

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Workbench payload written: {output_path}")


def _write_structured_payload(
    payload: dict[str, Any],
    output: str | Path | None,
    output_format: str,
    *,
    message: str,
) -> None:
    '''Write a JSON or YAML payload to a file or stdout.

    Parameters
    ----------
    payload : dict[str, Any]
        Serializable payload.
    output : str or pathlib.Path or None
        Optional output path. If None, the payload is printed to stdout.
    output_format : str
        Output format, either ``json`` or ``yaml``.
    message : str
        Message prefix printed after writing a file.
    '''

    if output_format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    elif output_format == "yaml":
        text = yaml.safe_dump(payload, sort_keys=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    if output is None:
        print(text, end="")
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"{message}: {output_path}")


def _write_text_payload(text: str, output: str | Path | None, *, message: str) -> None:
    '''Write a text payload to a file or stdout.

    Parameters
    ----------
    text : str
        Text payload.
    output : str or pathlib.Path or None
        Optional output path. If None, the payload is printed to stdout.
    message : str
        Message prefix printed after writing a file.
    '''

    if output is None:
        print(text, end="")
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"{message}: {output_path}")


def _shell_command(command: tuple[str, ...]) -> str:
    '''Format a planned command as a shell-safe string.

    Parameters
    ----------
    command : tuple[str, ...]
        Command argument tuple.

    Returns
    -------
    str
        Shell-quoted command string.
    '''

    return " ".join(shlex.quote(part) for part in command)


def _validation_payload(spec: WorkbenchSpec, spec_path: str | Path) -> dict[str, Any]:
    '''Build the validation result payload.

    Parameters
    ----------
    spec : WorkbenchSpec
        Validated Workbench spec.
    spec_path : str or pathlib.Path
        Input spec path.

    Returns
    -------
    dict[str, Any]
        Validation payload.
    '''

    return {
        "valid": True,
        "spec_path": str(spec_path),
        "spec_type": spec.type,
        "name": spec.name,
        "schema_version": spec.schema_version,
    }


## Public ##


def cmd_adopt_plan(args: argparse.Namespace) -> int:
    '''Build a dry-run adoption plan for existing output directories.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        plan = build_adoption_plan(
            args.source,
            max_depth=args.max_depth,
            spec_type=args.spec_type,
            status=args.status,
            run_id_prefix=args.run_id_prefix,
            max_metric_file_bytes=args.max_metric_bytes,
            require_metrics=getattr(args, "require_metrics", False),
        )
    except Exception as exc:
        print(f"Error: could not build Workbench adoption plan: {exc}")
        return 2
    _write_json_payload(model_to_data(plan), args.output)
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    '''Write Workbench manifests for existing output directories.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        result = write_adoption_workspace(
            args.source,
            args.destination,
            max_depth=args.max_depth,
            spec_type=args.spec_type,
            status=args.status,
            run_id_prefix=args.run_id_prefix,
            max_metric_file_bytes=args.max_metric_bytes,
            require_metrics=getattr(args, "require_metrics", False),
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Error: could not adopt existing outputs into Workbench: {exc}")
        return 2
    _write_json_payload(model_to_data(result), args.output)
    return 0


def cmd_ablations(args: argparse.Namespace) -> int:
    '''Build a read-only ablation comparison against a reference run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        metrics = tuple(parse_ablation_metric(value) for value in (args.metrics or ()))
        analysis = build_ablation_analysis(
            args.root,
            baseline_run_id=args.baseline,
            candidates=tuple(args.candidates or ()),
            metrics=metrics,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench ablation analysis: {exc}")
        return 2
    _write_json_payload(model_to_data(analysis), args.output)
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    '''Build a read-only cross-run artifact index.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        index = build_artifact_index(
            args.root,
            kinds=tuple(args.kinds or ()),
            roles=tuple(args.roles or ()),
            require_existing=args.require_existing,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench artifact index: {exc}")
        return 2
    _write_json_payload(model_to_data(index), args.output)
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    '''Build a read-only OCScore evidence index from adopted sources.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        index = build_evidence_index(
            args.root,
            max_depth=args.max_depth,
            source_depth=args.source_depth,
            max_entries=args.max_entries,
            max_csv_rows=args.max_csv_rows,
            max_series=args.max_series,
            max_shap_features=args.max_shap_features,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench evidence index: {exc}")
        return 2
    _write_json_payload(model_to_data(index), args.output)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    '''Build a read-only preflight report for a Workbench spec.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        report = preflight_spec_file(
            args.spec,
            ocdocker_executable=args.ocdocker_executable,
            snakemake_executable=args.snakemake_executable,
        )
    except Exception as exc:
        print(f"Error: could not preflight Workbench spec: {exc}")
        return 2
    _write_json_payload(model_to_data(report), args.output)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    '''Build a Workbench run bundle without executing it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        spec = _load_spec_for_cli(args.spec)
        bundle = build_run_bundle(
            spec,
            args.bundle_dir,
            run_id=args.run_id,
            overwrite=args.overwrite,
            ocdocker_executable=args.ocdocker_executable,
            snakemake_executable=args.snakemake_executable,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench bundle: {exc}")
        return 2
    _write_json_payload(model_to_data(bundle), args.output)
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    '''Build a starter Workbench spec template.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        payload = build_template_payload(args.template_name)
        _write_structured_payload(
            payload,
            args.output,
            args.output_format,
            message="Workbench template written",
        )
    except Exception as exc:
        print(f"Error: could not build Workbench template: {exc}")
        return 2
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    '''Validate a Workbench spec file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        spec = _load_spec_for_cli(args.spec)
    except Exception as exc:
        print(f"Error: invalid Workbench spec: {exc}")
        return 2
    _write_json_payload(_validation_payload(spec, args.spec), args.output)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    '''Build a read-only run comparison against a baseline result manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        metrics = tuple(parse_comparison_metric(value) for value in (args.metrics or ()))
        comparison = build_run_comparison(
            args.root,
            baseline_run_id=args.baseline,
            candidates=tuple(args.candidates or ()),
            metrics=metrics,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench comparison: {exc}")
        return 2
    _write_json_payload(model_to_data(comparison), args.output)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    '''Build a publishable export scaffold from a Workbench manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        export = build_publication_export(
            args.manifest,
            args.export_dir,
            copy_artifacts=args.copy_artifacts,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench export: {exc}")
        return 2
    _write_json_payload(model_to_data(export), args.output)
    return 0


def cmd_overview(args: argparse.Namespace) -> int:
    '''Build a read-only workspace overview for dashboard consumers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        overview = build_workspace_overview(
            args.root,
            max_depth=args.max_depth,
            recent_limit=args.recent_limit,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench overview: {exc}")
        return 2
    _write_json_payload(model_to_data(overview), args.output)
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    '''Build a read-only inventory for a Workbench root.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        inventory = scan_workspace(args.root, max_depth=args.max_depth)
    except Exception as exc:
        print(f"Error: could not inventory Workbench root: {exc}")
        return 2
    _write_json_payload(model_to_data(inventory), args.output)
    return 0


def cmd_launch_plan(args: argparse.Namespace) -> int:
    '''Build a non-executing launch plan for a prepared Workbench run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        plan = build_run_launch_plan(
            args.target,
            log_dir=args.log_dir,
            script_path=args.script_output,
        )
        if args.script_output is not None:
            plan = write_launch_script(plan, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Error: could not build Workbench launch plan: {exc}")
        return 2
    _write_json_payload(model_to_data(plan), args.output)
    return 0


def cmd_metrics_catalog(args: argparse.Namespace) -> int:
    '''Build a read-only metric coverage catalog from result manifests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        catalog = build_metrics_catalog(args.root, max_depth=args.max_depth)
    except Exception as exc:
        print(f"Error: could not build Workbench metrics catalog: {exc}")
        return 2
    _write_json_payload(model_to_data(catalog), args.output)
    return 0


def cmd_pareto(args: argparse.Namespace) -> int:
    '''Build a read-only Pareto front from result manifests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        objectives = tuple(parse_pareto_objective(value) for value in args.objectives)
        front = build_pareto_front(
            args.root,
            objectives=objectives,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench Pareto front: {exc}")
        return 2
    _write_json_payload(model_to_data(front), args.output)
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    '''Build a read-only metric leaderboard from result manifests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        leaderboard = build_metric_leaderboard(
            args.root,
            metric_name=args.metric,
            mode=args.mode,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench leaderboard: {exc}")
        return 2
    _write_json_payload(model_to_data(leaderboard), args.output)
    return 0


def cmd_metrics_matrix(args: argparse.Namespace) -> int:
    '''Build a read-only metric matrix from result manifests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        matrix = build_metric_matrix(
            args.root,
            metric_names=tuple(args.metrics or ()),
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench metrics matrix: {exc}")
        return 2
    _write_json_payload(model_to_data(matrix), args.output)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    '''Build a bounded read-only log preview for one Workbench run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        preview = preview_run_logs(
            args.target,
            lines=args.lines,
            max_bytes=args.max_bytes,
            encoding=args.encoding,
        )
    except Exception as exc:
        print(f"Error: could not preview Workbench logs: {exc}")
        return 2
    _write_json_payload(model_to_data(preview), args.output)
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    '''Build a read-only result summary for a Workbench manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        summary = summarize_results(args.manifest)
    except Exception as exc:
        print(f"Error: could not summarize Workbench results: {exc}")
        return 2
    _write_json_payload(model_to_data(summary), args.output)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    '''Build a Plotly-compatible Workbench plot payload.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        if args.kind == "leaderboard":
            metrics = tuple(args.metrics or ())
            if len(metrics) != 1:
                raise ValueError("leaderboard plots require exactly one --metric.")
            plot = build_leaderboard_plot(
                args.root,
                metric_name=metrics[0],
                mode=args.mode,
                max_depth=args.max_depth,
                top_n=args.top_n,
            )
        elif args.kind == "scatter":
            if args.x_metric is None or args.y_metric is None:
                raise ValueError("scatter plots require --x-metric and --y-metric.")
            plot = build_metric_scatter_plot(
                args.root,
                x_metric=args.x_metric,
                y_metric=args.y_metric,
                color_metric=args.color_metric,
                max_depth=args.max_depth,
            )
        elif args.kind == "parallel":
            plot = build_parallel_coordinates_plot(
                args.root,
                metric_names=tuple(args.metrics or ()),
                max_depth=args.max_depth,
            )
        elif args.kind == "pareto":
            objectives = tuple(parse_pareto_objective(value) for value in (args.objectives or ()))
            plot = build_pareto_scatter_plot(
                args.root,
                objectives=objectives,
                max_depth=args.max_depth,
            )
        else:
            raise ValueError(f"Unsupported plot kind: {args.kind}")
    except Exception as exc:
        print(f"Error: could not build Workbench plot payload: {exc}")
        return 2
    _write_json_payload(model_to_data(plot), args.output)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    '''Build a composed read-only analysis report for a workspace.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        leaderboards = tuple(parse_report_metric(value) for value in (args.leaderboards or ()))
        objectives = tuple(parse_pareto_objective(value) for value in (args.objectives or ()))
        report = build_analysis_report(
            args.root,
            leaderboards=leaderboards,
            pareto_objectives=objectives,
            max_depth=args.max_depth,
            recent_limit=args.recent_limit,
            top_n=args.top_n,
        )
    except Exception as exc:
        print(f"Error: could not build Workbench analysis report: {exc}")
        return 2
    if args.output_format == "markdown":
        _write_text_payload(
            render_analysis_report_markdown(report),
            args.output,
            message="Workbench report written",
        )
    else:
        _write_json_payload(model_to_data(report), args.output)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    '''Build JSON Schema payloads for Workbench models.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        names = tuple(args.names) if args.names else None
        catalog = build_schema_catalog(names)
    except Exception as exc:
        print(f"Error: could not build Workbench schema catalog: {exc}")
        return 2
    _write_json_payload(catalog, args.output)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    '''Serve the read-only local Workbench API.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        serve_workbench_api(
            args.root,
            host=args.host,
            port=args.port,
            max_depth=args.max_depth,
            optuna_dashboard_port_start=args.optuna_port_start,
            optuna_dashboard_port_end=args.optuna_port_end,
            optuna_dashboard_slots=args.optuna_slots,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("Workbench API stopped.")
        return 0
    except Exception as exc:
        print(f"Error: could not serve Workbench API: {exc}")
        return 2
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    '''Build a read-only status report for one Workbench run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        status = inspect_run_status(args.target)
    except Exception as exc:
        print(f"Error: could not inspect Workbench run: {exc}")
        return 2
    _write_json_payload(model_to_data(status), args.output)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    '''Plan a command from a Workbench spec file without executing it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    try:
        spec = _load_spec_for_cli(args.spec)
        plan = plan_command(
            spec,
            ocdocker_executable=args.ocdocker_executable,
            snakemake_executable=args.snakemake_executable,
        )
    except Exception as exc:
        print(f"Error: could not plan Workbench command: {exc}")
        return 2

    payload: dict[str, Any] = {
        "spec_path": str(args.spec),
        "spec": model_to_data(spec),
        "plan": plan.model_dump(mode="json", exclude_none=True),
        "shell_command": _shell_command(plan.command),
    }
    if args.run_id:
        manifest = build_run_manifest(spec, plan, run_id=args.run_id)
        payload["run_manifest"] = manifest.model_dump(mode="json", exclude_none=True)
        if args.manifest_output:
            write_model(args.manifest_output, manifest)
    _write_json_payload(payload, args.output)
    return 0


def cmd_workbench(args: argparse.Namespace) -> int:
    '''Dispatch a Workbench subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code.
    '''

    handler = getattr(args, "func", None)
    if handler is None:
        return 2
    return int(handler(args))


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``ocdocker workbench`` command group.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Main CLI subparser registry.
    '''

    parser = subparsers.add_parser(
        "workbench",
        description=(
            "Validate OCDocker Workbench specs, inspect manifests, and plan "
            "commands without executing runs.\n\n"
            "This command group is intended for GUI and automation integration. "
            "It does not launch Snakemake, OCScore, or docking jobs."
        ),
        help="Validate, inspect, and plan Workbench runs without execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=cmd_workbench)
    workbench_sub = parser.add_subparsers(dest="workbench_command", required=True)

    adopt_plan = workbench_sub.add_parser(
        "adopt-plan",
        help="Plan adoption of existing output folders without writing manifests",
        description=(
            "Scan an existing OCDocker output tree and report which folders can "
            "be represented as Workbench run/result manifests. The source tree "
            "is never modified."
        ),
    )
    adopt_plan.add_argument("source", help="Existing output root to inspect.")
    adopt_plan.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth below source to inspect. OCScore ablation policy folders are also discovered from reached train or ablations folders. Default: 3.",
    )
    adopt_plan.add_argument(
        "--spec-type",
        choices=("vs_campaign", "ocscore_study", "ocscore_ablation"),
        default="ocscore_ablation",
        help="Workbench spec type assigned to adopted runs. Default: ocscore_ablation.",
    )
    adopt_plan.add_argument(
        "--status",
        choices=("defined", "built", "dry_run", "running", "completed", "failed", "cancelled"),
        default=None,
        help="Optional status override. If omitted, completed is inferred when metrics are found.",
    )
    adopt_plan.add_argument(
        "--run-id-prefix",
        default="",
        help="Optional prefix applied to generated run ids.",
    )
    adopt_plan.add_argument(
        "--max-metric-bytes",
        type=int,
        default=1048576,
        help="Maximum metric file size parsed during scanning. Default: 1048576.",
    )
    adopt_plan.add_argument(
        "--require-metrics",
        action="store_true",
        default=False,
        help="Only include adopted directories with at least one parsed metric.",
    )
    adopt_plan.add_argument("--output", default=None, help="Optional JSON output path.")
    adopt_plan.set_defaults(func=cmd_adopt_plan)

    adopt = workbench_sub.add_parser(
        "adopt",
        help="Write Workbench manifests for existing output folders",
        description=(
            "Scan an existing output tree and write Workbench run/result "
            "manifests into a separate destination workspace. Original files "
            "are not moved, copied, deleted, or modified."
        ),
    )
    adopt.add_argument("source", help="Existing output root to inspect.")
    adopt.add_argument("destination", help="Workbench destination root to write.")
    adopt.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth below source to inspect. OCScore ablation policy folders are also discovered from reached train or ablations folders. Default: 3.",
    )
    adopt.add_argument(
        "--spec-type",
        choices=("vs_campaign", "ocscore_study", "ocscore_ablation"),
        default="ocscore_ablation",
        help="Workbench spec type assigned to adopted runs. Default: ocscore_ablation.",
    )
    adopt.add_argument(
        "--status",
        choices=("defined", "built", "dry_run", "running", "completed", "failed", "cancelled"),
        default=None,
        help="Optional status override. If omitted, completed is inferred when metrics are found.",
    )
    adopt.add_argument(
        "--run-id-prefix",
        default="",
        help="Optional prefix applied to generated run ids.",
    )
    adopt.add_argument(
        "--max-metric-bytes",
        type=int,
        default=1048576,
        help="Maximum metric file size parsed during scanning. Default: 1048576.",
    )
    adopt.add_argument(
        "--require-metrics",
        action="store_true",
        default=False,
        help="Only include adopted directories with at least one parsed metric.",
    )
    adopt.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination Workbench manifests for adopted runs.",
    )
    adopt.add_argument("--output", default=None, help="Optional JSON output path.")
    adopt.set_defaults(func=cmd_adopt)

    artifacts = workbench_sub.add_parser(
        "artifacts",
        help="Index declared Workbench artifacts across manifests",
        description=(
            "Scan Workbench run and result manifests below a root path and emit "
            "a read-only artifact browser payload with existence, kind, role, "
            "size, and source manifest metadata. No artifacts are copied or opened."
        ),
    )
    artifacts.add_argument("root", help="Workspace root or manifest file to inspect.")
    artifacts.add_argument(
        "--kind",
        dest="kinds",
        action="append",
        default=None,
        help="Artifact kind to include. May be supplied multiple times.",
    )
    artifacts.add_argument(
        "--role",
        dest="roles",
        action="append",
        default=None,
        help="Artifact role to include. May be supplied multiple times.",
    )
    artifacts.add_argument(
        "--require-existing",
        action="store_true",
        default=False,
        help="Include only artifacts that currently exist on disk.",
    )
    artifacts.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    artifacts.add_argument("--output", default=None, help="Optional JSON output path.")
    artifacts.set_defaults(func=cmd_artifacts)

    evidence = workbench_sub.add_parser(
        "evidence",
        help="Discover OCScore evidence files from adopted outputs",
        description=(
            "Scan adopted Workbench source paths and emit a read-only evidence "
            "payload for OCScore performance tables, Optuna traces, SHAP exports, "
            "and analysis figures. Source outputs are not modified."
        ),
    )
    evidence.add_argument("root", help="Workspace root or result manifest to inspect.")
    evidence.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum Workbench manifest scan depth. Default: 6.",
    )
    evidence.add_argument(
        "--source-depth",
        type=int,
        default=6,
        help="Maximum directory depth below each adopted source path. Default: 6.",
    )
    evidence.add_argument(
        "--max-entries",
        type=int,
        default=400,
        help="Maximum evidence file entries to return. Default: 400.",
    )
    evidence.add_argument(
        "--max-csv-rows",
        type=int,
        default=1000,
        help="Maximum CSV rows read per evidence file for previews. Default: 1000.",
    )
    evidence.add_argument(
        "--max-series",
        type=int,
        default=8,
        help="Maximum Optuna trial series to preview. Default: 8.",
    )
    evidence.add_argument(
        "--max-shap-features",
        type=int,
        default=30,
        help="Maximum SHAP features to preview. Default: 30.",
    )
    evidence.add_argument("--output", default=None, help="Optional JSON output path.")
    evidence.set_defaults(func=cmd_evidence)

    build = workbench_sub.add_parser(
        "build",
        help="Build a prepared run bundle without executing it",
        description=(
            "Validate a Workbench spec, plan its command, and write spec.yml, "
            "plan.json, run_manifest.yml, and bundle.json into a bundle directory. "
            "No run is launched."
        ),
    )
    build.add_argument("spec", help="Workbench spec path (.yml, .yaml, or .json).")
    build.add_argument("bundle_dir", help="Directory that will receive bundle files.")
    build.add_argument("--run-id", required=True, help="Stable run id for the bundle.")
    build.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing bundle files if they already exist.",
    )
    build.add_argument("--output", default=None, help="Optional JSON summary output path.")
    build.add_argument(
        "--ocdocker-executable",
        default="ocdocker",
        help="OCDocker executable used in planned OCScore commands.",
    )
    build.add_argument(
        "--snakemake-executable",
        default="snakemake",
        help="Snakemake executable used in planned VS campaign commands.",
    )
    build.set_defaults(func=cmd_build)

    check = workbench_sub.add_parser(
        "check",
        help="Preflight a Workbench spec without executing runs",
        description=(
            "Validate and preflight a Workbench spec by planning its command "
            "and checking declared inputs, outputs, and executable availability. "
            "No run is launched."
        ),
    )
    check.add_argument("spec", help="Workbench spec path (.yml, .yaml, or .json).")
    check.add_argument("--output", default=None, help="Optional JSON output path.")
    check.add_argument(
        "--ocdocker-executable",
        default="ocdocker",
        help="OCDocker executable used in planned OCScore commands.",
    )
    check.add_argument(
        "--snakemake-executable",
        default="snakemake",
        help="Snakemake executable used in planned VS campaign commands.",
    )
    check.set_defaults(func=cmd_check)

    validate = workbench_sub.add_parser(
        "validate",
        help="Validate a Workbench YAML or JSON spec",
        description="Validate a Workbench YAML or JSON spec and print a compact JSON summary.",
    )
    validate.add_argument("spec", help="Workbench spec path (.yml, .yaml, or .json).")
    validate.add_argument("--output", default=None, help="Optional JSON output path.")
    validate.set_defaults(func=cmd_validate)

    template = workbench_sub.add_parser(
        "template",
        help="Emit a starter Workbench spec without executing runs",
        description=(
            "Emit a validated starter Workbench spec for a supported campaign or "
            "study type. No project data is read and no run is launched."
        ),
    )
    template.add_argument(
        "template_name",
        choices=available_template_names(),
        help="Starter spec template to emit.",
    )
    template.add_argument(
        "--format",
        dest="output_format",
        choices=("yaml", "json"),
        default="yaml",
        help="Template output format. Default: yaml.",
    )
    template.add_argument("--output", default=None, help="Optional output path.")
    template.set_defaults(func=cmd_template)

    ablations = workbench_sub.add_parser(
        "ablations",
        help="Compare adopted OCScore ablation runs against a reference",
        description=(
            "Scan Workbench result manifests, detect adopted OCScore "
            "train/ablations/<policy> runs, and compare them against an "
            "explicit or auto-selected reference run. No files are modified."
        ),
    )
    ablations.add_argument("root", help="Workspace root or result manifest to inspect.")
    ablations.add_argument(
        "--baseline",
        default=None,
        help="Reference run id. If omitted, a non-ablation train/reference run is selected when available.",
    )
    ablations.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        default=None,
        help="Ablation run id or policy name to compare. May be repeated. If omitted, all detected ablations are compared.",
    )
    ablations.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric in metric or metric:min|max form. May be repeated. If omitted, numeric metrics are inferred.",
    )
    ablations.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    ablations.add_argument("--output", default=None, help="Optional JSON output path.")
    ablations.set_defaults(func=cmd_ablations)

    compare = workbench_sub.add_parser(
        "compare",
        help="Compare result manifests against a baseline run",
        description=(
            "Scan Workbench result manifests below a root path and compare one "
            "baseline run against selected candidates across explicit or inferred "
            "numeric metrics. No files are modified."
        ),
    )
    compare.add_argument("root", help="Workspace root or result manifest to inspect.")
    compare.add_argument(
        "--baseline",
        required=True,
        help="Baseline run id used for comparison.",
    )
    compare.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        default=None,
        help="Candidate run id to compare. May be repeated. If omitted, all non-baseline runs are compared.",
    )
    compare.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric in metric or metric:min|max form. May be repeated. If omitted, numeric metrics are inferred.",
    )
    compare.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    compare.add_argument("--output", default=None, help="Optional JSON output path.")
    compare.set_defaults(func=cmd_compare)

    export = workbench_sub.add_parser(
        "export",
        help="Build a publishable export scaffold without executing runs",
        description=(
            "Read a Workbench run/result manifest and write README.md plus "
            "publication_manifest.json. Artifacts are only copied when "
            "--copy-artifacts is supplied."
        ),
    )
    export.add_argument("manifest", help="Run or result manifest path.")
    export.add_argument("export_dir", help="Directory that will receive export files.")
    export.add_argument(
        "--copy-artifacts",
        action="store_true",
        default=False,
        help="Copy declared existing artifacts into the export artifacts directory.",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing export files and copied artifacts.",
    )
    export.add_argument("--output", default=None, help="Optional JSON summary output path.")
    export.set_defaults(func=cmd_export)

    overview = workbench_sub.add_parser(
        "overview",
        help="Summarize a Workbench workspace for GUI dashboards",
        description=(
            "Build a read-only dashboard payload from Workbench run/result "
            "manifests below a root path. No run is launched or controlled."
        ),
    )
    overview.add_argument("root", help="Workspace root or manifest file to inspect.")
    overview.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    overview.add_argument(
        "--recent-limit",
        type=int,
        default=20,
        help="Maximum recent runs to include. Default: 20.",
    )
    overview.add_argument("--output", default=None, help="Optional JSON output path.")
    overview.set_defaults(func=cmd_overview)

    inventory = workbench_sub.add_parser(
        "inventory",
        help="Summarize Workbench manifests below a root without executing runs",
        description=(
            "Build a read-only JSON inventory from Workbench run/result "
            "manifests below a root path. No run is launched."
        ),
    )
    inventory.add_argument("root", help="Workspace root or manifest file to inspect.")
    inventory.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    inventory.add_argument("--output", default=None, help="Optional JSON output path.")
    inventory.set_defaults(func=cmd_inventory)

    status = workbench_sub.add_parser(
        "status",
        help="Inspect one Workbench run manifest without executing runs",
        description=(
            "Read a Workbench run manifest or prepared bundle directory and emit "
            "a JSON status report. No run is launched or controlled."
        ),
    )
    status.add_argument("target", help="Run manifest path or prepared bundle directory.")
    status.add_argument("--output", default=None, help="Optional JSON output path.")
    status.set_defaults(func=cmd_status)

    launch_plan = workbench_sub.add_parser(
        "launch-plan",
        help="Prepare launch commands for a bundle without executing them",
        description=(
            "Read a Workbench run manifest or prepared bundle and emit foreground "
            "and background shell commands, log paths, and optional script output. "
            "No run is launched."
        ),
    )
    launch_plan.add_argument("target", help="Run manifest path or prepared bundle directory.")
    launch_plan.add_argument(
        "--log-dir",
        default="logs",
        help="Launch log directory, relative to the run workspace by default.",
    )
    launch_plan.add_argument(
        "--script-output",
        default=None,
        help="Optional shell script path to write without executing it.",
    )
    launch_plan.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing launch script when --script-output is supplied.",
    )
    launch_plan.add_argument("--output", default=None, help="Optional JSON output path.")
    launch_plan.set_defaults(func=cmd_launch_plan)

    metrics_catalog = workbench_sub.add_parser(
        "metrics-catalog",
        help="Summarize metric coverage across result manifests",
        description=(
            "Scan result manifests below a root path and summarize metric coverage, "
            "numeric ranges, means, and non-numeric values. No files are modified."
        ),
    )
    metrics_catalog.add_argument("root", help="Workspace root or result manifest to inspect.")
    metrics_catalog.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    metrics_catalog.add_argument("--output", default=None, help="Optional JSON output path.")
    metrics_catalog.set_defaults(func=cmd_metrics_catalog)

    pareto = workbench_sub.add_parser(
        "pareto",
        help="Build a multi-objective Pareto front from result manifests",
        description=(
            "Scan result manifests below a root path and identify non-dominated "
            "runs for one or more metric objectives. No files are modified."
        ),
    )
    pareto.add_argument("root", help="Workspace root or result manifest to inspect.")
    pareto.add_argument(
        "--objective",
        dest="objectives",
        action="append",
        required=True,
        help="Objective in metric or metric:min|max form. May be supplied multiple times.",
    )
    pareto.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    pareto.add_argument("--output", default=None, help="Optional JSON output path.")
    pareto.set_defaults(func=cmd_pareto)

    leaderboard = workbench_sub.add_parser(
        "leaderboard",
        help="Rank Workbench result manifests by a numeric metric",
        description=(
            "Scan result manifests below a root path and rank runs by one numeric "
            "metric. No runs are launched and no result files are modified."
        ),
    )
    leaderboard.add_argument("root", help="Workspace root or result manifest to inspect.")
    leaderboard.add_argument(
        "--metric",
        required=True,
        help="Metric key or dotted metric path to rank, for example auc or validation.auc.",
    )
    leaderboard.add_argument(
        "--mode",
        choices=("max", "min"),
        default="max",
        help="Ranking direction. Use max for larger-is-better metrics, min for lower-is-better metrics.",
    )
    leaderboard.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    leaderboard.add_argument("--output", default=None, help="Optional JSON output path.")
    leaderboard.set_defaults(func=cmd_leaderboard)

    metrics_matrix = workbench_sub.add_parser(
        "metrics-matrix",
        help="Build a plot-ready metric matrix from result manifests",
        description=(
            "Scan result manifests below a root path and emit flattened numeric "
            "metrics for GUI tables, plots, and exports. No files are modified."
        ),
    )
    metrics_matrix.add_argument("root", help="Workspace root or result manifest to inspect.")
    metrics_matrix.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric key or dotted metric path to include. May be supplied multiple times.",
    )
    metrics_matrix.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    metrics_matrix.add_argument("--output", default=None, help="Optional JSON output path.")
    metrics_matrix.set_defaults(func=cmd_metrics_matrix)

    logs = workbench_sub.add_parser(
        "logs",
        help="Preview declared Workbench run logs without streaming unbounded files",
        description=(
            "Read log files declared in a Workbench run manifest or prepared "
            "bundle and emit bounded tail previews as JSON. No run is launched "
            "or controlled."
        ),
    )
    logs.add_argument("target", help="Run manifest path or prepared bundle directory.")
    logs.add_argument(
        "--lines",
        type=int,
        default=80,
        help="Maximum returned lines per log file. Default: 80.",
    )
    logs.add_argument(
        "--max-bytes",
        type=int,
        default=65536,
        help="Maximum bytes read from the end of each log file. Default: 65536.",
    )
    logs.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used to decode log files. Default: utf-8.",
    )
    logs.add_argument("--output", default=None, help="Optional JSON output path.")
    logs.set_defaults(func=cmd_logs)

    results = workbench_sub.add_parser(
        "results",
        help="Summarize Workbench result artifacts and metrics without copying",
        description=(
            "Read a Workbench run or result manifest and emit declared artifact "
            "existence plus metrics. No files are copied or exported."
        ),
    )
    results.add_argument("manifest", help="Run or result manifest path.")
    results.add_argument("--output", default=None, help="Optional JSON output path.")
    results.set_defaults(func=cmd_results)

    plot = workbench_sub.add_parser(
        "plot",
        help="Build Plotly-compatible metric plot payloads",
        description=(
            "Scan Workbench result manifests and emit a Plotly-compatible JSON "
            "figure for leaderboard bars, metric scatter plots, parallel "
            "coordinates, or two-objective Pareto scatter plots. No files are "
            "modified unless --output is supplied."
        ),
    )
    plot.add_argument("root", help="Workspace root or result manifest to inspect.")
    plot.add_argument(
        "--kind",
        choices=("leaderboard", "scatter", "parallel", "pareto"),
        required=True,
        help="Plot payload kind to build.",
    )
    plot.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric name. Use once for leaderboard plots and multiple times for parallel plots.",
    )
    plot.add_argument("--x-metric", default=None, help="Scatter x-axis metric.")
    plot.add_argument("--y-metric", default=None, help="Scatter y-axis metric.")
    plot.add_argument(
        "--color-metric",
        default=None,
        help="Optional scatter marker color metric.",
    )
    plot.add_argument(
        "--objective",
        dest="objectives",
        action="append",
        default=None,
        help="Pareto objective in metric or metric:min|max form. Use exactly two for pareto plots.",
    )
    plot.add_argument(
        "--mode",
        choices=("max", "min"),
        default="max",
        help="Leaderboard ranking direction. Default: max.",
    )
    plot.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    plot.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum leaderboard bars to include. Default: 20.",
    )
    plot.add_argument("--output", default=None, help="Optional JSON output path.")
    plot.set_defaults(func=cmd_plot)

    report = workbench_sub.add_parser(
        "report",
        help="Build a composed analysis report from Workbench manifests",
        description=(
            "Scan Workbench manifests below a root path and compose a GUI-ready "
            "analysis report with overview, metric coverage, leaderboards, an "
            "optional Pareto front, findings, and Markdown text. No run is launched."
        ),
    )
    report.add_argument("root", help="Workspace root or manifest file to inspect.")
    report.add_argument(
        "--leaderboard",
        dest="leaderboards",
        action="append",
        default=None,
        help="Leaderboard metric in metric or metric:min|max form. May be repeated.",
    )
    report.add_argument(
        "--objective",
        dest="objectives",
        action="append",
        default=None,
        help="Pareto objective in metric or metric:min|max form. May be repeated.",
    )
    report.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum directory depth below root to scan. Default: 6.",
    )
    report.add_argument(
        "--recent-limit",
        type=int,
        default=20,
        help="Maximum recent runs to include. Default: 20.",
    )
    report.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Maximum top entries shown in repeated report sections. Default: 5.",
    )
    report.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "markdown"),
        default="json",
        help="Report output format. Default: json.",
    )
    report.add_argument("--output", default=None, help="Optional output path.")
    report.set_defaults(func=cmd_report)

    serve = workbench_sub.add_parser(
        "serve",
        help="Serve a local Workbench API for GUI development",
        description=(
            "Serve Workbench inspection payloads over a local HTTP API. The API "
            "is read-only except for local Optuna dashboard launch/stop helpers. "
            "For SSH workflows, bind to 127.0.0.1 and forward the selected port."
        ),
    )
    serve.add_argument("root", help="Workspace root or run directory to serve.")
    serve.add_argument(
        "--host",
        default=DEFAULT_WORKBENCH_API_HOST,
        help="Host interface to bind. Default: 127.0.0.1.",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WORKBENCH_API_PORT,
        help="TCP port to bind. Default: 8765.",
    )
    serve.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Default directory depth used by scan endpoints. Default: 6.",
    )
    serve.add_argument(
        "--optuna-port-start",
        type=int,
        default=None,
        help="Explicit first Optuna dashboard port. Default: first free port after --port.",
    )
    serve.add_argument(
        "--optuna-port-end",
        type=int,
        default=None,
        help="Explicit last Optuna dashboard port. Requires --optuna-port-start.",
    )
    serve.add_argument(
        "--optuna-slots",
        type=int,
        default=None,
        help=(
            "Override Optuna dashboard slot count when ports are auto-selected. "
            f"Default: baseline replica count, clamped to 1-{MAX_OPTUNA_DASHBOARD_SLOT_COUNT}."
        ),
    )
    serve.add_argument(
        "--verbose",
        action="store_true",
        help="Log each HTTP request to stderr.",
    )
    serve.set_defaults(func=cmd_serve)

    schema = workbench_sub.add_parser(
        "schema",
        help="Emit Workbench JSON Schemas for GUI form generation",
        description=(
            "Emit JSON Schema payloads for Workbench specs, manifests, and "
            "read-only report models. No project data is read or written unless "
            "--output is supplied."
        ),
    )
    schema.add_argument(
        "names",
        nargs="*",
        choices=available_schema_names(),
        help="Optional schema names. If omitted, all schemas are emitted.",
    )
    schema.add_argument("--output", default=None, help="Optional JSON output path.")
    schema.set_defaults(func=cmd_schema)

    plan = workbench_sub.add_parser(
        "plan",
        help="Plan the command for a Workbench spec without executing it",
        description=("Plan the command for a Workbench spec and emit the plan as JSON. No run is launched."),
    )
    plan.add_argument("spec", help="Workbench spec path (.yml, .yaml, or .json).")
    plan.add_argument("--output", default=None, help="Optional JSON output path for the command plan.")
    plan.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used to include a run manifest in the payload.",
    )
    plan.add_argument(
        "--manifest-output",
        default=None,
        help="Optional path to write the run manifest when --run-id is used.",
    )
    plan.add_argument(
        "--ocdocker-executable",
        default="ocdocker",
        help="OCDocker executable used in planned OCScore commands.",
    )
    plan.add_argument(
        "--snakemake-executable",
        default="snakemake",
        help="Snakemake executable used in planned VS campaign commands.",
    )
    plan.set_defaults(func=cmd_plan)


__all__ = [
    "cmd_ablations",
    "cmd_adopt",
    "cmd_adopt_plan",
    "cmd_artifacts",
    "cmd_build",
    "cmd_check",
    "cmd_compare",
    "cmd_evidence",
    "cmd_export",
    "cmd_inventory",
    "cmd_launch_plan",
    "cmd_leaderboard",
    "cmd_metrics_matrix",
    "cmd_pareto",
    "cmd_metrics_catalog",
    "cmd_logs",
    "cmd_overview",
    "cmd_plan",
    "cmd_plot",
    "cmd_report",
    "cmd_results",
    "cmd_schema",
    "cmd_serve",
    "cmd_status",
    "cmd_template",
    "cmd_validate",
    "cmd_workbench",
    "register_subparser",
]
