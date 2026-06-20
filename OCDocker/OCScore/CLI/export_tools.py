#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI tools for exported OCScore best_model bundles.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import OCDocker.OCScore.Analysis.Plotting.CrossValidationPlots as occvplot
import OCDocker.OCScore.Analysis.Plotting.ArchitecturePlots as ocarchplot
import OCDocker.OCScore.Analysis.SHAP.ExportRunner as ocexpshap
import OCDocker.OCScore.Optimization.ModelCrossValidation as occv
import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Utils.ExternalBlindEvaluation as ocextblind
import OCDocker.OCScore.Utils.IO as ocscoreio

REDUCED_DATASET_NAME = "reduced_dataset.csv"
DATASET_COLUMN_CANDIDATES = ["dataset", "source", "db"]
PDBBIND_DATASET_VALUES = {"pdbbind", "pdb-bind", "pdb_bind"}
DUDEZ_DATASET_VALUES = {"dudez", "dude-z", "dude_z"}

# License
###############################################################################
'''OCDocker
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
'''


# Classes
###############################################################################

# Functions
###############################################################################
## Private ##


def _resolve_export_dir(retrain_from: str | None, export_dir: str | None) -> Path:
    '''Resolve an exported best-model directory.

    Parameters
    ----------
    retrain_from : str or None
        Path to a retraining config or export directory.
    export_dir : str or None
        Explicit export directory.

    Returns
    -------
    pathlib.Path
        Resolved export directory path.

    Raises
    ------
    ValueError
        If neither input identifies an export directory.
    '''

    if export_dir:
        return Path(export_dir)
    if retrain_from:
        path = Path(retrain_from)
        if path.is_file():
            return path.parent
        return path
    raise ValueError("Provide --export-dir or --retrain-from.")


def _preview(values: Sequence[Any], *, limit: int = 8) -> dict[str, Any]:
    '''Create a compact preview for a sequence.

    Parameters
    ----------
    values : sequence
        Values to summarize.
    limit : int, default=8
        Maximum number of values to include in the preview.

    Returns
    -------
    dict
        Count, preview values, and omitted-value count.
    '''

    items = list(values or [])
    return {
        "count": len(items),
        "preview": items[:limit],
        "omitted": max(0, len(items) - limit),
    }


def _selected_feature_summary(features: Sequence[str]) -> dict[str, Any]:
    '''Summarize selected feature names without dumping all values.

    Parameters
    ----------
    features : sequence of str
        Selected feature names.

    Returns
    -------
    dict
        Compact selected-feature summary.
    '''

    summary = _preview(features, limit=8)
    return {
        "selected_features_count": summary["count"],
        "selected_features_preview": summary["preview"],
        "selected_features_omitted": summary["omitted"],
    }


def _compact_ocscore_wins(
    summary: Mapping[str, Any], *, limit: int = 8
) -> list[dict[str, Any]]:
    '''Compact OCScore fold-win rows for terminal output.

    Parameters
    ----------
    summary : Mapping
        Cross-validation scorer comparison summary.
    limit : int, default=8
        Maximum number of non-empty rows to return.

    Returns
    -------
    list of dict
        Compact metric win summaries.
    '''

    rows = summary.get("ocscore_wins", []) if summary else []
    compact = []
    for row in rows:
        compared = int(row.get("n_folds_compared", 0) or 0)
        if compared <= 0:
            continue
        compact.append(
            {
                "metric": row.get("metric"),
                "n_folds_won": int(row.get("n_folds_won", 0) or 0),
                "n_folds_compared": compared,
            }
        )
    return compact[:limit]


def _print_summary(payload: Mapping[str, Any]) -> None:
    '''Print a JSON summary payload.

    Parameters
    ----------
    payload : Mapping
        JSON-serializable command summary.
    '''

    print(json.dumps(payload, indent=2, default=str))


def _cmd_validate(args: argparse.Namespace) -> None:
    '''Validate an exported model bundle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed validate command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    result = ocexport.validate_export_bundle(export_path)
    _print_summary({"status": "export_valid", **result})


def _cmd_load(args: argparse.Namespace) -> None:
    '''Load an exported model bundle and print a compact summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed load command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    bundle = ocexport.load_exported_model(export_path, device=args.device)
    summary = bundle["summary"]
    _print_summary(
        {
            "status": "export_loaded",
            "export_dir": bundle["export_dir"],
            "task": summary.get("task"),
            "trial_number": summary.get("trial_number"),
            "objective_metric": summary.get("objective_metric"),
            "final_objective_value": summary.get("final_objective_value"),
            **_selected_feature_summary(bundle["selected_features"]),
            "model_type": type(bundle["model"]).__name__,
        }
    )


def _split_reduced_dataset(reduced, task: str):
    '''Split a merged reduced dataset for one export task.

    Parameters
    ----------
    reduced : pandas.DataFrame
        Reduced dataset containing a dataset/source column.
    task : str
        Export task name.

    Returns
    -------
    pandas.DataFrame
        Rows matching the requested task.

    Raises
    ------
    ValueError
        If the task or dataset/source column is invalid.
    '''

    import pandas as pd

    source_column = next(
        (column for column in DATASET_COLUMN_CANDIDATES if column in reduced.columns),
        None,
    )
    if source_column is None:
        raise ValueError(
            f"{REDUCED_DATASET_NAME!r} requires one dataset/source column from {DATASET_COLUMN_CANDIDATES}."
        )
    normalized = reduced[source_column].astype(str).str.strip().str.lower()
    if task == "pdbbind_regression":
        values = PDBBIND_DATASET_VALUES
    elif task == "dudez_screening":
        values = DUDEZ_DATASET_VALUES
    else:
        raise ValueError(f"Unsupported task for dataset loading: {task}")
    subset = reduced[normalized.isin(values)].copy()
    if subset.empty:
        raise ValueError(
            f"No rows found for task {task!r} in {REDUCED_DATASET_NAME!r}."
        )
    return subset


def _load_dataframe_for_export(export_path: Path, args: argparse.Namespace):
    '''Load reduced data appropriate for an export bundle.

    Parameters
    ----------
    export_path : pathlib.Path
        Export directory containing the retraining config.
    args : argparse.Namespace
        Parsed command arguments with CSV or reduction-archive inputs.

    Returns
    -------
    pandas.DataFrame
        Reduced task-specific dataframe.

    Raises
    ------
    ValueError
        If no usable data input is provided.
    '''

    import pandas as pd

    task = str(
        json.loads(
            (export_path / ocexport.RETRAIN_CONFIG_FILENAME).read_text(encoding="utf-8")
        )["task"]
    )
    if task == "pdbbind_regression" and args.pdbbind_csv:
        return pd.read_csv(args.pdbbind_csv, low_memory=False)
    if task == "dudez_screening" and args.dudez_csv:
        return pd.read_csv(args.dudez_csv, low_memory=False)
    if args.pdbbind_csv and not args.dudez_csv:
        return pd.read_csv(args.pdbbind_csv, low_memory=False)
    if args.dudez_csv and not args.pdbbind_csv:
        return pd.read_csv(args.dudez_csv, low_memory=False)
    if args.pdbbind_csv or args.dudez_csv:
        required = "--pdbbind-csv" if task == "pdbbind_regression" else "--dudez-csv"
        raise ValueError(
            f"Export task {task!r} requires {required} when CSV inputs are provided."
        )
    if not args.reduction_archive:
        raise ValueError(
            "Provide --reduction-archive, --pdbbind-csv, or --dudez-csv for cross-validation."
        )
    source = Path(args.reduction_archive)
    if source.is_dir():
        reduced = pd.read_csv(source / REDUCED_DATASET_NAME, low_memory=False)
    else:
        with tarfile.open(source, "r:*") as archive:
            reduced = pd.read_csv(
                io.BytesIO(archive.extractfile(REDUCED_DATASET_NAME).read()),
                low_memory=False,
            )
    return _split_reduced_dataset(reduced, task)


def _cmd_cross_validate(args: argparse.Namespace) -> None:
    '''Run cross-validation for an exported model bundle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed cross-validation command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    dataframe = _load_dataframe_for_export(export_path, args)
    cv_config = occv.CrossValidationConfig(
        n_folds=int(args.n_folds),
        epochs=int(args.epochs),
        random_seed=int(args.seed),
        shuffle=not args.no_shuffle,
        strategy=args.strategy,
        group_column=args.group_column,
        kind_column=args.kind_column,
        include_scoring_function_baselines=not args.no_scoring_baselines,
        include_descriptor_aggregate_baselines=not getattr(
            args, "no_descriptor_aggregates", False
        ),
        include_sf_consensus_baselines=not getattr(args, "no_sf_consensus", False),
    )
    result = occv.run_cross_validation_from_export(
        export_path,
        dataframe,
        config=cv_config,
        device=args.device,
        output_dir=args.output_dir,
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (export_path / "cross_validation").resolve()
    )
    _print_summary(
        {
            "status": "cross_validation_complete",
            "export_dir": result.export_dir,
            "task": result.task,
            "n_input_rows": int(len(dataframe)),
            "strategy": result.strategy,
            "folds": f"{result.effective_folds}/{result.n_folds}",
            "epochs": result.epochs,
            "objective_metric": result.objective_metric,
            "aggregate_validation_metrics": result.aggregate_validation_metrics,
            "scoring_function_baselines_count": len(result.scoring_function_columns),
            "ocscore_wins_preview": _compact_ocscore_wins(
                result.scorer_comparison_summary
            ),
            "output_dir": str(output_dir),
            "artifacts": {
                "fold_comparison_csv": str(
                    output_dir / "cross_validation_fold_comparison.csv"
                ),
                "scorer_mean_std_csv": str(
                    output_dir / "cross_validation_scorer_mean_std.csv"
                ),
                "ocscore_wins_csv": str(
                    output_dir / "cross_validation_ocscore_wins.csv"
                ),
                "fold_rankings_csv": str(
                    output_dir / "cross_validation_fold_rankings.csv"
                ),
                "per_target_csv": str(
                    output_dir / "cross_validation_per_target_metrics.csv"
                ),
            },
        }
    )


def _cmd_plot(args: argparse.Namespace) -> None:
    '''Render cross-validation plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed plot command arguments.
    '''

    cv_dir = args.cv_dir
    if cv_dir is None:
        export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
        cv_dir = str(export_path / "cross_validation")
    metrics = (
        [item.strip() for item in args.metrics.split(",") if item.strip()]
        if args.metrics
        else None
    )
    resolved_cv = occvplot.resolve_cross_validation_dir(cv_dir)
    figures_dir = (
        Path(args.figures_dir) if args.figures_dir else resolved_cv / "figures"
    )
    written = occvplot.save_cross_validation_figures(
        resolved_cv,
        figures_dir=figures_dir,
        metrics=metrics,
        top_n=args.top_n,
        dpi=int(args.dpi),
    )
    _print_summary(
        {
            "status": "plots_written",
            "figures_dir": str(figures_dir.resolve()),
            "n_plots": len(written),
            "plot_keys": sorted(written),
        }
    )


def _cmd_architecture_plot(args: argparse.Namespace) -> None:
    '''Render architecture plots from an export or architecture file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed architecture-plot command arguments.
    '''

    if args.architecture_file:
        source = Path(args.architecture_file)
        default_dir = source.parent / "figures"
    else:
        export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
        source = export_path
        default_dir = export_path / "figures"
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    document, architecture_path = ocarchplot.load_architecture_document(source)
    written = ocarchplot.save_architecture_figures(
        architecture_path,
        output_dir,
        formats=formats,
        dpi=int(args.dpi),
        title=args.title,
        basename=args.basename,
        include_decoder=args.show_decoder,
    )
    artifacts = {key: value for key, value in written.items() if key != "source"}
    _print_summary(
        {
            "status": "architecture_figures_written",
            "task": document.get("task", "architecture"),
            "architecture_file": str(architecture_path.resolve()),
            "figures_dir": str(output_dir.resolve()),
            "n_figures": len(artifacts),
            "artifacts": artifacts,
        }
    )


def _cmd_shap(args: argparse.Namespace) -> None:
    '''Run SHAP analysis for an exported model bundle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed SHAP command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    dataframe = _load_dataframe_for_export(export_path, args)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (export_path / "shap").resolve()
    )
    result = ocexpshap.run_export_shap_analysis(
        export_path,
        dataframe,
        output_dir,
        device=args.device,
        background_size=args.background_size,
        eval_size=args.eval_size,
        explainer=args.explainer,
        seed=int(args.seed),
        save_csv=not args.no_csv,
    )
    _print_summary(
        {
            "status": "shap_complete",
            "export_dir": str(export_path.resolve()),
            "output_dir": result.out_dir,
            "artifacts": {
                "feature_importance_png": result.feature_importance_png,
                "beeswarm_png": result.beeswarm_png,
                "shap_values_npy": result.shap_values_npy,
                "shap_values_csv": result.shap_values_csv,
            },
        }
    )


def _prepare_raw_dataframe_for_export(export_path: Path, args: argparse.Namespace):
    '''Prepare raw pipeline data for exported-model scoring.

    Parameters
    ----------
    export_path : pathlib.Path
        Export directory containing the retraining config.
    args : argparse.Namespace
        Parsed score command arguments.

    Returns
    -------
    pandas.DataFrame
        Task-specific raw modeling dataframe.

    Raises
    ------
    ValueError
        If the export task is unsupported for scoring.
    '''

    task = str(
        json.loads(
            (export_path / ocexport.RETRAIN_CONFIG_FILENAME).read_text(encoding="utf-8")
        )["task"]
    )
    raw = ocscoreio.load_pipeline_results_from_archive(
        args.raw_archive,
        member_name=args.archive_member,
    )
    if task == "pdbbind_regression":
        return ocscoreio.prepare_pdbbind_dataframe(raw)
    if task == "dudez_screening":
        return ocscoreio.prepare_dudez_dataframe(raw)
    raise ValueError(f"Unsupported export task for scoring: {task}")


def _cmd_external_blind(args: argparse.Namespace) -> None:
    '''Run external blind evaluation for an exported model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed external-blind command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    forbidden_hashes = [
        value.strip()
        for value in (args.forbidden_dataset_hashes or "").split(",")
        if value.strip()
    ]
    report = ocextblind.run_external_blind_evaluation(
        ocextblind.ExternalBlindConfig(
            export_dir=export_path,
            blind_csv=args.blind_csv,
            output_dir=args.output_dir,
            label_column=args.label_column,
            group_column=args.group_column,
            kind_column=args.kind_column,
            device=args.device,
            pdbbind_export_dir=args.pdbbind_export_dir,
            forbidden_dataset_hashes=forbidden_hashes or None,
            command=sys.argv,
        )
    )
    print(json.dumps(report, indent=2))


def _cmd_score(args: argparse.Namespace) -> None:
    '''Score raw pipeline archives with an exported model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed score command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    dataframe = _prepare_raw_dataframe_for_export(export_path, args)
    predictions = ocexport.predict_from_export(
        export_path,
        dataframe,
        device=args.device,
        pdbbind_export_dir=args.pdbbind_export_dir,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    column_summary = _preview(list(predictions.columns), limit=12)
    _print_summary(
        {
            "status": "score_complete",
            "export_dir": str(export_path.resolve()),
            "raw_archive": str(Path(args.raw_archive).resolve()),
            "output_csv": str(output_csv.resolve()),
            "n_predictions": int(len(predictions)),
            "output_columns_count": column_summary["count"],
            "output_columns_preview": column_summary["preview"],
            "output_columns_omitted": column_summary["omitted"],
        }
    )


def _cmd_retrain(args: argparse.Namespace) -> None:
    '''Prepare retraining objects from an exported model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed retrain command arguments.
    '''

    export_path = _resolve_export_dir(args.retrain_from, args.export_dir)
    pdbbind_df = None
    dudez_df = None
    if args.pdbbind_csv:
        import pandas as pd

        pdbbind_df = pd.read_csv(args.pdbbind_csv)
    if args.dudez_csv:
        import pandas as pd

        dudez_df = pd.read_csv(args.dudez_csv)
    payload = ocexport.retrain_from_export(
        export_path,
        pdbbind_df=pdbbind_df,
        dudez_df=dudez_df,
        device=args.device,
        use_saved_split_indices=not args.ignore_saved_splits,
    )
    _print_summary(
        {
            "status": "retrain_smoke_complete",
            "export_dir": str(export_path),
            "task": payload["retrain_config"]["task"],
            **_selected_feature_summary(payload["selected_features"]),
            "train_shape": list(payload["splits"]["X_train"].shape),
            "validation_shape": list(payload["splits"]["X_val"].shape),
            "test_shape": list(payload["splits"]["X_test"].shape),
            "model_type": type(payload["model"]).__name__,
        }
    )


def _shared_export_parser() -> argparse.ArgumentParser:
    '''Build shared exported-model CLI arguments.

    Returns
    -------
    argparse.ArgumentParser
        Parent parser containing export path and torch device arguments.
    '''

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Path to best_model/ export directory",
    )
    parser.add_argument(
        "--retrain-from",
        type=str,
        default=None,
        help="Path to retrain_config.json or best_model/ directory",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device for load/retrain/CV"
    )
    return parser


def _wrap_command(
    handler: Callable[[argparse.Namespace], None],
) -> Callable[[argparse.Namespace], int]:
    '''Wrap a command handler with an integer exit-code return.

    Parameters
    ----------
    handler : callable
        Command function accepting an argparse namespace.

    Returns
    -------
    callable
        Wrapped command function returning an integer exit code.
    '''

    def _wrapped(args: argparse.Namespace) -> int:
        '''Execute a wrapped command handler.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command arguments.

        Returns
        -------
        int
            Zero exit code on success.
        '''

        handler(args)
        return 0

    return _wrapped


def _register_export_commands(
    subparsers: argparse._SubParsersAction, *, required: bool
) -> None:
    '''Register exported-model CLI subcommands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser registry receiving export-tool commands.
    required : bool
        Whether required command arguments should be marked required.
    '''

    shared = _shared_export_parser()

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[shared],
        help="Validate an export bundle",
    )
    validate_parser.set_defaults(func=_wrap_command(_cmd_validate))

    load_parser = subparsers.add_parser(
        "load",
        parents=[shared],
        help="Load an exported model",
    )
    load_parser.set_defaults(func=_wrap_command(_cmd_load))

    retrain_parser = subparsers.add_parser(
        "retrain",
        parents=[shared],
        help="Prepare splits/model for retraining",
    )
    retrain_parser.add_argument("--pdbbind-csv", type=str, default=None)
    retrain_parser.add_argument("--dudez-csv", type=str, default=None)
    retrain_parser.add_argument(
        "--ignore-saved-splits",
        action="store_true",
        help="Recompute splits from split_config instead of saved indices",
    )
    retrain_parser.set_defaults(func=_wrap_command(_cmd_retrain))

    cv_parser = subparsers.add_parser(
        "cross-validate",
        parents=[shared],
        help="Run K-fold cross-validation with fixed exported hyperparameters",
    )
    cv_parser.add_argument(
        "--reduction-archive",
        type=str,
        default=None,
        help="Feature-reduction tar or directory with reduced_dataset.csv",
    )
    cv_parser.add_argument(
        "--pdbbind-csv", type=str, default=None, help="Reduced PDBbind CSV"
    )
    cv_parser.add_argument(
        "--dudez-csv", type=str, default=None, help="Reduced DUDEz CSV"
    )
    cv_parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of cross-validation folds"
    )
    cv_parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs per fold"
    )
    cv_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for folds and training"
    )
    cv_parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=occv.CROSS_VALIDATION_STRATEGIES,
        help="auto=receptor-grouped for DUDEz when receptor column exists, else row K-fold",
    )
    cv_parser.add_argument(
        "--group-column",
        type=str,
        default="receptor",
        help="Receptor column for grouped DUDEz CV",
    )
    cv_parser.add_argument(
        "--kind-column", type=str, default="kind", help="DUDEz kind column"
    )
    cv_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for cross_validation_results.json (default: <export-dir>/cross_validation)",
    )
    cv_parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of fold assignments",
    )
    cv_parser.add_argument(
        "--no-scoring-baselines",
        action="store_true",
        help="Skip per-fold evaluation of individual scoring-function columns (DUDEz only)",
    )
    cv_parser.add_argument(
        "--no-descriptor-aggregates",
        action="store_true",
        help="Skip desc_mean/desc_median/... over all model input features (DUDEz only)",
    )
    cv_parser.add_argument(
        "--no-sf-consensus",
        action="store_true",
        help="Skip sf_mean/sf_median/... across scoring-function columns only (DUDEz only)",
    )
    cv_parser.set_defaults(func=_wrap_command(_cmd_cross_validate))

    plot_parser = subparsers.add_parser(
        "plot",
        parents=[shared],
        help="Plot cross-validation artifacts (PNG figures)",
    )
    plot_parser.add_argument(
        "--cv-dir",
        type=str,
        default=None,
        help="Cross-validation output directory (default: <export-dir>/cross_validation)",
    )
    plot_parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Directory for PNG plots (default: <cv-dir>/figures)",
    )
    plot_parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metrics to plot (default: all comparison metrics)",
    )
    plot_parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Max scoring functions per chart (OCScore always included)",
    )
    plot_parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    plot_parser.set_defaults(func=_wrap_command(_cmd_plot))

    architecture_parser = subparsers.add_parser(
        "architecture-plot",
        parents=[shared],
        help="Render a publication-ready architecture diagram",
    )
    architecture_parser.add_argument(
        "--architecture-file",
        type=str,
        default=None,
        help="Architecture JSON/YAML file. If omitted, uses <export-dir>/architecture.json.",
    )
    architecture_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for architecture figures (default: <export-dir>/figures or <architecture-file>/../figures)",
    )
    architecture_parser.add_argument(
        "--formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats",
    )
    architecture_parser.add_argument(
        "--dpi", type=int, default=220, help="Raster output resolution"
    )
    architecture_parser.add_argument(
        "--title", type=str, default=None, help="Optional figure title"
    )
    architecture_parser.add_argument(
        "--basename", type=str, default="architecture", help="Output filename stem"
    )
    architecture_parser.add_argument(
        "--show-decoder",
        action="store_true",
        help="Include the auxiliary reconstruction decoder branch",
    )
    architecture_parser.set_defaults(func=_wrap_command(_cmd_architecture_plot))

    shap_parser = subparsers.add_parser(
        "shap",
        parents=[shared],
        help="Run SHAP on an exported best_model bundle",
        description=(
            "Pipeline-native SHAP for exported best_model/ bundles "
            "(validation background, test evaluation)."
        ),
    )
    shap_parser.add_argument(
        "--reduction-archive",
        type=str,
        default=None,
        help="Feature-reduction tar or directory with reduced_dataset.csv",
    )
    shap_parser.add_argument(
        "--pdbbind-csv", type=str, default=None, help="Reduced PDBbind CSV"
    )
    shap_parser.add_argument(
        "--dudez-csv", type=str, default=None, help="Reduced DUDEz CSV"
    )
    shap_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for SHAP artifacts (default: <export-dir>/shap)",
    )
    shap_parser.add_argument(
        "--background-size",
        type=int,
        default=None,
        help="Validation background sample size",
    )
    shap_parser.add_argument(
        "--eval-size", type=int, default=None, help="Test evaluation sample size"
    )
    shap_parser.add_argument(
        "--explainer",
        type=str,
        default="gradient",
        choices=("gradient", "deep", "kernel", "permutation"),
        help="SHAP explainer type",
    )
    shap_parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for SHAP subsampling"
    )
    shap_parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing shap_values.csv",
    )
    shap_parser.set_defaults(func=_wrap_command(_cmd_shap))

    score_parser = subparsers.add_parser(
        "score",
        parents=[shared],
        help="Score raw pipeline archives with an exported best_model bundle",
    )
    score_parser.add_argument(
        "--raw-archive",
        type=str,
        required=required,
        help=(
            "Raw pipeline input: a .csv file, directory, or tar.gz containing "
            "pipeline_results.csv, PDBbind.csv, or DUDEz.csv"
        ),
    )
    score_parser.add_argument(
        "--output-csv",
        type=str,
        required=required,
        help="Path for predictions CSV output",
    )
    score_parser.add_argument(
        "--pdbbind-export-dir",
        type=str,
        default=None,
        help="Linked PDBbind export directory for DUDEz transfer models",
    )
    score_parser.add_argument(
        "--archive-member",
        type=str,
        default=None,
        help="Explicit tar member path when multiple pipeline CSV files exist",
    )
    score_parser.set_defaults(func=_wrap_command(_cmd_score))

    external_blind_parser = subparsers.add_parser(
        "external-blind",
        parents=[shared],
        help="One-shot external blind evaluation with frozen export artifacts",
    )
    external_blind_parser.add_argument(
        "--blind-csv",
        type=str,
        required=required,
        help="External blind dataset CSV (wide features; extra columns ignored).",
    )
    external_blind_parser.add_argument(
        "--output-dir",
        type=str,
        required=required,
        help="Directory for external_blind_evaluation.json and predictions CSV.",
    )
    external_blind_parser.add_argument("--label-column", type=str, default=None)
    external_blind_parser.add_argument("--group-column", type=str, default="receptor")
    external_blind_parser.add_argument("--kind-column", type=str, default="kind")
    external_blind_parser.add_argument(
        "--forbidden-dataset-hashes",
        type=str,
        default=None,
        help="Comma-separated SHA-256 hashes that must not match the blind CSV.",
    )
    external_blind_parser.add_argument(
        "--pdbbind-export-dir",
        type=str,
        default=None,
        help="Linked PDBbind export directory for DUDEz transfer models.",
    )
    external_blind_parser.set_defaults(func=_wrap_command(_cmd_external_blind))


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    '''Register exported-model commands on the OCScore CLI.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        OCScore subparser registry.
    '''

    _register_export_commands(subparsers, required=True)


def _build_parser() -> argparse.ArgumentParser:
    '''Build the standalone exported-model tools parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser containing all exported-model tool subcommands.
    '''

    parser = argparse.ArgumentParser(description="OCScore exported model tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _register_export_commands(subparsers, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    '''Run exported-model tool CLI dispatch.

    Parameters
    ----------
    argv : list of str, optional
        Optional argument vector for tests or programmatic use.

    Returns
    -------
    int
        Process-style exit code.
    '''

    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result) if result is not None else 0
