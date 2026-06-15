#!/usr/bin/env python3

"""CLI for staged OCScore Optuna from raw unreduced modeling inputs.

The train command loads raw pipeline tables, creates a fixed outer split, fits
train-only feature reduction on PDBbind training rows, and runs replicated
staged Optuna on the frozen feature set.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged
import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Analysis.Metrics.Calibration import build_calibration_report_section
from OCDocker.OCScore.Analysis.Metrics.Calibration import validate_calibration_report_mode
from OCDocker.OCScore.Analysis.ProductionBaselines import ProductionBaselineConfig
from OCDocker.OCScore.Analysis.ProductionBaselines import run_and_write_production_baselines
from OCDocker.OCScore.Optimization.StagedTrainProtocol import StagedTrainProtocol
from OCDocker.OCScore.Optimization.StagedTrainProtocol import load_staged_train_protocol
from OCDocker.OCScore.Optimization.StagedTrainProtocol import resolve_protocol_path
from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.DUDEzScaling import DUDEzScalingConfig
from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_SUMMARY_CSV
from OCDocker.OCScore.Utils.FeaturePolicy import FEATURE_POLICY_SUMMARY_JSON
from OCDocker.OCScore.Utils.FeaturePolicy import FULL_OCSCORE_POLICY_NAME
from OCDocker.OCScore.Utils.FeaturePolicy import FeaturePolicy
from OCDocker.OCScore.Utils.FeaturePolicy import apply_feature_policy
from OCDocker.OCScore.Utils.FeaturePolicy import discover_candidate_model_features
from OCDocker.OCScore.Utils.FeaturePolicy import resolve_requested_feature_policies
from OCDocker.OCScore.Utils.FeaturePolicy import write_feature_policy_metadata
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FEATURE_SELECTION_JSON
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FeatureSelectionScope
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import load_feature_selection_json
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import validate_train_only_feature_selection
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import verify_selected_features_against_scope
from OCDocker.OCScore.Utils.DUDEzSplit import DUDEzSplitConfig
from OCDocker.OCScore.Utils.DUDEzSplit import split_dudez_by_receptor_and_kind
from OCDocker.OCScore.Utils.FixedOuterSplit import FixedOuterSplitAssignment
from OCDocker.OCScore.Utils.FixedOuterSplit import build_fixed_outer_split_assignment
from OCDocker.OCScore.Utils.FixedOuterSplit import validate_protocol_integrity
from OCDocker.OCScore.Utils.RawModelingInput import RawModelingInput
from OCDocker.OCScore.Utils.RawModelingInput import FORBIDDEN_TRAINING_ARTIFACTS
from OCDocker.OCScore.Utils.RawModelingInput import load_raw_modeling_input
from OCDocker.OCScore.Utils.RawModelingInput import reject_precomputed_training_artifacts
from OCDocker.OCScore.Utils.FixedOuterSplit import write_fixed_outer_split_json
from OCDocker.OCScore.Utils.PDBbindSplit import PDBbindSplitConfig
from OCDocker.OCScore.Utils.PDBbindSplit import split_pdbbind_regression
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import apply_frozen_feature_selection
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import drop_nonfinite_selected_feature_rows
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import feature_reduction_config_for_feature_blocks
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import fit_train_only_feature_reduction
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import split_wide_dataset_by_column
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import write_train_only_reduction_artifact
from OCDocker.OCScore.Utils.LeakageAudit import run_leakage_audit
from OCDocker.OCScore.Utils.LeakageAudit import write_leakage_audit_report
from OCDocker.OCScore.Utils.ProtocolProvenance import build_split_assignments_payload
from OCDocker.OCScore.Utils.ProtocolProvenance import write_production_provenance_bundle

from OCDocker.OCScore.Optimization.Protocol import ProtocolContext
from OCDocker.OCScore.Optimization.Protocol import ReplicatedProtocolResult
from OCDocker.OCScore.Optimization.Protocol import ReplicatedStagedProtocol

# License
###############################################################################
"""
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
"""

LOGGER = oclogging.get_logger("ocscore.cli.train")

REDUCED_PDBBIND_NAME = "reduced_pdbbind.csv"
REDUCED_DUDEZ_NAME = "reduced_dudez.csv"
REDUCED_DATASET_NAME = "reduced_dataset.csv"
SELECTED_FEATURES_JSON = "selected_features.json"
SELECTED_FEATURES_TXT = "selected_features.txt"
FEATURE_REDUCTION_PROTOCOL_JSON = "feature_reduction_protocol.json"
MERGED_INPUT_DATASET_NAME = "merged_input_dataset.csv"
LABEL_COLUMN = "label"
DATASET_COLUMN_CANDIDATES = ["dataset", "source", "db"]
PDBBIND_DATASET_VALUES = {"pdbbind", "pdbbind_refined", "pdbbind_general"}
DUDEZ_DATASET_VALUES = {"dudez", "dude-z", "dude_z"}
OPTIONAL_FEATURE_REDUCTION_ARTIFACTS = [
    FEATURE_REDUCTION_PROTOCOL_JSON,
    "feature_reduction_protocol.md",
    FEATURE_SELECTION_JSON,
    "config_used.json",
    "block_summary.csv",
    "dropped_features.csv",
    "dropped_rows_missing_values.csv",
    "missingness_by_column.csv",
    "missingness_by_block.csv",
    "intra_block_correlation_report.csv",
    "cross_block_pairwise_correlation_report.csv",
    "cross_block_predictability_report.csv",
    "cross_block_filter_report.csv",
]
NON_FEATURE_COLUMNS = list(ocstaged.OCSCORE_NON_FEATURE_COLUMNS)
FULL_FEATURE_BLOCKS = ("ligand", "receptor", "scoring")
ABLATION_FEATURE_BLOCKS = {
    "ligand_only": ("ligand",),
    "sf_only": ("scoring",),
    "ligand_sf": ("ligand", "scoring"),
    "receptor_sf": ("receptor", "scoring"),
}


# Classes
###############################################################################

@dataclass
class ReductionArtifacts:
    """Modeling artifacts produced after train-only feature reduction.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        PDBbind dataframe with frozen train-only selected features applied.
    dudez_df : pd.DataFrame
        DUDEz dataframe with frozen train-only selected features applied.
    selected_features : list[str]
        Selected feature columns from train-only reduction.
    artifact_paths : dict[str, str]
        Paths to written modeling and feature-selection artifacts.
    extracted_dir : pathlib.Path
        Base directory associated with the loaded or written modeling inputs.
    feature_selection : FeatureSelectionScope | None, optional
        Train-only feature-selection metadata.
    fixed_outer_split : FixedOuterSplitAssignment | None, optional
        Fixed outer split shared by all replicas.
    row_cleanup_summary : dict | None, optional
        Summary of selected-feature row cleanup before modeling.
    feature_policy_metadata : dict | None, optional
        Feature-policy provenance for this reduction pass.
    """

    pdbbind_df: pd.DataFrame
    dudez_df: pd.DataFrame
    selected_features: list[str]
    artifact_paths: dict[str, str]
    extracted_dir: Path
    feature_selection: Optional[FeatureSelectionScope] = None
    fixed_outer_split: Optional[FixedOuterSplitAssignment] = None
    row_cleanup_summary: Optional[dict[str, Any]] = None
    feature_policy_metadata: Optional[dict[str, Any]] = None


@dataclass
class StagedTrainingRun:
    """Result bundle for one full or ablation staged-training pass."""

    artifacts: ReductionArtifacts
    pdbbind_df: pd.DataFrame
    dudez_df: pd.DataFrame
    context: ProtocolContext
    metadata: dict[str, Any]
    result: ReplicatedProtocolResult
    written: dict[str, str]
    protocol_metadata: dict[str, Any]
    replica_alignments: list[dict[str, Any]]


# Functions
###############################################################################
## Private ##

def _as_jsonable(value: Any) -> Any:
    '''Convert common scientific Python values for JSON output.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-compatible representation.
    '''

    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _as_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_as_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_as_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_jsonable(item) for item in value]
    return str(value)


def _artifact_id(source: str | Path, member_name: str) -> str:
    '''Return a stable identifier for an archive member.

    Parameters
    ----------
    source : str or pathlib.Path
        Archive path.
    member_name : str
        Archive member name.

    Returns
    -------
    str
        Human-readable archive member identifier.
    '''

    return f"{Path(source)}::{member_name}"


def _is_tar_source(source: str | Path) -> bool:
    '''Return True when the source is a readable tar archive.

    Parameters
    ----------
    source : str or pathlib.Path
        Candidate feature-reduction source.

    Returns
    -------
    bool
        Whether ``source`` is a tar-compatible archive.
    '''

    path = Path(source)
    return path.is_file() and tarfile.is_tarfile(path)


def _safe_tar_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    '''Validate and return safe tar members.

    Parameters
    ----------
    tar : tarfile.TarFile
        Open tar archive.

    Returns
    -------
    list[tarfile.TarInfo]
        Safe tar members.

    Raises
    ------
    ValueError
        If the archive contains absolute paths, path traversal, links, or
        special file entries.
    '''

    safe_members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe tar member path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsafe tar member type: {member.name}")
        if member.isfile():
            safe_members.append(member)
    return safe_members


def _find_tar_member(members: Sequence[tarfile.TarInfo], filename: str) -> Optional[tarfile.TarInfo]:
    '''Find one regular tar member by basename.

    Parameters
    ----------
    members : Sequence[tarfile.TarInfo]
        Safe archive members.
    filename : str
        Basename to locate.

    Returns
    -------
    tarfile.TarInfo | None
        First matching member, or None if absent.
    '''

    matches = sorted(
        (member for member in members if Path(member.name).name == filename),
        key=lambda member: member.name,
    )
    return matches[0] if matches else None


def _read_tar_member_text(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    '''Read a tar member as UTF-8 text.

    Parameters
    ----------
    tar : tarfile.TarFile
        Open tar archive.
    member : tarfile.TarInfo
        Regular file member to read.

    Returns
    -------
    str
        Decoded text.
    '''

    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with fileobj:
        return fileobj.read().decode("utf-8")


def _read_tar_member_csv(tar: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
    '''Read a CSV tar member directly with pandas.

    Parameters
    ----------
    tar : tarfile.TarFile
        Open tar archive.
    member : tarfile.TarInfo
        Regular CSV member to read.

    Returns
    -------
    pd.DataFrame
        Loaded CSV dataframe.
    '''

    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with fileobj:
        return pd.read_csv(fileobj, low_memory=False)


def _write_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    '''Copy a regular tar member to a target file.

    Parameters
    ----------
    tar : tarfile.TarFile
        Open tar archive.
    member : tarfile.TarInfo
        Regular file member to copy.
    target : pathlib.Path
        Output file path.
    '''

    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise ValueError(f"Could not copy archive member: {member.name}")
    with fileobj, target.open("wb") as handle:
        shutil.copyfileobj(fileobj, handle)


def _find_one_file(base_dir: Path, filename: str) -> Optional[Path]:
    '''Find one file by basename under a directory.

    Parameters
    ----------
    base_dir : pathlib.Path
        Directory to search.
    filename : str
        Basename to locate.

    Returns
    -------
    pathlib.Path | None
        First matching file, or None if no file was found.
    '''

    matches = sorted(path for path in base_dir.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def _log(message: str) -> None:
    '''Print a concise run progress message.

    Parameters
    ----------
    message : str
        Message to emit.
    '''

    print(message, flush=True)


def _compact_paths(paths: dict[str, Any]) -> str:
    '''Return a compact ``key=path`` summary for output artifacts.'''

    compact = {key: value for key, value in paths.items() if value}
    if not compact:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(compact.items()))


def _replica_name(replica_summary: Any) -> str:
    '''Return the replica name from an aggregate best-replica payload.'''

    if isinstance(replica_summary, dict):
        name = replica_summary.get("replica_name")
        if name:
            return str(name)
    return "none"


def _normalize_kind(value: Any) -> str:
    '''Normalize a DUDEz kind value.

    Parameters
    ----------
    value : Any
        Raw kind value.

    Returns
    -------
    str
        Lowercase stripped kind value.
    '''

    return str(value).strip().lower()


def _normalize_source(value: Any) -> str:
    '''Normalize a dataset/source value.

    Parameters
    ----------
    value : Any
        Raw dataset/source value.

    Returns
    -------
    str
        Lowercase normalized dataset/source value.
    '''

    return str(value).strip().lower().replace(" ", "_")


def _write_metrics(stage_result: dict[str, Any], output_dir: Path, prefix: str) -> dict[str, str]:
    '''Write stage metrics as JSON and CSV.

    Parameters
    ----------
    stage_result : dict[str, Any]
        Stage result dictionary produced by the staged Optuna protocol.
    output_dir : pathlib.Path
        Directory where metric files are written.
    prefix : str
        File prefix, for example ``"pdbbind"`` or ``"dudez"``.

    Returns
    -------
    dict[str, str]
        Written metric file paths.
    '''

    payload = {
        "objective_metric": stage_result.get("objective_metric"),
        "best_value": stage_result.get("best_value"),
        "validation_metrics": stage_result.get("validation_metrics", {}),
        "test_metrics": stage_result.get("test_metrics", {}),
    }
    json_path = output_dir / f"{prefix}_metrics.json"
    csv_path = output_dir / f"{prefix}_metrics.csv"
    json_path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for split_name in ["validation_metrics", "test_metrics"]:
        for metric, value in stage_result.get(split_name, {}).items():
            rows.append({
                "split": split_name.replace("_metrics", ""),
                "metric": metric,
                "value": value,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def _format_mean_std(metric: dict[str, Any]) -> str:
    '''Format a mean/std aggregate metric for Markdown output.

    Parameters
    ----------
    metric : dict[str, Any]
        Metric summary with ``mean`` and ``std`` keys.

    Returns
    -------
    str
        Compact Markdown-ready metric text.
    '''

    mean = metric.get("mean")
    std = metric.get("std")
    if mean is None:
        return "not available"
    if std is None:
        return f"{mean:.6g}"
    return f"{mean:.6g} +/- {std:.6g}"


def _write_protocol_markdown(summary: dict[str, Any], output_path: Path) -> None:
    '''Write a compact Markdown summary for the example run.

    Parameters
    ----------
    summary : dict[str, Any]
        JSON-compatible protocol summary.
    output_path : pathlib.Path
        Markdown output path.
    '''

    aggregate = summary.get("aggregate_summary", {})
    metrics = aggregate.get("metrics", {})
    best_pdbbind = aggregate.get("best_pdbbind_replica") or {}
    best_dudez = aggregate.get("best_dudez_replica") or {}
    output_paths = summary.get("replicated_output_paths", {})
    lines = [
        "# OCScore Replicated Staged Optuna Protocol",
        "",
        "## Inputs",
        "",
        f"- Reduction archive: `{summary.get('reduction_archive')}`",
        f"- Reduction source: `{summary.get('reduction_source')}`",
        f"- Selected features: {summary.get('static_context', {}).get('n_selected_features')}",
        "",
        "## Replicas",
        "",
        f"- Replicas requested: {summary.get('n_replicas')}",
        f"- Successful replicas: {aggregate.get('n_successful_replicas')}",
        f"- Failed replicas: {aggregate.get('n_failed_replicas')}",
        "",
        "## Objectives",
        "",
        "- PDBbind: validation RMSE, minimize",
        f"- DUDEz: {best_dudez.get('dudez_primary_metric', 'BEDROC')}, maximize",
        "",
        "## Aggregate Metrics (scientific headline)",
        "",
        f"- PDBbind validation RMSE: {_format_mean_std(metrics.get('pdbbind_validation_rmse', {}))}",
        f"- PDBbind test RMSE: {_format_mean_std(metrics.get('pdbbind_test_rmse', {}))}",
        f"- DUDEz validation primary metric: {_format_mean_std(metrics.get('dudez_validation_primary_metric', {}))}",
        f"- DUDEz test PR-AUC: {_format_mean_std(metrics.get('dudez_test_pr_auc', {}))}",
        f"- DUDEz test BEDROC: {_format_mean_std(metrics.get('dudez_test_bedroc', {}))}",
        "",
        "## Reporting policy",
        "",
        f"- Primary claim: ranking/screening performance (`{summary.get('primary_claim', 'ranking_screening')}`)",
        f"- Calibration report mode: `{summary.get('calibration_report_mode', 'ranking_only')}`",
    ]
    if summary.get("calibration_report_mode", "ranking_only") == "ranking_only":
        lines.append(
            "- Post-hoc calibration metrics are diagnostic-only (`diagnostic_*` keys); "
            "not validated probability estimates."
        )
    lines.extend(
        [
        "",
        "## Export Candidates (deployment selection only)",
        "",
        f"- Best PDBbind replica: `{best_pdbbind.get('replica_name')}`",
        f"- Best DUDEz replica: `{best_dudez.get('replica_name')}`",
        "",
        "## Outputs",
        "",
        f"- Replicas summary: `{output_paths.get('replicas_summary_csv')}`",
        f"- Replicas protocol: `{output_paths.get('replicas_protocol_json')}`",
        "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


## Public ##

def add_arguments(parser: argparse.ArgumentParser) -> None:
    '''Register ``ocscore train`` command-line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser or subparser to extend.
    '''

    parser.add_argument(
        "--protocol",
        required=True,
        help=(
            "Path to a staged training protocol .yml file, or a bundled protocol name "
            "(production, smoke-test, development, example)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where staged Optuna outputs are written.",
    )
    parser.add_argument(
        "--merged-input",
        help="Path to a merged raw unreduced modeling CSV (for example merged_input_dataset.csv).",
    )
    parser.add_argument(
        "--pdbbind-input",
        help="Path to a raw unreduced PDBbind pipeline CSV or archive.",
    )
    parser.add_argument(
        "--dudez-input",
        help="Path to a raw unreduced DUDEz pipeline CSV or archive.",
    )
    parser.add_argument(
        "--raw-input-dir",
        help=(
            "Directory or tar archive containing merged_input_dataset.csv "
            "or separate raw PDBbind/DUDEz CSV files from ocscore reduce."
        ),
    )
    parser.add_argument(
        "--feature-policy",
        action="append",
        default=None,
        help=(
            "Feature-ablation policy name to run. May be repeated. "
            "Bundled policies are discovered from OCDocker/OCScore/Protocols/Ablations/. "
            "Defaults to full_ocscore when no feature policy is supplied. "
            "Focused bundled policies include no_shape_core_no_receptor_length_pair, "
            "no_shape_core_no_receptor_surface_counts, no_shape_core_no_receptor_surface_size, "
            "ligand_plus_scoring_function_no_shape_core, and "
            "ligand_plus_scoring_function_no_shape_size."
        ),
    )
    parser.add_argument(
        "--feature-policy-dir",
        action="append",
        default=None,
        help=(
            "Additional directory containing one feature-policy .yml file per policy. "
            "May be repeated; duplicate policy names fail."
        ),
    )
    parser.add_argument(
        "--feature-policy-yml",
        action="append",
        default=None,
        help=(
            "Explicit one-off feature-policy .yml file. Participates in duplicate-name "
            "conflict detection."
        ),
    )
    parser.add_argument(
        "--run-all-feature-policies",
        action="store_true",
        help="Run all discovered bundled, custom-directory, and explicit feature policies.",
    )
    parser.add_argument(
        "--reduction-archive",
        help=argparse.SUPPRESS,
    )


def build_argparser() -> argparse.ArgumentParser:
    '''Build the ``ocscore train`` command-line parser.'''

    parser = argparse.ArgumentParser(
        description=(
            "Run staged OCScore Optuna from raw unreduced inputs. "
            "Feature cleaning/reduction is fit only on training rows after splitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``train`` subcommand on the ``ocscore`` parser.'''

    parser = subparsers.add_parser(
        "train",
        help="Staged OCScore Optuna from raw unreduced modeling inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    parser.set_defaults(func=cmd_train)


def cmd_train(args: argparse.Namespace) -> int:
    '''Dispatch handler for ``ocscore train``.'''

    return main_from_args(args)


def resolve_reduction_source(archive_path: str | Path) -> Path:
    '''Validate and return a feature-reduction source path.

    Directory inputs are read in place; tar/tar.gz inputs are read member-by-member
    by pandas and JSON/text loaders.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Feature-reduction archive or directory path.

    Returns
    -------
    pathlib.Path
        Validated source path.

    Raises
    ------
    FileNotFoundError
        If the source path does not exist.
    ValueError
        If the source is neither a directory nor a tar archive.
    '''

    source = Path(archive_path)
    if not source.exists():
        raise FileNotFoundError(f"Reduction source not found: {source}")
    if source.is_dir() or _is_tar_source(source):
        return source
    raise ValueError(f"Reduction source must be a directory or tar archive: {source}")


def _load_selected_features_from_directory(source_dir: Path) -> tuple[list[str], str]:
    '''Load selected features from an unpacked reduction directory.

    Parameters
    ----------
    source_dir : pathlib.Path
        Directory containing feature-reduction outputs.

    Returns
    -------
    tuple[list[str], str]
        Selected feature list and source path string.
    '''

    json_path = _find_one_file(source_dir, SELECTED_FEATURES_JSON)
    txt_path = _find_one_file(source_dir, SELECTED_FEATURES_TXT)

    if json_path is not None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source = str(json_path)
    elif txt_path is not None:
        selected = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        source = str(txt_path)
        if not selected:
            raise ValueError("Selected feature list is empty.")
        return selected, source
    else:
        raise FileNotFoundError(
            f"Expected {SELECTED_FEATURES_JSON!r} or {SELECTED_FEATURES_TXT!r} in {source_dir}."
        )

    return _parse_selected_features_payload(payload, source), source


def _load_selected_features_from_tar(source: Path) -> tuple[list[str], str]:
    '''Load selected features directly from a tar archive.

    Parameters
    ----------
    source : pathlib.Path
        Tar archive containing feature-reduction outputs.

    Returns
    -------
    tuple[list[str], str]
        Selected feature list and archive member identifier.
    '''

    with tarfile.open(source, "r:*") as tar:
        members = _safe_tar_members(tar)
        json_member = _find_tar_member(members, SELECTED_FEATURES_JSON)
        txt_member = _find_tar_member(members, SELECTED_FEATURES_TXT)
        if json_member is not None:
            payload = json.loads(_read_tar_member_text(tar, json_member))
            source_id = _artifact_id(source, json_member.name)
            return _parse_selected_features_payload(payload, source_id), source_id
        if txt_member is not None:
            selected = [line.strip() for line in _read_tar_member_text(tar, txt_member).splitlines() if line.strip()]
            if not selected:
                raise ValueError("Selected feature list is empty.")
            return selected, _artifact_id(source, txt_member.name)
    raise FileNotFoundError(
        f"Expected {SELECTED_FEATURES_JSON!r} or {SELECTED_FEATURES_TXT!r} in {source}."
    )


def _parse_selected_features_payload(payload: Any, source: str) -> list[str]:
    '''Parse a selected-features JSON payload.

    Parameters
    ----------
    payload : Any
        JSON payload read from selected_features.json.
    source : str
        Source identifier used in error messages.

    Returns
    -------
    list[str]
        Selected feature names.
    '''

    if isinstance(payload, dict):
        payload = payload.get("selected_features", payload.get("features"))
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a JSON list or a dictionary with selected_features.")
    selected = [str(item) for item in payload]
    if not selected:
        raise ValueError("Selected feature list is empty.")
    return selected


def load_selected_features(extracted_dir: str | Path) -> tuple[list[str], Path | str]:
    '''Load selected features from a reduction directory or tar archive.

    Parameters
    ----------
    extracted_dir : str or pathlib.Path
        Directory or tar archive containing feature-reduction outputs.

    Returns
    -------
    tuple[list[str], pathlib.Path | str]
        Selected feature list and source identifier.
    '''

    source = Path(extracted_dir)
    if source.is_dir():
        return _load_selected_features_from_directory(source)
    if _is_tar_source(source):
        return _load_selected_features_from_tar(source)
    raise ValueError(f"Reduction source must be a directory or tar archive: {source}")


def _split_reduced_dataset(reduced: pd.DataFrame, paths: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    '''Split a combined reduced dataset into PDBbind and DUDEz rows.

    Parameters
    ----------
    reduced : pd.DataFrame
        Combined reduced dataset.
    paths : dict[str, str]
        Artifact path metadata to update.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]
        PDBbind dataframe, DUDEz dataframe, and artifact path metadata.
    '''

    source_column = next((column for column in DATASET_COLUMN_CANDIDATES if column in reduced.columns), None)
    if source_column is None:
        raise ValueError(
            f"{REDUCED_DATASET_NAME!r} requires one dataset/source column from {DATASET_COLUMN_CANDIDATES}."
        )

    normalized_source = reduced[source_column].map(_normalize_source)
    pdbbind = reduced[normalized_source.isin(PDBBIND_DATASET_VALUES)].copy()
    dudez = reduced[normalized_source.isin(DUDEZ_DATASET_VALUES)].copy()
    if pdbbind.empty or dudez.empty:
        raise ValueError(
            f"Could not split {REDUCED_DATASET_NAME!r} by {source_column!r}; "
            f"found PDBbind rows={len(pdbbind)}, DUDEz rows={len(dudez)}."
        )

    paths["source_column"] = source_column
    return pdbbind, dudez, paths


def _load_reduced_datasets_from_directory(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    '''Load reduced datasets from an unpacked directory.

    Parameters
    ----------
    source_dir : pathlib.Path
        Directory containing feature-reduction outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]
        PDBbind dataframe, DUDEz dataframe, and artifact path metadata.
    '''

    pdbbind_path = _find_one_file(source_dir, REDUCED_PDBBIND_NAME)
    dudez_path = _find_one_file(source_dir, REDUCED_DUDEZ_NAME)
    paths: dict[str, str] = {}

    if pdbbind_path is not None and dudez_path is not None:
        paths["reduced_pdbbind"] = str(pdbbind_path)
        paths["reduced_dudez"] = str(dudez_path)
        return pd.read_csv(pdbbind_path, low_memory=False), pd.read_csv(dudez_path, low_memory=False), paths

    reduced_path = _find_one_file(source_dir, REDUCED_DATASET_NAME)
    if reduced_path is None:
        raise FileNotFoundError(
            f"Expected {REDUCED_PDBBIND_NAME!r} + {REDUCED_DUDEZ_NAME!r}, or {REDUCED_DATASET_NAME!r}."
        )

    paths["reduced_dataset"] = str(reduced_path)
    return _split_reduced_dataset(pd.read_csv(reduced_path, low_memory=False), paths)


def _load_reduced_datasets_from_tar(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    '''Load reduced datasets directly from a tar archive.

    Parameters
    ----------
    source : pathlib.Path
        Tar archive containing feature-reduction outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]
        PDBbind dataframe, DUDEz dataframe, and artifact path metadata.
    '''

    with tarfile.open(source, "r:*") as tar:
        members = _safe_tar_members(tar)
        pdbbind_member = _find_tar_member(members, REDUCED_PDBBIND_NAME)
        dudez_member = _find_tar_member(members, REDUCED_DUDEZ_NAME)
        paths: dict[str, str] = {}

        if pdbbind_member is not None and dudez_member is not None:
            paths["reduced_pdbbind"] = _artifact_id(source, pdbbind_member.name)
            paths["reduced_dudez"] = _artifact_id(source, dudez_member.name)
            return _read_tar_member_csv(tar, pdbbind_member), _read_tar_member_csv(tar, dudez_member), paths

        reduced_member = _find_tar_member(members, REDUCED_DATASET_NAME)
        if reduced_member is None:
            raise FileNotFoundError(
                f"Expected {REDUCED_PDBBIND_NAME!r} + {REDUCED_DUDEZ_NAME!r}, or {REDUCED_DATASET_NAME!r}."
            )

        paths["reduced_dataset"] = _artifact_id(source, reduced_member.name)
        return _split_reduced_dataset(_read_tar_member_csv(tar, reduced_member), paths)


def load_reduced_datasets(extracted_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    '''Load reduced PDBbind and DUDEz datasets from a directory or tar archive.

    Parameters
    ----------
    extracted_dir : str or pathlib.Path
        Directory or tar archive containing feature-reduction outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]
        Reduced PDBbind dataframe, reduced DUDEz dataframe, and source paths.
    '''

    source = Path(extracted_dir)
    if source.is_dir():
        return _load_reduced_datasets_from_directory(source)
    if _is_tar_source(source):
        return _load_reduced_datasets_from_tar(source)
    raise ValueError(f"Reduction source must be a directory or tar archive: {source}")


def _find_protocol_artifact(source: str | Path) -> Optional[str]:
    '''Find the feature-reduction protocol artifact identifier.

    Parameters
    ----------
    source : str or pathlib.Path
        Directory or tar archive containing feature-reduction outputs.

    Returns
    -------
    str | None
        Artifact path/member identifier if present.
    '''

    path = Path(source)
    if path.is_dir():
        protocol_path = _find_one_file(path, FEATURE_REDUCTION_PROTOCOL_JSON)
        return str(protocol_path) if protocol_path is not None else None
    if _is_tar_source(path):
        with tarfile.open(path, "r:*") as tar:
            member = _find_tar_member(_safe_tar_members(tar), FEATURE_REDUCTION_PROTOCOL_JSON)
            return _artifact_id(path, member.name) if member is not None else None
    return None


def load_reduction_artifacts(extracted_dir: str | Path) -> ReductionArtifacts:
    '''Reject deprecated global/precomputed reduction archives for training.

    Parameters
    ----------
    extracted_dir : str or pathlib.Path
        Legacy reduction directory or archive path.

    Returns
    -------
    ReductionArtifacts
        Never returned; retained for backward-compatible import paths in tests.

    Raises
    ------
    ValueError
        If the source contains forbidden precomputed artifacts or the deprecated API is used.
    '''

    source = Path(extracted_dir)
    reject_precomputed_training_artifacts(source)
    forbidden_present = [
        name for name in FORBIDDEN_TRAINING_ARTIFACTS if _find_one_file(source, name) is not None
    ]
    if forbidden_present or _is_tar_source(source):
        raise ValueError(
            "Training from precomputed/global feature-reduction artifacts is not supported. "
            f"Source {source} contains forbidden training inputs: {forbidden_present or ['tar archive']}. "
            "Provide raw unreduced inputs via --merged-input, --raw-input-dir, "
            "or --pdbbind-input with --dudez-input."
        )
    raise ValueError(
        "load_reduction_artifacts is no longer supported for training. "
        "Use load_raw_modeling_input() with raw unreduced CSV inputs."
    )


def load_merged_input_dataset(source: str | Path) -> Optional[pd.DataFrame]:
    '''Load ``merged_input_dataset.csv`` from a directory or tar archive.

    Parameters
    ----------
    source : str or pathlib.Path
        Directory or tar archive that may contain a merged raw input table.

    Returns
    -------
    pd.DataFrame | None
        Merged raw input dataframe when present, otherwise ``None``.
    '''

    path = Path(source)
    if path.is_dir():
        merged_path = _find_one_file(path, MERGED_INPUT_DATASET_NAME)
        if merged_path is not None:
            return pd.read_csv(merged_path, low_memory=False)
        return None
    if _is_tar_source(path):
        with tarfile.open(path, "r:*") as tar:
            member = _find_tar_member(_safe_tar_members(tar), MERGED_INPUT_DATASET_NAME)
            if member is not None:
                return _read_tar_member_csv(tar, member)
    return None


def _resolve_dudez_outer_split_config(protocol: StagedTrainProtocol, outer_seed: int) -> DUDEzSplitConfig:
    return ocstaged.dudez_receptor_heldout_complete_config(
        random_seed=int(outer_seed),
        receptor_column="receptor",
        kind_column=protocol.dudez.kind_column,
        relaxed_split=False,
    )


def apply_train_only_feature_reduction(
        protocol: StagedTrainProtocol,
        raw_input: RawModelingInput,
        *,
        output_dir: Path,
        feature_blocks: Optional[Sequence[str]] = None,
        feature_policy: Optional[FeaturePolicy] = None,
        feature_policy_lookup_dirs: Sequence[str | Path] = (),
    ) -> ReductionArtifacts:
    '''Apply train-only feature reduction on a fixed outer split.

    Parameters
    ----------
    protocol : StagedTrainProtocol
        Loaded staged training protocol defining split policy and task columns.
    raw_input : RawModelingInput
        Validated raw unreduced modeling input.
    output_dir : pathlib.Path
        Directory where fixed split and train-only reduction artifacts are written.
    feature_blocks : Sequence[str] | None, optional
        Descriptor blocks eligible for this reduction pass. ``None`` keeps the
        full ligand+receptor+scoring protocol.
    feature_policy : FeaturePolicy | None, optional
        Feature-ablation policy used to constrain candidate model descriptors
        before train-only reduction is fitted.
    feature_policy_lookup_dirs : Sequence[str | pathlib.Path], optional
        Lookup directories recorded in feature-policy provenance.

    Returns
    -------
    ReductionArtifacts
        Modeling tables and metadata after frozen train-only feature selection.
    '''

    merged = raw_input.merged
    pdbbind_wide, dudez_wide = split_wide_dataset_by_column(merged)
    pdbbind_wide, _ = prepare_pdbbind_for_optuna(
        pdbbind_wide,
        target_column=protocol.pdbbind.target_column,
    )
    dudez_wide, _, _ = prepare_dudez_for_optuna(
        dudez_wide,
        kind_column=protocol.dudez.kind_column,
        positive_kind=protocol.dudez.positive_kind,
        negative_kind=protocol.dudez.negative_kind,
        ignore_unknown_kind=protocol.dudez.ignore_unknown_kind,
    )

    outer_seed = int(protocol.seed)
    pdbbind_split_config = protocol.pdbbind_split_config()

    pdbbind_reset = pdbbind_wide.reset_index(drop=True)
    split_result = split_pdbbind_regression(pdbbind_reset, pdbbind_split_config)
    train_df = pdbbind_reset.iloc[split_result.train_idx].copy()
    policy_metadata: Optional[dict[str, Any]] = None
    policy_candidate_features: Optional[list[str]] = None
    if feature_policy is not None:
        feature_reduction_config = feature_reduction_config_for_feature_blocks(
            target_column=protocol.pdbbind.target_column,
            feature_blocks=feature_blocks,
        )
        candidate_discovery = discover_candidate_model_features(
            train_df.columns,
            config=feature_reduction_config,
            non_feature_columns=NON_FEATURE_COLUMNS,
        )
        policy_application = apply_feature_policy(
            feature_policy,
            candidate_discovery.candidate_features,
            lookup_dirs=feature_policy_lookup_dirs,
        )
        policy_candidate_features = list(policy_application.final_candidate_features_before_reduction)
        policy_metadata = policy_application.to_metadata()
        policy_metadata["candidate_metadata_columns"] = candidate_discovery.metadata_columns
        policy_metadata["candidate_target_columns"] = candidate_discovery.target_columns
        policy_metadata["candidate_unmatched_columns"] = candidate_discovery.unmatched_columns
        policy_metadata["candidate_duplicate_assignments"] = candidate_discovery.duplicate_assignments

    frozen = fit_train_only_feature_reduction(
        train_df,
        target_column=protocol.pdbbind.target_column,
        feature_blocks=feature_blocks,
        candidate_features=policy_candidate_features,
    )
    reduced_pdbbind = apply_frozen_feature_selection(pdbbind_reset, frozen)
    reduced_dudez = apply_frozen_feature_selection(dudez_wide.reset_index(drop=True), frozen)

    pdbbind_cleanup = drop_nonfinite_selected_feature_rows(
        reduced_pdbbind,
        frozen.selected_features,
        label="PDBbind",
    )
    dudez_cleanup = drop_nonfinite_selected_feature_rows(
        reduced_dudez,
        frozen.selected_features,
        label="DUDEz",
    )
    reduced_pdbbind = pdbbind_cleanup.cleaned_df
    reduced_dudez = dudez_cleanup.cleaned_df

    def _remap_pdbbind_split(indices: Sequence[int], split_name: str) -> tuple[np.ndarray, int]:
        old_to_new = np.full(len(pdbbind_cleanup.kept_mask), -1, dtype=int)
        old_to_new[np.flatnonzero(pdbbind_cleanup.kept_mask)] = np.arange(
            int(pdbbind_cleanup.kept_mask.sum()),
            dtype=int,
        )
        remapped = old_to_new[np.asarray(indices, dtype=int)]
        dropped = int((remapped < 0).sum())
        remapped = remapped[remapped >= 0]
        if len(remapped) == 0:
            raise ValueError(
                f"PDBbind selected-feature cleanup removed all rows from the {split_name} split."
            )
        return remapped.astype(int), dropped

    pdbbind_train_idx, dropped_train_split_rows = _remap_pdbbind_split(split_result.train_idx, "train")
    pdbbind_val_idx, dropped_val_split_rows = _remap_pdbbind_split(split_result.val_idx, "validation")
    pdbbind_test_idx, dropped_test_split_rows = _remap_pdbbind_split(split_result.test_idx, "test")

    dudez_split_config = _resolve_dudez_outer_split_config(protocol, outer_seed)
    ocstaged.derive_dudez_labels(
        reduced_dudez,
        kind_column=protocol.dudez.kind_column,
    )
    dudez_split_result = split_dudez_by_receptor_and_kind(reduced_dudez, dudez_split_config)

    train_only_dir = output_dir / "train_only_feature_reduction"
    paths = write_train_only_reduction_artifact(train_only_dir, frozen)
    if policy_metadata is not None:
        policy_metadata["selected_features_after_train_only_reduction"] = list(frozen.selected_features)
        policy_metadata["selected_features_after_train_only_reduction_hash"] = hash_feature_list(frozen.selected_features)
        policy_metadata["removed_features_after_train_only_reduction"] = list(frozen.removed_features)
        policy_metadata["removed_features_after_train_only_reduction_hash"] = hash_feature_list(frozen.removed_features)
        policy_metadata_path = write_feature_policy_metadata(output_dir, _as_jsonable(policy_metadata))
        paths["feature_policy_metadata"] = str(policy_metadata_path)
    row_cleanup_dir = output_dir / "row_cleanup"
    row_cleanup_dir.mkdir(parents=True, exist_ok=True)
    pdbbind_dropped_path = row_cleanup_dir / "pdbbind_dropped_nonfinite_selected_features.csv"
    dudez_dropped_path = row_cleanup_dir / "dudez_dropped_nonfinite_selected_features.csv"
    row_cleanup_summary_path = row_cleanup_dir / "row_cleanup_summary.json"
    pdbbind_cleanup.dropped_rows.to_csv(pdbbind_dropped_path, index=False)
    dudez_cleanup.dropped_rows.to_csv(dudez_dropped_path, index=False)
    row_cleanup_summary = {
        "pdbbind": pdbbind_cleanup.summary,
        "dudez": dudez_cleanup.summary,
        "pdbbind_dropped_rows_by_original_split": {
            "train": dropped_train_split_rows,
            "validation": dropped_val_split_rows,
            "test": dropped_test_split_rows,
        },
    }
    row_cleanup_summary_path.write_text(
        json.dumps(_as_jsonable(row_cleanup_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fixed_outer = build_fixed_outer_split_assignment(
        outer_split_seed=outer_seed,
        pdbbind_train_indices=pdbbind_train_idx,
        pdbbind_validation_indices=pdbbind_val_idx,
        pdbbind_test_indices=pdbbind_test_idx,
        dudez_train_indices=dudez_split_result.train_idx,
        dudez_validation_indices=dudez_split_result.val_idx,
        dudez_test_indices=dudez_split_result.test_idx,
        feature_selection_fit_row_count=len(train_df),
        selected_features=frozen.selected_features,
        removed_features=frozen.removed_features,
        feature_reduction_artifact_path=str(train_only_dir),
    )
    write_fixed_outer_split_json(output_dir, fixed_outer)
    fixed_path = output_dir / "fixed_outer_split.json"
    modeling_pdbbind_path = output_dir / "modeling_pdbbind.csv"
    modeling_dudez_path = output_dir / "modeling_dudez.csv"
    reduced_pdbbind.to_csv(modeling_pdbbind_path, index=False)
    reduced_dudez.to_csv(modeling_dudez_path, index=False)

    _log(
        "Fixed outer split + train-only feature reduction: "
        f"outer_seed={outer_seed} fit_rows={len(train_df)} "
        f"feature_policy={(feature_policy.name if feature_policy is not None else FULL_OCSCORE_POLICY_NAME)} "
        f"feature_blocks={list(feature_blocks) if feature_blocks is not None else list(FULL_FEATURE_BLOCKS)} "
        f"policy_candidates={(len(policy_candidate_features) if policy_candidate_features is not None else 'all')} "
        f"selected={len(frozen.selected_features)} "
        f"dropped_nonfinite_rows={row_cleanup_summary['pdbbind']['n_rows_dropped']} PDBbind/"
        f"{row_cleanup_summary['dudez']['n_rows_dropped']} DUDEz"
    )
    return ReductionArtifacts(
        pdbbind_df=reduced_pdbbind,
        dudez_df=reduced_dudez,
        selected_features=frozen.selected_features,
        artifact_paths={
            **raw_input.artifact_paths,
            **paths,
            "fixed_outer_split": str(fixed_path),
            "modeling_pdbbind": str(modeling_pdbbind_path),
            "modeling_dudez": str(modeling_dudez_path),
            "pdbbind_dropped_nonfinite_selected_features": str(pdbbind_dropped_path),
            "dudez_dropped_nonfinite_selected_features": str(dudez_dropped_path),
            "row_cleanup_summary": str(row_cleanup_summary_path),
        },
        extracted_dir=output_dir,
        feature_selection=frozen.feature_selection,
        fixed_outer_split=fixed_outer,
        row_cleanup_summary=row_cleanup_summary,
        feature_policy_metadata=policy_metadata,
    )


def validate_selected_features(
        pdbbind_df: pd.DataFrame,
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
    ) -> None:
    '''Validate selected feature columns before staged Optuna modeling.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        Reduced PDBbind dataframe.
    dudez_df : pd.DataFrame
        Reduced DUDEz dataframe.
    selected_features : Sequence[str]
        Selected feature columns from feature reduction.

    Raises
    ------
    ValueError
        If selected features are missing, non-numeric, or contain NaN/Inf.
    '''

    reserved_selected = [feature for feature in selected_features if feature in NON_FEATURE_COLUMNS]
    if reserved_selected:
        raise ValueError(
            "Selected feature list contains metadata/target columns that must not be model inputs: "
            f"{reserved_selected}"
        )

    for name, df in [("PDBbind", pdbbind_df), ("DUDEz", dudez_df)]:
        missing = [feature for feature in selected_features if feature not in df.columns]
        if missing:
            raise ValueError(f"{name} reduced dataset is missing selected features: {missing}")

        feature_df = df.loc[:, list(selected_features)]
        non_numeric = [
            feature
            for feature in selected_features
            if not pd.api.types.is_numeric_dtype(feature_df[feature])
        ]
        if non_numeric:
            raise ValueError(f"{name} selected features must be numeric. Non-numeric columns: {non_numeric}")

        values = feature_df.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} selected feature columns contain NaN, +inf, or -inf values.")


def prepare_pdbbind_for_optuna(
        pdbbind_df: pd.DataFrame,
        target_column: str = "experimental",
    ) -> tuple[pd.DataFrame, int]:
    '''Prepare reduced PDBbind rows for regression Optuna.

    The default behavior is explicit: rows with missing or non-numeric target
    values are dropped and the dropped-row count is returned for logging.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        Reduced PDBbind dataframe.
    target_column : str, optional
        Regression target column, by default ``"experimental"``.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned PDBbind dataframe and number of dropped target rows.

    Raises
    ------
    ValueError
        If the target column is missing or no rows remain after cleanup.
    '''

    if target_column not in pdbbind_df.columns:
        raise ValueError(f"PDBbind target column is missing: {target_column!r}")

    prepared = pdbbind_df.copy()
    before = len(prepared)
    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")
    prepared = prepared[prepared[target_column].notna()].copy()
    dropped = before - len(prepared)
    if prepared.empty:
        raise ValueError("No PDBbind rows remain after dropping missing target values.")
    return prepared, dropped


def prepare_dudez_for_optuna(
        dudez_df: pd.DataFrame,
        kind_column: str = "kind",
        positive_kind: str = "ligands",
        negative_kind: str = "decoys",
        ignore_unknown_kind: bool = False,
    ) -> tuple[pd.DataFrame, dict[int, int], int]:
    '''Prepare reduced DUDEz rows for screening Optuna.

    The final kind values passed to the staged Optuna API are standardized to
    ``"ligands"`` and ``"decoys"`` so custom CLI kind names can still be used
    with the reusable DUDEz stage.

    Parameters
    ----------
    dudez_df : pd.DataFrame
        Reduced DUDEz dataframe.
    kind_column : str, optional
        Column containing active/decoy kind values, by default ``"kind"``.
    positive_kind : str, optional
        Raw kind value mapped to label 1, by default ``"ligands"``.
    negative_kind : str, optional
        Raw kind value mapped to label 0, by default ``"decoys"``.
    ignore_unknown_kind : bool, optional
        Drop rows with unknown kind values instead of failing, by default False.

    Returns
    -------
    tuple[pd.DataFrame, dict[int, int], int]
        Prepared DUDEz dataframe, class counts, and dropped unknown-kind rows.

    Raises
    ------
    ValueError
        If the kind column is missing, unknown kinds are present, or only one
        class remains.
    '''

    if kind_column not in dudez_df.columns:
        raise ValueError(f"DUDEz kind column is missing: {kind_column!r}")

    prepared = dudez_df.copy()
    normalized = prepared[kind_column].map(_normalize_kind)
    positive = _normalize_kind(positive_kind)
    negative = _normalize_kind(negative_kind)
    label = pd.Series(np.nan, index=prepared.index, dtype="float")
    label[normalized == positive] = 1.0
    label[normalized == negative] = 0.0
    unknown_mask = label.isna()

    if unknown_mask.any() and not ignore_unknown_kind:
        unknown_values = sorted(normalized[unknown_mask].dropna().unique().tolist())
        raise ValueError(f"Unknown DUDEz kind values: {unknown_values}")

    dropped = int(unknown_mask.sum())
    if ignore_unknown_kind and dropped:
        prepared = prepared.loc[~unknown_mask].copy()
        label = label.loc[~unknown_mask]

    prepared[LABEL_COLUMN] = label.astype(int)
    prepared[kind_column] = np.where(prepared[LABEL_COLUMN].to_numpy(dtype=int) == 1, "ligands", "decoys")
    class_counts = prepared[LABEL_COLUMN].value_counts().sort_index().astype(int).to_dict()
    if set(class_counts) != {0, 1}:
        raise ValueError(f"DUDEz data must contain both classes after label creation. Counts: {class_counts}")
    return prepared, class_counts, dropped


def copy_feature_reduction_artifacts(
        extracted_dir: str | Path,
        output_dir: str | Path,
    ) -> dict[str, str]:
    '''Copy optional feature-reduction reports into the Optuna output directory.

    Parameters
    ----------
    extracted_dir : str or pathlib.Path
        Directory or tar archive containing feature-reduction outputs.
    output_dir : str or pathlib.Path
        Staged Optuna output directory.

    Returns
    -------
    dict[str, str]
        Mapping from artifact filename to copied output path.
    '''

    source = Path(extracted_dir)
    artifact_dir = Path(output_dir) / "feature_reduction_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    filenames = [SELECTED_FEATURES_JSON, SELECTED_FEATURES_TXT, *OPTIONAL_FEATURE_REDUCTION_ARTIFACTS]

    if source.is_dir():
        for filename in filenames:
            found = _find_one_file(source, filename)
            if found is None:
                continue
            target = artifact_dir / filename
            shutil.copy2(found, target)
            copied[filename] = str(target)
    elif _is_tar_source(source):
        with tarfile.open(source, "r:*") as tar:
            members = _safe_tar_members(tar)
            for filename in filenames:
                member = _find_tar_member(members, filename)
                if member is None:
                    continue
                target = artifact_dir / filename
                _write_tar_member(tar, member, target)
                copied[filename] = str(target)
    else:
        raise ValueError(f"Reduction source must be a directory or tar archive: {source}")

    manifest_path = artifact_dir / "reduction_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(copied, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied["manifest"] = str(manifest_path)
    return copied


def build_protocol_context(
        pdbbind_df: pd.DataFrame,
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
        output_dir: str | Path,
        random_seed: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProtocolContext:
    '''Build the staged Optuna protocol context.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        Prepared reduced PDBbind dataframe.
    dudez_df : pd.DataFrame
        Prepared reduced DUDEz dataframe.
    selected_features : Sequence[str]
        Selected feature columns from feature reduction.
    output_dir : str or pathlib.Path
        Staged Optuna output directory.
    random_seed : int
        Random seed used by protocol stages.
    metadata : dict[str, Any], optional
        Dataset paths, reduction artifact paths, and run metadata.

    Returns
    -------
    ProtocolContext
        Context passed to the staged Optuna API.
    '''

    return ProtocolContext(
        pdbbind_df=pdbbind_df,
        dudez_df=dudez_df,
        selected_features=list(selected_features),
        output_dir=str(output_dir),
        random_seed=int(random_seed),
        metadata=metadata or {},
    )


def write_example_outputs(
        result: ReplicatedProtocolResult,
        base_context: ProtocolContext,
        reduction_archive: str | Path,
        extracted_dir: str | Path,
        output_dir: str | Path,
    ) -> dict[str, str]:
    '''Write example-level summaries around replicated protocol outputs.

    Parameters
    ----------
    result : ReplicatedProtocolResult
        Result returned by the replicated staged Optuna protocol.
    base_context : ProtocolContext
        Base protocol context shared by all modeling replicas.
    reduction_archive : str or pathlib.Path
        Source feature-reduction archive path.
    extracted_dir : str or pathlib.Path
        Directory or archive containing feature-reduction outputs.
    output_dir : str or pathlib.Path
        Staged Optuna output directory.

    Returns
    -------
    dict[str, str]
        Written summary output paths.
    '''

    out = Path(output_dir)
    paths: dict[str, str] = dict(result.output_paths)

    selected_path_value = paths.get("selected_features_json")
    if selected_path_value is None:
        selected_path = out / "selected_features.json"
        selected_path.write_text(json.dumps(list(base_context.selected_features), indent=2) + "\n", encoding="utf-8")
        paths["selected_features_json"] = str(selected_path)
    else:
        selected_path = Path(selected_path_value)

    static_context = {
        "n_selected_features": len(base_context.selected_features),
        "selected_features_hash": hash_feature_list(base_context.selected_features),
        "selected_features_json": str(selected_path),
    }
    if "static_context_json" in paths:
        static_context["static_context_json"] = paths["static_context_json"]

    summary = {
        "reduction_archive": str(reduction_archive),
        "reduction_source": str(extracted_dir),
        "static_context": static_context,
        "primary_claim": "ranking_screening",
        "calibration_report_mode": base_context.metadata.get("calibration_report_mode", "ranking_only"),
        "n_replicas": len(result.replica_results),
        "replicas": result.summary_df.to_dict(orient="records"),
        "aggregate_summary": result.aggregate_summary,
        "replicated_output_paths": result.output_paths,
        "failed_replicas": [replica.summary for replica in result.failed_replicas],
    }
    json_path = out / "staged_optuna_protocol.json"
    md_path = out / "staged_optuna_protocol.md"
    json_path.write_text(json.dumps(_as_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_protocol_markdown(_as_jsonable(summary), md_path)
    paths["staged_optuna_protocol_json"] = str(json_path)
    paths["staged_optuna_protocol_md"] = str(md_path)
    return paths



def _build_replicated_staged_protocol(protocol: StagedTrainProtocol) -> ReplicatedStagedProtocol:
    """Build the replicated staged Optuna protocol from YAML sections."""

    pdbbind_split_config = protocol.pdbbind_split_config()
    dudez_split_config = _resolve_dudez_outer_split_config(protocol, int(protocol.seed))
    if protocol.pdbbind.search_phase == "encoder_regression":
        pdbbind_config = ocstaged.pdbbind_phase1_experiment_config(
            target_column=protocol.pdbbind.target_column,
            n_trials=protocol.pdbbind.trials,
            epochs=protocol.pdbbind.epochs,
            n_jobs=protocol.pdbbind.n_jobs,
            random_seed=protocol.seed,
            sampler_seed=protocol.seed,
            use_gpu=protocol.runtime.use_gpu,
            enable_pruning=protocol.pdbbind.enable_pruning,
            split_config=pdbbind_split_config,
        )
    else:
        pdbbind_config = ocstaged.PDBbindOptunaConfig(
            target_column=protocol.pdbbind.target_column,
            n_trials=protocol.pdbbind.trials,
            epochs=protocol.pdbbind.epochs,
            n_jobs=protocol.pdbbind.n_jobs,
            random_seed=protocol.seed,
            sampler_seed=protocol.seed,
            use_gpu=protocol.runtime.use_gpu,
            search_phase=protocol.pdbbind.search_phase,
            enable_pruning=protocol.pdbbind.enable_pruning,
            split_config=pdbbind_split_config,
        )
    dudez_scaling_config = DUDEzScalingConfig(
        strategy=protocol.dudez.scaling_strategy,  # type: ignore[arg-type]
        strict=True,
    )
    dudez_config = ocstaged.DUDEzOptunaConfig(
        kind_column=protocol.dudez.kind_column,
        primary_metric=protocol.dudez.primary_metric,
        bedroc_alpha=protocol.dudez.bedroc_alpha,
        n_trials=protocol.dudez.trials,
        epochs=protocol.dudez.epochs,
        n_jobs=protocol.dudez.n_jobs,
        random_seed=protocol.seed,
        sampler_seed=protocol.seed,
        use_gpu=protocol.runtime.use_gpu,
        dudez_scaling_config=dudez_scaling_config,
        calibration_report_mode=protocol.reporting.calibration_report_mode,
        split_config=dudez_split_config,
    )
    if protocol.runtime.pdbbind_only:
        stages = [ocstaged.PDBbindOptunaStage(config=pdbbind_config)]
        stage_summary = "pdbbind"
    else:
        stages = [
            ocstaged.PDBbindOptunaStage(config=pdbbind_config),
            ocstaged.TransferFeatureExtractorStage(),
            ocstaged.DUDEzOptunaStage(config=dudez_config),
        ]
        stage_summary = "pdbbind -> transfer -> dudez"

    _log(
        "Modeling setup: "
        f"replicas={protocol.replicas} replica_jobs={protocol.runtime.replica_jobs} "
        f"resume_completed={protocol.runtime.resume_completed} "
        f"seed={protocol.seed} stages={stage_summary} "
        f"pdbbind_trials={protocol.pdbbind.trials} pdbbind_n_jobs={protocol.pdbbind.n_jobs} "
        f"pdbbind_phase={protocol.pdbbind.search_phase} "
        f"dudez_trials={0 if protocol.runtime.pdbbind_only else protocol.dudez.trials} "
        f"dudez_n_jobs={0 if protocol.runtime.pdbbind_only else protocol.dudez.n_jobs} "
        f"dudez_metric={protocol.dudez.primary_metric} "
        f"dudez_bedroc_alpha={protocol.dudez.bedroc_alpha}"
    )

    return ReplicatedStagedProtocol(
        stages=stages,
        n_replicas=protocol.replicas,
        base_seed=protocol.seed,
        replica_name_prefix="replica",
        replica_jobs=protocol.runtime.replica_jobs,
        resume_completed=protocol.runtime.resume_completed,
    )


def _collect_replica_alignments(result: ReplicatedProtocolResult) -> list[dict[str, Any]]:
    """Collect fixed-split alignment payloads from successful replicas."""

    replica_alignments: list[dict[str, Any]] = []
    for replica in result.replica_results:
        if replica.success and replica.context is not None:
            pdb_stage = replica.context.stage_results.get("pdbbind_optuna") or {}
            alignment = pdb_stage.get("split_alignment")
            if alignment:
                replica_alignments.append(alignment)
    return replica_alignments


def _run_staged_training_from_artifacts(
        *,
        protocol: StagedTrainProtocol,
        raw_input: RawModelingInput,
        artifacts: ReductionArtifacts,
        output_dir: Path,
        protocol_path: Path,
        raw_input_hashes: dict[str, Optional[str]],
        log_prefix: str,
        ablation_variant: Optional[str] = None,
        feature_blocks: Optional[Sequence[str]] = None,
    ) -> StagedTrainingRun:
    """Prepare model tables, run replicated staged Optuna, and write summaries."""

    preserved_non_features = sorted([
        column
        for column in NON_FEATURE_COLUMNS
        if column in artifacts.pdbbind_df.columns or column in artifacts.dudez_df.columns
    ])
    _log(
        f"{log_prefix}: prepared inputs "
        f"PDBbind={artifacts.pdbbind_df.shape[0]}x{artifacts.pdbbind_df.shape[1]} "
        f"DUDEz={artifacts.dudez_df.shape[0]}x{artifacts.dudez_df.shape[1]} "
        f"selected_features={len(artifacts.selected_features)} "
        f"metadata_columns_excluded={len(preserved_non_features)}"
    )

    validate_selected_features(artifacts.pdbbind_df, artifacts.dudez_df, artifacts.selected_features)
    if artifacts.feature_selection is not None:
        verify_selected_features_against_scope(artifacts.selected_features, artifacts.feature_selection)
        validate_train_only_feature_selection(artifacts.feature_selection)
    pdbbind_df, dropped_pdbbind_targets = prepare_pdbbind_for_optuna(
        artifacts.pdbbind_df,
        target_column=protocol.pdbbind.target_column,
    )
    dudez_df, dudez_class_counts, dropped_dudez_unknown = prepare_dudez_for_optuna(
        artifacts.dudez_df,
        kind_column=protocol.dudez.kind_column,
        positive_kind=protocol.dudez.positive_kind,
        negative_kind=protocol.dudez.negative_kind,
        ignore_unknown_kind=protocol.dudez.ignore_unknown_kind,
    )
    _log(
        f"{log_prefix}: modeling rows "
        f"PDBbind={len(pdbbind_df)} dropped_target={dropped_pdbbind_targets}; "
        f"DUDEz={len(dudez_df)} dropped_unknown_kind={dropped_dudez_unknown}; "
        f"DUDEz_classes={dudez_class_counts}"
    )

    fixed_outer_payload = artifacts.fixed_outer_split.to_dict() if artifacts.fixed_outer_split else None
    metadata = {
        "protocol_valid": True,
        "feature_selection_scope": "train_only",
        "feature_selection_fit_split": "train",
        "fixed_outer_split": fixed_outer_payload,
        "global_feature_reduction_used": False,
        "precomputed_features_used_for_training": False,
        "raw_input_hashes": raw_input_hashes,
        "raw_input_paths": raw_input.artifact_paths,
        "copied_feature_reduction_artifacts": {},
        "protocol": protocol.name,
        "protocol_path": str(protocol_path),
        "pdbbind_target_column": protocol.pdbbind.target_column,
        "dudez_kind_column": protocol.dudez.kind_column,
        "dudez_label_column": LABEL_COLUMN,
        "dudez_positive_kind": protocol.dudez.positive_kind,
        "dudez_negative_kind": protocol.dudez.negative_kind,
        "dudez_class_counts": dudez_class_counts,
        "dropped_pdbbind_target_rows": dropped_pdbbind_targets,
        "dropped_dudez_unknown_kind_rows": dropped_dudez_unknown,
        "selected_feature_row_cleanup": artifacts.row_cleanup_summary,
        "feature_selection": artifacts.feature_selection.to_dict() if artifacts.feature_selection else None,
        "fixed_outer_split_metadata": fixed_outer_payload,
        "calibration_report_mode": protocol.reporting.calibration_report_mode,
    }
    if artifacts.feature_policy_metadata is not None:
        metadata["feature_policy"] = artifacts.feature_policy_metadata
    if ablation_variant is not None:
        metadata["ablation"] = {
            "variant": ablation_variant,
            "feature_blocks": list(feature_blocks or []),
            "full_protocol_reference": False,
        }
    context = build_protocol_context(
        pdbbind_df=pdbbind_df,
        dudez_df=dudez_df,
        selected_features=artifacts.selected_features,
        output_dir=output_dir,
        random_seed=protocol.seed,
        metadata=metadata,
    )

    staged_protocol = _build_replicated_staged_protocol(protocol)
    result = staged_protocol.run(context)
    replica_alignments = _collect_replica_alignments(result)
    protocol_metadata = validate_protocol_integrity(
        feature_selection=metadata.get("feature_selection"),
        fixed_outer_split=metadata.get("fixed_outer_split_metadata"),
        replica_alignments=replica_alignments,
        raw_input_hashes=raw_input_hashes,
    )
    metadata["protocol_metadata"] = protocol_metadata

    written = write_example_outputs(
        result=result,
        base_context=context,
        reduction_archive=str(raw_input.merged_source or raw_input.artifact_paths.get("raw_pdbbind", output_dir)),
        extracted_dir=output_dir,
        output_dir=output_dir,
    )

    aggregate = result.aggregate_summary or {}
    n_successful = aggregate.get("n_successful_replicas", 0)
    n_failed = aggregate.get("n_failed_replicas", 0)
    _log(
        f"{log_prefix}: training complete "
        f"replicas={n_successful}/{len(result.replica_results)} failed={n_failed}; "
        f"best_pdbbind={_replica_name(aggregate.get('best_pdbbind_replica'))}; "
        f"best_dudez={_replica_name(aggregate.get('best_dudez_replica'))}"
    )
    _log(
        f"{log_prefix}: summaries "
        + _compact_paths({
            "protocol_json": written.get("staged_optuna_protocol_json"),
            "protocol_md": written.get("staged_optuna_protocol_md"),
            "replicas_csv": result.output_paths.get("replicas_summary_csv"),
        })
    )

    return StagedTrainingRun(
        artifacts=artifacts,
        pdbbind_df=pdbbind_df,
        dudez_df=dudez_df,
        context=context,
        metadata=metadata,
        result=result,
        written=written,
        protocol_metadata=protocol_metadata,
        replica_alignments=replica_alignments,
    )


def _build_ablation_summary_row(
        *,
        variant: str,
        feature_blocks: Sequence[str],
        output_dir: Path,
        selected_features: Sequence[str],
        result: ReplicatedProtocolResult,
        written: dict[str, str],
    ) -> dict[str, Any]:
    """Build one ablation summary row with flattened metric columns."""

    aggregate = result.aggregate_summary or {}
    row: dict[str, Any] = {
        "variant": variant,
        "feature_blocks": list(feature_blocks),
        "output_dir": str(output_dir),
        "n_selected_features": len(selected_features),
        "selected_features_hash": hash_feature_list(selected_features),
        "n_replicas": len(result.replica_results),
        "n_successful_replicas": aggregate.get("n_successful_replicas"),
        "n_failed_replicas": aggregate.get("n_failed_replicas"),
        "staged_optuna_protocol_json": written.get("staged_optuna_protocol_json"),
        "replicas_summary_csv": result.output_paths.get("replicas_summary_csv"),
        "aggregate_summary": aggregate,
        "output_paths": written,
    }
    metrics = aggregate.get("metrics") or {}
    if isinstance(metrics, dict):
        for metric_name, metric_summary in metrics.items():
            if isinstance(metric_summary, dict):
                for stat_name in ("mean", "std", "median", "min", "max", "n"):
                    if stat_name in metric_summary:
                        row[f"{metric_name}_{stat_name}"] = metric_summary.get(stat_name)
    return row


def _ablation_csv_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ablation rows for CSV output."""

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"aggregate_summary", "output_paths"}:
                continue
            if key == "feature_blocks":
                flat[key] = "+".join(str(item) for item in value)
            else:
                flat[key] = value
        csv_rows.append(flat)
    return csv_rows




def _policy_output_dir(parent_dir: Path, policy: FeaturePolicy) -> Path:
    name = str(policy.name).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Feature policy name is not safe for an output directory: {policy.name!r}")
    return parent_dir / name


def _metric_stat(aggregate: dict[str, Any], metric_name: str, stat_name: str) -> Any:
    metrics = aggregate.get("metrics") or {}
    summary = metrics.get(metric_name) or {}
    if isinstance(summary, dict):
        return summary.get(stat_name)
    return None


def _build_feature_policy_summary_row(
        *,
        policy: FeaturePolicy,
        output_dir: Path,
        selected_features: Sequence[str],
        result: ReplicatedProtocolResult,
        written: dict[str, str],
        feature_policy_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
    """Build one feature-policy comparison row."""

    aggregate = result.aggregate_summary or {}
    row: dict[str, Any] = {
        "feature_policy_name": policy.name,
        "feature_policy_description": policy.description,
        "feature_policy_source_path": str(policy.source_path),
        "feature_policy_source_kind": policy.source_kind,
        "feature_policy_hash": policy.source_hash,
        "output_dir": str(output_dir),
        "n_replicas": len(result.replica_results),
        "n_successful_replicas": aggregate.get("n_successful_replicas"),
        "n_failed_replicas": aggregate.get("n_failed_replicas"),
        "dudez_test_BEDROC_mean": _metric_stat(aggregate, "dudez_test_bedroc", "mean"),
        "dudez_test_BEDROC_std": _metric_stat(aggregate, "dudez_test_bedroc", "std"),
        "dudez_test_EF1_mean": _metric_stat(aggregate, "dudez_test_ef1", "mean"),
        "dudez_test_EF1_std": _metric_stat(aggregate, "dudez_test_ef1", "std"),
        "dudez_test_PR_AUC_mean": _metric_stat(aggregate, "dudez_test_pr_auc", "mean"),
        "dudez_test_PR_AUC_std": _metric_stat(aggregate, "dudez_test_pr_auc", "std"),
        "dudez_test_ROC_AUC_mean": _metric_stat(aggregate, "dudez_test_roc_auc", "mean"),
        "dudez_test_ROC_AUC_std": _metric_stat(aggregate, "dudez_test_roc_auc", "std"),
        "pdbbind_test_RMSE_mean": _metric_stat(aggregate, "pdbbind_test_rmse", "mean"),
        "pdbbind_test_RMSE_std": _metric_stat(aggregate, "pdbbind_test_rmse", "std"),
        "selected_feature_count_mean": float(len(selected_features)),
        "selected_feature_count_std": 0.0,
        "selected_features_hash": hash_feature_list(selected_features),
        "staged_optuna_protocol_json": written.get("staged_optuna_protocol_json"),
        "replicas_summary_csv": result.output_paths.get("replicas_summary_csv"),
        "feature_policy_metadata": feature_policy_metadata or {},
        "aggregate_summary": aggregate,
        "output_paths": written,
    }
    return row


def _feature_policy_csv_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten feature-policy summary rows for CSV output."""

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append({
            key: value
            for key, value in row.items()
            if key not in {"feature_policy_metadata", "aggregate_summary", "output_paths"}
        })
    return csv_rows


def _write_feature_policy_summary(
        output_dir: Path,
        *,
        protocol: StagedTrainProtocol,
        protocol_path: Path,
        rows: Sequence[dict[str, Any]],
    ) -> dict[str, str]:
    """Write aggregate feature-policy ablation summaries."""

    json_path = output_dir / FEATURE_POLICY_SUMMARY_JSON
    csv_path = output_dir / FEATURE_POLICY_SUMMARY_CSV
    payload = {
        "protocol": protocol.name,
        "protocol_path": str(protocol_path),
        "feature_policies": list(rows),
    }
    json_path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(_feature_policy_csv_rows(rows)).to_csv(csv_path, index=False)
    return {
        "feature_policy_ablation_summary_json": str(json_path),
        "feature_policy_ablation_summary_csv": str(csv_path),
    }


def _run_post_training_reports(
        *,
        protocol: StagedTrainProtocol,
        raw_input: RawModelingInput,
        output_dir: Path,
        protocol_path: Path,
        raw_input_hashes: dict[str, Optional[str]],
        run_output: StagedTrainingRun,
        extra_ablation_paths: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
    """Run leakage audit, baselines, final report, and provenance for one training run."""

    result = run_output.result
    metadata = run_output.metadata
    artifacts = run_output.artifacts
    dudez_df = run_output.dudez_df
    protocol_metadata = run_output.protocol_metadata
    replica_alignments = run_output.replica_alignments
    reporting_paths: dict[str, str] = {}

    if protocol.reporting.generate_final_report or protocol.reporting.run_leakage_audit:
        first_replica = result.replica_results[0] if result.replica_results else None
        pdbbind_diag = None
        dudez_diag = None
        scaling_meta = None
        if first_replica is not None and first_replica.context is not None:
            stage_results = first_replica.context.stage_results
            pdbbind_diag = (stage_results.get("pdbbind_optuna") or {}).get("split_diagnostics")
            dudez_diag = (stage_results.get("dudez_optuna") or {}).get("split_diagnostics")
            scaling_meta = (stage_results.get("dudez_optuna") or {}).get("scaling_metadata")
        audit = run_leakage_audit(
            feature_selection=metadata.get("feature_selection"),
            pdbbind_split_diagnostics=pdbbind_diag,
            dudez_split_diagnostics=dudez_diag,
            scaling_metadata=scaling_meta,
            strict=True,
        )
        audit_path = write_leakage_audit_report(output_dir, audit)
        reporting_paths["leakage_audit"] = str(audit_path)
        _log(f"Leakage audit: passed={audit.passed}; report={audit_path}")
        if not audit.passed:
            raise ValueError("Leakage audit failed. See leakage_audit.json for details.")

    baseline_paths: dict[str, str] = {}
    if protocol.reporting.run_baselines:
        baseline_paths = run_and_write_production_baselines(
            output_dir,
            dudez_df=dudez_df,
            selected_features=artifacts.selected_features,
            replica_results=result.replica_results,
            config=ProductionBaselineConfig(
                label_column=LABEL_COLUMN,
                group_column="receptor",
                bedroc_alpha=protocol.dudez.bedroc_alpha,
            ),
        )
        reporting_paths.update({f"baseline_{key}": value for key, value in baseline_paths.items()})
        _log(f"Production baselines: {_compact_paths(baseline_paths)}")

    if protocol.reporting.generate_final_report:
        calibration_mode = protocol.reporting.calibration_report_mode
        first_success = next(
            (replica for replica in result.replica_results if replica.success and replica.context),
            None,
        )
        calibration_section = None
        split_assignments = None
        if first_success is not None and first_success.context is not None:
            stage_results = first_success.context.stage_results
            dudez_stage = stage_results.get("dudez_optuna") or {}
            pdbbind_stage = stage_results.get("pdbbind_optuna") or {}
            split_assignments = build_split_assignments_payload(
                pdbbind_stage=pdbbind_stage,
                dudez_stage=dudez_stage,
            )
            val_metrics = dudez_stage.get("validation_metrics") or {}
            test_metrics = dudez_stage.get("test_metrics") or {}
            validate_calibration_report_mode(
                val_metrics,
                calibration_mode,  # type: ignore[arg-type]
                strict=True,
            )
            validate_calibration_report_mode(
                test_metrics,
                calibration_mode,  # type: ignore[arg-type]
                strict=True,
            )
            calibration_section = build_calibration_report_section(
                val_metrics,
                test_metrics,
                mode=calibration_mode,  # type: ignore[arg-type]
            )
        final_report = {
            "aggregate_summary": result.aggregate_summary,
            "protocol": protocol.name,
            "protocol_path": str(protocol_path),
            "budget": protocol.budget_dict(),
            "primary_claim": "ranking_screening",
            "calibration_report_mode": calibration_mode,
            "calibration": calibration_section,
            **protocol_metadata,
            "fixed_outer_split": metadata.get("fixed_outer_split_metadata"),
            "replica_split_alignment": replica_alignments,
        }
        if metadata.get("feature_policy"):
            final_report["feature_policy"] = metadata.get("feature_policy")
        if baseline_paths:
            final_report["baselines"] = {"output_paths": baseline_paths}
        if extra_ablation_paths:
            final_report["ablations"] = {"output_paths": extra_ablation_paths}
        provenance_paths = write_production_provenance_bundle(
            output_dir,
            feature_selection=metadata.get("feature_selection"),
            scaling={"note": "See replica stage results and scaling.json per replica when available."},
            split_assignments=split_assignments,
            data_provenance={
                "raw_input_paths": raw_input.artifact_paths,
                "raw_input_hashes": raw_input_hashes,
                "metadata": metadata,
            },
            command={
                "argv": sys.argv,
                "protocol": protocol.name,
                "protocol_path": str(protocol_path),
            },
            final_report=final_report,
        )
        reporting_paths.update(provenance_paths)
        _log(f"Production provenance bundle: {_compact_paths(provenance_paths)}")

    return reporting_paths


def _run_one_feature_policy_training(
        *,
        protocol: StagedTrainProtocol,
        raw_input: RawModelingInput,
        protocol_path: Path,
        raw_input_hashes: dict[str, Optional[str]],
        output_dir: Path,
        feature_policy: FeaturePolicy,
        feature_policy_lookup_dirs: Sequence[str | Path],
        log_prefix: str,
    ) -> StagedTrainingRun:
    """Run one production training pass for a resolved feature policy."""

    artifacts = apply_train_only_feature_reduction(
        protocol,
        raw_input,
        output_dir=output_dir,
        feature_policy=feature_policy,
        feature_policy_lookup_dirs=feature_policy_lookup_dirs,
    )
    return _run_staged_training_from_artifacts(
        protocol=protocol,
        raw_input=raw_input,
        artifacts=artifacts,
        output_dir=output_dir,
        protocol_path=protocol_path,
        raw_input_hashes=raw_input_hashes,
        log_prefix=log_prefix,
    )

def _run_ablation_protocols(
        *,
        protocol: StagedTrainProtocol,
        raw_input: RawModelingInput,
        protocol_path: Path,
        raw_input_hashes: dict[str, Optional[str]],
        output_dir: Path,
        full_run: StagedTrainingRun,
    ) -> dict[str, str]:
    """Run configured descriptor-block ablations with the same staged protocol."""

    if not protocol.ablation.enabled:
        return {}

    ablation_dir = output_dir / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "Ablation protocol enabled: "
        f"variants={list(protocol.ablation.variants)} output_dir={ablation_dir}"
    )

    rows: list[dict[str, Any]] = [
        _build_ablation_summary_row(
            variant="full",
            feature_blocks=FULL_FEATURE_BLOCKS,
            output_dir=output_dir,
            selected_features=full_run.artifacts.selected_features,
            result=full_run.result,
            written=full_run.written,
        )
    ]

    for variant in protocol.ablation.variants:
        feature_blocks = ABLATION_FEATURE_BLOCKS[str(variant)]
        variant_dir = ablation_dir / str(variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Ablation {variant}: feature blocks={list(feature_blocks)} output_dir={variant_dir}")
        variant_artifacts = apply_train_only_feature_reduction(
            protocol,
            raw_input,
            output_dir=variant_dir,
            feature_blocks=feature_blocks,
        )
        variant_run = _run_staged_training_from_artifacts(
            protocol=protocol,
            raw_input=raw_input,
            artifacts=variant_artifacts,
            output_dir=variant_dir,
            protocol_path=protocol_path,
            raw_input_hashes=raw_input_hashes,
            log_prefix=f"Ablation {variant}",
            ablation_variant=str(variant),
            feature_blocks=feature_blocks,
        )
        rows.append(
            _build_ablation_summary_row(
                variant=str(variant),
                feature_blocks=feature_blocks,
                output_dir=variant_dir,
                selected_features=variant_artifacts.selected_features,
                result=variant_run.result,
                written=variant_run.written,
            )
        )

    json_path = ablation_dir / "ablation_summary.json"
    csv_path = ablation_dir / "ablation_summary.csv"
    payload = {
        "protocol": protocol.name,
        "protocol_path": str(protocol_path),
        "full_protocol_output_dir": str(output_dir),
        "variants": rows,
    }
    json_path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(_ablation_csv_rows(rows)).to_csv(csv_path, index=False)
    _log(f"Ablation summaries: json={json_path} csv={csv_path}")
    return {"ablation_summary_json": str(json_path), "ablation_summary_csv": str(csv_path)}


def main(argv: Optional[list[str]] = None) -> int:
    '''Run staged OCScore Optuna from raw unreduced modeling inputs.

    Parameters
    ----------
    argv : list[str], optional
        Optional argument list for testing or programmatic execution.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    args = build_argparser().parse_args(argv)
    return main_from_args(args)


def main_from_args(args: argparse.Namespace) -> int:
    '''Run the unified leakage-safe staged training protocol.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process exit code. Zero indicates success.

    Raises
    ------
    ValueError
        If deprecated reduction inputs are supplied or protocol validation fails.
    '''

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "reduction_archive", None):
        raise ValueError(
            "The --reduction-archive option is no longer supported for training. "
            "Provide raw unreduced inputs via --merged-input, --raw-input-dir, "
            "or --pdbbind-input with --dudez-input."
        )

    protocol_path = resolve_protocol_path(args.protocol)
    protocol = load_staged_train_protocol(protocol_path)
    shutil.copy2(protocol_path, output_dir / protocol_path.name)
    _log(f"Protocol: {protocol.name} ({protocol_path})")

    raw_input = load_raw_modeling_input(
        merged_input=getattr(args, "merged_input", None),
        pdbbind_input=getattr(args, "pdbbind_input", None),
        dudez_input=getattr(args, "dudez_input", None),
        raw_input_dir=getattr(args, "raw_input_dir", None),
    )
    _log(f"Raw modeling input: merged rows={raw_input.merged.shape[0]} columns={raw_input.merged.shape[1]}")
    _log(f"Output directory: {output_dir}")

    raw_input_hashes = {
        "merged": raw_input.merged_hash,
        "pdbbind": raw_input.pdbbind_hash,
        "dudez": raw_input.dudez_hash,
    }
    feature_policies, policy_discovery = resolve_requested_feature_policies(
        requested_names=getattr(args, "feature_policy", None),
        run_all=bool(getattr(args, "run_all_feature_policies", False)),
        policy_dirs=getattr(args, "feature_policy_dir", None),
        explicit_ymls=getattr(args, "feature_policy_yml", None),
    )
    _log(
        "Feature policy selection: "
        f"policies={[policy.name for policy in feature_policies]} "
        f"lookup_dirs={[str(path) for path in policy_discovery.lookup_dirs]}"
    )

    if len(feature_policies) > 1:
        rows: list[dict[str, Any]] = []
        for policy in feature_policies:
            policy_dir = _policy_output_dir(output_dir, policy)
            policy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protocol_path, policy_dir / protocol_path.name)
            _log(f"Feature policy {policy.name}: output_dir={policy_dir}")
            run_output = _run_one_feature_policy_training(
                protocol=protocol,
                raw_input=raw_input,
                protocol_path=protocol_path,
                raw_input_hashes=raw_input_hashes,
                output_dir=policy_dir,
                feature_policy=policy,
                feature_policy_lookup_dirs=policy_discovery.lookup_dirs,
                log_prefix=f"Feature policy {policy.name}",
            )
            reporting_paths = _run_post_training_reports(
                protocol=protocol,
                raw_input=raw_input,
                output_dir=policy_dir,
                protocol_path=protocol_path,
                raw_input_hashes=raw_input_hashes,
                run_output=run_output,
            )
            row_written = {**run_output.written, **reporting_paths}
            rows.append(
                _build_feature_policy_summary_row(
                    policy=policy,
                    output_dir=policy_dir,
                    selected_features=run_output.artifacts.selected_features,
                    result=run_output.result,
                    written=row_written,
                    feature_policy_metadata=run_output.artifacts.feature_policy_metadata,
                )
            )
        summary_paths = _write_feature_policy_summary(
            output_dir,
            protocol=protocol,
            protocol_path=protocol_path,
            rows=rows,
        )
        _log(f"Feature-policy ablation summaries: {_compact_paths(summary_paths)}")
        _log("Training complete.")
        return 0

    feature_policy = feature_policies[0]
    run_output = _run_one_feature_policy_training(
        protocol=protocol,
        raw_input=raw_input,
        protocol_path=protocol_path,
        raw_input_hashes=raw_input_hashes,
        output_dir=output_dir,
        feature_policy=feature_policy,
        feature_policy_lookup_dirs=policy_discovery.lookup_dirs,
        log_prefix=f"Feature policy {feature_policy.name}",
    )
    _run_post_training_reports(
        protocol=protocol,
        raw_input=raw_input,
        output_dir=output_dir,
        protocol_path=protocol_path,
        raw_input_hashes=raw_input_hashes,
        run_output=run_output,
    )

    _log("Training complete.")

    return 0

