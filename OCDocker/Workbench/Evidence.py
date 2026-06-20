#!/usr/bin/env python3

# Description
###############################################################################
'''
Read-only OCScore evidence discovery for Workbench dashboards.
'''

# Imports
###############################################################################
from __future__ import annotations

import csv

from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterable

from OCDocker.Workbench.IO import read_result_manifest
from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import EvidenceKind
from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import RunStatus
from OCDocker.Workbench.Models import WorkbenchEvidenceEntry
from OCDocker.Workbench.Models import WorkbenchEvidenceIndex
from OCDocker.Workbench.Registry import discover_result_manifest_paths

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

CSV_SUFFIX = ".csv"
EVIDENCE_SUFFIXES = frozenset({".csv", ".json", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html", ".npy", ".db"})
EVIDENCE_ASSET_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
SOURCE_EXPORT_DIR = "export"
ABLATION_CONTAINER_NAME = "ablations"
PERFORMANCE_TABLE_NAMES = frozenset(
    {
        "baselines_per_fold.csv",
        "baselines_rank_table.csv",
        "baselines_summary.csv",
        "cross_validation_folds.csv",
        "cross_validation_fold_comparison.csv",
        "cross_validation_fold_rankings.csv",
    }
)
IDENTIFIER_COLUMNS = frozenset(
    {
        "baseline",
        "baseline_family",
        "split",
        "replica",
        "fold",
        "fold_index",
        "dataset",
        "run_id",
        "policy_name",
        "state",
    }
)

# Functions
###############################################################################
## Private ##


def _is_numeric(value: Any) -> bool:
    '''Return whether a value can be interpreted as finite numeric.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True when the value is finite numeric.
    '''

    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _float(value: Any) -> float | None:
    '''Return a finite float or None.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    float or None
        Numeric value when conversion succeeds.
    '''

    if not _is_numeric(value):
        return None
    return float(value)


def _modified_at(path: Path) -> datetime | None:
    '''Return a file modification timestamp when available.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    datetime or None
        UTC modification timestamp.
    '''

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _size_bytes(path: Path) -> int | None:
    '''Return a file size when available.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    int or None
        File size in bytes.
    '''

    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _source_path_from_result_manifest(manifest_path: Path) -> Path | None:
    '''Return the adopted source path for a result manifest.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Result manifest path.

    Returns
    -------
    pathlib.Path or None
        Adopted source path when available.
    '''

    run_manifest_path = manifest_path.parent / "run_manifest.yml"
    if not run_manifest_path.exists():
        return None
    try:
        run_manifest = read_run_manifest(run_manifest_path)
    except Exception:
        return None
    source_path = run_manifest.metadata.get("source_path") if run_manifest.metadata else None
    if source_path:
        return Path(str(source_path))
    return run_manifest.workspace


def _policy_from_path(path: Path | None, fallback: str) -> str:
    '''Infer an ablation policy label from a source path.

    Parameters
    ----------
    path : pathlib.Path or None
        Adopted source path.
    fallback : str
        Fallback run id.

    Returns
    -------
    str
        Policy label.
    '''

    if path is None:
        return fallback
    parts = path.parts
    for index, part in enumerate(parts):
        if part == ABLATION_CONTAINER_NAME and index + 1 < len(parts):
            return parts[index + 1]
    return path.name or fallback


def _policy_from_evidence_path(path: Path, fallback: str) -> str:
    '''Infer a comparison policy label from an evidence file path.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.
    fallback : str
        Fallback policy label.

    Returns
    -------
    str
        Policy label inferred from ablation or analysis path components.
    '''

    parts = path.parts
    lowered = [part.lower() for part in parts]
    for marker in (ABLATION_CONTAINER_NAME, "replica_analysis"):
        if marker in lowered:
            index = lowered.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return fallback


def _dataset_from_path(path: Path) -> str:
    '''Infer a dataset label from common OCScore path components.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.

    Returns
    -------
    str
        Dataset label when recognized.
    '''

    lowered = [part.lower() for part in path.parts]
    for name in ("dudez", "pdbbind"):
        if name in lowered:
            return name
    return ""


def _replica_from_path(path: Path) -> str:
    '''Infer a replica label from a path.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.

    Returns
    -------
    str
        Replica label when recognized.
    '''

    for part in path.parts:
        if part.startswith("replica_"):
            return part
    return ""


def _source_roots(source_path: Path | None) -> tuple[Path, ...]:
    '''Return source roots to scan for one adopted run.

    Parameters
    ----------
    source_path : pathlib.Path or None
        Adopted source path.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing source roots to scan.
    '''

    if source_path is None:
        return ()
    roots = []
    if source_path.name == "train":
        export_path = source_path.parent / SOURCE_EXPORT_DIR
        if export_path.exists():
            roots.append(export_path)
    roots.append(source_path)
    return tuple(dict.fromkeys(root for root in roots if root.exists()))


def _within_depth(root: Path, path: Path, max_depth: int) -> bool:
    '''Return whether a file is within a recursive scan depth.

    Parameters
    ----------
    root : pathlib.Path
        Scan root.
    path : pathlib.Path
        Candidate file path.
    max_depth : int
        Maximum child directory depth.

    Returns
    -------
    bool
        True when the file is within depth.
    '''

    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return max(0, len(relative.parts) - 1) <= max_depth


def _iter_source_files(source_root: Path, *, source_depth: int) -> Iterable[Path]:
    '''Yield candidate evidence files below a source root.

    Parameters
    ----------
    source_root : pathlib.Path
        Existing source root.
    source_depth : int
        Maximum child directory depth.

    Yields
    ------
    pathlib.Path
        Candidate file path.
    '''

    for path in source_root.rglob("*"):
        if not path.is_file() or not _within_depth(source_root, path, source_depth):
            continue
        if path.suffix.lower() in EVIDENCE_SUFFIXES:
            yield path


def _evidence_kind(path: Path) -> EvidenceKind | None:
    '''Classify one OCScore evidence file.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file path.

    Returns
    -------
    EvidenceKind or None
        Evidence kind, or None when the file is not currently indexed.
    '''

    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    suffix = path.suffix.lower()
    if name.startswith("shap") or "shap" in parts:
        return "shap"
    if name.endswith("_optuna_trials.csv") or name in {"optuna.db", "optuna_summary.json"}:
        return "optimization"
    if name in PERFORMANCE_TABLE_NAMES:
        return "performance"
    if "prediction" in name:
        return "prediction"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}:
        if any(token in name for token in ("roc", "pr-", "curve", "fold", "cv_", "heatmap", "boxplot")):
            return "figure"
    return None


def _figure_name(path: Path, kind: EvidenceKind) -> str:
    '''Return a human-readable figure or evidence family name.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.
    kind : EvidenceKind
        Evidence kind.

    Returns
    -------
    str
        Compact figure family label.
    '''

    stem = path.stem.replace("-", "_")
    lowered = stem.lower()
    if kind == "shap":
        if "beeswarm" in lowered:
            return "SHAP beeswarm"
        if "feature_importance" in lowered or "importance" in lowered:
            return "SHAP feature importance"
        if "dependence" in lowered:
            return "SHAP dependence"
        if "summary" in lowered:
            return "SHAP summary"
        if lowered == "shap_values":
            return "SHAP values"
        return f"SHAP {stem.replace('_', ' ')}".strip()
    if "roc" in lowered:
        return "ROC curve"
    if "pr" in lowered and "curve" in lowered:
        return "PR curve"
    if "heatmap" in lowered:
        return "Heatmap"
    if "boxplot" in lowered:
        return "Boxplot"
    if "cross_validation" in lowered:
        return "Cross-validation"
    if "fold" in lowered:
        return "Fold analysis"
    return stem.replace("_", " ").strip().title()


def _comparison_key(path: Path, kind: EvidenceKind, role: str) -> str:
    '''Return a stable key for comparing equivalent evidence figures.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.
    kind : EvidenceKind
        Evidence kind.
    role : str
        Evidence role.

    Returns
    -------
    str
        Grouping key suitable for comparing policies side by side.
    '''

    figure_name = _figure_name(path, kind).lower()
    dataset = _dataset_from_path(path) or "all"
    return "|".join((role, figure_name, dataset))


def _evidence_role(path: Path, kind: EvidenceKind) -> str:
    '''Return a compact role label for one evidence file.

    Parameters
    ----------
    path : pathlib.Path
        Evidence path.
    kind : EvidenceKind
        Evidence kind.

    Returns
    -------
    str
        Role label.
    '''

    name = path.name.lower()
    if kind == "shap" and name.endswith(".csv"):
        return "shap_values"
    if kind == "shap" and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "shap_figure"
    if kind == "optimization" and name.endswith("_optuna_trials.csv"):
        return "optuna_trials"
    if kind == "performance" and name == "baselines_per_fold.csv":
        return "baseline_per_fold"
    if kind == "performance" and name.startswith("cross_validation"):
        return "cross_validation"
    if kind == "figure":
        return "analysis_figure"
    return kind


def _csv_columns(path: Path) -> tuple[str, ...]:
    '''Return a CSV header row.

    Parameters
    ----------
    path : pathlib.Path
        CSV file path.

    Returns
    -------
    tuple[str, ...]
        Header columns.
    '''

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return ()
    return tuple(str(column) for column in header)


def _metric_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    '''Return likely metric columns from a CSV header.

    Parameters
    ----------
    columns : tuple[str, ...]
        CSV header columns.

    Returns
    -------
    tuple[str, ...]
        Candidate metric columns.
    '''

    metrics: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered in IDENTIFIER_COLUMNS:
            continue
        if lowered.startswith("params_") or lowered.startswith("datetime_") or lowered == "duration":
            continue
        if any(
            token in lowered
            for token in (
                "auc",
                "bedroc",
                "ef",
                "ndcg",
                "rmse",
                "mae",
                "mcc",
                "precision",
                "recall",
                "f1",
                "brier",
                "ece",
                "loss",
                "r2",
            )
        ):
            metrics.append(column)
    return tuple(metrics)


def _read_dict_rows(path: Path, *, max_rows: int) -> tuple[dict[str, str], ...]:
    '''Read CSV dictionary rows up to a limit.

    Parameters
    ----------
    path : pathlib.Path
        CSV file path.
    max_rows : int
        Maximum rows to read.

    Returns
    -------
    tuple[dict[str, str], ...]
        CSV rows.
    '''

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
    return tuple(rows)


def _performance_points(path: Path, *, run_id: str, policy_name: str, max_rows: int) -> tuple[dict[str, Any], ...]:
    '''Build aggregate performance points from known OCScore CSV files.

    Parameters
    ----------
    path : pathlib.Path
        Performance CSV path.
    run_id : str
        Workbench run id.
    policy_name : str
        Policy label.
    max_rows : int
        Maximum rows to read.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Plot-ready aggregate points.
    '''

    rows = _read_dict_rows(path, max_rows=max_rows)
    if not rows:
        return ()
    columns = tuple(rows[0].keys())
    metric_names = _metric_columns(columns)
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    name = path.name.lower()
    for row in rows:
        if name == "baselines_per_fold.csv":
            label = row.get("baseline") or path.stem
            group = row.get("baseline_family") or row.get("split") or "baseline"
        elif name == "cross_validation_folds.csv":
            label = _dataset_from_path(path) or path.parent.name
            group = "cross_validation"
        else:
            label = path.stem
            group = row.get("split") or "performance"
        for metric_name in metric_names:
            number = _float(row.get(metric_name))
            if number is not None:
                values[(label, group, metric_name, path.name)].append(number)
    points = []
    for (label, group, metric_name, file_name), numbers in values.items():
        if not numbers:
            continue
        points.append(
            {
                "run_id": run_id,
                "policy_name": policy_name,
                "label": label,
                "group": group,
                "metric_name": metric_name,
                "value": sum(numbers) / len(numbers),
                "count": len(numbers),
                "file_name": file_name,
                "path": str(path),
            }
        )
    return tuple(sorted(points, key=lambda item: (item["metric_name"], item["group"], item["label"])))


def _optimization_points(path: Path, *, run_id: str, policy_name: str, max_rows: int) -> tuple[dict[str, Any], ...]:
    '''Build Optuna trace points from a trials CSV.

    Parameters
    ----------
    path : pathlib.Path
        Optuna trials CSV path.
    run_id : str
        Workbench run id.
    policy_name : str
        Policy label.
    max_rows : int
        Maximum rows to read.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Plot-ready trace points.
    '''

    if not path.name.lower().endswith("_optuna_trials.csv"):
        return ()
    rows = _read_dict_rows(path, max_rows=max_rows)
    dataset = _dataset_from_path(path)
    replica = _replica_from_path(path)
    series = "/".join(part for part in (policy_name, replica, dataset) if part) or path.stem
    points: list[dict[str, Any]] = []
    best_value: float | None = None
    for row in rows:
        trial = _float(row.get("number"))
        value = _float(row.get("value"))
        if trial is None or value is None:
            continue
        best_value = value if best_value is None else max(best_value, value)
        points.append(
            {
                "run_id": run_id,
                "policy_name": policy_name,
                "series": series,
                "trial": int(trial),
                "value": value,
                "best_value": best_value,
                "dataset": dataset,
                "replica": replica,
                "path": str(path),
            }
        )
    return tuple(points)


def _shap_features(
    path: Path, *, run_id: str, policy_name: str, max_rows: int, top_n: int
) -> tuple[dict[str, Any], ...]:
    '''Build mean absolute SHAP feature previews from a CSV matrix.

    Parameters
    ----------
    path : pathlib.Path
        SHAP values CSV path.
    run_id : str
        Workbench run id.
    policy_name : str
        Policy label.
    max_rows : int
        Maximum rows to read.
    top_n : int
        Maximum features to return.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Feature importance previews.
    '''

    if path.name.lower() != "shap_values.csv":
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return ()
        sums = [0.0 for _column in header]
        counts = [0 for _column in header]
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            for index, value in enumerate(row[: len(header)]):
                number = _float(value)
                if number is None:
                    continue
                sums[index] += abs(number)
                counts[index] += 1
    features = []
    dataset = _dataset_from_path(path)
    for name, total, count in zip(header, sums, counts):
        if count:
            features.append(
                {
                    "run_id": run_id,
                    "policy_name": policy_name,
                    "dataset": dataset,
                    "feature": name,
                    "mean_abs_shap": total / count,
                    "sample_count": count,
                    "path": str(path),
                }
            )
    return tuple(sorted(features, key=lambda item: item["mean_abs_shap"], reverse=True)[:top_n])


def _evidence_entry(
    *,
    run_id: str,
    status: RunStatus,
    manifest_path: Path,
    source_path: Path | None,
    policy_name: str,
    path: Path,
    kind: EvidenceKind,
) -> WorkbenchEvidenceEntry:
    '''Build one evidence index entry.

    Parameters
    ----------
    run_id : str
        Workbench run id.
    status : RunStatus
        Run status.
    manifest_path : pathlib.Path
        Result manifest path.
    source_path : pathlib.Path or None
        Adopted source path.
    policy_name : str
        Policy label.
    path : pathlib.Path
        Evidence file path.
    kind : EvidenceKind
        Evidence kind.

    Returns
    -------
    WorkbenchEvidenceEntry
        Evidence entry.
    '''

    columns = _csv_columns(path) if path.suffix.lower() == CSV_SUFFIX else ()
    role = _evidence_role(path, kind)
    return WorkbenchEvidenceEntry(
        run_id=run_id,
        status=status,
        manifest_path=manifest_path,
        source_path=source_path,
        path=path,
        kind=kind,
        role=role,
        dataset=_dataset_from_path(path),
        policy_name=_policy_from_evidence_path(path, policy_name),
        replica=_replica_from_path(path),
        figure_name=_figure_name(path, kind),
        comparison_key=_comparison_key(path, kind, role),
        suffix=path.suffix.lower(),
        size_bytes=_size_bytes(path),
        modified_at=_modified_at(path),
        column_count=len(columns) if columns else None,
        metric_names=_metric_columns(columns),
    )


def _count_by(entries: tuple[WorkbenchEvidenceEntry, ...], field_name: str) -> dict[str, int]:
    '''Count evidence entries by a string field.

    Parameters
    ----------
    entries : tuple[WorkbenchEvidenceEntry, ...]
        Evidence entries.
    field_name : str
        Field to count.

    Returns
    -------
    dict[str, int]
        Sorted counts.
    '''

    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field_name) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


## Public ##


def resolve_evidence_asset(
    root: str | Path,
    asset_path: str | Path,
    *,
    max_depth: int = 6,
    source_depth: int = 6,
) -> tuple[Path, str]:
    '''Resolve a discovered image evidence asset for local API serving.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file used to discover adopted sources.
    asset_path : str or pathlib.Path
        Requested image path. Relative paths are resolved below ``root``.
    max_depth : int
        Maximum Workbench manifest scan depth.
    source_depth : int
        Maximum recursive depth below each adopted source path.

    Returns
    -------
    tuple[pathlib.Path, str]
        Resolved image path and HTTP content type.

    Raises
    ------
    FileNotFoundError
        If the requested asset is absent.
    ValueError
        If the requested path is not a supported discovered image asset.
    '''

    root_path = Path(root)
    requested = Path(asset_path)
    base_path = root_path if root_path.is_dir() else root_path.parent
    candidate = requested if requested.is_absolute() else base_path / requested
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Evidence asset does not exist: {asset_path}") from exc

    content_type = EVIDENCE_ASSET_CONTENT_TYPES.get(resolved.suffix.lower())
    if content_type is None:
        raise ValueError(f"Evidence asset must be a supported image file: {asset_path}")
    if not resolved.is_file():
        raise ValueError(f"Evidence asset must be a file: {asset_path}")

    manifest_paths = tuple(
        sorted(
            discover_result_manifest_paths(root_path, max_depth=max_depth),
            key=lambda path: (path.parent.name != "train", path.parent.name),
        )
    )
    for manifest_path in manifest_paths:
        source_path = _source_path_from_result_manifest(manifest_path)
        for source_root in _source_roots(source_path):
            resolved_root = source_root.resolve(strict=False)
            if not _within_depth(resolved_root, resolved, source_depth):
                continue
            kind = _evidence_kind(resolved)
            if kind in {"shap", "figure"}:
                return resolved, content_type
    raise ValueError(f"Evidence asset is not part of a discovered image evidence source: {asset_path}")


def build_evidence_index(
    root: str | Path,
    *,
    max_depth: int = 6,
    source_depth: int = 6,
    max_entries: int = 400,
    max_csv_rows: int = 1000,
    max_series: int = 8,
    max_shap_features: int = 30,
) -> WorkbenchEvidenceIndex:
    '''Build a read-only evidence index from adopted Workbench sources.

    Parameters
    ----------
    root : str or pathlib.Path
        Workspace root or result manifest file to scan.
    max_depth : int
        Maximum Workbench manifest scan depth.
    source_depth : int
        Maximum recursive depth below each adopted source path.
    max_entries : int
        Maximum evidence entries to return.
    max_csv_rows : int
        Maximum CSV rows read per evidence file for plot previews.
    max_series : int
        Maximum Optuna trial series included in preview points.
    max_shap_features : int
        Maximum SHAP feature previews returned per SHAP CSV file.

    Returns
    -------
    WorkbenchEvidenceIndex
        Evidence index and plot-ready preview points.
    '''

    root_path = Path(root)
    entries: list[WorkbenchEvidenceEntry] = []
    performance_points: list[dict[str, Any]] = []
    optimization_points: list[dict[str, Any]] = []
    shap_features: list[dict[str, Any]] = []
    issues: list[InventoryIssue] = []
    seen_entries: set[tuple[str, Path]] = set()
    optimization_series: set[str] = set()
    result_manifest_count = 0

    manifest_paths = tuple(
        sorted(
            discover_result_manifest_paths(root_path, max_depth=max_depth),
            key=lambda path: (path.parent.name != "train", path.parent.name),
        )
    )
    for manifest_path in manifest_paths:
        try:
            manifest = read_result_manifest(manifest_path)
        except Exception as exc:
            issues.append(InventoryIssue(path=manifest_path, message=str(exc)))
            continue
        result_manifest_count += 1
        source_path = _source_path_from_result_manifest(manifest_path)
        policy_name = _policy_from_path(source_path, manifest.run_id)
        for source_root in _source_roots(source_path):
            for path in _iter_source_files(source_root, source_depth=source_depth):
                kind = _evidence_kind(path)
                if kind is None:
                    continue
                key = (manifest.run_id, path.resolve(strict=False))
                if key in seen_entries:
                    continue
                seen_entries.add(key)
                try:
                    entry_policy_name = _policy_from_evidence_path(path, policy_name)
                    entry = _evidence_entry(
                        run_id=manifest.run_id,
                        status=manifest.status,
                        manifest_path=manifest_path,
                        source_path=source_path,
                        policy_name=policy_name,
                        path=path,
                        kind=kind,
                    )
                    entries.append(entry)
                    if kind == "performance" and len(performance_points) < max_entries:
                        performance_points.extend(
                            _performance_points(
                                path, run_id=manifest.run_id, policy_name=entry_policy_name, max_rows=max_csv_rows
                            )
                        )
                    if kind == "optimization" and path.name.lower().endswith("_optuna_trials.csv"):
                        series = f"{manifest.run_id}:{path}"
                        if series not in optimization_series and len(optimization_series) < max_series:
                            optimization_series.add(series)
                            optimization_points.extend(
                                _optimization_points(
                                    path, run_id=manifest.run_id, policy_name=entry_policy_name, max_rows=max_csv_rows
                                )
                            )
                    if kind == "shap" and path.name.lower() == "shap_values.csv":
                        shap_features.extend(
                            _shap_features(
                                path,
                                run_id=manifest.run_id,
                                policy_name=entry_policy_name,
                                max_rows=max_csv_rows,
                                top_n=max_shap_features,
                            )
                        )
                except Exception as exc:
                    issues.append(InventoryIssue(path=path, message=str(exc)))

    kind_priority = {"shap": 0, "figure": 1, "performance": 2, "optimization": 3, "prediction": 4, "other": 5}
    all_entries = tuple(
        sorted(
            entries,
            key=lambda entry: (
                kind_priority.get(entry.kind, 99),
                entry.suffix not in EVIDENCE_ASSET_CONTENT_TYPES,
                entry.comparison_key,
                entry.dataset,
                entry.policy_name,
                entry.replica,
                entry.run_id != "train",
                entry.run_id,
                entry.role,
                str(entry.path),
            ),
        )
    )
    evidence_entries = all_entries[:max_entries]
    return WorkbenchEvidenceIndex(
        root=root_path,
        max_depth=max_depth,
        source_depth=source_depth,
        result_manifest_count=result_manifest_count,
        evidence_count=len(all_entries),
        kind_counts=_count_by(all_entries, "kind"),
        role_counts=_count_by(all_entries, "role"),
        entries=evidence_entries,
        performance_points=tuple(performance_points[: max_entries * 4]),
        optimization_points=tuple(optimization_points[: max_entries * 4]),
        shap_features=tuple(
            sorted(shap_features, key=lambda item: item["mean_abs_shap"], reverse=True)[:max_shap_features]
        ),
        issue_count=len(issues),
        issues=tuple(issues),
    )


__all__ = ["build_evidence_index", "resolve_evidence_asset"]
