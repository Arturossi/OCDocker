#!/usr/bin/env python3

# Description
###############################################################################
"""
Strict OCScore output layout discovery for Workbench dashboards.
"""

# Imports
###############################################################################
from __future__ import annotations

import csv
import json
import math
import re

from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from OCDocker.OCScore.Optimization.OptunaStorage import DEFAULT_OPTUNA_DB_FILENAME
from statistics import fmean
from statistics import median
from statistics import stdev
from typing import Any, Literal

import yaml

from OCDocker.Workbench.Models import InventoryIssue
from OCDocker.Workbench.Models import OCScoreReplicaStatus
from OCDocker.Workbench.Models import OCScoreWorkspaceRole
from OCDocker.Workbench.Models import WorkbenchOCScoreCrossValidation
from OCDocker.Workbench.Models import WorkbenchOCScoreCrossValidationMetric
from OCDocker.Workbench.Models import WorkbenchOCScoreExternalBaseline
from OCDocker.Workbench.Models import WorkbenchOCScoreFigure
from OCDocker.Workbench.Models import WorkbenchOCScoreMetric
from OCDocker.Workbench.Models import WorkbenchOCScoreBaselineSource
from OCDocker.Workbench.Models import WorkbenchOCScoreProtocolSummary
from OCDocker.Workbench.Models import WorkbenchOCScoreReplica
from OCDocker.Workbench.Models import WorkbenchOCScoreRunContext
from OCDocker.Workbench.Models import WorkbenchOCScoreStudy
from OCDocker.Workbench.Models import WorkbenchOCScoreWorkspace

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

DEFAULT_OCSCORE_REPLICA_COUNT = 5
DEFAULT_OCSCORE_SCAN_DEPTH = 6
MIN_OPTUNA_DASHBOARD_SLOT_COUNT = 1
MAX_OPTUNA_DASHBOARD_SLOT_COUNT = 50
DEFAULT_OCSCORE_MAX_METRIC_FILE_BYTES = 1_048_576
REPLICA_PATTERN = re.compile(r"^replica[_ -]?(?P<index>\d+)$", re.IGNORECASE)
METRIC_SUFFIXES = frozenset({".csv", ".json", ".yaml", ".yml"})
FIGURE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg", ".pdf"})
LOG_SUFFIXES = frozenset({".log", ".out", ".err"})
FAILED_LOG_MARKERS = (
    "traceback",
    "exception",
    "error:",
    "failed",
    "killed",
    "segmentation fault",
)
CURATED_METRIC_DIRECTIONS: dict[str, Literal["max", "min"]] = {
    "bedroc": "max",
    "roc_auc": "max",
    "pr_auc": "max",
    "ef_1": "max",
    "ef_5": "max",
    "validation_metric": "max",
    "rmse": "min",
    "mae": "min",
    "r2": "max",
}
CURATED_METRIC_LABELS: dict[str, str] = {
    "bedroc": "BEDROC",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "ef_1": "EF 1%",
    "ef_5": "EF 5%",
    "validation_metric": "Best validation metric",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R2",
}
EXTERNAL_BASELINE_CSV_METRICS: dict[str, str] = {
    "BEDROC": "bedroc",
    "ROC-AUC": "roc_auc",
    "PR-AUC": "pr_auc",
    "EF1%": "ef_1",
    "EF5%": "ef_5",
    "RMSE": "rmse",
    "MAE": "mae",
    "R2": "r2",
}
EXTERNAL_BASELINE_FAMILY_ALIASES: dict[str, str] = {
    "scoring_function": "scoring_function",
    "learned_sf": "learned_sf",
    "sf_consensus": "sf_consensus",
    "descriptor_aggregate": "descriptor_aggregate",
    "sf": "scoring_function",
}
LEARNED_SF_BASELINE_NAMES = frozenset({"lr_sf", "rf_sf", "xgb_sf", "lgbm_sf", "shuffled_lr_sf"})
SF_CONSENSUS_BASELINE_NAMES = frozenset({"sf_mean", "sf_median", "sf_max", "sf_min"})
DUDEZ_COMPARISON_SKIP_SCORER_TYPES = frozenset({"model", "reference"})

# Functions
###############################################################################
## Private ##


def _issue(path: Path, message: str) -> InventoryIssue:
    '''Build an OCScore layout issue.

    Parameters
    ----------
    path : pathlib.Path
        Path related to the issue.
    message : str
        Issue message.

    Returns
    -------
    InventoryIssue
        Issue payload.
    '''

    return InventoryIssue(path=path, message=message)


def _is_hidden(path: Path) -> bool:
    '''Return whether a path is hidden from OCScore discovery.

    Parameters
    ----------
    path : pathlib.Path
        Path to inspect.

    Returns
    -------
    bool
        True when the path should be ignored.
    '''

    return path.name.startswith(".") or path.name == "__pycache__"


def _iter_files(root: Path, *, max_depth: int) -> tuple[Path, ...]:
    '''Return visible files below a root path up to a maximum depth.

    Parameters
    ----------
    root : pathlib.Path
        Directory to scan.
    max_depth : int
        Maximum recursive depth.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Files ordered by path.
    '''

    if not root.exists() or not root.is_dir():
        return ()
    files: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        directory, depth = queue.pop(0)
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if _is_hidden(child):
                continue
            if child.is_file():
                files.append(child)
            elif child.is_dir() and depth < max_depth:
                queue.append((child, depth + 1))
    return tuple(files)


def _normalise_metric_name(name: str) -> str:
    '''Normalize a raw metric name for conservative matching.

    Parameters
    ----------
    name : str
        Raw metric name.

    Returns
    -------
    str
        Normalized name.
    '''

    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _metric_base_name(name: str) -> str:
    '''Return the curated metric name without an optional split prefix.

    Parameters
    ----------
    name : str
        Scoped canonical metric name.

    Returns
    -------
    str
        Base metric name.
    '''

    for prefix in ("test_", "validation_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _metric_split_tag(normalized: str) -> str | None:
    '''Infer a DUDEz split tag encoded in a raw metric name.

    Parameters
    ----------
    normalized : str
        Normalized raw metric name.

    Returns
    -------
    str or None
        ``test``, ``validation``, or None when no split is encoded.
    '''

    if "dudez_validation" in normalized or normalized.startswith("validation_"):
        return "validation"
    if "dudez_test" in normalized or normalized.startswith("test_"):
        return "test"
    return None


def _strip_split_markers(normalized: str) -> str:
    '''Remove known split markers from a normalized metric name.

    Parameters
    ----------
    normalized : str
        Normalized raw metric name.

    Returns
    -------
    str
        Metric name core used for canonical classification.
    '''

    stripped = normalized
    for token in ("dudez_validation", "dudez_test"):
        stripped = stripped.replace(f"{token}_", "")
        if stripped.startswith(f"{token}"):
            stripped = stripped[len(token) :].lstrip("_")
    stripped = stripped.strip("_")
    return stripped or normalized


def _classify_metric_core(normalized: str, requested: set[str] | None) -> str | None:
    '''Classify one normalized metric core into the curated OCScore metric set.

    Parameters
    ----------
    normalized : str
        Normalized metric name without split markers.
    requested : set[str] or None
        Optional normalized user-requested metric names.

    Returns
    -------
    str or None
        Base canonical metric name, or None when intentionally ignored.
    '''

    if requested and normalized in requested:
        return normalized
    if "bedroc" in normalized:
        return "bedroc"
    if "pr_auc" in normalized or "prauc" in normalized or "average_precision" in normalized:
        return "pr_auc"
    if "roc_auc" in normalized or "auroc" in normalized or normalized.endswith("_auc"):
        return "roc_auc"
    if "ef_1" in normalized or "ef1" in normalized or "enrichment_factor_1" in normalized:
        return "ef_1"
    if "ef_5" in normalized or "ef5" in normalized or "enrichment_factor_5" in normalized:
        return "ef_5"
    if "primary_metric" in normalized or "best_validation_metric" in normalized or normalized.endswith("validation_metric"):
        return "validation_metric"
    if normalized.endswith("rmse") or normalized == "rmse":
        return "rmse"
    if normalized.endswith("mae") or normalized == "mae":
        return "mae"
    if normalized.endswith("r2") or normalized == "r2" or normalized.endswith("r_squared"):
        return "r2"
    if requested and any(token in normalized for token in requested):
        return normalized
    return None


def _classify_metric(name: str, requested: set[str] | None) -> str | None:
    '''Classify one raw metric name into the curated OCScore metric set.

    Parameters
    ----------
    name : str
        Raw metric name.
    requested : set[str] or None
        Optional normalized user-requested metric names.

    Returns
    -------
    str or None
        Scoped canonical metric name, or None when intentionally ignored.
    '''

    normalized = _normalise_metric_name(name)
    if requested and normalized in requested:
        return normalized
    split = _metric_split_tag(normalized)
    core = _strip_split_markers(normalized) if split else normalized
    base = _classify_metric_core(core, requested)
    if base is None:
        return None
    if split:
        if base == "validation_metric" and split == "validation":
            return "validation_metric"
        return f"{split}_{base}"
    if base in {"bedroc", "roc_auc", "pr_auc", "ef_1", "ef_5"}:
        return f"test_{base}"
    if base == "validation_metric":
        return "validation_metric"
    return base


def _numeric(value: Any) -> float | None:
    '''Convert a value to a finite float when possible.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    float or None
        Numeric value or None.
    '''

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _record_metric(
    values: dict[str, list[float]],
    sources: dict[str, set[Path]],
    metric_name: str,
    raw_value: Any,
    source_path: Path,
    requested: set[str] | None,
) -> None:
    '''Record one numeric value when its metric name is curated.

    Parameters
    ----------
    values : dict[str, list[float]]
        Metric value accumulator.
    sources : dict[str, set[pathlib.Path]]
        Metric source accumulator.
    metric_name : str
        Raw metric name.
    raw_value : Any
        Raw value.
    source_path : pathlib.Path
        Source file path.
    requested : set[str] or None
        Optional requested metric names.
    '''

    canonical = _classify_metric(metric_name, requested)
    number = _numeric(raw_value)
    if canonical is None or number is None:
        return
    values[canonical].append(number)
    sources[canonical].add(source_path)


def _read_csv_metrics(path: Path, requested: set[str] | None) -> tuple[dict[str, list[float]], dict[str, set[Path]]]:
    '''Read curated metrics from one CSV file.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.
    requested : set[str] or None
        Optional requested metric names.

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, set[pathlib.Path]]]
        Metric values and source paths.
    '''

    values: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[Path]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            return values, sources
        fields = tuple(reader.fieldnames)
        metric_field = next(
            (field for field in fields if _normalise_metric_name(field) in {"metric", "metric_name", "name"}), None
        )
        value_field = next(
            (field for field in fields if _normalise_metric_name(field) in {"value", "mean", "score"}), None
        )
        for row in reader:
            if metric_field and value_field:
                _record_metric(values, sources, str(row.get(metric_field, "")), row.get(value_field), path, requested)
                continue
            for field in fields:
                _record_metric(values, sources, field, row.get(field), path, requested)
    return values, sources


def _flatten_mapping(data: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    '''Flatten a nested mapping or sequence into dotted metric paths.

    Parameters
    ----------
    data : Any
        Input object.
    prefix : str
        Current key prefix.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        Flattened key/value pairs.
    '''

    pairs: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            pairs.extend(_flatten_mapping(value, name))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            name = f"{prefix}.{index}" if prefix else str(index)
            pairs.extend(_flatten_mapping(value, name))
    else:
        pairs.append((prefix, data))
    return tuple(pairs)


def _read_structured_metrics(
    path: Path, requested: set[str] | None
) -> tuple[dict[str, list[float]], dict[str, set[Path]]]:
    '''Read curated metrics from one JSON or YAML file.

    Parameters
    ----------
    path : pathlib.Path
        Structured metrics path.
    requested : set[str] or None
        Optional requested metric names.

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, set[pathlib.Path]]]
        Metric values and source paths.
    '''

    values: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[Path]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    for name, raw_value in _flatten_mapping(data):
        _record_metric(values, sources, name, raw_value, path, requested)
    return values, sources


def _merge_metric_values(
    target_values: dict[str, list[float]],
    target_sources: dict[str, set[Path]],
    values: dict[str, list[float]],
    sources: dict[str, set[Path]],
) -> None:
    '''Merge parsed metric accumulators.

    Parameters
    ----------
    target_values : dict[str, list[float]]
        Target value accumulator.
    target_sources : dict[str, set[pathlib.Path]]
        Target source accumulator.
    values : dict[str, list[float]]
        Source value accumulator.
    sources : dict[str, set[pathlib.Path]]
        Source path accumulator.
    '''

    for name, numbers in values.items():
        target_values[name].extend(numbers)
        target_sources[name].update(sources.get(name, ()))


def _metric_direction(name: str) -> Literal["max", "min"]:
    '''Return the optimization direction for one metric.

    Parameters
    ----------
    name : str
        Canonical metric name.

    Returns
    -------
    Literal["max", "min"]
        ``max`` or ``min``.
    '''

    return CURATED_METRIC_DIRECTIONS.get(_metric_base_name(name), "max")


def _metric_label(name: str) -> str:
    '''Return the display label for one metric.

    Parameters
    ----------
    name : str
        Canonical metric name.

    Returns
    -------
    str
        Display label.
    '''

    if name.startswith("test_"):
        base = _metric_base_name(name)
        base_label = CURATED_METRIC_LABELS.get(base, base.replace("_", " ").title())
        return f"Test {base_label}"
    if name.startswith("validation_"):
        base = _metric_base_name(name)
        base_label = CURATED_METRIC_LABELS.get(base, base.replace("_", " ").title())
        return f"Validation {base_label}"
    return CURATED_METRIC_LABELS.get(name, name.replace("_", " ").title())


def _build_metric_models(
    values: dict[str, list[float]], sources: dict[str, set[Path]]
) -> tuple[WorkbenchOCScoreMetric, ...]:
    '''Build per-replica metric models from parsed values.

    Parameters
    ----------
    values : dict[str, list[float]]
        Metric value accumulator.
    sources : dict[str, set[pathlib.Path]]
        Metric source accumulator.

    Returns
    -------
    tuple[WorkbenchOCScoreMetric, ...]
        Metric models ordered by name.
    '''

    metrics: list[WorkbenchOCScoreMetric] = []
    for name in sorted(values):
        numbers = values[name]
        if numbers:
            metrics.append(
                WorkbenchOCScoreMetric(
                    name=name,
                    label=_metric_label(name),
                    direction=_metric_direction(name),
                    value=fmean(numbers),
                    observation_count=len(numbers),
                    source_paths=tuple(sorted(sources.get(name, ()))),
                )
            )
    return tuple(metrics)


def _parse_replica_index(path: Path) -> int | None:
    '''Parse a replica index from a directory name.

    Parameters
    ----------
    path : pathlib.Path
        Replica path.

    Returns
    -------
    int or None
        Parsed index.
    '''

    match = REPLICA_PATTERN.match(path.name)
    return int(match.group("index")) if match is not None else None


def _replica_sort_key(path: Path) -> tuple[int, str]:
    '''Return a stable sort key for replica directories.

    Parameters
    ----------
    path : pathlib.Path
        Replica path.

    Returns
    -------
    tuple[int, str]
        Sort key.
    '''

    index = _parse_replica_index(path)
    return (index if index is not None else 10**9, path.name)


def _collect_replica_paths(parent: Path) -> dict[int, Path]:
    '''Collect replica directories below a study path.

    Parameters
    ----------
    parent : pathlib.Path
        Study directory.

    Returns
    -------
    dict[int, pathlib.Path]
        One-based replica slot to existing path.
    '''

    if not parent.exists() or not parent.is_dir():
        return {}
    candidates = sorted(
        (child for child in parent.iterdir() if child.is_dir() and REPLICA_PATTERN.match(child.name)),
        key=_replica_sort_key,
    )
    indexes = [index for index in (_parse_replica_index(path) for path in candidates) if index is not None]
    zero_based = bool(indexes) and min(indexes) == 0
    paths: dict[int, Path] = {}
    for path in candidates:
        index = _parse_replica_index(path)
        if index is None:
            continue
        slot = index + 1 if zero_based else index
        paths.setdefault(slot, path)
    return paths


def _file_size(path: Path) -> int | None:
    '''Return a file size when available.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    int or None
        File size.
    '''

    try:
        return path.stat().st_size
    except OSError:
        return None


def _modified_at(path: Path) -> datetime | None:
    '''Return a file modification timestamp when available.

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    datetime or None
        Modification timestamp.
    '''

    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _figure_role(path: Path) -> str:
    '''Infer a compact figure role from a path.

    Parameters
    ----------
    path : pathlib.Path
        Figure path.

    Returns
    -------
    str
        Figure role.
    '''

    normalized = _normalise_metric_name("_".join(path.with_suffix("").parts[-4:]))
    tokens = set(normalized.split("_"))
    if "beeswarm" in tokens:
        return "shap_beeswarm"
    if "feature_importance" in normalized or "importance" in tokens:
        return "shap_importance"
    if "dependence" in tokens:
        return "shap_dependence"
    if "shap" in tokens:
        return "shap"
    if "cv_mean_std" in normalized or "mean_std" in normalized:
        return "cv_mean_std"
    if "cv_heatmap" in normalized or "heatmap" in tokens:
        return "cv_heatmap"
    if "cv_fold_comparison" in normalized or "fold_comparison" in normalized:
        return "cv_fold_comparison"
    if "per_target" in normalized or "target_validation" in normalized or "validation_by_target" in normalized:
        return "per_target_validation"
    if "architecture" in tokens or "model_structure" in normalized:
        return "architecture"
    if "optuna" in tokens:
        return "optuna"
    if tokens.intersection({"roc", "bedroc", "performance"}) or "roc_auc" in normalized:
        return "performance"
    return "figure"


def _figure_metric(path: Path) -> str:
    '''Infer a curated OCScore metric name from a figure path.

    Parameters
    ----------
    path : pathlib.Path
        Figure path.

    Returns
    -------
    str
        Canonical metric name, or an empty string.
    '''

    normalized = _normalise_metric_name("_".join(path.with_suffix("").parts[-4:]))
    canonical = _classify_metric(normalized, None)
    if canonical is not None:
        return canonical
    tokens = set(normalized.split("_"))
    for name in ("rmse", "mae", "r2"):
        if name in tokens:
            return name
    return ""


def _figure_dataset(path: Path) -> str:
    '''Infer a known OCScore dataset name from a path.

    Parameters
    ----------
    path : pathlib.Path
        Figure path.

    Returns
    -------
    str
        Dataset name, or an empty string.
    '''

    for part in reversed(path.parts):
        normalized = _normalise_metric_name(part)
        if normalized in {"dudez", "pdbbind", "casf", "dekois", "lit_pcba"}:
            return normalized
    return ""


def _log_has_failure(path: Path, *, max_bytes: int) -> bool:
    '''Return whether a log file contains a common failure marker.

    Parameters
    ----------
    path : pathlib.Path
        Log path.
    max_bytes : int
        Maximum bytes to inspect.

    Returns
    -------
    bool
        True when a failure marker is present.
    '''

    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:max_bytes].lower()
    except OSError:
        return False
    return any(marker in text for marker in FAILED_LOG_MARKERS)


REPLICA_STAGE_NAMES = ("pdbbind_optuna", "dudez_optuna")


def _read_replica_protocol_log(replica_path: Path) -> dict[str, Any] | None:
    '''Return parsed ``protocol_log.json`` for one replica when readable.'''

    protocol_log_path = replica_path / "protocol_log.json"
    if not protocol_log_path.is_file():
        return None
    try:
        payload = json.loads(protocol_log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _protocol_stage_result(payload: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return None
    for entry in stages:
        if not isinstance(entry, dict) or entry.get("name") != stage_name:
            continue
        result = entry.get("result")
        return result if isinstance(result, dict) else None
    return None


def _stage_result_complete(stage_result: dict[str, Any]) -> bool:
    checkpoint = stage_result.get("checkpoint_path")
    if isinstance(checkpoint, str) and checkpoint and Path(checkpoint).is_file():
        return True
    export_dir = stage_result.get("best_model_export_dir")
    if isinstance(export_dir, str) and export_dir and Path(export_dir).is_dir():
        return True
    if stage_result.get("test_metrics") or stage_result.get("validation_metrics"):
        return True
    return False


def _replica_stage_complete_from_protocol(replica_path: Path, stage_name: str) -> bool:
    payload = _read_replica_protocol_log(replica_path)
    if payload is None:
        return False
    stage_result = _protocol_stage_result(payload, stage_name)
    if stage_result is None:
        return False
    return _stage_result_complete(stage_result)


def _replica_pdbbind_complete(replica_path: Path) -> bool:
    if _replica_stage_complete_from_protocol(replica_path, "pdbbind_optuna"):
        return True
    pdbbind_dir = replica_path / "pdbbind"
    if (pdbbind_dir / "pdbbind_best.pt").is_file():
        return True
    return (pdbbind_dir / "best_model").is_dir()


def _replica_dudez_complete(replica_path: Path) -> bool:
    if _replica_stage_complete_from_protocol(replica_path, "dudez_optuna"):
        return True
    dudez_dir = replica_path / "dudez"
    if (dudez_dir / "dudez_best.pt").is_file():
        return True
    if (dudez_dir / "summary.json").is_file():
        return True
    summary_path = replica_path / "summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            metrics = payload.get("aggregate_summary", {}).get("metrics", {})
            if isinstance(metrics, dict) and any("dudez" in str(name).lower() for name in metrics):
                return True
    return False


def _replica_pipeline_complete(replica_path: Path) -> bool:
    return _replica_pdbbind_complete(replica_path) and _replica_dudez_complete(replica_path)


def _resolve_replica_status(
    replica_path: Path,
    log_files: tuple[Path, ...],
    *,
    max_metric_file_bytes: int,
) -> OCScoreReplicaStatus:
    '''Classify replica progress: missing, failed, completed, or running.'''

    if any(_log_has_failure(path, max_bytes=max_metric_file_bytes) for path in log_files):
        return "failed"
    if _replica_pipeline_complete(replica_path):
        return "completed"
    return "running"


def _read_replica_metrics(
    replica_path: Path,
    *,
    requested: set[str] | None,
    max_depth: int,
    max_metric_file_bytes: int,
) -> tuple[tuple[WorkbenchOCScoreMetric, ...], tuple[str, ...]]:
    '''Read curated metrics from one replica directory.

    Parameters
    ----------
    replica_path : pathlib.Path
        Replica directory.
    requested : set[str] or None
        Optional requested metric names.
    max_depth : int
        Maximum recursive depth.
    max_metric_file_bytes : int
        Maximum metric file size to parse.
    figures : tuple[WorkbenchOCScoreFigure, ...]
        Study-level figures discovered outside replica directories.

    Returns
    -------
    tuple[tuple[WorkbenchOCScoreMetric, ...], tuple[str, ...]]
        Metrics and non-fatal issue messages.
    '''

    values: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[Path]] = defaultdict(set)
    issues: list[str] = []
    for path in _iter_files(replica_path, max_depth=max_depth):
        if path.suffix.lower() not in METRIC_SUFFIXES:
            continue
        size = _file_size(path)
        if size is None or size > max_metric_file_bytes:
            continue
        try:
            if path.suffix.lower() == ".csv":
                parsed_values, parsed_sources = _read_csv_metrics(path, requested)
            else:
                parsed_values, parsed_sources = _read_structured_metrics(path, requested)
        except Exception as exc:
            issues.append(f"Could not parse metrics from {path}: {exc}")
            continue
        _merge_metric_values(values, sources, parsed_values, parsed_sources)
    return _build_metric_models(values, sources), tuple(issues)


def _build_figures(
    replica_path: Path, *, policy_name: str, replica_name: str, max_depth: int
) -> tuple[WorkbenchOCScoreFigure, ...]:
    '''Build figure records for one replica or study export directory.

    Parameters
    ----------
    replica_path : pathlib.Path
        Replica or study export directory.
    policy_name : str
        Policy or study name.
    replica_name : str
        Replica label.
    max_depth : int
        Maximum recursive depth.

    Returns
    -------
    tuple[WorkbenchOCScoreFigure, ...]
        Figure records.
    '''

    figures: list[WorkbenchOCScoreFigure] = []
    for path in _iter_files(replica_path, max_depth=max_depth):
        if path.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        figures.append(
            WorkbenchOCScoreFigure(
                path=path,
                role=_figure_role(path),
                dataset=_figure_dataset(path),
                metric_name=_figure_metric(path),
                policy_name=policy_name,
                replica_name=replica_name,
                suffix=path.suffix.lower(),
                size_bytes=_file_size(path),
                modified_at=_modified_at(path),
            )
        )
    return tuple(figures)


def _output_root_for_layout(layout_root: Path) -> Path:
    '''Return the OCScore output root for a resolved train/layout root.

    Parameters
    ----------
    layout_root : pathlib.Path
        Resolved strict layout root.

    Returns
    -------
    pathlib.Path
        OCScore output root.
    '''

    return layout_root.parent if layout_root.name == "train" else layout_root


def _study_export_path(layout_root: Path, *, role: OCScoreWorkspaceRole, study_name: str) -> Path:
    '''Return the expected export figure directory for one study.

    Parameters
    ----------
    layout_root : pathlib.Path
        Resolved strict layout root.
    role : str
        Study role.
    study_name : str
        Study name.

    Returns
    -------
    pathlib.Path
        Expected export directory.
    '''

    export_root = _output_root_for_layout(layout_root) / "export"
    if role == "ablation":
        return export_root / "ablations" / study_name
    return export_root


def _build_baseline_export_figures(
    layout_root: Path, *, policy_name: str, replica_name: str, max_depth: int
) -> tuple[WorkbenchOCScoreFigure, ...]:
    '''Build baseline figures while excluding ablation export children.

    Parameters
    ----------
    layout_root : pathlib.Path
        Resolved strict layout root.
    policy_name : str
        Policy name.
    replica_name : str
        Figure scope label.
    max_depth : int
        Maximum recursive depth.

    Returns
    -------
    tuple[WorkbenchOCScoreFigure, ...]
        Baseline export figures.
    '''

    export_root = _study_export_path(layout_root, role="baseline", study_name="baseline")
    if not export_root.is_dir():
        return ()
    figures: list[WorkbenchOCScoreFigure] = []
    for child in sorted(export_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name in {"ablations", "replica_analysis"} or _is_hidden(child):
            continue
        figures.extend(
            _build_figures(child, policy_name=policy_name, replica_name=replica_name, max_depth=max_depth)
        )
    return tuple(figures)


def _build_replica(
    *,
    role: OCScoreWorkspaceRole,
    study_name: str,
    policy_name: str,
    replica_index: int,
    replica_path: Path | None,
    requested: set[str] | None,
    max_depth: int,
    max_metric_file_bytes: int,
) -> WorkbenchOCScoreReplica:
    '''Build a strict OCScore replica record.

    Parameters
    ----------
    role : str
        Study role.
    study_name : str
        Study name.
    policy_name : str
        Policy name.
    replica_index : int
        One-based replica index.
    replica_path : pathlib.Path or None
        Existing path, or None for a missing expected replica.
    requested : set[str] or None
        Optional requested metric names.
    max_depth : int
        Maximum recursive depth.
    max_metric_file_bytes : int
        Maximum metric file size to parse.

    Returns
    -------
    WorkbenchOCScoreReplica
        Replica record.
    '''

    if replica_path is None:
        return WorkbenchOCScoreReplica(
            role=role,
            study_name=study_name,
            policy_name=policy_name,
            replica_name=f"replica_{replica_index}",
            replica_index=replica_index,
            path=Path(""),
            exists=False,
            status="missing",
        )
    files = _iter_files(replica_path, max_depth=max_depth)
    log_files = tuple(path for path in files if path.suffix.lower() in LOG_SUFFIXES)
    metrics, metric_issues = _read_replica_metrics(
        replica_path,
        requested=requested,
        max_depth=max_depth,
        max_metric_file_bytes=max_metric_file_bytes,
    )
    figures = _build_figures(replica_path, policy_name=policy_name, replica_name=replica_path.name, max_depth=max_depth)
    optuna_db = replica_path / DEFAULT_OPTUNA_DB_FILENAME
    optuna_storage_path = optuna_db.resolve() if optuna_db.is_file() else None
    status = _resolve_replica_status(
        replica_path,
        log_files,
        max_metric_file_bytes=max_metric_file_bytes,
    )
    return WorkbenchOCScoreReplica(
        role=role,
        study_name=study_name,
        policy_name=policy_name,
        replica_name=replica_path.name,
        replica_index=replica_index,
        path=replica_path,
        exists=True,
        status=status,
        optuna_storage_path=optuna_storage_path,
        metrics=metrics,
        figures=figures,
        log_files=log_files,
        issues=metric_issues,
    )


def _metric_summary(replicas: tuple[WorkbenchOCScoreReplica, ...]) -> dict[str, dict[str, Any]]:
    '''Summarize curated metrics across replicas.

    Parameters
    ----------
    replicas : tuple[WorkbenchOCScoreReplica, ...]
        Replica records.

    Returns
    -------
    dict[str, dict[str, Any]]
        Metric summary keyed by canonical metric name.
    '''

    values: dict[str, list[float]] = defaultdict(list)
    for replica in replicas:
        for metric in replica.metrics:
            values[metric.name].append(metric.value)
    summary: dict[str, dict[str, Any]] = {}
    for name in sorted(values):
        numbers = values[name]
        summary[name] = {
            "label": _metric_label(name),
            "direction": _metric_direction(name),
            "count": len(numbers),
            "mean": fmean(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "std": stdev(numbers) if len(numbers) > 1 else 0.0,
        }
    return summary


def _external_baseline_family(raw: str) -> str:
    '''Normalize an external baseline family label.

    Parameters
    ----------
    raw : str
        Raw family label from CSV.

    Returns
    -------
    str
        Normalized family label.
    '''

    normalized = _normalise_metric_name(raw)
    return EXTERNAL_BASELINE_FAMILY_ALIASES.get(normalized, normalized or "other")


def _infer_baseline_family(baseline_name: str, raw_family: str = "") -> str:
    '''Infer baseline family from explicit CSV value or baseline name patterns.'''

    if raw_family.strip():
        family = _external_baseline_family(raw_family)
        if family != "other":
            return family
    name = str(baseline_name or "").strip()
    if name in LEARNED_SF_BASELINE_NAMES:
        return "learned_sf"
    if name.startswith("sf_"):
        return "sf_consensus"
    if name.startswith("desc_"):
        return "descriptor_aggregate"
    return "scoring_function"


def _merge_external_baseline_rows(
    existing: WorkbenchOCScoreExternalBaseline,
    incoming: WorkbenchOCScoreExternalBaseline,
) -> WorkbenchOCScoreExternalBaseline:
    '''Merge duplicate baseline rows, preferring richer metric coverage.'''

    merged_summary = dict(existing.metric_summary)
    merged_summary.update(incoming.metric_summary)
    family = incoming.baseline_family if incoming.baseline_family not in {"", "other"} else existing.baseline_family
    return WorkbenchOCScoreExternalBaseline(
        baseline_name=existing.baseline_name,
        baseline_family=family,
        split=existing.split,
        path=incoming.path,
        metric_summary=merged_summary,
        n_replicas=max(existing.n_replicas, incoming.n_replicas),
        synthesized=existing.synthesized or incoming.synthesized,
    )


def _external_baseline_search_paths(layout_root: Path, *, max_depth: int = 4) -> tuple[Path, ...]:
    '''Return candidate external-baseline CSV paths for one workspace root.'''

    output_root = _output_root_for_layout(layout_root)
    explicit = (
        layout_root / "baselines_summary.csv",
        output_root / "baselines_summary.csv",
        output_root / "export" / "baselines_summary.csv",
        layout_root / "dudez_sf_baseline_comparison.csv",
        output_root / "dudez_sf_baseline_comparison.csv",
        output_root / "export" / "dudez_sf_baseline_comparison.csv",
    )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in explicit:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        ordered.append(candidate)
    for root in (layout_root, output_root):
        for path in _iter_files(root, max_depth=max_depth):
            if path.name not in {"baselines_summary.csv", "dudez_sf_baseline_comparison.csv"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ordered.append(path)
    return tuple(ordered)


def _external_baseline_row(
    *,
    baseline_name: str,
    baseline_family: str,
    split: str,
    source_path: Path,
    metric_summary: dict[str, dict[str, Any]],
    n_replicas: int = 0,
    synthesized: bool = False,
) -> WorkbenchOCScoreExternalBaseline | None:
    '''Build one external baseline row when metrics are present.'''

    if not baseline_name or not metric_summary:
        return None
    return WorkbenchOCScoreExternalBaseline(
        baseline_name=baseline_name,
        baseline_family=_infer_baseline_family(baseline_name, baseline_family),
        split=split or "test",
        path=source_path,
        metric_summary=metric_summary,
        n_replicas=max(0, int(n_replicas)),
        synthesized=bool(synthesized),
    )


def _external_baseline_metric_summary(row: dict[str, str], *, split: str) -> dict[str, dict[str, Any]]:
    '''Build curated metric summary for one external baseline CSV row.

    Parameters
    ----------
    row : dict[str, str]
        One baselines_summary.csv row.
    split : str
        Dataset split for the row.

    Returns
    -------
    dict[str, dict[str, Any]]
        Metric summary keyed by scoped canonical metric name.
    '''

    summary: dict[str, dict[str, Any]] = {}
    split_tag = split if split in {"test", "validation"} else "test"
    for column, canonical in EXTERNAL_BASELINE_CSV_METRICS.items():
        number = _numeric(row.get(column))
        if number is None:
            continue
        scoped = f"{split_tag}_{canonical}"
        summary[scoped] = {
            "label": _metric_label(scoped),
            "direction": _metric_direction(scoped),
            "count": 1,
            "mean": number,
            "min": number,
            "max": number,
            "std": 0.0,
        }
    return summary


def _load_baselines_summary_csv(summary_path: Path) -> tuple[WorkbenchOCScoreExternalBaseline, ...]:
    '''Load rows from ``baselines_summary.csv``.'''

    if not summary_path.is_file() or summary_path.stat().st_size < 8:
        return ()
    rows: list[WorkbenchOCScoreExternalBaseline] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return ()
        for row in reader:
            baseline_name = str(row.get("baseline", "")).strip()
            split = str(row.get("split", "test")).strip() or "test"
            metric_summary = _external_baseline_metric_summary(row, split=split)
            n_replicas = _numeric(row.get("n_replicas"))
            item = _external_baseline_row(
                baseline_name=baseline_name,
                baseline_family=str(row.get("baseline_family", "")).strip(),
                split=split,
                source_path=summary_path,
                metric_summary=metric_summary,
                n_replicas=int(n_replicas) if n_replicas is not None else 0,
            )
            if item is not None:
                rows.append(item)
    return tuple(rows)


def _load_dudez_baseline_comparison_csv(comparison_path: Path) -> tuple[WorkbenchOCScoreExternalBaseline, ...]:
    '''Load pooled rows from ``dudez_sf_baseline_comparison.csv``.'''

    if not comparison_path.is_file() or comparison_path.stat().st_size < 8:
        return ()
    rows: list[WorkbenchOCScoreExternalBaseline] = []
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return ()
        for row in reader:
            scorer_type = str(row.get("scorer_type", "")).strip().lower()
            if scorer_type in DUDEZ_COMPARISON_SKIP_SCORER_TYPES:
                continue
            baseline_name = str(row.get("scorer", row.get("baseline", ""))).strip()
            split = str(row.get("split", "test")).strip() or "test"
            metric_summary = _external_baseline_metric_summary(row, split=split)
            item = _external_baseline_row(
                baseline_name=baseline_name,
                baseline_family=scorer_type,
                split=split,
                source_path=comparison_path,
                metric_summary=metric_summary,
                n_replicas=0,
            )
            if item is not None:
                rows.append(item)
    return tuple(rows)


def _aggregate_sf_consensus_value(values: tuple[float, ...], aggregator: str) -> float:
    '''Reduce one metric across individual scoring-function baselines.'''

    if aggregator == "mean":
        return float(fmean(values))
    if aggregator == "median":
        return float(median(values))
    if aggregator == "max":
        return float(max(values))
    if aggregator == "min":
        return float(min(values))
    raise ValueError(f"Unknown SF consensus aggregator: {aggregator}")


def _synthesize_sf_consensus_baselines(
    baselines: tuple[WorkbenchOCScoreExternalBaseline, ...],
) -> tuple[WorkbenchOCScoreExternalBaseline, ...]:
    '''Build sf_mean/sf_median/sf_max/sf_min rows from individual SF baselines.

    When production baseline CSVs omit row aggregates, derive headline metrics by
    aggregating each scoped metric across ``scoring_function`` rows in the same split.
    '''

    existing = {(item.baseline_name, item.split) for item in baselines}
    by_split: dict[str, list[WorkbenchOCScoreExternalBaseline]] = defaultdict(list)
    for baseline in baselines:
        if baseline.baseline_family != "scoring_function":
            continue
        by_split[baseline.split].append(baseline)

    synthesized: list[WorkbenchOCScoreExternalBaseline] = []
    for split, sf_rows in sorted(by_split.items()):
        if not sf_rows:
            continue
        metric_keys = sorted({key for row in sf_rows for key in row.metric_summary})
        for aggregator in ("mean", "median", "max", "min"):
            baseline_name = f"sf_{aggregator}"
            if (baseline_name, split) in existing:
                continue
            metric_summary: dict[str, dict[str, Any]] = {}
            for metric_key in metric_keys:
                values = tuple(
                    float(row.metric_summary[metric_key]["mean"])
                    for row in sf_rows
                    if metric_key in row.metric_summary
                    and _numeric(row.metric_summary[metric_key].get("mean")) is not None
                )
                if not values:
                    continue
                metric_summary[metric_key] = {
                    "label": _metric_label(metric_key),
                    "direction": sf_rows[0].metric_summary[metric_key].get("direction", _metric_direction(metric_key)),
                    "count": 1,
                    "mean": _aggregate_sf_consensus_value(values, aggregator),
                    "min": min(values),
                    "max": max(values),
                    "std": float(stdev(values)) if len(values) > 1 else 0.0,
                }
            if not metric_summary:
                continue
            item = _external_baseline_row(
                baseline_name=baseline_name,
                baseline_family="sf_consensus",
                split=split,
                source_path=sf_rows[0].path,
                metric_summary=metric_summary,
                n_replicas=0,
                synthesized=True,
            )
            if item is not None:
                synthesized.append(item)
                existing.add((baseline_name, split))
    return tuple(synthesized)


def _load_external_baselines(study_path: Path) -> tuple[WorkbenchOCScoreExternalBaseline, ...]:
    '''Load external baseline rows from workspace baseline CSV reports.

    Parameters
    ----------
    study_path : pathlib.Path
        Study or workspace root that may contain baseline CSV reports.

    Returns
    -------
    tuple[WorkbenchOCScoreExternalBaseline, ...]
        External baseline rows ordered by split and baseline name.
    '''

    merged: dict[tuple[str, str], WorkbenchOCScoreExternalBaseline] = {}
    for candidate in _external_baseline_search_paths(study_path):
        if candidate.name == "dudez_sf_baseline_comparison.csv":
            loaded = _load_dudez_baseline_comparison_csv(candidate)
        else:
            loaded = _load_baselines_summary_csv(candidate)
        for item in loaded:
            key = (item.baseline_name, item.split)
            if key in merged:
                merged[key] = _merge_external_baseline_rows(merged[key], item)
            else:
                merged[key] = item
    loaded = tuple(sorted(merged.values(), key=lambda item: (item.split, item.baseline_family, item.baseline_name)))
    return loaded + _synthesize_sf_consensus_baselines(loaded)


def _protocol_int(value: Any) -> int | None:
    '''Return an integer protocol field when present.'''

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _protocol_float(value: Any) -> float | None:
    '''Return a float protocol field when present.'''

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _protocol_stage_config(payload: dict[str, Any], stage_name: str) -> dict[str, Any]:
    '''Return one stage config block from a protocol JSON payload.'''

    for stage in payload.get("stage_list") or []:
        if not isinstance(stage, dict) or stage.get("name") != stage_name:
            continue
        config = stage.get("config")
        return config if isinstance(config, dict) else {}
    return {}


def _load_ablation_campaign(layout_root: Path) -> tuple[str, tuple[str, ...]]:
    '''Return ablation campaign metadata from ablation_summary.json when present.'''

    for container in _ablation_containers(layout_root):
        summary_path = container / "ablation_summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        protocol_name = str(payload.get("protocol") or "")
        variants: list[str] = []
        for item in payload.get("variants") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("feature_policy_name") or item.get("variant") or item.get("study_name")
            if name:
                variants.append(str(name))
        return protocol_name, tuple(dict.fromkeys(variants))
    return "", ()


def _load_protocol_summary(
    study_path: Path,
    *,
    feature_policy: str = "",
    ablation_variants: tuple[str, ...] = (),
) -> WorkbenchOCScoreProtocolSummary | None:
    '''Load curated protocol metadata for one OCScore study root.'''

    if not study_path.is_dir():
        return None

    json_payload: dict[str, Any] = {}
    yaml_payload: dict[str, Any] = {}
    source_path: Path | None = None
    source_kind = ""
    markdown_path: Path | None = None

    for name, kind in (
        ("replicas_protocol.json", "replicas_protocol"),
        ("staged_optuna_protocol.json", "staged_optuna_protocol"),
    ):
        candidate = study_path / name
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(loaded, dict):
            continue
        json_payload = {**json_payload, **loaded}
        if source_path is None:
            source_path = candidate
            source_kind = kind

    markdown_candidate = study_path / "staged_optuna_protocol.md"
    if markdown_candidate.is_file():
        markdown_path = markdown_candidate

    yaml_candidate = study_path / "protocol.generated.yml"
    if yaml_candidate.is_file():
        try:
            loaded = yaml.safe_load(yaml_candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            yaml_payload = loaded
            if source_path is None:
                source_path = yaml_candidate
                source_kind = "protocol_generated"

    if not json_payload and not yaml_payload:
        return None

    pdbbind = _protocol_stage_config(json_payload, "pdbbind_optuna")
    dudez = _protocol_stage_config(json_payload, "dudez_optuna")
    split_cfg = pdbbind.get("split_config")
    if not isinstance(split_cfg, dict):
        pdbbind_split = pdbbind.get("split")
        split_cfg = pdbbind_split if isinstance(pdbbind_split, dict) else {}

    yaml_pdbbind_raw = yaml_payload.get("pdbbind")
    yaml_pdbbind: dict[str, Any] = yaml_pdbbind_raw if isinstance(yaml_pdbbind_raw, dict) else {}
    yaml_dudez_raw = yaml_payload.get("dudez")
    yaml_dudez: dict[str, Any] = yaml_dudez_raw if isinstance(yaml_dudez_raw, dict) else {}
    yaml_runtime_raw = yaml_payload.get("runtime")
    yaml_runtime: dict[str, Any] = yaml_runtime_raw if isinstance(yaml_runtime_raw, dict) else {}
    yaml_reporting_raw = yaml_payload.get("reporting")
    yaml_reporting: dict[str, Any] = yaml_reporting_raw if isinstance(yaml_reporting_raw, dict) else {}
    yaml_split = yaml_pdbbind.get("split")
    yaml_split_cfg = yaml_split if isinstance(yaml_split, dict) else {}
    dudez_scaling = dudez.get("dudez_scaling_config")
    if not isinstance(dudez_scaling, dict):
        dudez_scaling = {}

    aggregate_summary = json_payload.get("aggregate_summary")
    reporting_policy_raw = (
        aggregate_summary.get("reporting_policy") if isinstance(aggregate_summary, dict) else None
    )
    reporting_policy = reporting_policy_raw if isinstance(reporting_policy_raw, dict) else {}

    stage_names = tuple(
        str(stage.get("name"))
        for stage in (json_payload.get("stage_list") or [])
        if isinstance(stage, dict) and stage.get("name")
    )
    replica_names = tuple(
        str(name)
        for name in (json_payload.get("replica_names") or [])
        if name
    )

    protocol_name = str(yaml_payload.get("name") or json_payload.get("protocol") or "")
    notes: list[str] = []
    if markdown_path is not None:
        notes.append(f"Markdown report: {markdown_path}")

    return WorkbenchOCScoreProtocolSummary(
        source_path=source_path,
        source_kind=source_kind,
        protocol_name=protocol_name,
        feature_policy=feature_policy,
        n_replicas=_protocol_int(json_payload.get("n_replicas") or yaml_payload.get("replicas")) or 0,
        base_seed=_protocol_int(json_payload.get("base_seed") or yaml_payload.get("seed")),
        replica_jobs=_protocol_int(json_payload.get("replica_jobs") or yaml_runtime.get("replica_jobs")),
        resume_completed=(
            bool(json_payload.get("resume_completed"))
            if "resume_completed" in json_payload
            else bool(yaml_runtime.get("resume_completed"))
            if "resume_completed" in yaml_runtime
            else None
        ),
        replica_names=replica_names,
        pdbbind_split_strategy=str(split_cfg.get("strategy") or yaml_split_cfg.get("strategy") or ""),
        pdbbind_train_size=_protocol_float(split_cfg.get("train_size") or yaml_split_cfg.get("train_size")),
        pdbbind_validation_size=_protocol_float(split_cfg.get("validation_size") or yaml_split_cfg.get("validation_size")),
        pdbbind_test_size=_protocol_float(split_cfg.get("test_size") or yaml_split_cfg.get("test_size")),
        pdbbind_objective_metric=str(pdbbind.get("objective_metric") or yaml_pdbbind.get("objective_metric") or "RMSE"),
        pdbbind_trials=_protocol_int(pdbbind.get("n_trials") or yaml_pdbbind.get("trials")),
        pdbbind_epochs=_protocol_int(pdbbind.get("epochs") or yaml_pdbbind.get("epochs")),
        dudez_primary_metric=str(
            dudez.get("primary_metric")
            or json_payload.get("dudez_primary_metric")
            or yaml_dudez.get("primary_metric")
            or "BEDROC"
        ),
        dudez_bedroc_alpha=_protocol_float(dudez.get("bedroc_alpha") or yaml_dudez.get("bedroc_alpha")),
        dudez_scaling_strategy=str(dudez_scaling.get("strategy") or yaml_dudez.get("scaling_strategy") or ""),
        dudez_trials=_protocol_int(dudez.get("n_trials") or yaml_dudez.get("trials")),
        dudez_epochs=_protocol_int(dudez.get("epochs") or yaml_dudez.get("epochs")),
        calibration_report_mode=str(
            json_payload.get("calibration_report_mode")
            or yaml_reporting.get("calibration_report_mode")
            or dudez.get("calibration_report_mode")
            or ""
        ),
        primary_claim=str(
            json_payload.get("primary_claim")
            or reporting_policy.get("primary_claim")
            or ""
        ),
        stage_names=stage_names,
        ablation_variants=ablation_variants,
        markdown_path=markdown_path,
        notes=tuple(notes),
    )


def _read_replica_count_from_protocol(path: Path) -> int | None:
    '''Return a planned replica count from OCScore protocol artifacts.

    Parameters
    ----------
    path : pathlib.Path
        Study or layout root that may contain protocol JSON files.

    Returns
    -------
    int or None
        Planned replica count when available.
    '''

    if not path.is_dir():
        return None
    for candidate in (path / "replicas_protocol.json", path / "staged_optuna_protocol.json"):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        raw = payload.get("n_replicas")
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int) and raw >= 1:
            return raw
        if isinstance(raw, float) and raw.is_integer() and int(raw) >= 1:
            return int(raw)
    return None


def _infer_study_replica_count(study_path: Path, *, layout_root: Path) -> int:
    '''Infer the expected replica count for one OCScore study.

    Parameters
    ----------
    study_path : pathlib.Path
        Study directory.
    layout_root : pathlib.Path
        Resolved OCScore layout root.

    Returns
    -------
    int
        Expected replica slots for the study.
    '''

    replica_paths = _collect_replica_paths(study_path)
    from_dirs = max(replica_paths.keys(), default=0)
    from_protocol = 0
    for candidate in (study_path, layout_root):
        count = _read_replica_count_from_protocol(candidate)
        if count is not None:
            from_protocol = max(from_protocol, count)
    inferred = max(from_dirs, from_protocol)
    return inferred if inferred >= 1 else DEFAULT_OCSCORE_REPLICA_COUNT


def _cross_validation_dir(layout_root: Path, *, role: OCScoreWorkspaceRole, study_name: str) -> Path:
    '''Return the expected cross-validation directory for one study export.'''

    return _study_export_path(layout_root, role=role, study_name=study_name) / "cross_validation"


def _load_cross_validation(cv_dir: Path) -> WorkbenchOCScoreCrossValidation | None:
    '''Load exported cross-validation scorer summaries when present.'''

    mean_std_path = cv_dir / "cross_validation_scorer_mean_std.csv"
    if not mean_std_path.is_file() or mean_std_path.stat().st_size < 8:
        return None
    metrics: list[WorkbenchOCScoreCrossValidationMetric] = []
    fold_count = 0
    folds_path = cv_dir / "cross_validation_folds.csv"
    if folds_path.is_file():
        try:
            with folds_path.open("r", encoding="utf-8", newline="") as handle:
                fold_count = sum(1 for _ in csv.DictReader(handle))
        except OSError:
            fold_count = 0
    task = ""
    results_path = cv_dir / "cross_validation_results.json"
    if results_path.is_file():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                task = str(payload.get("task", "") or "")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            task = ""
    with mean_std_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        for row in reader:
            scorer = str(row.get("scorer", "")).strip()
            metric = str(row.get("metric", "")).strip()
            mean = _numeric(row.get("mean"))
            if not scorer or not metric or mean is None:
                continue
            std = _numeric(row.get("std")) or 0.0
            n_folds_raw = _numeric(row.get("n_folds"))
            metrics.append(
                WorkbenchOCScoreCrossValidationMetric(
                    scorer=scorer,
                    metric=metric,
                    mean=mean,
                    std=std,
                    n_folds=int(n_folds_raw) if n_folds_raw is not None else fold_count,
                )
            )
    if not metrics:
        return None
    return WorkbenchOCScoreCrossValidation(
        path=cv_dir,
        task=task,
        fold_count=fold_count,
        metrics=tuple(metrics),
    )


def _build_study(
    *,
    role: OCScoreWorkspaceRole,
    study_name: str,
    policy_name: str,
    path: Path,
    layout_root: Path,
    expected_replica_count: int | None,
    requested: set[str] | None,
    max_depth: int,
    max_metric_file_bytes: int,
    figures: tuple[WorkbenchOCScoreFigure, ...] = (),
) -> WorkbenchOCScoreStudy:
    '''Build a strict OCScore study record.

    Parameters
    ----------
    role : str
        Study role.
    study_name : str
        Study name.
    policy_name : str
        Policy name.
    path : pathlib.Path
        Study root path.
    layout_root : pathlib.Path
        Resolved OCScore layout root.
    expected_replica_count : int or None
        Expected number of replicas. When None, inferred from replica folders and
        protocol artifacts.
    requested : set[str] or None
        Optional requested metric names.
    max_depth : int
        Maximum recursive depth.
    max_metric_file_bytes : int
        Maximum metric file size to parse.
    figures : tuple[WorkbenchOCScoreFigure, ...]
        Study-level figures discovered outside replica directories.

    Returns
    -------
    WorkbenchOCScoreStudy
        Study record.
    '''

    replica_count = (
        expected_replica_count
        if expected_replica_count is not None
        else _infer_study_replica_count(path, layout_root=layout_root)
    )
    replica_paths = _collect_replica_paths(path)
    max_slot = max((replica_count, *replica_paths.keys()), default=replica_count)
    replicas = tuple(
        _build_replica(
            role=role,
            study_name=study_name,
            policy_name=policy_name,
            replica_index=index,
            replica_path=replica_paths.get(index),
            requested=requested,
            max_depth=max_depth,
            max_metric_file_bytes=max_metric_file_bytes,
        )
        for index in range(1, max_slot + 1)
    )
    return WorkbenchOCScoreStudy(
        role=role,
        study_name=study_name,
        policy_name=policy_name,
        path=path,
        expected_replica_count=replica_count,
        detected_replica_count=sum(1 for replica in replicas if replica.exists),
        completed_count=sum(1 for replica in replicas if replica.status == "completed"),
        failed_count=sum(1 for replica in replicas if replica.status == "failed"),
        missing_count=sum(1 for replica in replicas if replica.status == "missing"),
        replicas=replicas,
        figures=figures,
        metric_summary=_metric_summary(replicas),
        cross_validation=_load_cross_validation(
            _cross_validation_dir(layout_root, role=role, study_name=study_name),
        ),
        protocol=_load_protocol_summary(path, feature_policy=policy_name),
    )


def _ablation_containers(root: Path) -> tuple[Path, ...]:
    '''Return supported ablation container directories.

    Parameters
    ----------
    root : pathlib.Path
        OCScore output root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing ablation containers.
    '''

    return tuple(path for path in (root / "ablation", root / "ablations") if path.is_dir())


def _has_layout_markers(path: Path) -> bool:
    '''Return whether a path contains strict OCScore layout markers.

    Parameters
    ----------
    path : pathlib.Path
        Candidate layout root.

    Returns
    -------
    bool
        True when the path has direct replicas or an ablation container.
    '''

    return bool(_collect_replica_paths(path)) or bool(_ablation_containers(path))


def _has_workbench_manifest_layout(path: Path) -> bool:
    '''Return whether a path looks like the removed generic Workbench layout.

    Parameters
    ----------
    path : pathlib.Path
        Candidate layout root.

    Returns
    -------
    bool
        True when child directories contain Workbench manifests.
    '''

    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir() or _is_hidden(child):
            continue
        if (child / "run_manifest.yml").is_file() or (child / "result_manifest.yml").is_file():
            return True
    return False


def _resolve_layout_root(root: Path) -> Path:
    '''Resolve the directory that actually contains OCScore replicas.

    Parameters
    ----------
    root : pathlib.Path
        User-provided output root.

    Returns
    -------
    pathlib.Path
        Layout root used by strict discovery.
    '''

    if _has_layout_markers(root):
        return root
    train_root = root / "train"
    if train_root.is_dir() and _has_layout_markers(train_root):
        return train_root
    return root


def _requested_metrics(metric_names: tuple[str, ...]) -> set[str] | None:
    '''Normalize user-requested metric names.

    Parameters
    ----------
    metric_names : tuple[str, ...]
        Requested metric names.

    Returns
    -------
    set[str] or None
        Normalized names, or None.
    '''

    if not metric_names:
        return None
    return {_normalise_metric_name(name) for name in metric_names if str(name).strip()}


## Public ##


def _baseline_source_entries(layout_root: Path, *, max_depth: int) -> tuple[WorkbenchOCScoreBaselineSource, ...]:
    '''Return discovered external baseline CSV sources with modification times.'''

    sources: list[WorkbenchOCScoreBaselineSource] = []
    for path in _external_baseline_search_paths(layout_root, max_depth=max_depth):
        modified_at: datetime | None = None
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            modified_at = None
        sources.append(WorkbenchOCScoreBaselineSource(path=path, modified_at=modified_at))
    return tuple(sources)


def _build_run_context(
    baseline_study: WorkbenchOCScoreStudy,
    protocol: WorkbenchOCScoreProtocolSummary | None,
    baseline_sources: tuple[WorkbenchOCScoreBaselineSource, ...],
) -> WorkbenchOCScoreRunContext:
    '''Build always-visible dashboard run context from study and protocol metadata.'''

    split_parts: list[str] = []
    if protocol is not None:
        if protocol.pdbbind_train_size is not None:
            split_parts.append(f"train {protocol.pdbbind_train_size:.0%}")
        if protocol.pdbbind_validation_size is not None:
            split_parts.append(f"val {protocol.pdbbind_validation_size:.0%}")
        if protocol.pdbbind_test_size is not None:
            split_parts.append(f"test {protocol.pdbbind_test_size:.0%}")
    split_summary = protocol.pdbbind_split_strategy if protocol is not None else ""
    if split_parts:
        split_summary = " · ".join(part for part in (split_summary, " / ".join(split_parts)) if part)

    return WorkbenchOCScoreRunContext(
        planned_replica_count=baseline_study.expected_replica_count,
        detected_replica_count=baseline_study.detected_replica_count,
        completed_replica_count=baseline_study.completed_count,
        pdbbind_split_strategy=protocol.pdbbind_split_strategy if protocol is not None else "",
        pdbbind_split_summary=split_summary,
        dudez_bedroc_alpha=protocol.dudez_bedroc_alpha if protocol is not None else None,
        baseline_sources=baseline_sources,
    )


def resolve_optuna_dashboard_slot_count(
    root: str | Path,
    *,
    override: int | None = None,
) -> int:
    '''Return the Optuna dashboard slot count for one served OCScore root.

    When ``override`` is omitted, the count follows the baseline replica layout:
    the greater of detected ``replica_*`` directories and ``n_replicas`` from
    protocol artifacts, clamped to ``MIN_OPTUNA_DASHBOARD_SLOT_COUNT`` through
    ``MAX_OPTUNA_DASHBOARD_SLOT_COUNT``.

    Parameters
    ----------
    root : str or pathlib.Path
        Served OCScore root.
    override : int or None
        Optional explicit slot count from CLI or API configuration.

    Returns
    -------
    int
        Resolved Optuna dashboard slot count.
    '''

    if override is not None:
        if override < MIN_OPTUNA_DASHBOARD_SLOT_COUNT:
            raise ValueError("optuna dashboard slot override must be >= 1")
        return min(override, MAX_OPTUNA_DASHBOARD_SLOT_COUNT)

    root_path = Path(root)
    layout_root = _resolve_layout_root(root_path)
    if not layout_root.is_dir():
        return MIN_OPTUNA_DASHBOARD_SLOT_COUNT

    replica_paths = _collect_replica_paths(layout_root)
    from_dirs = max(replica_paths.keys(), default=0)
    from_protocol = 0
    for candidate in (layout_root, root_path):
        count = _read_replica_count_from_protocol(candidate)
        if count is not None:
            from_protocol = max(from_protocol, count)
    count = max(from_dirs, from_protocol, MIN_OPTUNA_DASHBOARD_SLOT_COUNT)
    return min(count, MAX_OPTUNA_DASHBOARD_SLOT_COUNT)


def resolve_ocscore_layout_root(root: str | Path) -> Path:
    '''Resolve the directory that actually contains OCScore replicas.

    Parameters
    ----------
    root : str or pathlib.Path
        User-provided output root.

    Returns
    -------
    pathlib.Path
        Layout root used by strict discovery.
    '''

    return _resolve_layout_root(Path(root).expanduser().resolve())


def ablation_container_paths(root: str | Path) -> tuple[Path, ...]:
    '''Return supported ablation container directories under one OCScore layout root.

    Parameters
    ----------
    root : str or pathlib.Path
        User-provided OCScore output root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing ``ablation/`` and ``ablations/`` directories when present.
    '''

    layout_root = resolve_ocscore_layout_root(root)
    return _ablation_containers(layout_root)


def build_ocscore_workspace(
    root: str | Path,
    *,
    expected_replica_count: int | None = None,
    max_depth: int = DEFAULT_OCSCORE_SCAN_DEPTH,
    max_metric_file_bytes: int = DEFAULT_OCSCORE_MAX_METRIC_FILE_BYTES,
    metric_names: tuple[str, ...] = (),
) -> WorkbenchOCScoreWorkspace:
    '''Build a strict OCScore workspace summary from the canonical layout.

    Parameters
    ----------
    root : str or pathlib.Path
        OCScore output root. Baseline replicas are expected directly below this
        path, with ablation studies below ``ablation/`` or ``ablations/``.
    expected_replica_count : int or None
        Expected number of replicas per study. When None, each study infers its
        own count from replica folders and protocol artifacts.
    max_depth : int
        Maximum recursive depth inside each replica.
    max_metric_file_bytes : int
        Maximum metric or log file size to inspect.
    metric_names : tuple[str, ...]
        Optional extra metric names to keep in addition to the curated defaults.

    Returns
    -------
    WorkbenchOCScoreWorkspace
        Strict OCScore workspace payload for the dashboard and API.
    '''

    root_path = Path(root)
    layout_root = _resolve_layout_root(root_path)
    if expected_replica_count is not None and expected_replica_count < 1:
        raise ValueError("expected_replica_count must be greater than or equal to one.")
    if max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to zero.")
    if max_metric_file_bytes < 1:
        raise ValueError("max_metric_file_bytes must be greater than or equal to one.")

    requested = _requested_metrics(metric_names)
    issues: list[InventoryIssue] = []
    if not root_path.exists():
        issues.append(_issue(root_path, "OCScore root does not exist."))
    elif not root_path.is_dir():
        issues.append(_issue(root_path, "OCScore root is not a directory."))
    elif not _has_layout_markers(layout_root):
        if _has_workbench_manifest_layout(layout_root):
            issues.append(
                _issue(
                    layout_root,
                    "Unsupported generic Workbench manifest layout detected. Serve the OCScore output root "
                    "that contains baseline replica_* directories and ablation/ or ablations/ study folders.",
                )
            )
        else:
            issues.append(
                _issue(
                    layout_root,
                    "No strict OCScore replicas were detected. Expected replica_* directories directly under "
                    "the served root, or under its train/ child, with ablation/ or ablations/ study folders.",
                )
            )

    baseline_study = _build_study(
        role="baseline",
        study_name="baseline",
        policy_name="baseline",
        path=layout_root,
        layout_root=layout_root,
        expected_replica_count=expected_replica_count,
        requested=requested,
        max_depth=max_depth,
        max_metric_file_bytes=max_metric_file_bytes,
        figures=_build_baseline_export_figures(
            layout_root,
            policy_name="baseline",
            replica_name="study",
            max_depth=max_depth,
        ),
    )
    ablation_studies: list[WorkbenchOCScoreStudy] = []
    for container in _ablation_containers(layout_root):
        for study_path in sorted((path for path in container.iterdir() if path.is_dir()), key=lambda item: item.name):
            ablation_studies.append(
                _build_study(
                    role="ablation",
                    study_name=study_path.name,
                    policy_name=study_path.name,
                    path=study_path,
                    layout_root=layout_root,
                    expected_replica_count=expected_replica_count,
                    requested=requested,
                    max_depth=max_depth,
                    max_metric_file_bytes=max_metric_file_bytes,
                    figures=_build_figures(
                        _study_export_path(layout_root, role="ablation", study_name=study_path.name),
                        policy_name=study_path.name,
                        replica_name="study",
                        max_depth=max_depth,
                    ),
                )
            )
    studies = (baseline_study, *ablation_studies)
    external_baselines = _load_external_baselines(layout_root)
    campaign_name, ablation_variants = _load_ablation_campaign(layout_root)
    workspace_protocol = _load_protocol_summary(
        layout_root,
        feature_policy=baseline_study.policy_name,
        ablation_variants=ablation_variants,
    )
    if workspace_protocol is not None and campaign_name and not workspace_protocol.protocol_name:
        workspace_protocol = workspace_protocol.model_copy(update={"protocol_name": campaign_name})
    baseline_sources = _baseline_source_entries(layout_root, max_depth=max_depth)
    run_context = _build_run_context(baseline_study, workspace_protocol, baseline_sources)
    metric_names_seen = tuple(sorted({
        name for study in studies for name in study.metric_summary
    } | {
        name for baseline in external_baselines for name in baseline.metric_summary
    }))
    workspace_replica_count = (
        expected_replica_count
        if expected_replica_count is not None
        else baseline_study.expected_replica_count
    )
    return WorkbenchOCScoreWorkspace(
        root=layout_root,
        expected_replica_count=workspace_replica_count,
        max_depth=max_depth,
        baseline_study=baseline_study,
        ablation_studies=tuple(ablation_studies),
        external_baselines=external_baselines,
        study_count=len(studies),
        replica_count=sum(len(study.replicas) for study in studies),
        completed_count=sum(study.completed_count for study in studies),
        failed_count=sum(study.failed_count for study in studies),
        missing_count=sum(study.missing_count for study in studies),
        metric_names=metric_names_seen,
        issue_count=len(issues),
        issues=tuple(issues),
        protocol=workspace_protocol,
        run_context=run_context,
    )


__all__ = [
    "CURATED_METRIC_DIRECTIONS",
    "DEFAULT_OCSCORE_MAX_METRIC_FILE_BYTES",
    "DEFAULT_OCSCORE_REPLICA_COUNT",
    "DEFAULT_OCSCORE_SCAN_DEPTH",
    "MAX_OPTUNA_DASHBOARD_SLOT_COUNT",
    "MIN_OPTUNA_DASHBOARD_SLOT_COUNT",
    "ablation_container_paths",
    "build_ocscore_workspace",
    "resolve_ocscore_layout_root",
    "resolve_optuna_dashboard_slot_count",
]
