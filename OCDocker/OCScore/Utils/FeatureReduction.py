#!/usr/bin/env python3

# Description
###############################################################################
'''
Granular feature-reduction utilities for OCScore descriptor datasets.

Usage:

import OCDocker.OCScore.Utils.FeatureReduction as ocfeatures
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import math
import platform
import sys

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Reproducibility as ocrepro
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FeatureSelectionScope
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import attach_feature_hashes
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import write_feature_selection_json

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

LOGGER = oclogging.get_logger("ocscore.feature_reduction")

DEFAULT_METADATA_COLUMNS = ["receptor", "ligand", "name", "type", "db"]
OCSCORE_PIPELINE_METADATA_COLUMNS = [
    "database",
    "target",
    "dataset",
    "kind",
    "label",
    "PDB ID",
    "pdb_id",
    "complex_id",
    "Complex ID",
    "Protein",
    "resolution",
    "release_year",
    "-logKd/Ki",
    "Ki/Kd",
    "Ki/Kd_relation",
    "Ki/Kd_value",
    "Ki/Kd_order",
    "Ki/Kd_raw_value",
    "Ki/Kd_raw_unit",
    "dG",
    "dG_kcal_mol",
    "reference",
    "ligand_name",
    "index_comment",
]
DEFAULT_TARGET_COLUMNS = ["experimental"]
DEFAULT_RECEPTOR_PATTERNS = ["receptor_"]
DEFAULT_LIGAND_PATTERNS = ["ligand_"]
DEFAULT_SCORING_PATTERNS = ["vina_", "gnina_", "smina_", "plants_", "oddt_"]
DEFAULT_ID_COLUMNS = ["PDB ID", "pdb_id", "complex_id", "Complex ID", "name", "receptor", "ligand"]
BLOCK_NAMES = ("receptor", "ligand", "scoring")
MISSING_DROPPED_ROWS_COLUMNS = ["original_index", "n_missing_total", "missing_columns", "drop_reason"]
MISSINGNESS_BY_COLUMN_COLUMNS = ["column", "n_missing", "fraction_missing"]
MISSINGNESS_BY_BLOCK_COLUMNS = ["block", "n_columns", "n_missing_values", "n_rows_with_missing", "missing_columns"]
CONSTANT_FEATURE_COLUMNS = ["block", "feature", "reason", "n_unique", "representative_value"]
NEAR_CONSTANT_FEATURE_COLUMNS = ["block", "feature", "reason", "threshold", "top_value", "top_count", "top_fraction"]
DUPLICATE_FEATURE_COLUMNS = ["block", "kept_feature", "dropped_feature", "reason"]
INTRA_CORRELATION_COLUMNS = [
    "pair_order", "block", "feature_1", "feature_2", "correlation",
    "abs_correlation", "method", "threshold", "flagged",
]
CROSS_CORRELATION_COLUMNS = [
    "left_block", "right_block", "left_feature", "right_feature",
    "correlation", "abs_correlation", "method", "threshold", "flagged",
]
CORRELATION_FILTER_COLUMNS = [
    "block", "kept_feature", "dropped_feature", "correlation",
    "abs_correlation", "threshold", "retention_policy", "reason",
]
CROSS_PREDICTABILITY_COLUMNS = [
    "target_feature", "target_block", "predictor_block", "model",
    "mean_cv_r2", "std_cv_r2", "cv_folds", "requested_cv_folds",
    "n_predictors", "random_seed", "n_jobs", "interpretation",
]
CROSS_BLOCK_FILTER_COLUMNS = [
    "molecular_feature", "scoring_feature", "correlation", "abs_correlation",
    "threshold", "scoring_function_priority", "reason",
]


# Classes
###############################################################################

@dataclass
class DescriptorBlocks:
    """Descriptor columns split into conceptual blocks.

    Parameters
    ----------
    receptor : list[str]
        Receptor molecular descriptor columns.
    ligand : list[str]
        Ligand molecular descriptor columns.
    scoring : list[str]
        Scoring-function descriptor columns.
    metadata : list[str]
        Metadata columns preserved in reduced outputs.
    target : list[str]
        Target columns preserved in reduced outputs.
    unmatched : list[str]
        Non-metadata, non-target columns that were not assigned to a block.
    duplicate_assignments : dict[str, list[str]]
        Columns that matched more than one descriptor block.
    sources : dict[str, str]
        Detection source used for each descriptor block.
    """

    receptor: List[str] = field(default_factory=list)
    ligand: List[str] = field(default_factory=list)
    scoring: List[str] = field(default_factory=list)
    metadata: List[str] = field(default_factory=list)
    target: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    duplicate_assignments: Dict[str, List[str]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def all_descriptor_columns(self) -> List[str]:
        '''Return all descriptor columns in receptor, ligand, scoring order.

        Returns
        -------
        list[str]
            Descriptor columns from receptor, ligand, and scoring blocks.
        '''

        return compose_selected_features(
            receptor_columns=self.receptor,
            ligand_columns=self.ligand,
            scoring_columns=self.scoring,
        )

    @property
    def all_model_columns(self) -> List[str]:
        '''Return target plus all descriptor columns, preserving order.

        Returns
        -------
        list[str]
            Target columns followed by descriptor columns.
        '''

        return _unique_preserve_order([*self.target, *self.all_descriptor_columns])

    def items(self) -> Iterator[Tuple[str, List[str]]]:
        '''Iterate over descriptor block names and columns.

        Returns
        -------
        Iterator[tuple[str, list[str]]]
            Iterator over ``(block_name, columns)`` pairs for receptor, ligand,
            and scoring blocks.
        '''

        yield "receptor", self.receptor
        yield "ligand", self.ligand
        yield "scoring", self.scoring


@dataclass
class BlockDetectionConfig:
    """Configuration for descriptor block detection.

    Parameters
    ----------
    metadata_columns : list[str]
        Candidate metadata columns to preserve.
    target_columns : list[str]
        Candidate target columns to preserve and optionally include in missing-row checks.
    receptor_patterns : list[str]
        Prefixes used as a receptor descriptor fallback.
    ligand_patterns : list[str]
        Prefixes used as a ligand descriptor fallback.
    scoring_patterns : list[str]
        Prefixes used as a scoring-function descriptor fallback.
    use_ligand_class_descriptors : bool
        If True, use ``Ligand.allDescriptors`` when available.
    use_receptor_class_descriptors : bool
        If True, use ``Receptor.allDescriptors`` when available.
    use_scoring_model_descriptors : bool
        If True, use ``Complexes.allDescriptors`` when available.
    """

    metadata_columns: List[str] = field(default_factory=lambda: DEFAULT_METADATA_COLUMNS.copy())
    target_columns: List[str] = field(default_factory=lambda: DEFAULT_TARGET_COLUMNS.copy())
    receptor_patterns: List[str] = field(default_factory=lambda: DEFAULT_RECEPTOR_PATTERNS.copy())
    ligand_patterns: List[str] = field(default_factory=lambda: DEFAULT_LIGAND_PATTERNS.copy())
    scoring_patterns: List[str] = field(default_factory=lambda: DEFAULT_SCORING_PATTERNS.copy())
    use_ligand_class_descriptors: bool = True
    use_receptor_class_descriptors: bool = True
    use_scoring_model_descriptors: bool = True


@dataclass
class MissingRowsConfig:
    """Configuration for missing-row filtering.

    Parameters
    ----------
    enabled : bool
        If True, remove rows with missing values before feature reduction.
    subset : str
        Column subset checked for missing values.
    preserve_index : bool
        If True, keep the original DataFrame index in the cleaned DataFrame.
    """

    enabled: bool = True
    subset: str = "model_relevant_columns"
    preserve_index: bool = True


@dataclass
class ColumnQualityConfig:
    """Configuration for block-wise column-quality filters.

    Parameters
    ----------
    remove_constant : bool
        If True, remove columns with a single unique value.
    remove_near_constant : bool
        If True, remove columns dominated by one value.
    near_constant_threshold : float
        Fraction above which a dominant value marks a feature as near-constant.
    remove_duplicates : bool
        If True, remove exact duplicate numeric columns.
    """

    remove_constant: bool = True
    remove_near_constant: bool = True
    near_constant_threshold: float = 0.995
    remove_duplicates: bool = True


@dataclass
class IntraBlockCorrelationConfig:
    """Configuration for intra-block correlation filtering.

    Parameters
    ----------
    method : str
        Correlation method passed to ``pandas.DataFrame.corr``.
    receptor_threshold : float
        Absolute correlation threshold for receptor descriptors.
    ligand_threshold : float
        Absolute correlation threshold for ligand descriptors.
    scoring_threshold : float
        Absolute correlation threshold for scoring-function descriptors.
    retention_policy : str
        Deterministic policy used to keep one feature from a correlated pair.
    """

    method: str = "spearman"
    receptor_threshold: float = 0.98
    ligand_threshold: float = 0.98
    scoring_threshold: float = 0.99
    retention_policy: str = "first"

    def threshold_for_block(self, block_name: str) -> float:
        '''Return the configured threshold for a descriptor block.

        Parameters
        ----------
        block_name : str
            Descriptor block name. Must be ``"receptor"``, ``"ligand"``, or
            ``"scoring"``.

        Returns
        -------
        float
            Correlation threshold configured for the descriptor block.

        Raises
        ------
        ValueError
            If ``block_name`` is not a known descriptor block.
        '''

        if block_name == "receptor":
            return self.receptor_threshold
        if block_name == "ligand":
            return self.ligand_threshold
        if block_name == "scoring":
            return self.scoring_threshold
        raise ValueError(f"Unknown descriptor block: {block_name}")


@dataclass
class CrossBlockDiagnosticsConfig:
    """Configuration for cross-block diagnostics.

    Parameters
    ----------
    enabled : bool
        If True, compute cross-block diagnostics after intra-block filtering.
    correlation_threshold : float
        Absolute pairwise correlation threshold used to flag cross-block pairs.
    ridge_cv_folds : int
        Requested number of folds for Ridge CV predictability diagnostics.
    random_seed : int
        Seed used for deterministic cross-validation splits.
    n_jobs : int
        Number of parallel jobs for Ridge CV predictability diagnostics. Use 1
        for serial execution or -1 for all available cores.
    """

    enabled: bool = True
    correlation_threshold: float = 0.95
    ridge_cv_folds: int = 5
    random_seed: int = 42
    n_jobs: int = 1


@dataclass
class CrossBlockFilteringConfig:
    """Configuration for optional conservative cross-block filtering.

    Parameters
    ----------
    enabled : bool
        If True, run conservative cross-block filtering.
    scoring_function_priority : bool
        If True, scoring-function descriptors may be preferred over strongly
        correlated molecular descriptors.
    """

    enabled: bool = False
    scoring_function_priority: bool = False


@dataclass
class FeatureReductionConfig:
    """Configuration for the convenience feature-reduction orchestration helper.

    Parameters
    ----------
    block_detection : BlockDetectionConfig
        Descriptor block detection settings.
    missing_rows : MissingRowsConfig
        Missing-row filtering settings.
    column_quality : ColumnQualityConfig
        Constant, near-constant, and duplicate-column filtering settings.
    intra_block_correlation : IntraBlockCorrelationConfig
        Intra-block correlation filtering settings.
    cross_block_diagnostics : CrossBlockDiagnosticsConfig
        Cross-block diagnostic settings.
    cross_block_filtering : CrossBlockFilteringConfig
        Optional cross-block filtering settings.
    verbose : bool
        If True, emit step-level orchestration progress messages.
    """

    block_detection: BlockDetectionConfig = field(default_factory=BlockDetectionConfig)
    missing_rows: MissingRowsConfig = field(default_factory=MissingRowsConfig)
    column_quality: ColumnQualityConfig = field(default_factory=ColumnQualityConfig)
    intra_block_correlation: IntraBlockCorrelationConfig = field(default_factory=IntraBlockCorrelationConfig)
    cross_block_diagnostics: CrossBlockDiagnosticsConfig = field(default_factory=CrossBlockDiagnosticsConfig)
    cross_block_filtering: CrossBlockFilteringConfig = field(default_factory=CrossBlockFilteringConfig)
    verbose: bool = False


@dataclass
class MissingRowsResult:
    """Result from missing-row filtering.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        DataFrame after rows with missing values were removed.
    dropped_rows : pd.DataFrame
        Per-row report with original index, IDs, missing columns, and reason.
    missingness_by_column : pd.DataFrame
        Missing-value counts and fractions by checked column.
    missingness_by_block : pd.DataFrame
        Missing-value counts by descriptor block and target block.
    summary : dict[str, Any]
        Reproducibility summary of the row-filtering operation.
    """

    cleaned_df: pd.DataFrame
    dropped_rows: pd.DataFrame
    missingness_by_column: pd.DataFrame
    missingness_by_block: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class CorrelationReport:
    """Correlation matrix and long-form pairwise report.

    Parameters
    ----------
    matrix : pd.DataFrame
        Square correlation matrix.
    pairs : pd.DataFrame
        Long-form upper-triangle correlation report.
    method : str
        Correlation method used.
    block : str
        Descriptor block name.
    threshold : float, optional
        Threshold used to flag correlated pairs.
    """

    matrix: pd.DataFrame
    pairs: pd.DataFrame
    method: str
    block: str = ""
    threshold: Optional[float] = None


@dataclass
class CorrelationFilterResult:
    """Result from deterministic correlated-feature filtering.

    Parameters
    ----------
    kept_features : list[str]
        Feature columns retained after filtering.
    dropped_features : list[str]
        Feature columns removed by the filter.
    report : pd.DataFrame
        Per-feature drop report.
    """

    kept_features: List[str]
    dropped_features: List[str]
    report: pd.DataFrame


@dataclass
class CrossBlockFilterResult:
    """Result from optional conservative cross-block filtering.

    Parameters
    ----------
    kept_features : list[str]
        Molecular descriptor columns retained after filtering.
    dropped_features : list[str]
        Molecular descriptor columns removed by the filter.
    report : pd.DataFrame
        Cross-block filtering report.
    """

    kept_features: List[str]
    dropped_features: List[str]
    report: pd.DataFrame


@dataclass
class FeatureReductionResult:
    """Result from the convenience orchestration helper.

    Parameters
    ----------
    reduced_df : pd.DataFrame
        Reduced dataset containing metadata, targets, and selected features.
    selected_features : list[str]
        Final selected descriptor columns.
    blocks : DescriptorBlocks
        Descriptor block assignments detected from the input dataset.
    cleaned_blocks : dict[str, list[str]]
        Final retained columns by descriptor block.
    missing_result : MissingRowsResult
        Missing-row filtering result.
    protocol : dict[str, Any]
        JSON-serializable reproducibility protocol.
    reports : dict[str, pd.DataFrame]
        Report tables generated by the workflow.
    output_paths : dict[str, str]
        Paths written by ``write_feature_reduction_outputs``.
    """

    reduced_df: pd.DataFrame
    selected_features: List[str]
    blocks: DescriptorBlocks
    cleaned_blocks: Dict[str, List[str]]
    missing_result: MissingRowsResult
    protocol: Dict[str, Any]
    reports: Dict[str, pd.DataFrame]
    output_paths: Dict[str, str] = field(default_factory=dict)


# Functions
###############################################################################
## Private ##

def _as_list(values: Optional[Union[str, Sequence[str], pd.Index]]) -> List[str]:
    '''Normalize a string sequence to a list.'''

    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in list(values)]


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    '''Return unique values while preserving first occurrence order.'''

    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _columns_present(columns: Iterable[str], requested: Optional[Sequence[str]]) -> List[str]:
    '''Return requested columns that are present in columns.'''

    column_set = set(columns)
    return [col for col in _as_list(requested) if col in column_set]


def _validate_columns_exist(df: pd.DataFrame, columns: Sequence[str], label: str = "columns") -> None:
    '''Raise ValueError when requested columns are absent from a DataFrame.'''

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {label}: {missing}")


def _validate_probability_threshold(threshold: float, label: str = "threshold") -> None:
    '''Validate a correlation or frequency threshold in the interval (0, 1].'''

    if threshold <= 0 or threshold > 1:
        raise ValueError(f"{label} must be > 0 and <= 1.")


def _match_by_patterns(columns: Sequence[str], patterns: Sequence[str]) -> List[str]:
    '''Return columns whose names start with any pattern, case-insensitively.'''

    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    if not lowered_patterns:
        return []
    return [col for col in columns if col.lower().startswith(lowered_patterns)]


def _match_by_descriptor_names(columns: Sequence[str], descriptor_names: Sequence[str]) -> List[str]:
    '''Return columns matching descriptor names case-insensitively.'''

    descriptor_lookup = {name.lower() for name in descriptor_names}
    return [col for col in columns if col.lower() in descriptor_lookup]


def _get_scoring_descriptor_names_from_complexes_model() -> List[str]:
    '''Return scoring descriptor names from the Complexes model when importable.'''

    try:
        from OCDocker.DB.Models.Complexes import Complexes
    except (ImportError, ModuleNotFoundError):
        return []

    descriptors = getattr(Complexes, "allDescriptors", [])
    return [str(desc) for desc in descriptors]


def _get_ligand_descriptor_names_from_ligand_class() -> List[str]:
    '''Return ligand descriptor names when the Ligand class is importable.'''

    try:
        import OCDocker.Ligand as ocl
    except (ImportError, ModuleNotFoundError):
        return []

    return [str(desc) for desc in getattr(ocl.Ligand, "allDescriptors", [])]


def _get_receptor_descriptor_names_from_receptor_class() -> List[str]:
    '''Return receptor descriptor names when the Receptor class is importable.'''

    try:
        import OCDocker.Receptor as ocr
    except (ImportError, ModuleNotFoundError):
        return []

    return [str(desc) for desc in getattr(ocr.Receptor, "allDescriptors", [])]


def _long_form_upper_triangle(
    matrix: pd.DataFrame,
    *,
    method: str,
    block: str = "",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    '''Convert a square correlation matrix to a stable upper-triangle report.'''

    rows: List[Dict[str, Any]] = []
    columns = list(matrix.columns)
    pair_order = 0
    for i, feature_1 in enumerate(columns):
        for j in range(i + 1, len(columns)):
            feature_2 = columns[j]
            value = matrix.iloc[i, j]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            rows.append({
                "pair_order": pair_order,
                "block": block,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": float(value),
                "abs_correlation": abs_value,
                "method": method,
                "threshold": float(threshold) if threshold is not None else None,
                "flagged": bool(abs_value > threshold) if threshold is not None else False,
            })
            pair_order += 1

    return pd.DataFrame(rows, columns=INTRA_CORRELATION_COLUMNS)


def _dropped_features_from_any(dropped_features: Union[pd.DataFrame, Sequence[str], pd.Series]) -> List[str]:
    '''Extract dropped feature names from a report DataFrame or sequence.'''

    if isinstance(dropped_features, pd.DataFrame):
        for col in ("dropped_feature", "feature", "molecular_feature"):
            if col in dropped_features.columns:
                return [str(value) for value in dropped_features[col].dropna().tolist()]
        return []
    if isinstance(dropped_features, pd.Series):
        return [str(value) for value in dropped_features.dropna().tolist()]
    return [str(value) for value in dropped_features]


def _block_columns_from_blocks(blocks: Optional[DescriptorBlocks]) -> Dict[str, List[str]]:
    '''Return block column mapping, allowing an empty default.'''

    if blocks is None:
        return {block: [] for block in BLOCK_NAMES}
    return {
        "receptor": list(blocks.receptor),
        "ligand": list(blocks.ligand),
        "scoring": list(blocks.scoring),
    }


def _missing_rows_columns(
    *,
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    descriptor_columns: Optional[Sequence[str]],
    target_columns: Optional[Sequence[str]],
    blocks: Optional[DescriptorBlocks],
    subset: str,
) -> List[str]:
    '''Resolve which columns should be checked for missing values.'''

    if columns is not None:
        selected = _as_list(columns)
    else:
        if blocks is not None:
            descriptors = blocks.all_descriptor_columns
            targets = blocks.target
        else:
            descriptors = _as_list(descriptor_columns)
            targets = _as_list(target_columns)

        normalized_subset = subset.lower().strip()
        if normalized_subset in {"descriptor_columns", "descriptors", "descriptor"}:
            selected = descriptors
        elif normalized_subset in {"descriptor_and_target_columns", "descriptors_and_targets", "descriptor_plus_target"}:
            selected = [*descriptors, *targets]
        elif normalized_subset in {"model_relevant_columns", "model", "all_model_relevant"}:
            selected = [*targets, *descriptors] if targets else descriptors
        elif normalized_subset in {"all", "all_columns"}:
            selected = list(df.columns)
        else:
            raise ValueError(
                "subset must be one of 'model_relevant_columns', 'descriptor_columns', "
                "'descriptor_and_target_columns', or 'all_columns'."
            )

    selected = _unique_preserve_order(selected)
    _validate_columns_exist(df, selected, label="missing-value columns")
    return selected


def _summarize_missingness_by_block(
    missing_mask: pd.DataFrame,
    blocks: Optional[DescriptorBlocks],
    target_columns: Sequence[str],
) -> pd.DataFrame:
    '''Summarize missing values by descriptor block and target columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    blocks : DescriptorBlocks
        Descriptor block assignments.
    target_columns : Sequence[str], optional
        Target columns to include. If None, ``blocks.target`` is used.

    Returns
    -------
    pd.DataFrame
        Table with missing-value counts by receptor, ligand, scoring, and target blocks.

    Raises
    ------
    ValueError
        If any required column is missing.
    '''

    block_map = _block_columns_from_blocks(blocks)
    if target_columns:
        block_map["target"] = list(target_columns)

    rows: List[Dict[str, Any]] = []
    for block, cols in block_map.items():
        present_cols = [col for col in cols if col in missing_mask.columns]
        if not present_cols:
            rows.append({
                "block": block,
                "n_columns": 0,
                "n_missing_values": 0,
                "n_rows_with_missing": 0,
                "missing_columns": "",
            })
            continue
        block_mask = missing_mask[present_cols]
        missing_by_column = block_mask.sum(axis=0)
        missing_columns = [col for col, value in missing_by_column.items() if int(value) > 0]
        rows.append({
            "block": block,
            "n_columns": len(present_cols),
            "n_missing_values": int(block_mask.to_numpy().sum()),
            "n_rows_with_missing": int(block_mask.any(axis=1).sum()),
            "missing_columns": ",".join(missing_columns),
        })

    return pd.DataFrame(rows, columns=MISSINGNESS_BY_BLOCK_COLUMNS)


def _interpret_r2(mean_r2: float) -> str:
    '''Return a redundancy label for a mean R2 value.'''

    if not np.isfinite(mean_r2):
        return "not estimated"
    if mean_r2 < 0.3:
        return "low redundancy"
    if mean_r2 < 0.7:
        return "partial redundancy"
    if mean_r2 < 0.9:
        return "strong redundancy"
    return "very strong redundancy"


def _report_to_frame(report: Union[pd.DataFrame, CorrelationReport, CorrelationFilterResult, CrossBlockFilterResult]) -> pd.DataFrame:
    '''Return a DataFrame payload from supported report objects.'''

    if isinstance(report, pd.DataFrame):
        return report.copy()
    if isinstance(report, CorrelationReport):
        return report.pairs.copy()
    if isinstance(report, (CorrelationFilterResult, CrossBlockFilterResult)):
        return report.report.copy()
    raise TypeError(f"Unsupported report type: {type(report)}")


def _to_jsonable(value: Any) -> Any:
    '''Convert common scientific Python objects to JSON-serializable values.'''

    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_markdown_protocol(protocol: Mapping[str, Any], output_path: Path) -> None:
    '''Write a compact human-readable protocol summary.'''

    lines = [
        "# Feature Reduction Protocol",
        "",
        f"- Generated at: {protocol.get('generated_at_utc', '')}",
        f"- Input shape: {protocol.get('input', {}).get('shape', '')}",
        f"- Reduced shape: {protocol.get('final_output', {}).get('reduced_dataset_shape', '')}",
        f"- Selected features: {len(protocol.get('final_output', {}).get('selected_features', []))}",
        "",
        "## Row Filtering",
    ]
    row_filtering = protocol.get("row_filtering", {})
    for key in ("n_rows_before", "n_rows_after", "n_rows_dropped", "fraction_rows_dropped"):
        lines.append(f"- {key}: {row_filtering.get(key, '')}")

    lines.extend(["", "## Block Detection"])
    block_detection = protocol.get("block_detection", {})
    for key in ("n_receptor_descriptors", "n_ligand_descriptors", "n_scoring_function_descriptors"):
        lines.append(f"- {key}: {block_detection.get(key, '')}")

    lines.extend(["", "## Output Paths"])
    for key, path in protocol.get("final_output", {}).get("output_paths", {}).items():
        lines.append(f"- {key}: `{path}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


## Public ##

def default_ocscore_feature_reduction_config(
        target_column: str = "experimental",
    ) -> FeatureReductionConfig:
    '''Return feature-reduction config for OCScore pipeline wide tables.

    Parameters
    ----------
    target_column : str, optional
        Regression target column preserved outside descriptor blocks.

    Returns
    -------
    FeatureReductionConfig
        Config with pipeline metadata columns excluded from descriptor detection.
    '''

    config = FeatureReductionConfig()
    config.block_detection.metadata_columns = _unique_preserve_order([
        *config.block_detection.metadata_columns,
        *OCSCORE_PIPELINE_METADATA_COLUMNS,
    ])
    config.block_detection.target_columns = [target_column]
    config.missing_rows.subset = "descriptor_columns"
    config.missing_rows.preserve_index = True
    return config


def _validate_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    '''Validate that selected DataFrame columns are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to validate.
    columns : Sequence[str]
        Column names expected to exist and contain numeric values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any column is missing or non-numeric.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="numeric columns")
    non_numeric = [col for col in selected if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Descriptor columns must be numeric. Non-numeric columns: {non_numeric}")


def _validate_no_nan_inf(
    df: pd.DataFrame,
    columns: Sequence[str],
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> None:
    '''Validate that selected columns do not contain NaN or infinite values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to validate.
    columns : Sequence[str]
        Column names to check.
    allow_nan : bool, optional
        If True, allow NaN values, by default False.
    allow_inf : bool, optional
        If True, allow positive or negative infinite values, by default False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any column is missing, contains NaN when ``allow_nan`` is False, or
        contains infinite values when ``allow_inf`` is False.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="validated columns")
    data = df[selected]

    if not allow_nan:
        nan_columns = data.columns[data.isna().any(axis=0)].tolist()
        if nan_columns:
            raise ValueError(f"Columns contain NaN values after filtering: {nan_columns}")

    if not allow_inf:
        numeric = data.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric.to_numpy(dtype=float, copy=False))
        if inf_mask.any():
            inf_columns = numeric.columns[inf_mask.any(axis=0)].tolist()
            raise ValueError(f"Columns contain infinite values after filtering: {inf_columns}")


def validate_descriptor_frame(
    df: pd.DataFrame,
    descriptor_columns: Sequence[str],
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> None:
    '''Validate descriptor columns for existence, numeric dtype, NaN, and inf.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing descriptor columns.
    descriptor_columns : Sequence[str]
        Descriptor columns to validate.
    allow_nan : bool, optional
        If True, allow NaN values, by default False.
    allow_inf : bool, optional
        If True, allow positive or negative infinite values, by default False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no descriptors are provided, a column is missing, a descriptor is
        non-numeric, or forbidden NaN/inf values are present.
    '''

    selected = _as_list(descriptor_columns)
    if not selected:
        raise ValueError("descriptor_columns must contain at least one column.")
    _validate_numeric_columns(df, selected)
    _validate_no_nan_inf(df, selected, allow_nan=allow_nan, allow_inf=allow_inf)


def split_descriptor_blocks(
    columns: Sequence[str],
    metadata_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    receptor_patterns: Optional[Sequence[str]] = None,
    ligand_patterns: Optional[Sequence[str]] = None,
    scoring_patterns: Optional[Sequence[str]] = None,
    use_ligand_class_descriptors: bool = True,
    use_receptor_class_descriptors: bool = True,
    use_scoring_model_descriptors: bool = True,
) -> DescriptorBlocks:
    '''Split dataset columns into receptor, ligand, scoring, metadata, and target blocks.

    Parameters
    ----------
    columns : Sequence[str]
        Dataset columns to classify.
    metadata_columns : Sequence[str], optional
        Metadata column names to preserve. If None, OCDocker defaults are used.
    target_columns : Sequence[str], optional
        Target column names to preserve. If None, OCDocker defaults are used.
    receptor_patterns : Sequence[str], optional
        Prefixes used to detect receptor descriptors by name.
    ligand_patterns : Sequence[str], optional
        Prefixes used to detect ligand descriptors by name.
    scoring_patterns : Sequence[str], optional
        Prefixes used to detect scoring-function descriptors by name.
    use_ligand_class_descriptors : bool, optional
        If True, use ``Ligand.allDescriptors`` before pattern fallback.
    use_receptor_class_descriptors : bool, optional
        If True, use ``Receptor.allDescriptors`` before pattern fallback.
    use_scoring_model_descriptors : bool, optional
        If True, use ``Complexes.allDescriptors`` before pattern fallback.

    Returns
    -------
    DescriptorBlocks
        Column assignments, unmatched columns, duplicate assignments, and
        descriptor-source metadata.
    '''

    all_columns = _as_list(columns)
    metadata = _columns_present(all_columns, metadata_columns or DEFAULT_METADATA_COLUMNS)
    target = _columns_present(all_columns, target_columns or DEFAULT_TARGET_COLUMNS)
    excluded = set(metadata + target)
    candidate_columns = [col for col in all_columns if col not in excluded]

    receptor_matches: List[str] = []
    ligand_matches: List[str] = []
    scoring_matches: List[str] = []

    sources = {
        "receptor": "patterns",
        "ligand": "patterns",
        "scoring": "patterns",
    }

    if use_receptor_class_descriptors:
        receptor_from_class = _match_by_descriptor_names(candidate_columns, _get_receptor_descriptor_names_from_receptor_class())
        if receptor_from_class:
            receptor_matches.extend(receptor_from_class)
            sources["receptor"] = "Receptor.allDescriptors + patterns"
    if use_ligand_class_descriptors:
        ligand_from_class = _match_by_descriptor_names(candidate_columns, _get_ligand_descriptor_names_from_ligand_class())
        if ligand_from_class:
            ligand_matches.extend(ligand_from_class)
            sources["ligand"] = "Ligand.allDescriptors + patterns"
    if use_scoring_model_descriptors:
        scoring_from_model = _match_by_descriptor_names(candidate_columns, _get_scoring_descriptor_names_from_complexes_model())
        if scoring_from_model:
            scoring_matches.extend(scoring_from_model)
            sources["scoring"] = "Complexes.allDescriptors + patterns"

    receptor_pattern_values = DEFAULT_RECEPTOR_PATTERNS if receptor_patterns is None else receptor_patterns
    ligand_pattern_values = DEFAULT_LIGAND_PATTERNS if ligand_patterns is None else ligand_patterns
    scoring_pattern_values = DEFAULT_SCORING_PATTERNS if scoring_patterns is None else scoring_patterns
    receptor_matches.extend(_match_by_patterns(candidate_columns, receptor_pattern_values))
    ligand_matches.extend(_match_by_patterns(candidate_columns, ligand_pattern_values))
    scoring_matches.extend(_match_by_patterns(candidate_columns, scoring_pattern_values))

    receptor_matches = _unique_preserve_order(receptor_matches)
    ligand_matches = _unique_preserve_order(ligand_matches)
    scoring_matches = _unique_preserve_order(scoring_matches)

    assignments: Dict[str, List[str]] = {}
    for block_name, block_columns in (
        ("receptor", receptor_matches),
        ("ligand", ligand_matches),
        ("scoring", scoring_matches),
    ):
        for col in block_columns:
            assignments.setdefault(col, []).append(block_name)

    duplicate_assignments = {col: blocks for col, blocks in assignments.items() if len(blocks) > 1}
    assigned = set(assignments)
    unmatched = [col for col in candidate_columns if col not in assigned]

    return DescriptorBlocks(
        receptor=receptor_matches,
        ligand=ligand_matches,
        scoring=scoring_matches,
        metadata=metadata,
        target=target,
        unmatched=unmatched,
        duplicate_assignments=duplicate_assignments,
        sources=sources,
    )


def summarize_blocks(blocks: DescriptorBlocks) -> pd.DataFrame:
    '''Build a compact block summary table.

    Parameters
    ----------
    blocks : DescriptorBlocks
        Descriptor block assignments to summarize.

    Returns
    -------
    pd.DataFrame
        Table with block name, number of columns, detection source, and columns.
    '''

    rows = []
    for block, cols in blocks.items():
        rows.append({
            "block": block,
            "n_columns": len(cols),
            "source": blocks.sources.get(block, ""),
            "columns": ",".join(cols),
        })
    rows.append({
        "block": "metadata",
        "n_columns": len(blocks.metadata),
        "source": "configured metadata columns",
        "columns": ",".join(blocks.metadata),
    })
    rows.append({
        "block": "target",
        "n_columns": len(blocks.target),
        "source": "configured target columns",
        "columns": ",".join(blocks.target),
    })
    rows.append({
        "block": "unmatched",
        "n_columns": len(blocks.unmatched),
        "source": "not assigned",
        "columns": ",".join(blocks.unmatched),
    })
    return pd.DataFrame(rows)


def _summarize_missingness_by_column(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    '''Summarize missing values by selected column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    columns : Sequence[str]
        Columns to summarize.

    Returns
    -------
    pd.DataFrame
        Table with ``column``, ``n_missing``, and ``fraction_missing``.

    Raises
    ------
    ValueError
        If any requested column is missing.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="missingness columns")
    n_rows = len(df)
    rows = []
    for col in selected:
        n_missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "n_missing": n_missing,
            "fraction_missing": (n_missing / n_rows) if n_rows else 0.0,
        })
    return pd.DataFrame(rows, columns=MISSINGNESS_BY_COLUMN_COLUMNS)


def _summarize_missingness_by_block_from_frame(
    df: pd.DataFrame,
    blocks: DescriptorBlocks,
    target_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    '''Summarize missing values by descriptor block and target columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    blocks : DescriptorBlocks
        Descriptor block assignments.
    target_columns : Sequence[str], optional
        Target columns to include. If None, ``blocks.target`` is used.

    Returns
    -------
    pd.DataFrame
        Table with missing-value counts by receptor, ligand, scoring, and target blocks.

    Raises
    ------
    ValueError
        If any required column is missing.
    '''

    columns = _unique_preserve_order([*blocks.all_descriptor_columns, *_as_list(target_columns or blocks.target)])
    _validate_columns_exist(df, columns, label="missingness columns")
    return _summarize_missingness_by_block(df[columns].isna(), blocks, _as_list(target_columns or blocks.target))


def drop_rows_with_missing_values(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    subset: str = "model_relevant_columns",
    descriptor_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    blocks: Optional[DescriptorBlocks] = None,
    id_columns: Optional[Sequence[str]] = None,
    preserve_index: bool = True,
    return_report: bool = True,
) -> MissingRowsResult:
    '''Drop rows with missing values and return a full row-removal report.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. The input DataFrame is not mutated.
    columns : Sequence[str], optional
        Explicit columns to check. When provided, ``subset`` and block-derived
        columns are ignored.
    subset : str, optional
        Logical subset to check. Supported values are ``"model_relevant_columns"``,
        ``"descriptor_columns"``, ``"descriptor_and_target_columns"``, and
        ``"all_columns"``.
    descriptor_columns : Sequence[str], optional
        Descriptor columns used when ``blocks`` and ``columns`` are not provided.
    target_columns : Sequence[str], optional
        Target columns used when ``blocks`` and ``columns`` are not provided.
    blocks : DescriptorBlocks, optional
        Descriptor block assignments used to resolve model-relevant columns and
        block-level missingness.
    id_columns : Sequence[str], optional
        Identifier columns copied into the dropped-row report when present.
    preserve_index : bool, optional
        If True, preserve the original DataFrame index in ``cleaned_df``.
    return_report : bool, optional
        Kept for API readability. Reports are always returned to avoid silent
        row removal.

    Returns
    -------
    MissingRowsResult
        Cleaned DataFrame, dropped-row report, missingness summaries, and row
        filtering summary.

    Raises
    ------
    ValueError
        If the requested missing-value columns are not present.
    '''

    del return_report  # Reports are always returned to avoid silent row removal.
    selected = _missing_rows_columns(
        df=df,
        columns=columns,
        descriptor_columns=descriptor_columns,
        target_columns=target_columns,
        blocks=blocks,
        subset=subset,
    )

    missing_mask = df[selected].isna()
    rows_to_drop = missing_mask.any(axis=1)
    dropped_index = df.index[rows_to_drop]

    id_cols = _columns_present(df.columns, id_columns or DEFAULT_ID_COLUMNS)
    dropped_rows_records: List[Dict[str, Any]] = []
    for idx in dropped_index:
        row_mask = missing_mask.loc[idx]
        missing_columns = [col for col, missing in row_mask.items() if bool(missing)]
        record: Dict[str, Any] = {
            "original_index": idx,
            "n_missing_total": len(missing_columns),
            "missing_columns": ",".join(missing_columns),
            "drop_reason": "missing_values",
        }
        for id_col in id_cols:
            record[id_col] = df.loc[idx, id_col]
        dropped_rows_records.append(record)

    dropped_rows_columns = _unique_preserve_order([*MISSING_DROPPED_ROWS_COLUMNS, *id_cols])
    dropped_rows = pd.DataFrame(dropped_rows_records, columns=dropped_rows_columns)
    missingness_by_column = _summarize_missingness_by_column(df, selected)
    missingness_by_block = _summarize_missingness_by_block(
        missing_mask,
        blocks=blocks,
        target_columns=_as_list(target_columns or (blocks.target if blocks else [])),
    )

    cleaned_df = df.drop(index=dropped_index).copy()
    if not preserve_index:
        cleaned_df = cleaned_df.reset_index(drop=True)

    n_before = int(len(df))
    n_after = int(len(cleaned_df))
    n_dropped = int(n_before - n_after)
    summary = {
        "n_rows_before": n_before,
        "n_rows_after": n_after,
        "n_rows_dropped": n_dropped,
        "fraction_rows_dropped": (n_dropped / n_before) if n_before else 0.0,
        "subset": subset,
        "columns_checked": selected,
        "id_columns": id_cols,
        "missing_values_by_column": dict(zip(missingness_by_column["column"], missingness_by_column["n_missing"])),
        "missing_values_by_block": dict(zip(missingness_by_block["block"], missingness_by_block["n_missing_values"])),
    }

    if n_dropped:
        LOGGER.info("Dropped %s rows with missing values from %s rows.", n_dropped, n_before)

    return MissingRowsResult(
        cleaned_df=cleaned_df,
        dropped_rows=dropped_rows,
        missingness_by_column=missingness_by_column,
        missingness_by_block=missingness_by_block,
        summary=summary,
    )


def find_constant_features(df: pd.DataFrame, columns: Sequence[str], block: str = "") -> pd.DataFrame:
    '''Find columns with exactly one unique value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns.
    columns : Sequence[str]
        Feature columns to inspect.
    block : str, optional
        Descriptor block name stored in the report.

    Returns
    -------
    pd.DataFrame
        Drop report for constant columns. Empty when no columns are constant.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="constant-feature columns")
    rows = []
    for col in selected:
        n_unique = int(df[col].nunique(dropna=False))
        if n_unique <= 1:
            value = df[col].iloc[0] if len(df) else None
            rows.append({
                "block": block,
                "feature": col,
                "reason": "constant",
                "n_unique": n_unique,
                "representative_value": value,
            })
    return pd.DataFrame(rows, columns=CONSTANT_FEATURE_COLUMNS)


def find_near_constant_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    threshold: float = 0.995,
    block: str = "",
) -> pd.DataFrame:
    '''Find columns where one value accounts for more than ``threshold`` rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns.
    columns : Sequence[str]
        Feature columns to inspect.
    threshold : float, optional
        Dominant-value fraction above which a feature is near-constant.
    block : str, optional
        Descriptor block name stored in the report.

    Returns
    -------
    pd.DataFrame
        Drop report for near-constant columns. Empty when none are found.

    Raises
    ------
    ValueError
        If ``threshold`` is not in the interval ``(0, 1]``.
    '''

    _validate_probability_threshold(threshold)
    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="near-constant-feature columns")
    n_rows = len(df)
    rows: list[dict[str, Any]] = []
    if n_rows == 0:
        return pd.DataFrame(rows, columns=NEAR_CONSTANT_FEATURE_COLUMNS)
    for col in selected:
        counts = df[col].value_counts(dropna=False)
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        top_fraction = top_count / n_rows
        if top_fraction > threshold:
            rows.append({
                "block": block,
                "feature": col,
                "reason": "near_constant",
                "threshold": float(threshold),
                "top_value": top_value,
                "top_count": top_count,
                "top_fraction": top_fraction,
            })
    return pd.DataFrame(rows, columns=NEAR_CONSTANT_FEATURE_COLUMNS)


def find_duplicate_features(df: pd.DataFrame, columns: Sequence[str], block: str = "") -> pd.DataFrame:
    '''Find exact duplicate columns and keep the first stable representative.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric feature columns.
    columns : Sequence[str]
        Feature columns to inspect in retention order.
    block : str, optional
        Descriptor block name stored in the report.

    Returns
    -------
    pd.DataFrame
        Drop report with kept and dropped duplicate features.

    Raises
    ------
    ValueError
        If any requested column is missing or non-numeric.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="duplicate-feature columns")
    _validate_numeric_columns(df, selected)

    signatures: Dict[Tuple[int, ...], List[str]] = {}
    rows: List[Dict[str, Any]] = []
    for col in selected:
        hashed = pd.util.hash_pandas_object(df[col], index=False).to_numpy(dtype=np.uint64)
        signature = tuple(int(value) for value in hashed)
        representatives = signatures.setdefault(signature, [])
        kept_feature = next((candidate for candidate in representatives if df[col].equals(df[candidate])), None)
        if kept_feature is not None:
            rows.append({
                "block": block,
                "kept_feature": kept_feature,
                "dropped_feature": col,
                "reason": "duplicate",
            })
        else:
            representatives.append(col)
    return pd.DataFrame(rows, columns=DUPLICATE_FEATURE_COLUMNS)


def apply_feature_drops(
    columns: Sequence[str],
    dropped_features: Union[pd.DataFrame, Sequence[str], pd.Series],
) -> List[str]:
    '''Return columns after dropping requested features, preserving order.

    Parameters
    ----------
    columns : Sequence[str]
        Original feature columns.
    dropped_features : pd.DataFrame or Sequence[str] or pd.Series
        Features to remove. DataFrames may contain ``dropped_feature``,
        ``feature``, or ``molecular_feature`` columns.

    Returns
    -------
    list[str]
        Columns remaining after dropping requested features.
    '''

    drop_set = set(_dropped_features_from_any(dropped_features))
    return [col for col in _as_list(columns) if col not in drop_set]


def _compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "spearman",
) -> pd.DataFrame:
    '''Compute a correlation matrix for selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric feature columns.
    columns : Sequence[str]
        Columns included in the correlation matrix.
    method : str, optional
        Correlation method passed to ``pandas.DataFrame.corr``.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.

    Raises
    ------
    ValueError
        If any requested column is missing or non-numeric.
    '''

    selected = _as_list(columns)
    _validate_columns_exist(df, selected, label="correlation columns")
    if not selected:
        return pd.DataFrame()
    _validate_numeric_columns(df, selected)
    return df[selected].corr(method=method)


def compute_intra_block_correlations(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "spearman",
    threshold: Optional[float] = None,
    block: str = "",
) -> CorrelationReport:
    '''Compute intra-block correlations as a matrix and long-form pair report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric feature columns.
    columns : Sequence[str]
        Block-specific feature columns.
    method : str, optional
        Correlation method passed to ``pandas.DataFrame.corr``.
    threshold : float, optional
        Optional threshold used to flag pairs in the long-form report.
    block : str, optional
        Descriptor block name stored in the report.

    Returns
    -------
    CorrelationReport
        Correlation matrix and pairwise upper-triangle report.
    '''

    matrix = _compute_correlation_matrix(df, columns, method=method)
    pairs = _long_form_upper_triangle(matrix, method=method, block=block, threshold=threshold)
    return CorrelationReport(matrix=matrix, pairs=pairs, method=method, block=block, threshold=threshold)


def filter_correlated_features(
    corr_report: Union[CorrelationReport, pd.DataFrame],
    threshold: float,
    retention_policy: str = "first",
) -> CorrelationFilterResult:
    '''Filter correlated features deterministically from a pairwise correlation report.

    Parameters
    ----------
    corr_report : CorrelationReport or pd.DataFrame
        Correlation report returned by ``compute_intra_block_correlations`` or a
        compatible DataFrame.
    threshold : float
        Absolute correlation threshold above which a feature is dropped.
    retention_policy : str, optional
        Feature retention policy. Currently only ``"first"`` is supported.

    Returns
    -------
    CorrelationFilterResult
        Kept features, dropped features, and the filtering report.

    Raises
    ------
    ValueError
        If ``threshold`` is invalid, the retention policy is unsupported, or the
        report lacks required columns.
    '''

    _validate_probability_threshold(threshold)
    if retention_policy != "first":
        raise ValueError("Only retention_policy='first' is currently supported.")

    if isinstance(corr_report, CorrelationReport):
        pairs = corr_report.pairs.copy()
        initial_features = list(corr_report.matrix.columns)
        block = corr_report.block
    else:
        pairs = corr_report.copy()
        initial_features = _unique_preserve_order(
            [*pairs.get("feature_1", pd.Series(dtype=str)).dropna().tolist(), *pairs.get("feature_2", pd.Series(dtype=str)).dropna().tolist()]
        )
        block = ""

    retained = set(initial_features)
    dropped: List[str] = []
    report_rows: List[Dict[str, Any]] = []
    if pairs.empty:
        return CorrelationFilterResult(
            kept_features=initial_features,
            dropped_features=[],
            report=pd.DataFrame(columns=CORRELATION_FILTER_COLUMNS),
        )

    required = {"feature_1", "feature_2", "correlation", "abs_correlation"}
    missing_cols = required.difference(pairs.columns)
    if missing_cols:
        raise ValueError(f"Correlation report is missing required columns: {sorted(missing_cols)}")

    filtered_pairs = pairs[pairs["abs_correlation"] > threshold].copy()
    if "pair_order" not in filtered_pairs.columns:
        filtered_pairs["_pair_order"] = np.arange(len(filtered_pairs))
        order_column = "_pair_order"
    else:
        order_column = "pair_order"
    filtered_pairs = filtered_pairs.sort_values(by=order_column, kind="mergesort")

    for _, row in filtered_pairs.iterrows():
        feature_1 = str(row["feature_1"])
        feature_2 = str(row["feature_2"])
        if feature_1 not in retained or feature_2 not in retained:
            continue
        kept_feature = feature_1
        dropped_feature = feature_2
        retained.remove(dropped_feature)
        dropped.append(dropped_feature)
        report_rows.append({
            "block": row.get("block", block),
            "kept_feature": kept_feature,
            "dropped_feature": dropped_feature,
            "correlation": float(row["correlation"]),
            "abs_correlation": float(row["abs_correlation"]),
            "threshold": float(threshold),
            "retention_policy": retention_policy,
            "reason": "intra_block_correlation",
        })

    kept_features = [feature for feature in initial_features if feature in retained]
    return CorrelationFilterResult(
        kept_features=kept_features,
        dropped_features=dropped,
        report=pd.DataFrame(report_rows, columns=CORRELATION_FILTER_COLUMNS),
    )


def compute_cross_block_correlations(
    df: pd.DataFrame,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    method: str = "spearman",
    threshold: float = 0.95,
    left_block: str = "left",
    right_block: str = "right",
) -> pd.DataFrame:
    '''Compute pairwise cross-block correlations and flag values above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both descriptor blocks.
    left_columns : Sequence[str]
        Columns from the left descriptor block.
    right_columns : Sequence[str]
        Columns from the right descriptor block.
    method : str, optional
        Correlation method passed to ``pandas.DataFrame.corr``.
    threshold : float, optional
        Absolute correlation threshold used to flag pairs.
    left_block : str, optional
        Name stored for the left descriptor block.
    right_block : str, optional
        Name stored for the right descriptor block.

    Returns
    -------
    pd.DataFrame
        Pairwise cross-block correlation report. Empty when either block is empty.

    Raises
    ------
    ValueError
        If requested columns are missing or non-numeric.
    '''

    _validate_probability_threshold(threshold)
    left = _as_list(left_columns)
    right = _as_list(right_columns)
    selected = _unique_preserve_order([*left, *right])
    _validate_columns_exist(df, selected, label="cross-block correlation columns")
    if not left or not right:
        return pd.DataFrame(columns=CROSS_CORRELATION_COLUMNS)
    _validate_numeric_columns(df, selected)

    corr = df[selected].corr(method=method)
    rows: List[Dict[str, Any]] = []
    for left_col in left:
        for right_col in right:
            value = corr.loc[left_col, right_col]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            rows.append({
                "left_block": left_block,
                "right_block": right_block,
                "left_feature": left_col,
                "right_feature": right_col,
                "correlation": float(value),
                "abs_correlation": abs_value,
                "method": method,
                "threshold": float(threshold),
                "flagged": bool(abs_value > threshold),
            })
    return pd.DataFrame(rows, columns=CROSS_CORRELATION_COLUMNS)


def compute_cross_block_predictability(
    df: pd.DataFrame,
    predictor_columns: Sequence[str],
    target_columns: Sequence[str],
    model: str = "ridge",
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = 1,
    predictor_block: str = "predictor",
    target_block: str = "scoring",
) -> pd.DataFrame:
    '''Estimate how predictable target columns are from predictor columns using CV R2.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictors and targets.
    predictor_columns : Sequence[str]
        Numeric columns used as Ridge predictors.
    target_columns : Sequence[str]
        Numeric target columns predicted one at a time.
    model : str, optional
        Predictability model. Currently only ``"ridge"`` is supported.
    cv_folds : int, optional
        Requested number of cross-validation folds.
    random_seed : int, optional
        Seed used for deterministic KFold shuffling.
    n_jobs : int, optional
        Number of parallel jobs passed to ``sklearn.model_selection.cross_val_score``.
        Use 1 for serial execution or -1 for all available cores.
    predictor_block : str, optional
        Predictor block name stored in the report.
    target_block : str, optional
        Target block name stored in the report.

    Returns
    -------
    pd.DataFrame
        CV R2 report with mean, standard deviation, number of predictors, and
        redundancy interpretation. Empty when predictors or targets are empty.

    Raises
    ------
    ValueError
        If the model is unsupported, folds are invalid, data are too small, or
        requested columns are missing or non-numeric.
    '''

    if model != "ridge":
        raise ValueError("Only model='ridge' is currently supported.")
    predictors = _as_list(predictor_columns)
    targets = _as_list(target_columns)
    selected = _unique_preserve_order([*predictors, *targets])
    _validate_columns_exist(df, selected, label="predictability columns")
    if not predictors or not targets:
        return pd.DataFrame(columns=CROSS_PREDICTABILITY_COLUMNS)
    _validate_numeric_columns(df, selected)
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")
    if len(df) < 4:
        raise ValueError("At least 4 rows are required for Ridge CV predictability diagnostics.")

    effective_folds = min(cv_folds, max(2, len(df) // 2))
    cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_seed)
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_seed))
    X = df[predictors].to_numpy(dtype=float)

    rows = []
    for target in targets:
        y = df[target].to_numpy(dtype=float)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2", n_jobs=n_jobs)
        mean_r2 = float(np.nanmean(scores))
        std_r2 = float(np.nanstd(scores))
        rows.append({
            "target_feature": target,
            "target_block": target_block,
            "predictor_block": predictor_block,
            "model": model,
            "mean_cv_r2": mean_r2,
            "std_cv_r2": std_r2,
            "cv_folds": int(effective_folds),
            "requested_cv_folds": int(cv_folds),
            "n_predictors": len(predictors),
            "random_seed": int(random_seed),
            "n_jobs": int(n_jobs),
            "interpretation": _interpret_r2(mean_r2),
        })
    return pd.DataFrame(rows, columns=CROSS_PREDICTABILITY_COLUMNS)


def filter_cross_block_redundant_features(
    cross_corr_report: pd.DataFrame,
    molecular_columns: Sequence[str],
    scoring_columns: Sequence[str],
    threshold: float = 0.95,
    scoring_function_priority: bool = False,
) -> CrossBlockFilterResult:
    '''Optionally drop molecular descriptors correlated with scoring functions.

    Parameters
    ----------
    cross_corr_report : pd.DataFrame
        Cross-block pairwise correlation report.
    molecular_columns : Sequence[str]
        Molecular descriptor columns eligible for removal.
    scoring_columns : Sequence[str]
        Scoring-function descriptor columns that may be prioritized.
    threshold : float, optional
        Absolute correlation threshold required for removal.
    scoring_function_priority : bool, optional
        If False, no features are removed. If True, strongly correlated molecular
        descriptors may be dropped.

    Returns
    -------
    CrossBlockFilterResult
        Kept molecular descriptors, dropped descriptors, and filtering report.

    Raises
    ------
    ValueError
        If the correlation report lacks required columns.
    '''

    _validate_probability_threshold(threshold)
    molecular = _as_list(molecular_columns)
    scoring = set(_as_list(scoring_columns))
    retained = set(molecular)
    if cross_corr_report.empty or not scoring_function_priority:
        return CrossBlockFilterResult(
            kept_features=molecular,
            dropped_features=[],
            report=pd.DataFrame(columns=CROSS_BLOCK_FILTER_COLUMNS),
        )

    required = {"left_feature", "right_feature", "correlation", "abs_correlation"}
    missing_cols = required.difference(cross_corr_report.columns)
    if missing_cols:
        raise ValueError(f"Cross-correlation report is missing required columns: {sorted(missing_cols)}")

    rows: List[Dict[str, Any]] = []
    candidates = cross_corr_report[cross_corr_report["abs_correlation"] > threshold].copy()
    candidates = candidates.sort_values(by=["left_feature", "right_feature"], kind="mergesort")
    for _, row in candidates.iterrows():
        left_feature = str(row["left_feature"])
        right_feature = str(row["right_feature"])
        if left_feature in retained and right_feature in scoring:
            molecular_feature = left_feature
            scoring_feature = right_feature
        elif right_feature in retained and left_feature in scoring:
            molecular_feature = right_feature
            scoring_feature = left_feature
        else:
            continue
        if molecular_feature not in retained:
            continue
        retained.remove(molecular_feature)
        rows.append({
            "molecular_feature": molecular_feature,
            "scoring_feature": scoring_feature,
            "correlation": float(row["correlation"]),
            "abs_correlation": float(row["abs_correlation"]),
            "threshold": float(threshold),
            "scoring_function_priority": bool(scoring_function_priority),
            "reason": "cross_block_correlation_with_scoring_priority",
        })

    dropped = [row["molecular_feature"] for row in rows]
    return CrossBlockFilterResult(
        kept_features=[col for col in molecular if col in retained],
        dropped_features=dropped,
        report=pd.DataFrame(rows, columns=CROSS_BLOCK_FILTER_COLUMNS),
    )


def compose_selected_features(
    receptor_columns: Sequence[str],
    ligand_columns: Sequence[str],
    scoring_columns: Sequence[str],
) -> List[str]:
    '''Compose final selected feature names in receptor, ligand, scoring order.

    Parameters
    ----------
    receptor_columns : Sequence[str]
        Retained receptor descriptor columns.
    ligand_columns : Sequence[str]
        Retained ligand descriptor columns.
    scoring_columns : Sequence[str]
        Retained scoring-function descriptor columns.

    Returns
    -------
    list[str]
        Unique selected features, preserving receptor, ligand, then scoring order.
    '''

    return _unique_preserve_order([*_as_list(receptor_columns), *_as_list(ligand_columns), *_as_list(scoring_columns)])


def build_reduced_dataframe(
    df: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    target_columns: Optional[Sequence[str]] = None,
    selected_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    '''Build a reduced DataFrame with metadata, target, and selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame. The input is not mutated.
    metadata_columns : Sequence[str], optional
        Metadata columns to retain.
    target_columns : Sequence[str], optional
        Target columns to retain.
    selected_features : Sequence[str], optional
        Selected descriptor columns to retain.

    Returns
    -------
    pd.DataFrame
        Copy of the reduced DataFrame in metadata, target, feature order.

    Raises
    ------
    ValueError
        If any requested column is missing.
    '''

    metadata = _as_list(metadata_columns)
    targets = _as_list(target_columns)
    features = _as_list(selected_features)
    requested = _unique_preserve_order([*metadata, *targets, *features])
    _validate_columns_exist(df, requested, label="reduced DataFrame columns")
    return df[requested].copy()


def _merge_feature_drop_reports(*reports: Union[pd.DataFrame, CorrelationFilterResult, CrossBlockFilterResult]) -> pd.DataFrame:
    '''Merge feature-drop reports into one DataFrame.

    Parameters
    ----------
    *reports : pd.DataFrame or CorrelationFilterResult or CrossBlockFilterResult
        Drop reports or result objects containing a report DataFrame.

    Returns
    -------
    pd.DataFrame
        Concatenated report. Empty when all inputs are empty.
    '''

    frames = []
    for report in reports:
        frame = _report_to_frame(report)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_feature_reduction_protocol(
    config: FeatureReductionConfig,
    blocks: DescriptorBlocks,
    missing_result: MissingRowsResult,
    cleaned_blocks: Mapping[str, Sequence[str]],
    selected_features: Sequence[str],
    reduced_df: pd.DataFrame,
    input_path: Optional[str] = None,
    input_shape: Optional[Tuple[int, int]] = None,
    block_summary: Optional[pd.DataFrame] = None,
    dropped_features: Optional[pd.DataFrame] = None,
    intra_block_correlation_report: Optional[pd.DataFrame] = None,
    cross_block_pairwise_correlation_report: Optional[pd.DataFrame] = None,
    cross_block_predictability_report: Optional[pd.DataFrame] = None,
    cross_block_filter_report: Optional[pd.DataFrame] = None,
    output_paths: Optional[Mapping[str, str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    '''Build a JSON-serializable reproducibility protocol for feature reduction.

    Parameters
    ----------
    config : FeatureReductionConfig
        Configuration used for the run.
    blocks : DescriptorBlocks
        Descriptor block assignments detected from the input dataset.
    missing_result : MissingRowsResult
        Missing-row filtering result.
    cleaned_blocks : Mapping[str, Sequence[str]]
        Final retained columns by descriptor block.
    selected_features : Sequence[str]
        Final selected descriptor columns.
    reduced_df : pd.DataFrame
        Reduced dataset.
    input_path : str, optional
        Input file path when data were loaded from disk.
    input_shape : tuple[int, int], optional
        Shape of the raw input dataset.
    block_summary : pd.DataFrame, optional
        Block summary report.
    dropped_features : pd.DataFrame, optional
        Feature-drop report.
    intra_block_correlation_report : pd.DataFrame, optional
        Intra-block pairwise correlation report.
    cross_block_pairwise_correlation_report : pd.DataFrame, optional
        Cross-block pairwise correlation report.
    cross_block_predictability_report : pd.DataFrame, optional
        Ridge CV predictability report.
    cross_block_filter_report : pd.DataFrame, optional
        Optional cross-block filtering report.
    output_paths : Mapping[str, str], optional
        Output paths already written by an orchestration layer.
    warnings : Sequence[str], optional
        Warning messages to embed in the protocol.

    Returns
    -------
    dict[str, Any]
        JSON-serializable reproducibility protocol.
    '''

    manifest = ocrepro.generate_reproducibility_manifest(include_python_packages=False)
    shape = input_shape if input_shape is not None else tuple(missing_result.summary.get("input_shape", reduced_df.shape))
    block_summary_df = block_summary if block_summary is not None else summarize_blocks(blocks)

    protocol = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "input_path": input_path,
            "shape": list(shape),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "ocdocker": manifest.get("ocdocker", {}),
            "git": manifest.get("git", {}),
            "random_seed": config.cross_block_diagnostics.random_seed,
        },
        "configuration": _to_jsonable(config),
        "row_filtering": missing_result.summary,
        "block_detection": {
            "n_receptor_descriptors": len(blocks.receptor),
            "n_ligand_descriptors": len(blocks.ligand),
            "n_scoring_function_descriptors": len(blocks.scoring),
            "source_receptor": blocks.sources.get("receptor", ""),
            "source_ligand": blocks.sources.get("ligand", ""),
            "source_scoring": blocks.sources.get("scoring", ""),
            "used_receptor_class_metadata": config.block_detection.use_receptor_class_descriptors,
            "used_ligand_class_metadata": config.block_detection.use_ligand_class_descriptors,
            "used_scoring_model_metadata": config.block_detection.use_scoring_model_descriptors,
            "unmatched_descriptor_columns": blocks.unmatched,
            "duplicate_assignment_warnings": blocks.duplicate_assignments,
            "block_summary": _to_jsonable(block_summary_df),
        },
        "column_filtering": {
            "cleaned_blocks": {key: list(value) for key, value in cleaned_blocks.items()},
            "dropped_features": _to_jsonable(dropped_features if dropped_features is not None else pd.DataFrame()),
        },
        "intra_block_correlation_filtering": {
            "method": config.intra_block_correlation.method,
            "receptor_threshold": config.intra_block_correlation.receptor_threshold,
            "ligand_threshold": config.intra_block_correlation.ligand_threshold,
            "scoring_threshold": config.intra_block_correlation.scoring_threshold,
            "retention_policy": config.intra_block_correlation.retention_policy,
            "report": _to_jsonable(intra_block_correlation_report if intra_block_correlation_report is not None else pd.DataFrame()),
        },
        "cross_block_diagnostics": {
            "enabled": config.cross_block_diagnostics.enabled,
            "correlation_threshold": config.cross_block_diagnostics.correlation_threshold,
            "ridge_cv_folds": config.cross_block_diagnostics.ridge_cv_folds,
            "random_seed": config.cross_block_diagnostics.random_seed,
            "n_jobs": config.cross_block_diagnostics.n_jobs,
            "pairwise_report": _to_jsonable(cross_block_pairwise_correlation_report if cross_block_pairwise_correlation_report is not None else pd.DataFrame()),
            "predictability_report": _to_jsonable(cross_block_predictability_report if cross_block_predictability_report is not None else pd.DataFrame()),
        },
        "cross_block_filtering": {
            "enabled": config.cross_block_filtering.enabled,
            "scoring_function_priority": config.cross_block_filtering.scoring_function_priority,
            "report": _to_jsonable(cross_block_filter_report if cross_block_filter_report is not None else pd.DataFrame()),
        },
        "final_output": {
            "selected_features": list(selected_features),
            "reduced_dataset_shape": list(reduced_df.shape),
            "output_paths": dict(output_paths or {}),
            "warnings": list(warnings or []),
        },
    }
    return cast(Dict[str, Any], _to_jsonable(protocol))


def write_feature_reduction_outputs(
    output_dir: Union[str, Path],
    reduced_df: pd.DataFrame,
    selected_features: Sequence[str],
    protocol: Mapping[str, Any],
    missing_result: Optional[MissingRowsResult] = None,
    block_summary: Optional[pd.DataFrame] = None,
    dropped_features: Optional[pd.DataFrame] = None,
    intra_block_correlation_report: Optional[pd.DataFrame] = None,
    cross_block_pairwise_correlation_report: Optional[pd.DataFrame] = None,
    cross_block_predictability_report: Optional[pd.DataFrame] = None,
    cross_block_filter_report: Optional[pd.DataFrame] = None,
    config: Optional[FeatureReductionConfig] = None,
    write_markdown: bool = True,
) -> Dict[str, str]:
    '''Write reduced dataset, reports, config, and protocol files with stable names.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory where output files are written.
    reduced_df : pd.DataFrame
        Reduced dataset to write as ``reduced_dataset.csv``.
    selected_features : Sequence[str]
        Selected features written as JSON and text.
    protocol : Mapping[str, Any]
        Reproducibility protocol to write as JSON and optionally Markdown.
    missing_result : MissingRowsResult, optional
        Missing-row result used to write row-filtering reports.
    block_summary : pd.DataFrame, optional
        Block summary report.
    dropped_features : pd.DataFrame, optional
        Feature-drop report.
    intra_block_correlation_report : pd.DataFrame, optional
        Intra-block pairwise correlation report.
    cross_block_pairwise_correlation_report : pd.DataFrame, optional
        Cross-block pairwise correlation report.
    cross_block_predictability_report : pd.DataFrame, optional
        Ridge CV predictability report.
    cross_block_filter_report : pd.DataFrame, optional
        Optional cross-block filtering report.
    config : FeatureReductionConfig, optional
        Configuration written to ``config_used.json``.
    write_markdown : bool, optional
        If True, also write ``feature_reduction_protocol.md``.

    Returns
    -------
    dict[str, str]
        Mapping from output artifact names to written file paths.
    '''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, str] = {}

    def _write_csv(key: str, filename: str, df_to_write: Optional[pd.DataFrame]) -> None:
        path = out / filename
        frame = df_to_write if df_to_write is not None else pd.DataFrame()
        frame.to_csv(path, index=False)
        output_paths[key] = str(path)

    reduced_path = out / "reduced_dataset.csv"
    reduced_df.to_csv(reduced_path, index=False)
    output_paths["reduced_dataset"] = str(reduced_path)

    selected_json_path = out / "selected_features.json"
    selected_json_path.write_text(json.dumps(list(selected_features), indent=2) + "\n", encoding="utf-8")
    output_paths["selected_features_json"] = str(selected_json_path)

    selected_txt_path = out / "selected_features.txt"
    selected_txt_path.write_text("\n".join(selected_features) + "\n", encoding="utf-8")
    output_paths["selected_features_txt"] = str(selected_txt_path)

    feature_selection = FeatureSelectionScope.precomputed_global(
        fit_dataset="merged_pdbbind_dudez",
        selected_features_source="reduction_protocol",
        n_selected_features=len(selected_features),
    )
    removed_names = _dropped_features_from_any(dropped_features) if dropped_features is not None else []
    feature_selection = attach_feature_hashes(
        feature_selection,
        selected_features=list(selected_features),
        removed_features=removed_names,
    )
    feature_selection_path = write_feature_selection_json(out, feature_selection)
    output_paths["feature_selection_json"] = str(feature_selection_path)

    _write_csv("dropped_rows_missing_values", "dropped_rows_missing_values.csv", missing_result.dropped_rows if missing_result else None)
    _write_csv("missingness_by_column", "missingness_by_column.csv", missing_result.missingness_by_column if missing_result else None)
    _write_csv("missingness_by_block", "missingness_by_block.csv", missing_result.missingness_by_block if missing_result else None)
    _write_csv("block_summary", "block_summary.csv", block_summary)
    _write_csv("dropped_features", "dropped_features.csv", dropped_features)
    _write_csv("intra_block_correlation_report", "intra_block_correlation_report.csv", intra_block_correlation_report)
    _write_csv("cross_block_pairwise_correlation_report", "cross_block_pairwise_correlation_report.csv", cross_block_pairwise_correlation_report)
    _write_csv("cross_block_predictability_report", "cross_block_predictability_report.csv", cross_block_predictability_report)
    _write_csv("cross_block_filter_report", "cross_block_filter_report.csv", cross_block_filter_report)

    config_path = out / "config_used.json"
    config_payload = _to_jsonable(config) if config is not None else protocol.get("configuration", {})
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths["config_used"] = str(config_path)

    protocol_json_path = out / "feature_reduction_protocol.json"
    output_paths["feature_reduction_protocol_json"] = str(protocol_json_path)
    if write_markdown:
        protocol_md_path = out / "feature_reduction_protocol.md"
        output_paths["feature_reduction_protocol_md"] = str(protocol_md_path)
    else:
        protocol_md_path = None

    protocol_payload = dict(protocol)
    protocol_payload.setdefault("final_output", {})
    protocol_payload["final_output"]["output_paths"] = dict(output_paths)
    protocol_json_path.write_text(json.dumps(_to_jsonable(protocol_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if protocol_md_path is not None:
        _write_markdown_protocol(protocol_payload, protocol_md_path)

    return output_paths


def run_feature_reduction_protocol(
    df: Optional[pd.DataFrame] = None,
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[FeatureReductionConfig] = None,
    write_outputs: bool = False,
) -> FeatureReductionResult:
    '''Convenience orchestration over the granular feature-reduction API.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataset. The DataFrame is copied before processing.
    input_path : str or pathlib.Path, optional
        CSV file to load when ``df`` is not provided. Raw ``pandas.read_csv`` is
        used so missing rows can be reported before removal.
    output_dir : str or pathlib.Path, optional
        Directory where reports are written when ``write_outputs`` is True.
    config : FeatureReductionConfig, optional
        Run configuration. If None, scientific defaults are used.
    write_outputs : bool, optional
        If True, write reduced data, reports, and protocol files.

    Returns
    -------
    FeatureReductionResult
        Reduced dataset, selected features, descriptor blocks, reports, protocol,
        and output paths.

    Raises
    ------
    ValueError
        If input arguments are inconsistent, no descriptors are detected, or
        output writing is requested without ``output_dir``.
    '''

    if df is None and input_path is None:
        raise ValueError("Either df or input_path must be provided.")
    if df is not None and input_path is not None:
        raise ValueError("Provide either df or input_path, not both.")

    cfg = config or FeatureReductionConfig()

    def _log_verbose(message: str, *args: Any) -> None:
        if cfg.verbose:
            LOGGER.info(message, *args)

    _log_verbose("Starting feature-reduction protocol.")
    if df is None:
        source_path = str(input_path)
        raw_df = pd.read_csv(source_path)
    else:
        source_path = None
        raw_df = df.copy()

    input_shape = tuple(raw_df.shape)
    _log_verbose("Raw input shape: %s rows x %s columns.", input_shape[0], input_shape[1])
    blocks = split_descriptor_blocks(
        columns=raw_df.columns,
        metadata_columns=cfg.block_detection.metadata_columns,
        target_columns=cfg.block_detection.target_columns,
        receptor_patterns=cfg.block_detection.receptor_patterns,
        ligand_patterns=cfg.block_detection.ligand_patterns,
        scoring_patterns=cfg.block_detection.scoring_patterns,
        use_ligand_class_descriptors=cfg.block_detection.use_ligand_class_descriptors,
        use_receptor_class_descriptors=cfg.block_detection.use_receptor_class_descriptors,
        use_scoring_model_descriptors=cfg.block_detection.use_scoring_model_descriptors,
    )
    block_summary = summarize_blocks(blocks)
    _log_verbose(
        "Detected descriptor blocks: receptor=%s, ligand=%s, scoring=%s, unmatched=%s.",
        len(blocks.receptor),
        len(blocks.ligand),
        len(blocks.scoring),
        len(blocks.unmatched),
    )

    if not blocks.all_descriptor_columns:
        raise ValueError("No descriptor columns were detected. Check descriptor metadata or patterns.")

    if cfg.missing_rows.enabled:
        missing_result = drop_rows_with_missing_values(
            df=raw_df,
            subset=cfg.missing_rows.subset,
            blocks=blocks,
            id_columns=blocks.metadata,
            preserve_index=cfg.missing_rows.preserve_index,
        )
        complete_df = missing_result.cleaned_df
        _log_verbose(
            "Missing-row filtering kept %s/%s rows using subset=%s.",
            missing_result.summary["n_rows_after"],
            missing_result.summary["n_rows_before"],
            missing_result.summary["subset"],
        )
    else:
        complete_df = raw_df.copy()
        missing_result = MissingRowsResult(
            cleaned_df=complete_df,
            dropped_rows=pd.DataFrame(),
            missingness_by_column=_summarize_missingness_by_column(raw_df, blocks.all_model_columns),
            missingness_by_block=_summarize_missingness_by_block_from_frame(raw_df, blocks),
            summary={
                "n_rows_before": len(raw_df),
                "n_rows_after": len(raw_df),
                "n_rows_dropped": 0,
                "fraction_rows_dropped": 0.0,
                "subset": "disabled",
                "columns_checked": blocks.all_model_columns,
                "id_columns": blocks.metadata,
                "missing_values_by_column": {},
                "missing_values_by_block": {},
            },
        )
        _log_verbose("Missing-row filtering disabled; using all %s rows.", len(complete_df))

    _log_verbose("Validating %s descriptor columns after missing-row filtering.", len(blocks.all_descriptor_columns))
    validate_descriptor_frame(complete_df, blocks.all_descriptor_columns, allow_nan=False, allow_inf=False)

    cleaned_blocks: Dict[str, List[str]] = {}
    all_drop_reports: List[pd.DataFrame] = []
    intra_reports: List[pd.DataFrame] = []

    for block_name, block_columns in blocks.items():
        current_columns = list(block_columns)
        _log_verbose("Filtering %s block: starting with %s columns.", block_name, len(current_columns))
        if cfg.column_quality.remove_constant:
            constant_report = find_constant_features(complete_df, current_columns, block=block_name)
            current_columns = apply_feature_drops(current_columns, constant_report)
            _log_verbose("%s block: dropped %s constant columns.", block_name, len(constant_report))
            all_drop_reports.append(constant_report)
        if cfg.column_quality.remove_near_constant:
            near_constant_report = find_near_constant_features(
                complete_df,
                current_columns,
                threshold=cfg.column_quality.near_constant_threshold,
                block=block_name,
            )
            current_columns = apply_feature_drops(current_columns, near_constant_report)
            _log_verbose("%s block: dropped %s near-constant columns.", block_name, len(near_constant_report))
            all_drop_reports.append(near_constant_report)
        if cfg.column_quality.remove_duplicates:
            duplicate_report = find_duplicate_features(complete_df, current_columns, block=block_name)
            current_columns = apply_feature_drops(current_columns, duplicate_report)
            _log_verbose("%s block: dropped %s duplicate columns.", block_name, len(duplicate_report))
            all_drop_reports.append(duplicate_report)

        threshold = cfg.intra_block_correlation.threshold_for_block(block_name)
        corr_report = compute_intra_block_correlations(
            complete_df,
            current_columns,
            method=cfg.intra_block_correlation.method,
            threshold=threshold,
            block=block_name,
        )
        corr_filter = filter_correlated_features(
            corr_report,
            threshold=threshold,
            retention_policy=cfg.intra_block_correlation.retention_policy,
        )
        current_columns = apply_feature_drops(current_columns, corr_filter.dropped_features)
        _log_verbose(
            "%s block: dropped %s correlated columns; retained %s columns.",
            block_name,
            len(corr_filter.dropped_features),
            len(current_columns),
        )
        if not corr_report.pairs.empty:
            intra_reports.append(corr_report.pairs)
        all_drop_reports.append(corr_filter.report)
        cleaned_blocks[block_name] = current_columns

    dropped_features = _merge_feature_drop_reports(*all_drop_reports)
    intra_block_correlation_report = pd.concat(intra_reports, ignore_index=True, sort=False) if intra_reports else pd.DataFrame()

    cross_pairwise_reports: List[pd.DataFrame] = []
    cross_predictability_reports: List[pd.DataFrame] = []
    if cfg.cross_block_diagnostics.enabled or cfg.cross_block_filtering.enabled:
        _log_verbose("Computing cross-block pairwise correlation diagnostics.")
        receptor_scoring_corr = compute_cross_block_correlations(
            complete_df,
            cleaned_blocks["receptor"],
            cleaned_blocks["scoring"],
            method=cfg.intra_block_correlation.method,
            threshold=cfg.cross_block_diagnostics.correlation_threshold,
            left_block="receptor",
            right_block="scoring",
        )
        ligand_scoring_corr = compute_cross_block_correlations(
            complete_df,
            cleaned_blocks["ligand"],
            cleaned_blocks["scoring"],
            method=cfg.intra_block_correlation.method,
            threshold=cfg.cross_block_diagnostics.correlation_threshold,
            left_block="ligand",
            right_block="scoring",
        )
        cross_pairwise_reports.extend([receptor_scoring_corr, ligand_scoring_corr])

        if cfg.cross_block_diagnostics.enabled and cleaned_blocks["scoring"]:
            _log_verbose("Computing Ridge CV cross-block predictability diagnostics with n_jobs=%s.", cfg.cross_block_diagnostics.n_jobs)
            if cleaned_blocks["receptor"]:
                cross_predictability_reports.append(compute_cross_block_predictability(
                    complete_df,
                    predictor_columns=cleaned_blocks["receptor"],
                    target_columns=cleaned_blocks["scoring"],
                    cv_folds=cfg.cross_block_diagnostics.ridge_cv_folds,
                    random_seed=cfg.cross_block_diagnostics.random_seed,
                    n_jobs=cfg.cross_block_diagnostics.n_jobs,
                    predictor_block="receptor",
                    target_block="scoring",
                ))
            if cleaned_blocks["ligand"]:
                cross_predictability_reports.append(compute_cross_block_predictability(
                    complete_df,
                    predictor_columns=cleaned_blocks["ligand"],
                    target_columns=cleaned_blocks["scoring"],
                    cv_folds=cfg.cross_block_diagnostics.ridge_cv_folds,
                    random_seed=cfg.cross_block_diagnostics.random_seed,
                    n_jobs=cfg.cross_block_diagnostics.n_jobs,
                    predictor_block="ligand",
                    target_block="scoring",
                ))
            molecular_columns = [*cleaned_blocks["receptor"], *cleaned_blocks["ligand"]]
            if molecular_columns:
                cross_predictability_reports.append(compute_cross_block_predictability(
                    complete_df,
                    predictor_columns=molecular_columns,
                    target_columns=cleaned_blocks["scoring"],
                    cv_folds=cfg.cross_block_diagnostics.ridge_cv_folds,
                    random_seed=cfg.cross_block_diagnostics.random_seed,
                    n_jobs=cfg.cross_block_diagnostics.n_jobs,
                    predictor_block="receptor+ligand",
                    target_block="scoring",
                ))

    cross_block_pairwise_correlation_report = (
        pd.concat([frame for frame in cross_pairwise_reports if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in cross_pairwise_reports)
        else pd.DataFrame()
    )
    cross_block_predictability_report = (
        pd.concat([frame for frame in cross_predictability_reports if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in cross_predictability_reports)
        else pd.DataFrame()
    )

    cross_block_filter_report = pd.DataFrame()
    if cfg.cross_block_filtering.enabled:
        _log_verbose("Applying optional conservative cross-block filtering.")
        receptor_filter = filter_cross_block_redundant_features(
            cross_block_pairwise_correlation_report,
            molecular_columns=cleaned_blocks["receptor"],
            scoring_columns=cleaned_blocks["scoring"],
            threshold=cfg.cross_block_diagnostics.correlation_threshold,
            scoring_function_priority=cfg.cross_block_filtering.scoring_function_priority,
        )
        ligand_filter = filter_cross_block_redundant_features(
            cross_block_pairwise_correlation_report,
            molecular_columns=cleaned_blocks["ligand"],
            scoring_columns=cleaned_blocks["scoring"],
            threshold=cfg.cross_block_diagnostics.correlation_threshold,
            scoring_function_priority=cfg.cross_block_filtering.scoring_function_priority,
        )
        cleaned_blocks["receptor"] = receptor_filter.kept_features
        cleaned_blocks["ligand"] = ligand_filter.kept_features
        cross_block_filter_report = _merge_feature_drop_reports(receptor_filter, ligand_filter)
        dropped_features = _merge_feature_drop_reports(dropped_features, cross_block_filter_report)
        _log_verbose("Cross-block filtering dropped %s molecular descriptor columns.", len(cross_block_filter_report))

    selected_features = compose_selected_features(
        receptor_columns=cleaned_blocks["receptor"],
        ligand_columns=cleaned_blocks["ligand"],
        scoring_columns=cleaned_blocks["scoring"],
    )
    _log_verbose("Composed %s selected features.", len(selected_features))
    reduced_df = build_reduced_dataframe(
        complete_df,
        metadata_columns=blocks.metadata,
        target_columns=blocks.target,
        selected_features=selected_features,
    )

    reports = {
        "block_summary": block_summary,
        "dropped_features": dropped_features,
        "intra_block_correlation_report": intra_block_correlation_report,
        "cross_block_pairwise_correlation_report": cross_block_pairwise_correlation_report,
        "cross_block_predictability_report": cross_block_predictability_report,
        "cross_block_filter_report": cross_block_filter_report,
    }

    protocol = build_feature_reduction_protocol(
        config=cfg,
        blocks=blocks,
        missing_result=missing_result,
        cleaned_blocks=cleaned_blocks,
        selected_features=selected_features,
        reduced_df=reduced_df,
        input_path=source_path,
        input_shape=input_shape,
        block_summary=block_summary,
        dropped_features=dropped_features,
        intra_block_correlation_report=intra_block_correlation_report,
        cross_block_pairwise_correlation_report=cross_block_pairwise_correlation_report,
        cross_block_predictability_report=cross_block_predictability_report,
        cross_block_filter_report=cross_block_filter_report,
    )

    output_paths: Dict[str, str] = {}
    if write_outputs:
        if output_dir is None:
            raise ValueError("output_dir must be provided when write_outputs=True.")
        output_paths = write_feature_reduction_outputs(
            output_dir=output_dir,
            reduced_df=reduced_df,
            selected_features=selected_features,
            protocol=protocol,
            missing_result=missing_result,
            block_summary=block_summary,
            dropped_features=dropped_features,
            intra_block_correlation_report=intra_block_correlation_report,
            cross_block_pairwise_correlation_report=cross_block_pairwise_correlation_report,
            cross_block_predictability_report=cross_block_predictability_report,
            cross_block_filter_report=cross_block_filter_report,
            config=cfg,
        )
        protocol["final_output"]["output_paths"] = output_paths
        _log_verbose("Wrote feature-reduction outputs to %s.", output_dir)

    _log_verbose("Feature-reduction protocol complete: reduced shape=%s.", reduced_df.shape)
    return FeatureReductionResult(
        reduced_df=reduced_df,
        selected_features=selected_features,
        blocks=blocks,
        cleaned_blocks=cleaned_blocks,
        missing_result=missing_result,
        protocol=protocol,
        reports=reports,
        output_paths=output_paths,
    )


__all__ = [
    "BlockDetectionConfig",
    "ColumnQualityConfig",
    "CorrelationFilterResult",
    "CorrelationReport",
    "CrossBlockDiagnosticsConfig",
    "CrossBlockFilterResult",
    "CrossBlockFilteringConfig",
    "DescriptorBlocks",
    "FeatureReductionConfig",
    "FeatureReductionResult",
    "IntraBlockCorrelationConfig",
    "MissingRowsConfig",
    "MissingRowsResult",
    "apply_feature_drops",
    "build_feature_reduction_protocol",
    "build_reduced_dataframe",
    "compose_selected_features",
    "default_ocscore_feature_reduction_config",
    "compute_cross_block_correlations",
    "compute_cross_block_predictability",
    "compute_intra_block_correlations",
    "drop_rows_with_missing_values",
    "filter_correlated_features",
    "filter_cross_block_redundant_features",
    "find_constant_features",
    "find_duplicate_features",
    "find_near_constant_features",
    "run_feature_reduction_protocol",
    "split_descriptor_blocks",
    "summarize_blocks",
    "validate_descriptor_frame",
    "write_feature_reduction_outputs",
]
