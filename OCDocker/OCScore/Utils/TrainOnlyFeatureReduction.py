#!/usr/bin/env python3

# Description
###############################################################################
'''
Train-only feature reduction for the unified OCScore modeling protocol.

Fits unsupervised feature reduction on training rows only, freezes selected
features, and applies the frozen selection to validation/test and DUDEz rows.
'''

from __future__ import annotations

# Imports
###############################################################################
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

import OCDocker.OCScore.Utils.FeatureReduction as ocfr
import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Utils.ContentHash import hash_dataframe_partition
from OCDocker.OCScore.Utils.ContentHash import hash_json_dict
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FeatureSelectionScope
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import write_feature_selection_fit_rows_json
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import write_feature_selection_json

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

LOGGER = oclogging.get_logger("ocscore.utils.train_only_feature_reduction")

TRAIN_ONLY_ARTIFACT_JSON = "train_only_feature_reduction.json"
DATASET_COLUMN = "dataset"
FEATURE_BLOCK_NAMES = ("ligand", "receptor", "scoring")


# Classes
###############################################################################


@dataclass(frozen=True)
class SelectedFeatureRowCleanupResult:
    """Rows retained/dropped after selected-feature finite-value cleanup."""

    cleaned_df: pd.DataFrame
    dropped_rows: pd.DataFrame
    kept_mask: np.ndarray
    summary: dict[str, Any]


@dataclass
class TrainOnlyReductionArtifact:
    """Frozen train-only feature reduction artifact."""

    selected_features: list[str]
    removed_features: list[str]
    metadata_columns: list[str]
    target_columns: list[str]
    feature_selection: FeatureSelectionScope
    protocol: dict[str, Any] = field(default_factory=dict)
    transform_artifacts: list[str] = field(default_factory=list)
    transform_artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        '''Serialize the frozen reduction artifact for JSON persistence.

        Returns
        -------
        dict[str, Any]
            Selected features, scope metadata, and transform artifact hashes.
        '''

        return {
            "selected_features": self.selected_features,
            "removed_features": self.removed_features,
            "metadata_columns": self.metadata_columns,
            "target_columns": self.target_columns,
            "feature_selection": self.feature_selection.to_dict(),
            "protocol": self.protocol,
            "transform_artifacts": self.transform_artifacts,
            "transform_artifact_hashes": self.transform_artifact_hashes,
        }


# Functions
###############################################################################
## Private ##


def _collect_removed_features(
        fit_df: pd.DataFrame,
        selected_features: Sequence[str],
        blocks: ocfr.DescriptorBlocks,
    ) -> list[str]:
    descriptor_columns = list(blocks.receptor) + list(blocks.ligand) + list(blocks.scoring)
    return [column for column in descriptor_columns if column not in set(selected_features)]


def _default_transform_artifacts(config: ocfr.FeatureReductionConfig) -> list[str]:
    artifacts = ["missing_row_filter", "column_quality_filter", "intra_block_correlation_filter"]
    if config.cross_block_diagnostics.enabled:
        artifacts.append("cross_block_diagnostics")
    if config.cross_block_filtering.enabled:
        artifacts.append("cross_block_filtering")
    return artifacts


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _subset_fit_dataframe_for_candidate_features(
        train_df: pd.DataFrame,
        cfg: ocfr.FeatureReductionConfig,
        candidate_features: Sequence[str],
    ) -> pd.DataFrame:
    candidates = _unique_preserve_order([str(feature) for feature in candidate_features])
    if not candidates:
        raise ValueError("candidate_features must not be empty for train-only feature reduction.")
    missing = [feature for feature in candidates if feature not in train_df.columns]
    if missing:
        raise ValueError(f"Training dataframe is missing policy candidate features: {missing}")

    blocks = ocfr.split_descriptor_blocks(
        columns=train_df.columns,
        metadata_columns=cfg.block_detection.metadata_columns,
        target_columns=cfg.block_detection.target_columns,
        receptor_patterns=cfg.block_detection.receptor_patterns,
        ligand_patterns=cfg.block_detection.ligand_patterns,
        scoring_patterns=cfg.block_detection.scoring_patterns,
        use_ligand_class_descriptors=cfg.block_detection.use_ligand_class_descriptors,
        use_receptor_class_descriptors=cfg.block_detection.use_receptor_class_descriptors,
        use_scoring_model_descriptors=cfg.block_detection.use_scoring_model_descriptors,
    )
    preserved_columns = _unique_preserve_order([*blocks.metadata, *blocks.target, *candidates])
    return train_df.loc[:, preserved_columns].copy()


## Public ##


def feature_reduction_config_for_feature_blocks(
        target_column: str = "experimental",
        feature_blocks: Optional[Sequence[str]] = None,
    ) -> ocfr.FeatureReductionConfig:
    '''Return the OCScore train-only feature-reduction config for selected descriptor blocks.

    ``feature_blocks=None`` keeps the full protocol. Otherwise only the requested
    descriptor families are detectable and eligible for feature selection.
    '''

    cfg = ocfr.default_ocscore_feature_reduction_config(target_column=target_column)
    if feature_blocks is None:
        return cfg

    requested = {str(block).strip().lower().replace("-", "_") for block in feature_blocks}
    unknown = sorted(requested.difference(FEATURE_BLOCK_NAMES))
    if unknown:
        valid = ", ".join(FEATURE_BLOCK_NAMES)
        raise ValueError(f"Unknown feature block(s) for ablation: {unknown}. Expected: {valid}.")
    if not requested:
        raise ValueError("feature_blocks must contain at least one descriptor block.")

    if "ligand" not in requested:
        cfg.block_detection.ligand_patterns = []
        cfg.block_detection.use_ligand_class_descriptors = False
    if "receptor" not in requested:
        cfg.block_detection.receptor_patterns = []
        cfg.block_detection.use_receptor_class_descriptors = False
    if "scoring" not in requested:
        cfg.block_detection.scoring_patterns = []
        cfg.block_detection.use_scoring_model_descriptors = False
    return cfg


def fit_train_only_feature_reduction(
        train_df: pd.DataFrame,
        *,
        config: Optional[ocfr.FeatureReductionConfig] = None,
        target_column: str = "experimental",
        fit_split: str = "train",
        fit_dataset: str = "pdbbind_train",
        feature_blocks: Optional[Sequence[str]] = None,
        candidate_features: Optional[Sequence[str]] = None,
    ) -> TrainOnlyReductionArtifact:
    '''Fit feature reduction on training rows only and freeze selected features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Wide PDBbind training partition only.
    config : FeatureReductionConfig | None, optional
        Feature-reduction configuration.
    target_column : str, optional
        Regression target column preserved outside descriptor blocks.
    fit_split : str, optional
        Split label recorded in metadata.
    fit_dataset : str, optional
        Dataset label recorded in metadata.
    feature_blocks : Sequence[str] | None, optional
        Descriptor blocks eligible for selection. ``None`` keeps ligand,
        receptor, and scoring-function descriptors.
    candidate_features : Sequence[str] | None, optional
        Ordered policy-constrained candidate descriptors exposed to train-only
        feature reduction. Metadata and target columns are preserved separately.

    Returns
    -------
    TrainOnlyReductionArtifact
        Frozen selected/removed features and metadata.
    '''

    cfg = config or feature_reduction_config_for_feature_blocks(
        target_column=target_column,
        feature_blocks=feature_blocks,
    )
    fit_df = (
        _subset_fit_dataframe_for_candidate_features(train_df, cfg, candidate_features)
        if candidate_features is not None
        else train_df.copy()
    )
    result = ocfr.run_feature_reduction_protocol(df=fit_df, config=cfg, write_outputs=False)
    removed = _collect_removed_features(fit_df, result.selected_features, result.blocks)
    transform_artifacts = _default_transform_artifacts(cfg)
    transform_hashes = {
        name: hash_json_dict({"step": name, "config": cfg.block_detection.__class__.__name__})
        for name in transform_artifacts
    }
    fit_row_indices = [int(index) for index in train_df.index.tolist()]
    feature_selection = FeatureSelectionScope.train_only(
        fit_dataset=fit_dataset,
        fit_split=fit_split,
        fit_row_count=int(len(train_df)),
        fit_row_indices=fit_row_indices,
        selected_features=result.selected_features,
        removed_features=removed,
        transform_artifacts=transform_artifacts,
    )
    feature_selection.fit_row_content_hash = hash_dataframe_partition(
        fit_df.reset_index(drop=True),
        list(range(len(fit_df))),
    )
    feature_selection.transform_artifact_hashes = transform_hashes
    protocol = dict(result.protocol)
    if feature_blocks is not None:
        protocol["feature_blocks"] = list(feature_blocks)
    if candidate_features is not None:
        protocol["candidate_feature_count_before_reduction"] = len(candidate_features)
        protocol["candidate_features_before_reduction"] = list(candidate_features)
    return TrainOnlyReductionArtifact(
        selected_features=list(result.selected_features),
        removed_features=removed,
        metadata_columns=list(result.blocks.metadata),
        target_columns=list(result.blocks.target),
        feature_selection=feature_selection,
        protocol=protocol,
        transform_artifacts=transform_artifacts,
        transform_artifact_hashes=transform_hashes,
    )


def drop_nonfinite_selected_feature_rows(
        df: pd.DataFrame,
        selected_features: Sequence[str],
        *,
        label: str = "dataset",
        id_columns: Optional[Sequence[str]] = None,
        reset_index: bool = True,
    ) -> SelectedFeatureRowCleanupResult:
    '''Drop rows with NaN, +inf, or -inf values in selected features.

    Selected feature values are coerced to numeric first, so blank strings and
    non-numeric cells are treated as invalid and reported as dropped rows.
    '''

    features = list(selected_features)
    if not features:
        raise ValueError("selected_features must not be empty.")
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"{label} dataframe is missing selected features: {missing}")

    cleaned = df.copy()
    numeric_features = cleaned.loc[:, features].apply(pd.to_numeric, errors="coerce")
    feature_values = numeric_features.to_numpy(dtype=float, copy=False)
    invalid_mask = ~np.isfinite(feature_values)
    rows_to_drop = invalid_mask.any(axis=1)
    kept_mask = ~rows_to_drop

    for feature in features:
        cleaned[feature] = numeric_features[feature]

    present_id_columns = [
        column
        for column in (list(id_columns) if id_columns is not None else list(ocfr.DEFAULT_ID_COLUMNS))
        if column in df.columns
    ]
    dropped_records: list[dict[str, Any]] = []
    for position in np.flatnonzero(rows_to_drop):
        invalid_columns = [
            features[column_index]
            for column_index in np.flatnonzero(invalid_mask[position])
        ]
        record: dict[str, Any] = {
            "original_index": df.index[position],
            "original_position": int(position),
            "n_invalid_selected_features": int(len(invalid_columns)),
            "invalid_selected_features": ",".join(invalid_columns),
            "drop_reason": "nonfinite_selected_feature_values",
        }
        for id_column in present_id_columns:
            record[id_column] = df.iloc[position][id_column]
        dropped_records.append(record)

    report_columns = [
        "original_index",
        "original_position",
        "n_invalid_selected_features",
        "invalid_selected_features",
        "drop_reason",
        *present_id_columns,
    ]
    dropped_rows = pd.DataFrame(dropped_records, columns=report_columns)
    cleaned = cleaned.loc[kept_mask].copy()
    if reset_index:
        cleaned = cleaned.reset_index(drop=True)

    n_before = int(len(df))
    n_after = int(len(cleaned))
    n_dropped = int(n_before - n_after)
    summary = {
        "label": label,
        "n_rows_before": n_before,
        "n_rows_after": n_after,
        "n_rows_dropped": n_dropped,
        "fraction_rows_dropped": (n_dropped / n_before) if n_before else 0.0,
        "columns_checked": features,
        "id_columns": present_id_columns,
    }
    if n_before and n_after == 0:
        raise ValueError(f"No {label} rows remain after dropping non-finite selected feature values.")
    if n_dropped:
        LOGGER.info(
            "Dropped %s %s row(s) with non-finite selected feature values from %s rows.",
            n_dropped,
            label,
            n_before,
        )

    return SelectedFeatureRowCleanupResult(
        cleaned_df=cleaned,
        dropped_rows=dropped_rows,
        kept_mask=kept_mask,
        summary=summary,
    )


def apply_frozen_feature_selection(
        df: pd.DataFrame,
        artifact: TrainOnlyReductionArtifact,
    ) -> pd.DataFrame:
    '''Apply frozen selected features to a wide dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Wide input dataframe.
    artifact : TrainOnlyReductionArtifact
        Frozen train-only reduction artifact.

    Returns
    -------
    pd.DataFrame
        Reduced dataframe containing metadata, targets, and selected features.

    Raises
    ------
    ValueError
        If required selected features are missing.
    '''

    missing = [column for column in artifact.selected_features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required selected features for frozen application: {missing}")
    return ocfr.build_reduced_dataframe(
        df,
        metadata_columns=artifact.metadata_columns,
        target_columns=artifact.target_columns,
        selected_features=artifact.selected_features,
    )


def write_train_only_reduction_artifact(output_dir: str | Path, artifact: TrainOnlyReductionArtifact) -> dict[str, str]:
    '''Persist train-only reduction artifacts to ``output_dir``.'''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if artifact.feature_selection.fit_row_indices is not None:
        fit_rows_path = write_feature_selection_fit_rows_json(out, artifact.feature_selection)
        paths["feature_selection_fit_rows_json"] = str(fit_rows_path)
    artifact_path = out / TRAIN_ONLY_ARTIFACT_JSON
    artifact_path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["train_only_feature_reduction"] = str(artifact_path)
    selected_path = out / "selected_features.json"
    selected_path.write_text(json.dumps(artifact.selected_features, indent=2) + "\n", encoding="utf-8")
    paths["selected_features_json"] = str(selected_path)
    removed_path = out / "removed_features.json"
    removed_path.write_text(json.dumps(artifact.removed_features, indent=2) + "\n", encoding="utf-8")
    paths["removed_features_json"] = str(removed_path)
    paths["feature_selection_json"] = str(write_feature_selection_json(out, artifact.feature_selection))
    return paths


def load_train_only_reduction_artifact(source: str | Path) -> TrainOnlyReductionArtifact:
    '''Load a train-only reduction artifact from a directory or JSON file.'''

    path = Path(source)
    if path.is_dir():
        path = path / TRAIN_ONLY_ARTIFACT_JSON
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_selection = FeatureSelectionScope.from_dict(payload["feature_selection"])
    return TrainOnlyReductionArtifact(
        selected_features=list(payload["selected_features"]),
        removed_features=list(payload.get("removed_features") or []),
        metadata_columns=list(payload.get("metadata_columns") or []),
        target_columns=list(payload.get("target_columns") or []),
        feature_selection=feature_selection,
        protocol=dict(payload.get("protocol") or {}),
        transform_artifacts=list(payload.get("transform_artifacts") or []),
        transform_artifact_hashes=dict(payload.get("transform_artifact_hashes") or {}),
    )


def split_wide_dataset_by_column(
        merged_df: pd.DataFrame,
        *,
        dataset_column: str = DATASET_COLUMN,
        pdbbind_values: Optional[set[str]] = None,
        dudez_values: Optional[set[str]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Split a merged wide dataframe into PDBbind and DUDEz partitions.'''

    if dataset_column not in merged_df.columns:
        raise ValueError(f"Merged wide dataframe is missing dataset column {dataset_column!r}.")
    normalized = merged_df[dataset_column].astype(str).str.strip().str.lower()
    pdb_values = pdbbind_values or {"pdbbind", "pdbbind_refined", "pdbbind_general"}
    dude_values = dudez_values or {"dudez", "dude-z", "dude_z"}
    pdbbind_df = merged_df[normalized.isin(pdb_values)].copy()
    dudez_df = merged_df[normalized.isin(dude_values)].copy()
    if pdbbind_df.empty or dudez_df.empty:
        raise ValueError("Merged wide dataframe must contain both PDBbind and DUDEz rows.")
    return pdbbind_df, dudez_df


__all__ = [
    "FEATURE_BLOCK_NAMES",
    "TRAIN_ONLY_ARTIFACT_JSON",
    "SelectedFeatureRowCleanupResult",
    "TrainOnlyReductionArtifact",
    "apply_frozen_feature_selection",
    "drop_nonfinite_selected_feature_rows",
    "feature_reduction_config_for_feature_blocks",
    "fit_train_only_feature_reduction",
    "load_train_only_reduction_artifact",
    "split_wide_dataset_by_column",
    "write_train_only_reduction_artifact",
]
