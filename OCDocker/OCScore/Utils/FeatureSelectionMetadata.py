#!/usr/bin/env python3

# Description
###############################################################################
'''
Feature-selection scope metadata for OCScore staged protocols.

Records whether selected features were derived globally during ``reduce`` or fit
on training data only during ``train``, so staged runs fail closed on leakage.
'''

# Imports
###############################################################################
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.ContentHash import hash_split_indices

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

LOGGER = oclogging.get_logger("ocscore.utils.feature_selection_metadata")

FEATURE_SELECTION_JSON = "feature_selection.json"
FEATURE_SELECTION_FIT_ROWS_JSON = "feature_selection_fit_rows.json"
FeatureSelectionScopeName = Literal["precomputed_global", "train_only"]
FeatureSelectionSourceName = Literal["externally_supplied", "reduction_protocol", "train_derived"]
FeatureSelectionModeName = Literal["production-strict", "external-blind"]


@dataclass
class FeatureSelectionScope:
    """Provenance for how model input features were chosen.

    Parameters
    ----------
    scope : str
        ``precomputed_global`` when the reduce CLI used all rows before modeling splits;
        ``train_only`` when fit on training rows only during staged train.
    fit_dataset : str
        Dataset partition used to derive features (e.g. ``pdbbind_train``).
    fit_split : str | None
        Split name when scope is train-only (e.g. ``train``).
    selected_features_source : str
        ``reduction_protocol``, ``externally_supplied``, or ``train_derived``.
    uses_supervised_target : bool, optional
        True when target-aware selection was used, by default False.
    reduction_archive : str | None, optional
        Path to reduction archive when externally supplied.
    n_selected_features : int | None, optional
        Count of selected features for quick audit.
    fit_row_indices : list[int] | None, optional
        Row indices when scope is train-only. Kept in memory for optional
        artifact writing, but omitted from summary dictionaries by default.
    fit_row_count : int | None, optional
        Number of rows used to fit feature selection.
    fit_row_indices_hash : str | None, optional
        SHA-256 hash of the ordered fit-row index list.
    fit_row_content_hash : str | None, optional
        SHA-256 hash of fit-row identifiers/content used for audit.
    fit_row_indices_artifact : str | None, optional
        Path to the full fit-row index artifact when written.
    feature_selection_mode : str | None, optional
        Validation mode active when metadata was recorded.
    selected_features : list[str] | None, optional
        Ordered selected feature list for reproducibility checks.
    selected_features_hash : str | None, optional
        SHA-256 hash of ``selected_features``.
    removed_features : list[str] | None, optional
        Features removed during reduction relative to the fit input.
    removed_features_hash : str | None, optional
        SHA-256 hash of ``removed_features``.
    transform_artifacts : list[str], optional
        Names of frozen transform/selection steps (for example correlation filters).
    transform_artifact_hashes : dict[str, str], optional
        Optional hashes keyed by transform artifact name.
    notes : list[str], optional
        Human-readable warnings or context.
    """

    scope: FeatureSelectionScopeName
    fit_dataset: str
    fit_split: Optional[str] = None
    selected_features_source: FeatureSelectionSourceName = "reduction_protocol"
    uses_supervised_target: bool = False
    reduction_archive: Optional[str] = None
    n_selected_features: Optional[int] = None
    fit_row_indices: Optional[list[int]] = None
    fit_row_count: Optional[int] = None
    fit_row_indices_hash: Optional[str] = None
    fit_row_content_hash: Optional[str] = None
    fit_row_indices_artifact: Optional[str] = None
    feature_selection_mode: Optional[FeatureSelectionModeName] = None
    selected_features: Optional[list[str]] = None
    selected_features_hash: Optional[str] = None
    removed_features: Optional[list[str]] = None
    removed_features_hash: Optional[str] = None
    transform_artifacts: list[str] = field(default_factory=list)
    transform_artifact_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, include_fit_row_indices: bool = False) -> dict[str, Any]:
        '''Return a JSON-serializable dictionary.

        The full ``fit_row_indices`` list is intentionally omitted by default so
        summary/provenance reports stay readable. Write it to
        ``feature_selection_fit_rows.json`` when the full audit trail is needed.
        '''

        payload = asdict(self)
        if not include_fit_row_indices:
            payload.pop("fit_row_indices", None)
        payload["schema_version"] = 2
        payload["feature_selection_scope"] = self.scope
        payload["feature_selection_fit_split"] = self.fit_split
        payload["feature_selection_fit_row_count"] = self.fit_row_count
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSelectionScope":
        '''Build scope from a JSON-compatible mapping.'''

        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {key: value for key, value in payload.items() if key in known}
        if "scope" not in filtered and "feature_selection_scope" in payload:
            filtered["scope"] = payload["feature_selection_scope"]
        if "fit_split" not in filtered and "feature_selection_fit_split" in payload:
            filtered["fit_split"] = payload["feature_selection_fit_split"]
        if "fit_row_count" not in filtered and "feature_selection_fit_row_count" in payload:
            filtered["fit_row_count"] = payload["feature_selection_fit_row_count"]
        return cls(**filtered)

    @classmethod
    def precomputed_global(
            cls,
            *,
            fit_dataset: str = "merged_pdbbind_dudez",
            selected_features_source: FeatureSelectionSourceName = "reduction_protocol",
            reduction_archive: Optional[str] = None,
            n_selected_features: Optional[int] = None,
            notes: Optional[Sequence[str]] = None,
        ) -> "FeatureSelectionScope":
        '''Scope recorded by the reduce CLI only; not valid for staged train.'''

        default_notes = [
            "Unsupervised feature reduction ran on the merged PDBbind+DUDEz dataframe "
            "before train/validation/test splits. Staged train refits train-only selection "
            "from merged_input_dataset.csv and must not reuse this scope.",
        ]
        merged_notes = list(default_notes)
        if notes:
            merged_notes.extend(notes)
        return cls(
            scope="precomputed_global",
            fit_dataset=fit_dataset,
            fit_split=None,
            selected_features_source=selected_features_source,
            uses_supervised_target=False,
            reduction_archive=reduction_archive,
            n_selected_features=n_selected_features,
            notes=merged_notes,
        )

    @classmethod
    def train_only(
            cls,
            *,
            fit_dataset: str = "pdbbind_train",
            fit_split: str = "train",
            fit_row_count: int,
            fit_row_indices: Optional[list[int]] = None,
            selected_features: Sequence[str],
            removed_features: Optional[Sequence[str]] = None,
            transform_artifacts: Optional[Sequence[str]] = None,
            feature_selection_mode: FeatureSelectionModeName = "production-strict",
            notes: Optional[Sequence[str]] = None,
        ) -> "FeatureSelectionScope":
        '''Scope for train-only feature reduction after an outer split.'''

        selected = list(selected_features)
        removed = list(removed_features or [])
        scope = cls(
            scope="train_only",
            fit_dataset=fit_dataset,
            fit_split=fit_split,
            selected_features_source="train_derived",
            uses_supervised_target=False,
            n_selected_features=len(selected),
            fit_row_indices=list(fit_row_indices) if fit_row_indices is not None else None,
            fit_row_count=int(fit_row_count),
            fit_row_indices_hash=(
                hash_split_indices(fit_row_indices) if fit_row_indices is not None else None
            ),
            feature_selection_mode=feature_selection_mode,
            selected_features=selected,
            selected_features_hash=hash_feature_list(selected),
            removed_features=removed,
            removed_features_hash=hash_feature_list(removed) if removed else hash_feature_list([]),
            transform_artifacts=list(transform_artifacts or []),
            notes=list(notes or []),
        )
        return scope


def attach_feature_hashes(
        scope: FeatureSelectionScope,
        *,
        selected_features: Optional[Sequence[str]] = None,
        removed_features: Optional[Sequence[str]] = None,
    ) -> FeatureSelectionScope:
    '''Return a copy of ``scope`` with feature-list hashes populated.'''

    selected = list(selected_features if selected_features is not None else (scope.selected_features or []))
    removed = list(removed_features if removed_features is not None else (scope.removed_features or []))
    scope.selected_features = selected or scope.selected_features
    scope.removed_features = removed or scope.removed_features
    if selected:
        scope.selected_features_hash = hash_feature_list(selected)
        scope.n_selected_features = len(selected)
    if removed is not None:
        scope.removed_features_hash = hash_feature_list(removed)
    return scope


def write_feature_selection_json(
        output_dir: str | Path,
        scope: FeatureSelectionScope,
        filename: str = FEATURE_SELECTION_JSON,
    ) -> Path:
    '''Write ``feature_selection.json`` to ``output_dir``.'''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(scope.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_feature_selection_fit_rows_json(
        output_dir: str | Path,
        scope: FeatureSelectionScope,
        filename: str = FEATURE_SELECTION_FIT_ROWS_JSON,
    ) -> Path:
    '''Write full train-only fit-row indices as a separate audit artifact.'''

    if scope.fit_row_indices is None:
        raise ValueError("fit_row_indices are not available for artifact writing.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    payload = {
        "schema_version": 1,
        "fit_dataset": scope.fit_dataset,
        "fit_split": scope.fit_split,
        "fit_row_count": scope.fit_row_count,
        "fit_row_indices": [int(index) for index in scope.fit_row_indices],
        "fit_row_indices_hash": scope.fit_row_indices_hash,
        "fit_row_content_hash": scope.fit_row_content_hash,
        "selected_features_hash": scope.selected_features_hash,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scope.fit_row_indices_artifact = str(path)
    return path


def load_feature_selection_json(source: str | Path) -> FeatureSelectionScope:
    '''Load feature-selection scope from a directory or JSON file.'''

    path = Path(source)
    if path.is_dir():
        path = path / FEATURE_SELECTION_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Feature selection metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    scope_name = payload.get("scope") or payload.get("feature_selection_scope")
    if scope_name not in ("precomputed_global", "train_only"):
        raise ValueError(f"Invalid feature selection scope {scope_name!r} in {path}.")
    return FeatureSelectionScope.from_dict(payload)


def validate_train_only_feature_selection(scope: FeatureSelectionScope) -> None:
    '''Validate that feature selection metadata is train-only and reproducible.'''

    if scope.scope != "train_only":
        raise ValueError(
            "Staged train requires train-only feature selection. "
            f"Current feature_selection_scope={scope.scope!r}, fit_dataset={scope.fit_dataset!r}. "
            "Provide merged_input_dataset.csv in the reduction archive or a saved train-only artifact."
        )
    if scope.fit_split not in (None, "train", "pdbbind_train"):
        raise ValueError(
            f"Train-only feature selection requires feature_selection_fit_split='train' "
            f"(got {scope.fit_split!r})."
        )
    if scope.selected_features_hash is None and not scope.selected_features:
        raise ValueError(
            "Train-only feature selection metadata must include selected_features "
            "or selected_features_hash."
        )


def verify_selected_features_against_scope(
        selected_features: Sequence[str],
        scope: FeatureSelectionScope,
    ) -> None:
    '''Verify that ``selected_features`` matches saved scope metadata.'''

    if scope.selected_features is not None and list(selected_features) != list(scope.selected_features):
        raise ValueError(
            "Selected feature list does not match feature_selection.json ordering/content."
        )
    if scope.selected_features_hash is not None:
        actual = hash_feature_list(selected_features)
        if actual != scope.selected_features_hash:
            raise ValueError(
                "Selected feature hash mismatch: expected "
                f"{scope.selected_features_hash}, got {actual}."
            )


__all__ = [
    "FEATURE_SELECTION_FIT_ROWS_JSON",
    "FEATURE_SELECTION_JSON",
    "FeatureSelectionModeName",
    "FeatureSelectionScope",
    "FeatureSelectionScopeName",
    "FeatureSelectionSourceName",
    "attach_feature_hashes",
    "load_feature_selection_json",
    "validate_train_only_feature_selection",
    "verify_selected_features_against_scope",
    "write_feature_selection_fit_rows_json",
    "write_feature_selection_json",
]
