#!/usr/bin/env python3

# Description
###############################################################################
'''
Fixed outer train/validation/test splits for the unified OCScore modeling protocol.

One outer split is created before replicas; feature reduction is fit only on the
outer-train partition; all replicas reuse the same split assignments.
'''

from __future__ import annotations

# Imports
###############################################################################
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from OCDocker.OCScore.Utils.ContentHash import hash_dataframe_partition
from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.ContentHash import hash_json_dict
from OCDocker.OCScore.Utils.ContentHash import hash_split_indices

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

FIXED_OUTER_SPLIT_JSON = "fixed_outer_split.json"
ROW_ID_COLUMNS = ("name", "receptor", "ligand")

# Classes
###############################################################################


@dataclass
class FixedOuterSplitAssignment:
    """Frozen outer split shared by all replicas."""

    outer_split_id: str
    outer_split_seed: int
    pdbbind_train_indices: list[int]
    pdbbind_validation_indices: list[int]
    pdbbind_test_indices: list[int]
    dudez_train_indices: list[int]
    dudez_validation_indices: list[int]
    dudez_test_indices: list[int]
    pdbbind_train_indices_hash: str
    pdbbind_validation_indices_hash: str
    pdbbind_test_indices_hash: str
    dudez_train_indices_hash: str
    dudez_validation_indices_hash: str
    dudez_test_indices_hash: str
    outer_split_assignment_hash: str
    feature_selection_scope: str = "train_only"
    feature_selection_fit_split: str = "train"
    feature_selection_fit_row_count: int = 0
    feature_selection_split_hash: str = ""
    selected_features_hash: str = ""
    removed_features_hash: str = ""
    feature_reduction_artifact_path: Optional[str] = None
    replica_uses_fixed_outer_split: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pdbbind_train_row_count"] = len(self.pdbbind_train_indices)
        payload["pdbbind_validation_row_count"] = len(self.pdbbind_validation_indices)
        payload["pdbbind_test_row_count"] = len(self.pdbbind_test_indices)
        payload["dudez_train_row_count"] = len(self.dudez_train_indices)
        payload["dudez_validation_row_count"] = len(self.dudez_validation_indices)
        payload["dudez_test_row_count"] = len(self.dudez_test_indices)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedOuterSplitAssignment":
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {key: value for key, value in payload.items() if key in known}
        return cls(**filtered)


# Functions
###############################################################################
## Private ##


def _as_index_list(indices: Sequence[int] | np.ndarray) -> list[int]:
    return [int(value) for value in np.asarray(indices, dtype=int).tolist()]


def _assignment_hash(payload: dict[str, Any]) -> str:
    return hash_json_dict(
        {
            "outer_split_seed": payload["outer_split_seed"],
            "pdbbind_train_indices_hash": payload["pdbbind_train_indices_hash"],
            "pdbbind_validation_indices_hash": payload["pdbbind_validation_indices_hash"],
            "pdbbind_test_indices_hash": payload["pdbbind_test_indices_hash"],
            "dudez_train_indices_hash": payload["dudez_train_indices_hash"],
            "dudez_validation_indices_hash": payload["dudez_validation_indices_hash"],
            "dudez_test_indices_hash": payload["dudez_test_indices_hash"],
        }
    )


## Public ##


def build_fixed_outer_split_assignment(
        *,
        outer_split_seed: int,
        pdbbind_train_indices: Sequence[int],
        pdbbind_validation_indices: Sequence[int],
        pdbbind_test_indices: Sequence[int],
        dudez_train_indices: Sequence[int],
        dudez_validation_indices: Sequence[int],
        dudez_test_indices: Sequence[int],
        feature_selection_fit_row_count: int,
        selected_features: Sequence[str],
        removed_features: Sequence[str],
        feature_reduction_artifact_path: Optional[str] = None,
        outer_split_id: Optional[str] = None,
    ) -> FixedOuterSplitAssignment:
    '''Build a fixed outer split record with deterministic hashes.'''

    pdb_train = _as_index_list(pdbbind_train_indices)
    pdb_val = _as_index_list(pdbbind_validation_indices)
    pdb_test = _as_index_list(pdbbind_test_indices)
    dude_train = _as_index_list(dudez_train_indices)
    dude_val = _as_index_list(dudez_validation_indices)
    dude_test = _as_index_list(dudez_test_indices)
    payload = {
        "outer_split_seed": int(outer_split_seed),
        "pdbbind_train_indices_hash": hash_split_indices(pdb_train),
        "pdbbind_validation_indices_hash": hash_split_indices(pdb_val),
        "pdbbind_test_indices_hash": hash_split_indices(pdb_test),
        "dudez_train_indices_hash": hash_split_indices(dude_train),
        "dudez_validation_indices_hash": hash_split_indices(dude_val),
        "dudez_test_indices_hash": hash_split_indices(dude_test),
    }
    return FixedOuterSplitAssignment(
        outer_split_id=outer_split_id or str(uuid.uuid4()),
        outer_split_seed=int(outer_split_seed),
        pdbbind_train_indices=pdb_train,
        pdbbind_validation_indices=pdb_val,
        pdbbind_test_indices=pdb_test,
        dudez_train_indices=dude_train,
        dudez_validation_indices=dude_val,
        dudez_test_indices=dude_test,
        pdbbind_train_indices_hash=payload["pdbbind_train_indices_hash"],
        pdbbind_validation_indices_hash=payload["pdbbind_validation_indices_hash"],
        pdbbind_test_indices_hash=payload["pdbbind_test_indices_hash"],
        dudez_train_indices_hash=payload["dudez_train_indices_hash"],
        dudez_validation_indices_hash=payload["dudez_validation_indices_hash"],
        dudez_test_indices_hash=payload["dudez_test_indices_hash"],
        outer_split_assignment_hash=_assignment_hash(payload),
        feature_selection_fit_row_count=int(feature_selection_fit_row_count),
        feature_selection_split_hash=hash_split_indices(pdb_train),
        selected_features_hash=hash_feature_list(selected_features),
        removed_features_hash=hash_feature_list(removed_features),
        feature_reduction_artifact_path=feature_reduction_artifact_path,
    )


def write_fixed_outer_split_json(output_dir: str | Path, assignment: FixedOuterSplitAssignment) -> Path:
    '''Write ``fixed_outer_split.json`` to ``output_dir``.'''

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FIXED_OUTER_SPLIT_JSON
    path.write_text(json.dumps(assignment.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_fixed_outer_split(source: str | Path) -> FixedOuterSplitAssignment:
    '''Load a fixed outer split assignment from a directory or JSON file.'''

    path = Path(source)
    if path.is_dir():
        path = path / FIXED_OUTER_SPLIT_JSON
    if not path.is_file():
        raise FileNotFoundError(f"Fixed outer split metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FixedOuterSplitAssignment.from_dict(payload)


def build_replica_split_alignment_metadata(
        fixed: FixedOuterSplitAssignment,
        *,
        replica_seed: int,
        replica_name: str,
        pdbbind_split_indices: dict[str, list[int]],
        selected_features: Sequence[str],
        dudez_split_indices: Optional[dict[str, list[int]]] = None,
    ) -> dict[str, Any]:
    '''Build per-replica split-alignment metadata for protocol logs.'''

    pdb_train_hash = hash_split_indices(pdbbind_split_indices.get("train", []))
    pdb_val_hash = hash_split_indices(pdbbind_split_indices.get("validation", []))
    pdb_test_hash = hash_split_indices(pdbbind_split_indices.get("test", []))
    selected_hash = hash_feature_list(selected_features)
    metadata = {
        "replica_name": replica_name,
        "replica_seed": int(replica_seed),
        "outer_split_id": fixed.outer_split_id,
        "outer_split_seed": fixed.outer_split_seed,
        "feature_selection_scope": fixed.feature_selection_scope,
        "feature_selection_fit_split": fixed.feature_selection_fit_split,
        "feature_selection_fit_row_count": fixed.feature_selection_fit_row_count,
        "feature_selection_split_hash": fixed.feature_selection_split_hash,
        "pdbbind_train_indices_hash": pdb_train_hash,
        "pdbbind_validation_indices_hash": pdb_val_hash,
        "pdbbind_test_indices_hash": pdb_test_hash,
        "selected_features_hash": selected_hash,
        "removed_features_hash": fixed.removed_features_hash,
        "replica_uses_fixed_outer_split": True,
        "pdbbind_split_matches_fixed_outer_split": (
            pdb_train_hash == fixed.pdbbind_train_indices_hash
            and pdb_val_hash == fixed.pdbbind_validation_indices_hash
            and pdb_test_hash == fixed.pdbbind_test_indices_hash
        ),
        "selected_features_match_fixed": selected_hash == fixed.selected_features_hash,
    }
    if dudez_split_indices is not None:
        metadata["dudez_train_indices_hash"] = hash_split_indices(dudez_split_indices.get("train", []))
        metadata["dudez_validation_indices_hash"] = hash_split_indices(dudez_split_indices.get("validation", []))
        metadata["dudez_test_indices_hash"] = hash_split_indices(dudez_split_indices.get("test", []))
        metadata["dudez_split_matches_fixed_outer_split"] = (
            metadata["dudez_train_indices_hash"] == fixed.dudez_train_indices_hash
            and metadata["dudez_validation_indices_hash"] == fixed.dudez_validation_indices_hash
            and metadata["dudez_test_indices_hash"] == fixed.dudez_test_indices_hash
        )
    return metadata


def validate_replica_split_alignment(
        fixed: FixedOuterSplitAssignment,
        *,
        replica_name: str,
        pdbbind_split_indices: dict[str, list[int]],
        selected_features: Sequence[str],
        dudez_split_indices: Optional[dict[str, list[int]]] = None,
        strict: bool = True,
    ) -> None:
    '''Validate that a replica reused the fixed outer split and selected features.'''

    alignment = build_replica_split_alignment_metadata(
        fixed,
        replica_seed=-1,
        replica_name=replica_name,
        pdbbind_split_indices=pdbbind_split_indices,
        selected_features=selected_features,
        dudez_split_indices=dudez_split_indices,
    )
    errors: list[str] = []
    if not alignment["pdbbind_split_matches_fixed_outer_split"]:
        errors.append(
            f"Replica {replica_name!r} PDBbind split hashes differ from fixed outer split."
        )
    if not alignment["selected_features_match_fixed"]:
        errors.append(
            f"Replica {replica_name!r} selected_features_hash differs from fixed outer split."
        )
    if dudez_split_indices is not None and not alignment.get("dudez_split_matches_fixed_outer_split", False):
        errors.append(
            f"Replica {replica_name!r} DUDEz split hashes differ from fixed outer split."
        )
    if errors and strict:
        raise ValueError(" ".join(errors))


def validate_protocol_integrity(
        *,
        feature_selection: Optional[dict[str, Any]],
        fixed_outer_split: Optional[dict[str, Any] | FixedOuterSplitAssignment],
        replica_alignments: Sequence[dict[str, Any]],
        raw_input_hashes: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
    '''Validate unified OCScore training protocol metadata and fail on invalid runs.

    Parameters
    ----------
    feature_selection : dict[str, Any] | None
        Train-only feature-selection metadata from the modeling run.
    fixed_outer_split : dict[str, Any] | FixedOuterSplitAssignment | None
        Fixed outer split metadata shared by all replicas.
    replica_alignments : Sequence[dict[str, Any]]
        Per-replica split and selected-feature alignment records.
    raw_input_hashes : dict[str, Any] | None, optional
        Content hashes for raw modeling inputs.

    Returns
    -------
    dict[str, Any]
        Protocol metadata asserting a valid leakage-safe training run.

    Raises
    ------
    ValueError
        If train-only scope, split hashes, or replica alignment checks fail.
    '''

    invalid_reasons: list[str] = []

    scope = (feature_selection or {}).get("scope") or (feature_selection or {}).get("feature_selection_scope")
    if scope != "train_only":
        invalid_reasons.append("missing_train_only_feature_selection")
    fit_split = (feature_selection or {}).get("fit_split") or (feature_selection or {}).get("feature_selection_fit_split")
    if fit_split not in (None, "train", "pdbbind_train"):
        invalid_reasons.append("invalid_feature_selection_fit_split")

    fixed_dict = fixed_outer_split.to_dict() if isinstance(fixed_outer_split, FixedOuterSplitAssignment) else dict(fixed_outer_split or {})
    required_hashes = (
        "pdbbind_train_indices_hash",
        "pdbbind_validation_indices_hash",
        "pdbbind_test_indices_hash",
        "selected_features_hash",
    )
    if not fixed_dict or any(not fixed_dict.get(key) for key in required_hashes):
        invalid_reasons.append("missing_split_hashes")
    if not fixed_dict.get("fixed_outer_split", True):
        invalid_reasons.append("fixed_outer_split_not_set")

    for alignment in replica_alignments:
        if not alignment.get("pdbbind_split_matches_fixed_outer_split", False):
            invalid_reasons.append("replica_split_mismatch")
            break
        if not alignment.get("selected_features_match_fixed", False):
            invalid_reasons.append("replica_selected_features_mismatch")
            break

    unique_reasons = sorted(set(invalid_reasons))
    if unique_reasons:
        raise ValueError(
            "OCScore training protocol validation failed: " + ", ".join(unique_reasons)
        )

    return {
        "protocol_valid": True,
        "feature_selection_scope": "train_only",
        "feature_selection_fit_split": fit_split or "train",
        "fixed_outer_split": True,
        "global_feature_reduction_used": False,
        "precomputed_features_used_for_training": False,
        "raw_input_hashes": raw_input_hashes or {},
    }


def evaluate_production_claim_validity(
        *,
        perform_hard_checks: bool,
        feature_selection: Optional[dict[str, Any]],
        fixed_outer_split: Optional[dict[str, Any] | FixedOuterSplitAssignment],
        replica_alignments: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
    '''Deprecated alias that always validates the unified protocol.

    Parameters
    ----------
    perform_hard_checks : bool
        Ignored. Retained for backward-compatible call sites.
    feature_selection : dict[str, Any] | None
        Train-only feature-selection metadata from the modeling run.
    fixed_outer_split : dict[str, Any] | FixedOuterSplitAssignment | None
        Fixed outer split metadata shared by all replicas.
    replica_alignments : Sequence[dict[str, Any]]
        Per-replica split and selected-feature alignment records.

    Returns
    -------
    dict[str, Any]
        Protocol metadata from :func:`validate_protocol_integrity`.
    '''

    del perform_hard_checks
    return validate_protocol_integrity(
        feature_selection=feature_selection,
        fixed_outer_split=fixed_outer_split,
        replica_alignments=replica_alignments,
    )


__all__ = [
    "FIXED_OUTER_SPLIT_JSON",
    "FixedOuterSplitAssignment",
    "ROW_ID_COLUMNS",
    "build_fixed_outer_split_assignment",
    "build_replica_split_alignment_metadata",
    "evaluate_production_claim_validity",
    "validate_protocol_integrity",
    "load_fixed_outer_split",
    "validate_replica_split_alignment",
    "write_fixed_outer_split_json",
]
