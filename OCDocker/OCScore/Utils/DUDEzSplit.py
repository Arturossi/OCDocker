#!/usr/bin/env python3

# Description
###############################################################################
'''Receptor- and kind-aware train/validation/test splitting for DUDEz screening.

Strategies:

- ``receptor_heldout_complete`` (staged OCScore default): assigns **complete**
  receptor cases (all ligands + all decoys) to train, validation, or test. No
  receptor appears in more than one split. Validation and test are used for
  grouped early-enrichment metrics on unseen receptors.
- ``receptor_stratified_kind``: splits ligands and decoys separately within each
  receptor (historical row-wise stratification per receptor).
- ``receptor_held_out``: fraction-based receptor hold-out (historical).
- ``random_row``: global row shuffle (not recommended for grouped metrics).
'''

# Imports
###############################################################################
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

import OCDocker.Toolbox.Logging as oclogging

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

LOGGER = oclogging.get_logger("ocscore.utils.dudez_split")

DUDEZ_SPLIT_STRATEGIES = (
    "receptor_heldout_complete",
    "receptor_stratified_kind",
    "receptor_held_out",
    "random_row",
)

POSITIVE_KIND_ALIASES = frozenset({"ligands", "ligand", "actives", "active"})
NEGATIVE_KIND_ALIASES = frozenset({"decoys", "decoy"})


@dataclass
class DUDEzSplitConfig:
    """Configuration for DUDEz train/validation/test splitting.

    Parameters
    ----------
    strategy : str, optional
        Split strategy. See :data:`DUDEZ_SPLIT_STRATEGIES`. Default is
        ``receptor_stratified_kind`` for backward compatibility when constructing
        this dataclass directly; staged OCScore uses ``receptor_heldout_complete``.
    receptor_column : str, optional
        Receptor/target column, by default ``"receptor"``.
    kind_column : str, optional
        Ligand/decoy kind column, by default ``"kind"``.
    positive_kind : str, optional
        Canonical positive kind label for logging, by default ``"ligands"``.
    negative_kind : str, optional
        Canonical negative kind label for logging, by default ``"decoys"``.
    train_size : float, optional
        Training fraction for fraction-based strategies, by default ``0.6``.
    validation_size : float, optional
        Validation fraction for fraction-based strategies, by default ``0.2``.
    test_size : float, optional
        Test fraction for fraction-based strategies, by default ``0.2``.
    random_seed : int, optional
        Random seed for deterministic splits, by default ``42``.
    relaxed_split : bool, optional
        When False, raise if validity constraints cannot be met. When True,
        apply documented fallbacks, by default ``False``.
    min_kind_per_split : int, optional
        Minimum samples per kind in a receptor split when possible (stratified
        strategy only), by default ``1``.
    train_receptors : list[str] | None, optional
        Explicit training receptors for ``receptor_heldout_complete``.
    validation_receptors : list[str] | None, optional
        Explicit validation receptors for ``receptor_heldout_complete``.
    test_receptors : list[str] | None, optional
        Explicit test receptors for ``receptor_heldout_complete``.
    n_train_receptors : int, optional
        Target training receptor count when lists are not provided, by default ``31``.
    n_validation_receptors : int, optional
        Target validation receptor count, by default ``6``.
    n_test_receptors : int, optional
        Target test receptor count, by default ``6``.
    balance_by : str, optional
        Receptor ordering key for greedy assignment: ``"ligands"`` (default),
        ``"decoys"``, or ``"rows"``.
    shuffle_within_splits : bool, optional
        Shuffle rows within each split after receptor assignment, by default ``True``.
    """

    strategy: str = "receptor_stratified_kind"
    receptor_column: str = "receptor"
    kind_column: str = "kind"
    positive_kind: str = "ligands"
    negative_kind: str = "decoys"
    train_size: float = 0.6
    validation_size: float = 0.2
    test_size: float = 0.2
    random_seed: int = 42
    relaxed_split: bool = False
    min_kind_per_split: int = 1
    train_receptors: Optional[list[str]] = None
    validation_receptors: Optional[list[str]] = None
    test_receptors: Optional[list[str]] = None
    n_train_receptors: int = 31
    n_validation_receptors: int = 6
    n_test_receptors: int = 6
    balance_by: str = "ligands"
    shuffle_within_splits: bool = True


@dataclass
class DUDEzSplitResult:
    """Indices and diagnostics from a DUDEz split.

    Parameters
    ----------
    train_idx : np.ndarray
        Training row indices.
    val_idx : np.ndarray
        Validation row indices.
    test_idx : np.ndarray
        Test row indices.
    diagnostics : dict[str, Any]
        JSON-compatible split diagnostics and constraint reporting.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def split_dudez_by_receptor_and_kind(
        df: pd.DataFrame,
        config: Optional[DUDEzSplitConfig] = None,
        *,
        receptor_column: Optional[str] = None,
        kind_column: Optional[str] = None,
        positive_kind: Optional[str] = None,
        negative_kind: Optional[str] = None,
        train_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        test_size: Optional[float] = None,
        random_seed: Optional[int] = None,
        relaxed_split: Optional[bool] = None,
    ) -> DUDEzSplitResult:
    '''Split a DUDEz dataframe by receptor and ligand/decoy kind.

    Parameters
    ----------
    df : pd.DataFrame
        DUDEz dataframe with receptor and kind columns.
    config : DUDEzSplitConfig | None, optional
        Split configuration. Individual keyword arguments override fields when
        both are provided.
    receptor_column : str | None, optional
        Override for ``config.receptor_column``.
    kind_column : str | None, optional
        Override for ``config.kind_column``.
    positive_kind : str | None, optional
        Override for ``config.positive_kind`` (logging only).
    negative_kind : str | None, optional
        Override for ``config.negative_kind`` (logging only).
    train_size : float | None, optional
        Override for ``config.train_size``.
    validation_size : float | None, optional
        Override for ``config.validation_size``.
    test_size : float | None, optional
        Override for ``config.test_size``.
    random_seed : int | None, optional
        Override for ``config.random_seed``.
    relaxed_split : bool | None, optional
        Override for ``config.relaxed_split``.

    Returns
    -------
    DUDEzSplitResult
        Train/validation/test indices and diagnostics.

    Raises
    ------
    ValueError
        If fractions are invalid, required columns are missing, or strict validity
        constraints cannot be satisfied.
    '''

    cfg = _resolve_split_config(
        config=config,
        receptor_column=receptor_column,
        kind_column=kind_column,
        positive_kind=positive_kind,
        negative_kind=negative_kind,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_seed=random_seed,
        relaxed_split=relaxed_split,
    )
    if cfg.strategy not in DUDEZ_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported DUDEz split strategy {cfg.strategy!r}. "
            f"Expected one of {DUDEZ_SPLIT_STRATEGIES}."
        )

    if cfg.strategy != "receptor_heldout_complete":
        _validate_fractions(cfg.train_size, cfg.validation_size, cfg.test_size)

    receptors = df[cfg.receptor_column].astype(str).to_numpy()
    kinds = _normalize_kind_column(df, cfg.kind_column)
    labels = _kind_to_binary_labels(kinds)
    idx = np.arange(len(df), dtype=int)
    fractions = (cfg.train_size, cfg.validation_size, cfg.test_size)
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.strategy == "receptor_heldout_complete":
        train_idx, val_idx, test_idx, receptor_notes = _split_receptor_heldout_complete(
            idx,
            labels,
            receptors,
            rng,
            cfg,
        )
    elif cfg.strategy == "random_row":
        train_idx, val_idx, test_idx, receptor_notes = _split_random_row(idx, labels, fractions, rng)
    elif cfg.strategy == "receptor_held_out":
        train_idx, val_idx, test_idx, receptor_notes = _split_receptor_held_out(
            idx,
            labels,
            receptors,
            fractions,
            rng,
            cfg,
        )
    else:
        train_idx, val_idx, test_idx, receptor_notes = _split_receptor_stratified_kind(
            idx,
            labels,
            receptors,
            kinds,
            fractions,
            rng,
            cfg,
        )

    diagnostics = _build_split_diagnostics(
        df=df,
        labels=labels,
        receptors=receptors,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=cfg,
        receptor_notes=receptor_notes,
    )
    _validate_split_constraints(
        diagnostics,
        cfg,
        receptors=receptors,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    _log_split_diagnostics(diagnostics)
    return DUDEzSplitResult(
        train_idx=np.asarray(train_idx, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        diagnostics=diagnostics,
    )


def _resolve_split_config(
        *,
        config: Optional[DUDEzSplitConfig],
        receptor_column: Optional[str],
        kind_column: Optional[str],
        positive_kind: Optional[str],
        negative_kind: Optional[str],
        train_size: Optional[float],
        validation_size: Optional[float],
        test_size: Optional[float],
        random_seed: Optional[int],
        relaxed_split: Optional[bool],
    ) -> DUDEzSplitConfig:
    base = config or DUDEzSplitConfig()
    overrides = {
        "receptor_column": receptor_column,
        "kind_column": kind_column,
        "positive_kind": positive_kind,
        "negative_kind": negative_kind,
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "random_seed": random_seed,
        "relaxed_split": relaxed_split,
    }
    payload = asdict(base)
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    return DUDEzSplitConfig(**payload)


def _validate_fractions(train_size: float, validation_size: float, test_size: float) -> None:
    fractions = (float(train_size), float(validation_size), float(test_size))
    if any(value < 0.0 for value in fractions):
        raise ValueError("DUDEz split fractions must be non-negative.")
    total = sum(fractions)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"DUDEz split fractions must sum to 1.0 (got train={train_size}, "
            f"validation={validation_size}, test={test_size}, total={total})."
        )
    if fractions[0] <= 0.0:
        raise ValueError("DUDEz train_size must be > 0.")


def _normalize_kind_column(df: pd.DataFrame, kind_column: str) -> np.ndarray:
    source_column = kind_column
    if source_column not in df.columns and "type" in df.columns:
        source_column = "type"
    if source_column not in df.columns:
        raise ValueError(f"DUDEz dataframe must contain {kind_column!r} for kind-aware splitting.")
    return cast(np.ndarray, df[source_column].astype("string").str.strip().str.lower().to_numpy())


def _kind_to_binary_labels(kinds: np.ndarray) -> np.ndarray:
    labels = np.full(len(kinds), -1, dtype=int)
    for idx, kind in enumerate(kinds):
        if kind in POSITIVE_KIND_ALIASES:
            labels[idx] = 1
        elif kind in NEGATIVE_KIND_ALIASES:
            labels[idx] = 0
        else:
            raise ValueError(f"Unsupported DUDEz kind value: {kind!r}")
    return labels


def _split_indices_three_way(
        indices: np.ndarray,
        fractions: tuple[float, float, float],
        rng: np.random.Generator,
        *,
        require_all_splits: bool,
        min_per_split: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    '''Split row indices into train/validation/test with deterministic allocation.'''

    indices = np.asarray(indices, dtype=int)
    notes: dict[str, Any] = {}
    n = len(indices)
    if n == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            notes,
        )

    perm = rng.permutation(indices)
    train_f, val_f, test_f = fractions

    if n == 1:
        notes["assign_all_to_train"] = True
        return perm[:1], np.array([], dtype=int), np.array([], dtype=int), notes

    n_test = int(round(n * test_f))
    n_val = int(round(n * val_f))
    n_train = n - n_test - n_val

    if require_all_splits and test_f > 0.0 and n >= 3:
        n_test = max(n_test, min_per_split)
    if require_all_splits and val_f > 0.0 and n >= 2:
        n_val = max(n_val, min_per_split if n >= 3 else 1)

    while n_train + n_val + n_test > n:
        if n_test > min_per_split and n >= 3:
            n_test -= 1
        elif n_val > (min_per_split if n >= 3 else 1):
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)

    while n_train + n_val + n_test < n:
        n_train += 1

    if n_train <= 0:
        n_train = 1
        if n_val + n_test >= n:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    notes.update(
        {
            "n_total": int(n),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        }
    )
    return train_idx, val_idx, test_idx, notes


def dudez_receptor_heldout_complete_config(
        random_seed: int = 42,
        *,
        n_train_receptors: int = 31,
        n_validation_receptors: int = 6,
        n_test_receptors: int = 6,
        receptor_column: str = "receptor",
        kind_column: str = "kind",
        train_receptors: Optional[list[str]] = None,
        validation_receptors: Optional[list[str]] = None,
        test_receptors: Optional[list[str]] = None,
        relaxed_split: bool = False,
        balance_by: str = "ligands",
    ) -> DUDEzSplitConfig:
    '''Return a :class:`DUDEzSplitConfig` for complete receptor hold-out splitting.

    Parameters
    ----------
    random_seed : int, optional
        Random seed, by default ``42``.
    n_train_receptors : int, optional
        Training receptor count when lists are omitted, by default ``31``.
    n_validation_receptors : int, optional
        Validation receptor count, by default ``6``.
    n_test_receptors : int, optional
        Test receptor count, by default ``6``.
    receptor_column : str, optional
        Receptor column name, by default ``"receptor"``.
    kind_column : str, optional
        Kind column name, by default ``"kind"``.
    train_receptors : list[str] | None, optional
        Explicit training receptors.
    validation_receptors : list[str] | None, optional
        Explicit validation receptors.
    test_receptors : list[str] | None, optional
        Explicit test receptors.
    relaxed_split : bool, optional
        Allow incomplete receptors to be excluded, by default ``False``.
    balance_by : str, optional
        Greedy balancing key, by default ``"ligands"``.

    Returns
    -------
    DUDEzSplitConfig
        Configured split definition for staged DUDEz Optuna.
    '''

    return DUDEzSplitConfig(
        strategy="receptor_heldout_complete",
        receptor_column=receptor_column,
        kind_column=kind_column,
        random_seed=int(random_seed),
        relaxed_split=bool(relaxed_split),
        train_receptors=train_receptors,
        validation_receptors=validation_receptors,
        test_receptors=test_receptors,
        n_train_receptors=int(n_train_receptors),
        n_validation_receptors=int(n_validation_receptors),
        n_test_receptors=int(n_test_receptors),
        balance_by=str(balance_by),
        shuffle_within_splits=True,
    )


def _receptor_case_stats(
        idx: np.ndarray,
        labels: np.ndarray,
        receptors: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
    '''Summarize ligand/decoy counts per receptor.'''

    stats: dict[str, dict[str, Any]] = {}
    for receptor in np.unique(receptors):
        mask = receptors == receptor
        receptor_labels = labels[mask]
        n_ligands = int(np.sum(receptor_labels == 1))
        n_decoys = int(np.sum(receptor_labels == 0))
        n_rows = int(mask.sum())
        stats[str(receptor)] = {
            "name": str(receptor),
            "n_rows": n_rows,
            "n_ligands": n_ligands,
            "n_decoys": n_decoys,
            "active_fraction": float(n_ligands / n_rows) if n_rows else float("nan"),
            "indices": idx[mask],
            "complete": n_ligands > 0 and n_decoys > 0,
        }
    return stats


def _validate_explicit_receptor_lists(
        all_receptors: set[str],
        train_receptors: list[str],
        validation_receptors: list[str],
        test_receptors: list[str],
        complete_receptors: set[str],
        relaxed_split: bool,
    ) -> None:
    '''Validate user-provided receptor assignments.'''

    def _check_membership(names: list[str], split_name: str) -> None:
        unknown = sorted(set(names) - all_receptors)
        if unknown:
            raise ValueError(f"Unknown {split_name} receptors: {unknown[:10]}")
        incomplete = sorted(set(names) - complete_receptors)
        if incomplete and not relaxed_split:
            raise ValueError(
                f"{split_name} receptors must contain ligands and decoys: {incomplete[:10]}"
            )

    _check_membership(train_receptors, "train")
    _check_membership(validation_receptors, "validation")
    _check_membership(test_receptors, "test")

    train_set = set(train_receptors)
    val_set = set(validation_receptors)
    test_set = set(test_receptors)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "Explicit receptor lists overlap across splits: "
            f"train∩val={sorted(overlap_tv)[:5]}, "
            f"train∩test={sorted(overlap_tt)[:5]}, "
            f"val∩test={sorted(overlap_vt)[:5]}"
        )


def _balance_sort_key(record: dict[str, Any], balance_by: str) -> tuple[Any, ...]:
    if balance_by == "decoys":
        primary = -int(record["n_decoys"])
    elif balance_by == "rows":
        primary = -int(record["n_rows"])
    else:
        primary = -int(record["n_ligands"])
    return (primary, -int(record["n_decoys"]), -int(record["n_rows"]), str(record["name"]))


def _greedy_assign_receptors_heldout_complete(
        complete_stats: list[dict[str, Any]],
        n_train: int,
        n_validation: int,
        n_test: int,
        balance_by: str,
        rng: np.random.Generator,
    ) -> tuple[list[str], list[str], list[str]]:
    '''Assign complete receptors to train/validation/test with greedy balancing.'''

    ordered = sorted(
        complete_stats,
        key=lambda item: (
            *_balance_sort_key(item, balance_by)[:-1],
            float(rng.random()),
            str(item["name"]),
        ),
    )
    train_receptors: list[str] = []
    validation_receptors: list[str] = []
    test_receptors: list[str] = []
    val_ligands = 0
    val_decoys = 0
    test_ligands = 0
    test_decoys = 0

    for record in ordered:
        name = str(record["name"])
        n_ligands = int(record["n_ligands"])
        n_decoys = int(record["n_decoys"])
        if len(validation_receptors) < n_validation and len(test_receptors) < n_test:
            val_after = abs((val_ligands + n_ligands) - test_ligands) + abs((val_decoys + n_decoys) - test_decoys)
            test_after = abs(val_ligands - (test_ligands + n_ligands)) + abs(val_decoys - (test_decoys + n_decoys))
            if val_after <= test_after:
                validation_receptors.append(name)
                val_ligands += n_ligands
                val_decoys += n_decoys
            else:
                test_receptors.append(name)
                test_ligands += n_ligands
                test_decoys += n_decoys
        elif len(validation_receptors) < n_validation:
            validation_receptors.append(name)
            val_ligands += n_ligands
            val_decoys += n_decoys
        elif len(test_receptors) < n_test:
            test_receptors.append(name)
            test_ligands += n_ligands
            test_decoys += n_decoys
        else:
            train_receptors.append(name)

    assigned = set(train_receptors) | set(validation_receptors) | set(test_receptors)
    for record in ordered:
        name = str(record["name"])
        if name in assigned:
            continue
        train_receptors.append(name)

    if len(validation_receptors) != n_validation or len(test_receptors) != n_test:
        raise ValueError(
            "Could not assign the requested receptor counts. "
            f"Got validation={len(validation_receptors)} (expected {n_validation}), "
            f"test={len(test_receptors)} (expected {n_test}), "
            f"complete_receptors={len(ordered)}."
        )
    expected_train = len(ordered) - n_validation - n_test
    if len(train_receptors) != expected_train:
        raise ValueError(
            f"Training receptor count mismatch: got {len(train_receptors)}, expected {expected_train}."
        )
    return train_receptors, validation_receptors, test_receptors


def _indices_for_receptor_set(
        idx: np.ndarray,
        receptors: np.ndarray,
        receptor_names: set[str],
    ) -> np.ndarray:
    if not receptor_names:
        return np.array([], dtype=int)
    return idx[np.isin(receptors, list(receptor_names))]


def _shuffle_within_splits(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Shuffle row order inside each split.'''

    def _permute(indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return indices
        return indices[rng.permutation(len(indices))]

    return _permute(train_idx), _permute(val_idx), _permute(test_idx)


def _split_receptor_heldout_complete(
        idx: np.ndarray,
        labels: np.ndarray,
        receptors: np.ndarray,
        rng: np.random.Generator,
        config: DUDEzSplitConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    '''Assign complete receptor blocks to train, validation, and test.'''

    stats = _receptor_case_stats(idx, labels, receptors)
    all_receptors = set(stats.keys())
    complete = [record for record in stats.values() if record["complete"]]
    incomplete = sorted(name for name, record in stats.items() if not record["complete"])
    complete_names = {str(record["name"]) for record in complete}

    if incomplete and not config.relaxed_split:
        raise ValueError(
            "receptor_heldout_complete requires every receptor to contain ligands and decoys. "
            f"Incomplete receptors: {incomplete[:10]}"
        )

    explicit = (
        config.train_receptors is not None
        or config.validation_receptors is not None
        or config.test_receptors is not None
    )
    if explicit:
        if config.train_receptors is None or config.validation_receptors is None or config.test_receptors is None:
            raise ValueError(
                "receptor_heldout_complete requires train_receptors, validation_receptors, "
                "and test_receptors when any explicit list is provided."
            )
        train_receptors = [str(name) for name in config.train_receptors]
        validation_receptors = [str(name) for name in config.validation_receptors]
        test_receptors = [str(name) for name in config.test_receptors]
        _validate_explicit_receptor_lists(
            all_receptors,
            train_receptors,
            validation_receptors,
            test_receptors,
            complete_names,
            config.relaxed_split,
        )
    else:
        n_complete = len(complete)
        n_required = (
            int(config.n_train_receptors)
            + int(config.n_validation_receptors)
            + int(config.n_test_receptors)
        )
        if n_complete != n_required:
            raise ValueError(
                "receptor_heldout_complete automatic assignment requires exactly "
                f"{n_required} complete receptors (got {n_complete}). "
                "Provide explicit receptor lists or adjust n_*_receptors."
            )
        train_receptors, validation_receptors, test_receptors = _greedy_assign_receptors_heldout_complete(
            complete,
            n_train=int(config.n_train_receptors),
            n_validation=int(config.n_validation_receptors),
            n_test=int(config.n_test_receptors),
            balance_by=str(config.balance_by),
            rng=rng,
        )

    train_set = set(train_receptors)
    val_set = set(validation_receptors)
    test_set = set(test_receptors)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Receptor sets must be disjoint across train/validation/test.")

    train_idx = _indices_for_receptor_set(idx, receptors, train_set)
    val_idx = _indices_for_receptor_set(idx, receptors, val_set)
    test_idx = _indices_for_receptor_set(idx, receptors, test_set)

    if config.shuffle_within_splits:
        train_idx, val_idx, test_idx = _shuffle_within_splits(train_idx, val_idx, test_idx, rng)

    receptor_notes: dict[str, Any] = {
        "assignment_mode": "explicit" if explicit else "greedy_balanced",
        "balance_by": str(config.balance_by),
        "train_receptors": sorted(train_set),
        "validation_receptors": sorted(val_set),
        "test_receptors": sorted(test_set),
        "complete_receptors": sorted(complete_names),
        "incomplete_receptors": incomplete,
        "n_receptors_total": int(len(all_receptors)),
        "n_receptors_complete": int(len(complete_names)),
        "n_receptors_incomplete": int(len(incomplete)),
    }
    for receptor in sorted(train_set | val_set | test_set):
        split_name = (
            "train"
            if receptor in train_set
            else "validation"
            if receptor in val_set
            else "test"
        )
        receptor_notes.setdefault(str(receptor), {})["heldout_complete_split"] = split_name

    for receptor in incomplete:
        receptor_notes.setdefault(receptor, {})["excluded_incomplete"] = True

    return train_idx, val_idx, test_idx, receptor_notes


def _split_receptor_stratified_kind(
        idx: np.ndarray,
        labels: np.ndarray,
        receptors: np.ndarray,
        kinds: np.ndarray,
        fractions: tuple[float, float, float],
        rng: np.random.Generator,
        config: DUDEzSplitConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    receptor_notes: dict[str, Any] = {}

    for receptor in np.unique(receptors):
        receptor_mask = receptors == receptor
        receptor_idx = idx[receptor_mask]
        receptor_labels = labels[receptor_mask]
        positive_idx = receptor_idx[receptor_labels == 1]
        negative_idx = receptor_idx[receptor_labels == 0]
        receptor_key = str(receptor)
        receptor_seed = int(config.random_seed) + (zlib.adler32(receptor_key.encode("utf-8")) % 10_000)
        receptor_rng = np.random.default_rng(receptor_seed)

        min_required = max(1, int(config.min_kind_per_split))
        require_all = (
            len(positive_idx) >= 3 * min_required
            and len(negative_idx) >= 3 * min_required
        )

        pos_train, pos_val, pos_test, pos_notes = _split_indices_three_way(
            positive_idx,
            fractions,
            receptor_rng,
            require_all_splits=require_all,
            min_per_split=min_required,
        )
        neg_train, neg_val, neg_test, neg_notes = _split_indices_three_way(
            negative_idx,
            fractions,
            np.random.default_rng(receptor_seed + 1),
            require_all_splits=require_all,
            min_per_split=min_required,
        )

        note_entry: dict[str, Any] = {
            "n_ligands": int(len(positive_idx)),
            "n_decoys": int(len(negative_idx)),
            "ligand_split": pos_notes,
            "decoy_split": neg_notes,
            "in_all_splits": bool(
                len(pos_train) and len(pos_val) and len(pos_test)
                and len(neg_train) and len(neg_val) and len(neg_test)
            ),
        }

        if not require_all and config.relaxed_split:
            if len(pos_val) == 0 or len(pos_test) == 0:
                reassigned, pos_train, pos_val, pos_test = _move_indices_to_train(
                    pos_train, pos_val, pos_test
                )
                if reassigned:
                    note_entry["ligands_train_only_fallback"] = True
            if len(neg_val) == 0 or len(neg_test) == 0:
                reassigned, neg_train, neg_val, neg_test = _move_indices_to_train(
                    neg_train, neg_val, neg_test
                )
                if reassigned:
                    note_entry["decoys_train_only_fallback"] = True
        elif not require_all and not config.relaxed_split:
            if len(positive_idx) and (len(pos_val) == 0 or len(pos_test) == 0):
                note_entry["ligands_insufficient_for_all_splits"] = True
            if len(negative_idx) and (len(neg_val) == 0 or len(neg_test) == 0):
                note_entry["decoys_insufficient_for_all_splits"] = True

        train_parts.extend([pos_train, neg_train])
        val_parts.extend([pos_val, neg_val])
        test_parts.extend([pos_test, neg_test])
        receptor_notes[receptor_key] = note_entry

    return (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=int),
        np.concatenate(val_parts) if val_parts else np.array([], dtype=int),
        np.concatenate(test_parts) if test_parts else np.array([], dtype=int),
        receptor_notes,
    )


def _move_indices_to_train(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    moved = np.concatenate([val_idx, test_idx]) if len(val_idx) or len(test_idx) else np.array([], dtype=int)
    if len(moved) == 0:
        return False, train_idx, val_idx, test_idx
    train_idx = np.concatenate([train_idx, moved])
    return True, train_idx, np.array([], dtype=int), np.array([], dtype=int)


def _split_receptor_held_out(
        idx: np.ndarray,
        labels: np.ndarray,
        receptors: np.ndarray,
        fractions: tuple[float, float, float],
        rng: np.random.Generator,
        config: DUDEzSplitConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    unique_receptors = np.unique(receptors)
    receptor_notes: dict[str, Any] = {}
    valid_receptors: list[str] = []
    for receptor in unique_receptors:
        mask = receptors == receptor
        has_positive = bool(np.any(labels[mask] == 1))
        has_negative = bool(np.any(labels[mask] == 0))
        receptor_key = str(receptor)
        if has_positive and has_negative:
            valid_receptors.append(receptor_key)
        else:
            receptor_notes[receptor_key] = {
                "skipped_held_out": True,
                "reason": "missing_ligands_or_decoys",
            }

    if not valid_receptors:
        raise ValueError("No receptors contain both ligands and decoys for receptor-held-out splitting.")

    valid_array = np.array(valid_receptors, dtype=object)
    perm = rng.permutation(valid_array)
    n_receptors = len(perm)
    n_test = int(round(n_receptors * fractions[2]))
    n_val = int(round(n_receptors * fractions[1]))
    n_train = n_receptors - n_test - n_val
    if n_train <= 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    train_receptors = set(perm[:n_train].tolist())
    val_receptors = set(perm[n_train : n_train + n_val].tolist())
    test_receptors = set(perm[n_train + n_val :].tolist())

    train_idx = idx[np.isin(receptors, list(train_receptors))]
    val_idx = idx[np.isin(receptors, list(val_receptors))]
    test_idx = idx[np.isin(receptors, list(test_receptors))]

    for receptor in valid_receptors:
        receptor_notes.setdefault(receptor, {})["held_out_split"] = (
            "train" if receptor in train_receptors else "validation" if receptor in val_receptors else "test"
        )

    skipped = [r for r in unique_receptors if str(r) not in valid_receptors]
    if skipped and not config.relaxed_split:
        raise ValueError(
            "Receptor-held-out split requires every receptor to contain ligands and decoys. "
            f"Invalid receptors: {skipped[:10]}"
        )

    return train_idx, val_idx, test_idx, receptor_notes


def _split_random_row(
        idx: np.ndarray,
        labels: np.ndarray,
        fractions: tuple[float, float, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    LOGGER.warning(
        "DUDEz random_row split ignores receptor structure; grouped validation metrics may be invalid."
    )
    train_idx, val_idx, test_idx, notes = _split_indices_three_way(
        idx,
        fractions,
        rng,
        require_all_splits=True,
        min_per_split=1,
    )
    return train_idx, val_idx, test_idx, {"random_row_notes": notes}


def _split_summary_for_indices(
        indices: np.ndarray,
        labels: np.ndarray,
        receptors: np.ndarray,
        split_name: str,
    ) -> dict[str, Any]:
    split_labels = labels[indices]
    actives = int(np.sum(split_labels == 1))
    decoys = int(np.sum(split_labels == 0))
    total = int(len(indices))
    split_receptors = receptors[indices]
    unique_targets = np.unique(split_receptors)
    per_target: dict[str, dict[str, int]] = {}
    zero_actives: list[str] = []
    zero_decoys: list[str] = []
    valid_metric_groups: list[str] = []
    invalid_metric_groups: list[str] = []
    for target in unique_targets:
        mask = split_receptors == target
        target_labels = split_labels[mask]
        target_actives = int(np.sum(target_labels == 1))
        target_decoys = int(np.sum(target_labels == 0))
        target_name = str(target)
        per_target[target_name] = {
            "n_rows": int(mask.sum()),
            "n_ligands": target_actives,
            "n_decoys": target_decoys,
        }
        if target_actives == 0:
            zero_actives.append(target_name)
        if target_decoys == 0:
            zero_decoys.append(target_name)
        if target_actives > 0 and target_decoys > 0:
            valid_metric_groups.append(target_name)
        else:
            invalid_metric_groups.append(target_name)

    return {
        "split": split_name,
        "n_rows": total,
        "n_ligands": actives,
        "n_actives": actives,
        "n_decoys": decoys,
        "active_fraction": float(actives / total) if total else float("nan"),
        "n_receptors": int(len(unique_targets)),
        "n_targets": int(len(unique_targets)),
        "targets_with_zero_actives": zero_actives,
        "targets_with_zero_decoys": zero_decoys,
        "valid_metric_groups": valid_metric_groups,
        "invalid_metric_groups": invalid_metric_groups,
        "n_valid_metric_groups": int(len(valid_metric_groups)),
        "n_invalid_metric_groups": int(len(invalid_metric_groups)),
        "per_target_counts": per_target,
    }


def _build_split_diagnostics(
        *,
        df: pd.DataFrame,
        labels: np.ndarray,
        receptors: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        config: DUDEzSplitConfig,
        receptor_notes: dict[str, Any],
) -> dict[str, Any]:
    all_receptors = {str(value) for value in np.unique(receptors)}
    val_receptors = {str(value) for value in np.unique(receptors[val_idx])} if len(val_idx) else set()
    test_receptors = {str(value) for value in np.unique(receptors[test_idx])} if len(test_idx) else set()
    receptors_missing_from_validation = sorted(all_receptors - val_receptors)
    receptors_missing_from_test = sorted(all_receptors - test_receptors)

    val_summary = _split_summary_for_indices(val_idx, labels, receptors, "validation")
    test_summary = _split_summary_for_indices(test_idx, labels, receptors, "test")

    relaxed_constraints: list[str] = []
    for receptor, note in receptor_notes.items():
        if isinstance(note, dict) and note.get("ligands_train_only_fallback"):
            relaxed_constraints.append(f"{receptor}:ligands_train_only")
        if isinstance(note, dict) and note.get("decoys_train_only_fallback"):
            relaxed_constraints.append(f"{receptor}:decoys_train_only")

    payload: dict[str, Any] = {
        "strategy": config.strategy,
        "random_seed": int(config.random_seed),
        "relaxed_split": bool(config.relaxed_split),
        "receptor_column": config.receptor_column,
        "kind_column": config.kind_column,
        "positive_kind": config.positive_kind,
        "negative_kind": config.negative_kind,
        "fractions": {
            "train": float(config.train_size),
            "validation": float(config.validation_size),
            "test": float(config.test_size),
        },
        "n_rows_total": int(len(df)),
        "splits": {
            "train": _split_summary_for_indices(train_idx, labels, receptors, "train"),
            "validation": val_summary,
            "test": test_summary,
        },
        "receptors_missing_from_validation": receptors_missing_from_validation,
        "receptors_missing_from_test": receptors_missing_from_test,
        "receptor_notes": receptor_notes,
        "relaxed_constraints": relaxed_constraints,
        "n_groups_total_validation": val_summary["n_receptors"],
        "n_groups_used_validation": val_summary["n_valid_metric_groups"],
        "n_groups_total_test": test_summary["n_receptors"],
        "n_groups_used_test": test_summary["n_valid_metric_groups"],
    }
    if config.strategy == "receptor_heldout_complete":
        payload.update(
            {
                "n_receptors_total": receptor_notes.get("n_receptors_total"),
                "n_receptors_complete": receptor_notes.get("n_receptors_complete"),
                "n_receptors_incomplete": receptor_notes.get("n_receptors_incomplete"),
                "train_receptors": receptor_notes.get("train_receptors", []),
                "validation_receptors": receptor_notes.get("validation_receptors", []),
                "test_receptors": receptor_notes.get("test_receptors", []),
                "complete_receptors": receptor_notes.get("complete_receptors", []),
                "incomplete_receptors": receptor_notes.get("incomplete_receptors", []),
                "balance_by": receptor_notes.get("balance_by"),
                "assignment_mode": receptor_notes.get("assignment_mode"),
                "shuffle_within_splits": bool(config.shuffle_within_splits),
                "receptor_counts": {
                    "train": int(len(receptor_notes.get("train_receptors") or [])),
                    "validation": int(len(receptor_notes.get("validation_receptors") or [])),
                    "test": int(len(receptor_notes.get("test_receptors") or [])),
                },
            }
        )
    return payload


def _validate_split_constraints(
        diagnostics: dict[str, Any],
        config: DUDEzSplitConfig,
        *,
        receptors: Optional[np.ndarray] = None,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
    ) -> None:
    failures: list[str] = []
    if config.strategy == "receptor_heldout_complete":
        train_receptors = set(diagnostics.get("train_receptors") or [])
        val_receptors = set(diagnostics.get("validation_receptors") or [])
        test_receptors = set(diagnostics.get("test_receptors") or [])
        if train_receptors & val_receptors:
            failures.append("train and validation receptors overlap")
        if train_receptors & test_receptors:
            failures.append("train and test receptors overlap")
        if val_receptors & test_receptors:
            failures.append("validation and test receptors overlap")

        if receptors is not None and train_idx is not None and val_idx is not None and test_idx is not None:
            for receptor in sorted(train_receptors | val_receptors | test_receptors):
                in_train = bool(np.any(receptors[train_idx] == receptor)) if len(train_idx) else False
                in_val = bool(np.any(receptors[val_idx] == receptor)) if len(val_idx) else False
                in_test = bool(np.any(receptors[test_idx] == receptor)) if len(test_idx) else False
                if int(in_train) + int(in_val) + int(in_test) != 1:
                    failures.append(f"receptor {receptor} is fragmented across splits")

    for split_name in ("train", "validation", "test"):
        summary = diagnostics["splits"][split_name]
        if summary["n_ligands"] < 1:
            failures.append(f"{split_name} has zero ligands")
        if summary["n_decoys"] < 1:
            failures.append(f"{split_name} has zero decoys")

    for split_name in ("validation", "test"):
        summary = diagnostics["splits"][split_name]
        if summary["n_valid_metric_groups"] < 1:
            failures.append(f"{split_name} has no receptor groups with both ligands and decoys")
        if config.strategy == "receptor_heldout_complete":
            if summary["n_valid_metric_groups"] != summary["n_receptors"]:
                failures.append(
                    f"{split_name} grouped metrics require all receptor groups valid "
                    f"(used={summary['n_valid_metric_groups']}, total={summary['n_receptors']})"
                )

    insufficient = [
        receptor
        for receptor, note in diagnostics.get("receptor_notes", {}).items()
        if isinstance(note, dict)
        and (
            note.get("ligands_insufficient_for_all_splits")
            or note.get("decoys_insufficient_for_all_splits")
        )
    ]
    if insufficient and not config.relaxed_split:
        failures.append(
            "receptors with insufficient ligands/decoys for all splits: "
            + ", ".join(insufficient[:10])
        )

    if failures and not config.relaxed_split:
        raise ValueError(
            "DUDEz split failed validity constraints (set relaxed_split=True to allow documented fallbacks): "
            + "; ".join(failures)
        )

    if failures and config.relaxed_split:
        diagnostics["relaxed_constraints"] = list(diagnostics.get("relaxed_constraints", [])) + failures
        LOGGER.warning(
            "DUDEz split relaxed constraints: %s",
            "; ".join(failures),
        )


def _log_split_diagnostics(diagnostics: dict[str, Any]) -> None:
    LOGGER.debug(
        "DUDEz split strategy=%s seed=%s relaxed=%s rows=%s",
        diagnostics.get("strategy"),
        diagnostics.get("random_seed"),
        diagnostics.get("relaxed_split"),
        diagnostics.get("n_rows_total"),
    )
    for split_name, summary in diagnostics.get("splits", {}).items():
        LOGGER.debug(
            "DUDEz %s: rows=%s ligands=%s decoys=%s active_fraction=%.4f receptors=%s "
            "valid_metric_groups=%s invalid_metric_groups=%s",
            split_name,
            summary.get("n_rows"),
            summary.get("n_ligands"),
            summary.get("n_decoys"),
            summary.get("active_fraction"),
            summary.get("n_receptors"),
            summary.get("n_valid_metric_groups"),
            summary.get("n_invalid_metric_groups"),
        )
        invalid = summary.get("invalid_metric_groups") or []
        if invalid:
            LOGGER.warning(
                "DUDEz %s invalid metric groups (single-class receptors): %s",
                split_name,
                invalid[:10],
            )
    missing_val = diagnostics.get("receptors_missing_from_validation") or []
    missing_test = diagnostics.get("receptors_missing_from_test") or []
    if missing_val:
        LOGGER.debug("DUDEz receptors absent from validation: %s", missing_val[:10])
    if missing_test:
        LOGGER.debug("DUDEz receptors absent from test: %s", missing_test[:10])
    relaxed = diagnostics.get("relaxed_constraints") or []
    if relaxed:
        LOGGER.warning("DUDEz split applied relaxed constraints: %s", relaxed[:20])


__all__ = [
    "DUDEZ_SPLIT_STRATEGIES",
    "DUDEzSplitConfig",
    "DUDEzSplitResult",
    "dudez_receptor_heldout_complete_config",
    "split_dudez_by_receptor_and_kind",
]
