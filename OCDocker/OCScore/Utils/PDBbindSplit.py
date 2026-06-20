#!/usr/bin/env python3

# Description
###############################################################################
'''Affinity-aware train/validation/test splitting for PDBbind regression.

Default strategy ``affinity_quantile_stratified`` bins the continuous affinity
target into quantiles and uses those bins for stratified splitting. This keeps
weak/medium/strong binders represented across train/validation/test so RMSE
selection is scientifically meaningful.
'''

# Imports
###############################################################################
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

import OCDocker.Toolbox.Logging as oclogging

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

LOGGER = oclogging.get_logger("ocscore.utils.pdbbind_split")

PDBBIND_SPLIT_STRATEGIES = (
    "affinity_quantile_stratified",
    "receptor_heldout",
    "random_row",
)


@dataclass
class PDBbindSplitConfig:
    """Configuration for PDBbind regression splitting.

    Parameters
    ----------
    strategy : str, optional
        Split strategy. Defaults to ``"affinity_quantile_stratified"``.
    target_column : str, optional
        Continuous regression target column, by default ``"experimental"``.
    receptor_column : str | None, optional
        Optional receptor/target column for diagnostics, by default ``"receptor"``.
        If missing in the dataframe, receptor diagnostics are skipped.
    n_affinity_bins : int, optional
        Requested number of affinity quantile bins, by default 5.
    train_size : float, optional
        Training fraction, by default 0.6.
    validation_size : float, optional
        Validation fraction, by default 0.2.
    test_size : float, optional
        Test fraction, by default 0.2.
    random_seed : int, optional
        Random seed for deterministic splits, by default 42.
    relaxed_split : bool, optional
        If False, fail loudly when quantile binning/stratification is invalid.
        If True, reduce the number of bins until stratification is feasible and
        drop rows with NaN targets, by default False.
    """

    strategy: str = "affinity_quantile_stratified"
    target_column: str = "experimental"
    receptor_column: Optional[str] = "receptor"
    n_affinity_bins: int = 5
    train_size: float = 0.6
    validation_size: float = 0.2
    test_size: float = 0.2
    random_seed: int = 42
    relaxed_split: bool = False


@dataclass
class PDBbindSplitResult:
    """Indices and diagnostics from a PDBbind regression split.

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


def split_pdbbind_regression(
        df: pd.DataFrame,
        config: Optional[PDBbindSplitConfig] = None,
        *,
        target: Optional[np.ndarray] = None,
    ) -> PDBbindSplitResult:
    '''Split PDBbind regression rows into train/validation/test.

    Parameters
    ----------
    df : pd.DataFrame
        PDBbind dataframe containing a continuous affinity target.
    config : PDBbindSplitConfig | None, optional
        Split configuration. Defaults to :class:`PDBbindSplitConfig`.
    target : np.ndarray | None, optional
        Optional override target array aligned with ``df``. When provided, it is
        used instead of ``df[config.target_column]``. This exists to support
        upstream preprocessing while keeping a single splitting implementation.

    Returns
    -------
    PDBbindSplitResult
        Indices for each split plus JSON-compatible diagnostics.
    '''

    cfg = config or PDBbindSplitConfig()
    _validate_fractions(cfg.train_size, cfg.validation_size, cfg.test_size)
    if cfg.strategy not in PDBBIND_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported PDBbind split strategy {cfg.strategy!r}. Expected one of {PDBBIND_SPLIT_STRATEGIES}."
        )

    y_full = np.asarray(target if target is not None else df[cfg.target_column].to_numpy(), dtype=float).reshape(-1)
    if y_full.shape[0] != len(df):
        raise ValueError("PDBbind split target must align with dataframe length.")

    idx_all = np.arange(len(df), dtype=int)
    nan_mask = np.isnan(y_full)
    dropped_nan = 0
    if nan_mask.any():
        if not cfg.relaxed_split:
            raise ValueError(
                f"PDBbind target column {cfg.target_column!r} contains NaN values; "
                "set relaxed_split=True to drop them explicitly."
            )
        keep = ~nan_mask
        dropped_nan = int(nan_mask.sum())
        idx_all = idx_all[keep]
        y = y_full[keep]
    else:
        y = y_full

    rng = np.random.default_rng(int(cfg.random_seed))
    if cfg.strategy == "random_row":
        perm = rng.permutation(idx_all)
        train_idx, val_idx, test_idx = _split_three_way_from_perm(
            perm,
            (cfg.train_size, cfg.validation_size, cfg.test_size),
        )
        diagnostics = _build_diagnostics(df, y, train_idx, val_idx, test_idx, cfg, dropped_nan, bins=None)
        _validate_split_nonempty(train_idx, val_idx, test_idx)
        _log_diagnostics(diagnostics)
        return PDBbindSplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, diagnostics=diagnostics)

    if cfg.strategy == "receptor_heldout":
        receptor_col = _resolve_receptor_column(df, cfg.receptor_column)
        if cfg.receptor_column != receptor_col:
            cfg = PDBbindSplitConfig(**{**asdict(cfg), "receptor_column": receptor_col})
        train_idx, val_idx, test_idx, heldout_notes = _split_receptor_heldout(
            df.iloc[idx_all].reset_index(drop=True),
            y,
            idx_all,
            cfg,
            rng,
        )
        diagnostics = _build_diagnostics(df, y, train_idx, val_idx, test_idx, cfg, dropped_nan, bins=None)
        if heldout_notes:
            diagnostics["heldout_notes"] = heldout_notes
        _validate_receptor_disjoint(diagnostics, strict=True)
        _validate_split_nonempty(train_idx, val_idx, test_idx)
        _log_diagnostics(diagnostics)
        return PDBbindSplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, diagnostics=diagnostics)

    if cfg.strategy != "affinity_quantile_stratified":
        raise ValueError(f"Unhandled PDBbind split strategy {cfg.strategy!r}.")

    bins, edges, used_bins, relaxed_notes = _make_affinity_bins(y, int(cfg.n_affinity_bins), bool(cfg.relaxed_split))
    train_idx, val_idx, test_idx = _stratified_three_way_split(
        idx_all,
        bins,
        (cfg.train_size, cfg.validation_size, cfg.test_size),
        int(cfg.random_seed),
        relaxed_split=bool(cfg.relaxed_split),
    )
    train_pos = np.searchsorted(idx_all, train_idx)
    val_pos = np.searchsorted(idx_all, val_idx)
    test_pos = np.searchsorted(idx_all, test_idx)
    diagnostics = _build_diagnostics(
        df,
        y,
        train_idx,
        val_idx,
        test_idx,
        cfg,
        dropped_nan,
        bins={
            "requested_bins": int(cfg.n_affinity_bins),
            "used_bins": int(used_bins),
            "bin_edges": [float(v) for v in edges] if edges is not None else None,
            "bin_counts_total": _counts_json(bins),
            "bin_counts_train": _counts_json(bins[train_pos] if len(train_pos) else np.array([], dtype=int)),
            "bin_counts_validation": _counts_json(bins[val_pos] if len(val_pos) else np.array([], dtype=int)),
            "bin_counts_test": _counts_json(bins[test_pos] if len(test_pos) else np.array([], dtype=int)),
        },
    )
    if relaxed_notes:
        diagnostics["relaxed_notes"] = relaxed_notes
    _validate_split_nonempty(train_idx, val_idx, test_idx)
    _log_diagnostics(diagnostics)
    return PDBbindSplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, diagnostics=diagnostics)


def _validate_fractions(train_size: float, validation_size: float, test_size: float) -> None:
    fractions = (float(train_size), float(validation_size), float(test_size))
    if any(value < 0.0 for value in fractions):
        raise ValueError("PDBbind split fractions must be non-negative.")
    total = sum(fractions)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"PDBbind split fractions must sum to 1.0 (got train={train_size}, validation={validation_size}, "
            f"test={test_size}, total={total})."
        )
    if fractions[0] <= 0.0:
        raise ValueError("PDBbind train_size must be > 0.")


def _make_affinity_bins(
        y: np.ndarray,
        requested_bins: int,
        relaxed_split: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray], int, list[str]]:
    if requested_bins < 2:
        raise ValueError("n_affinity_bins must be >= 2.")
    notes: list[str] = []
    y = np.asarray(y, dtype=float).reshape(-1)
    unique = int(len(np.unique(y)))
    if unique < 2:
        raise ValueError("PDBbind target has <2 unique values; affinity stratification is impossible.")

    bins_to_try = min(int(requested_bins), unique)
    while bins_to_try >= 2:
        try:
            cats, edges = pd.qcut(y, q=bins_to_try, retbins=True, labels=False, duplicates="drop")
            labels = np.asarray(cats, dtype=int)
            used_bins = int(len(np.unique(labels)))
            if used_bins < 2:
                raise ValueError("qcut produced <2 bins")
            counts = np.bincount(labels, minlength=used_bins)
            if counts.min() < 2:
                raise ValueError("some bins have <2 samples; stratification may fail")
            return labels, np.asarray(edges, dtype=float), used_bins, notes
        except Exception as exc:
            if not relaxed_split:
                raise ValueError(
                    f"Failed to create {requested_bins} affinity quantile bins for stratification: {exc}. "
                    "Try fewer bins or set relaxed_split=True to auto-reduce bin count."
                ) from exc
            notes.append(f"reduced_bins_from_{bins_to_try}")
            bins_to_try -= 1

    raise ValueError("Unable to create valid affinity bins for PDBbind splitting.")


def _stratified_three_way_split(
        idx: np.ndarray,
        strat_labels: np.ndarray,
        fractions: tuple[float, float, float],
        seed: int,
        *,
        relaxed_split: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.asarray(idx, dtype=int)
    strat_labels = np.asarray(strat_labels, dtype=int)
    if idx.size != strat_labels.size:
        raise ValueError("Stratification labels must align with indices.")

    train_f, val_f, test_f = fractions
    test_size = float(test_f)
    val_fraction = float(val_f) / max(1e-8, 1.0 - test_size)

    try:
        train_val_idx, test_idx, train_val_bins, _ = train_test_split(
            idx,
            strat_labels,
            test_size=test_size,
            random_state=int(seed),
            stratify=strat_labels,
        )
        train_idx, val_idx = train_test_split(
            np.asarray(train_val_idx, dtype=int),
            test_size=val_fraction,
            random_state=int(seed) + 1,
            stratify=np.asarray(train_val_bins, dtype=int),
        )
        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)
    except Exception as exc:
        if not relaxed_split:
            raise ValueError(
                f"PDBbind affinity-bin stratified split failed: {exc}. "
                "Set relaxed_split=True to allow bin reduction."
            ) from exc
        raise


def _split_three_way_from_perm(
        perm: np.ndarray,
        fractions: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(perm))
    train_f, val_f, test_f = fractions
    n_test = int(round(n * float(test_f)))
    n_val = int(round(n * float(val_f)))
    n_train = n - n_test - n_val
    if n_train <= 0:
        n_train = max(1, n_train)
        n_val = max(0, n_val - 1) if n_val > 0 else n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _resolve_receptor_column(df: pd.DataFrame, configured: Optional[str]) -> str:
    '''Resolve the receptor grouping column for held-out splits.'''

    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(["receptor", "Protein", "protein"])
    for column in candidates:
        if column in df.columns:
            return column
    available = sorted(str(col) for col in df.columns)
    raise ValueError(
        "receptor_heldout requires a receptor column (tried "
        f"{candidates}). Available columns: {available[:20]}"
        + (" ..." if len(available) > 20 else "")
    )


def _split_receptor_heldout(
        df_used: pd.DataFrame,
        y: np.ndarray,
        idx_all: np.ndarray,
        config: PDBbindSplitConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    '''Assign whole receptors to train/validation/test splits.'''

    receptor_column = _resolve_receptor_column(df_used, config.receptor_column)
    receptors = df_used[receptor_column].astype(str).to_numpy()
    unique_receptors = sorted(set(receptors.tolist()))
    n_receptors = len(unique_receptors)
    if n_receptors < 3:
        raise ValueError(
            f"receptor_heldout requires at least 3 receptors (got {n_receptors}). "
            "Use affinity_quantile_stratified or random_row for tiny datasets."
        )

    n_val = max(1, int(round(n_receptors * float(config.validation_size))))
    n_test = max(1, int(round(n_receptors * float(config.test_size))))
    if n_val + n_test >= n_receptors:
        raise ValueError(
            f"receptor_heldout cannot reserve validation+test receptors "
            f"(validation={n_val}, test={n_test}, total receptors={n_receptors}). "
            "Reduce validation_size/test_size or use row-level splitting."
        )
    n_train = n_receptors - n_val - n_test

    receptor_stats = []
    for name in unique_receptors:
        mask = receptors == name
        receptor_stats.append({
            "name": name,
            "n_rows": int(mask.sum()),
            "mean_target": float(np.mean(y[mask])),
        })
    ordered = sorted(
        receptor_stats,
        key=lambda item: (-int(item["n_rows"]), float(rng.random()), str(item["name"])),
    )
    val_receptors: list[str] = []
    test_receptors: list[str] = []
    val_rows = 0
    test_rows = 0
    for record in ordered:
        name = str(record["name"])
        n_rows = int(record["n_rows"])
        if len(val_receptors) < n_val and len(test_receptors) < n_test:
            if abs((val_rows + n_rows) - test_rows) <= abs(val_rows - (test_rows + n_rows)):
                val_receptors.append(name)
                val_rows += n_rows
            else:
                test_receptors.append(name)
                test_rows += n_rows
        elif len(val_receptors) < n_val:
            val_receptors.append(name)
            val_rows += n_rows
        elif len(test_receptors) < n_test:
            test_receptors.append(name)
            test_rows += n_rows
    train_receptors = [
        str(record["name"])
        for record in ordered
        if str(record["name"]) not in set(val_receptors) | set(test_receptors)
    ]
    if len(train_receptors) != n_train:
        train_receptors = [
            str(record["name"])
            for record in ordered
            if str(record["name"]) not in set(val_receptors) | set(test_receptors)
        ]

    def _indices_for(names: set[str]) -> np.ndarray:
        if not names:
            return np.array([], dtype=int)
        local_mask = np.isin(receptors, list(names))
        return idx_all[local_mask]

    train_idx = _indices_for(set(train_receptors))
    val_idx = _indices_for(set(val_receptors))
    test_idx = _indices_for(set(test_receptors))
    notes = [
        f"receptor_column={receptor_column}",
        f"n_receptors_total={n_receptors}",
        f"n_receptors_train={len(train_receptors)}",
        f"n_receptors_validation={len(val_receptors)}",
        f"n_receptors_test={len(test_receptors)}",
    ]
    return train_idx, val_idx, test_idx, notes


def _validate_receptor_disjoint(diagnostics: dict[str, Any], *, strict: bool) -> None:
    '''Raise when receptor overlap exists across splits in held-out mode.'''

    if diagnostics.get("strategy") != "receptor_heldout":
        return
    overlap = diagnostics.get("receptor_overlap") or {}
    total_overlap = sum(int(value) for value in overlap.values())
    if total_overlap > 0:
        message = f"PDBbind receptor_heldout split has receptor overlap: {overlap}"
        if strict:
            raise ValueError(message)
        LOGGER.warning(message)


def _validate_split_nonempty(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"PDBbind split produced empty split(s): train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}."
        )


def _counts_json(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _target_stats(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    return {
        "min": float(np.min(y)) if y.size else float("nan"),
        "mean": float(np.mean(y)) if y.size else float("nan"),
        "std": float(np.std(y)) if y.size else float("nan"),
        "max": float(np.max(y)) if y.size else float("nan"),
    }


def _build_diagnostics(
        df: pd.DataFrame,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        config: PDBbindSplitConfig,
        dropped_nan: int,
        bins: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
    receptor_column = config.receptor_column
    receptor_counts = None
    receptor_overlap = None
    if receptor_column and receptor_column in df.columns:
        receptors = df[receptor_column].astype(str).to_numpy()
        train_r = set(receptors[train_idx].tolist())
        val_r = set(receptors[val_idx].tolist())
        test_r = set(receptors[test_idx].tolist())
        receptor_counts = {
            "train": int(len(train_r)),
            "validation": int(len(val_r)),
            "test": int(len(test_r)),
        }
        receptor_overlap = {
            "train∩validation": int(len(train_r & val_r)),
            "train∩test": int(len(train_r & test_r)),
            "validation∩test": int(len(val_r & test_r)),
        }

    idx_used = np.arange(len(df), dtype=int)
    if dropped_nan:
        idx_used = idx_used[~np.isnan(df[config.target_column].to_numpy(dtype=float))]
    train_pos = np.searchsorted(idx_used, train_idx)
    val_pos = np.searchsorted(idx_used, val_idx)
    test_pos = np.searchsorted(idx_used, test_idx)

    return {
        "strategy": config.strategy,
        "random_seed": int(config.random_seed),
        "relaxed_split": bool(config.relaxed_split),
        "target_column": config.target_column,
        "receptor_column": receptor_column if receptor_column in df.columns else None,
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(y)),
        "n_rows_dropped_nan_target": int(dropped_nan),
        "fractions": {
            "train": float(config.train_size),
            "validation": float(config.validation_size),
            "test": float(config.test_size),
        },
        "splits": {
            "train": {"n_rows": int(len(train_idx)), "target": _target_stats(y[train_pos])},
            "validation": {"n_rows": int(len(val_idx)), "target": _target_stats(y[val_pos])},
            "test": {"n_rows": int(len(test_idx)), "target": _target_stats(y[test_pos])},
        },
        "affinity_bins": bins,
        "receptor_counts": receptor_counts,
        "receptor_overlap": receptor_overlap,
        "scaler_fit_rule": "fit_on_train_only",
    }


def _log_diagnostics(diagnostics: dict[str, Any]) -> None:
    LOGGER.debug(
        "PDBbind split strategy=%s seed=%s relaxed=%s rows=%s used=%s bins=%s",
        diagnostics.get("strategy"),
        diagnostics.get("random_seed"),
        diagnostics.get("relaxed_split"),
        diagnostics.get("n_rows_total"),
        diagnostics.get("n_rows_used"),
        (diagnostics.get("affinity_bins") or {}).get("used_bins"),
    )
    for split_name, summary in (diagnostics.get("splits") or {}).items():
        target = summary.get("target") or {}
        LOGGER.debug(
            "PDBbind %s: rows=%s target[min=%.3f mean=%.3f std=%.3f max=%.3f]",
            split_name,
            summary.get("n_rows"),
            target.get("min", float("nan")),
            target.get("mean", float("nan")),
            target.get("std", float("nan")),
            target.get("max", float("nan")),
        )
    receptor_counts = diagnostics.get("receptor_counts")
    if receptor_counts:
        LOGGER.debug("PDBbind receptors: train=%s validation=%s test=%s", receptor_counts["train"], receptor_counts["validation"], receptor_counts["test"])
    receptor_overlap = diagnostics.get("receptor_overlap")
    if receptor_overlap:
        LOGGER.debug(
            "PDBbind receptor overlap: train∩validation=%s train∩test=%s validation∩test=%s",
            receptor_overlap.get("train∩validation"),
            receptor_overlap.get("train∩test"),
            receptor_overlap.get("validation∩test"),
        )
    if diagnostics.get("n_rows_dropped_nan_target"):
        LOGGER.warning("PDBbind dropped rows with NaN targets: %s", diagnostics["n_rows_dropped_nan_target"])
    if diagnostics.get("relaxed_notes"):
        LOGGER.warning("PDBbind split relaxed notes: %s", diagnostics["relaxed_notes"][:10])


__all__ = [
    "PDBBIND_SPLIT_STRATEGIES",
    "PDBbindSplitConfig",
    "PDBbindSplitResult",
    "split_pdbbind_regression",
]

