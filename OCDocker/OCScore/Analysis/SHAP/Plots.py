#!/usr/bin/env python3

# Description
###############################################################################
'''
Reusable SHAP plotting utilities.

Usage:

from OCDocker.OCScore.Analysis.SHAP.Plots import save_shap_plot_suite
'''

# Imports
###############################################################################
from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import shap

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURE_FAMILIES: dict[str, list[str]] = {
    "PMI": ["ligand_PMI*"],
    "AUTOCORR2D": ["ligand_AUTOCORR2D_*"],
    "VSA/EState": [
        "ligand_EState_VSA*",
        "ligand_PEOE_VSA*",
        "ligand_SMR_VSA*",
        "ligand_SlogP_VSA*",
        "ligand_VSA_EState*",
    ],
    "shape/size/topology": [
        "ligand_NPR*",
        "ligand_Asphericity",
        "ligand_Eccentricity",
        "ligand_InertialShapeFactor",
        "ligand_RadiusOfGyration",
        "ligand_SpherocityIndex",
        "ligand_BertzCT",
        "ligand_MolWt",
        "ligand_ExactMolWt",
        "ligand_HeavyAtomMolWt",
        "ligand_HeavyAtomCount",
        "ligand_TPSA",
        "ligand_MolLogP",
        "ligand_MolMR",
        "ligand_RingCount",
        "ligand_Num*Ring*",
        "ligand_Num*Cycle*",
        "ligand_NumHAcceptors",
        "ligand_NumHDonors",
        "ligand_NumHeteroatoms",
        "ligand_NumValenceElectrons",
        "ligand_FractionCSP3",
    ],
    "fragments": ["ligand_fr_*"],
    "BCUT2D": ["ligand_BCUT2D_*"],
    "Chi/Kappa/topological indices": [
        "ligand_Chi*",
        "ligand_Kappa*",
        "ligand_BalabanJ",
        "ligand_HallKierAlpha",
    ],
    "partial-charge / scalar EState": [
        "ligand_*PartialCharge",
        "ligand_MaxEStateIndex",
        "ligand_MinEStateIndex",
        "ligand_MaxAbsEStateIndex",
        "ligand_MinAbsEStateIndex",
    ],
    "PLANTS": ["plants_*"],
    "Vina/Smina": ["vina_*", "smina_*"],
    "GNINA": ["gnina_*"],
    "ODDT": ["oddt_*"],
    "receptor": ["receptor_*"],
    "other ligand": ["ligand_*"],
    "other": ["*"],
}

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def _ensure_dir(path: Union[str, Path]) -> None:
    '''Ensure that a directory exists.

    Parameters
    ----------
    path : str | Path
        Directory path to create.
    '''

    path = Path(path)
    if str(path):
        path.mkdir(parents=True, exist_ok=True)


def _safe_policy_name(policy: str) -> str:
    '''Normalize a policy label for file names.

    Parameters
    ----------
    policy : str
        User-provided policy label.

    Returns
    -------
    str
        File-safe policy label.
    '''

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(policy).strip())
    return safe.strip("_") or "policy"


def _normalize_shap_values(shap_values: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    '''Normalize SHAP values to a two-dimensional array.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values with shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Float SHAP array.
    '''

    arr = shap_values.to_numpy(dtype=float) if isinstance(shap_values, pd.DataFrame) else np.asarray(shap_values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"SHAP values must be 2D. Got shape {arr.shape}.")
    return arr


def _feature_names_from_values(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
    ) -> list[str]:
    '''Resolve feature names from explicit input or SHAP DataFrame columns.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Explicit feature names.

    Returns
    -------
    list[str]
        Feature names.
    '''

    if feature_names is not None:
        return [str(name) for name in feature_names]
    if isinstance(shap_values, pd.DataFrame):
        return [str(name) for name in shap_values.columns]
    raise ValueError("feature_names must be provided when shap_values is not a DataFrame.")


def _validate_feature_width(shap_2d: np.ndarray, feature_names: Sequence[str]) -> None:
    '''Validate SHAP width against feature names.

    Parameters
    ----------
    shap_2d : np.ndarray
        SHAP values.
    feature_names : sequence[str]
        Feature names.
    '''

    if shap_2d.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP width and feature_names differ: {shap_2d.shape[1]} != {len(feature_names)}."
        )


def _coerce_feature_frame(
        feature_matrix: Union[str, Path, np.ndarray, pd.DataFrame],
        feature_names: Sequence[str],
        n_rows: int,
    ) -> pd.DataFrame:
    '''Load or normalize a feature matrix.

    Parameters
    ----------
    feature_matrix : str | Path | np.ndarray | pd.DataFrame
        Feature matrix source.
    feature_names : sequence[str]
        Feature names in model order.
    n_rows : int
        Number of SHAP rows to retain.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to SHAP rows and columns.
    '''

    if isinstance(feature_matrix, (str, Path)):
        frame = pd.read_csv(feature_matrix)
    elif isinstance(feature_matrix, pd.DataFrame):
        frame = feature_matrix.copy()
    else:
        frame = pd.DataFrame(np.asarray(feature_matrix), columns=list(feature_names))
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing SHAP features: {', '.join(missing[:5])}")
    return frame.loc[:, list(feature_names)].iloc[:n_rows].reset_index(drop=True)


def _load_table(path: Union[str, Path]) -> pd.DataFrame:
    '''Load a CSV table.

    Parameters
    ----------
    path : str | Path
        CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    '''

    return pd.read_csv(path)


def _read_shap_values(path: Union[str, Path]) -> Union[np.ndarray, pd.DataFrame]:
    '''Read SHAP values from CSV or NPY.

    Parameters
    ----------
    path : str | Path
        SHAP values path.

    Returns
    -------
    np.ndarray | pd.DataFrame
        Loaded SHAP values.
    '''

    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return pd.read_csv(path)


def _read_feature_names(path: Optional[Union[str, Path]], shap_values: Union[np.ndarray, pd.DataFrame]) -> Optional[list[str]]:
    '''Read feature names from a one-column file or SHAP DataFrame columns.

    Parameters
    ----------
    path : str | Path | None
        Feature-name file.
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.

    Returns
    -------
    list[str] | None
        Feature names when available.
    '''

    if path is None:
        if isinstance(shap_values, pd.DataFrame):
            return [str(name) for name in shap_values.columns]
        return None
    path = Path(path)
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return [str(value) for value in loaded]
    table = pd.read_csv(path)
    if table.shape[1] == 1:
        return [str(value) for value in table.iloc[:, 0].tolist()]
    if "feature" in table.columns:
        return [str(value) for value in table["feature"].tolist()]
    return [str(value) for value in table.columns]


def _plot_height(n_rows: int, minimum: float = 4.5, row_height: float = 0.32) -> float:
    '''Return a readable plot height for row-oriented plots.

    Parameters
    ----------
    n_rows : int
        Number of visible rows.
    minimum : float, optional
        Minimum height.
    row_height : float, optional
        Height per row.

    Returns
    -------
    float
        Plot height.
    '''

    return max(minimum, min(18.0, 1.6 + row_height * max(1, int(n_rows))))


def _ordered_top(table: pd.DataFrame, value_col: str, top_n: Optional[int]) -> pd.DataFrame:
    '''Return top rows sorted by value.

    Parameters
    ----------
    table : pd.DataFrame
        Input table.
    value_col : str
        Value column.
    top_n : int | None
        Number of rows to keep.

    Returns
    -------
    pd.DataFrame
        Sorted top rows.
    '''

    ordered = table.sort_values(value_col, ascending=False).reset_index(drop=True)
    if top_n is None:
        return ordered
    return ordered.head(max(0, int(top_n))).copy()


def _family_matches(feature: str, patterns: Sequence[str]) -> bool:
    '''Return whether a feature matches any family pattern.

    Parameters
    ----------
    feature : str
        Feature name.
    patterns : sequence[str]
        Shell-style wildcard patterns.

    Returns
    -------
    bool
        True when matched.
    '''

    return any(fnmatch.fnmatchcase(feature, pattern) for pattern in patterns)


def _family_spec_from_loaded(loaded: Any) -> dict[str, list[str]]:
    '''Normalize loaded family configuration.

    Parameters
    ----------
    loaded : Any
        Loaded JSON/YAML content.

    Returns
    -------
    dict[str, list[str]]
        Family specification.
    '''

    if isinstance(loaded, Mapping) and "families" in loaded:
        loaded = loaded["families"]
    if not isinstance(loaded, Mapping):
        raise ValueError("Family specification must be a mapping or contain a 'families' mapping.")
    normalized: dict[str, list[str]] = {}
    for name, patterns in loaded.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, Sequence):
            raise ValueError(f"Family {name!r} must contain a sequence of patterns.")
        normalized[str(name)] = [str(pattern) for pattern in patterns]
    return normalized


def _write_csv(table: pd.DataFrame, path: Union[str, Path]) -> str:
    '''Write a CSV table and return its path.

    Parameters
    ----------
    table : pd.DataFrame
        Table to write.
    path : str | Path
        Output path.

    Returns
    -------
    str
        Output path.
    '''

    path = Path(path)
    _ensure_dir(path.parent)
    table.to_csv(path, index=False)
    return str(path)


def _update_policy_csv(table: pd.DataFrame, path: Union[str, Path], policy: str) -> str:
    '''Write or update a cross-policy CSV table.

    Parameters
    ----------
    table : pd.DataFrame
        Table containing a ``policy`` column.
    path : str | Path
        Output CSV path.
    policy : str
        Policy label to replace if already present.

    Returns
    -------
    str
        Output path.
    '''

    path = Path(path)
    _ensure_dir(path.parent)
    if path.exists():
        existing = pd.read_csv(path)
        if "policy" in existing.columns:
            existing = existing[existing["policy"].astype(str) != str(policy)]
        combined = pd.concat([existing, table], ignore_index=True)
    else:
        combined = table.copy()
    combined.to_csv(path, index=False)
    return str(path)


def _plot_importance_bar(
        table: pd.DataFrame,
        value_col: str,
        label_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        out_png: Union[str, Path],
        color: str,
        dpi: int,
        figsize: Optional[Tuple[float, float]] = None,
        log_x: bool = False,
    ) -> str:
    '''Save a horizontal importance bar plot.

    Parameters
    ----------
    table : pd.DataFrame
        Importance table.
    value_col : str
        Numeric value column.
    label_col : str
        Label column.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    out_png : str | Path
        Output PNG path.
    color : str
        Bar color.
    dpi : int
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.
    log_x : bool, optional
        Use logarithmic x-axis.

    Returns
    -------
    str
        Output PNG path.
    '''

    if table.empty:
        LOGGER.warning("No rows available for SHAP importance plot: %s", out_png)
        return str(out_png)
    fig_height = _plot_height(len(table), minimum=5.0, row_height=0.36) if figsize is None else figsize[1]
    fig_width = 10.0 if figsize is None else figsize[0]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.barplot(
        data=table,
        y=label_col,
        x=value_col,
        color=color,
        ax=ax,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_png)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _filter_log_importance_rows(
        table: pd.DataFrame,
        value_col: str,
        filter_zero_rows: bool,
    ) -> pd.DataFrame:
    '''Prepare importance rows for log-scale plotting.

    Parameters
    ----------
    table : pd.DataFrame
        Importance table.
    value_col : str
        Numeric value column.
    filter_zero_rows : bool
        Remove non-positive rows when True.

    Returns
    -------
    pd.DataFrame
        Log-plot rows.
    '''

    if filter_zero_rows:
        return table[table[value_col] > 0].copy()
    log_table = table.copy()
    positive = log_table[value_col] > 0
    if not positive.any():
        return log_table.iloc[0:0].copy()
    floor = float(log_table.loc[positive, value_col].min()) * 0.5
    log_table.loc[~positive, value_col] = floor
    return log_table


def _prepare_log_heatmap_matrix(
        matrix: pd.DataFrame,
        filter_zero_rows: bool,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    '''Prepare a heatmap matrix for log-color plotting.

    Parameters
    ----------
    matrix : pd.DataFrame
        Linear heatmap matrix.
    filter_zero_rows : bool
        Remove all-zero rows and columns when True. When False, non-positive
        cells are plotted with a small positive floor.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        Matrix to plot and optional mask.
    '''

    log_matrix = matrix.copy()
    if filter_zero_rows:
        log_matrix = log_matrix.loc[(log_matrix > 0).any(axis=1), (log_matrix > 0).any(axis=0)]
        if log_matrix.empty:
            return log_matrix, None
        return log_matrix, log_matrix <= 0
    positive = log_matrix > 0
    if not positive.any().any():
        return log_matrix.iloc[0:0, 0:0], None
    floor = float(log_matrix.where(positive).min().min()) * 0.5
    log_matrix = log_matrix.mask(~positive, floor)
    return log_matrix, None


def _plot_target_family_heatmap(
        matrix: pd.DataFrame,
        out_png: Union[str, Path],
        dpi: int,
        figsize: Optional[Tuple[float, float]] = None,
        log_color: bool = False,
        mask: Optional[pd.DataFrame] = None,
    ) -> str:
    '''Save a target-family heatmap.

    Parameters
    ----------
    matrix : pd.DataFrame
        Heatmap matrix.
    out_png : str | Path
        Output PNG path.
    dpi : int
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.
    log_color : bool, optional
        Use a logarithmic color scale when True.
    mask : pd.DataFrame | None, optional
        Heatmap mask.

    Returns
    -------
    str
        Output PNG path.
    '''

    out_path = Path(out_png)
    if matrix.empty:
        LOGGER.warning("No rows available for SHAP target-family heatmap: %s", out_path)
        return str(out_path)
    fig_width = max(8.0, min(24.0, 0.28 * max(1, matrix.shape[1]) + 4.0)) if figsize is None else figsize[0]
    fig_height = max(5.0, min(18.0, 0.35 * max(1, matrix.shape[0]) + 2.0)) if figsize is None else figsize[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap_kwargs: dict[str, Any] = {"cmap": "YlOrRd", "ax": ax}
    if mask is not None:
        heatmap_kwargs["mask"] = mask
    if log_color:
        positive_values = matrix.to_numpy(dtype=float)
        positive_values = positive_values[positive_values > 0]
        if positive_values.size == 0:
            plt.close(fig)
            LOGGER.warning("Skipping log-color SHAP target-family heatmap because no positive values were available.")
            return str(out_path)
        heatmap_kwargs["norm"] = LogNorm(
            vmin=float(np.nanmin(positive_values)),
            vmax=float(np.nanmax(positive_values)),
        )
    sns.heatmap(matrix, **heatmap_kwargs)
    title = "Per-Target SHAP Family Importance"
    if log_color:
        title = f"{title} (log color scale)"
    ax.set_title(title)
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


## Public ##


def load_family_spec(family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None) -> dict[str, list[str]]:
    '''Load a feature-family specification.

    Parameters
    ----------
    family_spec : str | Path | mapping | None, optional
        Family specification as a dict, JSON path, YAML path, or None for
        suggested defaults.

    Returns
    -------
    dict[str, list[str]]
        Family names mapped to shell-style wildcard patterns.
    '''

    if family_spec is None:
        return {name: list(patterns) for name, patterns in DEFAULT_FEATURE_FAMILIES.items()}
    if isinstance(family_spec, Mapping):
        return _family_spec_from_loaded(family_spec)
    path = Path(family_spec)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML SHAP family specifications.")
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return _family_spec_from_loaded(loaded)


def assign_feature_families(
        feature_names: Sequence[str],
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    ) -> pd.DataFrame:
    '''Assign features to configurable families.

    Parameters
    ----------
    feature_names : sequence[str]
        Feature names.
    family_spec : str | Path | mapping | None, optional
        Family specification.

    Returns
    -------
    pd.DataFrame
        Table with ``feature`` and ``family`` columns.
    '''

    spec = load_family_spec(family_spec)
    rows: list[dict[str, str]] = []
    for feature in feature_names:
        assigned = None
        for family, patterns in spec.items():
            if _family_matches(str(feature), patterns):
                assigned = family
                break
        rows.append({"feature": str(feature), "family": assigned or "other"})
    return pd.DataFrame(rows)


def compute_feature_importance_table(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
    '''Compute global SHAP feature importance.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None, optional
        Feature names.

    Returns
    -------
    pd.DataFrame
        Ranked feature-importance table.
    '''

    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    mean_abs = np.mean(np.abs(shap_2d), axis=0)
    total = float(mean_abs.sum())
    relative = np.zeros_like(mean_abs) if total <= 0 else (mean_abs / total) * 100.0
    table = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": mean_abs,
            "relative_importance_pct": relative,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "feature", "mean_abs_shap", "relative_importance_pct"]].reset_index(drop=True)


def save_global_feature_importance_plot(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        output_dir: Union[str, Path],
        policy: str,
        top_n: int = 20,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
        include_log_plot: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> dict[str, str]:
    '''Save a global SHAP feature-importance plot and CSV.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    output_dir : str | Path
        Output directory.
    policy : str
        File-name policy prefix.
    top_n : int, optional
        Number of visible features.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.
    include_log_plot : bool, optional
        Save a log-scale companion plot when True.
    filter_zero_rows_log : bool, optional
        Remove zero rows from log-scale plots when True. When False, zero rows
        are plotted with a small positive floor.

    Returns
    -------
    dict[str, str]
        Output artifact paths.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    table = compute_feature_importance_table(shap_values, feature_names)
    top_table = _ordered_top(table, "mean_abs_shap", top_n)
    csv_path = _write_csv(top_table, output_path / f"{prefix}_shap_top_features.csv")

    png_path = output_path / f"{prefix}_shap_feature_importance.png"
    _plot_importance_bar(
        top_table,
        "relative_importance_pct",
        "feature",
        "Global SHAP Feature Importance",
        "Relative importance (%)",
        "Feature",
        png_path,
        "#2f7ebc",
        dpi,
        figsize=figsize,
    )
    artifacts = {"feature_importance_png": str(png_path), "top_features_csv": csv_path}
    if include_log_plot:
        log_table = _filter_log_importance_rows(
            top_table,
            "relative_importance_pct",
            filter_zero_rows_log,
        )
        if log_table.empty:
            LOGGER.warning("Skipping log-scale SHAP feature importance plot because no positive values were available.")
        else:
            log_png_path = output_path / f"{prefix}_shap_feature_importance_logx.png"
            artifacts["feature_importance_logx_png"] = _plot_importance_bar(
                log_table,
                "relative_importance_pct",
                "feature",
                "Global SHAP Feature Importance (log x-axis)",
                "Relative importance (%)",
                "Feature",
                log_png_path,
                "#2f7ebc",
                dpi,
                figsize=figsize,
                log_x=True,
            )
    return artifacts


def save_beeswarm_plot(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_matrix: Union[str, Path, np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        output_dir: Union[str, Path],
        policy: str,
        top_n: int = 20,
        dpi: int = 300,
        figsize: Tuple[float, float] = (10.0, 7.0),
        rng_seed: Optional[int] = 0,
    ) -> dict[str, str]:
    '''Save a SHAP beeswarm plot.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_matrix : str | Path | np.ndarray | pd.DataFrame
        Feature matrix.
    feature_names : sequence[str] | None
        Feature names.
    output_dir : str | Path
        Output directory.
    policy : str
        File-name policy prefix.
    top_n : int, optional
        Number of visible features.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float], optional
        Figure size.
    rng_seed : int | None, optional
        Optional local random seed for SHAP versions that support it.

    Returns
    -------
    dict[str, str]
        Output artifact paths.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    X_eval = _coerce_feature_frame(feature_matrix, names, shap_2d.shape[0])
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    try:
        shap.summary_plot(
            shap_2d,
            X_eval,
            feature_names=names,
            max_display=int(top_n),
            show=False,
            plot_size=figsize,
            rng=rng,
        )
    except TypeError:
        shap.summary_plot(
            shap_2d,
            X_eval,
            feature_names=names,
            max_display=int(top_n),
            show=False,
            plot_size=figsize,
        )
    png_path = output_path / f"{prefix}_shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return {"beeswarm_png": str(png_path)}


def save_dependence_plots(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_matrix: Union[str, Path, np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        requested_features: Sequence[str],
        output_dir: Union[str, Path],
        policy: str,
        dpi: int = 300,
        figsize: Tuple[float, float] = (7.0, 5.0),
    ) -> dict[str, Any]:
    '''Save SHAP dependence plots for requested features.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_matrix : str | Path | np.ndarray | pd.DataFrame
        Feature matrix.
    feature_names : sequence[str] | None
        Feature names.
    requested_features : sequence[str]
        Features to plot.
    output_dir : str | Path
        Output directory.
    policy : str
        File-name policy prefix.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float], optional
        Figure size.

    Returns
    -------
    dict[str, Any]
        Written dependence plots and skipped features.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    X_eval = _coerce_feature_frame(feature_matrix, names, shap_2d.shape[0])
    available = set(names)
    written: dict[str, str] = {}
    skipped: list[str] = []
    for feature in requested_features:
        if feature not in available:
            LOGGER.warning("Skipping missing SHAP dependence feature: %s", feature)
            skipped.append(str(feature))
            continue
        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature,
            shap_2d,
            X_eval,
            feature_names=names,
            show=False,
        )
        feature_safe = _safe_policy_name(str(feature))
        png_path = output_path / f"{prefix}_shap_dependence_{feature_safe}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        written[str(feature)] = str(png_path)
    return {"dependence_pngs": written, "skipped_features": skipped}


def compute_family_importance_table(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        policy: Optional[str] = None,
    ) -> pd.DataFrame:
    '''Compute SHAP importance aggregated by feature family.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None, optional
        Feature names.
    family_spec : str | Path | mapping | None, optional
        Family specification.
    policy : str | None, optional
        Optional policy label for cross-policy aggregation.

    Returns
    -------
    pd.DataFrame
        Family-importance table.
    '''

    feature_table = compute_feature_importance_table(shap_values, feature_names)
    family_map = assign_feature_families(feature_table["feature"].tolist(), family_spec)
    merged = feature_table.merge(family_map, on="feature", how="left")
    rows = (
        merged.groupby("family", dropna=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            relative_importance_pct=("relative_importance_pct", "sum"),
            n_features=("feature", "count"),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    rows["rank"] = np.arange(1, len(rows) + 1)
    if policy is not None:
        rows.insert(0, "policy", str(policy))
    return rows


def save_family_importance_plot(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        output_dir: Union[str, Path],
        policy: str,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
        include_log_plot: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> dict[str, str]:
    '''Save feature-family SHAP aggregation plot and CSV files.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    output_dir : str | Path
        Output directory.
    policy : str
        Policy label.
    family_spec : str | Path | mapping | None, optional
        Family specification.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.
    include_log_plot : bool, optional
        Save a log-scale companion plot when True.
    filter_zero_rows_log : bool, optional
        Remove zero rows from log-scale plots when True. When False, zero rows
        are plotted with a small positive floor.

    Returns
    -------
    dict[str, str]
        Output artifact paths.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    table = compute_family_importance_table(shap_values, feature_names, family_spec, policy=None)
    table_with_policy = compute_family_importance_table(shap_values, feature_names, family_spec, policy=policy)
    csv_path = _write_csv(table, output_path / f"{prefix}_shap_family_importance.csv")
    all_policy_csv = _update_policy_csv(
        table_with_policy,
        output_path / "shap_family_importance_all_policies.csv",
        policy,
    )

    nonempty = table[table["n_features"] > 0].copy()
    if nonempty.empty:
        LOGGER.warning("No non-empty SHAP families were available for plotting.")
        return {
            "family_importance_csv": csv_path,
            "family_importance_all_policies_csv": all_policy_csv,
        }
    png_path = output_path / f"{prefix}_shap_family_importance.png"
    _plot_importance_bar(
        nonempty,
        "relative_importance_pct",
        "family",
        "Feature-Family SHAP Importance",
        "Relative importance (%)",
        "Feature family",
        png_path,
        "#3c8d5a",
        dpi,
        figsize=figsize,
    )
    artifacts = {
        "family_importance_png": str(png_path),
        "family_importance_csv": csv_path,
        "family_importance_all_policies_csv": all_policy_csv,
    }
    if include_log_plot:
        log_table = _filter_log_importance_rows(
            nonempty,
            "relative_importance_pct",
            filter_zero_rows_log,
        )
        if log_table.empty:
            LOGGER.warning("Skipping log-scale SHAP family importance plot because no positive values were available.")
        else:
            log_png_path = output_path / f"{prefix}_shap_family_importance_logx.png"
            artifacts["family_importance_logx_png"] = _plot_importance_bar(
                log_table,
                "relative_importance_pct",
                "family",
                "Feature-Family SHAP Importance (log x-axis)",
                "Relative importance (%)",
                "Feature family",
                log_png_path,
                "#3c8d5a",
                dpi,
                figsize=figsize,
                log_x=True,
            )
    return artifacts


def compute_target_family_shap_table(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        sample_metadata: Union[str, Path, pd.DataFrame],
        target_column: str,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    ) -> pd.DataFrame:
    '''Compute per-target mean absolute SHAP by feature family.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    sample_metadata : str | Path | pd.DataFrame
        Sample metadata.
    target_column : str
        Metadata column containing target IDs.
    family_spec : str | Path | mapping | None, optional
        Family specification.

    Returns
    -------
    pd.DataFrame
        Long-form target-family table.
    '''

    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    metadata = _load_table(sample_metadata) if isinstance(sample_metadata, (str, Path)) else sample_metadata.copy()
    if target_column not in metadata.columns:
        raise ValueError(f"sample_metadata is missing target column {target_column!r}.")
    metadata = metadata.iloc[: shap_2d.shape[0]].reset_index(drop=True)
    families = assign_feature_families(names, family_spec)
    family_names = list(dict.fromkeys(families["family"].tolist()))
    rows: list[dict[str, Any]] = []
    abs_values = np.abs(shap_2d)
    for target, row_indices in metadata.groupby(target_column, dropna=False).indices.items():
        for family in family_names:
            feature_indices = families.index[families["family"] == family].to_numpy()
            if feature_indices.size == 0:
                LOGGER.warning("Skipping empty SHAP family for target heatmap: %s", family)
                continue
            values = abs_values[np.asarray(row_indices), :][:, feature_indices]
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "mean_abs_shap": float(np.nanmean(values)),
                    "n_samples": int(len(row_indices)),
                    "n_features": int(feature_indices.size),
                }
            )
    return pd.DataFrame(rows)


def save_target_family_heatmap(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        sample_metadata: Union[str, Path, pd.DataFrame],
        target_column: str,
        output_dir: Union[str, Path],
        policy: str,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
        include_log_plot: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> dict[str, str]:
    '''Save a per-target SHAP family heatmap and CSV.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    sample_metadata : str | Path | pd.DataFrame
        Sample metadata.
    target_column : str
        Metadata column containing target IDs.
    output_dir : str | Path
        Output directory.
    policy : str
        File-name policy prefix.
    family_spec : str | Path | mapping | None, optional
        Family specification.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.
    include_log_plot : bool, optional
        Save a log-color companion heatmap when True.
    filter_zero_rows_log : bool, optional
        Remove all-zero rows and columns from log-color heatmaps when True.
        When False, zero cells are plotted with a small positive floor.

    Returns
    -------
    dict[str, str]
        Output artifact paths.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    table = compute_target_family_shap_table(
        shap_values,
        feature_names,
        sample_metadata,
        target_column,
        family_spec,
    )
    csv_path = _write_csv(table, output_path / f"{prefix}_target_family_shap_heatmap.csv")
    if table.empty:
        LOGGER.warning("No target-family SHAP values were available for heatmap plotting.")
        return {"target_family_heatmap_csv": csv_path}
    matrix = table.pivot_table(
        index="family",
        columns="target",
        values="mean_abs_shap",
        aggfunc="mean",
        fill_value=0.0,
    )
    png_path = output_path / f"{prefix}_target_family_shap_heatmap.png"
    artifacts = {
        "target_family_heatmap_png": _plot_target_family_heatmap(
            matrix,
            png_path,
            dpi,
            figsize=figsize,
        ),
        "target_family_heatmap_csv": csv_path,
    }
    if include_log_plot:
        log_matrix, mask = _prepare_log_heatmap_matrix(matrix, filter_zero_rows_log)
        if log_matrix.empty:
            LOGGER.warning("Skipping log-color SHAP target-family heatmap because no positive values were available.")
        else:
            log_png_path = output_path / f"{prefix}_target_family_shap_heatmap_logcolor.png"
            artifacts["target_family_heatmap_logcolor_png"] = _plot_target_family_heatmap(
                log_matrix,
                log_png_path,
                dpi,
                figsize=figsize,
                log_color=True,
                mask=mask,
            )
    return artifacts


def compute_label_family_distribution_table(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        labels: Union[str, Path, Sequence[Any], pd.Series, pd.DataFrame],
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        label_column: Optional[str] = None,
    ) -> pd.DataFrame:
    '''Compute sample-level SHAP family scores grouped by labels.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    labels : str | Path | sequence | pd.Series | pd.DataFrame
        Sample labels.
    family_spec : str | Path | mapping | None, optional
        Family specification.
    label_column : str | None, optional
        Label column when labels are provided as a table.

    Returns
    -------
    pd.DataFrame
        Long-form sample-family table.
    '''

    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    if isinstance(labels, (str, Path)):
        label_frame = pd.read_csv(labels)
        column = label_column or label_frame.columns[0]
        label_values = label_frame[column].iloc[: shap_2d.shape[0]].reset_index(drop=True)
    elif isinstance(labels, pd.DataFrame):
        column = label_column or labels.columns[0]
        label_values = labels[column].iloc[: shap_2d.shape[0]].reset_index(drop=True)
    elif isinstance(labels, pd.Series):
        label_values = labels.iloc[: shap_2d.shape[0]].reset_index(drop=True)
    else:
        label_values = pd.Series(list(labels)[: shap_2d.shape[0]])
    families = assign_feature_families(names, family_spec)
    abs_values = np.abs(shap_2d)
    rows: list[dict[str, Any]] = []
    for sample_index, label in enumerate(label_values.tolist()):
        for family in dict.fromkeys(families["family"].tolist()):
            feature_indices = families.index[families["family"] == family].to_numpy()
            if feature_indices.size == 0:
                LOGGER.warning("Skipping empty SHAP family for label distribution: %s", family)
                continue
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "label": label,
                    "family": family,
                    "mean_abs_shap": float(np.nanmean(abs_values[sample_index, feature_indices])),
                    "sum_abs_shap": float(np.nansum(abs_values[sample_index, feature_indices])),
                    "n_features": int(feature_indices.size),
                }
            )
    return pd.DataFrame(rows)


def save_label_family_distribution_plot(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        labels: Union[str, Path, Sequence[Any], pd.Series, pd.DataFrame],
        output_dir: Union[str, Path],
        policy: str,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        label_column: Optional[str] = None,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> dict[str, str]:
    '''Save active-vs-decoy SHAP family distribution plot and CSV.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    labels : str | Path | sequence | pd.Series | pd.DataFrame
        Sample labels.
    output_dir : str | Path
        Output directory.
    policy : str
        File-name policy prefix.
    family_spec : str | Path | mapping | None, optional
        Family specification.
    label_column : str | None, optional
        Label column when labels are provided as a table.
    dpi : int, optional
        Figure DPI.
    figsize : tuple[float, float] | None, optional
        Figure size.

    Returns
    -------
    dict[str, str]
        Output artifact paths.
    '''

    output_path = Path(output_dir)
    _ensure_dir(output_path)
    prefix = _safe_policy_name(policy)
    table = compute_label_family_distribution_table(
        shap_values,
        feature_names,
        labels,
        family_spec,
        label_column=label_column,
    )
    csv_path = _write_csv(table, output_path / f"{prefix}_active_decoy_shap_family_distribution.csv")
    if table.empty:
        LOGGER.warning("No label-family SHAP values were available for distribution plotting.")
        return {"active_decoy_family_distribution_csv": csv_path}
    fig_height = _plot_height(table["family"].nunique(), minimum=5.0, row_height=0.4) if figsize is None else figsize[1]
    fig_width = 11.0 if figsize is None else figsize[0]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    families = list(dict.fromkeys(table["family"].astype(str).tolist()))
    labels_order = list(dict.fromkeys(table["label"].astype(str).tolist()))
    palette = sns.color_palette("Set2", n_colors=max(1, len(labels_order)))
    offsets = np.linspace(-0.22, 0.22, max(1, len(labels_order)))
    box_width = 0.34 / max(1, len(labels_order))
    for label_index, label in enumerate(labels_order):
        values_by_family = []
        positions = []
        for family_index, family in enumerate(families):
            values = table[
                (table["family"].astype(str) == family)
                & (table["label"].astype(str) == label)
            ]["mean_abs_shap"].to_numpy(dtype=float)
            values_by_family.append(values if values.size else np.array([np.nan]))
            positions.append(family_index + offsets[label_index])
        parts = ax.boxplot(
            values_by_family,
            positions=positions,
            orientation="horizontal",
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": palette[label_index],
                "markeredgecolor": palette[label_index],
                "alpha": 0.65,
            },
        )
        for patch in parts["boxes"]:
            patch.set_facecolor(palette[label_index])
            patch.set_alpha(0.42)
            patch.set_edgecolor(palette[label_index])
        for key in ("whiskers", "caps", "medians"):
            for artist in parts[key]:
                artist.set_color(palette[label_index])
        ax.scatter([], [], color=palette[label_index], label=str(label))
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    ax.set_title("Active-vs-Decoy SHAP Family Distribution")
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_ylabel("Feature family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Label", frameon=False)
    fig.tight_layout()
    png_path = output_path / f"{prefix}_active_decoy_shap_family_distribution.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "active_decoy_family_distribution_png": str(png_path),
        "active_decoy_family_distribution_csv": csv_path,
    }


def save_shap_plot_suite(
        shap_values: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]],
        output_dir: Union[str, Path],
        policy: str = "policy",
        feature_matrix: Optional[Union[str, Path, np.ndarray, pd.DataFrame]] = None,
        dependence_features: Optional[Sequence[str]] = None,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        sample_metadata: Optional[Union[str, Path, pd.DataFrame]] = None,
        target_column: Optional[str] = None,
        labels: Optional[Union[str, Path, Sequence[Any], pd.Series, pd.DataFrame]] = None,
        label_column: Optional[str] = None,
        top_n: int = 20,
        dpi: int = 300,
        rng_seed: Optional[int] = 0,
        include_log_importance_plots: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> dict[str, Any]:
    '''Save reusable SHAP plots for a policy.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    feature_names : sequence[str] | None
        Feature names.
    output_dir : str | Path
        Output directory.
    policy : str, optional
        File-name policy prefix.
    feature_matrix : str | Path | np.ndarray | pd.DataFrame | None, optional
        Feature matrix for beeswarm and dependence plots.
    dependence_features : sequence[str] | None, optional
        Features for dependence plots.
    family_spec : str | Path | mapping | None, optional
        Feature-family specification.
    sample_metadata : str | Path | pd.DataFrame | None, optional
        Sample metadata for target-family heatmap.
    target_column : str | None, optional
        Metadata target column.
    labels : str | Path | sequence | pd.Series | pd.DataFrame | None, optional
        Labels for active-vs-decoy distribution.
    label_column : str | None, optional
        Label column for table labels.
    top_n : int, optional
        Number of visible features.
    dpi : int, optional
        Figure DPI.
    rng_seed : int | None, optional
        Optional local random seed for SHAP versions that support it.
    include_log_importance_plots : bool, optional
        Save log-scale feature and family importance companion plots.
    filter_zero_rows_log : bool, optional
        Remove zero rows from log-scale plots when True. When False, zero rows
        are plotted with a small positive floor.

    Returns
    -------
    dict[str, Any]
        Output artifact paths.
    '''

    artifacts: dict[str, Any] = {}
    artifacts.update(
        save_global_feature_importance_plot(
            shap_values,
            feature_names,
            output_dir,
            policy,
            top_n=top_n,
            dpi=dpi,
            include_log_plot=include_log_importance_plots,
            filter_zero_rows_log=filter_zero_rows_log,
        )
    )
    artifacts.update(
        save_family_importance_plot(
            shap_values,
            feature_names,
            output_dir,
            policy,
            family_spec=family_spec,
            dpi=dpi,
            include_log_plot=include_log_importance_plots,
            filter_zero_rows_log=filter_zero_rows_log,
        )
    )
    if feature_matrix is not None:
        artifacts.update(
            save_beeswarm_plot(
                shap_values,
                feature_matrix,
                feature_names,
                output_dir,
                policy,
                top_n=top_n,
                dpi=dpi,
                rng_seed=rng_seed,
            )
        )
        if dependence_features:
            artifacts.update(
                save_dependence_plots(
                    shap_values,
                    feature_matrix,
                    feature_names,
                    dependence_features,
                    output_dir,
                    policy,
                    dpi=dpi,
                )
            )
    if sample_metadata is not None and target_column:
        artifacts.update(
            save_target_family_heatmap(
                shap_values,
                feature_names,
                sample_metadata,
                target_column,
                output_dir,
                policy,
                family_spec=family_spec,
                dpi=dpi,
                include_log_plot=include_log_importance_plots,
                filter_zero_rows_log=filter_zero_rows_log,
            )
        )
    elif sample_metadata is not None:
        LOGGER.warning("Skipping target-family heatmap because target_column was not provided.")
    if labels is not None:
        artifacts.update(
            save_label_family_distribution_plot(
                shap_values,
                feature_names,
                labels,
                output_dir,
                policy,
                family_spec=family_spec,
                label_column=label_column,
                dpi=dpi,
            )
        )
    return artifacts


def save_shap_plot_suite_from_paths(
        shap_values_path: Union[str, Path],
        output_dir: Union[str, Path],
        policy: str = "policy",
        feature_names_path: Optional[Union[str, Path]] = None,
        feature_matrix_path: Optional[Union[str, Path]] = None,
        dependence_features: Optional[Sequence[str]] = None,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        sample_metadata_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
        labels_path: Optional[Union[str, Path]] = None,
        label_column: Optional[str] = None,
        top_n: int = 20,
        dpi: int = 300,
        rng_seed: Optional[int] = 0,
        include_log_importance_plots: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> dict[str, Any]:
    '''Save reusable SHAP plots from explicit input paths.

    Parameters
    ----------
    shap_values_path : str | Path
        SHAP values CSV or NPY path.
    output_dir : str | Path
        Output directory.
    policy : str, optional
        File-name policy prefix.
    feature_names_path : str | Path | None, optional
        Feature-name source for NPY SHAP values.
    feature_matrix_path : str | Path | None, optional
        Feature matrix CSV path.
    dependence_features : sequence[str] | None, optional
        Features for dependence plots.
    family_spec : str | Path | mapping | None, optional
        Feature-family specification.
    sample_metadata_path : str | Path | None, optional
        Sample metadata CSV path.
    target_column : str | None, optional
        Target column in metadata.
    labels_path : str | Path | None, optional
        Label CSV path.
    label_column : str | None, optional
        Label column.
    top_n : int, optional
        Number of visible features.
    dpi : int, optional
        Figure DPI.
    rng_seed : int | None, optional
        Optional local random seed for SHAP versions that support it.
    include_log_importance_plots : bool, optional
        Save log-scale feature and family importance companion plots.
    filter_zero_rows_log : bool, optional
        Remove zero rows from log-scale plots when True. When False, zero rows
        are plotted with a small positive floor.

    Returns
    -------
    dict[str, Any]
        Output artifact paths.
    '''

    shap_values = _read_shap_values(shap_values_path)
    feature_names = _read_feature_names(feature_names_path, shap_values)
    return save_shap_plot_suite(
        shap_values,
        feature_names,
        output_dir,
        policy=policy,
        feature_matrix=feature_matrix_path,
        dependence_features=dependence_features,
        family_spec=family_spec,
        sample_metadata=sample_metadata_path,
        target_column=target_column,
        labels=labels_path,
        label_column=label_column,
        top_n=top_n,
        dpi=dpi,
        rng_seed=rng_seed,
        include_log_importance_plots=include_log_importance_plots,
        filter_zero_rows_log=filter_zero_rows_log,
    )


def beeswarm(
        shap_2d: np.ndarray,
        X_eval: pd.DataFrame,
        out_png: str,
        figsize: Tuple[int, int] = (10, 6),
        rng_seed: Optional[int] = 0,
    ) -> str:
    '''Wrapper around SHAP beeswarm plotting.

    Parameters
    ----------
    shap_2d : np.ndarray
        SHAP values with shape ``(n_samples, n_features)``.
    X_eval : pd.DataFrame
        Evaluation features.
    out_png : str
        Output PNG path.
    figsize : tuple[int, int], optional
        Figure size.
    rng_seed : int | None, optional
        Optional local random seed.

    Returns
    -------
    str
        Output path.
    '''

    artifacts = save_beeswarm_plot(
        shap_2d,
        X_eval,
        list(X_eval.columns),
        Path(out_png).parent,
        Path(out_png).stem.replace("_shap_beeswarm_plot", "").replace("_shap_beeswarm", "") or "policy",
        top_n=min(20, shap_2d.shape[1]),
        figsize=figsize,
        rng_seed=rng_seed,
    )
    generated = Path(artifacts["beeswarm_png"])
    target = Path(out_png)
    if generated != target:
        generated.replace(target)
    return str(target)


def feature_importance_barh(
        shap_2d: np.ndarray,
        feature_names: Sequence[str],
        out_png: str,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
    '''Horizontal bar chart of relative SHAP importance per feature.

    Parameters
    ----------
    shap_2d : np.ndarray
        SHAP values.
    feature_names : sequence[str]
        Feature names.
    out_png : str
        Output PNG path.
    top_k : int, optional
        Number of top features.
    figsize : tuple[int, int], optional
        Figure size.

    Returns
    -------
    str
        Output path.
    '''

    artifacts = save_global_feature_importance_plot(
        shap_2d,
        feature_names,
        Path(out_png).parent,
        Path(out_png).stem.replace("_shap_feature_importance", "") or "policy",
        top_n=top_k,
        figsize=figsize,
    )
    generated = Path(artifacts["feature_importance_png"])
    target = Path(out_png)
    if generated != target:
        generated.replace(target)
    return str(target)


def shap_correlation_heatmap(
        shap_values: Union[np.ndarray, pd.DataFrame],
        out_png: str,
        feature_names: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> str:
    '''Plot a heatmap of SHAP value correlations across features.

    Parameters
    ----------
    shap_values : np.ndarray | pd.DataFrame
        SHAP values.
    out_png : str
        Output PNG path.
    feature_names : sequence[str] | None, optional
        Feature names.
    figsize : tuple[int, int], optional
        Figure size.

    Returns
    -------
    str
        Output path.
    '''

    _ensure_dir(Path(out_png).parent)
    names = _feature_names_from_values(shap_values, feature_names)
    shap_2d = _normalize_shap_values(shap_values)
    _validate_feature_width(shap_2d, names)
    df = pd.DataFrame(shap_2d, columns=names)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("SHAP value correlations")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


__all__ = [
    "DEFAULT_FEATURE_FAMILIES",
    "assign_feature_families",
    "beeswarm",
    "compute_feature_importance_table",
    "compute_family_importance_table",
    "compute_label_family_distribution_table",
    "compute_target_family_shap_table",
    "feature_importance_barh",
    "load_family_spec",
    "save_beeswarm_plot",
    "save_dependence_plots",
    "save_family_importance_plot",
    "save_global_feature_importance_plot",
    "save_label_family_distribution_plot",
    "save_shap_plot_suite",
    "save_shap_plot_suite_from_paths",
    "save_target_family_heatmap",
    "shap_correlation_heatmap",
]
