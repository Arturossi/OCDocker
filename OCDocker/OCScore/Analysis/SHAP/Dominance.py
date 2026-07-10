#!/usr/bin/env python3

# Description
###############################################################################
'''
Cross-replica SHAP aggregation for feature-ablation policies.

Aggregates per-replica SHAP exports (``export/replica_analysis/...``, written
by ``ExportRunner.run_export_shap_analysis`` for the reference model and for
each ablation policy) into per-policy summaries: family-level importance
composition averaged across replicas, and single-feature "shortcut" dominance
(mean/max share of total importance held by the single most important feature
across replicas).

Usage:

import OCDocker.OCScore.Analysis.SHAP.Dominance as ocshapdominance
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from OCDocker.OCScore.Analysis.SHAP.Plots import compute_family_importance_table
from OCDocker.OCScore.Analysis.SHAP.Plots import compute_feature_importance_table
from OCDocker.OCScore.Utils.FeaturePolicy import FULL_OCSCORE_POLICY_NAME

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
DEFAULT_TASK = "dudez"
DEFAULT_N_REPLICAS = 5
REPLICA_ANALYSIS_DIRNAME = "replica_analysis"

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def _replica_shap_dir(
        export_root: Union[str, Path],
        policy: str,
        replica_index: int,
        task: str,
        reference_policy_name: str,
    ) -> Path:
    '''Resolve a single replica's SHAP export directory for a policy.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree (containing ``replica_analysis``).
    policy : str
        Feature-policy name.
    replica_index : int
        Replica index (0-based).
    task : str
        Evaluation task subdirectory (``"dudez"`` or ``"pdbbind"``).
    reference_policy_name : str
        Policy name treated as the unablated reference model.

    Returns
    -------
    Path
        Replica SHAP directory (may not exist).
    '''

    root = Path(export_root) / REPLICA_ANALYSIS_DIRNAME
    replica_name = f"replica_{replica_index:03d}"
    if policy == reference_policy_name:
        return root / "full" / replica_name / task / "shap"
    return root / "ablations" / policy / replica_name / task / "shap"


def _replica_index_from_csv(csv_path: Path) -> int:
    '''Recover the replica index from a ``.../replica_XXX/<task>/shap/shap_values.csv`` path.

    Parameters
    ----------
    csv_path : Path
        Per-replica SHAP values CSV path.

    Returns
    -------
    int
        Replica index.
    '''

    replica_dir_name = csv_path.parents[2].name
    return int(replica_dir_name.rsplit("_", 1)[-1])


## Public ##

def discover_replica_shap_csvs(
        export_root: Union[str, Path],
        policy: str,
        *,
        task: str = DEFAULT_TASK,
        n_replicas: int = DEFAULT_N_REPLICAS,
        reference_policy_name: str = FULL_OCSCORE_POLICY_NAME,
    ) -> list[Path]:
    '''Resolve the per-replica ``shap_values.csv`` paths for a policy.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree (containing ``replica_analysis``).
    policy : str
        Feature-policy name.
    task : str, optional
        Evaluation task subdirectory, by default ``"dudez"``.
    n_replicas : int, optional
        Number of replicas to look for, by default 5.
    reference_policy_name : str, optional
        Policy name treated as the unablated reference model, by default ``"full_ocscore"``.

    Returns
    -------
    list[Path]
        Existing per-replica ``shap_values.csv`` paths, ordered by replica index.

    Raises
    ------
    FileNotFoundError
        If no per-replica SHAP export is found for the policy.
    '''

    candidates = [
        _replica_shap_dir(export_root, policy, replica_index, task, reference_policy_name) / "shap_values.csv"
        for replica_index in range(n_replicas)
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No per-replica SHAP export found for policy {policy!r} under "
            f"{Path(export_root) / REPLICA_ANALYSIS_DIRNAME!s}."
        )
    return existing


def compute_replica_family_importance(
        export_root: Union[str, Path],
        policy: str,
        *,
        task: str = DEFAULT_TASK,
        n_replicas: int = DEFAULT_N_REPLICAS,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        reference_policy_name: str = FULL_OCSCORE_POLICY_NAME,
    ) -> pd.DataFrame:
    '''Compute per-replica, family-level SHAP importance for one policy.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree.
    policy : str
        Feature-policy name.
    task : str, optional
        Evaluation task subdirectory, by default ``"dudez"``.
    n_replicas : int, optional
        Number of replicas to look for, by default 5.
    family_spec : str | Path | mapping | None, optional
        Family specification passed through to ``SHAP.Plots.compute_family_importance_table``.
    reference_policy_name : str, optional
        Policy name treated as the unablated reference model, by default ``"full_ocscore"``.

    Returns
    -------
    pd.DataFrame
        One row per (replica, family), with columns ``policy``, ``replica_index``,
        ``family``, ``mean_abs_shap``, ``relative_importance_pct``, ``n_features``, ``rank``.
    '''

    csv_paths = discover_replica_shap_csvs(
        export_root, policy, task=task, n_replicas=n_replicas, reference_policy_name=reference_policy_name,
    )
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        shap_df = pd.read_csv(csv_path)
        table = compute_family_importance_table(shap_df, family_spec=family_spec, policy=policy)
        table.insert(1, "replica_index", _replica_index_from_csv(csv_path))
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def compute_replica_dominant_feature(
        export_root: Union[str, Path],
        policy: str,
        *,
        task: str = DEFAULT_TASK,
        n_replicas: int = DEFAULT_N_REPLICAS,
        reference_policy_name: str = FULL_OCSCORE_POLICY_NAME,
    ) -> pd.DataFrame:
    '''Compute the single most important SHAP feature for each replica of one policy.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree.
    policy : str
        Feature-policy name.
    task : str, optional
        Evaluation task subdirectory, by default ``"dudez"``.
    n_replicas : int, optional
        Number of replicas to look for, by default 5.
    reference_policy_name : str, optional
        Policy name treated as the unablated reference model, by default ``"full_ocscore"``.

    Returns
    -------
    pd.DataFrame
        One row per replica, with columns ``policy``, ``replica_index``, ``feature``
        (the top-ranked feature), ``relative_importance_pct`` (its share of total importance).
    '''

    csv_paths = discover_replica_shap_csvs(
        export_root, policy, task=task, n_replicas=n_replicas, reference_policy_name=reference_policy_name,
    )
    rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        shap_df = pd.read_csv(csv_path)
        top = compute_feature_importance_table(shap_df).iloc[0]
        rows.append({
            "policy": policy,
            "replica_index": _replica_index_from_csv(csv_path),
            "feature": top["feature"],
            "relative_importance_pct": float(top["relative_importance_pct"]),
        })
    return pd.DataFrame(rows)


def aggregate_family_composition(
        export_root: Union[str, Path],
        policies: Sequence[str],
        *,
        task: str = DEFAULT_TASK,
        n_replicas: int = DEFAULT_N_REPLICAS,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        reference_policy_name: str = FULL_OCSCORE_POLICY_NAME,
    ) -> pd.DataFrame:
    '''Average per-family SHAP importance across replicas, for a set of policies.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree.
    policies : sequence[str]
        Feature-policy names to aggregate.
    task : str, optional
        Evaluation task subdirectory, by default ``"dudez"``.
    n_replicas : int, optional
        Number of replicas to look for, by default 5.
    family_spec : str | Path | mapping | None, optional
        Family specification passed through to ``SHAP.Plots.compute_family_importance_table``.
    reference_policy_name : str, optional
        Policy name treated as the unablated reference model, by default ``"full_ocscore"``.

    Returns
    -------
    pd.DataFrame
        One row per (policy, family), with columns ``policy``, ``family``,
        ``relative_importance_pct_mean``, ``n_replicas``.
    '''

    frames = [
        compute_replica_family_importance(
            export_root, policy, task=task, n_replicas=n_replicas,
            family_spec=family_spec, reference_policy_name=reference_policy_name,
        )
        for policy in policies
    ]
    long_df = pd.concat(frames, ignore_index=True)
    return (
        long_df.groupby(["policy", "family"], sort=False)["relative_importance_pct"]
        .agg(relative_importance_pct_mean="mean", n_replicas="count")
        .reset_index()
    )


def aggregate_dominant_feature_risk(
        export_root: Union[str, Path],
        policies: Sequence[str],
        *,
        task: str = DEFAULT_TASK,
        n_replicas: int = DEFAULT_N_REPLICAS,
        reference_policy_name: str = FULL_OCSCORE_POLICY_NAME,
    ) -> pd.DataFrame:
    '''Summarize single-feature "shortcut" dominance across replicas, for a set of policies.

    Parameters
    ----------
    export_root : str | Path
        Root of the pipeline's ``export`` output tree.
    policies : sequence[str]
        Feature-policy names to aggregate.
    task : str, optional
        Evaluation task subdirectory, by default ``"dudez"``.
    n_replicas : int, optional
        Number of replicas to look for, by default 5.
    reference_policy_name : str, optional
        Policy name treated as the unablated reference model, by default ``"full_ocscore"``.

    Returns
    -------
    pd.DataFrame
        One row per policy, with columns ``policy``, ``dominant_feature`` (most
        frequent top-1 feature across replicas), ``top1_pct_mean``, ``top1_pct_max``,
        ``n_replicas``.
    '''

    rows: list[dict[str, object]] = []
    for policy in policies:
        replica_df = compute_replica_dominant_feature(
            export_root, policy, task=task, n_replicas=n_replicas, reference_policy_name=reference_policy_name,
        )
        rows.append({
            "policy": policy,
            "dominant_feature": replica_df["feature"].mode().iloc[0],
            "top1_pct_mean": float(replica_df["relative_importance_pct"].mean()),
            "top1_pct_max": float(replica_df["relative_importance_pct"].max()),
            "n_replicas": int(len(replica_df)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "aggregate_dominant_feature_risk",
    "aggregate_family_composition",
    "compute_replica_dominant_feature",
    "compute_replica_family_importance",
    "discover_replica_shap_csvs",
]
