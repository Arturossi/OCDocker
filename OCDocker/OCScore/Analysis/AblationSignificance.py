#!/usr/bin/env python3

# Description
###############################################################################
'''Significance testing for feature-ablation BEDROC comparisons.

Replicas are paired across feature policies: ``ReplicatedProtocolConfig``
derives every replica seed as ``base_seed + replica_index`` (see
``OCScore.Optimization.Protocol``), and that seed controls the PDBbind/DUDEz
data split (``PDBbindSplitConfig.seed``). Replica ``i`` therefore evaluates
the same train/validation/test split for every policy, differing only in the
feature set - so policies are compared with a paired test on matched replica
seeds rather than an independent-samples test.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats as sstats

from statsmodels.stats.multitest import multipletests

from OCDocker.OCScore.Utils.FeaturePolicy import FULL_OCSCORE_POLICY_NAME

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

DEFAULT_METRIC_COLUMN = "dudez_test_bedroc"
DEFAULT_POLICY_COLUMN = "feature_policy_name"
DEFAULT_REPLICAS_CSV_COLUMN = "replicas_summary_csv"
DEFAULT_PAIRING_COLUMN = "seed"
PAIRED_TEST_METHODS = ("paired_ttest", "wilcoxon")


# Classes
###############################################################################

@dataclass(frozen=True)
class AblationSignificanceConfig:
    """Configuration for paired ablation-vs-reference significance testing.

    Parameters
    ----------
    reference_policy : str, optional
        Feature-policy name treated as the comparison baseline, by default ``"full_ocscore"``.
    metric_column : str, optional
        Per-replica metric column to compare, by default ``"dudez_test_bedroc"``.
    pairing_column : str, optional
        Column used to match replicas across policies, by default ``"seed"``.
    method : str, optional
        Paired test to run: ``"paired_ttest"`` (default) or ``"wilcoxon"``. With the typical
        ``n_replicas=5`` design, a two-sided Wilcoxon signed-rank test cannot reach p<0.05
        regardless of effect size (minimum exact p is 0.0625), so ``"paired_ttest"`` is the
        default; it assumes approximately normal paired differences.
    correction_method : str, optional
        Multiple-comparisons correction passed to ``statsmodels.stats.multitest.multipletests``,
        by default ``"holm"``.
    alpha : float, optional
        Family-wise significance threshold, by default 0.05.
    """

    reference_policy: str = FULL_OCSCORE_POLICY_NAME
    metric_column: str = DEFAULT_METRIC_COLUMN
    pairing_column: str = DEFAULT_PAIRING_COLUMN
    method: str = "paired_ttest"
    correction_method: str = "holm"
    alpha: float = 0.05


# Functions
###############################################################################
## Private ##

def _run_paired_test(reference_values: np.ndarray, policy_values: np.ndarray, *, method: str) -> tuple[float, float]:
    '''Run one paired test between matched reference and policy values.

    Parameters
    ----------
    reference_values : np.ndarray
        Reference-policy metric values, ordered to match ``policy_values`` pairwise.
    policy_values : np.ndarray
        Comparison-policy metric values, ordered to match ``reference_values`` pairwise.
    method : str
        Either ``"paired_ttest"`` or ``"wilcoxon"``.

    Returns
    -------
    tuple[float, float]
        ``(statistic, pvalue)``, both NaN if the test cannot be computed.

    Raises
    ------
    ValueError
        If ``method`` is not a supported paired test name.
    '''

    if method not in PAIRED_TEST_METHODS:
        raise ValueError(f"Unknown paired test method: {method!r}. Expected one of {PAIRED_TEST_METHODS}.")

    if method == "paired_ttest":
        result = sstats.ttest_rel(policy_values, reference_values)
        return float(result.statistic), float(result.pvalue)

    diffs = policy_values - reference_values
    if not np.any(diffs != 0):
        return float("nan"), float("nan")
    try:
        result = sstats.wilcoxon(diffs)
    except ValueError:
        # All-zero or otherwise degenerate difference vector
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


## Public ##

def load_ablation_bedroc_long(
        ablation_summary_csv: str | Path,
        *,
        metric_column: str = DEFAULT_METRIC_COLUMN,
        policy_column: str = DEFAULT_POLICY_COLUMN,
        replicas_csv_column: str = DEFAULT_REPLICAS_CSV_COLUMN,
        pairing_column: str = DEFAULT_PAIRING_COLUMN,
    ) -> pd.DataFrame:
    '''Load per-replica metric values for every policy listed in an ablation summary CSV.

    Parameters
    ----------
    ablation_summary_csv : str or pathlib.Path
        Path to an ``ablation_summary.csv`` (or equivalent) listing one row per feature
        policy, with a policy-name column and a path to that policy's ``replicas_summary.csv``.
    metric_column : str, optional
        Per-replica metric column to extract, by default ``"dudez_test_bedroc"``.
    policy_column : str, optional
        Policy-name column in ``ablation_summary_csv``, by default ``"feature_policy_name"``.
    replicas_csv_column : str, optional
        Column holding each policy's per-replica CSV path, by default ``"replicas_summary_csv"``.
    pairing_column : str, optional
        Column used to match replicas across policies, by default ``"seed"``.

    Returns
    -------
    pd.DataFrame
        Long-format frame with columns ``policy``, ``replica_index``, ``pairing_column``,
        and ``metric_column``.

    Raises
    ------
    FileNotFoundError
        If ``ablation_summary_csv`` or a referenced per-policy CSV does not exist.
    ValueError
        If required columns are missing from either CSV.
    '''

    summary_path = Path(ablation_summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Ablation summary CSV not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    required_summary_columns = {policy_column, replicas_csv_column}
    missing_summary_columns = required_summary_columns - set(summary_df.columns)
    if missing_summary_columns:
        raise ValueError(
            f"Ablation summary CSV {summary_path} is missing required column(s): {sorted(missing_summary_columns)}"
        )

    frames: list[pd.DataFrame] = []
    for _, row in summary_df.iterrows():
        policy_name = str(row[policy_column])
        replicas_path = Path(str(row[replicas_csv_column]))
        if not replicas_path.is_file():
            raise FileNotFoundError(f"Per-replica CSV for policy {policy_name!r} not found: {replicas_path}")

        replicas_df = pd.read_csv(replicas_path)
        required_replica_columns = {"replica_index", pairing_column, metric_column}
        missing_replica_columns = required_replica_columns - set(replicas_df.columns)
        if missing_replica_columns:
            raise ValueError(
                f"Replicas CSV {replicas_path} for policy {policy_name!r} is missing "
                f"required column(s): {sorted(missing_replica_columns)}"
            )

        frames.append(
            pd.DataFrame({
                "policy": policy_name,
                "replica_index": replicas_df["replica_index"],
                pairing_column: replicas_df[pairing_column],
                metric_column: replicas_df[metric_column],
            })
        )

    return pd.concat(frames, ignore_index=True)


def compute_ablation_significance(
        replica_long_df: pd.DataFrame,
        *,
        config: Optional[AblationSignificanceConfig] = None,
    ) -> pd.DataFrame:
    '''Run a paired significance test for every policy against a reference policy.

    Parameters
    ----------
    replica_long_df : pd.DataFrame
        Long-format per-replica metric values, as returned by ``load_ablation_bedroc_long``.
    config : AblationSignificanceConfig, optional
        Test configuration. Defaults to comparing ``dudez_test_bedroc`` against
        ``full_ocscore`` with a paired t-test and Holm correction.

    Returns
    -------
    pd.DataFrame
        One row per non-reference policy, sorted by corrected p-value ascending, with columns
        ``policy``, ``n_pairs``, ``reference_mean``, ``policy_mean``, ``mean_diff``, ``statistic``,
        ``pvalue``, ``pvalue_corrected``, ``reject_null``, and ``direction``
        (``"better"``, ``"worse"``, or ``"not_significant"``).

    Raises
    ------
    ValueError
        If the reference policy is absent from ``replica_long_df``.
    '''

    cfg = config or AblationSignificanceConfig()
    metric_column = cfg.metric_column
    pairing_column = cfg.pairing_column

    reference_df = replica_long_df[replica_long_df["policy"] == cfg.reference_policy]
    if reference_df.empty:
        raise ValueError(f"Reference policy {cfg.reference_policy!r} not found in replica_long_df.")
    reference_df = reference_df[[pairing_column, metric_column]].rename(
        columns={metric_column: "reference_value"}
    )

    other_policies = [
        policy_name
        for policy_name in replica_long_df["policy"].unique()
        if policy_name != cfg.reference_policy
    ]

    rows: list[dict[str, object]] = []
    for policy_name in other_policies:
        policy_df = replica_long_df[replica_long_df["policy"] == policy_name][[pairing_column, metric_column]]
        policy_df = policy_df.rename(columns={metric_column: "policy_value"})
        paired = policy_df.merge(reference_df, on=pairing_column, how="inner")

        reference_values = paired["reference_value"].to_numpy(dtype=float)
        policy_values = paired["policy_value"].to_numpy(dtype=float)
        statistic, pvalue = _run_paired_test(reference_values, policy_values, method=cfg.method)

        rows.append({
            "policy": policy_name,
            "n_pairs": int(len(paired)),
            "reference_mean": float(np.mean(reference_values)) if len(reference_values) else float("nan"),
            "policy_mean": float(np.mean(policy_values)) if len(policy_values) else float("nan"),
            "mean_diff": float(np.mean(policy_values - reference_values)) if len(paired) else float("nan"),
            "statistic": statistic,
            "pvalue": pvalue,
        })

    result_df = pd.DataFrame(rows)
    valid_mask = result_df["pvalue"].notna()
    result_df["pvalue_corrected"] = np.nan
    result_df["reject_null"] = False
    if valid_mask.any():
        reject, corrected, _, _ = multipletests(
            result_df.loc[valid_mask, "pvalue"].to_numpy(),
            alpha=cfg.alpha,
            method=cfg.correction_method,
        )
        result_df.loc[valid_mask, "pvalue_corrected"] = corrected
        result_df.loc[valid_mask, "reject_null"] = reject

    def _direction(row: pd.Series) -> str:
        if not row["reject_null"]:
            return "not_significant"
        return "better" if row["mean_diff"] > 0 else "worse"

    result_df["direction"] = result_df.apply(_direction, axis=1)
    return result_df.sort_values("pvalue_corrected", na_position="last").reset_index(drop=True)


def build_ablation_bedroc_significance_table(
        ablation_summary_csv: str | Path,
        *,
        config: Optional[AblationSignificanceConfig] = None,
        policy_column: str = DEFAULT_POLICY_COLUMN,
        replicas_csv_column: str = DEFAULT_REPLICAS_CSV_COLUMN,
    ) -> pd.DataFrame:
    '''Load per-replica BEDROC values and compute significance vs the reference policy.

    Parameters
    ----------
    ablation_summary_csv : str or pathlib.Path
        Path to an ``ablation_summary.csv`` listing one row per feature policy.
    config : AblationSignificanceConfig, optional
        Test configuration. Defaults to comparing ``dudez_test_bedroc`` against
        ``full_ocscore`` with a paired t-test and Holm correction.
    policy_column : str, optional
        Policy-name column in ``ablation_summary_csv``, by default ``"feature_policy_name"``.
    replicas_csv_column : str, optional
        Column holding each policy's per-replica CSV path, by default ``"replicas_summary_csv"``.

    Returns
    -------
    pd.DataFrame
        Significance table as returned by ``compute_ablation_significance``.
    '''

    cfg = config or AblationSignificanceConfig()
    replica_long_df = load_ablation_bedroc_long(
        ablation_summary_csv,
        metric_column=cfg.metric_column,
        policy_column=policy_column,
        replicas_csv_column=replicas_csv_column,
        pairing_column=cfg.pairing_column,
    )
    return compute_ablation_significance(replica_long_df, config=cfg)


__all__ = [
    "AblationSignificanceConfig",
    "build_ablation_bedroc_significance_table",
    "compute_ablation_significance",
    "load_ablation_bedroc_long",
]
