#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for affinity-aware PDBbind regression splitting."""

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

from OCDocker.OCScore.Utils.PDBbindSplit import PDBbindSplitConfig
from OCDocker.OCScore.Utils.PDBbindSplit import split_pdbbind_regression


# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


def _pdbbind_df(n: int = 120, n_features: int = 6, seed: int = 0, with_receptor: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = np.concatenate(
        [
            rng.normal(loc=-2.0, scale=0.3, size=n // 3),
            rng.normal(loc=0.0, scale=0.3, size=n // 3),
            rng.normal(loc=2.0, scale=0.3, size=n - 2 * (n // 3)),
        ]
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["experimental"] = y.astype(float)
    if with_receptor:
        df["receptor"] = [f"P{i % 10}" for i in range(n)]
    return df


@pytest.mark.order(282)
def test_split_does_not_require_kind():
    df = _pdbbind_df(with_receptor=False)
    cfg = PDBbindSplitConfig(target_column="experimental", random_seed=7, relaxed_split=False)
    result = split_pdbbind_regression(df, cfg)
    assert len(result.train_idx) and len(result.val_idx) and len(result.test_idx)


@pytest.mark.order(283)
def test_split_stratifies_by_affinity_bins_and_preserves_distribution_roughly():
    df = _pdbbind_df(n=150)
    cfg = PDBbindSplitConfig(n_affinity_bins=5, random_seed=42, relaxed_split=False)
    result = split_pdbbind_regression(df, cfg)
    diag = result.diagnostics["affinity_bins"]
    assert diag["requested_bins"] == 5
    assert diag["used_bins"] >= 2

    total = diag["bin_counts_total"]
    train = diag["bin_counts_train"]
    val = diag["bin_counts_validation"]
    test = diag["bin_counts_test"]
    assert total and train and val and test

    # Each split should cover multiple bins when data is large enough.
    assert len(train) >= 3
    assert len(val) >= 2
    assert len(test) >= 2

    # Distributions shouldn't collapse: each split should have some mass in most bins.
    for split_counts in (train, val, test):
        assert sum(split_counts.values()) > 0


@pytest.mark.order(284)
def test_split_is_deterministic_for_same_seed():
    df = _pdbbind_df(n=120)
    cfg = PDBbindSplitConfig(random_seed=11, relaxed_split=False)
    first = split_pdbbind_regression(df, cfg)
    second = split_pdbbind_regression(df, cfg)
    assert np.array_equal(first.train_idx, second.train_idx)
    assert np.array_equal(first.val_idx, second.val_idx)
    assert np.array_equal(first.test_idx, second.test_idx)


@pytest.mark.order(285)
def test_split_changes_with_different_seed():
    df = _pdbbind_df(n=120)
    first = split_pdbbind_regression(df, PDBbindSplitConfig(random_seed=1, relaxed_split=False))
    second = split_pdbbind_regression(df, PDBbindSplitConfig(random_seed=2, relaxed_split=False))
    assert not np.array_equal(np.sort(first.test_idx), np.sort(second.test_idx))


@pytest.mark.order(286)
def test_nan_targets_fail_when_relaxed_false_and_drop_when_relaxed_true():
    df = _pdbbind_df(n=60)
    df.loc[0, "experimental"] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        split_pdbbind_regression(df, PDBbindSplitConfig(relaxed_split=False))

    result = split_pdbbind_regression(df, PDBbindSplitConfig(relaxed_split=True))
    assert result.diagnostics["n_rows_dropped_nan_target"] == 1


@pytest.mark.order(287)
def test_prepare_pdbbind_regression_data_fits_scaler_on_train_only():
    df = _pdbbind_df(n=90, n_features=4)
    features = [c for c in df.columns if c.startswith("f")]
    cfg = ocstaged.PDBbindSplitConfig(target_column="experimental", random_seed=9, relaxed_split=False)
    splits = ocstaged.prepare_pdbbind_regression_data(df, features, split_config=cfg)
    scaler = splits["scaler"]
    train_idx = splits["train_indices"]
    X = df[features].to_numpy(dtype=float)
    assert np.allclose(scaler.mean_, np.mean(X[train_idx], axis=0))


@pytest.mark.order(288)
def test_split_diagnostics_are_returned():
    df = _pdbbind_df(n=90)
    cfg = ocstaged.PDBbindSplitConfig(target_column="experimental", random_seed=9, relaxed_split=False)
    splits = ocstaged.prepare_pdbbind_regression_data(
        df,
        [c for c in df.columns if c.startswith("f")],
        split_config=cfg,
    )
    assert "split_diagnostics" in splits
    assert splits["split_diagnostics"]["strategy"] == "affinity_quantile_stratified"


@pytest.mark.order(289)
def test_receptor_heldout_has_no_receptor_overlap():
    df = _pdbbind_df(n=90)
    cfg = ocstaged.PDBbindSplitConfig(
        strategy="receptor_heldout",
        target_column="experimental",
        random_seed=11,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
    )
    result = ocstaged.split_pdbbind_regression(df, cfg)
    overlap = result.diagnostics["receptor_overlap"]
    assert overlap["train∩validation"] == 0
    assert overlap["train∩test"] == 0
    assert overlap["validation∩test"] == 0

