#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for fixed outer split alignment in production-grade OCScore protocols.'''

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged
from OCDocker.OCScore.Utils.ContentHash import hash_split_indices
from OCDocker.OCScore.Utils.FixedOuterSplit import build_fixed_outer_split_assignment
from OCDocker.OCScore.Utils.FixedOuterSplit import validate_protocol_integrity
from OCDocker.OCScore.Utils.FixedOuterSplit import validate_replica_split_alignment
from OCDocker.OCScore.Utils.PDBbindSplit import PDBbindSplitConfig

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


def _pdbbind_df(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(n_rows)],
            "receptor": [f"r{i % 3}" for i in range(n_rows)],
            "experimental": np.linspace(1.0, 2.1, n_rows),
            "f0": np.linspace(0.1, 1.2, n_rows),
            "f1": np.linspace(0.2, 1.3, n_rows),
        }
    )


@pytest.mark.order(430)
def test_prepare_pdbbind_regression_data_reuses_fixed_indices():
    df = _pdbbind_df()
    split_config = PDBbindSplitConfig(strategy="receptor_heldout", random_seed=99)
    fixed_train = [0, 1, 2, 3, 4, 5, 6]
    fixed_val = [7, 8, 9]
    fixed_test = [10, 11]
    splits = ocstaged.prepare_pdbbind_regression_data(
        df,
        ["f0", "f1"],
        split_config,
        fixed_train_indices=fixed_train,
        fixed_validation_indices=fixed_val,
        fixed_test_indices=fixed_test,
    )
    assert splits["split_diagnostics"]["fixed_outer_split"] is True
    assert splits["train_indices"].tolist() == fixed_train
    assert splits["validation_indices"].tolist() == fixed_val
    assert splits["test_indices"].tolist() == fixed_test


@pytest.mark.order(431)
def test_replica_split_mismatch_fails_under_strict_mode():
    fixed = build_fixed_outer_split_assignment(
        outer_split_seed=42,
        pdbbind_train_indices=[0, 1],
        pdbbind_validation_indices=[2],
        pdbbind_test_indices=[3],
        dudez_train_indices=[0],
        dudez_validation_indices=[1],
        dudez_test_indices=[2],
        feature_selection_fit_row_count=2,
        selected_features=["f0", "f1"],
        removed_features=["f2"],
    )
    with pytest.raises(ValueError, match="split hashes differ"):
        validate_replica_split_alignment(
            fixed,
            replica_name="replica_000",
            pdbbind_split_indices={"train": [0], "validation": [2], "test": [3]},
            selected_features=["f0", "f1"],
            strict=True,
        )


@pytest.mark.order(432)
def test_replica_seeds_can_differ_while_split_hashes_match():
    fixed = build_fixed_outer_split_assignment(
        outer_split_seed=42,
        pdbbind_train_indices=[0, 1, 2],
        pdbbind_validation_indices=[3, 4],
        pdbbind_test_indices=[5, 6],
        dudez_train_indices=[0, 1],
        dudez_validation_indices=[2],
        dudez_test_indices=[3],
        feature_selection_fit_row_count=3,
        selected_features=["f0", "f1"],
        removed_features=[],
    )
    for replica_seed in (42, 43, 44):
        validate_replica_split_alignment(
            fixed,
            replica_name=f"replica_{replica_seed}",
            pdbbind_split_indices={
                "train": fixed.pdbbind_train_indices,
                "validation": fixed.pdbbind_validation_indices,
                "test": fixed.pdbbind_test_indices,
            },
            selected_features=["f0", "f1"],
            strict=True,
        )
        assert hash_split_indices(fixed.pdbbind_train_indices) == fixed.pdbbind_train_indices_hash


@pytest.mark.order(433)
def test_protocol_invalid_for_precomputed_global_scope():
    fixed = build_fixed_outer_split_assignment(
        outer_split_seed=1,
        pdbbind_train_indices=[0],
        pdbbind_validation_indices=[1],
        pdbbind_test_indices=[2],
        dudez_train_indices=[0],
        dudez_validation_indices=[1],
        dudez_test_indices=[2],
        feature_selection_fit_row_count=1,
        selected_features=["a"],
        removed_features=[],
    ).to_dict()
    with pytest.raises(ValueError, match="missing_train_only_feature_selection"):
        validate_protocol_integrity(
            feature_selection={"scope": "precomputed_global"},
            fixed_outer_split=fixed,
            replica_alignments=[{"pdbbind_split_matches_fixed_outer_split": True, "selected_features_match_fixed": True}],
        )


@pytest.mark.order(434)
def test_protocol_invalid_when_split_hashes_missing():
    with pytest.raises(ValueError, match="missing_split_hashes"):
        validate_protocol_integrity(
            feature_selection={"scope": "train_only", "fit_split": "train"},
            fixed_outer_split={},
            replica_alignments=[],
        )
