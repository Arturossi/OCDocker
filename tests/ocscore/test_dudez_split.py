#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for receptor- and kind-aware DUDEz splitting."""

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

from OCDocker.OCScore.Utils.DUDEzSplit import DUDEzSplitConfig
from OCDocker.OCScore.Utils.DUDEzSplit import dudez_receptor_heldout_complete_config
from OCDocker.OCScore.Utils.DUDEzSplit import split_dudez_by_receptor_and_kind

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Private ##

def _make_dudez_df(receptor_counts: dict[str, tuple[int, int]], n_features: int = 4) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for receptor, (n_ligands, n_decoys) in receptor_counts.items():
        for idx in range(n_ligands):
            row = {f"f{i}": float(rng.normal()) for i in range(n_features)}
            row.update(
                {
                    "receptor": receptor,
                    "ligand": f"{receptor}_lig_{idx}",
                    "name": f"{receptor}_lig_{idx}",
                    "kind": "ligands",
                }
            )
            rows.append(row)
        for idx in range(n_decoys):
            row = {f"f{i}": float(rng.normal()) for i in range(n_features)}
            row.update(
                {
                    "receptor": receptor,
                    "ligand": f"{receptor}_dec_{idx}",
                    "name": f"{receptor}_dec_{idx}",
                    "kind": "decoys",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _heldout_config(**overrides) -> DUDEzSplitConfig:
    payload = {
        "strategy": "receptor_heldout_complete",
        "n_train_receptors": 31,
        "n_validation_receptors": 6,
        "n_test_receptors": 6,
        "random_seed": 42,
        "relaxed_split": False,
        "shuffle_within_splits": True,
    }
    payload.update(overrides)
    return DUDEzSplitConfig(**payload)


def _make_43_receptor_df(n_features: int = 4) -> pd.DataFrame:
    return _make_dudez_df({f"R{i:02d}": (3, 10) for i in range(43)}, n_features=n_features)


def _default_config(**overrides) -> DUDEzSplitConfig:
    payload = {
        "strategy": "receptor_stratified_kind",
        "train_size": 0.6,
        "validation_size": 0.2,
        "test_size": 0.2,
        "random_seed": 42,
        "relaxed_split": False,
    }
    payload.update(overrides)
    return DUDEzSplitConfig(**payload)


## Public ##

@pytest.mark.order(270)
def test_split_preserves_global_ligands_and_decoys_in_all_splits():
    df = _make_dudez_df({"A": (4, 10), "B": (4, 10), "C": (4, 10), "D": (4, 10)})
    result = split_dudez_by_receptor_and_kind(df, _default_config())
    labels = ocstaged.derive_dudez_labels(df)
    for split_name, indices in (
        ("train", result.train_idx),
        ("validation", result.val_idx),
        ("test", result.test_idx),
    ):
        split_labels = labels[indices]
        assert np.any(split_labels == 1), split_name
        assert np.any(split_labels == 0), split_name


@pytest.mark.order(271)
def test_split_preserves_receptor_diversity_across_splits():
    df = _make_dudez_df({"A": (4, 10), "B": (4, 10), "C": (4, 10), "D": (4, 10)})
    result = split_dudez_by_receptor_and_kind(df, _default_config())
    receptors = df["receptor"].to_numpy()
    train_targets = set(receptors[result.train_idx])
    val_targets = set(receptors[result.val_idx])
    test_targets = set(receptors[result.test_idx])
    assert len(train_targets) >= 3
    assert len(val_targets) >= 2
    assert len(test_targets) >= 2


@pytest.mark.order(272)
def test_receptors_with_enough_samples_appear_in_all_splits():
    df = _make_dudez_df({"A": (5, 15), "B": (5, 15)})
    result = split_dudez_by_receptor_and_kind(df, _default_config())
    receptors = df["receptor"].to_numpy()
    for receptor in ("A", "B"):
        for indices in (result.train_idx, result.val_idx, result.test_idx):
            mask = receptors[indices] == receptor
            split_labels = ocstaged.derive_dudez_labels(df)[indices][mask]
            assert np.any(split_labels == 1)
            assert np.any(split_labels == 0)


@pytest.mark.order(273)
def test_validation_and_test_splits_have_valid_metric_groups():
    df = _make_dudez_df({"A": (4, 12), "B": (4, 12), "C": (4, 12)})
    result = split_dudez_by_receptor_and_kind(df, _default_config())
    assert result.diagnostics["n_groups_used_validation"] >= 1
    assert result.diagnostics["n_groups_used_test"] >= 1
    assert result.diagnostics["splits"]["validation"]["n_invalid_metric_groups"] >= 0


@pytest.mark.order(274)
def test_split_is_deterministic_for_same_seed():
    df = _make_dudez_df({"A": (4, 12), "B": (4, 12), "C": (4, 12)})
    first = split_dudez_by_receptor_and_kind(df, _default_config(random_seed=7))
    second = split_dudez_by_receptor_and_kind(df, _default_config(random_seed=7))
    assert np.array_equal(first.train_idx, second.train_idx)
    assert np.array_equal(first.val_idx, second.val_idx)
    assert np.array_equal(first.test_idx, second.test_idx)


@pytest.mark.order(275)
def test_different_seeds_produce_different_splits():
    df = _make_dudez_df({"A": (4, 12), "B": (4, 12), "C": (4, 12), "D": (4, 12)})
    first = split_dudez_by_receptor_and_kind(df, _default_config(random_seed=1))
    second = split_dudez_by_receptor_and_kind(df, _default_config(random_seed=2))
    assert not np.array_equal(np.sort(first.test_idx), np.sort(second.test_idx))


@pytest.mark.order(276)
def test_small_receptor_groups_are_documented_in_receptor_notes():
    df = _make_dudez_df({"sparse": (1, 8), "rich": (5, 15)})
    result = split_dudez_by_receptor_and_kind(df, _default_config(relaxed_split=True))
    sparse_note = result.diagnostics["receptor_notes"]["sparse"]
    assert (
        sparse_note.get("ligands_train_only_fallback")
        or sparse_note.get("ligands_insufficient_for_all_splits")
        or sparse_note.get("ligand_split", {}).get("assign_all_to_train")
    )


@pytest.mark.order(277)
def test_strict_split_fails_when_constraints_cannot_be_met():
    df = _make_dudez_df({"only_train_ligands": (1, 20)})
    with pytest.raises(ValueError, match="validity constraints"):
        split_dudez_by_receptor_and_kind(df, _default_config(relaxed_split=False))


@pytest.mark.order(278)
def test_relaxed_split_allows_documented_fallback_for_sparse_receptors():
    df = _make_dudez_df({"sparse": (1, 30), "rich": (5, 20), "rich2": (5, 20)})
    result = split_dudez_by_receptor_and_kind(df, _default_config(relaxed_split=True))
    assert "sparse" in result.diagnostics["receptor_notes"]
    assert result.diagnostics["splits"]["validation"]["n_ligands"] >= 1
    assert result.diagnostics["splits"]["validation"]["n_decoys"] >= 1


@pytest.mark.order(279)
def test_prepare_dudez_screening_data_aligns_metric_groups():
    features = [f"f{i}" for i in range(4)]
    df = _make_dudez_df({"A": (4, 12), "B": (4, 12), "C": (4, 12)}, n_features=len(features))
    labels = ocstaged.derive_dudez_labels(df)
    groups = df["receptor"].to_numpy()
    split_config = _default_config()
    splits = ocstaged.prepare_dudez_screening_data(
        df,
        features,
        labels,
        groups=groups,
        split_config=split_config,
        target_group_column="receptor",
    )
    assert splits["val_groups"] is not None
    assert len(splits["val_groups"]) == len(splits["y_val"])
    metrics = ocstaged.evaluate_screening_metrics(
        splits["y_val"],
        np.linspace(0.0, 1.0, len(splits["y_val"])),
        groups=splits["val_groups"],
    )
    assert metrics["n_groups_used"] >= 1.0


@pytest.mark.order(280)
def test_pdbbind_splitting_is_unaffected_by_dudez_split_module():
    features = [f"f{i}" for i in range(4)]
    rng = np.random.default_rng(0)
    pdbbind_df = pd.DataFrame(rng.normal(size=(40, len(features))), columns=features)
    pdbbind_df["experimental"] = rng.normal(size=40)
    splits = ocstaged.prepare_pdbbind_regression_data(
        pdbbind_df,
        features,
        split_config=ocstaged.PDBbindSplitConfig(
            target_column="experimental",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_seed=99,
            relaxed_split=True,
        ),
    )
    assert splits["X_train"].shape[0] + splits["X_val"].shape[0] + splits["X_test"].shape[0] == 40


@pytest.mark.order(281)
def test_split_strategy_is_recorded_in_diagnostics():
    df = _make_dudez_df({"A": (4, 12), "B": (4, 12)})
    result = split_dudez_by_receptor_and_kind(df, _default_config())
    assert result.diagnostics["strategy"] == "receptor_stratified_kind"
    assert result.diagnostics["random_seed"] == 42


@pytest.mark.order(282)
def test_heldout_complete_assigns_whole_receptors_to_one_split():
    df = _make_43_receptor_df()
    result = split_dudez_by_receptor_and_kind(df, _heldout_config())
    receptors = df["receptor"].to_numpy()
    for split_indices in (result.train_idx, result.val_idx, result.test_idx):
        for receptor in np.unique(receptors[split_indices]):
            train_mask = receptors[result.train_idx] == receptor
            val_mask = receptors[result.val_idx] == receptor
            test_mask = receptors[result.test_idx] == receptor
            assert int(train_mask.any()) + int(val_mask.any()) + int(test_mask.any()) == 1


@pytest.mark.order(283)
def test_heldout_complete_receptor_sets_are_disjoint():
    df = _make_43_receptor_df()
    diagnostics = split_dudez_by_receptor_and_kind(df, _heldout_config()).diagnostics
    train_set = set(diagnostics["train_receptors"])
    val_set = set(diagnostics["validation_receptors"])
    test_set = set(diagnostics["test_receptors"])
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set


@pytest.mark.order(284)
def test_heldout_complete_receptor_counts_31_6_6():
    df = _make_43_receptor_df()
    diagnostics = split_dudez_by_receptor_and_kind(df, _heldout_config()).diagnostics
    assert diagnostics["receptor_counts"]["train"] == 31
    assert diagnostics["receptor_counts"]["validation"] == 6
    assert diagnostics["receptor_counts"]["test"] == 6


@pytest.mark.order(285)
def test_heldout_complete_validation_and_test_receptors_are_complete():
    df = _make_43_receptor_df()
    result = split_dudez_by_receptor_and_kind(df, _heldout_config())
    labels = ocstaged.derive_dudez_labels(df)
    receptors = df["receptor"].to_numpy()
    for split_indices in (result.val_idx, result.test_idx):
        for receptor in np.unique(receptors[split_indices]):
            mask = receptors[split_indices] == receptor
            split_labels = labels[split_indices][mask]
            assert np.any(split_labels == 1)
            assert np.any(split_labels == 0)


@pytest.mark.order(286)
def test_heldout_complete_validation_test_groups_all_valid():
    df = _make_43_receptor_df()
    diagnostics = split_dudez_by_receptor_and_kind(df, _heldout_config()).diagnostics
    assert diagnostics["n_groups_total_validation"] == diagnostics["n_groups_used_validation"]
    assert diagnostics["n_groups_total_test"] == diagnostics["n_groups_used_test"]
    assert diagnostics["splits"]["validation"]["n_invalid_metric_groups"] == 0
    assert diagnostics["splits"]["test"]["n_invalid_metric_groups"] == 0


@pytest.mark.order(287)
def test_heldout_complete_split_is_deterministic():
    df = _make_43_receptor_df()
    first = split_dudez_by_receptor_and_kind(df, _heldout_config(random_seed=11))
    second = split_dudez_by_receptor_and_kind(df, _heldout_config(random_seed=11))
    assert np.array_equal(np.sort(first.train_idx), np.sort(second.train_idx))
    assert np.array_equal(first.train_idx, second.train_idx)
    assert set(first.diagnostics["validation_receptors"]) == set(second.diagnostics["validation_receptors"])


@pytest.mark.order(288)
def test_heldout_complete_different_seeds_change_assignment():
    df = _make_43_receptor_df()
    first = split_dudez_by_receptor_and_kind(df, _heldout_config(random_seed=1))
    second = split_dudez_by_receptor_and_kind(df, _heldout_config(random_seed=2))
    assert set(first.diagnostics["validation_receptors"]) != set(second.diagnostics["validation_receptors"])


@pytest.mark.order(289)
def test_heldout_complete_explicit_receptor_lists_are_respected():
    df = _make_43_receptor_df()
    all_receptors = sorted(df["receptor"].unique())
    train = all_receptors[:31]
    validation = all_receptors[31:37]
    test = all_receptors[37:43]
    config = _heldout_config(
        train_receptors=train,
        validation_receptors=validation,
        test_receptors=test,
    )
    diagnostics = split_dudez_by_receptor_and_kind(df, config).diagnostics
    assert set(diagnostics["train_receptors"]) == set(train)
    assert set(diagnostics["validation_receptors"]) == set(validation)
    assert set(diagnostics["test_receptors"]) == set(test)


@pytest.mark.order(290)
def test_heldout_complete_invalid_explicit_lists_fail():
    df = _make_43_receptor_df()
    all_receptors = sorted(df["receptor"].unique())
    with pytest.raises(ValueError, match="overlap"):
        split_dudez_by_receptor_and_kind(
            df,
            _heldout_config(
                train_receptors=all_receptors[:10],
                validation_receptors=all_receptors[:6],
                test_receptors=all_receptors[6:12],
            ),
        )
    with pytest.raises(ValueError, match="Unknown"):
        split_dudez_by_receptor_and_kind(
            df,
            _heldout_config(
                train_receptors=["missing"],
                validation_receptors=all_receptors[31:37],
                test_receptors=all_receptors[37:43],
            ),
        )
    df_incomplete = _make_dudez_df({"bad": (2, 0), **{f"R{i:02d}": (3, 10) for i in range(42)}})
    complete = sorted(name for name in df_incomplete["receptor"].unique() if name != "bad")
    with pytest.raises(ValueError, match="ligands and decoys"):
        split_dudez_by_receptor_and_kind(
            df_incomplete,
            _heldout_config(
                train_receptors=complete[:31],
                validation_receptors=["bad"],
                test_receptors=complete[31:37],
            ),
        )


@pytest.mark.order(291)
def test_heldout_complete_shuffles_rows_within_splits():
    df = _make_43_receptor_df()
    shuffled = split_dudez_by_receptor_and_kind(df, _heldout_config(shuffle_within_splits=True))
    unshuffled = split_dudez_by_receptor_and_kind(df, _heldout_config(shuffle_within_splits=False))
    assert np.array_equal(np.sort(shuffled.train_idx), np.sort(unshuffled.train_idx))
    assert not np.array_equal(shuffled.train_idx, unshuffled.train_idx)


@pytest.mark.order(292)
def test_prepare_dudez_screening_data_uses_heldout_splits_for_optuna():
    features = [f"f{i}" for i in range(4)]
    df = _make_43_receptor_df(n_features=len(features))
    labels = ocstaged.derive_dudez_labels(df)
    groups = df["receptor"].to_numpy()
    split_config = dudez_receptor_heldout_complete_config(random_seed=99)
    splits = ocstaged.prepare_dudez_screening_data(
        df,
        features,
        labels,
        groups=groups,
        split_config=split_config,
        target_group_column="receptor",
    )
    train_receptors = set(groups[splits["train_indices"]])
    val_receptors = set(groups[splits["validation_indices"]])
    test_receptors = set(groups[splits["test_indices"]])
    assert len(train_receptors) == 31
    assert len(val_receptors) == 6
    assert len(test_receptors) == 6
    assert not train_receptors & val_receptors
    assert not train_receptors & test_receptors
    assert not val_receptors & test_receptors


@pytest.mark.order(293)
def test_staged_optuna_default_dudez_split_is_heldout_complete():
    config = ocstaged.DUDEzOptunaConfig()
    resolved = ocstaged._resolve_dudez_split_config(config, random_seed=123)
    assert resolved.strategy == "receptor_heldout_complete"
    assert resolved.n_train_receptors == 31
    assert resolved.n_validation_receptors == 6
    assert resolved.n_test_receptors == 6
