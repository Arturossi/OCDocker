#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for train-only feature reduction and validation modes.'''

# Imports
###############################################################################
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import FeatureSelectionScope
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import validate_train_only_feature_selection
from OCDocker.OCScore.Utils.FeatureSelectionMetadata import verify_selected_features_against_scope
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import apply_frozen_feature_selection
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import drop_nonfinite_selected_feature_rows
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import feature_reduction_config_for_feature_blocks
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import fit_train_only_feature_reduction
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import write_train_only_reduction_artifact

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


def _wide_df(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(n_rows)],
            "receptor": ["r1"] * n_rows,
            "ligand": [f"l{i}" for i in range(n_rows)],
            "dataset": ["pdbbind"] * n_rows,
            "experimental": np.linspace(1.0, 2.1, n_rows),
            "countA": np.linspace(1, n_rows, n_rows),
            "SASA": np.linspace(10.0, 10.0 + n_rows - 1, n_rows),
            "MolWt": np.linspace(100.0, 100.0 + n_rows - 1, n_rows),
            "TPSA": np.linspace(200.0, 200.0 + 2 * (n_rows - 1), n_rows),
            "receptor_countA": np.linspace(1, n_rows, n_rows),
            "receptor_SASA": np.linspace(10.0, 10.0 + n_rows - 1, n_rows),
            "ligand_MolWt": np.linspace(100.0, 100.0 + n_rows - 1, n_rows),
            "ligand_TPSA": np.linspace(200.0, 200.0 + 2 * (n_rows - 1), n_rows),
            "VINA_VINA": np.linspace(-7.0, -3.8, n_rows),
            "SMINA_VINA": np.linspace(-7.1, -3.7, n_rows),
        }
    )


@pytest.mark.order(410)
def test_fit_train_only_feature_reduction_records_train_scope():
    train_df = _wide_df(8)
    artifact = fit_train_only_feature_reduction(train_df)
    assert artifact.feature_selection.scope == "train_only"
    assert artifact.feature_selection.fit_split == "train"
    assert artifact.feature_selection.fit_row_count == 8
    assert artifact.selected_features
    assert artifact.feature_selection.selected_features_hash == hash_feature_list(artifact.selected_features)


@pytest.mark.order(411)
def test_apply_frozen_feature_selection_requires_all_selected_features():
    train_df = _wide_df(8)
    artifact = fit_train_only_feature_reduction(train_df)
    full_df = _wide_df(10)
    missing = artifact.selected_features[0]
    trimmed = full_df.drop(columns=[missing])
    with pytest.raises(ValueError, match="Missing required selected features"):
        apply_frozen_feature_selection(trimmed, artifact)


@pytest.mark.order(412)
def test_feature_order_is_stable_after_frozen_application():
    train_df = _wide_df(8)
    artifact = fit_train_only_feature_reduction(train_df)
    reduced = apply_frozen_feature_selection(_wide_df(10), artifact)
    feature_columns = [
        column
        for column in reduced.columns
        if column in artifact.selected_features
    ]
    assert feature_columns == artifact.selected_features


@pytest.mark.order(413)
def test_staged_train_rejects_precomputed_global_scope():
    scope = FeatureSelectionScope.precomputed_global()
    with pytest.raises(ValueError, match="Staged train requires train-only"):
        validate_train_only_feature_selection(scope)


@pytest.mark.order(414)
def test_train_only_scope_passes_validation():
    scope = FeatureSelectionScope.train_only(
        fit_row_count=8,
        selected_features=["SASA", "VINA_VINA"],
    )
    validate_train_only_feature_selection(scope)


@pytest.mark.order(415)
def test_verify_selected_features_hash_mismatch_fails():
    scope = FeatureSelectionScope(
        scope="train_only",
        fit_dataset="pdbbind_train",
        fit_split="train",
        fit_row_count=4,
        selected_features_hash=hash_feature_list(["a", "b"]),
    )
    with pytest.raises(ValueError, match="Selected feature hash mismatch"):
        verify_selected_features_against_scope(["a", "c"], scope)



@pytest.mark.order(416)
def test_train_only_excludes_ligand_name_from_descriptors():
    train_df = _wide_df(8)
    train_df["ligand_name"] = [f"lig{i}" for i in range(8)]
    artifact = fit_train_only_feature_reduction(train_df)
    assert "ligand_name" not in artifact.selected_features
    assert "ligand_name" in artifact.metadata_columns


@pytest.mark.order(417)
def test_train_only_artifact_round_trip(tmp_path):
    artifact = fit_train_only_feature_reduction(_wide_df(8))
    paths = write_train_only_reduction_artifact(tmp_path, artifact)
    payload = json.loads(Path(paths["train_only_feature_reduction"]).read_text(encoding="utf-8"))
    feature_selection = payload["feature_selection"]
    assert feature_selection["scope"] == "train_only"
    assert "fit_row_indices" not in feature_selection
    assert feature_selection["fit_row_count"] == 8
    assert feature_selection["fit_row_indices_hash"]
    assert feature_selection["fit_row_content_hash"]
    assert feature_selection["fit_row_indices_artifact"] == paths["feature_selection_fit_rows_json"]
    fit_rows = json.loads(Path(paths["feature_selection_fit_rows_json"]).read_text(encoding="utf-8"))
    assert fit_rows["fit_row_indices"] == list(range(8))
    assert fit_rows["fit_row_indices_hash"] == feature_selection["fit_row_indices_hash"]
    feature_selection_payload = json.loads(Path(paths["feature_selection_json"]).read_text(encoding="utf-8"))
    assert "fit_row_indices" not in feature_selection_payload
    assert payload["selected_features"]


@pytest.mark.order(417)
def test_train_only_reduction_uses_training_rows_only(monkeypatch):
    observed_rows: list[int] = []

    import OCDocker.OCScore.Utils.FeatureReduction as ocfr
    import OCDocker.OCScore.Utils.TrainOnlyFeatureReduction as tofr

    original = ocfr.run_feature_reduction_protocol

    def _capture(df, config=None, write_outputs=False):
        observed_rows.append(len(df))
        return original(df=df, config=config, write_outputs=write_outputs)

    monkeypatch.setattr(tofr.ocfr, "run_feature_reduction_protocol", _capture)
    train_df = _wide_df(6)
    val_df = _wide_df(4)
    artifact = fit_train_only_feature_reduction(train_df)
    assert observed_rows == [6]
    apply_frozen_feature_selection(val_df, artifact)

@pytest.mark.order(418)
def test_selected_feature_cleanup_drops_nonfinite_rows():
    frame = _wide_df(6)
    selected = ["SASA", "VINA_VINA"]
    frame.loc[1, "SASA"] = np.nan
    frame.loc[4, "VINA_VINA"] = np.inf

    result = drop_nonfinite_selected_feature_rows(frame, selected, label="PDBbind")

    assert result.summary["n_rows_before"] == 6
    assert result.summary["n_rows_dropped"] == 2
    assert result.cleaned_df.shape[0] == 4
    assert result.dropped_rows["original_position"].tolist() == [1, 4]
    assert np.isfinite(result.cleaned_df[selected].to_numpy(dtype=float)).all()


@pytest.mark.order(419)
def test_selected_feature_cleanup_fails_when_all_rows_drop():
    frame = _wide_df(2)
    frame["SASA"] = np.nan

    with pytest.raises(ValueError, match="No PDBbind rows remain"):
        drop_nonfinite_selected_feature_rows(frame, ["SASA"], label="PDBbind")


@pytest.mark.order(420)
@pytest.mark.parametrize(
    ("feature_blocks", "allowed"),
    [
        (("ligand",), {"MolWt", "TPSA", "ligand_MolWt", "ligand_TPSA"}),
        (("receptor",), {"countA", "SASA", "receptor_countA", "receptor_SASA"}),
        (("scoring",), {"VINA_VINA", "SMINA_VINA"}),
        (("ligand", "scoring"), {"MolWt", "TPSA", "ligand_MolWt", "ligand_TPSA", "VINA_VINA", "SMINA_VINA"}),
        (("receptor", "scoring"), {"countA", "SASA", "receptor_countA", "receptor_SASA", "VINA_VINA", "SMINA_VINA"}),
    ],
)
def test_train_only_ablation_feature_blocks_restrict_selected_descriptors(feature_blocks, allowed):
    artifact = fit_train_only_feature_reduction(_wide_df(8), feature_blocks=feature_blocks)

    assert artifact.selected_features
    assert set(artifact.selected_features).issubset(allowed)
    assert artifact.protocol["feature_blocks"] == list(feature_blocks)


@pytest.mark.order(421)
def test_train_only_candidate_features_restrict_selected_descriptors():
    allowed = ["ligand_MolWt", "VINA_VINA"]
    artifact = fit_train_only_feature_reduction(_wide_df(8), candidate_features=allowed)

    assert artifact.selected_features
    assert set(artifact.selected_features).issubset(set(allowed))
    assert artifact.protocol["candidate_feature_count_before_reduction"] == len(allowed)
    assert artifact.protocol["candidate_features_before_reduction"] == allowed


@pytest.mark.order(422)
def test_ablation_feature_block_config_rejects_unknown_block():
    with pytest.raises(ValueError, match="Unknown feature block"):
        feature_reduction_config_for_feature_blocks(feature_blocks=["pocket"])
