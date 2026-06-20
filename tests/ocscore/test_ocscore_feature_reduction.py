#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for granular OCScore feature-reduction utilities.
'''

# Imports
###############################################################################
import json

import numpy as np
import pandas as pd
import pytest

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.OCScore.Utils.FeatureReduction as ocfr

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


# Functions
###############################################################################
## Private ##

def _base_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(12)],
            "receptor": ["r1"] * 12,
            "ligand": [f"l{i}" for i in range(12)],
            "db": ["PDBbind"] * 12,
            "experimental": np.linspace(1.0, 2.1, 12),
            "countA": [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6],
            "SASA": np.linspace(10.0, 21.0, 12),
            "MolWt": np.linspace(100.0, 111.0, 12),
            "TPSA": np.linspace(200.0, 222.0, 12),
            "VINA_VINA": [-7.0, -6.8, -6.4, -6.1, -5.9, -5.5, -5.1, -4.9, -4.6, -4.2, -4.0, -3.8],
            "SMINA_VINA": [-7.1, -6.7, -6.5, -6.0, -5.8, -5.6, -5.0, -4.8, -4.5, -4.3, -3.9, -3.7],
        }
    )


## Public ##

@pytest.mark.order(390)
def test_split_descriptor_blocks_from_patterns():
    blocks = ocfr.split_descriptor_blocks(
        columns=["id", "y", "receptor_a", "ligand_b", "vina_score", "other"],
        metadata_columns=["id"],
        target_columns=["y"],
        receptor_patterns=["receptor_"],
        ligand_patterns=["ligand_"],
        scoring_patterns=["vina_"],
        use_ligand_class_descriptors=False,
        use_receptor_class_descriptors=False,
        use_scoring_model_descriptors=False,
    )

    assert blocks.metadata == ["id"]
    assert blocks.target == ["y"]
    assert blocks.receptor == ["receptor_a"]
    assert blocks.ligand == ["ligand_b"]
    assert blocks.scoring == ["vina_score"]
    assert blocks.unmatched == ["other"]


@pytest.mark.order(391)
def test_split_descriptor_blocks_uses_ligand_and_receptor_metadata():
    df = _base_feature_df()
    blocks = ocfr.split_descriptor_blocks(df.columns)

    assert "MolWt" in blocks.ligand
    assert "MolWt" in ocl.Ligand.allDescriptors
    assert "countA" in blocks.receptor
    assert "countA" in ocr.Receptor.allDescriptors
    assert "VINA_VINA" in blocks.scoring
    assert "Ligand.allDescriptors" in blocks.sources["ligand"]
    assert "Receptor.allDescriptors" in blocks.sources["receptor"]


@pytest.mark.order(392)
def test_missing_rows_report_descriptor_and_target_modes():
    df = _base_feature_df()
    df.loc[2, "MolWt"] = np.nan
    df.loc[5, "experimental"] = np.nan
    blocks = ocfr.split_descriptor_blocks(df.columns)

    descriptor_result = ocfr.drop_rows_with_missing_values(
        df,
        subset="descriptor_columns",
        blocks=blocks,
        id_columns=["name"],
    )
    assert descriptor_result.summary["n_rows_dropped"] == 1
    assert descriptor_result.dropped_rows.loc[0, "original_index"] == 2
    assert descriptor_result.dropped_rows.loc[0, "name"] == "c2"
    assert descriptor_result.dropped_rows.loc[0, "n_missing_total"] == 1
    assert descriptor_result.dropped_rows.loc[0, "missing_columns"] == "MolWt"

    model_result = ocfr.drop_rows_with_missing_values(
        df,
        subset="model_relevant_columns",
        blocks=blocks,
        id_columns=["name"],
    )
    assert model_result.summary["n_rows_dropped"] == 2
    assert set(model_result.dropped_rows["original_index"]) == {2, 5}
    assert model_result.summary["missing_values_by_column"]["MolWt"] == 1
    target_missing = model_result.missingness_by_block.set_index("block").loc["target", "n_missing_values"]
    assert target_missing == 1


@pytest.mark.order(393)
def test_missing_rows_report_has_schema_when_no_rows_are_dropped():
    df = _base_feature_df()
    blocks = ocfr.split_descriptor_blocks(df.columns)

    result = ocfr.drop_rows_with_missing_values(
        df,
        subset="model_relevant_columns",
        blocks=blocks,
        id_columns=["name"],
    )

    assert result.summary["n_rows_dropped"] == 0
    assert result.dropped_rows.empty
    assert list(result.dropped_rows.columns) == [
        "original_index",
        "n_missing_total",
        "missing_columns",
        "drop_reason",
        "name",
    ]
    assert result.missingness_by_column["n_missing"].sum() == 0
    assert set(result.missingness_by_block["block"]) == {"receptor", "ligand", "scoring", "target"}


@pytest.mark.order(394)
def test_validation_fails_on_nan_and_inf_after_row_drop():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.inf]})
    with pytest.raises(ValueError, match="infinite"):
        ocfr.validate_descriptor_frame(df, ["a", "b"])

    df_nan = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError, match="NaN"):
        ocfr.validate_descriptor_frame(df_nan, ["a"])


@pytest.mark.order(395)
def test_column_quality_filters_and_apply_feature_drops():
    df = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "near": [2, 2, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0],
            "x_dup": [1.0, 2.0, 3.0, 4.0],
        }
    )

    constant = ocfr.find_constant_features(df, ["constant", "near", "x"])
    near = ocfr.find_near_constant_features(df, ["near", "x"], threshold=0.70)
    duplicate = ocfr.find_duplicate_features(df, ["x", "x_dup"])

    assert constant["feature"].tolist() == ["constant"]
    assert near["feature"].tolist() == ["near"]
    assert duplicate.loc[0, "kept_feature"] == "x"
    assert duplicate.loc[0, "dropped_feature"] == "x_dup"
    assert ocfr.apply_feature_drops(["constant", "near", "x"], constant) == ["near", "x"]


@pytest.mark.order(396)
def test_intra_block_correlations_and_filtering_are_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 1, 4, 2, 3]})
    corr = ocfr.compute_intra_block_correlations(df, ["a", "b", "c"], method="spearman", block="receptor")
    filt = ocfr.filter_correlated_features(corr, threshold=0.98, retention_policy="first")

    assert corr.matrix.loc["a", "b"] == pytest.approx(1.0)
    assert filt.dropped_features == ["b"]
    assert "a" in filt.kept_features
    assert filt.report.loc[0, "reason"] == "intra_block_correlation"


@pytest.mark.order(397)
def test_cross_block_correlations_and_predictability_reports():
    df = pd.DataFrame(
        {
            "r1": np.arange(12, dtype=float),
            "r2": [1.0, 0.0] * 6,
            "s1": np.arange(12, dtype=float) * 2.0,
            "s2": [3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0, 7.0],
        }
    )

    pairwise = ocfr.compute_cross_block_correlations(
        df,
        left_columns=["r1", "r2"],
        right_columns=["s1", "s2"],
        threshold=0.95,
        left_block="receptor",
        right_block="scoring",
    )
    predictability = ocfr.compute_cross_block_predictability(
        df,
        predictor_columns=["r1", "r2"],
        target_columns=["s1", "s2"],
        cv_folds=3,
        random_seed=1,
        predictor_block="receptor",
    )

    assert pairwise[pairwise["left_feature"] == "r1"]["flagged"].any()
    assert set(predictability["target_feature"]) == {"s1", "s2"}
    assert predictability["n_jobs"].tolist() == [1, 1]
    assert set(predictability["interpretation"]).issubset(
        {"low redundancy", "partial redundancy", "strong redundancy", "very strong redundancy", "not estimated"}
    )

    with pytest.raises(ValueError, match="n_jobs"):
        ocfr.compute_cross_block_predictability(
            df,
            predictor_columns=["r1", "r2"],
            target_columns=["s1"],
            cv_folds=3,
            n_jobs=0,
        )


@pytest.mark.order(398)
def test_cross_block_filtering_requires_scoring_priority():
    report = pd.DataFrame(
        {
            "left_feature": ["MolWt"],
            "right_feature": ["VINA_VINA"],
            "correlation": [0.99],
            "abs_correlation": [0.99],
        }
    )

    disabled = ocfr.filter_cross_block_redundant_features(
        report,
        molecular_columns=["MolWt"],
        scoring_columns=["VINA_VINA"],
        scoring_function_priority=False,
    )
    enabled = ocfr.filter_cross_block_redundant_features(
        report,
        molecular_columns=["MolWt"],
        scoring_columns=["VINA_VINA"],
        scoring_function_priority=True,
    )

    assert disabled.dropped_features == []
    assert enabled.dropped_features == ["MolWt"]
    assert enabled.report.loc[0, "reason"] == "cross_block_correlation_with_scoring_priority"


@pytest.mark.order(399)
def test_cross_block_filtering_runs_when_diagnostics_are_disabled():
    df = pd.DataFrame(
        {
            "name": [f"c{i}" for i in range(12)],
            "experimental": np.linspace(1.0, 2.1, 12),
            "receptor_x": np.arange(12, dtype=float),
            "ligand_x": [1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 6.0, 10.0, 7.0, 11.0, 9.0, 12.0],
            "vina_score": np.arange(12, dtype=float),
        }
    )
    cfg = ocfr.FeatureReductionConfig()
    cfg.block_detection.use_ligand_class_descriptors = False
    cfg.block_detection.use_receptor_class_descriptors = False
    cfg.block_detection.use_scoring_model_descriptors = False
    cfg.cross_block_diagnostics.enabled = False
    cfg.cross_block_filtering.enabled = True
    cfg.cross_block_filtering.scoring_function_priority = True

    result = ocfr.run_feature_reduction_protocol(df=df, config=cfg)

    assert "receptor_x" not in result.selected_features
    assert "ligand_x" in result.selected_features
    assert "vina_score" in result.selected_features
    assert result.reports["cross_block_predictability_report"].empty
    assert not result.reports["cross_block_pairwise_correlation_report"].empty
    assert result.reports["cross_block_filter_report"].loc[0, "molecular_feature"] == "receptor_x"
    assert result.protocol["cross_block_diagnostics"]["enabled"] is False
    assert result.protocol["cross_block_filtering"]["enabled"] is True


@pytest.mark.order(400)
def test_compose_selected_features_build_reduced_dataframe_and_no_input_mutation():
    df = _base_feature_df()
    before = df.copy(deep=True)
    selected = ocfr.compose_selected_features(["countA"], ["MolWt"], ["VINA_VINA"])
    reduced = ocfr.build_reduced_dataframe(df, metadata_columns=["name"], target_columns=["experimental"], selected_features=selected)

    assert selected == ["countA", "MolWt", "VINA_VINA"]
    assert list(reduced.columns) == ["name", "experimental", "countA", "MolWt", "VINA_VINA"]
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.order(401)
def test_run_feature_reduction_protocol_writes_reports_and_protocol(tmp_path):
    df = _base_feature_df()
    df["constant_rec"] = 1.0
    cfg = ocfr.FeatureReductionConfig()
    cfg.block_detection.receptor_patterns.append("constant_")
    cfg.cross_block_diagnostics.ridge_cv_folds = 3
    cfg.cross_block_diagnostics.n_jobs = 1
    cfg.verbose = True

    result = ocfr.run_feature_reduction_protocol(
        df=df,
        output_dir=tmp_path,
        config=cfg,
        write_outputs=True,
    )

    assert "countA" in result.blocks.receptor
    assert result.reduced_df.shape[0] == df.shape[0]
    assert result.protocol["row_filtering"]["n_rows_before"] == df.shape[0]
    assert result.protocol["row_filtering"]["n_rows_after"] == df.shape[0]
    assert result.protocol["block_detection"]["used_ligand_class_metadata"] is True
    assert result.protocol["block_detection"]["used_receptor_class_metadata"] is True
    assert "feature_reduction_protocol_json" in result.output_paths

    protocol_path = tmp_path / "feature_reduction_protocol.json"
    selected_path = tmp_path / "selected_features.json"
    assert protocol_path.exists()
    assert selected_path.exists()
    assert (tmp_path / "cross_block_filter_report.csv").exists()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert payload["final_output"]["reduced_dataset_shape"] == list(result.reduced_df.shape)
    assert payload["configuration"]["cross_block_diagnostics"]["n_jobs"] == 1
    assert payload["configuration"]["verbose"] is True
    assert payload["cross_block_diagnostics"]["n_jobs"] == 1
    assert "feature_reduction_protocol_json" in payload["final_output"]["output_paths"]
    assert "feature_reduction_protocol_md" in payload["final_output"]["output_paths"]
    feature_selection_path = tmp_path / "feature_selection.json"
    assert feature_selection_path.exists()
    selection_payload = json.loads(feature_selection_path.read_text(encoding="utf-8"))
    assert selection_payload["scope"] == "precomputed_global"
    assert "cross_block_filter_report" in payload["final_output"]["output_paths"]
    assert json.loads(selected_path.read_text(encoding="utf-8")) == result.selected_features
