#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Release inference with the shipped generation bundle.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

from pathlib import Path

import OCDocker.OCScore.Release.Baselines as ocrelbaselines
import OCDocker.OCScore.Release.Bundle as ocrelbundle
import OCDocker.OCScore.Release.Context as ocrelcontext
import OCDocker.OCScore.Release.Catalog as ocrelcatalog
import OCDocker.OCScore.Release.Inference as ocrelinference

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Private ##

def _bundle_or_skip() -> ocrelbundle.FeatureContract:
    '''Load the shipped feature contract, skipping when the bundle is absent.

    Returns
    -------
    OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded contract.
    '''

    path = ocrelcatalog.contract_dir()
    if not (path / "feature_manifest.json").is_file():
        pytest.skip("Gen3B model bundle is not present in this checkout.")
    return ocrelbundle.FeatureContract(path)


def _synthetic_frame(contract: ocrelbundle.FeatureContract, rows: int = 24) -> pd.DataFrame:
    '''Build a table carrying every contract column.

    Parameters
    ----------
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded contract.
    rows : int, optional
        Number of rows to generate.

    Returns
    -------
    pandas.DataFrame
        Random but complete table.
    '''

    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=rows) for c in contract.feature_order}
    frame = pd.DataFrame(data)
    frame["receptor"] = "TEST"
    return frame


## Public ##

def test_catalog_derives_block_masks_from_arm_names() -> None:
    '''Arm directory names translate into the block letters they were trained with.'''

    assert ocrelcatalog._blocks_of_arm("S") == "A"
    assert ocrelcatalog._blocks_of_arm("S_R") == "AC"
    assert ocrelcatalog._blocks_of_arm("S_L_R_P") == "ABCD"
    with pytest.raises(ValueError):
        ocrelcatalog._blocks_of_arm("S_X")


def test_contract_mask_selects_only_live_blocks() -> None:
    '''The mask keeps exactly the columns of the requested blocks.'''

    contract = _bundle_or_skip()
    groups = contract.groups
    assert contract.mask("ABCD").sum() == len(contract.feature_order)
    assert contract.mask("A").sum() == len(groups["A_scores"])
    assert contract.mask("AC").sum() == len(groups["A_scores"]) + len(groups["C_receptor"])


def test_ablation_scores_ignore_columns_outside_their_blocks() -> None:
    '''An ablation arm must not react to columns outside the blocks it takes.'''

    contract = _bundle_or_skip()
    specs = [s for s in ocrelcatalog.discover(families=("ablation",)) if s.live_blocks == "A"]
    if not specs:
        pytest.skip("No score-only ablation arm in the bundle.")

    frame = _synthetic_frame(contract)
    groups = frame["receptor"].to_numpy()
    baseline = ocrelinference.score_models(contract, specs, frame, groups)

    perturbed = frame.copy()
    dead = [c for c, letter in zip(contract.feature_order, contract.block_letter) if letter != "A"]
    rng = np.random.default_rng(1)
    perturbed[dead] = rng.normal(scale=50.0, size=(len(frame), len(dead)))
    after = ocrelinference.score_models(contract, specs, perturbed, groups)

    for column in baseline.columns:
        np.testing.assert_allclose(baseline[column], after[column], rtol=0, atol=1e-6)


def test_normalise_columns_maps_pipeline_names() -> None:
    '''Upper-case score names and bare ligand descriptor names map onto the contract.'''

    contract = _bundle_or_skip()
    frame = pd.DataFrame({"VINA_VINA": [1.0, 2.0], "PLANTS_PLP": [3.0, 4.0]})
    bare = contract.groups["B_ligand"][0][len("ligand_"):]
    frame[bare] = [5.0, 6.0]

    out = ocrelinference.normalise_columns(frame, contract)

    assert "vina_vina" in out.columns
    assert "plants_plp" in out.columns
    assert f"ligand_{bare}" in out.columns


def test_control_verdict_flags_a_control_above_chance() -> None:
    '''A randomised-label control above chance fails the check.'''

    passing = pd.DataFrame({"model": ["release_m1_1", "control_shuffled_labels"],
                            "variant": ["ensemble", "ensemble"],
                            "ROC-AUC": [0.80, 0.51]})
    failing = passing.copy()
    failing.loc[1, "ROC-AUC"] = 0.80

    assert ocrelinference.control_verdict(passing)["status"] == "pass"
    assert ocrelinference.control_verdict(failing)["status"] == "FAIL"
    assert ocrelinference.control_verdict(passing.iloc[:1])["status"] == "not_run"


def test_linear_baselines_span_the_contract() -> None:
    '''Every shipped linear baseline carries one coefficient per contract feature.'''

    contract = _bundle_or_skip()
    frame = _synthetic_frame(contract)
    features = contract.featurize(frame, frame["receptor"].to_numpy())
    scores = ocrelbaselines.score_linear(ocrelcatalog.default_root(), features)

    if not scores:
        pytest.skip("No linear baselines in the bundle.")
    assert {"ridge_S|linear", "elasticnet_S|linear"} <= set(scores)
    for values in scores.values():
        assert values.shape == (len(frame),)
        assert np.isfinite(values).all()


def test_consensus_uses_only_the_score_block() -> None:
    '''Consensus rules must not move when non-score blocks change.'''

    contract = _bundle_or_skip()
    frame = _synthetic_frame(contract)
    groups = frame["receptor"].to_numpy()
    before = ocrelbaselines.score_consensus(contract, contract.featurize(frame, groups))

    perturbed = frame.copy()
    dead = [c for c, letter in zip(contract.feature_order, contract.block_letter) if letter != "A"]
    rng = np.random.default_rng(2)
    perturbed[dead] = rng.normal(scale=50.0, size=(len(frame), len(dead)))
    after = ocrelbaselines.score_consensus(contract, contract.featurize(perturbed, groups))

    for key in before:
        np.testing.assert_allclose(before[key], after[key], rtol=0, atol=1e-9)


def test_bare_context_columns_resolve_to_the_right_block() -> None:
    '''Unprefixed protein descriptors are assigned using the names unique to each set.'''

    contract = _bundle_or_skip()
    groups = contract.groups
    receptor_only = {c.split("_", 1)[1] for c in groups["C_receptor"]} - {
        c.split("_", 1)[1] for c in groups["D_pocket"]}
    pocket_only = {c.split("_", 1)[1] for c in groups["D_pocket"]} - {
        c.split("_", 1)[1] for c in groups["C_receptor"]}

    assert ocrelinference._infer_context_prefix(set(receptor_only), contract) == "receptor"
    assert ocrelinference._infer_context_prefix(set(pocket_only), contract) == "pocket"
    assert ocrelinference._infer_context_prefix(receptor_only | pocket_only, contract) is None
    assert ocrelinference._infer_context_prefix(set(), contract) is None


def test_context_mapping_shape_is_detected() -> None:
    '''A single-target mapping and a per-receptor mapping are told apart.'''

    assert not ocrelcontext.is_multi_target({"receptor": {}, "pocket": {}})
    assert ocrelcontext.is_multi_target({"A": {"receptor": {}, "pocket": {}}})
    assert ocrelcontext.is_multi_target({})


def test_compute_context_takes_the_pocket_from_the_docking_centre(tmp_path) -> None:
    '''The pocket comes from the docking centre alone, given as coordinates or as a box file.'''

    import OCDocker.Pocket as ocpocket
    import OCDocker.Receptor as ocreceptor

    receptor_file = Path(__file__).resolve().parents[2] / "test_files" / "test_ptn1" / "receptor.pdb"
    _, structure = ocreceptor.load_mol(str(receptor_file), name="receptor", compute_sasa=False, clean=False, canonicalize_pdb=False)
    first = next(r for r in structure[0].get_residues() if r.id[0] == ' ')
    center = tuple(float(v) for v in first["CA"].get_coord())
    box_file = tmp_path / "box0.pdb"
    box_file.write_text(f"REMARK    CENTER (X Y Z)      {center[0]:8.3f}{center[1]:8.3f}{center[2]:8.3f}\n")

    from_center = ocrelcontext.compute_context(receptor_file, center)
    from_box = ocrelcontext.compute_context(receptor_file, box_file)

    assert ocrelcontext.DEFAULT_RADIUS == pytest.approx(12.0)
    assert set(from_center["pocket"]) == {f"pocket_{d}" for d in ocpocket.Pocket.allDescriptors}
    assert all(k.startswith("receptor_") for k in from_center["receptor"])
    assert from_center["pocket"]["pocket_TotalAALength"] >= 1
    assert from_box["pocket"] == pytest.approx(from_center["pocket"], abs=1e-3)


def test_descriptors_cli_takes_one_docking_centre() -> None:
    '''The descriptors command needs the centre or a box, never both and never a ligand.'''

    import argparse

    import OCDocker.OCScore.CLI.descriptors as ocdescriptors

    parser = argparse.ArgumentParser()
    ocdescriptors.add_arguments(parser)

    args = parser.parse_args(["--receptor", "r.pdb", "--out", "o.json", "--center", "1", "2", "3"])
    assert args.center == [1.0, 2.0, 3.0] and args.box is None
    assert args.radius == pytest.approx(12.0)
    assert parser.parse_args(["--receptor", "r.pdb", "--out", "o.json", "--box", "box0.pdb"]).box == "box0.pdb"

    for argv in (["--receptor", "r.pdb", "--out", "o.json"],
                 ["--receptor", "r.pdb", "--out", "o.json", "--center", "1", "2", "3", "--box", "box0.pdb"],
                 ["--receptor", "r.pdb", "--out", "o.json", "--reference-ligand", "l.sdf"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_attach_context_applies_per_receptor_values() -> None:
    '''Each receptor's rows receive that receptor's own context.'''

    contract = _bundle_or_skip()
    receptor_column = contract.groups["C_receptor"][0]
    pocket_column = contract.groups["D_pocket"][0]
    frame = pd.DataFrame({"receptor": ["A", "A", "B"]})
    descriptors = {
        "A": {"receptor": {receptor_column: 1.0}, "pocket": {pocket_column: 10.0}},
        "B": {"receptor": {receptor_column: 2.0}, "pocket": {pocket_column: 20.0}},
    }

    out = ocrelinference.attach_context(frame, descriptors, contract,
                                        groups=frame["receptor"].to_numpy())

    assert list(out[receptor_column]) == [1.0, 1.0, 2.0]
    assert list(out[pocket_column]) == [10.0, 10.0, 20.0]


def test_attach_context_refuses_an_unknown_receptor() -> None:
    '''A receptor with no descriptors is an error, not a silently empty block.'''

    contract = _bundle_or_skip()
    frame = pd.DataFrame({"receptor": ["A", "B"]})
    descriptors = {"A": {"receptor": {}, "pocket": {}}}

    with pytest.raises(ValueError, match="No descriptors supplied"):
        ocrelinference.attach_context(frame, descriptors, contract,
                                      groups=frame["receptor"].to_numpy())


def test_attach_context_requires_groups_for_a_keyed_mapping() -> None:
    '''A per-receptor mapping cannot be applied without knowing each row's receptor.'''

    contract = _bundle_or_skip()
    frame = pd.DataFrame({"receptor": ["A"]})

    with pytest.raises(ValueError, match="keyed by receptor"):
        ocrelinference.attach_context(frame, {"A": {"receptor": {}, "pocket": {}}}, contract)


def test_ablation_checkpoints_declare_their_inputs() -> None:
    '''Each shipped arm takes exactly the contract columns of its blocks, in contract order.'''

    contract = _bundle_or_skip()
    for spec in ocrelcatalog.discover(families=("ablation",)):
        names = contract.columns(spec.live_blocks)
        for path in spec.checkpoints:
            net, payload = ocrelbundle.load_network(path)
            assert payload["d_in"] == len(names)
            assert list(payload["input_features"]) == names
            assert net.trunk[0].in_features == len(names)


def test_featurize_computes_only_requested_blocks() -> None:
    '''Unrequested blocks need not be present and are left at zero.'''

    contract = _bundle_or_skip()
    frame = _synthetic_frame(contract)
    keep = contract.columns("AB")
    partial = frame[keep + ["receptor"]]
    features = contract.featurize(partial, partial["receptor"].to_numpy(), blocks="AB")

    live = contract.mask("AB").astype(bool)
    assert np.all(features[:, ~live] == 0)
    with pytest.raises(ValueError, match="required feature columns are absent"):
        contract.featurize(partial, partial["receptor"].to_numpy(), blocks="ABC")


def test_baseline_requirements_follow_each_arm() -> None:
    '''Linear baselines need exactly the blocks of their arm; consensus needs only the scores.'''

    _bundle_or_skip()
    needs = ocrelbaselines.baseline_requirements(ocrelcatalog.default_root())
    assert needs["consensus_mean"] == "A"
    assert needs["random_ranking"] == ""
    if "ridge_S" in needs:
        assert needs["ridge_S"] == "A"
        assert needs["ridge_S+L"] == "AB"
        assert needs["ridge_S+L+R+P"] == "ABCD"


def _write_table_without_pocket(contract: ocrelbundle.FeatureContract, path) -> None:
    '''Write a labelled table that carries every block except the pocket descriptors.'''

    frame = _synthetic_frame(contract, rows=40)
    frame = frame[contract.columns("ABC") + ["receptor"]].copy()
    frame["kind"] = ["active"] * 8 + ["inactive"] * 32
    frame.to_csv(path, index=False)


def test_run_prediction_refuses_a_table_missing_a_needed_block(tmp_path) -> None:
    '''By default every requested model must be scorable.'''

    contract = _bundle_or_skip()
    table = tmp_path / "table.csv"
    _write_table_without_pocket(contract, table)
    config = ocrelinference.PredictionConfig(csv=table, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="does not carry every block"):
        ocrelinference.run_prediction(config)


def test_run_prediction_can_skip_models_the_table_cannot_support(tmp_path) -> None:
    '''With skipping enabled, only models whose blocks are present run, and the gap is reported.'''

    contract = _bundle_or_skip()
    table = tmp_path / "table.csv"
    _write_table_without_pocket(contract, table)
    config = ocrelinference.PredictionConfig(csv=table, output_dir=tmp_path / "out",
                                             skip_unavailable=True)
    report = ocrelinference.run_prediction(config)

    assert report["blocks_present"] == "ABC"
    assert all("P" not in name.split("_", 1)[-1] for name in report["models_scored"])
    assert "release_m1_1" in report["models_skipped"]
    assert report["control_check"]["status"] == "not_run"
