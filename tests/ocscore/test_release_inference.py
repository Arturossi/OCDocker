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


def test_ablation_scores_ignore_masked_blocks() -> None:
    '''An ablation arm must not react to blocks it was never trained on.

    This is the invariant that makes an ablation result meaningful: feeding an arm the blocks it
    never saw silently turns it into a different model.
    '''

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
