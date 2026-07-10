#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.SHAP.Dominance and .DominancePlots.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.SHAP.Dominance as ocshapdominance
import OCDocker.OCScore.Analysis.SHAP.DominancePlots as ocshapdominanceplots

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

FAMILY_SPEC = {
    "ligand_PMI": ["ligand_PMI*"],
    "receptor": ["receptor_*"],
    "scoring_function": ["plants_*"],
    "ligand_other": ["ligand_*"],
}

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def _write_replica_shap_csv(export_root, policy: str, replica_index: int, values: dict[str, list[float]]) -> None:
    '''Write a synthetic per-replica ``shap_values.csv`` at the pipeline's real path convention.

    Parameters
    ----------
    export_root : pathlib.Path
        Root of a fake ``export`` output tree.
    policy : str
        Feature-policy name (``"full_ocscore"`` maps to the ``full`` replica tree).
    replica_index : int
        Replica index.
    values : dict[str, list[float]]
        Feature-name to SHAP-value-column mapping.
    '''

    policy_dir = "full" if policy == "full_ocscore" else f"ablations/{policy}"
    shap_dir = export_root / "replica_analysis" / policy_dir / f"replica_{replica_index:03d}" / "dudez" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(shap_dir / "shap_values.csv", index=False)


def _seed_two_policies(export_root) -> None:
    '''Seed a fake export tree with ``full_ocscore`` (PMI-dominated) and ``no_pmi`` (distributed) replicas.

    Parameters
    ----------
    export_root : pathlib.Path
        Root of a fake ``export`` output tree.
    '''

    full_values = {
        "ligand_PMI1": [5.0, -5.0, 5.0],
        "ligand_PMI2": [0.1, 0.1, 0.1],
        "receptor_SASA": [0.2, 0.2, 0.2],
        "plants_plp": [0.1, 0.1, 0.1],
    }
    no_pmi_values = {
        "ligand_PMI1": [0.0, 0.0, 0.0],
        "ligand_PMI2": [0.0, 0.0, 0.0],
        "receptor_SASA": [1.0, -1.0, 1.0],
        "plants_plp": [1.0, 1.0, 1.0],
    }
    for replica_index in range(3):
        _write_replica_shap_csv(export_root, "full_ocscore", replica_index, full_values)
        _write_replica_shap_csv(export_root, "no_pmi", replica_index, no_pmi_values)


## Public ##

@pytest.mark.order(440)
def test_discover_replica_shap_csvs_resolves_full_and_ablation_paths(tmp_path):
    '''Replica discovery should resolve the reference and ablation directory conventions.'''

    _seed_two_policies(tmp_path)

    full_paths = ocshapdominance.discover_replica_shap_csvs(tmp_path, "full_ocscore", n_replicas=3)
    ablation_paths = ocshapdominance.discover_replica_shap_csvs(tmp_path, "no_pmi", n_replicas=3)

    assert len(full_paths) == 3
    assert len(ablation_paths) == 3
    assert "replica_analysis/full/replica_000" in str(full_paths[0]).replace("\\", "/")
    assert "replica_analysis/ablations/no_pmi/replica_000" in str(ablation_paths[0]).replace("\\", "/")


@pytest.mark.order(441)
def test_discover_replica_shap_csvs_raises_when_missing(tmp_path):
    '''Replica discovery should raise when no per-replica export exists for a policy.'''

    with pytest.raises(FileNotFoundError):
        ocshapdominance.discover_replica_shap_csvs(tmp_path, "missing_policy")


@pytest.mark.order(442)
def test_aggregate_family_composition_averages_across_replicas_and_sums_to_100(tmp_path):
    '''Family composition should be averaged per policy and each policy should sum to ~100%.'''

    _seed_two_policies(tmp_path)

    composition = ocshapdominance.aggregate_family_composition(
        tmp_path, ["full_ocscore", "no_pmi"], n_replicas=3, family_spec=FAMILY_SPEC,
    )

    totals = composition.groupby("policy")["relative_importance_pct_mean"].sum()
    assert totals["full_ocscore"] == pytest.approx(100.0)
    assert totals["no_pmi"] == pytest.approx(100.0)

    full_pmi_pct = composition.loc[
        (composition["policy"] == "full_ocscore") & (composition["family"] == "ligand_PMI"),
        "relative_importance_pct_mean",
    ].iloc[0]
    no_pmi_pct = composition.loc[
        (composition["policy"] == "no_pmi") & (composition["family"] == "ligand_PMI"),
        "relative_importance_pct_mean",
    ].iloc[0]
    assert full_pmi_pct > 50.0
    assert no_pmi_pct == pytest.approx(0.0)


@pytest.mark.order(443)
def test_aggregate_dominant_feature_risk_reports_mean_and_max(tmp_path):
    '''Dominant-feature risk should report the mean/max top-1 share across replicas.'''

    _seed_two_policies(tmp_path)

    risk = ocshapdominance.aggregate_dominant_feature_risk(tmp_path, ["full_ocscore", "no_pmi"], n_replicas=3)
    risk = risk.set_index("policy")

    assert risk.loc["full_ocscore", "dominant_feature"] == "ligand_PMI1"
    assert risk.loc["full_ocscore", "top1_pct_mean"] == pytest.approx(risk.loc["full_ocscore", "top1_pct_max"])
    assert risk.loc["no_pmi", "n_replicas"] == 3


@pytest.mark.order(444)
def test_save_family_composition_stacked_plot_writes_png_and_csv(tmp_path):
    '''The stacked-bar plot should write a PNG and its underlying pivot CSV.'''

    composition = pd.DataFrame({
        "policy": ["full_ocscore", "full_ocscore", "no_pmi", "no_pmi"],
        "family": ["ligand_PMI", "receptor", "ligand_PMI", "receptor"],
        "relative_importance_pct_mean": [80.0, 20.0, 0.0, 100.0],
    })

    artifacts = ocshapdominanceplots.save_family_composition_stacked_plot(
        composition,
        ["full_ocscore", "no_pmi"],
        str(tmp_path),
        policy_labels={"full_ocscore": "14", "no_pmi": "05"},
    )

    assert (tmp_path / "shap_family_composition.png").is_file()
    assert (tmp_path / "shap_family_composition.csv").is_file()
    assert artifacts["family_composition_png"].endswith("shap_family_composition.png")
