#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for the unified leakage-safe OCScore modeling protocol.'''

# Imports
###############################################################################
import json
from pathlib import Path

import pandas as pd
import pytest

from OCDocker.OCScore.CLI import train as train_cli
from OCDocker.OCScore.Optimization.StagedTrainProtocol import load_staged_train_protocol
from OCDocker.OCScore.Optimization.StagedTrainProtocol import resolve_protocol_path
from OCDocker.OCScore.Utils.RawModelingInput import FORBIDDEN_TRAINING_ARTIFACTS
from OCDocker.OCScore.Utils.RawModelingInput import load_raw_modeling_input
from OCDocker.OCScore.Utils.RawModelingInput import reject_precomputed_training_artifacts
from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import fit_train_only_feature_reduction

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


def _raw_wide_df(n_pdb: int = 8, n_dudez: int = 6) -> pd.DataFrame:
    pdb = pd.DataFrame(
        {
            "name": [f"p{i}" for i in range(n_pdb)],
            "receptor": ["r1"] * n_pdb,
            "ligand": [f"l{i}" for i in range(n_pdb)],
            "dataset": ["pdbbind"] * n_pdb,
            "experimental": [7.0 + 0.1 * i for i in range(n_pdb)],
            "SASA": [100.0 + i for i in range(n_pdb)],
            "VINA_VINA": [-7.0 + 0.1 * i for i in range(n_pdb)],
            "SMINA_VINA": [-7.1 + 0.1 * i for i in range(n_pdb)],
        }
    )
    dudez = pd.DataFrame(
        {
            "name": [f"d{i}" for i in range(n_dudez)],
            "receptor": ["r1"] * (n_dudez // 2) + ["r2"] * (n_dudez - n_dudez // 2),
            "ligand": [f"dl{i}" for i in range(n_dudez)],
            "dataset": ["dudez"] * n_dudez,
            "kind": ["ligands", "decoys"] * (n_dudez // 2),
            "SASA": [200.0 + i for i in range(n_dudez)],
            "VINA_VINA": [-6.0 + 0.1 * i for i in range(n_dudez)],
            "SMINA_VINA": [-6.1 + 0.1 * i for i in range(n_dudez)],
        }
    )
    return pd.concat([pdb, dudez], ignore_index=True)


def _write_raw_prepare_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw_prepare"
    raw_dir.mkdir()
    merged = _raw_wide_df()
    merged.to_csv(raw_dir / "merged_input_dataset.csv", index=False)
    return raw_dir


def _write_forbidden_dir(tmp_path: Path, artifact_name: str) -> Path:
    payload = tmp_path / artifact_name.replace(".", "_")
    payload.mkdir()
    if artifact_name.endswith(".json"):
        payload.joinpath(artifact_name).write_text(json.dumps(["f0"]) + "\n", encoding="utf-8")
    else:
        pd.DataFrame({"f0": [1.0]}).to_csv(payload / artifact_name, index=False)
    return payload


@pytest.mark.parametrize("artifact_name", FORBIDDEN_TRAINING_ARTIFACTS)
@pytest.mark.order(420)
def test_training_rejects_forbidden_artifacts(tmp_path, artifact_name):
    payload = _write_forbidden_dir(tmp_path, artifact_name)
    with pytest.raises(ValueError, match="not supported"):
        reject_precomputed_training_artifacts(payload)


@pytest.mark.order(421)
def test_training_rejects_reduction_archive_flag(tmp_path):
    raw_dir = _write_raw_prepare_dir(tmp_path)
    protocol = resolve_protocol_path("smoke-test")
    output_dir = tmp_path / "train_out"
    args = train_cli.build_argparser().parse_args([
        "--protocol", str(protocol),
        "--output-dir", str(output_dir),
        "--raw-input-dir", str(raw_dir),
        "--reduction-archive", str(raw_dir),
    ])
    with pytest.raises(ValueError, match="no longer supported"):
        train_cli.main_from_args(args)


@pytest.mark.order(422)
def test_training_accepts_merged_raw_dataset(tmp_path):
    raw_dir = _write_raw_prepare_dir(tmp_path)
    loaded = load_raw_modeling_input(merged_input=raw_dir / "merged_input_dataset.csv")
    assert loaded.merged.shape[0] == 14
    assert loaded.merged_hash


@pytest.mark.order(423)
def test_training_drops_empty_rows_from_merged_raw_dataset(tmp_path):
    raw_dir = _write_raw_prepare_dir(tmp_path)
    merged_path = raw_dir / "merged_input_dataset.csv"
    n_columns = len(pd.read_csv(merged_path, nrows=0).columns)
    with merged_path.open("a", encoding="utf-8") as handle:
        handle.write("," * (n_columns - 1) + "\n")

    loaded = load_raw_modeling_input(merged_input=merged_path)

    assert loaded.merged.shape[0] == 14


@pytest.mark.order(423)
def test_training_accepts_separate_raw_pdbbind_and_dudez(tmp_path):
    merged = _raw_wide_df()
    pdb = merged[merged["dataset"] == "pdbbind"].drop(columns=["kind"], errors="ignore")
    dudez = merged[merged["dataset"] == "dudez"]
    pdb_path = tmp_path / "PDBbind.csv"
    dudez_path = tmp_path / "DUDEz.csv"
    pdb.to_csv(pdb_path, index=False)
    dudez.to_csv(dudez_path, index=False)
    loaded = load_raw_modeling_input(pdbbind_input=pdb_path, dudez_input=dudez_path)
    assert loaded.pdbbind_hash
    assert loaded.dudez_hash
    assert loaded.merged.shape[0] == 14


@pytest.mark.order(424)
def test_feature_reduction_fit_uses_training_rows_only():
    merged = _raw_wide_df(n_pdb=12, n_dudez=0)
    train_df = merged.iloc[:6].copy()
    holdout = merged.iloc[6:].copy()
    train_artifact = fit_train_only_feature_reduction(train_df)
    holdout_artifact = fit_train_only_feature_reduction(holdout)
    assert train_artifact.feature_selection.fit_row_count == 6
    assert holdout_artifact.feature_selection.fit_row_count == 6
    assert train_artifact.feature_selection.fit_split == "train"
    assert train_artifact.feature_selection.scope == "train_only"


@pytest.mark.order(425)
def test_ambiguous_merged_and_separate_inputs_fail(tmp_path):
    raw_dir = _write_raw_prepare_dir(tmp_path)
    with pytest.raises(ValueError, match="exactly one raw input mode"):
        load_raw_modeling_input(
            merged_input=raw_dir / "merged_input_dataset.csv",
            raw_input_dir=raw_dir,
        )


@pytest.mark.order(426)
def test_protocol_dataclass_has_no_perform_hard_checks():
    protocol = load_staged_train_protocol(resolve_protocol_path("production"))
    assert "perform_hard_checks" not in protocol.__dataclass_fields__


@pytest.mark.order(427)
def test_train_cli_help_does_not_advertise_reduction_archive():
    parser = train_cli.build_argparser()
    help_text = parser.format_help()
    assert "--reduction-archive" not in help_text
    assert "raw unreduced" in help_text.lower()


@pytest.mark.order(428)
def test_load_reduction_artifacts_rejects_reduced_csv(tmp_path):
    payload = tmp_path / "deprecated"
    payload.mkdir()
    pd.DataFrame({"f0": [1.0]}).to_csv(payload / "reduced_pdbbind.csv", index=False)
    with pytest.raises(ValueError, match="not supported"):
        train_cli.load_reduction_artifacts(payload)
