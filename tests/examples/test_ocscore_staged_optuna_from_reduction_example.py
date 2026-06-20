#!/usr/bin/env python3

# Description
###############################################################################
"""
Smoke tests for staged Optuna with raw unreduced modeling inputs.
"""

# Imports
###############################################################################
import json

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

# License
###############################################################################
"""OCDocker
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
"""


def _load_example_module():
    from OCDocker.OCScore.CLI import train

    return train


def _write_raw_merged_payload(tmp_path: Path) -> Path:
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    pdbbind = pd.DataFrame({
        "name": ["p1", "p2", "p3"],
        "receptor": ["r1", "r1", "r2"],
        "dataset": ["pdbbind", "pdbbind", "pdbbind"],
        "experimental": [7.1, 6.8, 6.5],
        "f0": [0.1, 0.2, 0.3],
        "f1": [1.0, 1.1, 1.2],
    })
    dudez = pd.DataFrame({
        "name": ["d1", "d2", "d3", "d4"],
        "kind": ["ligands", "decoys", "ligands", "decoys"],
        "receptor": ["r1", "r1", "r2", "r2"],
        "dataset": ["dudez", "dudez", "dudez", "dudez"],
        "f0": [0.9, -0.2, 0.8, -0.1],
        "f1": [1.9, -1.2, 1.8, -1.1],
    })
    merged = pd.concat([pdbbind, dudez], ignore_index=True)
    merged.to_csv(payload_dir / "merged_input_dataset.csv", index=False)
    return payload_dir


@pytest.mark.order(495)
def test_staged_optuna_example_loads_and_validates_raw_input(tmp_path):
    example = _load_example_module()
    payload_dir = _write_raw_merged_payload(tmp_path)
    output_dir = tmp_path / "optuna_output"
    output_dir.mkdir()

    from OCDocker.OCScore.Utils.RawModelingInput import load_raw_modeling_input

    raw_input = load_raw_modeling_input(raw_input_dir=payload_dir)
    assert raw_input.merged.shape[0] == 7

    with pytest.raises(ValueError, match="no longer supported"):
        example.load_reduction_artifacts(payload_dir)

    pdbbind = raw_input.merged[raw_input.merged["dataset"] == "pdbbind"].copy()
    dudez = raw_input.merged[raw_input.merged["dataset"] == "dudez"].copy()
    selected = ["f0", "f1"]
    example.validate_selected_features(pdbbind, dudez, selected)
    with pytest.raises(ValueError, match="metadata/target columns"):
        example.validate_selected_features(pdbbind, dudez, ["experimental"])
    pdbbind_clean, dropped_targets = example.prepare_pdbbind_for_optuna(
        pdbbind,
        target_column="experimental",
    )
    dudez_clean, class_counts, dropped_unknown = example.prepare_dudez_for_optuna(
        dudez,
        kind_column="kind",
        positive_kind="ligands",
        negative_kind="decoys",
    )
    context = example.build_protocol_context(
        pdbbind_df=pdbbind_clean,
        dudez_df=dudez_clean,
        selected_features=selected,
        output_dir=output_dir,
        random_seed=42,
        metadata={"raw_input_dir": str(payload_dir)},
    )

    assert dropped_targets == 0
    assert dropped_unknown == 0
    assert class_counts == {0: 2, 1: 2}
    assert context.selected_features == selected
