#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for OCScore architecture figure generation.
'''

# Imports
###############################################################################
from __future__ import annotations

import json

import pytest

import OCDocker.OCScore.Analysis.Plotting.ArchitecturePlots as ocarchplot
import OCDocker.OCScore.Optimization.ModelExport as ocexport


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

def _pdbbind_architecture() -> dict:
    '''Build a representative PDBbind architecture document.

    Returns
    -------
    dict
        Architecture document used by figure-generation tests.
    '''

    return {
        "task": "pdbbind_regression",
        "input_size": 64,
        "encoder": {
            "hidden_sizes": [32, 16],
            "latent_dim": 8,
            "activation": "GELU",
            "resolved": {
                "hidden_sizes": [32, 16],
                "latent_dim": 8,
                "projection_dim": 4,
            },
        },
        "projection": {"enabled": True, "projection_dim": 4},
        "decoder": {"enabled": True, "hidden_sizes": [16, 32], "lambda_rec": 0.1},
        "dae": {
            "enabled": True,
            "noise_type": "mask",
            "mask_prob": 0.1,
            "gaussian_std": 0.0,
        },
        "regression_head": {"loss": "mse"},
    }


## Public ##

@pytest.mark.order(446)
def test_save_architecture_figures_from_export_dir(tmp_path):
    '''Test architecture figure generation from an export directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    '''

    export_dir = tmp_path / "best_model"
    export_dir.mkdir()
    (export_dir / ocexport.ARCHITECTURE_FILENAME).write_text(
        json.dumps(_pdbbind_architecture()),
        encoding="utf-8",
    )

    written = ocarchplot.save_architecture_figures(
        export_dir,
        tmp_path / "figures",
        formats=("png", "svg"),
        dpi=80,
    )

    assert written["source"].endswith(ocexport.ARCHITECTURE_FILENAME)
    for key in ("png", "svg"):
        path = tmp_path / "figures" / f"architecture.{key}"
        assert written[key] == str(path.resolve())
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.order(447)
def test_manual_layer_architecture_yaml_is_supported(tmp_path):
    '''Test manual YAML architecture parsing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    '''

    path = tmp_path / "manual_architecture.yaml"
    path.write_text(
        "\n".join(
            [
                "task: custom_ocscore",
                "layers:",
                "  - {label: Input, dim: 128, kind: input}",
                "  - {label: Hidden 1, dim: 64, kind: encoder, activation: GELU}",
                "  - {label: Output, dim: 1, kind: head}",
            ]
        ),
        encoding="utf-8",
    )

    document, source = ocarchplot.load_architecture_document(path)
    diagram = ocarchplot.normalize_architecture(document)

    assert source == path
    assert diagram.task == "custom_ocscore"
    assert [block.dim for block in diagram.main] == [128, 64, 1]
    assert diagram.main[1].detail == "GELU"
