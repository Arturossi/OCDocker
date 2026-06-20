#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for reproducibility manifest helpers.
'''

# Imports
###############################################################################
import json

from pathlib import Path

import pytest

import OCDocker.Toolbox.Reproducibility as ocrepro

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

## Public ##

@pytest.mark.order(468)
def test_write_reproducibility_manifest_writes_json(monkeypatch, tmp_path):
    expected = {"schema_version": 1, "ocdocker": {"version": "0.0.test"}}
    monkeypatch.setattr(
        ocrepro,
        "generate_reproducibility_manifest",
        lambda include_python_packages=True: {
            **expected,
            "python_package_count": 1 if include_python_packages else 0,
        },
    )

    output = tmp_path / "manifest" / "manifest.json"
    result = ocrepro.write_reproducibility_manifest(str(output), include_python_packages=False)

    assert result["python_package_count"] == 0
    assert output.exists()

    loaded = json.loads(Path(output).read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["ocdocker"]["version"] == "0.0.test"
