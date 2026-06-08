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
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
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
