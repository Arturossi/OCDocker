#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Smina log parsing helpers.

Usage:

pytest tests/test_Smina_logs.py
'''

# Imports
###############################################################################
import pytest

from pathlib import Path

import OCDocker.Docking.Smina as ocsmina

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

def make_log(path: Path, lines: str) -> Path:
    path.write_text(lines)
    return path


@pytest.mark.order(9)
def test_read_log(tmp_path):
    log_file = tmp_path / "dock.log"
    # minimal smina log section
    lines = (
        "-----+------------+----------+----------+\n"
        "    1 -7.5 0 0\n"
        "    2 -6.5 0 0\n"
    )
    make_log(log_file, lines)

    from OCDocker.Config import get_config
    config = get_config()
    smina_scoring = config.smina.scoring
    
    data = ocsmina.read_log(str(log_file))
    assert data[1][smina_scoring] == -7.5 # type: ignore
    assert data[2][smina_scoring] == -6.5 # type: ignore

    best = ocsmina.read_log(str(log_file), onlyBest=True)
    assert list(best.keys()) == [1]
    assert best[1][smina_scoring] == -7.5 # type: ignore


@pytest.mark.order(10)
def test_rescoring_logs(tmp_path):
    f1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    f2 = tmp_path / "lig_split_2_vinardo_rescoring.log"
    make_log(f1, "Affinity:            -7.0 (kcal/mol)\n")
    make_log(f2, "Affinity:            -6.5 (kcal/mol)\n")

    paths = ocsmina.get_rescore_log_paths(str(tmp_path))
    assert set(paths) == {str(f1), str(f2)}

    val1 = ocsmina.read_rescoring_log(str(f1))
    assert val1 == -7.0

    data = ocsmina.read_rescore_logs(paths)
    expected = {
        "rescoring_vinardo_1": -7.0,
        "rescoring_vinardo_2": -6.5,
    }
    assert data == expected

    best = ocsmina.read_rescore_logs(paths, onlyBest=True)
    assert best == {"rescoring_vinardo_1": -7.0}
