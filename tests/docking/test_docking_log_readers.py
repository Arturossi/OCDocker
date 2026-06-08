#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Vina/Smina log readers using real module imports.
'''

# Imports
###############################################################################
import pytest

pytest.importorskip("rdkit")

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


# Functions
###############################################################################
## Public ##

def test_smina_log_parsing(tmp_path):
    from OCDocker.Config import get_config
    from OCDocker.Docking.Smina import read_log, read_rescoring_log

    log_file = tmp_path / "smina.log"
    log_file.write_text(
        "header\n-----+------------+----------+----------+\n"
        "    1 -6.0 0.0 0.0\n",
        encoding="utf-8",
    )

    config = get_config()
    smina_scoring = config.smina.scoring

    result = read_log(str(log_file))
    expected = {1: {smina_scoring: -6.0}}
    assert result == expected

    rescoring = tmp_path / "smina_res.log"
    rescoring.write_text("Header\nAffinity: -6.5 (kcal/mol)\n", encoding="utf-8")
    affinity = read_rescoring_log(str(rescoring))
    assert affinity == -6.5


def test_vina_log_parsing(tmp_path):
    from OCDocker.Config import get_config
    from OCDocker.Docking.Vina import read_log, read_rescoring_log

    log_file = tmp_path / "vina.log"
    log_file.write_text(
        "header\n-----+------------+----------+----------+\n"
        "    1 -8.0 0.0 0.0\n    2 -7.5 1.0 2.0\n",
        encoding="utf-8",
    )

    config = get_config()
    vina_scoring = config.vina.scoring

    result = read_log(str(log_file))
    expected = {
        1: {vina_scoring: -8.0},
        2: {vina_scoring: -7.5},
    }
    assert result == expected

    rescoring = tmp_path / "vina_res.log"
    rescoring.write_text(
        "Some line\nEstimated Free Energy of Binding    -8.3 (kcal/mol)\n",
        encoding="utf-8",
    )
    affinity = read_rescoring_log(str(rescoring))
    assert affinity == -8.3
