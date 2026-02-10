#!/usr/bin/env python3

# Description
###############################################################################
'''
Regression tests for OCDocker.Initialise config parsing behavior.
'''

# Imports
###############################################################################
import OCDocker.Initialise as ocinit

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


def test_parse_config_preserves_hash_inside_value(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    cfg.write_text(
        "PASSWORD = abc#123\n"
        "vina_exhaustiveness = 5\n"
        "plants_cluster_structures = 3\n"
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["PASSWORD"] == "abc#123"


def test_parse_config_supports_inline_comments_without_corrupting_hash_values(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    cfg.write_text(
        "USER = alice # inline comment\n"
        "vina_scoring_functions = vina#,vinardo\n"
        "vina_exhaustiveness = 5\n"
        "plants_cluster_structures = 3\n"
    )

    parsed = ocinit._parse_config_file(str(cfg))

    assert parsed["USER"] == "alice"
    assert parsed["vina_scoring_functions"] == ["vina#", "vinardo"]
