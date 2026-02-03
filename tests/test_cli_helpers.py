#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI helper coverage for lightweight helpers.

Usage:

pytest tests/test_cli_helpers.py
'''

# Imports
###############################################################################
import argparse
import pytest

from pathlib import Path

from OCDocker.CLI.__init__ import _preparse_global_args, _require_file, build_parser

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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(21)
def test_build_parser_subcommands_and_parse():
    parser = build_parser()
    # A couple of subcommands should parse cleanly
    ns = parser.parse_args(["version"])  # sets func
    assert callable(getattr(ns, "func", None))
    ns2 = parser.parse_args(["init-config"])  # also sets func
    assert callable(getattr(ns2, "func", None))


@pytest.mark.order(19)
def test_preparse_global_args_reads_scattered_flags(tmp_path):
    cfg = tmp_path / "OCDocker.cfg"
    argv = [
        "vs", "--engine", "vina", "--output-level", "4",
        "--conf", str(cfg), "--overwrite", "--no-stdout-log",
        "--multiprocess", "-u",
    ]
    ns = _preparse_global_args(argv)
    assert ns.output_level == 4
    assert ns.config_file == str(cfg)
    assert ns.overwrite is True
    assert ns.no_stdout_log is True
    assert ns.multiprocess is True
    assert ns.update is True


@pytest.mark.order(20)
def test_require_file_valid_and_errors(tmp_path):
    # Valid path returns Path
    f = tmp_path / "ok.txt"
    f.write_text("x")
    p = _require_file(str(f), "--file")
    assert isinstance(p, Path)
    assert p.exists()

    # Missing path raises SystemExit with code 2
    with pytest.raises(SystemExit) as ei:
        _require_file(str(tmp_path / "missing.txt"), "--file")
    assert ei.value.code == 2

    # Ellipsis placeholder triggers exit 2
    with pytest.raises(SystemExit) as ei2:
        _require_file("…/placeholder", "--file")
    assert ei2.value.code == 2
