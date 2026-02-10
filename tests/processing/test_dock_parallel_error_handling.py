#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for parallel docking error normalization in OCDocker.Processing.Dock.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Processing.Dock as ocdock

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


@pytest.mark.order(97)
def test_run_dock_parallel_normalizes_non_io_worker_exceptions(monkeypatch):
    class DummyConfig:
        available_cores = 2
        logdir = "/tmp"

    class DummyPool:
        def __init__(self, _cores):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, _fn, _args):
            raise RuntimeError("simulated worker crash")

    logged = {}
    sentinel = 987654321

    def fake_print_error_log(message, path):
        logged["message"] = message
        logged["path"] = path

    def fake_docking_failed(message, level):
        logged["docking_failed_message"] = message
        logged["level"] = level
        return sentinel

    monkeypatch.setattr(ocdock, "Pool", DummyPool)
    monkeypatch.setattr(ocdock, "get_config", lambda: DummyConfig())
    monkeypatch.setattr(ocdock, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(ocdock.ocprint, "print_error_log", fake_print_error_log)
    monkeypatch.setattr(ocdock.ocerror.Error, "docking_failed", fake_docking_failed)

    rc = ocdock.__run_dock_parallel(  # type: ignore[attr-defined]
        complexList=[("/tmp/proteinA", ["/tmp/proteinA/lig1"])],
        archive="dudez",
        dockingAlgorithm="vina",
        overwrite=False,
        digestFormat="json",
        desc="Docking test",
        all_boxes=False,
    )

    assert rc == sentinel
    assert "RuntimeError" in logged["message"]
    assert "simulated worker crash" in logged["message"]
    assert logged["path"].endswith("/dudez_docking_report.log")
