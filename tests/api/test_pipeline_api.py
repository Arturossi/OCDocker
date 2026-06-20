#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for scheduler-friendly pipeline API wrappers.
'''

# Imports
###############################################################################
from pathlib import Path

from OCDocker.API.Pipeline import PipelineRunResult, run_pipeline

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
## Public ##


def test_run_pipeline_builds_scheduler_aware_namespace(monkeypatch, tmp_path):
    '''API wrapper should build a scheduler-aware CLI namespace.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    captured = {}

    def fake_cmd_pipeline(args):
        '''Capture CLI arguments without running external docking tools.

        Parameters
        ----------
        args : argparse.Namespace
            Namespace passed by the API wrapper.

        Returns
        -------
        int
            Simulated successful return code.
        '''

        captured["args"] = args
        return 0

    import OCDocker.CLI.pipeline as cli_pipeline
    monkeypatch.setattr(cli_pipeline, "cmd_pipeline", fake_cmd_pipeline)

    result = run_pipeline(
        receptor="receptor.pdb",
        ligand="ligand.smi",
        box="box0.pdb",
        outdir=tmp_path / "out",
        engines=["vina", "smina"],
        rescoring_engines=["vina", "oddt"],
        workers=0,
        tmp_dir=tmp_path / "tmp",
        strict_engines=True,
        done_marker=tmp_path / "done.json",
        config_file="OCDocker.cfg",
        overwrite=True,
    )

    args = captured["args"]
    assert isinstance(result, PipelineRunResult)
    assert result.ok is True
    assert result.summary == Path(tmp_path / "out" / "summary.json")
    assert args._api_call is True
    assert args.engines == "vina,smina"
    assert args.rescoring_engines == "vina,oddt"
    assert args.threads == 1
    assert args.multiprocess is False
    assert args.tmp_dir == str(tmp_path / "tmp")
    assert args.strict_engines is True
    assert args.done_marker == str(tmp_path / "done.json")
    assert args.config_file == "OCDocker.cfg"
    assert args.overwrite is True
