#!/usr/bin/env python3

# Description
###############################################################################
'''
Scheduler-friendly API entry points for the OCDocker docking pipeline.

This module is imported as:

from OCDocker.API.Pipeline import run_pipeline
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

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


@dataclass(frozen=True)
class PipelineRunResult:
    """Result metadata returned by :func:`run_pipeline`.

    Parameters
    ----------
    return_code : int
        Pipeline return code.
    outdir : Path
        Pipeline output directory.
    summary : Path
        Expected ``summary.json`` path.
    done_marker : Path, optional
        Optional workflow completion marker path.
    """

    return_code: int
    outdir: Path
    summary: Path
    done_marker: Optional[Path] = None

    @property
    def ok(self) -> bool:
        '''Return True when the pipeline completed successfully.

        Returns
        -------
        bool
            True if ``return_code`` equals zero.
        '''

        return self.return_code == 0


# Functions
###############################################################################
## Private ##


def _join_engines(engines: Union[str, Sequence[str]]) -> str:
    '''Normalize an engine list to the CLI comma-separated form.

    Parameters
    ----------
    engines : str or Sequence[str]
        Engine names.

    Returns
    -------
    str
        Comma-separated engine list.
    '''

    if isinstance(engines, str):
        return engines
    return ",".join(str(engine).strip() for engine in engines if str(engine).strip())


## Public ##


def run_pipeline(
        *,
        receptor: Union[str, Path],
        ligand: Union[str, Path],
        box: Union[str, Path],
        outdir: Union[str, Path],
        engines: Union[str, Sequence[str]] = ("vina", "smina", "plants"),
        rescoring_engines: Optional[Union[str, Sequence[str]]] = None,
        name: Optional[str] = None,
        all_boxes: bool = False,
        cluster_min: float = 10.0,
        cluster_max: float = 20.0,
        cluster_step: float = 0.1,
        store_db: bool = False,
        timeout: Optional[int] = None,
        config_file: Optional[Union[str, Path]] = None,
        workers: Optional[int] = None,
        tmp_dir: Optional[Union[str, Path]] = None,
        strict_engines: bool = False,
        done_marker: Optional[Union[str, Path]] = None,
        reset_runtime: bool = False,
        overwrite: bool = False,
        log_file: Optional[Union[str, Path]] = None,
        no_stdout_log: bool = False,
        no_splash: bool = True,
) -> PipelineRunResult:
    '''Run the multi-engine docking pipeline from Python.

    Parameters
    ----------
    receptor : str or Path
        Receptor structure path.
    ligand : str or Path
        Ligand structure path.
    box : str or Path
        Binding-site box path.
    outdir : str or Path
        Output directory.
    engines : str or Sequence[str], optional
        Docking engines to run, by default (``vina``, ``smina``, ``plants``).
    rescoring_engines : str or Sequence[str], optional
        Rescoring engines. If None, uses the docking engines.
    name : str, optional
        Job name. If None, the CLI default is used.
    all_boxes : bool, optional
        Run all ``box*.pdb`` files, by default False.
    cluster_min : float, optional
        Minimum RMSD clustering threshold, by default 10.0.
    cluster_max : float, optional
        Maximum RMSD clustering threshold, by default 20.0.
    cluster_step : float, optional
        RMSD clustering threshold step, by default 0.1.
    store_db : bool, optional
        Store results in the configured database, by default False.
    timeout : int, optional
        External tool timeout in seconds.
    config_file : str or Path, optional
        OCDocker config path.
    workers : int, optional
        Scheduler-provided worker count. Maps to bootstrap ``threads``.
    tmp_dir : str or Path, optional
        Job-local temporary directory.
    strict_engines : bool, optional
        Fail when any requested docking engine fails, by default False.
    done_marker : str or Path, optional
        Optional completion marker path written atomically.
    reset_runtime : bool, optional
        Reset global OCDocker runtime before running, by default False.
    overwrite : bool, optional
        Allow overwriting existing outputs, by default False.
    log_file : str or Path, optional
        Optional log file.
    no_stdout_log : bool, optional
        Disable stdout logging, by default False.
    no_splash : bool, optional
        Disable splash output, by default True.

    Returns
    -------
    PipelineRunResult
        Pipeline return code and expected output paths.
    '''

    if reset_runtime:
        import OCDocker.Initialise as ocinit
        ocinit.reset_runtime()

    outdir_path = Path(outdir)
    done_marker_path = Path(done_marker) if done_marker is not None else None
    threads = int(workers) if workers is not None else None
    if threads is not None and threads < 1:
        threads = 1

    from OCDocker.CLI.pipeline import cmd_pipeline

    args = argparse.Namespace(
        receptor=str(receptor),
        ligand=str(ligand),
        box=str(box),
        outdir=str(outdir_path),
        engines=_join_engines(engines),
        rescoring_engines=(
            None if rescoring_engines is None else _join_engines(rescoring_engines)
        ),
        name=name,
        all_boxes=all_boxes,
        cluster_min=cluster_min,
        cluster_max=cluster_max,
        cluster_step=cluster_step,
        store_db=store_db,
        timeout=timeout,
        strict_engines=strict_engines,
        done_marker=str(done_marker_path) if done_marker_path is not None else None,
        config_file=str(config_file) if config_file is not None else None,
        threads=threads,
        tmp_dir=str(tmp_dir) if tmp_dir is not None else None,
        multiprocess=True if threads is None else threads > 1,
        update=False,
        output_level=1,
        overwrite=overwrite,
        log_file=str(log_file) if log_file is not None else None,
        no_stdout_log=no_stdout_log,
        no_splash=no_splash,
        _api_call=True,
    )
    return_code = cmd_pipeline(args)

    return PipelineRunResult(
        return_code=return_code,
        outdir=outdir_path,
        summary=outdir_path / "summary.json",
        done_marker=done_marker_path,
    )
