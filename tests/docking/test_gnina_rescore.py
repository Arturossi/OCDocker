#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Gnina rescoring helpers.

Usage:

pytest tests/docking/test_gnina_rescore.py
'''

# Imports
###############################################################################
from __future__ import annotations

import types

from pathlib import Path

import pytest

import OCDocker.Docking.Gnina as ocgnina

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

def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding = 'utf-8')


## Public ##

@pytest.mark.order(40)
def test_gnina_rescore_log_discovery_and_parsing(tmp_path, monkeypatch):
    '''Test Gnina rescoring log discovery and parser helpers.'''

    f1 = tmp_path / 'lig_split_1_default_rescoring.log'
    f2 = tmp_path / 'lig_split_2_default_rescoring.log'
    f3 = tmp_path / 'lig_split_1_cnn_dense_1_3_rescoring.log'
    _write(f1, 'Affinity:            -7.0 (kcal/mol)\n')
    _write(f2, 'Affinity:            -6.5 (kcal/mol)\n')
    _write(f3, 'Affinity:            -6.8 (kcal/mol)\n')

    cfg = types.SimpleNamespace(
        gnina = types.SimpleNamespace(
            scoring = 'default',
            scoring_functions = ['default'],
            cnn = 'default',
            cnn_models = ['dense_1_3'],
        )
    )
    monkeypatch.setattr(ocgnina, 'get_config', lambda: cfg)

    paths = ocgnina.get_rescore_log_paths(str(tmp_path))
    assert set(paths) == {str(f1), str(f2), str(f3)}

    data = ocgnina.read_rescore_logs(paths)
    assert data == {
        'rescoring_default_1': -7.0,
        'rescoring_default_2': -6.5,
        'rescoring_cnn_dense_1_3_1': -6.8,
    }

    best = ocgnina.read_rescore_logs(paths, onlyBest = True)
    assert best == {
        'rescoring_default_1': -7.0,
        'rescoring_cnn_dense_1_3_1': -6.8,
    }


@pytest.mark.order(41)
def test_gnina_run_rescore_builds_expected_command(tmp_path, monkeypatch):
    '''Test Gnina rescoring command generation and execution path.'''

    conf_file = tmp_path / 'conf_gnina.conf'
    _write(
        conf_file,
        'receptor = rec.pdbqt\n'
        'center_x = 0\n'
        'center_y = 0\n'
        'center_z = 0\n'
        'size_x = 10\n'
        'size_y = 10\n'
        'size_z = 10\n',
    )

    out_path = tmp_path / 'rescoring'
    out_path.mkdir()
    lig = out_path / 'lig_split_1.pdbqt'
    _write(lig, 'MODEL 1\nATOM\nENDMDL\n')

    cfg = types.SimpleNamespace(
        gnina = types.SimpleNamespace(
            executable = '/fake/gnina',
            no_gpu = 'yes',
            device = '0',
        )
    )
    monkeypatch.setattr(ocgnina, 'get_config', lambda: cfg)

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, logFile = '', cwd = '', timeout = None):
        _ = logFile, cwd, timeout
        seen_cmds.append(list(cmd))
        if "--log" in cmd:
            log_index = cmd.index("--log")
            if log_index + 1 < len(cmd):
                Path(cmd[log_index + 1]).write_text('Affinity: -7.2 (kcal/mol)\n', encoding = 'utf-8')
        return 0

    monkeypatch.setattr(ocgnina.ocrun, 'run', _fake_run)

    result = ocgnina.run_rescore(
        confFile = str(conf_file),
        ligands = [str(lig)],
        outPath = str(out_path),
        scoring_function = 'default',
        splitLigand = False,
        overwrite = True,
    )

    assert result is None
    assert seen_cmds, 'Expected at least one gnina rescoring command'
    cmd = seen_cmds[0]
    assert '--score_only' in cmd
    assert '--scoring' in cmd
    assert 'default' in cmd
    assert '--no_gpu' in cmd


@pytest.mark.order(42)
def test_gnina_instance_run_rescore_uses_default_scoring_fallback(monkeypatch):
    '''Test class-level Gnina run_rescore fallback to default scoring key.'''

    cfg = types.SimpleNamespace(
        gnina = types.SimpleNamespace(
            scoring = 'default',
            scoring_functions = ['default'],
            cnn = 'default',
            cnn_models = ['dense'],
        )
    )
    monkeypatch.setattr(ocgnina, 'get_config', lambda: cfg)

    calls: list[tuple[str, str, bool]] = []

    def _fake_run_rescore(confFile, ligands, outPath, scoring_function, logFile = '', splitLigand = True, overwrite = False, cnn_model = '', disable_cnn = False):
        _ = confFile, ligands, outPath, logFile, splitLigand, overwrite
        calls.append((scoring_function, cnn_model, disable_cnn))
        return None

    monkeypatch.setattr(ocgnina, 'run_rescore', _fake_run_rescore)

    runner = ocgnina.Gnina.__new__(ocgnina.Gnina)
    runner.config = 'conf_gnina.conf'

    runner.run_rescore('out', 'ligand.pdbqt', splitLigand = False, skipDefaultScoring = False)
    assert calls == [
        ('default', '', True),
        ('default', 'dense', False),
    ]

    calls.clear()
    runner.run_rescore('out', 'ligand.pdbqt', splitLigand = False, skipDefaultScoring = True)
    assert calls == [
        ('default', 'dense', False),
    ]
