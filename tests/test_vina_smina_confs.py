#!/usr/bin/env python3

'''
Increase coverage of Vina/Smina config writers by asserting content under
different Initialise settings (custom/user grid flags).
'''

from __future__ import annotations
import sys
import types
from pathlib import Path

import pytest

import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import importlib, sys


def _install_init_defaults(monkeypatch, **overrides):
    init = types.ModuleType('OCDocker.Initialise')
    # Defaults for Vina
    init.vina_energy_range = "10"
    init.vina_exhaustiveness = "5"
    init.vina_num_modes = "3"
    init.vina_scoring = "vina"
    # Defaults for Smina
    init.smina_custom_scoring = 'no'
    init.smina_custom_atoms = 'no'
    init.smina_minimize_iters = '0'
    init.smina_approximation = 'spline'
    init.smina_factor = '32'
    init.smina_force_cap = '10'
    init.smina_user_grid = 'no'
    init.smina_user_grid_lambda = 'no'
    init.smina_energy_range = '10'
    init.smina_exhaustiveness = '5'
    init.smina_num_modes = '3'
    init.smina_scoring = 'vinardo'
    init.smina_scoring_functions = ['vinardo']
    # Printing stub
    err = types.ModuleType('OCDocker.Error')
    class RL(int):
        DEBUG=5; SUCCESS=4; INFO=3; WARNING=2; ERROR=1; NONE=0


    class E:
        @staticmethod
        def ok(*a, **k):
            return 0

        @staticmethod
        def file_not_exist(*a, **k): 
            return 101

        @staticmethod
        def read_file(*a, **k):
            return 102

        @staticmethod
        def write_file(*a, **k):
            return 103

        @staticmethod
        def wrong_type(*a, **k):
            return 200


    err.Error = E  # type: ignore
    err.ReportLevel = RL  # type: ignore
    init.ocerror = err  # type: ignore
    # Merge overrides
    for k, v in overrides.items():
        setattr(init, k, v)
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init)


@pytest.mark.order(97)
def test_box_to_vina_creates_conf(tmp_path, monkeypatch):
    _install_init_defaults(monkeypatch)
    # Use the tracked box fixture from the repo
    root = Path(__file__).resolve()
    while root.name != 'OCDocker':
        root = root.parent
    box = root / 'test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb'
    conf = tmp_path / 'nested' / 'vina_conf.txt'
    rc = ocvina.box_to_vina(str(box), str(conf), 'rec.pdbqt')
    assert rc == 0
    txt = conf.read_text()
    assert 'receptor = rec.pdbqt' in txt
    assert 'center_x = 36.552' in txt
    assert 'size_z = 102.582' in txt


@pytest.mark.order(98)
def test_gen_smina_conf_with_custom_flags(tmp_path, monkeypatch):
    '''Test gen_smina_conf with custom flags from Config.'''
    
    # Mock Config with custom attributes
    def mock_get_config():
        class MockSminaConfig:
            custom_scoring = "custom.score"
            custom_atoms = "atoms.def"
            minimize_iters = "5"
            user_grid = "grid.map"
            user_grid_lambda = "0.5"
            approximation = "spline"
            factor = "32"
            force_cap = "10"
            energy_range = "10"
            exhaustiveness = "5"
            num_modes = "3"
        class MockConfig:
            smina = MockSminaConfig()
            logdir = str(tmp_path)
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    root = Path(__file__).resolve()
    while root.name != 'OCDocker':
        root = root.parent
    box = root / 'test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb'
    conf = tmp_path / 'smina' / 'conf.txt'
    conf.parent.mkdir(parents=True, exist_ok=True)
    rc = ocsmina.gen_smina_conf(str(box), str(conf), 'rec.pdbqt')
    assert rc == 0
    txt = conf.read_text()
    # Check custom flags are written
    assert 'receptor = rec.pdbqt' in txt
    assert 'custom_scoring = custom.score' in txt
    assert 'custom_atoms = atoms.def' in txt
    assert 'minimize_iters = 5' in txt
    assert 'user_grid = grid.map' in txt
    assert 'user_grid_lambda = 0.5' in txt
    assert 'center_x = 36.552' in txt
    assert 'size_z = 102.582' in txt
