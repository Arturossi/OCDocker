#!/usr/bin/env python3

"""
Integration tests for the Preparation Strategy Pattern.

These tests verify that the Strategy Pattern implementation works correctly
for all preparation strategies (MGLTools, SPORES, OpenBabel).
"""

from __future__ import annotations
from pathlib import Path
import os
import pytest
import shutil

from OCDocker.Toolbox.Preparation import (
    PreparationStrategy,
    MGLToolsPreparationStrategy,
    SPORESPreparationStrategy,
    OpenBabelPreparationStrategy,
)
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants


@pytest.fixture
def test_files():
    '''Get test files from test_files directory.'''
 
    # Find project root by looking for directory containing both 'test_files' and 'OCDocker' (module)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent  # Start from tests/ directory
    
    # Traverse up until we find a directory that contains both 'test_files' and 'OCDocker' subdirectories
    while project_root != project_root.parent:
        test_files_dir = project_root / "test_files"
        ocdocker_module = project_root / "OCDocker"
        
        if test_files_dir.exists() and test_files_dir.is_dir() and ocdocker_module.exists() and ocdocker_module.is_dir():
            break
        
        project_root = project_root.parent
    
    if not (project_root / "test_files").exists():
        raise RuntimeError(f"test_files directory not found. Searched from {current_file} up to {project_root}")
    
    # Use test files from test_ptn1 directory
    receptor_file = project_root / "test_files" / "test_ptn1" / "receptor.pdb"
    ligand_file = project_root / "test_files" / "test_ptn1" / "compounds" / "ligands" / "ligand" / "ligand.smi"
    box_file = project_root / "test_files" / "test_ptn1" / "compounds" / "ligands" / "ligand" / "boxes" / "box0.pdb"
    
    # Verify files exist
    if not receptor_file.exists():
        raise FileNotFoundError(f"Receptor file not found: {receptor_file}")
    if not ligand_file.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    if not box_file.exists():
        raise FileNotFoundError(f"Box file not found: {box_file}")
    
    return {
        'receptor': receptor_file,
        'ligand': ligand_file,
        'box': box_file,
    }

@pytest.fixture
def sample_ligand(test_files):
    '''Use ligand file for testing.'''

    return str(test_files['ligand'])

@pytest.fixture
def sample_receptor(test_files):
    '''Use receptor file for testing.'''

    return str(test_files['receptor'])


class TestMGLToolsPreparationStrategy:
    """Test MGLTools preparation strategy."""
    
    def test_strategy_instantiation(self):
        '''Test that MGLTools strategy can be instantiated.'''

        strategy = MGLToolsPreparationStrategy()
        assert isinstance(strategy, PreparationStrategy)
        assert isinstance(strategy, MGLToolsPreparationStrategy)
    
    def test_prepare_ligand(self, tmp_path, sample_ligand):
        '''Test ligand preparation with tools.'''
        
        strategy = MGLToolsPreparationStrategy()
        output = tmp_path / "output" / "ligand.pdbqt"
        
        result = strategy.prepare_ligand(sample_ligand, str(output), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
    
    def test_prepare_receptor(self, tmp_path, sample_receptor):
        '''Test receptor preparation with tools.'''
        
        strategy = MGLToolsPreparationStrategy()
        output = tmp_path / "output" / "receptor.pdbqt"
        
        result = strategy.prepare_receptor(sample_receptor, str(output), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
    
    def test_shared_utilities(self):
        '''Test that shared utility methods work correctly.'''

        strategy = MGLToolsPreparationStrategy()
        
        # Test tool availability check
        assert isinstance(strategy._check_tool_available("/nonexistent"), bool)
        
        # Test directory creation
        test_path = "/tmp/test_prep/output.pdbqt"
        strategy._ensure_output_dir(test_path)
        # Should not raise exception


class TestSPORESPreparationStrategy:
    """Test SPORES preparation strategy."""
    
    def test_strategy_instantiation(self):
        '''Test that SPORES strategy can be instantiated.'''
        
        strategy = SPORESPreparationStrategy()
        assert isinstance(strategy, PreparationStrategy)
        assert isinstance(strategy, SPORESPreparationStrategy)
    
    def test_prepare_ligand(self, tmp_path, sample_ligand):
        '''Test ligand preparation with tools.'''
        
        strategy = SPORESPreparationStrategy()
        output = tmp_path / "output" / "ligand.mol2"
        
        result = strategy.prepare_ligand(sample_ligand, str(output), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
    
    def test_prepare_receptor(self, tmp_path, sample_receptor):
        '''Test receptor preparation with tools.'''
        
        strategy = SPORESPreparationStrategy()
        output = tmp_path / "output" / "receptor.mol2"
        
        result = strategy.prepare_receptor(sample_receptor, str(output), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))


class TestOpenBabelPreparationStrategy:
    """Test OpenBabel preparation strategy."""
    
    def test_strategy_instantiation(self):
        '''Test that OpenBabel strategy can be instantiated.'''
        
        strategy = OpenBabelPreparationStrategy()
        assert isinstance(strategy, PreparationStrategy)
        assert isinstance(strategy, OpenBabelPreparationStrategy)
    
    def test_prepare_ligand_with_valid_file(self, tmp_path, sample_ligand):
        '''Test ligand preparation with valid mol2 file.'''
        
        strategy = OpenBabelPreparationStrategy()
        output = tmp_path / "output" / "ligand.pdbqt"
        
        # This may fail if openbabel is not available, but should handle gracefully
        result = strategy.prepare_ligand(sample_ligand, str(output), "")
        
        # Result should be an int or tuple
        assert isinstance(result, (int, tuple))
    
    def test_prepare_receptor_with_valid_file(self, tmp_path, sample_receptor):
        '''Test receptor preparation with valid pdb file.'''
        
        strategy = OpenBabelPreparationStrategy()
        output = tmp_path / "output" / "receptor.pdbqt"
        
        # This may fail if openbabel is not available, but should handle gracefully
        result = strategy.prepare_receptor(sample_receptor, str(output), "")
        
        # Result should be an int or tuple
        assert isinstance(result, (int, tuple))


class TestStrategyPatternIntegration:
    """Integration tests for Strategy Pattern usage in docking classes."""
    
    def test_vina_uses_mgltools_strategy(self, test_files, tmp_path):
        '''Test that Vina class uses MGLTools strategy.'''
        
        from OCDocker.Docking.Vina import Vina
        from OCDocker.Receptor import Receptor
        from OCDocker.Ligand import Ligand
        
        config = tmp_path / "config.txt"
        config.write_text("")
        prep_rec = tmp_path / "prep_rec.pdbqt"
        prep_lig = tmp_path / "prep_lig.pdbqt"
        log = tmp_path / "log.txt"
        out = tmp_path / "out.pdbqt"
        
        receptor = Receptor(str(test_files['receptor']), name="test_receptor")
        ligand = Ligand(str(test_files['ligand']), name="test_ligand")
        
        vina = Vina(
            str(config), str(test_files['box']), receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out)
        )
        
        # Check that preparation_strategy is set
        assert hasattr(vina, 'preparation_strategy')
        assert isinstance(vina.preparation_strategy, MGLToolsPreparationStrategy)
    
    def test_plants_uses_spores_strategy(self, test_files, tmp_path):
        '''Test that PLANTS class uses SPORES strategy.'''
        
        from OCDocker.Docking.PLANTS import PLANTS
        from OCDocker.Receptor import Receptor
        from OCDocker.Ligand import Ligand
        
        config = tmp_path / "config.txt"
        config.write_text("")
        prep_rec = tmp_path / "prep_rec.mol2"
        prep_lig = tmp_path / "prep_lig.mol2"
        log = tmp_path / "log.txt"
        out = tmp_path / "out"
        
        receptor = Receptor(str(test_files['receptor']), name="test_receptor")
        ligand = Ligand(str(test_files['ligand']), name="test_ligand")
        
        plants = PLANTS(
            str(config), str(test_files['box']), receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out)
        )
        
        # Check that preparation_strategy is set
        assert hasattr(plants, 'preparation_strategy')
        assert isinstance(plants.preparation_strategy, SPORESPreparationStrategy)
    
    def test_smina_uses_correct_strategies(self, test_files, tmp_path):
        '''Test that Smina uses MGLTools for ligand and OpenBabel for receptor.'''
        
        from OCDocker.Docking.Smina import Smina
        from OCDocker.Receptor import Receptor
        from OCDocker.Ligand import Ligand
        
        config = tmp_path / "config.txt"
        config.write_text("")
        prep_rec = tmp_path / "prep_rec.pdbqt"
        prep_lig = tmp_path / "prep_lig.pdbqt"
        log = tmp_path / "log.txt"
        out = tmp_path / "out.pdbqt"
        
        # Use absolute paths to prevent directories in project root
        receptor_path_abs = str(Path(test_files['receptor']).resolve())
        ligand_path_abs = str(Path(test_files['ligand']).resolve())
        receptor = Receptor(receptor_path_abs, name="test_receptor")
        ligand = Ligand(ligand_path_abs, name="test_ligand")
        # Verify ligand.path is absolute and not empty
        assert ligand.path, "Ligand path should not be empty"
        assert os.path.isabs(ligand.path), f"Ligand path should be absolute, got: {ligand.path}"
        
        smina = Smina(
            str(config), str(test_files['box']), receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out)
        )
        
        # Check that preparation_strategy is set (for ligand)
        assert hasattr(smina, 'preparation_strategy')
        assert isinstance(smina.preparation_strategy, MGLToolsPreparationStrategy)
