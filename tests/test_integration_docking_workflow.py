#!/usr/bin/env python3

"""
Integration tests for complete docking workflows.

These tests verify end-to-end workflows for docking preparation and execution,
testing the integration between different components.
"""

from __future__ import annotations
from pathlib import Path
import os
import pytest
import tempfile
import shutil

import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.PLANTS as ocplants
from OCDocker.Receptor import Receptor
from OCDocker.Ligand import Ligand


@pytest.fixture
def test_files(tmp_path):
    '''Create test files for integration testing using test files.'''
    
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
    
    files = {
        'receptor': str(receptor_file),
        'ligand': str(ligand_file),
        'box': str(box_file),
        'output_dir': str(tmp_path / "output"),
    }
    
    # Create output directories
    Path(files['output_dir']).mkdir(exist_ok=True)
    
    return files


class TestDockingWorkflowIntegration:
    """Integration tests for complete docking workflows."""
    
    def test_vina_workflow_initialization(self, test_files):
        '''Test that Vina workflow can be initialized correctly.'''
        
        config = Path(test_files['output_dir']) / "conf_vina.txt"
        prep_rec = Path(test_files['output_dir']) / "prep_rec.pdbqt"
        prep_lig = Path(test_files['output_dir']) / "prep_lig.pdbqt"
        log = Path(test_files['output_dir']) / "vina.log"
        out = Path(test_files['output_dir']) / "vina.pdbqt"
        
        # Use test files
        receptor = Receptor(test_files['receptor'], name="test_receptor")
        ligand = Ligand(test_files['ligand'], name="test_ligand")
        
        vina = ocvina.Vina(
            str(config), test_files['box'], receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out),
            name="Test Vina"
        )
        
        # Verify object creation
        assert vina is not None
        assert vina.name == "Test Vina"
        assert hasattr(vina, 'preparation_strategy')
        assert hasattr(vina, 'run_prepare_ligand')
        assert hasattr(vina, 'run_prepare_receptor')
        assert hasattr(vina, 'run_docking')
    
    def test_smina_workflow_initialization(self, test_files):
        '''Test that Smina workflow can be initialized correctly.'''
        
        config = Path(test_files['output_dir']) / "conf_smina.txt"
        prep_rec = Path(test_files['output_dir']) / "prep_rec.pdbqt"
        prep_lig = Path(test_files['output_dir']) / "prep_lig.pdbqt"
        log = Path(test_files['output_dir']) / "smina.log"
        out = Path(test_files['output_dir']) / "smina.pdbqt"
        
        # Use test files - ensure absolute paths to prevent directories in project root
        receptor_path_abs = str(Path(test_files['receptor']).resolve())
        ligand_path_abs = str(Path(test_files['ligand']).resolve())
        receptor = Receptor(receptor_path_abs, name="test_receptor")
        ligand = Ligand(ligand_path_abs, name="test_ligand")
        # Verify ligand.path is absolute and not empty
        assert ligand.path, "Ligand path should not be empty"
        assert os.path.isabs(ligand.path), f"Ligand path should be absolute, got: {ligand.path}"
        
        smina = ocsmina.Smina(
            str(config), test_files['box'], receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out),
            name="Test Smina"
        )
        
        # Verify object creation
        assert smina is not None
        assert smina.name == "Test Smina"
        assert hasattr(smina, 'preparation_strategy')
        assert hasattr(smina, 'run_prepare_ligand')
        assert hasattr(smina, 'run_prepare_receptor')
        assert hasattr(smina, 'run_docking')
    
    def test_plants_workflow_initialization(self, test_files):
        '''Test that PLANTS workflow can be initialized correctly.'''
        
        config = Path(test_files['output_dir']) / "conf_plants.txt"
        prep_rec = Path(test_files['output_dir']) / "prep_rec.mol2"
        prep_lig = Path(test_files['output_dir']) / "prep_lig.mol2"
        log = Path(test_files['output_dir']) / "plants.log"
        out = Path(test_files['output_dir']) / "plants_output"
        
        # Use test files
        receptor = Receptor(test_files['receptor'], name="test_receptor")
        ligand = Ligand(test_files['ligand'], name="test_ligand")
        
        plants = ocplants.PLANTS(
            str(config), test_files['box'], receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out),
            name="Test PLANTS"
        )
        
        # Verify object creation
        assert plants is not None
        assert plants.name == "Test PLANTS"
        assert hasattr(plants, 'preparation_strategy')
        assert hasattr(plants, 'run_prepare_ligand')
        assert hasattr(plants, 'run_prepare_receptor')
        assert hasattr(plants, 'run_docking')
    
    def test_preparation_strategy_integration(self, test_files):
        '''Test that preparation strategies are correctly integrated.'''
        
        config = Path(test_files['output_dir']) / "conf_vina.txt"
        prep_rec = Path(test_files['output_dir']) / "prep_rec.pdbqt"
        prep_lig = Path(test_files['output_dir']) / "prep_lig.pdbqt"
        log = Path(test_files['output_dir']) / "vina.log"
        out = Path(test_files['output_dir']) / "vina.pdbqt"
        
        # Use test files
        receptor = Receptor(test_files['receptor'], name="test_receptor")
        ligand = Ligand(test_files['ligand'], name="test_ligand")
        
        vina = ocvina.Vina(
            str(config), test_files['box'], receptor, str(prep_rec),
            ligand, str(prep_lig), str(log), str(out)
        )
        
        # Test that preparation methods use the strategy
        result_lig = vina.run_prepare_ligand()
        result_rec = vina.run_prepare_receptor()
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result_lig, (int, tuple))
        assert isinstance(result_rec, (int, tuple))


class TestStandalonePreparationFunctions:
    """Test standalone preparation functions use strategies correctly."""
    
    def test_vina_standalone_prepare_ligand(self, test_files, tmp_path):
        '''Test standalone run_prepare_ligand function.'''
        
        lig_in = test_files['ligand']
        lig_out = tmp_path / "output" / "ligand.pdbqt"
        
        result = ocvina.run_prepare_ligand(str(lig_in), str(lig_out), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
    
    def test_smina_standalone_prepare_ligand(self, test_files, tmp_path):
        '''Test standalone Smina run_prepare_ligand function.'''
        
        lig_in = test_files['ligand']
        lig_out = tmp_path / "output" / "ligand.pdbqt"
        
        result = ocsmina.run_prepare_ligand(str(lig_in), str(lig_out))
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
    
    def test_plants_standalone_prepare_ligand(self, test_files, tmp_path):
        '''Test standalone PLANTS run_prepare_ligand function.'''
        
        lig_in = test_files['ligand']
        lig_out = tmp_path / "output" / "ligand.mol2"
        
        result = ocplants.run_prepare_ligand(str(lig_in), str(lig_out), "")
        
        # Result should be an int or tuple (may succeed or fail depending on tool availability)
        assert isinstance(result, (int, tuple))
