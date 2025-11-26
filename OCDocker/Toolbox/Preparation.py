#!/usr/bin/env python3

# Description
###############################################################################
'''
Strategy Pattern implementation for molecule preparation.

This module provides an abstract interface and concrete implementations for
preparing ligands and receptors using different tools (MGLTools, SPORES, OpenBabel).

They are imported as::

    from OCDocker.Toolbox.Preparation import (
        PreparationStrategy,
        MGLToolsPreparationStrategy,
        SPORESPreparationStrategy,
        OpenBabelPreparationStrategy
    )
'''

# Imports
###############################################################################
from abc import ABC, abstractmethod
from typing import Union, Tuple
import os
import shutil

from OCDocker.Config import get_config
from OCDocker.Toolbox import Running as ocrun
from OCDocker.Toolbox.Running import is_tool_available
from OCDocker.Toolbox.FilesFolders import ensure_parent_dir
import OCDocker.Error as ocerror

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

class PreparationStrategy(ABC):
    """Abstract base class for molecule preparation strategies."""
    
    @abstractmethod
    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a ligand molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        pass
    
    @abstractmethod
    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a receptor molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        pass
    
    def get_ligand_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a ligand.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
            
        Returns
        -------
        list[str]
            Command list that would be executed
        '''
        
        # Default implementation - should be overridden by subclasses
        return []
    
    def get_receptor_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a receptor.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
            
        Returns
        -------
        list[str]
            Command list that would be executed
        '''
        
        # Default implementation - should be overridden by subclasses
        return []
    
    def _check_tool_available(self, exe: str) -> bool:
        '''Check if tool executable is available (shared utility).
        
        Parameters
        ----------
        exe : str
            Path to the tool executable
            
        Returns
        -------
        bool
            True if the tool executable is available, False otherwise
        '''

        return is_tool_available(exe)
    
    def _ensure_output_dir(self, output_path: str) -> None:
        '''Ensure output directory exists (shared utility).
        
        Parameters
        ----------
        output_path : str
            Path to the output directory
        '''

        ensure_parent_dir(output_path)
    
    def _fallback_copy(
        self,
        input_path: str,
        output_path: str,
        tool_name: str
    ) -> Union[int, Tuple[int, str]]:
        '''Fallback to copying file if tool unavailable (shared utility).
        
        Parameters
        ----------
        input_path : str
            Path to the input file
        output_path : str
            Path to the output file
        tool_name : str
            Name of the tool
        
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        try:
            shutil.copyfile(input_path, output_path)
            return ocerror.Error.ok()  # type: ignore
        except Exception as e:
            return ocerror.Error.subprocess(
                message=f"{tool_name} not available and copy failed: {e}",
                level=ocerror.ReportLevel.ERROR
            )  # type: ignore


class MGLToolsPreparationStrategy(PreparationStrategy):
    """Preparation strategy using MGLTools (prepare_ligand4.py/prepare_receptor4.py)."""
    
    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a ligand molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        config = get_config()
        exe = str(config.tools.pythonsh)
        
        if not self._check_tool_available(exe):
            self._ensure_output_dir(output_path)
            return self._fallback_copy(input_path, output_path, "pythonsh")
        
        # Print verbosity
        from OCDocker.Toolbox import Printing as ocprint
        ocprint.printv(f"Running '{config.tools.prepare_ligand}' for '{input_path}'.")
        
        self._ensure_output_dir(output_path)
        
        # Create command
        cmd = [
            config.tools.pythonsh,
            config.tools.prepare_ligand,
            "-l", input_path,
            "-C", "-o", output_path
        ]
        
        return ocrun.run(cmd, logFile=log_file, cwd=os.path.dirname(input_path))
    
    def get_ligand_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a ligand.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
            
        Returns
        -------
        list[str]
            Command list that would be executed
        '''
        
        config = get_config()
        return [
            config.tools.pythonsh,
            config.tools.prepare_ligand,
            "-l", input_path,
            "-C", "-o", output_path
        ]
    
    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a receptor molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        config = get_config()
        exe = str(config.tools.pythonsh)
        
        if not self._check_tool_available(exe):
            self._ensure_output_dir(output_path)
            return self._fallback_copy(input_path, output_path, "pythonsh")
        
        # Print verbosity
        from OCDocker.Toolbox import Printing as ocprint
        ocprint.printv(f"Running '{config.tools.prepare_receptor}' for '{input_path}'.")
        
        self._ensure_output_dir(output_path)
        
        # Create command
        cmd = [
            config.tools.pythonsh,
            config.tools.prepare_receptor,
            "-r", input_path,
            "-o", output_path,
            "-A", "hydrogens",
            "-U", "nphs_lps_waters"
        ]
        
        return ocrun.run(cmd, logFile=log_file)
    
    def get_receptor_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a receptor.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
            
        Returns
        -------
        list[str]
            Command list that would be executed
        '''
        
        config = get_config()
        return [
            config.tools.pythonsh,
            config.tools.prepare_receptor,
            "-r", input_path,
            "-o", output_path,
            "-A", "hydrogens",
            "-U", "nphs_lps_waters"
        ]
    

class SPORESPreparationStrategy(PreparationStrategy):
    """Preparation strategy using SPORES."""
    
    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a ligand molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        config = get_config()
        exe = str(config.tools.spores)
        
        if not self._check_tool_available(exe):
            self._ensure_output_dir(output_path)
            return self._fallback_copy(input_path, output_path, "SPORES")
        
        self._ensure_output_dir(output_path)
        
        # Create command
        cmd = [
            config.tools.spores,
            "--mode", "complete",
            input_path,
            output_path
        ]
        
        # Print verbosity
        from OCDocker.Toolbox import Printing as ocprint
        ocprint.printv(f"Running '{config.tools.spores}' for '{input_path}'.")
        
        return ocrun.run(cmd, logFile=log_file)
    
    def get_ligand_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a ligand.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
            
        Returns
        -------
        list[str]
            Command list that would be executed
        '''
        
        config = get_config()
        return [
            config.tools.spores,
            "--mode", "complete",
            input_path,
            output_path
        ]
    
    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        # Same as ligand for SPORES
        return self.prepare_ligand(input_path, output_path, log_file)
    
    def get_receptor_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a receptor.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
            
        Returns
        -------
        list[str]
            Command list that would be executed (same as ligand for SPORES)
        '''
        
        return self.get_ligand_command(input_path, output_path)


class OpenBabelPreparationStrategy(PreparationStrategy):
    """Preparation strategy using OpenBabel (for Gnina and similar)."""
    
    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a ligand molecule.

        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
            
        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        # OpenBabel strategy may include extension validation
        from OCDocker.Toolbox import Validation as ocvalidation
        from OCDocker.Initialise import clrs
        
        extension = ocvalidation.validate_obabel_extension(input_path)
        if type(extension) != str:
            from OCDocker.Toolbox import Printing as ocprint
            ocprint.print_error(f"Problems while reading the ligand file '{input_path}'.")
            return extension  # type: ignore
        
        # Discover if the output extension is pdbqt (to warn user if it is not)
        out_extension = os.path.splitext(output_path)[1]
        if out_extension != ".pdbqt":
            from OCDocker.Toolbox import Printing as ocprint
            ocprint.print_warning(
                f"The output extension is not '.pdbqt', is {out_extension}. "
                f"This function converts {clrs['r']}ONLY{clrs['n']} to '.pdbqt'. "
                f"Please pay attention, since this might be a problem in the future for you!"
            )
        
        # Handle SMILES files if needed
        if extension in ["smi", "smiles"]:
            from OCDocker.Toolbox import Printing as ocprint
            ocprint.print_warning(
                f"The input ligand is a smiles file, it is supposed that there will be "
                f"also a mol2 file within the same folder, so I am changing the file "
                f"extension to '.mol2' to be able to read it."
            )
            input_path = f"{os.path.dirname(input_path)}/ligand.mol2"
        
        # Use conversion utility
        from OCDocker.Toolbox import Conversion as occonversion
        return occonversion.convert_mols(input_path, output_path)  # type: ignore
    
    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = ""
    ) -> Union[int, Tuple[int, str]]:
        '''Prepare a receptor molecule.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)

        Returns
        -------
        Union[int, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        # Similar to ligand but for receptor
        from OCDocker.Toolbox import Conversion as occonversion
        return occonversion.convert_mols(input_path, output_path)  # type: ignore
    
    def get_ligand_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a ligand.
        
        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
            
        Returns
        -------
        list[str]
            Command list that would be executed (OpenBabel conversion)
        '''
        
        config = get_config()
        # OpenBabel uses obabel command
        exe = str(config.tools.obabel)
        return [
            exe,
            input_path,
            "-O", output_path
        ]
    
    def get_receptor_command(self, input_path: str, output_path: str) -> list[str]:
        '''Get the command list that would be used to prepare a receptor.
        
        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
            
        Returns
        -------
        list[str]
            Command list that would be executed (OpenBabel conversion)
        '''
        
        # Same as ligand for OpenBabel
        return self.get_ligand_command(input_path, output_path)
