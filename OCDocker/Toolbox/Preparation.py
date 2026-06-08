#!/usr/bin/env python3

# Description
###############################################################################
'''
Strategy Pattern implementation for molecule preparation.

This module provides an abstract interface and concrete implementations for
preparing ligands and receptors using different tools (MGLTools, SPORES, OpenBabel).

Usage:

from OCDocker.Toolbox.Preparation import (
    PreparationStrategy,
    MGLToolsPreparationStrategy,
    SPORESPreparationStrategy,
    OpenBabelPreparationStrategy
)
'''

# Imports
###############################################################################
import os
import shutil

from abc import ABC, abstractmethod
from typing import Tuple, Union

import OCDocker.Error as ocerror

from OCDocker.Config import get_config
from OCDocker.Toolbox.FilesFolders import ensure_parent_dir
from OCDocker.Toolbox.Printing import print_warning
from OCDocker.Toolbox.Running import is_tool_available
from OCDocker.Toolbox import Running as ocrun

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

class PreparationStrategy(ABC):
    """Abstract base for ligand and receptor preparation backends.

    Concrete strategies build external-tool command lines and run preparation
    with shared availability checks, output-directory creation, and overwrite
    handling.
    """


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
    ) -> Union[int, str, Tuple[int, str]]:
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
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        try:
            shutil.copyfile(input_path, output_path)
            return ocerror.Error.ok()
        except Exception as e:
            return ocerror.Error.subprocess(
                message=f"{tool_name} not available and copy failed: {e}",
                level=ocerror.ReportLevel.ERROR
            )

    def _handle_existing_output(
        self,
        output_path: str,
        overwrite: bool,
        entity_label: str
    ) -> Union[int, None]:
        '''Handle existing output files (shared utility).

        Parameters
        ----------
        output_path : str
            Path to the output file
        overwrite : bool
            Whether to overwrite existing output file
        entity_label : str
            Label for the entity being prepared (e.g., "ligand", "receptor")

        Returns
        -------
        int | None
            Error code if skipping, otherwise None to continue
        '''

        if os.path.exists(output_path):
            if overwrite:
                try:
                    os.remove(output_path)
                except (OSError, PermissionError):
                    pass
            else:
                print_warning(
                    f"Prepared {entity_label} '{output_path}' already exists and overwrite is False. "
                    f"Skipping preparation."
                )
                return ocerror.Error.ok()
        return None

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

    @abstractmethod
    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a ligand molecule.

        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        pass

    @abstractmethod
    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a receptor molecule.

        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        pass


class MGLToolsPreparationStrategy(PreparationStrategy):
    """Prepare ligands and receptors with MGLTools scripts.

    Uses ``prepare_ligand4.py`` and ``prepare_receptor4.py`` via the configured
    ``pythonsh`` interpreter from :class:`OCDocker.Config.ToolsConfig`.
    """


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

    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a ligand molecule.

        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        result = self._handle_existing_output(output_path, overwrite, "ligand")
        if result is not None:
            return result

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

    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a receptor molecule.

        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        result = self._handle_existing_output(output_path, overwrite, "receptor")
        if result is not None:
            return result

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

        return ocrun.run(cmd, logFile=log_file, cwd=os.path.dirname(input_path))


class SPORESPreparationStrategy(PreparationStrategy):
    """Prepare receptors with the SPORES external tool.

    Converts receptor structures to MOL2 and applies SPORES protonation and
    typing using the configured ``spores`` executable.
    """


    def _prepare(
        self,
        input_path: str,
        output_path: str,
        log_file: str,
        overwrite: bool,
        entity_label: str
    ) -> Union[int, str, Tuple[int, str]]:
        result = self._handle_existing_output(output_path, overwrite, entity_label)
        if result is not None:
            return result

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

    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a ligand molecule.

        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        return self._prepare(
            input_path,
            output_path,
            log_file,
            overwrite,
            "ligand"
        )

    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a receptor molecule.

        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)
        '''

        return self._prepare(
            input_path,
            output_path,
            log_file,
            overwrite,
            "receptor"
        )


class OpenBabelPreparationStrategy(PreparationStrategy):
    """Prepare ligands and receptors with Open Babel.

    Uses the configured ``obabel`` executable for format conversion and
    optional protonation when MGLTools is unavailable (e.g. Gnina workflows).
    """


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

    def prepare_ligand(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a ligand molecule.

        Parameters
        ----------
        input_path : str
            Path to input ligand file
        output_path : str
            Path to output prepared ligand file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        result = self._handle_existing_output(output_path, overwrite, "ligand")
        if result is not None:
            return result

        # OpenBabel strategy may include extension validation
        from OCDocker.Toolbox import Validation as ocvalidation

        extension = ocvalidation.validate_obabel_extension(input_path)
        if not isinstance(extension, str):
            from OCDocker.Toolbox import Printing as ocprint
            ocprint.print_error(f"Problems while reading the ligand file '{input_path}'.")
            return extension

        # Discover if the output extension is pdbqt (to warn user if it is not)
        out_extension = os.path.splitext(output_path)[1]
        if out_extension != ".pdbqt":
            from OCDocker.Toolbox import Printing as ocprint
            ocprint.print_warning(
                f"The output extension is not '.pdbqt', is {out_extension}. "
                f"If you expected a .pdbqt file, please double-check your output path, "
                f"since downstream tools may assume that format."
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
            if not os.path.isfile(input_path):
                ocprint.print_error(
                    f"Expected companion mol2 file for smiles input was not found: '{input_path}'."
                )
                return ocerror.Error.file_not_exist(
                    message=f"Companion mol2 file missing: '{input_path}'.",
                    level=ocerror.ReportLevel.ERROR
                )

        # Use conversion utility
        from OCDocker.Toolbox import Conversion as occonversion
        return occonversion.convert_mols(input_path, output_path, return_molecule=False, overwrite=overwrite)


# Functions
###############################################################################
## Private ##

## Public ##

    def prepare_receptor(
        self,
        input_path: str,
        output_path: str,
        log_file: str = "",
        overwrite: bool = False
    ) -> Union[int, str, Tuple[int, str]]:
        '''Prepare a receptor molecule.

        Parameters
        ----------
        input_path : str
            Path to input receptor file
        output_path : str
            Path to output prepared receptor file
        log_file : str, optional
            Path to log file (empty to suppress)
        overwrite : bool, optional
            Whether to overwrite existing output file (default is False)

        Returns
        -------
        Union[int, str, Tuple[int, str]]
            Error code or tuple of (error_code, stderr)
        '''

        result = self._handle_existing_output(output_path, overwrite, "receptor")
        if result is not None:
            return result

        # Similar to ligand but for receptor
        from OCDocker.Toolbox import Conversion as occonversion
        return occonversion.convert_mols(input_path, output_path, return_molecule=False, overwrite=overwrite)
