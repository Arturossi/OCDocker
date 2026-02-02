#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to extract and process information
of any kind of molecule.

They are imported as:

import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
'''

# Imports
###############################################################################
import os
from functools import lru_cache

from spyrmsd import io, rmsd
from threading import Lock
from typing import Dict, List, Tuple, Union

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

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
AtomKey = Tuple[str, str]                  # (resname, atomname) in CHARMM scheme
AtomVal = Tuple[str, str]                  # (resname, atomname) in canonical scheme

# Optional: collapse PDB2PQR “protonation/patch” residue names back to common PDB ones
_CANON_COLLAPSE = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    "CYM": "CYS", "CYX": "CYS",
    "TYM": "TYR",
}


def _format_atom_name(atom: str, element: str) -> str:
    '''Format atom name into PDB columns 13-16.
    
    Parameters
    ----------
    atom : str
        The atom name.
    element : str
        The element symbol.

    Returns
    -------
    str
        The formatted atom name.
    '''

    atom = atom.strip()
    element = (element or "").strip().upper()

    if len(atom) >= 4:
        return atom[:4]

    if atom and atom[0].isdigit():
        # e.g. 1HG1: left-justify
        return f"{atom:<4}"

    # If element is 1-letter (C, N, O, S, P, H...), PDB usually uses a leading space
    if len(element) == 1:
        return f" {atom:<3}"

    # Otherwise right-justify
    return f"{atom:>4}"


@lru_cache(maxsize=1)
def build_charmm_to_canonical_map() -> Dict[AtomKey, AtomVal]:
    '''Build a complete CHARMM -> canonical mapping using PDB2PQR definitions and the CHARMM forcefield naming rules.

    Returns
    -------
    Dict[AtomKey, AtomVal]
        The mapping from CHARMM (key) to canonical (value).
    '''

    try:
        from pdb2pqr import io as pqr_io
        from pdb2pqr import forcefield as pqr_ff
    except ImportError as e:
        raise ImportError("pdb2pqr is required to build CHARMM->canonical maps.") from e

    definition = pqr_io.get_definitions()
    ff = pqr_ff.Forcefield("charmm", definition, userff=None, usernames=None)

    rev: Dict[AtomKey, AtomVal] = {}

    for canon_resname, defres in definition.map.items():
        # defres.map: canonical atomname -> definition atom object
        for canon_atomname in defres.map.keys():
            ff_res, ff_atom = ff.get_names(canon_resname, canon_atomname)
            if not ff_res or not ff_atom:
                continue
            # Invert: CHARMM -> canonical
            rev.setdefault((ff_res, ff_atom), (canon_resname, canon_atomname))

    return rev


def needs_canonical_pdb_fix(
    pdb_path: Union[str, os.PathLike],
    *,
    collapse_resnames: bool = True,
) -> bool:
    '''Check if a PDB file contains CHARMM-style names that should be canonicalized.'''

    input_path = os.fspath(pdb_path)
    if not os.path.isfile(input_path):
        return False

    try:
        rev_map = build_charmm_to_canonical_map()
    except ImportError as e:
        ocprint.print_warning(f"pdb2pqr not available; skipping canonical name detection. Error: {e}")
        return False
    except Exception as e:
        ocprint.print_warning(f"Failed to build CHARMM->canonical map. Error: {e}")
        return False

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                    continue

                atom = line[12:16].strip()
                res = line[17:20].strip()
                new_res, new_atom = rev_map.get((res, atom), (res, atom))

                if collapse_resnames:
                    new_res = _CANON_COLLAPSE.get(new_res, new_res)

                if new_res != res or new_atom != atom:
                    return True
    except Exception as e:
        ocprint.print_warning(f"Failed while scanning '{input_path}' for CHARMM names. Error: {e}")
        return False

    return False

## Public ##
def split_poses(ligand: str, ligandName: str, outPath: str, suffix: str = "", logFile: str = "") -> Union[int, Tuple[int, str]]:
    '''Split the input ligand into its poses.

    Parameters
    ----------
    ligand : str
        The path to the input ligand.
    ligandName : str
        The name of the input ligand.
    outPath : str
        The path to the output folder.
    suffix : str, optional
        The suffix to be added to the output files, by default "".
    logFile : str, optional
        The path to the log file, by default "".
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Split the input ligand
    config = get_config()
    
    # Ensure ligand input path is absolute and normalized (remove duplicate directory components)
    ligand = ocff.normalize_path(ligand)
    # Ensure outPath is normalized (removes duplicate slashes, . and .. components, and duplicate directories) and absolute
    outPath = ocff.normalize_path(outPath)
    os.makedirs(outPath, exist_ok=True)
    # Construct the output file prefix (vina_split will append numbers like _0, _1, etc.)
    # Use normalize again to ensure no duplicate path components
    output_prefix = ocff.normalize_path(os.path.join(outPath, f"{ligandName}{suffix}"))
    cmd = [config.vina.split_executable, "--input", ligand, "--flex", "''", "--ligand", output_prefix]

    # Print verbosity
    ocprint.printv(f"Spliting the ligand '{ligand}'.")

    # Run the command
    return ocrun.run(cmd, logFile = logFile)


def get_rmsd_matrix(molecules: List[str]) -> Dict[str, Dict[str, float]]:
    '''Get the rmsd matrix between a list of molecules.

    Parameters
    ----------
    molecules : List[str]
        The list of molecules.

    Returns
    -------
    Dict[str, Dict[str, float]]
        The rmsd matrix.
    '''

    # Initialise the rmsd matrix
    rmsdMatrix = {}

    # For each molecule in molecules
    for molecule in molecules:
        # Initialise the row of the rmsd matrix
        rmsdRow = {}

        # For each molecule in molecules
        for otherMolecule in molecules:
            # If the molecule is the same as the otherMolecule
            if molecule == otherMolecule:
                # Append 0 to the row
                rmsdRow[otherMolecule] = 0
            else:
                # Get the rmsd between the molecule and the other molecule
                tmpMolecule = get_rmsd(molecule, otherMolecule)
                # Get the rmsd between the molecule and the other molecule
                rmsdRow[otherMolecule] = tmpMolecule if isinstance(tmpMolecule, float) else tmpMolecule[0] # type: ignore

        # Append the row to the rmsd matrix
        rmsdMatrix[molecule] = rmsdRow 

    # Return the rmsd matrix
    return rmsdMatrix


def get_rmsd(reference: str, molecule: str) -> Union[List[float], float]:
    '''Get the rmsd between a reference and a molecule file (it supports more than one molecule in this second file).

    Parameters
    ----------
    reference : str
        The reference file.
    molecule : str
        The molecule file.

    Returns
    -------
    List[float] | float
        The rmsd between the reference and the molecule file.
    '''

    # Load reference
    ref = io.loadmol(reference)

    # Remove its hydrogens
    ref.strip()

    # Load all molecules (if only one, a list with a single element will be generated)
    mols = io.loadallmols(molecule)
    
    # For each molecule in molecules
    for mol in mols:
        # Remove its hydrogens
        mol.strip() # type: ignore

    # Get the reference and molecules coordinates
    refCoordinates = ref.coordinates
    molCoordinates = [mol.coordinates for mol in mols] # type: ignore

    # Get the reference and molecules atomicnums
    refAtmNum = ref.atomicnums
    molAtmNum = mols[0].atomicnums # type: ignore

    # Get the reference and molecules adjacency_matrix
    refAdjMat = ref.adjacency_matrix
    molAdjMat = mols[0].adjacency_matrix # type: ignore

    # Return the symmetric rmsd (account for symmetry because it is important)
    return rmsd.symmrmsd(refCoordinates, molCoordinates, refAtmNum, molAtmNum, refAdjMat, molAdjMat)


def clean_for_dssp(structurePath: str) -> int:
    '''Make a pdb file with HEADER (empty), then CRYST1 and finally ATOM records only.

    Parameters
    ----------
    structurePath : str
        The path to the structure file.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Initialise hasCryst1 flag
    hasCryst1 = False

    # Inigialise hasHeader flag
    hasHeader = False
    
    # List of lines and dssp lines
    lines = []

    # Check if structurePath is a valid file
    if os.path.isfile(structurePath):
        # Open it (for cleaning)
        with open(structurePath, 'r') as pdbFile:
            # For each line in pdbFile
            for line in pdbFile:
                if not line.startswith("HEADER") and not hasHeader:
                    # Set the hasHeader flag to True
                    hasHeader = True
                    
                    # Add the line to the list
                    lines.append("HEADER    \n")

                if not line.startswith("CRYST1") and hasHeader and not hasCryst1:
                    # Set the hasCryst1 flag to True
                    hasCryst1 = True
                    
                    # Add the line to the list
                    lines.append("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")

                # Check if the line starts with ATOM
                if line.startswith("ATOM"):
                    # Check if there is a chain in the line (all the lines should have a chain)
                    if line[21] == " ":
                        # Assume that the protein has only one chain and call it A
                        line = f"{line[:21]}A{line[22:]}"
                    
                    # Add the line to the list
                    lines.append(line)

        # Create a lock for multithreading
        lock = Lock()
        
        # Start the lock with statement
        with lock:
            # Write the lines to the file
            with open(structurePath, 'w') as pdbFile:
                # Write the lines list to the file
                pdbFile.writelines(lines)
        
        return ocerror.Error.ok()
    else:
        return ocerror.Error.file_not_exist(message = f"The file '{structurePath}' does not exist!", level = ocerror.ReportLevel.ERROR)


def clean_pdb_file(
    structurePath: str,
    outputPath: str,
    overwrite: bool = False,
    keep_hetatm: bool = False,
) -> int:
    '''Create a cleaned PDB file with HEADER, CRYST1, and ATOM (optionally HETATM) records.

    Parameters
    ----------
    structurePath : str
        The path to the input structure file.
    outputPath : str
        The path to the cleaned output file.
    overwrite : bool, optional
        If True, overwrites the output file when it already exists.
    keep_hetatm : bool, optional
        If True, keep HETATM lines in the output.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    if not os.path.isfile(structurePath):
        return ocerror.Error.file_not_exist(message=f"The file '{structurePath}' does not exist!", level=ocerror.ReportLevel.ERROR)

    if os.path.isfile(outputPath) and not overwrite:
        return ocerror.Error.file_exists(message=f"The file '{outputPath}' already exists, aborting cleaning.", level=ocerror.ReportLevel.WARNING) # type: ignore

    # Initialise flags
    hasCryst1 = False
    hasHeader = False

    lines: List[str] = []

    try:
        with open(structurePath, 'r') as pdbFile:
            for line in pdbFile:
                if not line.startswith("HEADER") and not hasHeader:
                    hasHeader = True
                    lines.append("HEADER    \n")

                if not line.startswith("CRYST1") and hasHeader and not hasCryst1:
                    hasCryst1 = True
                    lines.append("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")

                if line.startswith("ATOM") or (keep_hetatm and line.startswith("HETATM")):
                    if line[21] == " ":
                        line = f"{line[:21]}A{line[22:]}"
                    lines.append(line)

        # Write the cleaned output
        lock = Lock()
        with lock:
            with open(outputPath, 'w') as pdbFile:
                pdbFile.writelines(lines)
    except Exception as e:
        return ocerror.Error.write_file(message=f"Could not clean PDB file '{structurePath}'. Error: {e}", level=ocerror.ReportLevel.ERROR) # type: ignore

    return ocerror.Error.ok()


def convert_pdb_charmm_to_canonical(
    pdb_in: Union[str, os.PathLike],
    pdb_out: Union[str, os.PathLike],
    *,
    collapse_resnames: bool = True,
    overwrite: bool = False,
    in_place: bool = False,
) -> int:
    """Convert a CHARMM-named PDB to canonical PDB names using PDB2PQR mappings.

    Parameters
    ----------
    pdb_in : str | os.PathLike
        Input PDB file.
    pdb_out : str | os.PathLike
        Output PDB file (ignored if in_place=True).
    collapse_resnames : bool, optional
        Collapse patched/protonated residue names to common PDB names.
    overwrite : bool, optional
        Whether to overwrite output file when in_place=False.
    in_place : bool, optional
        If True, overwrite the input file path atomically.
    """

    input_path = os.fspath(pdb_in)
    output_path = os.fspath(pdb_out)
    if in_place:
        output_path = input_path

    if not os.path.isfile(input_path):
        return ocerror.Error.file_not_exist(
            message=f"The file '{input_path}' does not exist!",
            level=ocerror.ReportLevel.ERROR,
        )

    if os.path.isfile(output_path) and not overwrite and not in_place:
        return ocerror.Error.file_exists(
            message=f"The file '{output_path}' already exists, aborting conversion.",
            level=ocerror.ReportLevel.WARNING,
        )

    try:
        rev_map = build_charmm_to_canonical_map()
    except ImportError as e:
        return ocerror.Error.value_error(
            message=f"pdb2pqr is required for CHARMM->canonical PDB renaming. Error: {e}",
            level=ocerror.ReportLevel.ERROR,
        )
    except Exception as e:
        return ocerror.Error.unknown(
            message=f"Failed to build CHARMM->canonical map. Error: {e}",
            level=ocerror.ReportLevel.ERROR,
        )

    out_lines: List[str] = []
    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                    out_lines.append(line)
                    continue

                atom = line[12:16].strip()
                res = line[17:20].strip()
                element = line[76:78].strip() if len(line) >= 78 else ""

                new_res, new_atom = rev_map.get((res, atom), (res, atom))

                if collapse_resnames:
                    new_res = _CANON_COLLAPSE.get(new_res, new_res)

                line_list = list(line.rstrip("\n"))
                if len(line_list) < 80:
                    line_list.extend([" "] * (80 - len(line_list)))

                atom_field = _format_atom_name(new_atom, element)
                res_field = f"{new_res:>3}"[:3]

                line_list[12:16] = list(atom_field)
                line_list[17:20] = list(res_field)

                out_lines.append("".join(line_list) + "\n")

        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if in_place:
            tmp_path = f"{output_path}.canonical_tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.writelines(out_lines)
            os.replace(tmp_path, output_path)
        else:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.writelines(out_lines)
    except Exception as e:
        return ocerror.Error.write_file(
            message=f"Failed to convert '{input_path}' to canonical PDB names. Error: {e}",
            level=ocerror.ReportLevel.ERROR,
        )

    ocprint.print_success(f"Converted '{input_path}' to canonical PDB '{output_path}'.")
    return ocerror.Error.ok()
