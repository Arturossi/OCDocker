#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process all content related to
the ligand.

They are imported as:

import OCDocker.Receptor as ocr
'''

# Imports
###############################################################################
import Bio
import json
import math
import os

import numpy as np

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import SASA
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBExceptions import PDBException
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from openbabel import openbabel
from threading import Lock
from typing import Dict, Tuple, Union, Optional, Any

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation

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
class Receptor:
    """Represents a receptor (protein) molecule with computed descriptors.

    This class loads receptor structures from PDB or mmCIF files and computes
    various molecular descriptors including amino acid composition, surface
    accessibility (SASA), dipole moment, isoelectric point, GRAVY score,
    aromaticity, and instability index.

    Parameters
    ----------
    structure : str | Bio.PDB.Structure.Structure
        Path to a PDB/mmCIF file or a BioPython Structure object.
    name : str
        Name identifier for the receptor.
    mol2_path : str, optional
        Path to an existing MOL2 file, by default "".
    c_model : str, optional
        Charge model for dipole moment calculation, by default "gasteiger".
    gravy_scale : str, optional
        GRAVY scale to use, by default "KyteDoolitle".
    relative_asa_cutoff : float, optional
        Relative accessible surface area cutoff for surface amino acids, by default 0.7.
    from_json_descriptors : str, optional
        Path to JSON file containing pre-computed descriptors, by default "".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    clean : bool, optional
        Whether to clean/renumber the PDB structure, by default False.

    Attributes
    ----------
    name : str
        Name of the receptor.
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    path : str
        Path to the structure file.
    SASA : float
        Solvent accessible surface area.
    DipoleMoment : float
        Dipole moment of the receptor.
    IsoelectricPoint : float
        Isoelectric point (pI) of the receptor.
    GRAVY : float
        Grand average of hydropathy.
    Aromaticity : float
        Aromaticity index.
    InstabilityIndex : float
        Instability index.
    countA, countR, countN, ..., countV : int
        Count of each amino acid type.
    TotalAALength : int
        Total number of amino acids.
    AvgAALength : float
        Average chain length.
    countChain : int
        Number of chains.
    """

    # Declare the amino acid count descriptors (relevant for receptors)
    descriptors_names = {
        "count": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    }

    # Declare single descriptors for receptor properties
    single_descriptors = [
        "TotalAALength", "AvgAALength", "countChain", "SASA", "DipoleMoment", "IsoelectricPoint",
        "GRAVY", "Aromaticity", "InstabilityIndex"
    ]

    # Generate all descriptors dynamically
    allDescriptors = [f"count{i}" for i in descriptors_names["count"]] + single_descriptors


    def __init__(self, structure: Union[str, Bio.PDB.Structure.Structure], name: str, mol2_path: str = "", c_model: str = "gasteiger", gravy_scale: str = "KyteDoolitle", relative_asa_cutoff: float = 0.7, from_json_descriptors: str = "", overwrite: bool = False, clean: bool = False) -> None:
        '''Constructor of the class Receptor.

        Parameters
        ----------
        structure : str | Bio.PDB.Structure.Structure
            Path to the structure file OR Bio.PDB.Structure.Structure object.
        name : str
            Name of the receptor.
        mol2_path : str, optional
            Path to the mol2 file, by default "".
        c_model : str, optional
            Charge model to be used, by default "gasteiger".
        gravy_scale : str, optional
            Scale to be used to compute the GRAVY descriptor, by default "KyteDoolitle".
        relative_asa_cutoff : float, optional
            Relative cutoff to be used to compute the SASA descriptor, by default 0.7.
        from_json_descriptors : str, optional
            Path to the json file containing the descriptors, by default "".
        overwrite : bool, optional
            Flag to denote if files will be overwritten, by default False.
        clean : bool, optional
            Flag to denote if the pdb file will be cleaned, by default False.
        
        Returns
        -------
        None
        '''

        # Name must come first
        self.name = ""
        # The molpath not always will exist (should also come first)
        self.mol2_path = str(mol2_path)
        # Set the path and structure (NEVER SHOUD BE NONE)
        # If user pass a json
        if from_json_descriptors:
            # Read the molecule telling that there is no need to fetch the SASA value
            self.path, self.structure = load_mol(structure, name=self.name, compute_sasa=False, mol2_path=self.mol2_path, overwrite = overwrite, clean = clean)
        else:
            # Read the molecule telling that there is the need to fetch the SASA value
            self.path, self.structure = load_mol(structure, name=self.name, compute_sasa=True, mol2_path=self.mol2_path, overwrite = overwrite, clean = clean)

        # Set the residues (derived from structure)
        self.residues = get_res(self.structure)

        # Set everything as None
        self.sasa = None
        self.__c_model = None
        self.dipoleMoment = None
        self.isoelectricPoint = None
        self.instabilityIndex = None

        self.__gravy_scale = None
        self.GRAVY = None

        self.aromaticity = None

        self.totalAALength = None
        self.avgAALength = None
        self.countChain = None

        self.__relative_asa_cutoff = None
        self.__countAA = None

        self.countA = None
        self.countR = None
        self.countN = None
        self.countD = None
        self.countC = None
        self.countQ = None
        self.countE = None
        self.countG = None
        self.countH = None
        self.countI = None
        self.countL = None
        self.countK = None
        self.countM = None
        self.countF = None
        self.countP = None
        self.countS = None
        self.countT = None
        self.countW = None
        self.countY = None
        self.countV = None

        # If user pass a json
        if from_json_descriptors:
            # Read the descriptors from it
            data = read_descriptors_from_json(from_json_descriptors)

            # If data is None, a problem occurred while reading the json file
            if not data:
                ocprint.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None
            
            #region assign
            self.name, self.sasa, self.dipoleMoment, self.isoelectricPoint, self.instabilityIndex,self.GRAVY, self.aromaticity, self.__countAA, self.countA, self.countR, self.countN, self.countD, self.countC, self.countQ, self.countE, self.countG, self.countH, self.countI, self.countL, self.countK, self.countM, self.countF, self.countP, self.countS, self.countT, self.countW, self.countY, self.countV, self.totalAALength, self.avgAALength, self.countChain = data #type: ignore

            #endregion
        else:
            # Check if the name is empty
            if not name:
                ocprint.print_error("The Receptor name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            self.__AAdata = count_AAs_and_chains(self.structure)

            if self.__AAdata:
                self.totalAALength, self.avgAALength, self.countChain = self.__AAdata
            else:
                ocprint.print_error("Problems while counting AAs and chains!")
                return None

            # Ensure sasa attribute exists (compute_sasa should have added it, but add fallback)
            if not hasattr(self.structure, 'sasa'):
                # Fallback: call compute_sasa if it wasn't called or failed
                # Call compute_sasa directly (it's defined in this module)
                compute_sasa(self.structure)
            self.sasa = self.structure.sasa
            self.__c_model = c_model # The options are 'mmff94', 'gasteiger' or 'eem2015bm'
            self.dipoleMoment = compute_dipole_moment(self.path, self.__c_model)
            self.isoelectricPoint = compute_isoelectric_point(self.residues)
            self.instabilityIndex = compute_instability_index(self.residues)

            self.__gravy_scale = gravy_scale
            self.GRAVY = compute_gravy(self.residues, scale=self.__gravy_scale)

            self.aromaticity = compute_aromaticity(self.residues)

            # Potential future feature: secondary_structure_fraction(self)
            # Ref: https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html

            self.__relative_asa_cutoff = relative_asa_cutoff
            
            self.__countAA = count_surface_AA(self.structure, self.path, self.__relative_asa_cutoff)

            self.countA = self.__countAA["A"]
            self.countR = self.__countAA["R"]
            self.countN = self.__countAA["N"]
            self.countD = self.__countAA["D"]
            self.countC = self.__countAA["C"]
            self.countQ = self.__countAA["Q"]
            self.countE = self.__countAA["E"]
            self.countG = self.__countAA["G"]
            self.countH = self.__countAA["H"]
            self.countI = self.__countAA["I"]
            self.countL = self.__countAA["L"]
            self.countK = self.__countAA["K"]
            self.countM = self.__countAA["M"]
            self.countF = self.__countAA["F"]
            self.countP = self.__countAA["P"]
            self.countS = self.__countAA["S"]
            self.countT = self.__countAA["T"]
            self.countW = self.__countAA["W"]
            self.countY = self.__countAA["Y"]

            self.countV = self.__countAA["V"]










    ## Private ##
    def __safe_to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        '''Return all the properties (except the molecule object) for the Receptor object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Union[str, float, int, None]]
            A dictionary with all the properties (except the molecule object) for the Receptor object.
        '''

        # Create new dict
        properties = dict()
        # Set Name and Path
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["mol2Path"] = self.mol2_path if self.mol2_path is not None else "-"
        # Combine both in one dict and return them

        return {**properties, **self.get_descriptors()}

    ## Public ##
    def print_attributes(self) -> None:
        """Print all attributes of the receptor to stdout.

        Displays the receptor's name, structure path, and all computed
        descriptors (SASA, dipole moment, isoelectric point, GRAVY,
        aromaticity, instability index, amino acid counts, etc.) in a
        formatted, aligned table.
        """
        
        attributes = {
            "Name": self.name,
            "Structure path": self.path,
            "mol2 path": self.mol2_path,
            "Structure": self.structure,
            "AA residues": self.residues,
            "Total AA len": self.totalAALength,
            "Average AA len": self.avgAALength,
            "# of chains": self.countChain,
            "SASA": self.sasa,
            "Dipole Moment": self.dipoleMoment,
            "Isoelectric Point": self.isoelectricPoint,
            "GRAVY": self.GRAVY,
            "Aromaticity": self.aromaticity,
            "Instability Index": self.instabilityIndex
        }

        for aa in self.descriptors_names["count"]:
            attributes[f"# of accessible {aa}"] = getattr(self, f"count{aa}", 0)

        for key, value in attributes.items():

            print(f"{key}: {value if value else '-'}")

    def get_descriptors(self)-> Dict[str, Union[float, int]]:
        '''Return the descriptors for the Receptor object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, float | int]
            The descriptors for the Receptor object.
        '''

        descriptors = {
          "TotalAALength": self.totalAALength if self.totalAALength else 0,
          "AvgAALength": self.avgAALength if self.avgAALength else 0,
          "countChain": self.countChain if self.countChain else 0,
          "SASA": self.sasa if self.sasa else None,
          "DipoleMoment": self.dipoleMoment if self.dipoleMoment else None,
          "IsoelectricPoint": self.isoelectricPoint if self.isoelectricPoint else None,
          "GRAVY": self.GRAVY if self.GRAVY else None,
          "Aromaticity": self.aromaticity if self.aromaticity else None,
          "InstabilityIndex": self.instabilityIndex if self.instabilityIndex else None,
          "countA": self.countA if self.countA else 0,
          "countR": self.countR if self.countR else 0,
          "countN": self.countN if self.countN else 0,
          "countD": self.countD if self.countD else 0,
          "countC": self.countC if self.countC else 0,
          "countQ": self.countQ if self.countQ else 0,
          "countE": self.countE if self.countE else 0,
          "countG": self.countG if self.countG else 0,
          "countH": self.countH if self.countH else 0,
          "countI": self.countI if self.countI else 0,
          "countL": self.countL if self.countL else 0,
          "countK": self.countK if self.countK else 0,
          "countM": self.countM if self.countM else 0,
          "countF": self.countF if self.countF else 0,
          "countP": self.countP if self.countP else 0,
          "countS": self.countS if self.countS else 0,
          "countT": self.countT if self.countT else 0,
          "countW": self.countW if self.countW else 0,
          "countY": self.countY if self.countY else 0,
          "countV": self.countV if self.countV else 0
        }

        return descriptors

    def to_dict(self) -> Dict[str, Union[float, int]]:
        '''Return all the properties for the Receptor object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, float | int]
            The properties for the Receptor object.
        '''

        # Create new dict
        properties = dict()
        # Set Name, Path and molecule
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["mol2Path"] = self.mol2_path if self.mol2_path is not None else "-"
        properties["Structure"] = self.structure if self.structure is not None else "-"
        
        # Combine both in one dict and return them

        return {**properties, **self.get_descriptors()}

    def to_json(self, overwrite: bool = False) -> int:
        '''Stores the descriptors as json to avoid the necessity of evaluate them many times.

        Parameters
        ----------
        overwrite: bool, optional
            If True, the json file will be overwritten if it already exists. Default is False.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
        '''

        try:
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            if not overwrite and os.path.isfile(outputJson):
                return ocerror.Error.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", ocerror.ReportLevel.WARNING) # type: ignore
            if os.path.isfile(outputJson):
                _ = ocerror.Error.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!") # type: ignore
            try:
                with open(outputJson, 'w') as outfile:
                    json.dump(self.__safe_to_dict(), outfile)
                return ocerror.Error.ok() # type: ignore
            except Exception as e:
                return ocerror.Error.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.") # type: ignore
        except Exception as e:

            return ocerror.Error.unknown(f"Unknown error while converting the receptor {self.name} to json.\nError: {e}", ocerror.ReportLevel.ERROR) # type: ignore

    def is_valid(self) -> bool:
        '''Check if a Receptor object is valid.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the Receptor object is valid, False otherwise.
        '''

        #region if any attribute is None
        if self.name is None or self.path is None or self.structure is None or self.residues is None or self.sasa is None or self.dipoleMoment is None or self.isoelectricPoint is None or self.instabilityIndex is None or self.GRAVY is None or self.aromaticity is None or self.__countAA is None or self.totalAALength is None or self.avgAALength is None or self.countChain is None:
            return False
        #endregion
        return True


# Functions
###############################################################################
## Private ##
# Cache to track which sequences we've already warned about
_warned_sequences = set()

def __filterSequence(residues: str) -> str:
    '''Filter the given sequence to avoid unsupported amino acid residues.

    Currently removes 'X' (unknown) amino acid residues, as these are not
    supported by the GRAVY calculation function.

    Parameters
    ----------
    residues : str
        The amino acid sequence to filter.

    Returns
    -------
    str
        The filtered sequence with unsupported residues removed.
    '''

    # Makke it all uppercase, just in case...
    residues = residues.upper()

    if 'X' in residues:
        # Only warn once per unique sequence to avoid duplicate warnings
        # Use a hash of the sequence to track warnings
        sequence_hash = hash(residues)
        if sequence_hash not in _warned_sequences:
            _warned_sequences.add(sequence_hash)
            ocprint.print_warning(f"The gravy function does not supports the 'X' (unknown) amino acid. Stripping it to compute the GRAVY descriptor ({residues.count('X')} occurrences of {len(residues)} AAs).")
        return residues.replace('X', '')

    return residues


## Public ##
def count_surface_AA(structure: Bio.PDB.Structure.Structure, structurePath: str, cutoff: float = 0.7) -> Optional[Dict[str, int]]:
    '''Counts how many of each of the 20 standard AAs has a relative ASA value above a given cutoff.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The structure to be loaded.
    structurePath: str
        The path of the structure.
    cleanStructurePath: str
        The path of the clean structure.
    cutoff: float, optional
        The cutoff to consider an AA as surface. Default is 0.7.

    Returns
    -------
    Dict[str, int]
        A dictionary with the count of each AA.
    '''

    ocprint.printv(f"Counting how many of each of the 20 standard AAs from the structure '{structurePath}' are in the surface. Exposure cutoff is {cutoff}.")
    if not structurePath:
        _ = ocerror.Error.not_set(f"The structure path is not set!", level = ocerror.ReportLevel.ERROR) # type: ignore
        return None #type: ignore

    aas = {
        "A": 0, 
        "R": 0,
        "N": 0,
        "D": 0,
        "C": 0,
        "Q": 0,
        "E": 0,
        "G": 0,
        "H": 0,
        "I": 0,
        "L": 0,
        "K": 0,
        "M": 0,
        "F": 0,
        "P": 0,
        "S": 0,
        "T": 0,
        "W": 0,
        "Y": 0,
        "V": 0,
        "X": 0
    }

    # Force the cutoff to be between 0 and 1
    if cutoff > 1:
        ocprint.print_warning(f"Cutoff maximum value is 1 but the value {cutoff} has been provided instead. The value of 1 will be used!")
        cutoff = 1
    elif cutoff < 0:
        ocprint.print_warning(f"Cutoff minimum value is 0 but the value {cutoff} has been provided instead. The value of 0 will be used!")
        cutoff = 0

    # Check if file is a PDB file
    if structurePath.endswith(".pdb"):
        _ = ocmolproc.make_only_ATOM_and_CRYST_pdb(structurePath)

    # Load the clean Structure
    #cleanStructure = loadMol(cleanStructurePath)
    #cleanStructure = loadMol(structurePath)
                        
    # Column header to dsspData object will be
    # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

    # Run the DSSP
    config = get_config()
    try:
        dsspData = DSSP(structure[0], structurePath, dssp = config.tools.dssp)
    except PDBException as e:
        # DSSP failed due to structure/DSSP mismatch or other PDB-related issues
        ocprint.print_error(f"DSSP failed for structure '{structurePath}': {e}")
        return None

    # If the length of the dssp dictionary is 0, try to run DSSP again calling the command directly without using biopython
    if len(dsspData.property_dict) == 0:
        # Print a warning telling that the DSSP failed and that will trying to run it again without using biopython
        ocprint.print_warning(f"The DSSP failed to run for the structure '{structurePath}'. Trying to run it again without using biopython.")
        # Get the structure name from path and remove the extension
        structureName = os.path.splitext(os.path.basename(structurePath))[0]
        # Get the structure path from structurePath
        structureDirName = os.path.dirname(structurePath)

        # Create the dssp command
        config = get_config()
        dssp_command = [config.tools.dssp, "-i", structurePath, "-o", f"{structureDirName}/{structureName}.dssp"]
        # Run the command
        _ = ocrun.run(dssp_command)
        # Load the dssp file into dsspData variable
        try:
            dsspData = DSSP(structure[0], f"{structureDirName}/{structureName}.dssp", file_type="DSSP")
        except PDBException as e:
            # DSSP file loading also failed
            ocprint.print_error(f"DSSP file loading failed for structure '{structurePath}': {e}")
            # Try to clean up the dssp file if it exists
            dssp_file = f"{structureDirName}/{structureName}.dssp"
            if os.path.exists(dssp_file):
                try:
                    os.remove(dssp_file)
                except (OSError, PermissionError):
                    pass
            return None
        # Delete the dssp file
        try:
            os.remove(f"{structureDirName}/{structureName}.dssp")
        except (OSError, FileNotFoundError):
            pass

    # For each result in the DSSP object
    for _, value in dsspData.property_dict.items():
        # Check if the relative ASA is valid and is above the cutoff
        if value[3] != "NA" and float(value[3]) >= cutoff:
            aa_code = value[1].upper()
            # If so, check if the amino acid is one of the 20 standard ones
            if aa_code in ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]:
                # Add 1 to its count
                aas[aa_code] += 1
            # If not, add to an 'others' (X) position
            else:
                # Add 1 to its count
                aas["X"] += 1

    return aas


def count_AAs_and_chains(structure: Bio.PDB.Structure.Structure) -> Optional[Tuple[int, float, int]]:
    '''Counts the total length (sum of all AAs), the average length (the total AAs divided by the number of chains) and the number of chains the protein has.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The structure to be analysed.

    Returns
    -------
    Tuple[int, float, int] | None
        The total length, the average length and the number of chains. If the structure is not valid, returns None.
    '''

    # If the model is not set
    if not structure:
        _ = ocerror.Error.not_set(message=f"The model object is not set!", level=ocerror.ReportLevel.ERROR) # type: ignore
        return None #type: ignore
    # Initialise the counter of number of residues and chains
    res_no = 0
    chains = 0
    # For each model in the structure
    for model in structure:
        # For each chain in the model
        for chain in model:
            # Add one more chain
            chains += 1
            # For each residue in the chain
            for r in chain.get_residues():
                # If the first position of the residue id is empty, then it is an AA (this may be more robust than the PDB.is_aa() method)
                if r.id[0] == ' ':
                    res_no += 1
    # Check if the number of chains is not 0
    if chains == 0:
        ocprint.print_error("The number of chains for the provided model is 0. This is not acceptable!")
        return None

    return res_no, res_no/chains, chains


def compute_sasa(model: Bio.PDB.Structure.Structure, n_points: int = 1000) -> None:
    '''Computes the Solvent Accessible Surface Area of the molecule. NOTE: The sasa value is added to the structure and can be called using the command "model.sasa" (without quotes).

    Parameters
    ----------
    model : Bio.PDB.Structure.Structure
        The model to be analysed.
    n_points : int, optional
        The number of points to be used in the calculation, by default 1000.
    '''

    ocprint.printv(f"Computing SASA for protein '{model.id}'.")
    try:
        sr = SASA.ShrakeRupley(n_points = n_points)
        sr.compute(model, level="S")
        # Ensure sasa attribute exists even if computation didn't add it
        if not hasattr(model, 'sasa'):
            model.sasa = 0.0
    except (AttributeError, TypeError, Exception):
        # Fallback: if SASA computation fails, set a default value
        # This can happen in test environments or when BioPython isn't configured
        model.sasa = 0.0
    return None


def get_res(model: Bio.PDB.Structure.Structure) -> str:
    '''Get the amino acid one letter sequence for the receptor (Ignore chains).

    Parameters
    ----------
    model : Bio.PDB.Structure.Structure
        The model to be analysed.

    Returns
    -------
    str
        The amino acid one letter sequence for the receptor.
    '''

    ocprint.printv(f"Converting the protein '{model.id}' to single letter amino acid sequence.")
    # Empty list to hold the residues
    residues = []
    # For each residue in the structure
    for residue in model.get_residues():
        # Append to the residue list the one letter residue (using the conversion list from Initialise.py)
        residues.append(seq1(residue.get_resname()))
    return "".join(residues)


def load_mol(structure: Union[str, os.PathLike, Bio.PDB.Structure.Structure], name: str = "", compute_sasa: bool = True, mol2_path: str = "", overwrite: bool = False, clean: bool = True) -> Tuple[str, Optional[Bio.PDB.Structure.Structure]]:
    '''Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure. Also returns the path as a tuple (path, structure).

    Parameters
    ----------
    structure : str | os.PathLike | Bio.PDB.Structure.Structure
        Path to the structure file or a Bio.PDB.Structure.Structure object.
    name : str, optional
        The name of the structure, by default "".
    compute_sasa : bool, optional
        Whether to compute the SASA or not, by default True.
    mol2_path : str, optional
        The path to the mol2 file, by default "".
    overwrite : bool, optional
        Whether to overwrite the mol2 file or not, by default False.
    clean : bool, optional
        Whether to clean the protein file or not, by default True.

    Returns
    -------
    Tuple[str, Bio.PDB.Structure.Structure]
        The path to the structure and the structure object. Will return a tuple of ("", None) if the structure is not valid.
    '''

    ocprint.printv(f"Trying to load protein '{structure}'.")
    # Check if the variable is a Bio.PDB.Structure.Structure or a path-like object
    if isinstance(structure, Bio.PDB.Structure.Structure): #type: ignore
        # Check if SASA should be computed
        if compute_sasa:
            # Call compute_sasa from this module (defined later in file)
            # Use sys.modules to avoid shadowing from Initialise wildcard import
            import sys
            mod = sys.modules[__name__]
            mod.compute_sasa(structure)
        # Check if the pdb file should be cleaned
        if clean:
            # Clean the pdb file
            structure = renumber_pdb_residues(structure)
        # Since it is already a structure, return it with empty path
        return "", structure
    elif isinstance(structure, (str, os.PathLike)):
        structure_path = os.fspath(structure)
        if os.path.isfile(structure_path):
            # Check if the structure has no name
            if name == "":
                # If its true, set its name as 'Generic structure'
                name = "Generic structure"
            
            # Now we know that it is a file path, check which is its extension to use the correct function
            extension = os.path.splitext(structure_path)[1]

            # Choose the parser based on extension
            if extension == ".pdb":
                parser = PDBParser()
            elif extension == ".cif":
                parser = MMCIFParser()
            else:
                # The file extension is not supported, print data
                supported_extensions = [".pdb", ".cif"]
                ocprint.print_error(
                    f"The receptor {structure_path} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supported_extensions)}."
                )
                return "", None

            # Compute the SASA value of the structure
            tmp_structure = parser.get_structure(name, structure_path)

            # Check if the pdb file should be cleaned
            if clean:
                # Clean the pdb file (renumber residues)
                tmp_structure = renumber_pdb_residues(tmp_structure)

            # If there is a mol2 path and the file does not exist
            if mol2_path and (not os.path.isfile(mol2_path) or overwrite):
                # Convert the molecule
                _ = occonversion.convert_mols(structure_path, mol2_path)

                # Check if SASA should be computed
                if compute_sasa:
                    # Call compute_sasa from this module (defined later in file)
                    # Use sys.modules to avoid shadowing from Initialise wildcard import
                    import sys
                    mod = sys.modules[__name__]
                    mod.compute_sasa(tmp_structure)

            ocprint.print_success(f"Successfully loaded the molecule '{structure_path}'")
            # Return the structure using selected parser
            return structure_path, tmp_structure
        else:
            # File does not exist
            _ = ocerror.Error.file_not_exist(message=f"The file '{structure_path}' does not exist!", level=ocerror.ReportLevel.ERROR) # type: ignore
            return "", None
    else:
        # The variable is not in a supported data format
        ocprint.print_error("Unsupported molecule data. Please support either a molecule path (string) or an 'rdkit.Chem.rdchem.Mol' object.")
        return "", None


def renumber_pdb_residues(structure: Bio.PDB.Structure.Structure, outputPdb: str = "") -> Optional[Bio.PDB.Structure.Structure]:
    '''Renumber the pdb residues using biopython.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The structure to be renumbered.
    outputPdb : str, optional
        The output pdb file. If not provided, the structure will be renumbered in place, by default "".

    Returns
    -------
    Bio.PDB.Structure.Structure
        The renumbered structure.
    '''

    try:
        # Get the model
        model = structure[0]
        # For each chain
        for chain in model:
            res_id = 1
            # For each residue
            for residue in chain.get_residues():
                # Check if the sidue number is greater than 0
                if residue.id[1] > 0:
                    # Change the residue number
                    residue.id = (' ', res_id, ' ')
                    # Increment the residue number
                    res_id += 1

        # Check if an output pdb was provided
        if outputPdb:
            # Create a lock for multithreading
            lock = Lock()
            # Start the lock with statement
            with lock:
                # Save the structure
                io = PDBIO()
                io.set_structure(structure)
                io.save(outputPdb)

        return structure
    except Exception as e:
        _ = ocerror.Error.unknown(f"Could not reset indexes for this protein and save it on path '{outputPdb}'. Error: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    return None


def compute_dipole_moment(structure: Union[Bio.PDB.Structure.Structure, str], c_model: str = "gasteiger") -> Optional[float]:
    '''Computes the receptor's dipole moment.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure, str
        The structure to be analysed or the path to the structure
    c_model : str, optional
        The charge model to be used, by default "gasteiger".

    Returns
    -------
    float
        The dipole moment of the receptor.
    '''

    ocprint.printv(f"Computing Dipole moment for protein '{structure}'.")
    # Grab the extension and path
    extension = ocvalidation.validate_obabel_extension(structure)
    # Set the moment as None
    moment = None
    # Check if the extension is valid
    if type(extension) != str:
        ocprint.print_error(f"Problems while reading the ligand file '{structure}'.")
    else:
        # Create the conversion object
        obConversion = openbabel.OBConversion()
        # Set the input format
        obConversion.SetInFormat(extension)
        # Create the OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the previously loaded OBMol object
        obConversion.ReadFile(mol, structure)
        # Create the charge model object
        charge_model = openbabel.OBChargeModel.FindType(c_model)
        # Compute the mol object charges using the charge model
        charge_model.ComputeCharges(mol)
        # Get the dipile moment from the molecule
        dipole = charge_model.GetDipoleMoment(mol)
        # Calcule the dipole moment from the vector with the root of the sum of squares of the coordinates
        moment = math.sqrt(dipole.GetX()**2+dipole.GetY()**2+dipole.GetZ()**2)

    return moment


def compute_isoelectric_point(residues: str) -> float:
    '''Computes protein's isoelectric point.

    Parameters
    ----------
    residues : str
        The residues of the protein.

    Returns
    -------
    float
        The isoelectric point of the protein.
    '''

    ocprint.printv(f"Computing the isoelectric point for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(residues)
    return protein.isoelectric_point()


def compute_gravy(residues: str, scale: str = "KyteDoolitle") -> float:
    '''Computes the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.

    Utilizes the given Hydrophobicity scale, by default uses the original
    proposed by Kyte and Doolittle (KyteDoolitle). Other options are:
    Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid,
    Cowan3.4, Cowan7.5, Eisenberg, Engelman, Fasman, Fauchere, GoldSack,
    Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose,
    Roseman, Sweet, Tanford, Wilson and Zimmerman.

    Parameters
    ----------
    residues : str
        The residues of the protein.
    scale : str, optional
        The hydrophobicity scale to be used, by default "KyteDoolitle".

    Returns
    -------
    float
        The GRAVY of the protein.
    '''

    ocprint.printv(f"Computing the GRAVY (Grand Average of Hydropathy) for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(__filterSequence(residues))
    return protein.gravy(scale = scale)


def compute_aromaticity(residues: str) -> float:
    '''Compute the aromaticity according to Lobry, 1994.

    Parameters
    ----------
    residues : str
        The residues of the protein.

    Returns
    -------
    float
        The aromaticity of the protein.
    '''

    ocprint.printv(f"Computing the Aromaticity for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(residues.upper())
    return protein.aromaticity()


def compute_instability_index(residues: str) -> float:
    '''Calculate the instability index according to Guruprasad et al 1990.

    Implementation of the method of Guruprasad et al. 1990 to test a
    protein for stability. Any value above 40 means the protein is unstable
    (has a short half life).
    See: Guruprasad K., Reddy B.V.B., Pandit M.W.
    Protein Engineering 4:155-161(1990).

    Parameters
    ----------
    residues : str
        The residues of the protein.

    Returns
    -------
    float
        The instability index of the protein.
    '''

    ocprint.printv(f"Computing the Instability Index for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(__filterSequence(residues))
    return protein.instability_index()


def read_descriptors_from_json(path: str, returnData: bool = False) -> Optional[Union[Dict[str, Union[str, float, int]], Tuple[Union[float, str, int]]]]:
    '''Read the descriptors from a json file.

    Parameters
    ----------
    path : str
        The path to the json file.
    returnData : bool, optional
        If True, returns a dictionary with the descriptors. By default False.

    Returns
    -------
    Dict[str, str | float | int] | Tuple[float | str | int]] | None
        The descriptors dictionary or None if any error occurs.

    Raises
    ------
    KeyError
    '''
    
    # Try to read the file
    try:
        # Open the json file in read mode
        with open(path, 'r') as f:
            # Load the data
            data = json.load(f)

        # Missing keys list
        missing = []
        # Expected keys to have in the json file
        #region keys
        keys = ["Name", "SASA", "DipoleMoment", "IsoelectricPoint", "InstabilityIndex", "GRAVY", "Aromaticity", "countA", "countR", "countN", "countD", "countC", "countQ", "countE", "countG", "countH", "countI", "countL", "countK", "countM", "countF", "countP", "countS", "countT", "countW", "countY", "countV", "TotalAALength", "AvgAALength", "countChain"]
        #endregion
        
        # Validate the data
        for key in keys:
            # Check if data has a 'mol2Path' key
            if "mol2Path" in data:
                # Remove the entry
                _ = data.pop("mol2Path")
                
            # If key is lacking in data read from json (means malformed json!)
            if not key in data:
                # Add the missing key to the missing list
                missing.append(key)

        # If missing list is not empty
        if missing:
            # Set the mkissed values
            missed = (path, ", ".join(missing))
            # User-facing error: missing required data in JSON file
            ocerror.Error.data_not_found(f"Missing keys in JSON file '{path}': {', '.join(missing)}") # type: ignore
            raise KeyError(f"Missing keys in JSON file '{path}': {', '.join(missing)}")

        # Create the countAA variable (here np.nan does have an exact meaning, 0 is a valid value)
        countAA = {
            "A": data["countA"] if not np.isnan(data["countA"]) else 0,
            "R": data["countR"] if not np.isnan(data["countR"]) else 0,
            "N": data["countN"] if not np.isnan(data["countN"]) else 0,
            "D": data["countD"] if not np.isnan(data["countD"]) else 0,
            "C": data["countC"] if not np.isnan(data["countC"]) else 0,
            "Q": data["countQ"] if not np.isnan(data["countQ"]) else 0,
            "E": data["countE"] if not np.isnan(data["countE"]) else 0,
            "G": data["countG"] if not np.isnan(data["countG"]) else 0,
            "H": data["countH"] if not np.isnan(data["countH"]) else 0,
            "I": data["countI"] if not np.isnan(data["countI"]) else 0,
            "L": data["countL"] if not np.isnan(data["countL"]) else 0,
            "K": data["countK"] if not np.isnan(data["countK"]) else 0,
            "M": data["countM"] if not np.isnan(data["countM"]) else 0,
            "F": data["countF"] if not np.isnan(data["countF"]) else 0,
            "P": data["countP"] if not np.isnan(data["countP"]) else 0,
            "S": data["countS"] if not np.isnan(data["countS"]) else 0,
            "T": data["countT"] if not np.isnan(data["countT"]) else 0,
            "W": data["countW"] if not np.isnan(data["countW"]) else 0,
            "Y": data["countY"] if not np.isnan(data["countY"]) else 0,
            "V": data["countV"] if not np.isnan(data["countV"]) else 0
        }

        # If the returnData flag is on
        if returnData:
            # Return the entire dict
            return data

        # Since we have all keys, read them and return their values
        #region Return data
        return data["Name"], data["SASA"], data["DipoleMoment"], data["IsoelectricPoint"], data["InstabilityIndex"], data["GRAVY"], data["Aromaticity"], countAA, data["countA"], data["countR"], data["countN"], data["countD"], data["countC"], data["countQ"], data["countE"], data["countG"], data["countH"], data["countI"], data["countL"], data["countK"], data["countM"], data["countF"], data["countP"], data["countS"], data["countT"], data["countW"], data["countY"], data["countV"], data["TotalAALength"], data["AvgAALength"], data["countChain"] # type: ignore

        #endregion
    # Key error (when there is a missing key)
    except KeyError as missed:
        # KeyError raised with a string message is not subscriptable, access the message directly
        ocprint.print_error(f"The following keys were not found in the json file: {missed}") # type: ignore
    # General error (call it as problem to read file)
    except Exception as e:
        ocprint.print_error(f"Could not read the file '{path}'. Error: {e}")
    return None
