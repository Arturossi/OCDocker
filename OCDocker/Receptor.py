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

from Bio.PDB import MMCIFParser, PDBParser, PDBIO, SASA
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from openbabel import openbabel
from threading import Lock
from typing import Dict, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun
import OCDocker.Toolbox.Validation as ocvalidation

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################
class Receptor:
    """Load and compute receptor descriptors."""

    def __init__(self, structure: Union[str, Bio.PDB.Structure.Structure], name: str, mol2Path: str = "", cModel: str = "gasteiger", gravyScale: str = "KyteDoolitle", relativeASAcutoff: float = 0.7, from_json_descriptors: str = "", overwrite: bool = False, clean: bool = False) -> None:  # type: ignore
        '''Constructor of the class Receptor.

        Parameters
        ----------
        structure : str | Bio.PDB.Structure.Structure
            Path to the structure file OR Bio.PDB.Structure.Structure object.
        name : str
            Name of the receptor.
        mol2Path : str, optional
            Path to the mol2 file, by default "".
        cModel : str, optional
            Charge model to be used, by default "gasteiger".
        gravyScale : str, optional
            Scale to be used to compute the GRAVY descriptor, by default "KyteDoolitle".
        relativeASAcutoff : float, optional
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
        self.mol2Path = str(mol2Path)
        # Set the path and structure (NEVER SHOUD BE NONE)
        # If user pass a json
        if from_json_descriptors:
            # Read the molecule telling that there is no need to fetch the SASA value
            self.path, self.structure = loadMol(structure, name=self.name, computeSASA=False, mol2Path=self.mol2Path, overwrite = overwrite, clean = clean)
        else:
            # Read the molecule telling that there is the need to fetch the SASA value
            self.path, self.structure = loadMol(structure, name=self.name, computeSASA=True, mol2Path=self.mol2Path, overwrite = overwrite, clean = clean)

        # Set the residues (derived from structure)
        self.residues = getRes(self.structure)

        # Set everything as None
        self.sasa = None
        self.__cModel = None
        self.dipoleMoment = None
        self.isoelectricPoint = None
        self.instabilityIndex = None

        self.__gravyScale = None
        self.GRAVY = None

        self.aromaticity = None

        self.totalAALength = None
        self.avgAALength = None
        self.countChain = None

        self.__relativeASAcutoff = None
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

            self.sasa = self.structure.sasa
            self.__cModel = cModel # The options are 'mmff94', 'gasteiger' or 'eem2015bm'
            self.dipoleMoment = computeDipoleMoment(self.path, self.__cModel)
            self.isoelectricPoint = computeIsoelectricPoint(self.residues)
            self.instabilityIndex = computeInstabilityIndex(self.residues)

            self.__gravyScale = gravyScale
            self.GRAVY = computeGravy(self.residues, scale=self.__gravyScale)

            self.aromaticity = computeAromaticity(self.residues)

            # Será que seria interessante? secondary_structure_fraction(self) https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html

            self.__relativeASAcutoff = relativeASAcutoff
            
            self.__countAA = count_surface_AA(self.structure, self.path, self.__relativeASAcutoff)

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
            self.countW = self.__countAA["R"]
            self.countY = self.__countAA["Y"]
            self.countV = self.__countAA["V"]

    ## Private ##
    def __safe_to_dict(self) -> Dict:
        '''Return all the properties (except the molecule object) for the Receptor object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict
            A dictionary with all the properties (except the molecule object) for the Receptor object.
        '''

        # Create new dict
        properties = dict()
        # Set Name and Path
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["mol2Path"] = self.path if self.path is not None else "-"
        # Combine both in one dict and return them
        return {**properties, **self.get_descriptors()}

    ## Public ##
    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print(f"Name:                 '{self.name if self.name else '-' }'")
        print(f"Structure path:       '{self.path if self.path else '-' }'")
        print(f"mol2 path:            '{self.mol2Path if self.mol2Path else '-' }'")
        print(f"Structure:            '{self.structure if self.structure else '-' }'")
        print(f"AA residues:          '{self.residues if self.residues else '-' }'")
        print(f"Total AA len:         '{self.totalAALength if self.totalAALength else '0' }'")
        print(f"Average AA len:       '{self.avgAALength if self.avgAALength else '0' }'")
        print(f"# of chains:          '{self.countChain if self.countChain else '0' }'")
        print(f"SASA:                 '{self.sasa if self.sasa else '0.0' }'")
        print(f"Dipole Moment:        '{self.dipoleMoment if self.dipoleMoment else '-' }'")
        print(f"Isoelectric Point:    '{self.isoelectricPoint if self.isoelectricPoint else '-' }'")
        print(f"GRAVY:                '{self.GRAVY if self.GRAVY else '-' }'")
        print(f"Aromaticity:          '{self.aromaticity if self.aromaticity else '-' }'")
        print(f"Instability Index:    '{self.instabilityIndex if self.instabilityIndex else '-' }'")
        print(f"# of accessible A:    '{self.countA if self.countA else '0' }'")
        print(f"# of accessible R:    '{self.countR if self.countR else '0' }'")
        print(f"# of accessible N:    '{self.countN if self.countN else '0' }'")
        print(f"# of accessible D:    '{self.countD if self.countD else '0' }'")
        print(f"# of accessible C:    '{self.countC if self.countC else '0' }'")
        print(f"# of accessible Q:    '{self.countQ if self.countQ else '0' }'")
        print(f"# of accessible E:    '{self.countE if self.countE else '0' }'")
        print(f"# of accessible G:    '{self.countG if self.countG else '0' }'")
        print(f"# of accessible H:    '{self.countH if self.countH else '0' }'")
        print(f"# of accessible I:    '{self.countI if self.countI else '0' }'")
        print(f"# of accessible L:    '{self.countL if self.countL else '0' }'")
        print(f"# of accessible K:    '{self.countK if self.countK else '0' }'")
        print(f"# of accessible M:    '{self.countM if self.countM else '0' }'")
        print(f"# of accessible F:    '{self.countF if self.countF else '0' }'")
        print(f"# of accessible P:    '{self.countP if self.countP else '0' }'")
        print(f"# of accessible S:    '{self.countS if self.countS else '0' }'")
        print(f"# of accessible T:    '{self.countT if self.countT else '0' }'")
        print(f"# of accessible W:    '{self.countW if self.countW else '0' }'")
        print(f"# of accessible Y:    '{self.countY if self.countY else '0' }'")
        print(f"# of accessible V:    '{self.countV if self.countV else '0' }'")

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
        properties["mol2Path"] = self.mol2Path if self.mol2Path is not None else "-"
        properties["Structure"] = self.structure if self.structure is not None else "-"
        
        # Combine both in one dict and return them
        return {**properties, **self.get_descriptors()}

    def to_json(self, overwrite = False) -> int:
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
            return ocerror.Error.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", ocerror.ReportLevel.ERROR) # type: ignore

    def is_valid(self) -> bool:
        '''Check if a Ligand object is valid.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the Ligand object is valid, False otherwise.
        '''

        #region if any attribute is None
        if self.name is None or self.path is None or self.structure is None or self.residues is None or self.sasa is None or self.dipoleMoment is None or self.isoelectricPoint is None or self.instabilityIndex is None or self.GRAVY is None or self.aromaticity is None or self.__countAA is None or self.totalAALength is None or self.avgAALength is None or self.countChain is None:
            return False
        #endregion
        return True

# Functions
###############################################################################
## Private ##
def __filterSequence(residues: str) -> str:
    '''Filter the given sequence to avoid unsupported amino acid residues. (Currently: X)

    Parameters
    ----------
    residues: str
        The sequence to filter.

    Returns
    -------
    str
        The filtered sequence.
    '''

    # Makke it all uppercase, just in case...
    residues = residues.upper()

    if 'X' in residues:
        ocprint.print_warning(f"The gravy function does not supports the 'X' (unknown) amino acid. Stripping it to compute the GRAVY descriptor ({residues.count('X')} occurrences of {len(residues)} AAs).")
        return residues.replace('X', '')

    return residues

## Public ##
def count_surface_AA(structure: Bio.PDB.Structure.Structure, structurePath: str, cutoff: float = 0.7) -> Dict[str, int]: #type: ignore
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
    dsspData = DSSP(structure[0], structurePath, dssp = dssp)

    # If the length of the dssp dictionary is 0, try to run DSSP again calling the command directly without using biopython
    if len(dsspData.property_dict) == 0:
        # Print a warning telling that the DSSP failed and that will trying to run it again without using biopython
        ocprint.print_warning(f"The DSSP failed to run for the structure '{structurePath}'. Trying to run it again without using biopython.")
        # Get the structure name from path and remove the extension
        structureName = os.path.splitext(os.path.basename(structurePath))[0]
        # Get the structure path from structurePath
        structureDirName = os.path.dirname(structurePath)

        # Create the dssp command
        dssp_command = [dssp, "-i", structurePath, "-o", f"{structureDirName}/{structureName}.dssp"]
        # Run the command
        _ = ocrun.run(dssp_command)
        # Load the dssp file into dsspData variable
        dsspData = DSSP(structure[0], f"{structureDirName}/{structureName}.dssp", file_type="DSSP")
        # Delete the dssp file
        os.remove(f"{structureDirName}/{structureName}.dssp")

    # For each result in the DSSP object
    for _, value in dsspData.property_dict.items():
        # Check if the relative ASA is valid and is above the cutoff
        if value[3] != "NA" and float(value[3]) >= cutoff:
            # If so, check if the amino acid is one of the 20 standard ones
            if value[1] in ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "w", "Y", "V"]:
                # Add 1 to its count
                aas[value[1]] += 1
            # If not, add to an 'others' (X) position
            else:
                # Add 1 to its count
                aas["X"] += 1

    return aas

def count_AAs_and_chains(structure: Bio.PDB.Structure.Structure) -> Union[Tuple[int, float, int], None]: #type: ignore
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

def compute_sasa(model: Bio.PDB.Structure.Structure, n_points: int = 1000) -> None: #type: ignore
    '''Computes the Solvent Accessible Surface Area of the molecule. NOTE: The sasa value is added to the structure and can be called using the command "model.sasa" (without quotes).

    Parameters
    ----------
    model : Bio.PDB.Structure.Structure
        The model to be analysed.
    n_points : int, optional
        The number of points to be used in the calculation, by default 1000.

    Returns
    -------
    None
    '''

    ocprint.printv(f"Computing SASA for protein '{model.id}'.")
    sr = SASA.ShrakeRupley(n_points = n_points)
    sr.compute(model, level="S")

    return None

def getRes(model: Bio.PDB.Structure.Structure) -> str: #type: ignore
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

def loadMol(structure: Bio.PDB.Structure.Structure, name: str = "", computeSASA: bool = True, mol2Path: str = "", overwrite: bool = False, clean: bool = True) -> Tuple[str, Bio.PDB.Structure.Structure]: #type: ignore
    '''Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure. Also returns the path as a tuple (path, structure).

    Parameters
    ----------
    structure : str or Bio.PDB.Structure.Structure
        The structure to be loaded.
    name : str, optional
        The name of the structure, by default "".
    computeSASA : bool, optional
        Whether to compute the SASA or not, by default True.
    mol2Path : str, optional
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
    # Check if the type of the variable structure is a string or a Bio.PDB.Structure.Structure
    if type(structure) == Bio.PDB.Structure.Structure: #type: ignore
        # Check if SASA should be computed
        if computeSASA:
            compute_sasa(structure)
        # Check if the pdb file should be cleaned
        if clean:
            # Clean the pdb file
            structure = renumber_pdb_residues(structure)
        # Since is already a structure, assign it to the class
        return structure, None
    elif type(structure) == str:
        if os.path.isfile(structure):
            # Check if the structure has no name
            if name == "":
                # If its true, set its name as 'Generic structure'
                name = "Generic structure"
            
            # Now we know that it is a file path, check which is its extension to use the correct function
            extension = os.path.splitext(structure)[1]

            # Choose the parser based on extension
            if extension == ".pdb":
                parser = PDBParser()
            elif extension == ".cif":
                parser = MMCIFParser()
            else:
                # The file extension is not supported, print data
                supportedExtensions = [".pdb", ".cif"]
                ocprint.print_error(f"The receptor {structure} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.")
                return "", None

            # Compute the SASA value of the structure
            tmpStructure = parser.get_structure(name, structure)

            # Check if the pdb file should be cleaned
            if clean:
                # Clean the pdb file
                tmpStructure = renumber_pdb_residues(tmpStructure)

            # If there is a mol2 path and the file does not exist
            if mol2Path and (not os.path.isfile(mol2Path) or overwrite):
                # Convert the molecule
                _ = occonversion.convertMols(structure, mol2Path)

            # Check if SASA should be computed
            if computeSASA:
                compute_sasa(tmpStructure)

            ocprint.print_success(f"Successfully loaded the molecule '{structure}'")
            # Return the structure using selected parser
            return structure, tmpStructure
        else:
            # File does not exist
            _ = ocerror.Error.file_not_exist(message=f"The file '{structure}' does not exist!", level=ocerror.ReportLevel.ERROR) # type: ignore
            return "", None
    else:
        # The variable is not in a supported data format
        ocprint.print_error("Unsupported molecule data. Please support either a molecule path (string) or an 'rdkit.Chem.rdchem.Mol' object.")
        return "", None

def renumber_pdb_residues(structure: Bio.PDB.Structure.Structure, outputPdb: str = "") -> Bio.PDB.Structure.Structure: #type: ignore
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

def computeDipoleMoment(structure: Union[Bio.PDB.Structure.Structure, str], cModel: str = "gasteiger"): #type: ignore
    '''Computes the receptor's dipole moment.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure, str
        The structure to be analysed or the path to the structure
    cModel : str, optional
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
        chargeModel = openbabel.OBChargeModel.FindType(cModel)
        # Compute the mol object charges using the charge model
        chargeModel.ComputeCharges(mol)
        # Get the dipile moment from the molecule
        dipole = chargeModel.GetDipoleMoment(mol)
        # Calcule the dipole moment from the vector with the root of the sum of squares of the coordinates
        moment = math.sqrt(dipole.GetX()**2+dipole.GetY()**2+dipole.GetZ()**2)

    return moment

def computeIsoelectricPoint(residues: str) -> float:
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

def computeGravy(residues: str, scale: str = "KyteDoolitle") -> float:
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

def computeAromaticity(residues: str) -> float:
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

def computeInstabilityIndex(residues: str) -> float:
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

def read_descriptors_from_json(path: str, returnData: bool = False) -> Union[Dict[str, Union[str, float, int]], Tuple[Union[float, str, int]], None]:
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
            # Raise a Key error passing the file and the missing keys joined with ', '
            raise KeyError

        # Create the countAA variable (here np.NaN does have an exact meaning, 0 is a valid value)
        countAA = {
            "A": data["countA"] if data["countA"] != np.NaN else 0,
            'r': data["countR"] if data["countR"] != np.NaN else 0,
            "N": data["countN"] if data["countN"] != np.NaN else 0,
            "D": data["countD"] if data["countD"] != np.NaN else 0,
            "C": data["countC"] if data["countC"] != np.NaN else 0,
            "Q": data["countQ"] if data["countQ"] != np.NaN else 0,
            "E": data["countE"] if data["countE"] != np.NaN else 0,
            "G": data["countG"] if data["countG"] != np.NaN else 0,
            "H": data["countH"] if data["countH"] != np.NaN else 0,
            "I": data["countI"] if data["countI"] != np.NaN else 0,
            "L": data["countL"] if data["countL"] != np.NaN else 0,
            "K": data["countK"] if data["countK"] != np.NaN else 0,
            "M": data["countM"] if data["countM"] != np.NaN else 0,
            "F": data["countF"] if data["countF"] != np.NaN else 0,
            "P": data["countP"] if data["countP"] != np.NaN else 0,
            "S": data["countS"] if data["countS"] != np.NaN else 0,
            "T": data["countT"] if data["countT"] != np.NaN else 0,
            'w': data["countW"] if data["countW"] != np.NaN else 0,
            "Y": data["countY"] if data["countY"] != np.NaN else 0,
            "V": data["countV"] if data["countV"] != np.NaN else 0
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
        ocprint.print_error(f"The following keys were not found in the json file '{missed[0]}': {missed[1]}.") # type: ignore
    # General error (call it as problem to read file)
    except Exception as e:
        ocprint.print_error(f"Could not read the file '{path}'. Error: {e}")
    return None
