#!/usr/lib/python3

# Imports
###############################################################################
import os
import json
import math

from Bio.PDB import *
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from openbabel import openbabel

from OCDocker.Initialise import *
import OCDocker.Toolbox as octools

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

# Description
###############################################################################
'''
Sets of classes and functions that are used to process all content related to
the ligand.

They are imported as:

import OCDocker.Receptor as ocr
'''

# Classes
###############################################################################
class Receptor:
    """
    Load and compute receptor descriptors.
    """

    def __init__(self, structure, name, mol2Path = "", cModel='gasteiger', gravyScale="KyteDoolitle", relativeASAcutoff=0.7, from_json_descriptors = "", overwrite=False):
        # Name must come first
        self.name = ""
        # The molpath not always will exist (should also come first)
        self.mol2Path = str(mol2Path)
        # Set the path and structure (NEVER SHOUD BE NONE)
        # If user pass a json
        if from_json_descriptors:
            # Read the molecule telling that there is no need to fetch the SASA value
            self.path, self.structure = self.__loadMol(structure, computeSASA=False,overwrite=overwrite)
        else:
            # Read the molecule telling that there is the need to fetch the SASA value
            self.path, self.structure = self.__loadMol(structure, computeSASA=True, overwrite=overwrite)

        # Set the residues (derived from structure)
        self.residues = self.__getRes()

        # Set everything as None
        self.sasa = None
        self.__cModel = None
        self.dipoleMoment = None
        self.isoelectricPoint = None
        self.instabilityIndex = None

        self.__gravyScale = None
        self.GRAVY = None

        self.aromaticity = None

        self.totalLen = None
        self.avgLen = None
        self.chainNumber = None

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
            data = self.__read_descriptors_from_json(from_json_descriptors)
            # If data is None, a problem occurred while reading the json file
            if not data:
                octools.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None
            # <editor-fold> assign
            self.name, self.sasa, self.dipoleMoment, self.isoelectricPoint, self.instabilityIndex,self.GRAVY, self.aromaticity, self.__countAA, self.countA, self.countR, self.countN, self.countD, self.countC, self.countQ, self.countE, self.countG, self.countH, self.countI, self.countL, self.countK, self.countM, self.countF, self.countP, self.countS, self.countT, self.countW, self.countY, self.countV, self.totalLen, self.avgLen, self.chainNumber = data

            # </editor-fold>
        else:
            # Check if the name is empty
            if not name:
                octools.print_error("The Receptor name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            self.totalLen, self.avgLen, self.chainNumber = self.__count_AAs_and_chains()

            self.sasa = self.structure.sasa
            self.__cModel = cModel # The options are 'mmff94', 'gasteiger' or 'eem2015bm'
            self.dipoleMoment = self.__computeDipoleMoment()
            self.isoelectricPoint = self.__computeIsoelectricPoint()
            self.instabilityIndex = self.__computeInstabilityIndex()

            self.__gravyScale = gravyScale
            self.GRAVY = self.__computeGravy()

            self.aromaticity = self.__computeAromaticity()

            # Será que seria interessante? secondary_structure_fraction(self) https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html

            self.__relativeASAcutoff = relativeASAcutoff
            self.__countAA = self.__count_surface_AA()

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
    def __safe_to_dict(self):
        '''
        Return all the properties (except the molecule object) for the Receptor object.
        Input:
          -
        Return:
          [dict of mixed]
        '''
        # Create new dict
        properties = dict()
        # Set Name and Path
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["mol2Path"] = self.path if self.path is not None else "-"
        # Combine both in one dict and return them
        return {**properties, **self.get_descriptors()}

    def __read_descriptors_from_json(self, path):
        '''
        Read the descriptors from a json file.
        Input:
          -
        Return:
          [list(mixed)] - Descriptors read from the json file. If fails, returns null
        '''
        return read_descriptors_from_json(path)

    def __count_surface_AA(self):
        '''
        Counts how many of each of the 20 standard AAs has a relative Accessible surface area (ASA) value above a given cutoff.
        Input:
          -
        Return:
          [dict(string)] - A dict containing the number of each AA with a relative ASA value greater than the cutoff
          [None]         - If the model path is not set
        '''
        if not self.path:
            _ = errors.not_set(message=f"The model path is not set!", level="error")
            return None
        return count_surface_AA(self.structure, self.path, self.__relativeASAcutoff)

    def __count_AAs_and_chains(self):
        '''
        Counts the total length (sum of all AAs), the average length (the total AAs divided by the number of chains) and the number of chains the protein has.
        Input:
          -
        Return:
          [tuple(int, float, int)] - A tuple of the total lenght, the average length and the number of chains
          [None, None, None]     - If the model path is not set
        '''
        if not self.path:
            _ = errors.not_set(message=f"The model path is not set!", level="error")
            return None
        return count_AAs_and_chains(self.structure)

    def __loadMol(self, structure, computeSASA=True, overwrite=False):
        '''
        Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure.
        Input:
          structure   [string/Bio.PDB.Structure.Structure]                - Path to the structure file OR Bio.PDB.Structure.Structure object.
          computeSASA [Bool]                               DEFAULT: True  - Flag to denote if it is needed to compute the SASA descriptor.
          overwrite   [Bool]                               DEFAULT: False - Flag to denote if files will be overwritten.
        Return:
          [Bio.PDB.Structure.Structure] - If the object has been correctly parsed.
          [None]                        - If the object has not been correctly parsed.
        '''
        return loadMol(structure, name=self.name, computeSASA=True, mol2Path=self.mol2Path, overwrite=overwrite)

    def __getRes(self):
        '''
        Get the amino acid one letter sequence for the receptor (Ignore chains).
        Input:
          -
        Return:
          [string] The amino acid one letter sequence.
        '''
        return getRes(self.structure)

    def __computeDipoleMoment(self):
        '''
        Computes the receptor's dipole moment.
        Input:
          -
        Return:
          [float] - The dipole moment value.
          [None]  - If the model path is not set.
        '''
        return computeDipoleMoment(self.path, self.__cModel)

    def __computeIsoelectricPoint(self):
        '''
        Computes protein's isoelectric point.
        Input:
          -
        Return:
          [float] - Isoelectric point.
        '''
        return computeIsoelectricPoint(self.residues)

    def __computeGravy(self):
        '''
        Computes the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
            Utilizes the given Hydrophobicity scale, by default uses the original
            proposed by Kyte and Doolittle (KyteDoolitle). Other options are:
            Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid,
            Cowan3.4, Cowan7.5, Eisenberg, Engelman, Fasman, Fauchere, GoldSack,
            Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose,
            Roseman, Sweet, Tanford, Wilson and Zimmerman.
        Input:
          -
        Return:
          [float] - GRAVY value.
        '''
        return computeGravy(self.residues, scale=self.__gravyScale)

    def __computeAromaticity(self):
        '''
        Compute the aromaticity according to Lobry, 1994.
        Input:
          -
        Return:
          [float] - Aromaticity value.
        '''
        return computeAromaticity(self.residues)

    def __computeInstabilityIndex(self):
        '''
        Calculate the instability index according to Guruprasad et al 1990.
            Implementation of the method of Guruprasad et al. 1990 to test a
            protein for stability. Any value above 40 means the protein is unstable
            (has a short half life).
            See: Guruprasad K., Reddy B.V.B., Pandit M.W.
            Protein Engineering 4:155-161(1990).
        Input:
          -
        Return:
          [float] - Instability Index value.
        '''
        return computeInstabilityIndex(self.residues)

    ## Public ##
    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:              '{self.name if self.name else '-' }'")
        print(f"Structure path:    '{self.path if self.path else '-' }'")
        print(f"mol2 path:         '{self.mol2Path if self.mol2Path else '-' }'")
        print(f"Structure:         '{self.structure if self.structure else '-' }'")
        print(f"AA residues:       '{self.residues if self.residues else '-' }'")
        print(f"Total AA len:      '{self.totalLen if self.totalLen else '0' }'")
        print(f"Average AA len:    '{self.avgLen if self.avgLen else '0' }'")
        print(f"# of chains:       '{self.chainNumber if self.chainNumber else '0' }'")
        print(f"SASA:              '{self.sasa if self.sasa else '0.0' }'")
        print(f"Dipole Moment:     '{self.dipoleMoment if self.dipoleMoment else '-' }'")
        print(f"Isoelectric Point: '{self.isoelectricPoint if self.isoelectricPoint else '-' }'")
        print(f"GRAVY:             '{self.GRAVY if self.GRAVY else '-' }'")
        print(f"Aromaticity:       '{self.aromaticity if self.aromaticity else '-' }'")
        print(f"Instability Index: '{self.instabilityIndex if self.instabilityIndex else '-' }'")
        print(f"# of accessible A: '{self.countA if self.countA else '0' }'")
        print(f"# of accessible R: '{self.countR if self.countR else '0' }'")
        print(f"# of accessible N: '{self.countN if self.countN else '0' }'")
        print(f"# of accessible D: '{self.countD if self.countD else '0' }'")
        print(f"# of accessible C: '{self.countC if self.countC else '0' }'")
        print(f"# of accessible Q: '{self.countQ if self.countQ else '0' }'")
        print(f"# of accessible E: '{self.countE if self.countE else '0' }'")
        print(f"# of accessible G: '{self.countG if self.countG else '0' }'")
        print(f"# of accessible H: '{self.countH if self.countH else '0' }'")
        print(f"# of accessible I: '{self.countI if self.countI else '0' }'")
        print(f"# of accessible L: '{self.countL if self.countL else '0' }'")
        print(f"# of accessible K: '{self.countK if self.countK else '0' }'")
        print(f"# of accessible M: '{self.countM if self.countM else '0' }'")
        print(f"# of accessible F: '{self.countF if self.countF else '0' }'")
        print(f"# of accessible P: '{self.countP if self.countP else '0' }'")
        print(f"# of accessible S: '{self.countS if self.countS else '0' }'")
        print(f"# of accessible T: '{self.countT if self.countT else '0' }'")
        print(f"# of accessible W: '{self.countW if self.countW else '0' }'")
        print(f"# of accessible Y: '{self.countY if self.countY else '0' }'")
        print(f"# of accessible V: '{self.countV if self.countV else '0' }'")

    def get_descriptors(self):
        '''
        Return the descriptors for the Receptor object.
        Input:
          -
        Return:
          [dict] - Dictionary of descriptors for the recpetor.
        '''
        descriptors = {
          "TotalAALength": self.totalLen if self.totalLen else 0,
          "AvgAALength": self.avgLen if self.avgLen else 0.0,
          "countChain": self.chainNumber if self.chainNumber else 0,
          "SASA": self.sasa if self.sasa else 0.0,
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

    def to_dict(self):
        '''
        Return all the properties for the Receptor object.
        Input:
          -
        Return:
          -
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

    def to_json(self, overwrite = False):
        '''
        Stores the descriptors as json to avoid the necessity of evaluate them many times.
        Input:
          overwrite [bool] DEFAULT: False - Flag to allow overwriting the target file.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        try:
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            if not overwrite and os.path.isfile(outputJson):
                return errors.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", "warn")
            if os.path.isfile(outputJson):
                _ = errors.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!")
            try:
                with open(outputJson, "w") as outfile:
                    json.dump(self.__safe_to_dict(), outfile)
                return errors.ok()
            except Exception as e:
                return errors.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.")
        except Exception as e:
            return errors.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", "error")

    def is_valid(self):
        '''
        Check if a Ligand object is valid.
        Input:
          -
        Return:
          [bool]
            True  - if valid
            False - if not valid
        '''
        # <editor-fold> if any attribute is None
        if self.name is None or self.path is None or self.structure is None or self.residues is None or self.sasa is None or self.dipoleMoment is None or self.isoelectricPoint is None or self.instabilityIndex is None or self.GRAVY is None or self.aromaticity is None or self.__countAA is None or self.totalLen is None or self.avgLen is None or self.chainNumber is None:
            return False
        # </editor-fold>
        return True

# Functions
###############################################################################
## Private ##
def __filterSequence(residues):
    '''
    Filter the given sequence to avoid unsupported amino acid residues. (Currently: X)
    Input:
      residues [string]                         - The one letter amino acid sequence for the protein
      scale    [string] DEFAULT: "KyteDoolitle" - Scale to be used
    Return:
      [float] - GRAVY value.
    '''
    residues = residues.upper()
    if 'X' in residues:
        octools.print_warning(f"The gravy function does not supports the 'X' (unknown) amino acid. Stripping it to compute the GRAVY descriptor ({residues.count('X')} occurrences of {len(residues)} AAs).")
        return residues.replace('X', '')
    return residues

## Public ##
def count_surface_AA(model, modelPath, cutoff=0.7):
    '''
    Counts how many of each of the 20 standard AAs has a relative ASA value above a given cutoff.
    Input:
      model     [Bio.PDB.Structure.Structure] - The model to be evaluated.
      modelPath [string]                      - The path to the model which will be evaluated.
      cutoff    [float] Default: 0.7          - Relative ASA cutoff value (Ranges from 0 to 1).
    Return:
      [dict(string)] - A dict containing the number of each AA with a relative ASA value greater than the cutoff.
      [None]         - If the model path is not set.
    '''
    octools.printv(f"Counting how many of each of the 20 standard AAs from the model '{model.id}' are in the surface. Exposure cutoff is {cutoff}.")
    if not modelPath:
        _ = errors.not_set(message=f"The model path is not set!", level="error")
        return None

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
        octools.print_warning(f"Cutoff maximum value is 1 but the value {cutoff} has been provided instead. The value of 1 will be used!")
        cutoff = 1
    elif cutoff < 0:
        octools.print_warning(f"Cutoff minimum value is 0 but the value {cutoff} has been provided instead. The value of 0 will be used!")
        cutoff = 0

    # Column header to dsspData object will be
    # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

    # Run the DSSP
    dsspData = DSSP(model[0], modelPath, dssp=dssp)

    # For each result in the DSSP object
    for key, value in dsspData.property_dict.items():
        # Check if the relative ASA is valid and is above the cutoff
        if value[3] != "NA" and float(value[3]) >= cutoff:
            # If so, check if the amino acid is one of the 20 standard ones
            if value[1] in ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]:
                # Add 1 to its count
                aas[value[1]] += 1
            # If not, add to an 'others' (X) position
            else:
                # Add 1 to its count
                aas["X"] += 1
    return aas

def count_AAs_and_chains(model):
    '''
    Counts the total length (sum of all AAs), the average length (the total AAs divided by the number of chains) and the number of chains the protein has.
    Input:
      model [Bio.PDB.Structure.Structure] - The model to be evaluated.
    Return:
      [tuple(int, float, int)] - A tuple of the total lenght, the average length and the number of chains
      [None, None, None]     - If the model path is not set
    '''
    # If the model is not set
    if not model:
        _ = errors.not_set(message=f"The model object is not set!", level="error")
        return None, None, None
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
        octools.print_error("The number of chains for the provided model is 0. This is not acceptable!")
        return None, None, None

    return res_no, res_no/chains, chains

def compute_sasa(model):
    '''
    Computes the Solvent Accessible Surface Area of the molecule.
        NOTE: The sasa value is added to the structure and can be called like model.sasa
    Input:
      structure [Bio.PDB.Structure.Structure] - The Bio.PDB.Structure.Structure object to compute the SASA value.
    Return:
      -
    '''
    octools.printv(f"Computing SASA for protein '{model.id}'.")
    sr = SASA.ShrakeRupley(n_points=1000)
    sr.compute(model, level="S")
    return

def getRes(model):
    '''
    Get the amino acid one letter sequence for the receptor (Ignore chains).
    Input:
      model [Bio.PDB.Structure.Structure] - The structure structure.
    Return:
      [string] The amino acid one letter sequence.
    '''
    octools.printv(f"Converting the protein '{model.id}' to single letter amino acid sequence.")
    # Empty list to hold the residues
    residues = []
    # For each residue in the structure
    for residue in model.get_residues():
        # Append to the residue list the one letter residue (using the conversion list from Initialise.py)
        residues.append(seq1(residue.get_resname()))
    return "".join(residues)

def loadMol(structure, name="", computeSASA=True, mol2Path="", overwrite=False):
    '''
    Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure. Also returns the path as a tuple (path, structure).
    Input:
      name      [string]                             DEFAULT: ""    - Name of the structure (if empty the structure's name will be 'Generic structure').
      structure [string/Bio.PDB.Structure.Structure]                - Path to the structure file (.pdb or .cif) OR Bio.PDB.Structure.Structure object.
      computeSASA [Bool]                             DEFAULT: True  - Flag to denote if it is needed to compute the SASA descriptor.
      mol2Path  [string]                             DEFAULT: ""    - Path of the mol2 file (if empty no mol2 file will be generated).
      overwrite [Bool]                               DEFAULT: False - Flag to denote if files will be overwritten.
    Return:
      [string, Bio.PDB.Structure.Structure]
      [string, object] - If the object has been correctly parsed.
      [string, None]   - If the object has not been correctly parsed.
    '''
    octools.printv(f"Trying to load protein '{structure}'.")
    # Check if the type of the variable structure is a string or a Bio.PDB.Structure.Structure
    if type(structure) == Structure.Structure:
        # Check if SASA should be computed
        if computeSASA:
            compute_sasa(structure)
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
                octools.print_error(f"The receptor {structure} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.")
                return "", None
            # Compute the SASA value of the structure
            tmpStructure = parser.get_structure(name, structure)
            # If there is a mol2 path and the file does not exist
            if mol2Path and (not os.path.isfile(mol2Path) or overwrite):
                # Convert the molecule
                _ = octools.convertMols(structure, mol2Path)
            # Check if SASA should be computed
            if computeSASA:
                compute_sasa(tmpStructure)
            octools.print_success(f"Successfully loaded the molecule '{structure}'")
            # Return the structure using selected parser
            return structure, tmpStructure
        else:
            # File does not exist
            _ = errors.file_do_not_exist(message=f"The file '{structure}' does not exist!", level="error")
            return "", None
    else:
        # The variable is not in a supported data format
        octools.print_error("Unsupported molecule data. Please support either a molecule path (string) or an 'rdkit.Chem.rdchem.Mol' object.")
        return "", None

def computeDipoleMoment(structure, cModel='gasteiger'):
    '''
    Computes the receptor's dipole moment.
    Input:
      structure [string]                      - Path to the structure to be evaluated.
      cModel    [string] DEFAULT: 'gasteiger' - Charge model to be used. The options are 'mmff94', 'gasteiger' or 'eem2015bm'
    Return:
      [float] - The dipole moment value.
      [None]  - If the model path is not set.
    '''
    octools.printv(f"Computing Dipole moment for protein '{structure}'.")
    # Grab the extension and path
    extension = octools.validate_obabel_extension(structure)
    # Set the moment as None
    moment = None
    # Check if the extension is valid
    if type(extension) != str:
        octools.print_error(f"Problems while reading the ligand file '{inputLigandPath}'.")
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
        chargeModel = openbabel.OBChargeModel_FindType(cModel)
        # Compute the mol object charges using the charge model
        chargeModel.ComputeCharges(mol)
        # Get the dipile moment from the molecule
        dipole = chargeModel.GetDipoleMoment(mol)
        # Calcule the dipole moment from the vector with the root of the sum of squares of the coordinates
        moment = math.sqrt(dipole.GetX()**2+dipole.GetY()**2+dipole.GetZ()**2)

    return moment

def computeIsoelectricPoint(residues):
    '''
    Computes protein's isoelectric point.
    Input:
      residues [string] - The one letter amino acid sequence for the protein.
    Return:
      [float] - Isoelectric point.
    '''
    octools.printv(f"Computing the isoelectric point for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(residues)
    return protein.isoelectric_point()

def computeGravy(residues, scale="KyteDoolitle"):
    '''
    Computes the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
        Utilizes the given Hydrophobicity scale, by default uses the original
        proposed by Kyte and Doolittle (KyteDoolitle). Other options are:
        Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid,
        Cowan3.4, Cowan7.5, Eisenberg, Engelman, Fasman, Fauchere, GoldSack,
        Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose,
        Roseman, Sweet, Tanford, Wilson and Zimmerman.
    Input:
      residues [string]                         - The one letter amino acid sequence for the protein.
      scale    [string] DEFAULT: "KyteDoolitle" - Scale to be used.
    Return:
      [float] - GRAVY value.
    '''
    octools.printv(f"Computing the GRAVY (Grand Average of Hydropathy) for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(__filterSequence(residues))
    return protein.gravy()

def computeAromaticity(residues):
    '''
    Compute the aromaticity according to Lobry, 1994.
    Input:
      residues [string] - The one letter amino acid sequence for the protein.
    Return:
      [float] - Aromaticity value.
    '''
    octools.printv(f"Computing the Aromaticity for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(residues.upper())
    return protein.aromaticity()

def computeInstabilityIndex(residues):
    '''
    Calculate the instability index according to Guruprasad et al 1990.
        Implementation of the method of Guruprasad et al. 1990 to test a
        protein for stability. Any value above 40 means the protein is unstable
        (has a short half life).
        See: Guruprasad K., Reddy B.V.B., Pandit M.W.
        Protein Engineering 4:155-161(1990).
    Input:
      residues [string] - The one letter amino acid sequence for the protein.
    Return:
      [float] - Instability Index value.
    '''
    octools.printv(f"Computing the Instability Index for protein with amino acid sequence of '{residues}'.")
    protein = ProteinAnalysis(__filterSequence(residues))
    return protein.instability_index()

def read_descriptors_from_json(path):
    '''
    Read the descriptors from a json file.
    Input:
      -
    Return:
      [list(mixed)] - Descriptors read from the json file. If fails, returns null.
    '''
    # Try to read the file
    try:
        # Open the json file in read mode
        with open(path, "r") as f:
            # Load the data
            data = json.load(f)
        # Missing keys list
        missing = []
        # Expected keys to have in the json file
        # <editor-fold> keys
        keys = ["Name", "SASA", "DipoleMoment", "IsoelectricPoint", "InstabilityIndex", "GRAVY", "Aromaticity", "countA", "countR", "countN", "countD", "countC", "countQ", "countE", "countG", "countH", "countI", "countL", "countK", "countM", "countF", "countP", "countS", "countT", "countW", "countY", "countV", "totalLen", "avgLen", "chainNumber"]

        # </editor-fold>
        # Validate the data
        for key in keys:
            # If key is lacking in data read from json (means malformed json!)
            if not key in data:
                # Add the missing key to the missing list
                missing.append(key)
        # If missing list is not empty
        if missing:
            # Raise a Key error passing the file and the missing keys joined with ', '
            raise KeyError((path, ", ".join(missing)))
        # Create the countAA variable
        countAA = {
            "A": data["countA"],
            "R": data["countR"],
            "N": data["countN"],
            "D": data["countD"],
            "C": data["countC"],
            "Q": data["countQ"],
            "E": data["countE"],
            "G": data["countG"],
            "H": data["countH"],
            "I": data["countI"],
            "L": data["countL"],
            "K": data["countK"],
            "M": data["countM"],
            "F": data["countF"],
            "P": data["countP"],
            "S": data["countS"],
            "T": data["countT"],
            "W": data["countW"],
            "Y": data["countY"],
            "V": data["countV"]
        }
        # Since we have all keys, read them and return their values
        # <editor-fold> Return data
        return data["Name"], data["SASA"], data["DipoleMoment"], data["IsoelectricPoint"], data["InstabilityIndex"], data["GRAVY"], data["Aromaticity"], countAA, data["countA"], data["countR"], data["countN"], data["countD"], data["countC"], data["countQ"], data["countE"], data["countG"], data["countH"], data["countI"], data["countL"], data["countK"], data["countM"], data["countF"], data["countP"], data["countS"], data["countT"], data["countW"], data["countY"], data["totalLen"], data["avgLen"], data["chainNumber"]

        # </editor-fold>
    # Key error (when there is a missing key)
    except KeyError as k:
        octools.print_error(f"The following keys were not found in the json file '{k[0]}': {k[1]}.")
    # General error (call it as problem to read file)
    except Exception as e:
        octools.print_error(f"Could not read the file '{path}'. Error: {e}")
    return None
