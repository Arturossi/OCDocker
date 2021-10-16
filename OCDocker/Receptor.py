#!/usr/lib/python3

# Imports
###############################################################################
import os
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

    def __init__(self, structure, cModel='gasteiger', gravyScale="KyteDoolitle", relativeASAcutoff=0.7, name=""):
        self.name = name
        self.path, self.structure = self.__loadMol(structure)
        self.residues = self.__getRes()
        self.sasa = self.structure.sasa
        self.__cModel = cModel # The options are 'mmff94', 'gasteiger' or 'eem2015bm'
        self.dipoleMoment = self.__computeDipoleMoment()
        self.isoelectricPoint = self.__computeIsoelectricPoint()

        self.gravyScale = gravyScale
        self.GRAVY = self.__computeGravy()


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

    def __count_surface_AA(self):
        '''
        Counts how many of each of the 20 standard AAs has a relative Accessible surface area (ASA) value above a given cutoff.
        Input:
          -
        Return:
          [dict(string)] - A dict containing the number of each AA with a relative ASA value greater than the cutoff.
          [None]         - If the model path is not set.
        '''
        if not self.path:
            _ = errors.not_set(message=f"The model path is not set!", level="error")
            return None
        return count_surface_AA(self.structure, self.path, self.__relativeASAcutoff)

    def __loadMol(self, structure):
        '''
        Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure.
        Input:
          structure [string/Bio.PDB.Structure.Structure] - Path to the structure file OR Bio.PDB.Structure.Structure object.
        Return:
          [Bio.PDB.Structure.Structure] - If the object has been correctly parsed.
          [None]                        - If the object has not been correctly parsed.
        '''
        return loadMol(structure, name=self.name)

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
        return computeGravy(self.residues, scale=self.gravyScale)

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
        print(f"Structure:         '{self.structure if self.structure else '-' }'")
        print(f"AA residues:       '{self.residues if self.residues else '-' }'")
        print(f"SASA:              '{self.sasa if self.sasa else '0.0' }'")
        print(f"Dipole Moment:     '{self.dipoleMoment if self.dipoleMoment else '-' }'")
        print(f"IsoelectricPoint:  '{self.isoelectricPoint if self.isoelectricPoint else '-' }'")
        print(f"GRAVY:             '{self.GRAVY if self.GRAVY else '-' }'")
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
          "SASA": self.sasa if self.sasa else 0.0,
          "DipoleMoment": self.dipoleMoment if self.dipoleMoment else None,
          "IsoelectricPoint": self.isoelectricPoint if self.isoelectricPoint else None,
          "GRAVY": self.GRAVY if self.GRAVY else None,
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
        properties = {
          "Name": self.name if self.name else "-",
          "Path": self.path if self.path else "-",
          "Structure": self.structure if self.structure else "-",
          "Residues": self.residues if self.residues else "-",
          "SASA": self.sasa if self.sasa else 0.0,
          "DipoleMoment": self.dipoleMoment if self.dipoleMoment else "-",
          "IsoelectricPoint": self.isoelectricPoint if self.isoelectricPoint else "-",
          "GRAVY": self.GRAVY if self.GRAVY else "-",
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
        return properties

# Functions
###############################################################################
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

def loadMol(structure, name=""):
    '''
    Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure. Also returns the path as a tuple (path, structure).
    Input:
      name      [string] DEFAULT: ""                 - Name of the structure (if empty the structure's name will be 'Generic structure').
      structure [string/Bio.PDB.Structure.Structure] - Path to the structure file OR Bio.PDB.Structure.Structure object.
    Return:
      [string, Bio.PDB.Structure.Structure]
      [string, object] - If the object has been correctly parsed.
      [string, None]   - If the object has not been correctly parsed.
    '''
    octools.printv(f"Trying to load protein '{structure}'.")
    # Check if the type of the variable structure is a string or a Bio.PDB.Structure.Structure
    if type(structure) == Structure.Structure:
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

def computeIsoelectricPoint(residues):
    '''
    Computes protein's isoelectric point.
    Input:
      residues [string] - The one letter amino acid sequence for the protein.
    Return:
      [float] - Isoelectric point.
    '''
    octools.printv(f"Computing Isoelectric Point for protein with amino acid sequence of '{residues}'.")
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
    residues = residues.upper()
    if 'X' in residues.upper():
        octools.print_warning(f"The gravy function does not supports the 'X' (unknown) amino acid. Stripping it to compute the GRAVY descriptor ({residues.count('X')} occurrences of {len(residues)} AAs).")
        residues = residues.replace('X', '')
    protein = ProteinAnalysis(residues)
    return protein.gravy()
