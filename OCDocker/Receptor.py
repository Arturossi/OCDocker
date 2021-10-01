#!/usr/lib/python3

# Imports
###############################################################################
import os
from Bio.PDB import *
from Bio.SeqUtils import seq1

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

    def __init__(self, structure, name=""):
        self.name = name
        self.structure = self.__loadMol(structure)
        self.residues = self.__getRes()

    def __loadMol(self, structure):
        '''
        Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure.
        Input:
          structure [string/Bio.PDB.Structure.Structure] - Path to the structure file OR Bio.PDB.Structure.Structure object.
        Return:
          [Bio.PDB.Structure.Structure]
          [object] - If the object has been correctly parsed.
          [None]   - If the object has not been correctly parsed.
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

    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:        '{self.name if self.name else '-' }'")
        print(f"Structure:   '{self.structure if self.structure else '-' }'")
        print(f"AA residues: '{self.residues if self.residues else '-' }'")

# Functions
###############################################################################
def getRes(model):
    '''
    Get the amino acid one letter sequence for the receptor (Ignore chains).
    Input:
      model [Bio.PDB.Structure.Structure] - The structure structure.
    Return:
      [string] The amino acid one letter sequence.
    '''
    # Empty list to hold the residues
    residues = []
    # For each residue in the structure
    for residue in model.get_residues():
        # Append to the residue list the one letter residue (using the conversion list from Initialise.py)
        residues.append(seq1(residue.get_resname()))
    return ''.join(residues)

def loadMol(structure, name=""):
    '''
    Load a structure pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the structure.
    Input:
      name      [string] DEFAULT: ""                 - Name of the structure (if empty the structure's name will be 'Generic structure').
      structure [string/Bio.PDB.Structure.Structure] - Path to the structure file OR Bio.PDB.Structure.Structure object.
    Return:
      [Bio.PDB.Structure.Structure]
      [object] - If the object has been correctly parsed.
      [None]   - If the object has not been correctly parsed.
    '''
    # Check if the type of the variable structure is a string or a Bio.PDB.Structure.Structure
    if type(structure) == Structure.Structure:
        # Since is already a structure, assign it to the class
        return structure
    elif type(structure) == str:
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
            return None
        # Return the structure using selected parser
        return parser.get_structure(name, structure)
    else:
        # The variable is not in a supported data format
        octools.print_error("Unsupported molecule data. Please support either a molecule path (string) or an 'rdkit.Chem.rdchem.Mol' object.")
        return None
