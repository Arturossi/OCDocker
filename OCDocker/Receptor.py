#!/usr/lib/python3

# Imports
###############################################################################
import os
from Bio.PDB import *

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

import OCDocker.Receptor as ocpr
'''

# Classes
###############################################################################
class Receptor:
    """
    Load and compute receptor descriptors.
    """

    def __init__(self, molecule, name=""):
        self.name = name
        self.molecule = self.__loadMol(molecule)
        self.residues = self.__getRes()

    def __loadMol(self, molecule):
        '''
        Load a molecule pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the molecule.
        Input:
          molecule [string/Bio.PDB.Structure.Structure] - Path to the molecule file OR Bio.PDB.Structure.Structure object.
        Return:
          [Bio.PDB.Structure.Structure]
          [object] - If the object has been correctly parsed.
          None     - If the object has not been correctly parsed.
        '''
        return loadMol(self, molecule)

    def __getRes(self):
        '''
        Get the amino acid one letter sequence for the receptor (Ignore chains).
        Input:
          -
        Return:
          [string] The amino acid one letter sequence.
        '''
        return getRes(self.molecule)

# Functions
###############################################################################
def getRes(model):
    '''
    Get the amino acid one letter sequence for the receptor (Ignore chains).
    Input:
      model [Bio.PDB.Structure.Structure] - The molecule structure.
    Return:
      [string] The amino acid one letter sequence.
    '''
    # Empty list to hold the residues
    residues = []
    # For each residue in the structure
    for residue in self.residues:
        # Append to the residue list the one letter residue (using the conversion list from Initialise.py)
        residues.append(aa3to1[self.molecule[(int(residue) - 1)].get_resname()])
    return residues.join()

def loadMol(molecule):
    '''
    Load a molecule pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the molecule.
    Input:
      molecule   [string/Bio.PDB.Structure.Structure] - Path to the molecule file OR Bio.PDB.Structure.Structure object.
    Return:
      [Bio.PDB.Structure.Structure]
      [object] - If the object has been correctly parsed.
      None     - If the object has not been correctly parsed.
    '''
    # Check if the type of the variable molecule is a string or a Bio.PDB.Structure.Structure
    if type(molecule) == Bio.PDB.Structure.Structure:
        # Since is already a molecule, assign it to the class
        return molecule
    elif type(molecule) == str:
        # Now we know that it is a file path, check which is its extension to use the correct function
        extension = os.path.splitext(molecule)[1]
        # Choose the parser based on extension
        if extension == ".pdb":
            parser = PDBParser()
        elif extension == ".cif":
            parser = MMCIFParser()
        else:
            # The file extension is not supported, print data
            supportedExtensions = [".pdb", ".cif"]
            octools.print_error(f"The receptor {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.")
            return None
        # Return the molecule using selected parser
        return parser.get_structure("PHA-L", molecule)
    else:
        # The variable is not in a supported data format
        octools.print_error("Unsupported molecule data. Please support either a molecule path (string) or a rdkit.Chem.rdchem.Mol object.")
        return None
