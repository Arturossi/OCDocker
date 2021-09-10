#!/usr/lib/python3

# Imports
###############################################################################
import os
from Bio.PDB import *

import Toolbox as octools

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

import OCDocker.ProcessReceptor as ocpr
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
        """
        Load a molecule pdb/cif if a path is provided or just assign the Bio.PDB.Structure.Structure object to the molecule
        """
        # Check if the type of the variable molecule is a string or a Bio.PDB.Structure.Structure
        if type(molecule) == Bio.PDB.Structure.Structure:
            # Since is already a molecule, assign it to the class
            return molecule
        elif type(molecule) == str:
            # Now its a file path, check which is its extension to use the correct function
            extension = os.path.splitext(molecule)[1]
            if extension == ".pdb":
                parser = PDBParser()
                return parser.get_structure("PHA-L", molecule)
            elif extension == ".cif":
                parser = MMCIFParser()
                return parser.get_structure("PHA-L", molecule)
            else:
                # The file extension is not supported, print data
                supportedExtensions = ['.pdb', '.cif']
                octools.print_error(f"The receptor {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.")
                return None
        else:
            # The variable is not in a supported data format
            octools.print_error("Unsupported molecule data. Please support either a molecule path (string) or a rdkit.Chem.rdchem.Mol object.")
            return None

    def __getres(self):
        rescodes = []
        for residue in self.residues:
            rescodes.append(aa3to1[self.molecule[(int(residue) - 1)].get_resname()])
        return rescodes.join()

# Functions
###############################################################################
