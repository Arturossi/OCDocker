#!/usr/lib/python3

# Imports
###############################################################################
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
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

import OCDocker.Complex as occ
'''

# Classes
###############################################################################
class Complex:
    """
    Load and compute Complex descriptors.
    """

    def __init__(self, molecule, ligand, name=""):
        self.name = name
        self.molecule = self.__loadMol(molecule)
        self.ligand = ligand
        #self.real_energy = self.__read_real_energy()

    ## Private ##
    def __loadMol(self, molecule):
        '''
        Load a molecule pdb if a path is provided or just assign the Mol object to the molecule.
        '''
        return loadMol(molecule)

    ## Public ##
    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:     '{self.name if self.name else '-' }'")
        print(f"Molecule: '{self.molecule if self.molecule else '-' }'")

        return

# Functions
###############################################################################
## Private ##

## Public ##
def loadMol(molecule):
    '''
    Load a molecule pdb if a path is provided or just assign the Mol object to the molecule.
    '''
    return molecule
