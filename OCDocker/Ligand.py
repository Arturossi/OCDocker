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

import OCDocker.Ligand as ocl
'''

# Classes
###############################################################################
class Ligand:
    """
    Load and compute ligand descriptors. You can provide either a molecule file
    (pdb/sdf/mol/mol2) or a rdkit.Chem.rdchem.Mol object. A name to indentify
    the molecule can be provided aswell.
    """
    def __init__(self, molecule, name=""):
        self.name = name
        self.path, self.molecule = self.__loadMol(molecule)
        self.ExactMolWt = self.__findExactMolWt()
        self.FpDensityMorgan1 = self.__findFpDensityMorgan1()
        self.FpDensityMorgan2 = self.__findFpDensityMorgan2()
        self.FpDensityMorgan3 = self.__findFpDensityMorgan3()
        self.HeavyAtomMolWt = self.__findHeavyAtomMolWt()
        self.MaxAbsPartialCharge = self.__findMaxAbsPartialCharge()
        self.MaxPartialCharge = self.__findMaxPartialCharge()
        self.MinAbsPartialCharge = self.__findMinAbsPartialCharge()
        self.MinPartialCharge = self.__findMinPartialCharge()
        self.MolWt = self.__findMolWt()
        self.NumRadicalElectrons = self.__findNumRadicalElectrons()
        self.NumValenceElectrons = self.__findNumValenceElectrons()

    def __loadMol(self, molecule):
        '''
        Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.
        Input:
          molecule [string/rdkit.Chem.rdchem.Mol] - If a path is provided, parse the molecule (only for single) and return a tuple path, rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return an empty string and the object itself.
        Return:
          [string, rdkit.Chem.rdchem.Mol] - The molecule object.
        '''
        return loadMol(molecule)

    def __findExactMolWt(self):
        '''
        Compute the exact molecular weight of the molecule.
        Input:
          -
        Return:
          [double] - The exact molecular weight.
        '''
        return findExactMolWt(self.molecule)

    def __findFpDensityMorgan1(self):
        '''
        Compute the Morgan fingerprint, radius 1 descriptor of the molecule.
        Input:
          -
        Return:
          [double] - The Morgan fingerprint, radius 1.
        '''
        return findFpDensityMorgan1(self.molecule)

    def __findFpDensityMorgan2(self):
        '''
        Compute the Morgan fingerprint, radius 2 descriptor of the molecule.
        Input:
          -
        Return:
          [double] - The Morgan fingerprint, radius 2.
        '''
        return findFpDensityMorgan2(self.molecule)

    def __findFpDensityMorgan3(self):
        '''
        Compute the Morgan fingerprint, radius 3 descriptor of the molecule.
        Input:
          -
        Return:
          [double] - The Morgan fingerprint, radius 3.
        '''
        return findFpDensityMorgan3(self.molecule)

    def __findHeavyAtomMolWt(self):
        '''
        Compute the heavy atom molecular weight of the molecule.
        Input:
          -
        Return:
          [double] - The heavy atom molecular weight.
        '''
        return findHeavyAtomMolWt(self.molecule)

    def __findMaxAbsPartialCharge(self):
        '''
        Compute the maximum absolute partial charge of the molecule.
        Input:
          -
        Return:
          [double] - The maximum absolute partial charge.
        '''
        return findMaxAbsPartialCharge(self.molecule)

    def __findMaxPartialCharge(self):
        '''
        Compute the absolute partial charge of the molecule.
        Input:
          -
        Return:
          [double] - The absolute partial partial charge.
        '''
        return findMaxPartialCharge(self.molecule)

    def __findMinAbsPartialCharge(self):
        '''
        Compute the minimum absolute partial charge of the molecule.
        Input:
          -
        Return:
          [double] - The minimum absolute partial partial charge.
        '''
        return findMinAbsPartialCharge(self.molecule)

    def __findMinPartialCharge(self):
        '''
        Compute the minimum partial charge of the molecule.
        Input:
          -
        Return:
          [double] - The minimum partial partial charge.
        '''
        return findMinPartialCharge(self.molecule)

    def __findMolWt(self):
        '''
        Compute the molecular weight of the molecule.
        Input:
          -
        Return:
          [double] - The molecular weight.
        '''
        return findMolWt(self.molecule)

    def __findNumRadicalElectrons(self):
        '''
        Compute the number of radical electrons in the molecule.
        Input:
          -
        Return:
          [int] - The number of radical electrons.
        '''
        return findNumRadicalElectrons(self.molecule)

    def __findNumValenceElectrons(self):
        '''
        Compute the number of valence electrons in the molecule.
        Input:
          -
        Return:
          [int] - The number of valence electrons.
        '''
        return findNumValenceElectrons(self.molecule)

    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"Name:                            '{self.name if self.name else '-' }'")
        print(f"Molecule:                        '{self.molecule if self.molecule else '-' }'")
        print(f"Molecule path:                   '{self.path if self.path else '-' }'")
        print(f"Molecular weight:                '{self.MolWt if self.MolWt else '-' }'")
        print(f"Exact molecular weight:          '{self.ExactMolWt if self.ExactMolWt else '-' }'")
        print(f"Morgan fingerprint radius 1:     '{self.FpDensityMorgan1 if self.FpDensityMorgan1 else '-' }'")
        print(f"Morgan fingerprint radius 2:     '{self.FpDensityMorgan2 if self.FpDensityMorgan2 else '-' }'")
        print(f"Morgan fingerprint radius 3:     '{self.FpDensityMorgan3 if self.FpDensityMorgan3 else '-' }'")
        print(f"Heavy atoms molecular weight:    '{self.HeavyAtomMolWt if self.HeavyAtomMolWt else '-' }'")
        print(f"Maximum absolute partial charge: '{self.MaxAbsPartialCharge if self.MaxAbsPartialCharge else '-' }'")
        print(f"Maximum partial charge:          '{self.MaxPartialCharge if self.MaxPartialCharge else '-' }'")
        print(f"Minimum absolute partial charge: '{self.MinAbsPartialCharge if self.MinAbsPartialCharge else '-' }'")
        print(f"Minimum partial charge:          '{self.MinPartialCharge if self.MinPartialCharge else '-' }'")
        print(f"Number of radical electrons:     '{self.NumRadicalElectrons if self.NumRadicalElectrons else '0' }'")
        print(f"Number of valence electrons:     '{self.NumValenceElectrons if self.NumValenceElectrons else '0' }'")

        return

# Functions
###############################################################################
def multipleMoleculesSDF(molecule):
    '''
    Parse a .sdf file with multiple molecules returning a list of ligands.
    Input:
      molecule [string/rdkit.Chem.rdchem.Mol] - If a path is provided, parse the molecule (only for single) and return the rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return itself.
    Return:
      [list(Ligand)] - A list of Ligand objects.
      [None]         - If any problem occurs.
    '''
    # List to hold multiple Ligand objects
    ligands = []
    # Check if the path is a string (it is assumed that the provided path is already a sdf)
    if type(molecule) == str:
        # Check if file exists
        if os.path.isfile(molecule):
            # Check if the extension of the file is .sdf
            if os.path.splitext(molecule)[1] == ".sdf":
                # Get the molecules
                suppl = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule)
                # For each molecule
                for mol in suppl:
                    # Append an instance of the class of the molecule
                    temporaryLigand = Ligand(mol)
                    # Set the path
                    temporaryLigand.path = molecule
                    # Append to the list
                    ligands.append(temporaryLigand)
                return ligands
            else:
                # This case the return code is suppressed because it is needed to return None in case of failure
                _ = errors.wrong_type(message=f"The molecule file MUST be the .sdf format!", level="error")
                return None
        else:
            # File does not exist
            _ = errors.file_do_not_exist(message=f"The file '{molecule}' does not exist!", level="error")
            return None
    else:
        # This case the return code is suppressed because it is needed to return None in case of failure
        _ = errors.wrong_type(message=f"The molecule file path MUST be a string!", level="error")
    return None

def loadMol(molecule):
    '''
    Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.
    Input:
      molecule [string/rdkit.Chem.rdchem.Mol] - If a path is provided, parse the molecule (only for single) and return a tuple path, rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return an empty string and the object itself.
    Return:
      [string, rdkit.Chem.rdchem.Mol]
       [string, object] - The molecule object.
       [string, None]   - If fails to parse the molecule file.
    '''
    # Check if the type of the variable molecule is a string or a rdkit.Chem.rdchem.Mol
    if type(molecule) == rdkit.Chem.rdchem.Mol:
        # Since is already a molecule, assign it to the class
        return "", molecule
    elif type(molecule) == str:
        # Check if file exists
        if os.path.isfile(molecule):
            # Now its a file path, check which is its extension to use the correct function
            extension = os.path.splitext(molecule)[1]

            # Check the extension to see if its needed to convert to mol2
            if extension == ".mol2":
                return molecule, rdkit.Chem.rdmolfiles.MolFromMol2File(molecule)
            else:
                # Since is needed to convert the ligand, create the output path
                outputMoleculePath = f"{os.path.dirname(molecule)}/{os.path.splitext(os.path.basename(molecule))[0]}.mol2"
                print(outputMoleculePath)

                # Process the ligand
                octools.convert2mol2(molecule, outputMoleculePath, logFile = "")

                if extension == ".pdb":
                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule)
                elif extension == ".sdf":
                    # Since the sdf file can hold more than one molecule...
                    mols = molecule, rdkit.Chem.rdmolfiles.SDMolSupplier(molecule)
                    # If has multiple molecules, indicate the user to use the right function
                    if len(mols) > 1:
                        octools.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function multipleMoleculesSDF instead, otherwise just the first molecule will be processed.")
                    # Return just the first molecule
                    return outputMoleculePath, mols[0]
                elif extension == ".mol":
                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromMolFile(molecule)
                else:
                    # The file extension is not supported, print data
                    supportedExtensions = ['.pdb', '.sdf', '.mol', '.mol2']
                    # This case the return code is suppressed because it is needed to return None in case of failure
                    _ = errors.unsupported_extension(message=f"The ligand {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.", level="error")
                    return "", None
        else:
            # File does not exist
            _ = errors.file_do_not_exist(message=f"The file '{molecule}' does not exist!", level="error")
            return "", None
    else:
        # The variable is not in a supported data format
        _ = errors.unsupported_extension(message=f"Unsupported molecule data. Please support either a molecule path (string) or a rdkit.Chem.rdchem.Mol object.", level="error")
        return "", None

def findExactMolWt(molecule):
    '''
    Compute the exact molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The exact molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.ExactMolWt(molecule)
    return None

def findFpDensityMorgan1(molecule):
    '''
    Compute the Morgan fingerprint, radius 1 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Morgan fingerprint, radius 1.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.FpDensityMorgan1(molecule)
    return None

def findFpDensityMorgan2(molecule):
    '''
    Compute the Morgan fingerprint, radius 2 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Morgan fingerprint, radius 2.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.FpDensityMorgan2(molecule)
    return None

def findFpDensityMorgan3(molecule):
    '''
    Compute the Morgan fingerprint, radius 3 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Morgan fingerprint, radius 3.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.FpDensityMorgan3(molecule)
    return None

def findHeavyAtomMolWt(molecule):
    '''
    Compute the heavy atom molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The heavy atom molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.HeavyAtomMolWt(molecule)
    return None

def findMaxAbsPartialCharge(molecule):
    '''
    Compute the maximum absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The maximum absolute partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.MaxAbsPartialCharge(molecule)
    return None

def findMaxPartialCharge(molecule):
    '''
    Compute the absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The absolute partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.MaxPartialCharge(molecule)
    return None

def findMinAbsPartialCharge(molecule):
    '''
    Compute the minimum absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The minimum absolute partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.MinAbsPartialCharge(molecule)
    return None

def findMinPartialCharge(molecule):
    '''
    Compute the minimum partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The minimum partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.MinPartialCharge(molecule)
    return None

def findMolWt(molecule):
    '''
    Compute the molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.MolWt(molecule)
    return None

def findNumRadicalElectrons(molecule):
    '''
    Compute the number of radical electrons in the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The number of radical electrons.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.NumRadicalElectrons(molecule)
    return None

def findNumValenceElectrons(molecule):
    '''
    Compute the number of valence electrons in the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The number of valence electrons.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        return rdkit.Chem.Descriptors.NumValenceElectrons(molecule)
    return None
