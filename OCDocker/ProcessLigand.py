#!/usr/lib/python3

# Imports
###############################################################################
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
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

import OCDocker.ProcessLigand as ocpl
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
        self.molecule = self.__loadMol(molecule)
        self.ExactMolWt = self.__findExactMolWt()
        self.FpDensityMorgan1 = self.__findFpDensityMorgan1()
        self.FpDensityMorgan2 = self.__findFpDensityMorgan2()
        self.FpDensityMorgan3 = self.__findFpDensityMorgan3()
        self.HeavyAtomMolWt = self.__findHeavyAtomMolWt()
        self.MaxAbsPartialCharge = self.__findMaxAbsPartialCharge()
        self.MaxPartialCharge = self.__findMaxPartialCharge()
        self.MaxAbsPartialCharge = self.__findMaxAbsPartialCharge()
        self.MaxPartialCharge = self.__findMaxPartialCharge()
        self.MinAbsPartialCharge = self.__findMinAbsPartialCharge()
        self.MinPartialCharge = self.__findMinPartialCharge()
        self.MolWt = self.__findMolWt()
        self.NumRadicalElectrons = self.__findNumRadicalElectrons()
        self.NumValenceElectrons = self.__findNumValenceElectrons()

    def __loadMol(self, molecule):
        """
        Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule
        """
        # Check if the type of the variable molecule is a string or a rdkit.Chem.rdchem.Mol
        if type(molecule) == rdkit.Chem.rdchem.Mol:
            # Since is already a molecule, assign it to the class
            return molecule
        elif type(molecule) == str:
            # Now its a file path, check which is its extension to use the correct function
            extension = os.path.splitext(molecule)[1]
            if extension == ".pdb":
                return rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule)
            elif extension == ".sdf":
                # Since the sdf file can hold more than one molecule...
                mols = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule)
                # If has multiple molecules, indicate the user to use the right function
                if len(mols) > 1:
                    octools.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function multipleMoleculesSDF instead, otherwise just the first molecule will be processed.")
                # Return just the first molecule
                return mols[0]
            elif extension == ".mol":
                return rdkit.Chem.rdmolfiles.MolFromMolFile(molecule)
            elif extension == ".mol2":
                return rdkit.Chem.rdmolfiles.MolFromMol2File(molecule)
            else:
                # The file extension is not supported, print data
                supportedExtensions = ['.pdb', '.sdf', '.mol', '.mol2']
                octools.print_error(f"The molecule {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.")
                return None
        else:
            # The variable is not in a supported data format
            octools.print_error("Unsupported molecule data. Please support either a molecule path (string) or a rdkit.Chem.rdchem.Mol object.")
            return None

    def __findExactMolWt(self):
        """
        Compute the exact molecular weight of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.ExactMolWt(self.molecule)
        return None

    def __findFpDensityMorgan1(self):
        """
        Compute the Morgan fingerprint, radius 1 descriptor of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.FpDensityMorgan1(self.molecule)
        return None

    def __findFpDensityMorgan2(self):
        """
        Compute the Morgan fingerprint, radius 2 descriptor of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.FpDensityMorgan2(self.molecule)
        return None

    def __findFpDensityMorgan3(self):
        """
        Compute the Morgan fingerprint, radius 3 descriptor of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.FpDensityMorgan3(self.molecule)
        return None

    def __findHeavyAtomMolWt(self):
        """
        Compute the heavy atom molecular weight of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.HeavyAtomMolWt(self.molecule)
        return None

    def __findMaxAbsPartialCharge(self):
        """
        Compute the maximum absolute partial charge of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.MaxAbsPartialCharge(self.molecule)
        return None

    def __findMaxPartialCharge(self):
        """
        Compute the absolute partial charge of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.MaxPartialCharge(self.molecule)
        return None

    def __findMinAbsPartialCharge(self):
        """
        Compute the minimum absolute partial charge of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.MinAbsPartialCharge(self.molecule)
        return None

    def __findMinPartialCharge(self):
        """
        Compute the minimum partial charge of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.MinPartialCharge(self.molecule)
        return None

    def __findMolWt(self):
        """
        Compute the molecular weight of the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.MolWt(self.molecule)
        return None

    def __findNumRadicalElectrons(self):
        """
        Compute the number of radical electrons in the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.NumRadicalElectrons(self.molecule)
        return None

    def __findNumValenceElectrons(self):
        """
        Compute the number of valence electrons in the molecule
        """
        if self.molecule:
            return rdkit.Chem.Descriptors.NumValenceElectrons(self.molecule)
        return None

# Functions
###############################################################################
def multipleMoleculesSDF(molecule):
    ligands = []
    # Check if the path is a string (it is assumed that the provided path is already a sdf)
    if type(molecule) == str:
        # Get the molecules
        suppl = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule)
        # For each molecule
        for mol in suppl:
            # Append an instance of the class of the molecule
            ligands.append(Ligand(mol))
        return ligands
    else:
        octools.print_error("The molecule file path MUST be a string")
    return None
