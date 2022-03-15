#!/usr/lib/python3

# Imports
###############################################################################
import os
import rdkit
import json
from glob import glob

from rdkit import Chem
from rdkit.Chem import Descriptors
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
    def __init__(self, molecule, name, sanitize = True):
        self.name = name.replace(" ", "_")
        self.path, self.molecule = self.__loadMol(molecule, sanitize)
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

    ## Private ##
    def __loadMol(self, molecule, sanitize):
        '''
        Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.
        Input:
          molecule [string/rdkit.Chem.rdchem.Mol] - If a path is provided, parse the molecule (only for single) and return a tuple path, rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return an empty string and the object itself.
        Return:
          [string, rdkit.Chem.rdchem.Mol] - The molecule object.
        '''
        return loadMol(molecule, sanitize)

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

    def __safe_to_dict(self):
        '''
        Return all the properties for the Ligand object.
        Input:
          -
        Return:
          -
        '''
        properties = {
          "Name": self.name if self.name else "-",
          "Path": self.path if self.path else "-",
          "ExactMolWt": self.ExactMolWt if self.ExactMolWt else 0.0,
          "FpDensityMorgan1": self.FpDensityMorgan1 if self.FpDensityMorgan1 else 0,
          "FpDensityMorgan2": self.FpDensityMorgan2 if self.FpDensityMorgan2 else 0,
          "FpDensityMorgan3": self.FpDensityMorgan3 if self.FpDensityMorgan3 else 0,
          "HeavyAtomMolWt": self.HeavyAtomMolWt if self.HeavyAtomMolWt else 0,
          "MaxAbsPartialCharge": self.MaxAbsPartialCharge if self.MaxAbsPartialCharge else 0,
          "MaxPartialCharge": self.MaxPartialCharge if self.MaxPartialCharge else 0,
          "MinAbsPartialCharge": self.MinAbsPartialCharge if self.MinAbsPartialCharge else 0,
          "MinPartialCharge": self.MinPartialCharge if self.MinPartialCharge else 0,
          "MolWt": self.MolWt if self.MolWt else 0,
          "NumRadicalElectrons": self.NumRadicalElectrons if self.NumRadicalElectrons else 0,
          "NumValenceElectrons": self.NumValenceElectrons if self.NumValenceElectrons else 0,
        }
        return properties

    ## Public ##
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

    def get_descriptors(self):
        '''
        Return the descriptors for the Ligand object.
        Input:
          -
        Return:
          [dict] - Dictionary of descriptors for the recpetor.
        '''
        descriptors = {
          "ExactMolWt": self.ExactMolWt if self.ExactMolWt else 0.0,
          "FpDensityMorgan1": self.FpDensityMorgan1 if self.FpDensityMorgan1 else 0,
          "FpDensityMorgan2": self.FpDensityMorgan2 if self.FpDensityMorgan2 else 0,
          "FpDensityMorgan3": self.FpDensityMorgan3 if self.FpDensityMorgan3 else 0,
          "HeavyAtomMolWt": self.HeavyAtomMolWt if self.HeavyAtomMolWt else 0,
          "MaxAbsPartialCharge": self.MaxAbsPartialCharge if self.MaxAbsPartialCharge else 0,
          "MaxPartialCharge": self.MaxPartialCharge if self.MaxPartialCharge else 0,
          "MinAbsPartialCharge": self.MinAbsPartialCharge if self.MinAbsPartialCharge else 0,
          "MinPartialCharge": self.MinPartialCharge if self.MinPartialCharge else 0,
          "MolWt": self.MolWt if self.MolWt else 0,
          "NumRadicalElectrons": self.NumRadicalElectrons if self.NumRadicalElectrons else 0,
          "NumValenceElectrons": self.NumValenceElectrons if self.NumValenceElectrons else 0,
        }
        return descriptors

    def to_dict(self):
        '''
        Return all the properties for the Ligand object.
        Input:
          -
        Return:
          -
        '''
        properties = {
          "Name": self.name if self.name else "-",
          "Path": self.path if self.path else "-",
          "Molecule": self.molecule if self.molecule else "-",
          "ExactMolWt": self.ExactMolWt if self.ExactMolWt else 0.0,
          "FpDensityMorgan1": self.FpDensityMorgan1 if self.FpDensityMorgan1 else 0,
          "FpDensityMorgan2": self.FpDensityMorgan2 if self.FpDensityMorgan2 else 0,
          "FpDensityMorgan3": self.FpDensityMorgan3 if self.FpDensityMorgan3 else 0,
          "HeavyAtomMolWt": self.HeavyAtomMolWt if self.HeavyAtomMolWt else 0,
          "MaxAbsPartialCharge": self.MaxAbsPartialCharge if self.MaxAbsPartialCharge else 0,
          "MaxPartialCharge": self.MaxPartialCharge if self.MaxPartialCharge else 0,
          "MinAbsPartialCharge": self.MinAbsPartialCharge if self.MinAbsPartialCharge else 0,
          "MinPartialCharge": self.MinPartialCharge if self.MinPartialCharge else 0,
          "MolWt": self.MolWt if self.MolWt else 0,
          "NumRadicalElectrons": self.NumRadicalElectrons if self.NumRadicalElectrons else 0,
          "NumValenceElectrons": self.NumValenceElectrons if self.NumValenceElectrons else 0,
        }
        return properties

    def to_json(self, overwrite = False):
        '''
        Stores the descriptors as json to avoid the necessity of evaluate them many times.
        Input:
          ligand [ocl.Ligand] - The ligand to has its descriptors stored as json.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        try:
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            if not overwrite and os.path.isfile(outputJson):
                return errors.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.")
            if os.path.isfile(outputJson):
                octools.print_warning(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!")
            try:
                with open(outputJson, "w") as outfile:
                    json.dump(self.__safe_to_dict(), outfile)
                return errors.ok()
            except Exception as e:
                return errors.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.")
        except Exception as e:
            return errors.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", "error")

# Functions
###############################################################################
## Private ##

## Public ##
def splitMolecules_legacy(molecule, outputDir="", prefix="ligand"):
    '''
    Given a molecule file, checks if it has more than one ligand, if positive, splits the file into multiple single molecule files. Uses external obabel. [DEPRECATED]
    Input:
      molecule  [string]                   - Path to the molecule.
      outputDir [string] DEFAULT: ""       - The output directory. If it is empty the outputDir will be the input dir plus an extra dir called ligand.
      prefix    [string] DEFAULT: "ligand" - The output prefix for ligand file name.
    Return:
      [list(string)] - A list of paths to the new files.
    '''
    # Grab the extension and path
    extension = os.path.splitext(molecule)[1]
    path = os.path.split(os.path.abspath(molecule))[0]
    if not outputDir:
        outputDir = f"{path}/ligands"
    # Initialise an empty list to hold all files paths
    ligand_files = []

    # Check the extension
    if extension == ".sdf":
        # Create the dir if does not exist
        octools.safe_create_dir(outputDir)
        # Create the command
        cmd = [obabel, "-isdf", molecule, "-omol2", "-O", f"{outputDir}/{prefix}.mol2", "-m"]
        # Convert it
        octools.run(cmd, logFile="")
        # Get all mol2 files
        ligand_files = glob(f"{path}/{prefix}*.mol2")
        # Remove the molecule from the list (if it is included)
        if molecule in ligand_files:
            ligand_files(molecule)
    elif extension == ".mol2":
        # Create the dir if does not exist
        octools.safe_create_dir(outputDir)
        # Create the command
        cmd = [obabel, "-imol2", molecule, "-omol2", "-O", f"{outputDir}/{prefix}.mol2", "-m"]
         # Convert it
        octools.run(cmd, logFile="")
        # Get all mol2 files
        ligand_files = glob(f"{path}/{prefix}*.mol2")
        # Remove the molecule from the list (if it is included)
        if molecule in ligand_files:
            ligand_files.remove(molecule)
    else:
        # Since no supported extension has been found, throw the exception
        supportedExtensions = [".sdf", ".mol2"]
        _ = errors.unsupported_extension(message=f"The ligand {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supportedExtensions)}.", level="error")

    return ligand_files

def splitMolecules(molecule, outputDir="", prefix="ligand"):
    '''
    Given a molecule file, checks if it has more than one ligand, if positive, splits the file into multiple single molecule files. Uses openbabel python library.
    Input:
      molecule  [string]                   - Path to the molecule.
      outputDir [string] DEFAULT: ""       - The output directory. If it is empty the outputDir will be the input dir plus an extra dir called ligand.
      prefix    [string] DEFAULT: "ligand" - The output prefix for ligand file name.
    Return:
      [list(string)] - A list of paths to the new files.
    '''
    # Initialise an empty list to hold all files paths
    ligand_files = []
    # Grab the extension and path
    extension = octools.validate_obabel_extension(molecule)
    path = os.path.split(os.path.abspath(molecule))[0]
    # Check if the extension is valid
    if type(extension) != str:
        octools.print_error(f"Problems while reading the ligand file '{inputLigandPath}'.")
    else:
        # Check if outputDir is not set
        if not outputDir:
            outputDir = f"{path}/ligands"
        # Create the conversion object
        obConversion = openbabel.OBConversion()
        # Set the input/output format
        obConversion.SetInAndOutFormats(extension, "mol2")
        # Create the OBMol object
        mol = openbabel.OBMol()
        # Read the first molecule
        molecules = obConversion.ReadFile(mol, molecule)
        # Counter for files
        molNum = 1
        # For each molecule in the file
        while molecules:
            out_path = f"{outputDir}/{prefix}_{molNum}.mol2"
            # Write the mol object to the output performing the conversion
            obConversion.WriteFile(mol, out_path)
            # Recreate mol object
            mol = openbabel.OBMol()
            # Read it again
            molecules = obConversion.Read(mol)
            # Increase the counter
            molNum += 1
            # Add the path to the ligand_files list
            ligand_files.append(out_path)
    return ligand_files

def multipleMoleculesSDF(molecule):
    '''
    Parse a .sdf or .mol2 file with multiple molecules returning a list of ligands.
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
            extension = os.path.splitext(molecule)[1]
            if extension in [".sdf", ".mol2"]:
                # Split the mol file
                molsPaths = splitMolecules(molecule)
                # For each molecule
                for molPath in molsPaths:
                    # Get molecule name
                    name = os.path.splitext(os.path.basename(molecule))[0]
                    # Append to the list
                    ligands.append(Ligand(molPath, name=name))
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

def multipleMoleculesSDF_legacy(molecule):
    '''
    [DEPRECATED]
    Parse a .sdf file with multiple molecules returning a list of ligands.
    Input:
      molecule [string/rdkit.Chem.rdchem.Mol] - If a path is provided, parse the molecule (only for single) and return the rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return itself.
    Return:
      [list(Ligand)] - A list of Ligand objects.
      [None]         - If any problem occurs.
    '''
    octools.print_warning("This function is deprecated since the add of the path attribute in Ligand class. Please use the multipleMoleculesSDF function instead.")
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

def loadMol(molecule, sanitize = True):
    '''
    Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.
    Input:
      molecule [string/rdkit.Chem.rdchem.Mol]               - If a path is provided, parse the molecule (only for single) and return
      a tuple path, rdkit.Chem.rdchem.Mol object. If the molecule is a rdkit.Chem.rdchem.Mol object, return an empty string and the
      object itself.
      sanitize [bool]                         DEFAULT: True - Flag to control if the molecule should be sanitized. (Turn to False only
      if you need and know what you are doing)
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
                # If sanitize is off
                if not sanitize:
                    # Load the molecule
                    m = rdkit.Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = False)
                    # Turn off the property cache
                    m.UpdatePropertyCache(strict = False)
                    # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                    Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                    # Return the sanitized molecule
                    return molecule, m

                return molecule, rdkit.Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = True)
            else:
                # Since is needed to convert the ligand, create the output path
                outputMoleculePath = f"{os.path.dirname(molecule)}/{os.path.splitext(os.path.basename(molecule))[0]}.mol2"

                # Process the ligand
                octools.convert2mol2(molecule, outputMoleculePath)

                if extension == ".pdb":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule
                        m = rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule, sanitize = False)
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                        # Return the sanitized molecule
                        return molecule, m

                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule, sanitize = True)
                elif extension == ".sdf":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule (Since the sdf file can hold more than one molecule...)
                        mol = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = False)
                        if len(mol) > 1:
                            octools.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function splitMolecules to split the ligand into multiple ligand files. Otherwise just the first molecule will be processed.")
                        # Get the first molecule
                        m = mol[0]
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                        # Return the sanitized molecule
                        return molecule, m

                    # Since the sdf file can hold more than one molecule...
                    mols = molecule, rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = True)
                    # If has multiple molecules, indicate the user to use the right function
                    if len(mols) > 1:
                        octools.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function splitMolecules to split the ligand into multiple ligand files. Otherwise just the first molecule will be processed.")

                    # Return just the first molecule
                    return outputMoleculePath, mols[0]
                elif extension == ".mol":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule
                        m = rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = False)
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                        # Return the sanitized molecule
                        return molecule, m

                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = True)
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

# Descriptors functions #

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
