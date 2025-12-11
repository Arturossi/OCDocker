#!/usr/lib/python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process all content related to
the ligand.

They are imported as:

import OCDocker.Ligand as ocl
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import os
import rdkit

from openbabel import openbabel
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from threading import Lock
from typing import Callable, Dict, List, Tuple, Union, Optional

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

# Set output levels for openbabel
ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(ocerror.Error.output_level)
if ocerror.Error.output_level == ocerror.ReportLevel.NONE:
    RDLogger.DisableLog("rdApp.*") # type: ignore

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################
class Ligand:
    """Represents a ligand (small molecule) with computed molecular descriptors.

    This class loads ligand structures from various file formats (PDB, SDF, MOL,
    MOL2, SMILES) or accepts RDKit molecule objects, and computes a wide range
    of molecular descriptors including 2D descriptors (BalabanJ, BertzCT, etc.)
    and 3D descriptors (RadiusOfGyration, Asphericity, etc.).

    Parameters
    ----------
    molecule : str | rdkit.Chem.rdchem.Mol
        Path to a molecule file or an RDKit molecule object.
    name : str
        Name identifier for the ligand.
    sanitize : bool, optional
        Whether to sanitize the molecule after loading, by default True.
    from_json_descriptors : str, optional
        Path to JSON file containing pre-computed descriptors, by default "".    
    """

    # Declare the descriptors names as class attributes
    descriptors_names = {
        'AUTOCORR2D_': range(1, 193),
        'BCUT2D_': ["CHGHI", "CHGLO", "LOGPHI", "LOGPLOW", "MRHI", "MRLOW", "MWHI", "MWLOW"],
        'fr_': ["Al_COO", "Al_OH", "Al_OH_noTert", "ArN", "Ar_COO", "Ar_N", "Ar_NH", "Ar_OH", "COO", "COO2", "C_O", "C_O_noCOO", "C_S", "HOCCN", "Imine", "NH0", "NH1", "NH2", "N_O", "Ndealkylation1", "Ndealkylation2", "Nhpyrrole", "SH", "aldehyde", "alkyl_carbamate", "alkyl_halide", "allylic_oxid", "amide", "amidine", "aniline", "aryl_methyl", "azide", "azo", "barbitur", "benzene", "benzodiazepine", "bicyclic", "diazo", "dihydropyridine", "epoxide", "ester", "ether", "furan", "guanido", "halogen", "hdrzine", "hdrzone", "imidazole", "imide", "isocyan", "isothiocyan", "ketone", "ketone_Topliss", "lactam", "lactone", "methoxy", "morpholine", "nitrile", "nitro", "nitro_arom", "nitro_arom_nonortho", "nitroso", "oxazole", "oxime", "para_hydroxylation", "phenol", "phenol_noOrthoHbond", "phos_acid", "phos_ester", "piperdine", "piperzine", "priamide", "prisulfonamd", "pyridine", "quatN", "sulfide", "sulfonamd", "sulfone", "term_acetylene", "tetrazole", "thiazole", "thiocyan", "thiophene", "unbrch_alkane", "urea"],
        'Chi': [str(j) + i if j < 2 else f"{j}{i}" for j in range(5) for i in ('', 'v', 'n') if not (j >= 2 and i == '')],
        'EState_VSA': range(1, 12),
        'FpDensityMorgan': range(1, 4),
        'Kappa': range(1, 4),
        'Mol': ["LogP", "MR", "Wt"],
        'Num': ["AliphaticCarbocycles", "AliphaticHeterocycles", "AliphaticRings", "AromaticCarbocycles", "AromaticHeterocycles", "AromaticRings", "HAcceptors", "HDonors", "Heteroatoms", "RadicalElectrons", "RotatableBonds", "SaturatedCarbocycles", "SaturatedHeterocycles", "SaturatedRings", "ValenceElectrons"],
        'NPR': range(1, 3),
        'PMI': range(1, 4),
        'PEOE_VSA': range(1, 15),
        'SMR_VSA': range(1, 11),
        'SlogP_VSA': range(1, 13),
        'VSA_EState': range(1, 11),
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = [
        'BalabanJ', 'BertzCT', 'ExactMolWt', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomMolWt', 
        'HeavyAtomCount', 'LabuteASA', 'TPSA', 'MaxAbsEStateIndex', 'MaxEStateIndex', 
        'MinAbsEStateIndex', 'MinEStateIndex', 'MaxAbsPartialCharge', 'MaxPartialCharge', 
        'MinAbsPartialCharge', 'MinPartialCharge', 'qed', 'RingCount', 'Asphericity', 'Eccentricity',
        'InertialShapeFactor', 'RadiusOfGyration', 'SpherocityIndex', 'NHOHCount', 'NOCount'
    ]

    # Create all the descriptors to be class attributes
    allDescriptors = [f'{desc_prefix}{i}' for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

    def __init__(self, molecule: Union[str, rdkit.Chem.rdchem.Mol], name: str, sanitize: bool = True, from_json_descriptors: str = "") -> Optional[int]:
        ''' Constructor for the Ligand class.
        
        Parameters
        ----------
        molecule : str | rdkit.Chem.rdchem.Mol
            The molecule to be processed. If a string is provided, it is assumed to be a path to a molecule file (pdb/sdf/mol/mol2). If a rdkit.Chem.rdchem.Mol object is provided, it is assumed to be a molecule object.
        name : str
            The name of the molecule.
        sanitize : bool
            If True, the molecule will be sanitized.
        from_json_descriptors : str
            If a path to a json file is provided, the descriptors will be read from the file instead of being computed.

        Returns
        -------
        int | None
            Returns None if the molecule was loaded successfully, otherwise an error code.
        '''

        # Set the path and structure (NEVER SHOUD BE NONE)
        self.path, self.molecule = load_mol(molecule, sanitize) # type: ignore
        # Set the box_path (removing the file from the path)
        self.box_path = os.path.join(os.path.dirname(self.path), "boxes/box0.pdb")
        
        # Define everything as None, except for the name
        if not "_split_" in name:
            self.name = name
        else:
            return ocerror.Error.invalid_molecule_name("The name of the ligand cannot contain the string '_split_'") # type: ignore

        # All attribute initializations
        for desc in Ligand.allDescriptors:
            setattr(self, desc, None)

        # Set the from_json_descriptors attribute
        self.from_json_descriptors = from_json_descriptors

        # Set the sanitize attribute
        self.sanitize = sanitize

        # If user pass a json
        if from_json_descriptors:
            # Read the descriptors from it
            data = read_descriptors_from_json(from_json_descriptors, return_data = True)
            # If data is None, a problem occurred while reading the json file
            if not data:
                ocprint.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None
            
            #region assign
            # Handle both 'Name' and 'Ligand' keys (read_descriptors_from_json with return_data=True renames 'Name' to 'Ligand')
            self.name = data.get("Name") or data.get("Ligand") # type: ignore

            # All attribute initializations
            for desc in Ligand.allDescriptors:
                setattr(self, desc, f"{data[desc]}") # type: ignore
            #endregion

        else:
            # Check if the name is empty
            if not name:
                ocprint.print_error("The Ligand name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            # Single attribute initializations
            for desc in Ligand.allDescriptors:
                result = globals()[f"find{desc}"](self.molecule)

                setattr(self, f"{desc}", result)

    ## Private ##
    def __safe_to_dict(self) -> Dict[str, Union[int, float]]:
        '''Return all the properties (except the molecule object) for the Ligand object.

        Parameters
        ----------
        None
        
        Returns
        -------
        Dict[str, int | float]
            The properties of the Ligand object.
        '''

        # Create new dict
        properties = dict()

        # Set Name and Path
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"

        # Combine both in one dict and return them

        return {**properties, **self.get_descriptors()}

    def __repr__(self) -> str:
        '''Return a string representation of the Ligand object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A string representation of the Ligand object.
        '''

        return f"Ligand(molecule={self.molecule}, name={self.name}, sanitize={self.sanitize}, from_json_descriptors={'True' if self.from_json_descriptors else 'False'})"

    ## Public ##
    def print_attributes(self) -> None:
        '''Print all attributes of the ligand to stdout.

        Displays the ligand's name, molecule object, and all computed
        molecular descriptors (BalabanJ, BertzCT, RadiusOfGyration, etc.)
        in a formatted, aligned table.
        '''

        #region prints
        
        # Initialize descriptors in each category
        # Calculate maximum length of descriptor names for both categories
        max_length = max(
            max(len(desc) for desc in Ligand.allDescriptors),
            len("Name"),
            len("Molecule")
        ) + 5

        print(f"{'Name'.ljust(max_length)}: '{self.name if self.name is not None else '-'}'")
        print(f"{'Molecule'.ljust(max_length)}: '{self.molecule if self.molecule is not None else '-'}'")

        # All attribute initializations
        for desc in Ligand.allDescriptors:
            attribute = getattr(self, desc)
            # Pad the attribute name to align
            print(f"{desc.ljust(max_length)}: '{attribute if attribute is not None else '-'}'")
        #endregion

        return None

    def get_descriptors(self) -> Dict[str, Union[int, float]]:
        '''Return the descriptors for the Ligand object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Union[int, float]]
            A dictionary of the descriptors for the Ligand object.
        '''

        descriptors = {}

        # All attribute initializations
        for desc in Ligand.allDescriptors:
            attribute = getattr(self, desc)
            descriptors[desc] = attribute if attribute is not None else 0.0
        
        return descriptors # type: ignore

    def to_dict(self) -> Dict[str, Union[int, float, str]]:
        '''Return all the properties for the Ligand object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Union[int, float, str]]
            A dictionary of all the properties for the Ligand object.
        '''

        # Create new dict
        properties = dict()

        # Set Name, Path and molecule
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["Molecule"] = self.molecule if self.molecule is not None else "-"

        # Combine both in one dict and return them

        return {**properties, **self.get_descriptors()}

    def to_json(self, overwrite: bool = False) -> int:
        '''Stores the descriptors as json to avoid the necessity of evaluate them many times.

        Parameters
        ----------
        overwrite : bool, optional
            If True, the json file will be overwritten, by default False.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
        '''

        try:
            # Parameterize the path
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            # Check if the file exists
            if os.path.isfile(outputJson):
                # Check if the user wants to overwrite the file
                if not overwrite:
                    # If the file exists and overwrite is False, return the file exists error
                    return ocerror.Error.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", ocerror.ReportLevel.WARNING) # type: ignore
                # Warns the user that the file will be overwritten
                _ = ocerror.Error.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!") # type: ignore

            try:
                # Create a lock for multithreading
                lock = Lock()
                with lock:
                    # Open the file for writing
                    with open(outputJson, 'w') as outfile:
                        # Write the json file
                        json.dump(self.__safe_to_dict(), outfile)
                return ocerror.Error.ok() # type: ignore
            except Exception as e:
                return ocerror.Error.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.") # type: ignore
        except Exception as e:

            return ocerror.Error.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", ocerror.ReportLevel.WARNING) # type: ignore

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

        #region if any attribute is None (will check for every attribute in the ligand object)
        if any(getattr(self, attr) is None for attr in Ligand.allDescriptors):
            return False
        #endregion

        return True

    def to_smiles(self) -> Union[str, int]:
        '''Return the smiles of the molecule.

        Parameters
        ----------
        None

        Returns
        -------
        str | int
            The smiles of the molecule, if fails the exit code of the command (based on the Error.py code table).
        '''

        return get_smiles(self.molecule)

    def is_same_molecule(self, molecule: Union[rdkit.Chem.rdchem.Mol, Ligand], sanitize: bool = True) -> Union[bool, int]:
        '''Compare two molecules to check if they are the same using their MACCSkeys.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol | ocl.Ligand
            The molecule to compare with.
        sanitize : bool, optional
            Flag to allow, or not, molecules sanitization. (default is True)

        Returns
        -------
        bool | int
            If both molecules are the same, return True. If both molecules are not the same, return False. If fails, return an error code.
        '''

        # Get the MACCSKeys for the ligand object
        ligandMACCSSKeys = MACCSkeys.GenMACCSKeys(self.molecule) # type: ignore

        # Check if the type of the molecule is a Ligand
        if type(molecule) == Ligand:
            # If yes, get its MACCSKeys
            targetMACCSSKeys = MACCSkeys.GenMACCSKeys(molecule.molecule) # type: ignore
        # Otherwise check if it is a Chem.rdchem.Mol object
        elif type(molecule) == Chem.rdchem.Mol:
            # If it is, get its smiles using the Ligand public function, get_smiles()
            targetMACCSSKeys = MACCSkeys.GenMACCSKeys(molecule) # type: ignore
        # If is neither both types above
        else:
            # Return an error
            return ocerror.Error.wrong_type(f"The provided variable is a '{type(molecule)}' and was expected a 'rdkit.Chem.rdchem.Mol' or 'ocl.Ligand'.") # type: ignore
        
        # Check if the Fingerprints are the same using the Tanimoto similarity
        if DataStructs.FingerprintSimilarity(ligandMACCSSKeys, targetMACCSSKeys) == 1.0:
            # If they are the same, return True
            return True

        # Otherwise (they are not the same)
        return False

    def is_same_molecule_SMILES(self, molecule: Union[rdkit.Chem.rdchem.Mol, Ligand], sanitize: bool = True) -> Union[bool, int]:
        '''Compare two molecules to check if they are the same using their SMILES and FpDensityMorgan 1 2 and 3.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol | ocl.Ligand
            The molecule to compare with.
        sanitize : bool, optional
            Flag to allow, or not, molecules sanitization. (default is True)

        Returns
        -------
        bool | int
            If both molecules are the same, return True. If both molecules are not the same, return False. If fails, return an error code.
        '''

        try:
            # Check if the molecule is a Ligand and get the properties directly
            if isinstance(molecule, Ligand):
                target_molecule_smiles = molecule.to_smiles()
                target_mol_morgan_fps = (
                    molecule.FpDensityMorgan1, # type: ignore
                    molecule.FpDensityMorgan2, # type: ignore
                    molecule.FpDensityMorgan3, # type: ignore
                )
            # If it's an RDKit Mol, calculate the properties
            elif isinstance(molecule, Chem.rdchem.Mol):
                mol = load_mol(molecule, sanitize=sanitize)
                target_molecule_smiles = get_smiles(mol)
                target_mol_morgan_fps = (
                    findFpDensityMorgan1(mol), # type: ignore
                    findFpDensityMorgan2(mol), # type: ignore
                    findFpDensityMorgan3(mol), # type: ignore
                )
            else:
                _ = ocerror.Error.wrong_type(f"The provided variable is of type '{type(molecule)}'; expected 'rdkit.Chem.rdchem.Mol' or 'ocl.Ligand'.") # type: ignore
                return False

            # Compare SMILES first to short-circuit if they are different
            if self.to_smiles() != target_molecule_smiles:
                return False

            # Compare fingerprints
            self_mol_morgan_fps = (
                self.FpDensityMorgan1, # type: ignore
                self.FpDensityMorgan2, # type: ignore
                self.FpDensityMorgan3, # type: ignore
            )

            return self_mol_morgan_fps == target_mol_morgan_fps
        except Exception as e:
            _ = ocerror.Error.unknown(f"Unknown error while checking smiles: {str(e)}") # type: ignore

        return False

    def get_centroid(self, sanitize: bool = True) -> rdkit.Geometry.rdGeometry.Point3D:
        '''Get the centroid of the molecule.

        Parameters
        ----------
        sanitize : bool, optional
            Flag to allow, or not, molecules sanitization. (default is True)

        Returns
        -------
        rdkit.Geometry.rdGeometry.Point3D
            The centroid of the molecule.
        '''

        # Compute the centroid of the molecule and return it
        return get_centroid(self.molecule, sanitize = sanitize)

    def create_box(self, centroid: Optional[Tuple[float, float, float]] = None, save_path: str = "", box_length: float = 2.9, overwrite: bool = False) -> Optional[int]:
        '''Create a box file to be used by docking software.

        Parameters
        ----------
        centroid : tuple | None, optional
            The centroid of the box. If not provided, the centroid of the molecule will be used. (default is None)
        save_path : str, optional
            The path to save the box file. If not provided, the box file will be saved in the same path as the molecule. (default is "", which turns into self.box_path)
        box_length : float, optional
            The length of the box. (default is 2.9)
        overwrite : bool, optional
            Flag to allow, or not, the overwrite of the box file. (default is False)

        Returns
        -------
        int | None
            If the box file was created, return None. If fails, return the exit code of the command (based on the Error.py code table).
        '''

        # Check if the box file already exists
        if os.path.isfile(f"{save_path}/box0.pdb") and not overwrite:
            # If it exists and the overwrite flag is False, return an error
            return ocerror.Error.file_exists(f"The box file '{save_path}' already exists. If you want to overwrite it, set the 'overwrite' flag to True.") # type: ignore
            
        # If the centroid is not defined
        if not centroid:
            # Compute it
            centroid = self.get_centroid()

        # Check if the centroid is the type rdkit.Geometry.rdGeometry.Point3D
        if type(centroid) == rdkit.Geometry.rdGeometry.Point3D: # type: ignore
            centroid = (centroid.x, centroid.y, centroid.z) # type: ignore

        # Get the partial size for each axis (to determine how much should be expanded in each direction)
        partialSize = (box_length * self.RadiusOfGyration) / 2 # type: ignore

        # Create the box using this Centroid
        box = {
            "min_x": centroid[0] - partialSize, # type: ignore
            "max_x": centroid[0] + partialSize, # type: ignore
            "min_y": centroid[1] - partialSize, # type: ignore
            "max_y": centroid[1] + partialSize, # type: ignore
            "min_z": centroid[2] - partialSize, # type: ignore
            "max_z": centroid[2] + partialSize  # type: ignore
        }

        # Get dimensions for each axis and its center (round to 3 decimals)
        dim_x = abs(round(box["max_x"] - box["min_x"], 3))
        dim_y = abs(round(box["max_y"] - box["min_y"], 3))
        dim_z = abs(round(box["max_z"] - box["min_z"], 3))

        # Get the size of the center (starting from the origin) (not using dim because I want to round only once)
        center_x = abs((box["max_x"] - box["min_x"])/2)
        center_y = abs((box["max_y"] - box["min_y"])/2)
        center_z = abs((box["max_z"] - box["min_z"])/2)
        # Since the boxes might not have one corner at the origin, shift it in all directions X,Y,Z
        center_x = round(center_x + box["min_x"], 3)
        center_y = round(center_y + box["min_y"], 3)
        center_z = round(center_z + box["min_z"], 3)

        # Convert the values found above to string with 8 chars (complete with spaces to the left) as the .pdb file model
        min_x = " " * (8 - len(str(round(box["min_x"], 3)))) + str(round(box["min_x"], 3))
        max_x = " " * (8 - len(str(round(box["max_x"], 3)))) + str(round(box["max_x"], 3))
        min_y = " " * (8 - len(str(round(box["min_y"], 3)))) + str(round(box["min_y"], 3))
        max_y = " " * (8 - len(str(round(box["max_y"], 3)))) + str(round(box["max_y"], 3))
        min_z = " " * (8 - len(str(round(box["min_z"], 3)))) + str(round(box["min_z"], 3))
        max_z = " " * (8 - len(str(round(box["max_z"], 3)))) + str(round(box["max_z"], 3))

        dim_x = " " * (8 - len(str(round(dim_x, 3)))) + str(round(dim_x, 3))
        dim_y = " " * (8 - len(str(round(dim_y, 3)))) + str(round(dim_y, 3))
        dim_z = " " * (8 - len(str(round(dim_z, 3)))) + str(round(dim_z, 3))

        center_x = " " * (8 - len(str(round(center_x, 3)))) + str(round(center_x, 3))
        center_y = " " * (8 - len(str(round(center_y, 3)))) + str(round(center_y, 3))
        center_z = " " * (8 - len(str(round(center_z, 3)))) + str(round(center_z, 3))

        # If the save_path is not defined
        if not save_path:
            # Set it as the same dir as the ligand
            save_path = os.path.join(os.path.split(self.path)[0], 'boxes')
            # If the save_path does not exist
            if not os.path.exists(save_path):
                # Create it
                _ = ocff.safe_create_dir(save_path)
        else:
            # If the save_path does not exist, warn the user
            if not os.path.exists(save_path):
                _ =  ocerror.Error.dir_not_exist(f"The save_path '{save_path}' does not exist. Creating it.", level = ocerror.ReportLevel.WARNING) # type: ignore
                os.mkdir(save_path)

        # Write out the box file (following the one given in the DUD-E database)
        with open(f"{save_path}/box0.pdb", 'w') as f:
            f.write(f"HEADER    CORNERS OF BOX      {min_x}{min_y}{min_z}{max_x}{max_y}{max_z}\n")
            f.write(f"REMARK    CENTER (X Y Z)      {center_x}{center_y}{center_z}\n")
            f.write(f"REMARK    DIMENSIONS (X Y Z)  {dim_x}{dim_y}{dim_z}\n")
            f.write(f"ATOM      1  DUA BOX     1    {min_x}{min_y}{min_z}\n")
            f.write(f"ATOM      2  DUB BOX     1    {max_x}{min_y}{min_z}\n")
            f.write(f"ATOM      3  DUC BOX     1    {max_x}{min_y}{max_z}\n")
            f.write(f"ATOM      4  DUD BOX     1    {min_x}{min_y}{max_z}\n")
            f.write(f"ATOM      5  DUE BOX     1    {min_x}{max_y}{min_z}\n")
            f.write(f"ATOM      6  DUF BOX     1    {max_x}{max_y}{min_z}\n")
            f.write(f"ATOM      7  DUG BOX     1    {max_x}{max_y}{max_z}\n")
            f.write(f"ATOM      8  DUH BOX     1    {min_x}{max_y}{max_z}\n")
            f.write("CONECT    1    2    4    5\n")
            f.write("CONECT    2    1    3    6\n")
            f.write("CONECT    3    2    4    7\n")
            f.write("CONECT    4    1    3    8\n")
            f.write("CONECT    5    1    6    8\n")
            f.write("CONECT    6    2    5    7\n")
            f.write("CONECT    7    3    6    8\n")
            f.write("CONECT    8    4    5    7\n")
        
        return None


# Functions
###############################################################################
## Private ##

## Public ##
def split_molecules(molecule: str, output_dir: str = "", prefix: str = "ligand") -> List[str]:
    ''' Given a molecule file, checks if it has more than one ligand, if positive, splits the file into multiple single molecule files. Uses openbabel python library. TODO: Make this function work better with the new database structure.

    Parameters
    ----------
    molecule : str
        The path to the molecule file.
    output_dir : str, optional
        The path to the output directory, by default ""
    prefix : str, optional
        The prefix for the output files, by default "ligand"

    Returns
    -------
    List[str]
        A list of paths to the new files.
    '''

    # Initialise an empty list to hold all files paths
    ligand_files = []
    # Grab the extension and path
    extension = ocvalidation.validate_obabel_extension(molecule)

    # If the output_dir is not defined
    if not output_dir:
        # Set it as the same dir as the ligand
        output_dir = f"{os.path.split(os.path.abspath(molecule))[0]}/compounds"

    # Check if the extension is valid
    if type(extension) != str:
        ocprint.print_error(f"Problems while reading the ligand file '{molecule}'.")
    else:
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
            out_path = f"{output_dir}/{prefix}_{molNum}.mol2"
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


def multiple_molecules_sdf(molecule: Union[str, rdkit.Chem.rdchem.Mol]) -> List[Ligand]:
    ''' Parse a .sdf or .mol2 file with multiple molecules returning a list of ligands.

    Parameters
    ----------
    molecule : str | rdkit.Chem.rdchem.Mol
        Path to a molecule file or an RDKit molecule object

    Returns
    -------
    List[Ligand]
        A list of ligands.
    '''

    # List to hold multiple Ligand objects
    ligands = []
    # Check if the path is a string (it is assumed that the provided path is already a sdf)
    if isinstance(molecule, str):
        # Check if file exists
        if os.path.isfile(molecule):
            # Check if the extension of the file is .sdf
            extension = os.path.splitext(molecule)[1]
            if extension in [".sdf", ".mol2"]:
                # Split the mol file
                molsPaths = split_molecules(molecule)
                # For each molecule
                for molPath in molsPaths:
                    # Get molecule name
                    name = os.path.splitext(os.path.basename(molecule))[0]
                    # Append to the list
                    ligands.append(Ligand(molPath, name=name))
            else:
                # This case the return code is suppressed because it is needed to return None in case of failure
                _ = ocerror.Error.wrong_type(message=f"The molecule file MUST be the .sdf format!", level=ocerror.ReportLevel.WARNING) # type: ignore
        elif isinstance(molecule, Chem.rdchem.Mol):
            name = molecule.GetProp("_Name") if molecule.HasProp("_Name") else "ligand"
            ligands.append(Ligand(molecule, name=name))
        else:
            # File does not exist
            _ = ocerror.Error.wrong_type(message=f"The molecule MUST be either a file path or rdkit.Chem.rdchem.Mol!", level=ocerror.ReportLevel.WARNING) # type: ignore
    else:
        # This case the return code is suppressed because it is needed to return None in case of failure
        _ = ocerror.Error.wrong_type(message=f"The molecule file path MUST be a string!", level=ocerror.ReportLevel.WARNING) # type: ignore

    return ligands


def load_mol(molecule: Union[str, Chem.rdchem.Mol], sanitize: bool = True) -> Tuple[str, Optional[Chem.rdchem.Mol]]:
    ''' Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.

    Parameters
    ----------
    molecule : str/rdkit.Chem.rdchem.Mol
        The molecule path or the Mol object.

    Returns
    -------
    Tuple[str, Chem.rdchem.Mol | None]
        The molecule object and the path to the molecule.
    '''

    # Check if the type of the variable molecule is a string or a rdkit.Chem.rdchem.Mol
    if isinstance(molecule, Chem.rdchem.Mol):
        # Since is already a molecule, assign it to the class
        return "", molecule

    if isinstance(molecule, str):
        # Check if file exists
        if not os.path.isfile(molecule):
            # File does not exist
            _ = ocerror.Error.file_not_exist(message=f"The file '{molecule}' does not exist!", level = ocerror.ReportLevel.WARNING) # type: ignore
            return "", None

        # Determine the extension and use appropriate RDKit functions
        extension = os.path.splitext(molecule)[1].lower()
        supported_extensions = ['.pdb', '.sdf', '.mol', '.mol2', '.smi', '.smiles']

        # The file extension is not supported, print data
        if extension not in supported_extensions:
            # This case the return code is suppressed because it is needed to return None in case of failure
            _ = ocerror.Error.unsupported_extension(message=f"The ligand {molecule} has a unsupported extension.\nCurrently the supported extensions are {', '.join(supported_extensions)}.", level = ocerror.ReportLevel.WARNING) # type: ignore
            return "", None

        # Function map for file extension to RDKit loading function
        load_functions = {
            ".pdb": Chem.rdmolfiles.MolFromPDBFile, # type: ignore
            ".sdf": Chem.rdmolfiles.SDMolSupplier, # type: ignore
            ".mol": Chem.rdmolfiles.MolFromMolFile, # type: ignore
            ".mol2": Chem.rdmolfiles.MolFromMol2File, # type: ignore
        }

        # Load the molecule with appropriate loader
        if extension in load_functions:
            # Get the function to load the molecule
            load_func = load_functions[extension]
            # Load the molecule
            mol = load_func(molecule, sanitize = sanitize)
        elif extension in ['.smi', '.smiles']:
            # Read the smiles file into a string
            with open(molecule, 'r') as file:
                smiles = file.read().strip()

            # Load the molecule
            mol = Chem.MolFromSmiles(smiles, sanitize = sanitize) # type: ignore

            # If the molecule was loaded
            if mol:
                # Find its name (without extension)
                name = os.path.splitext(os.path.basename(molecule))[0]

                # Set its name
                mol.SetProp("_Name", name)

                # Remove the salts
                remover = SaltRemover()
                mol = remover.StripMol(mol)

                # Add the hydrogens
                mol = Chem.AddHs(mol) # type: ignore

                # Embed the molecule
                AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # type: ignore

                # Ensure that the ring information is initialized
                _ = mol.GetRingInfo()
    
                # If sanitize is True, sanitize the molecule
                if sanitize:
                    # Sanitize the molecule
                    Chem.SanitizeMol(mol)

                # Optimize the molecule
                AllChem.UFFOptimizeMolecule(mol) # type: ignore
            else:
                # The molecule could not be loaded
                _ = ocerror.Error.parse_molecule(f"The molecule '{molecule}' could not be parsed from smiles.", level = ocerror.ReportLevel.WARNING) # type: ignore
                return "", None

        # Handling of multiple molecules in a .sdf file
        if isinstance(mol, Chem.rdmolfiles.SDMolSupplier): # type: ignore
            mol_list = list(mol) # type: ignore
            if len(mol_list) > 1:
                print("Warning: The .sdf file contains more than one molecule. Only the first will be processed.")
            mol = mol_list[0] if mol_list else None

        # Check if the molecule was loaded
        if mol is None: # type: ignore
            _ = ocerror.Error.parse_molecule(f"The molecule '{molecule}' could not be parsed.", level = ocerror.ReportLevel.WARNING) # type: ignore
            return "", None

        # If sanitize is off
        if not sanitize:
            # Turn off the property cache
            mol.UpdatePropertyCache(strict=False) # type: ignore
            # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
            Chem.SanitizeMol( # type: ignore
                mol, # type: ignore
                Chem.SANITIZE_FINDRADICALS | # type: ignore
                Chem.SANITIZE_KEKULIZE | # type: ignore
                Chem.SANITIZE_SETAROMATICITY | # type: ignore
                Chem.SANITIZE_SETCONJUGATION | # type: ignore
                Chem.SANITIZE_SETHYBRIDIZATION | # type: ignore
                Chem.SANITIZE_SYMMRINGS, # type: ignore
                catchErrors=True
            )
        
        # If the molecule is not in mol2 format
        if extension != ".mol2":
            # Use temporary directory instead of input directory to avoid polluting input folder
            try:
                from OCDocker.Config import get_config
                config = get_config()
                tmp_dir = config.tmp_dir if config.tmp_dir else None
            except (ImportError, AttributeError):
                tmp_dir = None
            
            # Fallback to system temp directory if config tmp_dir is not available
            if not tmp_dir or not os.path.isdir(tmp_dir):
                import tempfile
                tmp_dir = tempfile.gettempdir()
            
            # Ensure tmp_dir exists
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except (OSError, PermissionError):
                # If we can't create tmp_dir, fall back to input directory (original behavior)
                tmp_dir = os.path.dirname(molecule)
            
            # Create a unique filename to avoid conflicts in multiprocessing scenarios
            import hashlib
            molecule_hash = hashlib.md5(os.path.abspath(molecule).encode()).hexdigest()[:8]
            ligand_basename = os.path.splitext(os.path.basename(molecule))[0]
            outputMoleculePath = os.path.join(tmp_dir, f"{ligand_basename}_{molecule_hash}.mol2")
            
            # Check if it is not a smiles file
            if extension not in [".smi", ".smiles"]:
                # Convert the molecule
                occonversion.convert_mols(molecule, outputMoleculePath)
            else:
                # Convert the molecule
                occonversion.convert_mols_from_string("", outputMoleculePath, mol = mol) # type: ignore
            
            # Return the molecule
            return outputMoleculePath, mol # type: ignore

        # Return the molecule
        return molecule, mol # type: ignore

    # The variable is not in a supported data format
    _ = ocerror.Error.unsupported_extension(message = f"Unsupported molecule data. Please support either a molecule path (string) or a rdkit.Chem.rdchem.Mol object.", level = ocerror.ReportLevel.WARNING) # type: ignore

    return "", None


def read_descriptors_from_json(path: str, return_data: bool = False) -> Optional[Union[Dict[str, Union[str, float, int]], Tuple[Union[str, float, int]]]]:
    ''' Read the descriptors from a JSON file.

    Parameters
    ----------
    path : str
        The path to the JSON file.
    return_data : bool, optional
        If True, returns a dictionary with the descriptors. If False, returns a dictionary with the descriptors, by default False.
    
    Returns
    -------
    Dict[str, str | float | int]] | Tuple[str | float | int]] | None
        The descriptors or None if an error occurs.
    '''

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Construct the list of expected keys
        keys = ["Name"] + Ligand.allDescriptors

        # Check for missing keys
        missing_keys = [key for key in keys if key not in data]
        if missing_keys:
            # User-facing error: missing required data in JSON file
            ocerror.Error.data_not_found(f"Missing keys in JSON file '{path}': {', '.join(missing_keys)}") # type: ignore
            return None

        if return_data:
            # Rename 'Name' to 'Ligand' and remove 'Path' if they exist
            if 'Name' in data:
                data['Ligand'] = data.pop('Name')
            data.pop('Path', None)  # Remove 'Path' if it exists

            return data

        # If not returning as data, return tuple of values
        return tuple(data[key] for key in keys) # type: ignore

    except KeyError as e:
        # This should rarely happen now since we check for missing keys above
        # But keep it as a fallback for other KeyError cases
        ocerror.Error.value_error(f"Error: {e}") # type: ignore
    except Exception as e:
        ocerror.Error.read_file(f"Could not read the file '{path}'. Error: {e}") # type: ignore

    return None


def get_smiles(molecule: rdkit.Chem.rdchem.Mol) -> Union[str, int]:
    ''' Return the smiles of the molecule

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to get the smiles from.

    Returns
    -------
    str | int
        The smiles of the molecule or the error code or the exit code of the command (based on the Error.py code table).
    '''

    if molecule:
        if type(molecule) == rdkit.Chem.rdchem.Mol: # type: ignore
            return Chem.MolToSmiles(molecule) # type: ignore
        return ocerror.Error.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'") # type: ignore

    return ocerror.Error.not_set(f"The variable is not set.") # type: ignore


def get_centroid(molecule: Union[str, rdkit.Chem.rdchem.Mol], sanitize: bool = True) -> rdkit.Geometry.rdGeometry.Point3D:
    ''' Get the centroid of the molecule.

    Parameters
    ----------
    molecule : str | rdkit.Chem.rdchem.Mol
        The molecule to get the centroid or its path.
    sanitize : bool, optional
        If the molecule should be sanitized, by default True.

    Returns
    -------
    rdkit.Geometry.rdGeometry.Point3D
        The centroid of the molecule.
    
    Raises
    ------
    ValueError
        If the molecule cannot be loaded from the provided path.
    '''

    # Check if the molecule is a string (means that it is a path)
    if type(molecule) == str:
        # Store the path before loading
        molecule_path = molecule
        # Load it
        _, molecule = load_mol(molecule, sanitize = sanitize)
        # Check if molecule was loaded successfully
        if molecule is None:
            # User-facing error: molecule loading failure
            ocerror.Error.parse_molecule(f"Could not load molecule from path: {molecule_path}") # type: ignore
            raise ValueError(f"Could not load molecule from path: {molecule_path}")  # Still raise to maintain API contract

    # Get the molecule conformer
    conf = molecule.GetConformer()  # type: ignore

    # Compute the centroid of the molecule and return it
    return ComputeCentroid(conf)


# Factory functions and dynamic methods
##############################################################################

def __descriptor_function_factory(descriptor_name: str) -> Callable[[rdkit.Chem.rdchem.Mol], Optional[float]]:
    '''Factory function to create a function that computes a descriptor by its name.

    This function creates a descriptor computation function dynamically based on
    the descriptor name. It handles both 2D and 3D descriptors from RDKit.

    Parameters
    ----------
    descriptor_name : str
        The name of the descriptor to compute (e.g., "BalabanJ", "BertzCT",
        "RadiusOfGyration"). Must match a descriptor function name in either
        rdkit.Chem.Descriptors or rdkit.Chem.Descriptors3D.

    Returns
    -------
    Callable[[rdkit.Chem.rdchem.Mol], Optional[float]]
        A function that takes a molecule and returns the descriptor value,
        or None if computation fails.
    '''

    if descriptor_name in ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]:
        descriptor_func = getattr(Descriptors3D, descriptor_name)
    else:
        descriptor_func = getattr(Descriptors, descriptor_name)

    # TODO: Check how to avoid this function to spam print the same error multiple times
    def __compute_descriptor(molecule: rdkit.Chem.rdchem.Mol) -> Optional[float]:
        '''Compute a descriptor value for a molecule.

        This is a nested function created by the factory to compute a specific
        molecular descriptor using RDKit descriptor functions.

        Parameters
        ----------
        molecule : rdkit.Chem.rdchem.Mol
            The molecule object to compute the descriptor for.

        Returns
        -------
        Optional[float]
            The descriptor value, or None if computation fails or molecule is invalid.
        '''
        
        if molecule:
            if isinstance(molecule, rdkit.Chem.rdchem.Mol): # type: ignore
                try:
                    return descriptor_func(molecule)
                except Exception as e:
                    _ = ocerror.Error.unknown(f"Error while creating the function in factory: {str(e)}") # type: ignore
            else:
                _ = ocerror.Error.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'") # type: ignore
        else:
            pass
            # _ = ocerror.Error.not_set("The molecule is not set.") # type: ignore
        return None

    return __compute_descriptor


def __descriptor_function_factory_class(descriptor_name: str) -> Callable[[rdkit.Chem.rdchem.Mol], Optional[float]]:
    '''Factory function to create a class method that computes a descriptor.

    This function creates a descriptor computation method dynamically for the
    Ligand class. The created method uses self.molecule as the input molecule.

    Parameters
    ----------
    descriptor_name : str
        The name of the descriptor to compute (e.g., "BalabanJ", "BertzCT",
        "RadiusOfGyration"). Must match a descriptor function name in either
        rdkit.Chem.Descriptors or rdkit.Chem.Descriptors3D.

    Returns
    -------
    Callable[[rdkit.Chem.rdchem.Mol], Optional[float]]
        A method that can be bound to the Ligand class. When called, it computes
        the descriptor for self.molecule and returns the value, or None if
        computation fails.
    '''

    # Check if the descriptor is 3D (in this case use the Descriptors3D module)
    if descriptor_name in ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]:
        descriptor_func = getattr(Descriptors3D, descriptor_name)
    else:
        descriptor_func = getattr(Descriptors, descriptor_name)

    # TODO: Check how to avoid this function to spam print the same error multiple times

    # Create the nested function as a method of the Ligand class
    def __compute_descriptor_class(self) -> Optional[float]:
        '''Compute the descriptor for the Ligand object.

        This is a nested function created by the factory to compute a specific
        molecular descriptor for the Ligand instance using self.molecule.

        Returns
        -------
        Optional[float]
            The descriptor value for self.molecule, or None if computation fails
            or molecule is invalid.
        '''

        molecule = self.molecule  # Use self.molecule from the Ligand instance

        # Check if the molecule is set
        if molecule:
            # Check if the molecule is a rdkit.Chem.rdchem.Mol
            if isinstance(molecule, rdkit.Chem.rdchem.Mol): # type: ignore
                try:
                    # Return the function
                    return descriptor_func(molecule)
                except Exception as e:
                    _ = ocerror.Error.unknown(f"Error while creating the function in factory: {str(e)}") # type: ignore
            else:
                _ = ocerror.Error.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'") # type: ignore
        else:
            pass
            #_ = ocerror.Error.not_set("The molecule is not set.") # type: ignore
        return None

    return __compute_descriptor_class


# Using the factory function to create the descriptor functions add them to the global namespace and to the Ligand class
for desc in Ligand.allDescriptors:
    # Set the description
    func_description = f"Compute the {desc} descriptor for the Ligand object.\n\n    Parameters\n    ----------\n    molecule : rdkit.Chem.rdchem.Mol\n        The molecule to be evaluated.\n\n    Returns\n    -------\n    float | None\n        The {desc} value or None if parsing the descriptor fails."
    # Create the function using the factory
    func = __descriptor_function_factory(desc)
    # Set the docstring
    func.__doc__ = func_description
    # Add the function to the global namespace with the desired name
    globals()[f'find{desc}'] = func

    # Assemble the function name
    func_name = f'find{desc}'
    # Create the function using the factory
    class_func = __descriptor_function_factory_class(desc)
    # Set the docstring
    class_func.__doc__ = func_description
    # Add the function to the Ligand class
    setattr(Ligand, func_name, class_func)
