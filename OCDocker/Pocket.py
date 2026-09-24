#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to process all content related to
the binding pocket of a receptor.

A pocket is defined by a reference ligand: every standard receptor residue with
at least one heavy atom within a distance cutoff of any reference-ligand heavy
atom belongs to the pocket. One receptor may hold several pockets, each built
from its own reference ligand.

Usage:

import OCDocker.Pocket as ocpocket
'''

# Imports
###############################################################################
from __future__ import annotations

import Bio
import json
import os

import numpy as np

from Bio.PDB import SASA
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.SeqUtils import seq1
from Bio.SeqUtils.IsoelectricPoint import negative_pKs, positive_pKs
from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################
class Pocket:
    """Represents a receptor binding pocket defined by a reference ligand.

    The pocket is the set of standard receptor residues with at least one
    heavy atom within ``cutoff`` angstroms of any heavy atom of the reference
    ligand. Descriptors are computed on those residues only: residue counts,
    number of chains contributing, solvent accessible surface area in the
    context of the full receptor, GRAVY, aromaticity, side-chain net charge and
    side-chain hydrogen-bond donor and acceptor atoms lining the pocket.

    Parameters
    ----------
    structure : str | Bio.PDB.Structure.Structure
        Path to a PDB/mmCIF file or a BioPython Structure object of the receptor.
    reference_ligand : str | rdkit.Chem.rdchem.Mol
        Path to the reference ligand (pdb/sdf/mol/mol2) or an RDKit Mol with 3D
        coordinates. Not required when ``from_json_descriptors`` is given.
    name : str
        Name identifier for the pocket.
    cutoff : float, optional
        Heavy-atom distance cutoff in angstroms defining the pocket, by default 8.0.
    ph : float, optional
        pH used to compute the side-chain net charge, by default 7.4.
    gravy_scale : str, optional
        GRAVY scale to use, by default "KyteDoolitle".
    from_json_descriptors : str, optional
        Path to JSON file containing pre-computed descriptors, by default "".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    clean : bool, optional
        Whether the receptor pdb file will be cleaned, by default False.
    canonicalize_pdb : bool | str, optional
        Whether to canonicalize CHARMM-style PDB atom/residue names. Use
        True, False, or ``"auto"``, by default ``"auto"``.

    Attributes
    ----------
    name : str
        Name of the pocket.
    path : str
        Path to the receptor structure file.
    reference_ligand_path : str
        Path to the reference ligand used to define the pocket.
    structure : Bio.PDB.Structure.Structure
        BioPython structure object of the receptor.
    residues : str
        One-letter sequence of the pocket residues, in structure order.
    residue_ids : list[str]
        Pocket residues as ``chain:number:insertion:resname`` identifiers.
    cutoff : float
        Heavy-atom distance cutoff in angstroms.
    SASA : float
        Solvent accessible surface area of the pocket residues, computed in
        the context of the full receptor.
    GRAVY : float
        Grand average of hydropathy of the pocket residues.
    Aromaticity : float
        Aromaticity index of the pocket residues.
    NetCharge : float
        Side-chain net charge of the pocket residues at ``ph``.
    countA, countR, countN, ..., countV : int
        Count of each amino acid type among the pocket residues. Unlike
        :class:`OCDocker.Receptor.Receptor`, where these count only surface
        residues, here every pocket residue is counted.
    TotalAALength : int
        Number of pocket residues.
    countChain : int
        Number of chains contributing residues to the pocket.
    countHBondDonors : int
        Side-chain hydrogen-bond donor atoms within ``cutoff`` of the reference ligand.
    countHBondAcceptors : int
        Side-chain hydrogen-bond acceptor atoms within ``cutoff`` of the reference ligand.
    """

    # Declare the amino acid count descriptors (relevant for pockets)
    descriptors_names = {
        "count": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    }

    # Declare single descriptors for pocket properties
    single_descriptors = [
        "TotalAALength", "countChain", "SASA", "GRAVY", "Aromaticity", "NetCharge",
        "countHBondDonors", "countHBondAcceptors"
    ]

    # Generate all descriptors dynamically
    allDescriptors = [f"count{i}" for i in descriptors_names["count"]] + single_descriptors

    def __init__(self, structure: Union[str, Bio.PDB.Structure.Structure], reference_ligand: Union[str, Chem.rdchem.Mol], name: str, cutoff: float = 8.0, ph: float = 7.4, gravy_scale: str = "KyteDoolitle", from_json_descriptors: str = "", overwrite: bool = False, clean: bool = False, canonicalize_pdb: Union[bool, str] = "auto") -> None:
        '''Constructor of the class Pocket.

        Parameters
        ----------
        structure : str | Bio.PDB.Structure.Structure
            Path to the receptor structure file OR Bio.PDB.Structure.Structure object.
        reference_ligand : str | rdkit.Chem.rdchem.Mol
            Path to the reference ligand OR RDKit Mol with 3D coordinates.
        name : str
            Name of the pocket.
        cutoff : float, optional
            Heavy-atom distance cutoff in angstroms defining the pocket, by default 8.0.
        ph : float, optional
            pH used to compute the side-chain net charge, by default 7.4.
        gravy_scale : str, optional
            Scale to be used to compute the GRAVY descriptor, by default "KyteDoolitle".
        from_json_descriptors : str, optional
            Path to the json file containing the descriptors, by default "".
        overwrite : bool, optional
            Flag to denote if files will be overwritten, by default False.
        clean : bool, optional
            Flag to denote if the receptor pdb file will be cleaned, by default False.
        canonicalize_pdb : bool | str, optional
            Whether to canonicalize CHARMM-style PDB names. Use True, False, or
            ``"auto"``, by default ``"auto"``.
        '''

        # Name must come first
        self.name: str = ""
        self.path: str = ""
        self.reference_ligand_path: str = os.fspath(reference_ligand) if isinstance(reference_ligand, (str, os.PathLike)) else ""
        self.structure: Optional[Bio.PDB.Structure.Structure] = None
        self.residues: str = ""
        self.residue_ids: List[str] = []
        self.cutoff: float = float(cutoff)
        # Descriptor-related fields
        self.__ph: float = float(ph)
        self.__gravy_scale: str = gravy_scale
        self.sasa: Optional[float] = None
        self.GRAVY: Optional[float] = None
        self.aromaticity: Optional[float] = None
        self.netCharge: Optional[float] = None
        self.totalAALength: Optional[int] = None
        self.countChain: Optional[int] = None
        self.countHBondDonors: Optional[int] = None
        self.countHBondAcceptors: Optional[int] = None
        self.__countAA: Optional[Dict[str, int]] = None
        # Individual AA counts
        self.countA: Optional[int] = None
        self.countR: Optional[int] = None
        self.countN: Optional[int] = None
        self.countD: Optional[int] = None
        self.countC: Optional[int] = None
        self.countQ: Optional[int] = None
        self.countE: Optional[int] = None
        self.countG: Optional[int] = None
        self.countH: Optional[int] = None
        self.countI: Optional[int] = None
        self.countL: Optional[int] = None
        self.countK: Optional[int] = None
        self.countM: Optional[int] = None
        self.countF: Optional[int] = None
        self.countP: Optional[int] = None
        self.countS: Optional[int] = None
        self.countT: Optional[int] = None
        self.countW: Optional[int] = None
        self.countY: Optional[int] = None
        self.countV: Optional[int] = None

        # Load the receptor without computing the whole-receptor SASA (the pocket SASA is computed per residue)
        self.path, self.structure = ocr.load_mol(structure, name=name, compute_sasa=False, overwrite=overwrite, clean=clean, canonicalize_pdb=canonicalize_pdb)

        # Ensure structure is valid before continuing
        if self.structure is None:
            ocprint.print_error(f"Could not load receptor structure for pocket: '{structure}'.")
            return None

        # If user pass a json
        if from_json_descriptors:
            # Read the descriptors from it
            data = read_descriptors_from_json(from_json_descriptors, returnData=True)

            # If data is None, a problem occurred while reading the json file
            if not data or not isinstance(data, dict):
                ocprint.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None

            self.name = str(data["Name"])
            self.cutoff = float(cast(float, data["Cutoff"]))
            self.reference_ligand_path = str(data.get("ReferenceLigand", self.reference_ligand_path))
            self.residue_ids = [rid for rid in str(data["Residues"]).split(";") if rid]
            self.residues = "".join(seq1(rid.split(":")[-1]) for rid in self.residue_ids)
            self.sasa = _to_float(data["SASA"])
            self.GRAVY = _to_float(data["GRAVY"])
            self.aromaticity = _to_float(data["Aromaticity"])
            self.netCharge = _to_float(data["NetCharge"])
            self.totalAALength = _to_int(data["TotalAALength"]) or 0
            self.countChain = _to_int(data["countChain"]) or 0
            self.countHBondDonors = _to_int(data["countHBondDonors"]) or 0
            self.countHBondAcceptors = _to_int(data["countHBondAcceptors"]) or 0
            self.__countAA = {aa: _to_int(data[f"count{aa}"]) or 0 for aa in self.descriptors_names["count"]}
        else:
            # Check if the name is empty
            if not name:
                ocprint.print_error("The Pocket name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            # Get the reference ligand heavy-atom coordinates
            ligand_coordinates = get_ligand_heavy_atom_coordinates(reference_ligand)
            if ligand_coordinates is None:
                ocprint.print_error(f"Could not read the reference ligand coordinates: '{reference_ligand}'.")
                return None

            # Select the pocket residues
            pocket_residues = select_pocket_residues(self.structure, ligand_coordinates, self.cutoff)
            if not pocket_residues:
                ocprint.print_error(f"No receptor residue within {self.cutoff} A of the reference ligand '{reference_ligand}'.")
                return None

            self.residue_ids = [_residue_id(residue) for residue in pocket_residues]
            self.residues = "".join(seq1(residue.get_resname()) for residue in pocket_residues)
            self.totalAALength = len(pocket_residues)
            self.countChain = len({residue.get_parent().id for residue in pocket_residues})
            self.__countAA = count_pocket_AAs(pocket_residues)
            self.sasa = compute_pocket_sasa(self.structure, pocket_residues)
            self.GRAVY = ocr.compute_gravy(self.residues, scale=self.__gravy_scale)
            self.aromaticity = ocr.compute_aromaticity(self.residues)
            self.netCharge = compute_net_charge(self.residues, ph=self.__ph)
            self.countHBondDonors, self.countHBondAcceptors = count_hbond_atoms(pocket_residues, ligand_coordinates, self.cutoff)

        # Set the individual AA counts
        for aa in self.descriptors_names["count"]:
            setattr(self, f"count{aa}", self.__countAA.get(aa, 0))

    ## Private ##
    def __safe_to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        '''Return all the properties (except the molecule object) for the Pocket object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Union[str, float, int, None]]
            A dictionary with all the properties (except the molecule object) for the Pocket object.
        '''

        # Create new dict
        properties: Dict[str, Union[str, float, int, None]] = dict()
        # Set Name, Path, reference ligand, cutoff and residues
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["ReferenceLigand"] = self.reference_ligand_path if self.reference_ligand_path else "-"
        properties["Cutoff"] = self.cutoff
        properties["Residues"] = ";".join(self.residue_ids)

        return {**properties, **self.get_descriptors()}

    ## Public ##

    def get_descriptors(self) -> Dict[str, Union[float, int, None]]:
        '''Return the descriptors for the Pocket object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, float | int | None]
            The descriptors for the Pocket object.
        '''

        descriptors: Dict[str, Union[float, int, None]] = {
            "TotalAALength": self.totalAALength if self.totalAALength is not None else 0,
            "countChain": self.countChain if self.countChain is not None else 0,
            "SASA": self.sasa if self.sasa is not None else None,
            "GRAVY": self.GRAVY if self.GRAVY is not None else None,
            "Aromaticity": self.aromaticity if self.aromaticity is not None else None,
            "NetCharge": self.netCharge if self.netCharge is not None else None,
            "countHBondDonors": self.countHBondDonors if self.countHBondDonors is not None else 0,
            "countHBondAcceptors": self.countHBondAcceptors if self.countHBondAcceptors is not None else 0,
        }

        # Add the AA counts
        for aa in self.descriptors_names["count"]:
            value = getattr(self, f"count{aa}", None)
            descriptors[f"count{aa}"] = value if value is not None else 0

        return descriptors

    def is_valid(self) -> bool:
        '''Check if a Pocket object is valid.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the Pocket object is valid, False otherwise.
        '''

        #region if any attribute is None
        if not self.name or self.structure is None or not self.residue_ids or self.sasa is None or self.GRAVY is None or self.aromaticity is None or self.netCharge is None or self.__countAA is None or self.totalAALength is None or self.countChain is None or self.countHBondDonors is None or self.countHBondAcceptors is None:
            return False
        #endregion
        return True

    def print_attributes(self) -> None:
        '''Print all attributes of the pocket to stdout.

        Displays the pocket's name, receptor and reference ligand paths, cutoff
        and all computed descriptors in a formatted table.
        '''

        attributes: Dict[str, Any] = {
            "Name": self.name,
            "Structure path": self.path,
            "Reference ligand": self.reference_ligand_path,
            "Cutoff (A)": self.cutoff,
            "Structure": self.structure,
            "Pocket residues": self.residues,
            "# of residues": self.totalAALength,
            "# of chains": self.countChain,
            "SASA": self.sasa,
            "GRAVY": self.GRAVY,
            "Aromaticity": self.aromaticity,
            "Net charge": self.netCharge,
            "# of H-bond donor atoms": self.countHBondDonors,
            "# of H-bond acceptor atoms": self.countHBondAcceptors
        }

        for aa in self.descriptors_names["count"]:
            attributes[f"# of {aa}"] = getattr(self, f"count{aa}", 0)

        for key, value in attributes.items():
            print(f"{key}: {value if value else '-'}")

    def to_dict(self) -> Dict[str, Union[str, float, int, None]]:
        '''Return all the properties for the Pocket object.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, float | int]
            The properties for the Pocket object.
        '''

        # Create new dict
        properties: Dict[str, Union[str, float, int, None]] = dict()
        # Set Name, Path, reference ligand and molecule
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["ReferenceLigand"] = self.reference_ligand_path if self.reference_ligand_path else "-"
        properties["Cutoff"] = self.cutoff
        properties["Residues"] = ";".join(self.residue_ids)
        properties["Structure"] = str(self.structure) if self.structure is not None else "-"

        return {**properties, **self.get_descriptors()}

    def to_json(self, overwrite: bool = False) -> int:
        '''Stores the descriptors as json to avoid the necessity of evaluate them many times.

        Parameters
        ----------
        overwrite: bool, optional
            If True, the json file will be overwritten if it already exists. Default is False.

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
        '''

        try:
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            if not overwrite and os.path.isfile(outputJson):
                return ocerror.Error.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", ocerror.ReportLevel.WARNING)
            if os.path.isfile(outputJson):
                _ = ocerror.Error.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!")
            try:
                with open(outputJson, 'w') as outfile:
                    json.dump(self.__safe_to_dict(), outfile)
                return ocerror.Error.ok()
            except Exception as e:
                return ocerror.Error.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.")
        except Exception as e:
            return ocerror.Error.unknown(f"Unknown error while converting the pocket {self.name} to json.\nError: {e}", ocerror.ReportLevel.ERROR)


# Functions
###############################################################################
## Private ##

# Side-chain heavy atoms that can donate a hydrogen bond, per residue
_HBOND_DONORS: Dict[str, Tuple[str, ...]] = {
    "ARG": ("NE", "NH1", "NH2"),
    "ASN": ("ND2",),
    "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"),
    "LYS": ("NZ",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1",),
    "TYR": ("OH",),
}

# Side-chain heavy atoms that can accept a hydrogen bond, per residue
_HBOND_ACCEPTORS: Dict[str, Tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "ASN": ("OD1",),
    "GLN": ("OE1",),
    "HIS": ("ND1", "NE2"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
}

def _residue_id(residue: Bio.PDB.Residue.Residue) -> str:
    '''Build a stable identifier for a residue.

    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        The residue to be identified.

    Returns
    -------
    str
        The identifier as ``chain:number:insertion:resname``.
    '''

    _, number, insertion = residue.id
    return f"{residue.get_parent().id}:{number}:{insertion.strip()}:{residue.get_resname()}"

def _to_float(val: object) -> Optional[float]:
    '''Convert a json value to float.

    Parameters
    ----------
    val : object
        The value to be converted.

    Returns
    -------
    float | None
        The converted value or None if it cannot be converted.
    '''

    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None

def _to_int(val: object) -> Optional[int]:
    '''Convert a json value to int.

    Parameters
    ----------
    val : object
        The value to be converted.

    Returns
    -------
    int | None
        The converted value or None if it cannot be converted.
    '''

    if isinstance(val, (bool, int)):
        return int(val)
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return None
    return None

## Public ##

def compute_net_charge(residues: str, ph: float = 7.4) -> float:
    '''Compute the side-chain net charge of a set of residues at a given pH.

    Uses the Henderson-Hasselbalch equation with the side-chain pKa values
    from Biopython's IsoelectricPoint module. Chain termini are not included
    because pocket residues are not a contiguous chain.

    Parameters
    ----------
    residues : str
        The one-letter residues of the pocket.
    ph : float, optional
        The pH, by default 7.4.

    Returns
    -------
    float
        The side-chain net charge.
    '''

    ocprint.printv(f"Computing the side-chain net charge at pH {ph} for residues '{residues}'.")
    charge = 0.0
    for aa in residues.upper():
        # Positively charged side chains
        if aa in positive_pKs and aa != "Nterm":
            charge += 1.0 / (1.0 + 10 ** (ph - positive_pKs[aa]))
        # Negatively charged side chains
        elif aa in negative_pKs and aa != "Cterm":
            charge -= 1.0 / (1.0 + 10 ** (negative_pKs[aa] - ph))
    return float(charge)

def compute_pocket_sasa(structure: Bio.PDB.Structure.Structure, residues: List[Bio.PDB.Residue.Residue], n_points: int = 100) -> Optional[float]:
    '''Compute the solvent accessible surface area of the pocket residues.

    The SASA is computed per residue on the first model of the full receptor,
    so each pocket residue is shielded by its receptor neighbours, and then
    summed over the pocket residues.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The receptor structure.
    residues : List[Bio.PDB.Residue.Residue]
        The pocket residues (belonging to the first model of ``structure``).
    n_points : int, optional
        The number of points per atom used in the calculation, by default 100.

    Returns
    -------
    float | None
        The pocket SASA, or None if it cannot be computed.
    '''

    ocprint.printv(f"Computing the pocket SASA for protein '{structure.id}'.")
    try:
        sr = SASA.ShrakeRupley(n_points = n_points)
        sr.compute(structure[0], level="R")
        return float(sum(float(getattr(residue, "sasa", 0.0)) for residue in residues))
    except Exception as e:
        _ = ocerror.Error.unknown(f"Could not compute the pocket SASA for protein '{structure.id}'. Error: {e}", level = ocerror.ReportLevel.ERROR)
    return None

def count_hbond_atoms(residues: List[Bio.PDB.Residue.Residue], ligand_coordinates: np.ndarray, cutoff: float = 8.0) -> Tuple[int, int]:
    '''Count side-chain hydrogen-bond donor and acceptor atoms lining the pocket.

    Only side-chain heavy atoms within ``cutoff`` of any reference-ligand heavy
    atom are counted. Atoms that can both donate and accept (e.g. SER OG) are
    counted in both groups.

    Parameters
    ----------
    residues : List[Bio.PDB.Residue.Residue]
        The pocket residues.
    ligand_coordinates : np.ndarray
        The reference ligand heavy-atom coordinates, shape ``(n, 3)``.
    cutoff : float, optional
        The distance cutoff in angstroms, by default 8.0.

    Returns
    -------
    Tuple[int, int]
        The number of donor and acceptor atoms.
    '''

    donors = 0
    acceptors = 0
    for residue in residues:
        resname = residue.get_resname()
        for atom in residue:
            atom_name = atom.get_id()
            is_donor = atom_name in _HBOND_DONORS.get(resname, ())
            is_acceptor = atom_name in _HBOND_ACCEPTORS.get(resname, ())
            if not is_donor and not is_acceptor:
                continue
            # Keep only the atoms that line the pocket
            if float(np.min(np.linalg.norm(ligand_coordinates - atom.get_coord(), axis=1))) > cutoff:
                continue
            donors += int(is_donor)
            acceptors += int(is_acceptor)
    return donors, acceptors

def count_pocket_AAs(residues: List[Bio.PDB.Residue.Residue]) -> Dict[str, int]:
    '''Count how many of each of the 20 standard AAs are in the pocket.

    Parameters
    ----------
    residues : List[Bio.PDB.Residue.Residue]
        The pocket residues.

    Returns
    -------
    Dict[str, int]
        A dictionary with the count of each AA (non-standard residues are counted as 'X').
    '''

    aas = {aa: 0 for aa in Pocket.descriptors_names["count"]}
    aas["X"] = 0
    for residue in residues:
        aa_code = seq1(residue.get_resname()).upper()
        # Check if the amino acid is one of the 20 standard ones
        if aa_code in aas and aa_code != "X":
            aas[aa_code] += 1
        else:
            aas["X"] += 1
    return aas

def get_ligand_heavy_atom_coordinates(reference_ligand: Union[str, Chem.rdchem.Mol]) -> Optional[np.ndarray]:
    '''Get the heavy-atom coordinates of the reference ligand.

    The ligand is loaded without sanitization, since only its coordinates are
    needed and charge-state issues must not discard a valid pocket.

    Parameters
    ----------
    reference_ligand : str | rdkit.Chem.rdchem.Mol
        The reference ligand path or an RDKit Mol with 3D coordinates.

    Returns
    -------
    np.ndarray | None
        The heavy-atom coordinates, shape ``(n, 3)``, or None if they cannot be read.
    '''

    # Normalize to a concrete RDKit Mol
    if isinstance(reference_ligand, Chem.rdchem.Mol):
        mol = reference_ligand
    elif isinstance(reference_ligand, (str, os.PathLike)):
        _, loaded = ocl.load_mol(os.fspath(reference_ligand), sanitize = False, write_mol2 = False)
        if loaded is None:
            _ = ocerror.Error.parse_molecule(f"Could not load the reference ligand '{reference_ligand}'.")
            return None
        mol = loaded
    else:
        _ = ocerror.Error.wrong_type(f"Expected a path or an RDKit Mol, got {type(reference_ligand)}.")
        return None

    # A conformer with real coordinates is required
    if mol.GetNumConformers() == 0:
        _ = ocerror.Error.parse_molecule(f"The reference ligand '{reference_ligand}' has no conformer.")
        return None

    conformer = mol.GetConformer()
    coordinates = [
        list(conformer.GetAtomPosition(atom.GetIdx()))
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not coordinates:
        _ = ocerror.Error.parse_molecule(f"The reference ligand '{reference_ligand}' has no heavy atoms.")
        return None
    return np.asarray(coordinates, dtype=float)

def read_descriptors_from_json(path: str, returnData: bool = False) -> Optional[Union[Dict[str, Union[str, float, int]], Tuple[Union[str, float, int], ...]]]:
    '''Read the descriptors from a json file.

    Parameters
    ----------
    path : str
        The path to the json file.
    returnData : bool, optional
        If True, returns a dictionary with the descriptors. By default False.

    Returns
    -------
    Dict[str, str | float | int] | Tuple[str | float | int, ...] | None
        The descriptors dictionary (or a tuple ordered as ``Name``, ``Cutoff``,
        ``Residues`` followed by :attr:`Pocket.allDescriptors`) or None if any
        error occurs.
    '''

    # Try to read the file
    try:
        # Open the json file in read mode
        with open(path, 'r') as f:
            # Load the data
            data = json.load(f)

        # Expected keys to have in the json file
        keys = ["Name", "Cutoff", "Residues"] + Pocket.allDescriptors
        missing = [key for key in keys if key not in data]

        # If missing list is not empty
        if missing:
            # User-facing error: missing required data in JSON file
            ocerror.Error.data_not_found(f"Missing keys in JSON file '{path}': {', '.join(missing)}")
            raise KeyError(f"Missing keys in JSON file '{path}': {', '.join(missing)}")

        # If the returnData flag is on
        if returnData:
            # Return the entire dict
            return cast(Dict[str, Union[str, float, int]], data)

        return tuple(data[key] for key in keys)
    # Key error (when there is a missing key)
    except KeyError as missed:
        ocprint.print_error(f"The following keys were not found in the json file: {missed}")
    # General error (call it as problem to read file)
    except Exception as e:
        ocprint.print_error(f"Could not read the file '{path}'. Error: {e}")
    return None

def select_pocket_residues(structure: Bio.PDB.Structure.Structure, ligand_coordinates: np.ndarray, cutoff: float = 8.0) -> List[Bio.PDB.Residue.Residue]:
    '''Select the standard residues with any heavy atom within the cutoff of the ligand.

    Only the first model is considered, water and hetero residues are
    ignored, and hydrogens are excluded from the distance search.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The receptor structure.
    ligand_coordinates : np.ndarray
        The reference ligand heavy-atom coordinates, shape ``(n, 3)``.
    cutoff : float, optional
        The distance cutoff in angstroms, by default 8.0.

    Returns
    -------
    List[Bio.PDB.Residue.Residue]
        The pocket residues, in structure order.
    '''

    model = structure[0]
    # Standard residues only (same rule as count_AAs_and_chains)
    atoms = [
        atom
        for residue in model.get_residues()
        if residue.id[0] == ' '
        for atom in residue
        if atom.element != "H"
    ]
    if not atoms:
        return []

    search = NeighborSearch(atoms)
    selected = set()
    for coordinate in ligand_coordinates:
        for residue in search.search(np.asarray(coordinate, dtype="f"), cutoff, level="R"):
            selected.add(residue.get_full_id())

    return [residue for residue in model.get_residues() if residue.get_full_id() in selected]
