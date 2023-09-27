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
import vaex

import vaex.dataframe as vdf

from openbabel import openbabel
from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from threading import Lock
from typing import Dict, List, Tuple, Union

from OCDocker.Initialise import *

import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Validation as ocvalidation

# Set output levels for openbabel
ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(args.output_level)
if args.output_level == 0:
    RDLogger.DisableLog('rdApp.*')

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

# Classes
###############################################################################
class Ligand:
    """Load and compute ligand descriptors. You can provide either a molecule file
    (pdb/sdf/mol/mol2) or a rdkit.Chem.rdchem.Mol object. A name to indentify
    the molecule can be provided aswell."""

    def __init__(self, molecule: Union[str, rdkit.Chem.rdchem.Mol], name: str, sanitize: bool = True, from_json_descriptors: str = "") -> Union[int, None]: # type: ignore
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
        self.path, self.molecule = loadMol(molecule, sanitize) # type: ignore
        # Set the boxPath (removing the file from the path)
        self.boxPath = os.path.join(os.path.dirname(self.path), "boxes/box0.pdb")
        
        # Define everything as None, except for the name
        if not "_split_" in name:
            self.name = name
        else:
            return errors.invalid_molecule_name("The name of the ligand cannot contain the string '_split_'")

        #region AUTOCORR descriptors
        self.AUTOCORR2D_1 = None
        self.AUTOCORR2D_2 = None
        self.AUTOCORR2D_3 = None
        self.AUTOCORR2D_4 = None
        self.AUTOCORR2D_5 = None
        self.AUTOCORR2D_6 = None
        self.AUTOCORR2D_7 = None
        self.AUTOCORR2D_8 = None
        self.AUTOCORR2D_9 = None
        self.AUTOCORR2D_10 = None
        self.AUTOCORR2D_11 = None
        self.AUTOCORR2D_12 = None
        self.AUTOCORR2D_13 = None
        self.AUTOCORR2D_14 = None
        self.AUTOCORR2D_15 = None
        self.AUTOCORR2D_16 = None
        self.AUTOCORR2D_17 = None
        self.AUTOCORR2D_18 = None
        self.AUTOCORR2D_19 = None
        self.AUTOCORR2D_20 = None
        self.AUTOCORR2D_21 = None
        self.AUTOCORR2D_22 = None
        self.AUTOCORR2D_23 = None
        self.AUTOCORR2D_24 = None
        self.AUTOCORR2D_25 = None
        self.AUTOCORR2D_26 = None
        self.AUTOCORR2D_27 = None
        self.AUTOCORR2D_28 = None
        self.AUTOCORR2D_29 = None
        self.AUTOCORR2D_30 = None
        self.AUTOCORR2D_31 = None
        self.AUTOCORR2D_32 = None
        self.AUTOCORR2D_33 = None
        self.AUTOCORR2D_34 = None
        self.AUTOCORR2D_35 = None
        self.AUTOCORR2D_36 = None
        self.AUTOCORR2D_37 = None
        self.AUTOCORR2D_38 = None
        self.AUTOCORR2D_39 = None
        self.AUTOCORR2D_40 = None
        self.AUTOCORR2D_41 = None
        self.AUTOCORR2D_42 = None
        self.AUTOCORR2D_43 = None
        self.AUTOCORR2D_44 = None
        self.AUTOCORR2D_45 = None
        self.AUTOCORR2D_46 = None
        self.AUTOCORR2D_47 = None
        self.AUTOCORR2D_48 = None
        self.AUTOCORR2D_49 = None
        self.AUTOCORR2D_50 = None
        self.AUTOCORR2D_51 = None
        self.AUTOCORR2D_52 = None
        self.AUTOCORR2D_53 = None
        self.AUTOCORR2D_54 = None
        self.AUTOCORR2D_55 = None
        self.AUTOCORR2D_56 = None
        self.AUTOCORR2D_57 = None
        self.AUTOCORR2D_58 = None
        self.AUTOCORR2D_59 = None
        self.AUTOCORR2D_60 = None
        self.AUTOCORR2D_61 = None
        self.AUTOCORR2D_62 = None
        self.AUTOCORR2D_63 = None
        self.AUTOCORR2D_64 = None
        self.AUTOCORR2D_65 = None
        self.AUTOCORR2D_66 = None
        self.AUTOCORR2D_67 = None
        self.AUTOCORR2D_68 = None
        self.AUTOCORR2D_69 = None
        self.AUTOCORR2D_70 = None
        self.AUTOCORR2D_71 = None
        self.AUTOCORR2D_72 = None
        self.AUTOCORR2D_73 = None
        self.AUTOCORR2D_74 = None
        self.AUTOCORR2D_75 = None
        self.AUTOCORR2D_76 = None
        self.AUTOCORR2D_77 = None
        self.AUTOCORR2D_78 = None
        self.AUTOCORR2D_79 = None
        self.AUTOCORR2D_80 = None
        self.AUTOCORR2D_81 = None
        self.AUTOCORR2D_82 = None
        self.AUTOCORR2D_83 = None
        self.AUTOCORR2D_84 = None
        self.AUTOCORR2D_85 = None
        self.AUTOCORR2D_86 = None
        self.AUTOCORR2D_87 = None
        self.AUTOCORR2D_88 = None
        self.AUTOCORR2D_89 = None
        self.AUTOCORR2D_90 = None
        self.AUTOCORR2D_91 = None
        self.AUTOCORR2D_92 = None
        self.AUTOCORR2D_93 = None
        self.AUTOCORR2D_94 = None
        self.AUTOCORR2D_95 = None
        self.AUTOCORR2D_96 = None
        self.AUTOCORR2D_97 = None
        self.AUTOCORR2D_98 = None
        self.AUTOCORR2D_99 = None
        self.AUTOCORR2D_100 = None
        self.AUTOCORR2D_101 = None
        self.AUTOCORR2D_102 = None
        self.AUTOCORR2D_103 = None
        self.AUTOCORR2D_104 = None
        self.AUTOCORR2D_105 = None
        self.AUTOCORR2D_106 = None
        self.AUTOCORR2D_107 = None
        self.AUTOCORR2D_108 = None
        self.AUTOCORR2D_109 = None
        self.AUTOCORR2D_110 = None
        self.AUTOCORR2D_111 = None
        self.AUTOCORR2D_112 = None
        self.AUTOCORR2D_113 = None
        self.AUTOCORR2D_114 = None
        self.AUTOCORR2D_115 = None
        self.AUTOCORR2D_116 = None
        self.AUTOCORR2D_117 = None
        self.AUTOCORR2D_118 = None
        self.AUTOCORR2D_119 = None
        self.AUTOCORR2D_120 = None
        self.AUTOCORR2D_121 = None
        self.AUTOCORR2D_122 = None
        self.AUTOCORR2D_123 = None
        self.AUTOCORR2D_124 = None
        self.AUTOCORR2D_125 = None
        self.AUTOCORR2D_126 = None
        self.AUTOCORR2D_127 = None
        self.AUTOCORR2D_128 = None
        self.AUTOCORR2D_129 = None
        self.AUTOCORR2D_130 = None
        self.AUTOCORR2D_131 = None
        self.AUTOCORR2D_132 = None
        self.AUTOCORR2D_133 = None
        self.AUTOCORR2D_134 = None
        self.AUTOCORR2D_135 = None
        self.AUTOCORR2D_136 = None
        self.AUTOCORR2D_137 = None
        self.AUTOCORR2D_138 = None
        self.AUTOCORR2D_139 = None
        self.AUTOCORR2D_140 = None
        self.AUTOCORR2D_141 = None
        self.AUTOCORR2D_142 = None
        self.AUTOCORR2D_143 = None
        self.AUTOCORR2D_144 = None
        self.AUTOCORR2D_145 = None
        self.AUTOCORR2D_146 = None
        self.AUTOCORR2D_147 = None
        self.AUTOCORR2D_148 = None
        self.AUTOCORR2D_149 = None
        self.AUTOCORR2D_150 = None
        self.AUTOCORR2D_151 = None
        self.AUTOCORR2D_152 = None
        self.AUTOCORR2D_153 = None
        self.AUTOCORR2D_154 = None
        self.AUTOCORR2D_155 = None
        self.AUTOCORR2D_156 = None
        self.AUTOCORR2D_157 = None
        self.AUTOCORR2D_158 = None
        self.AUTOCORR2D_159 = None
        self.AUTOCORR2D_160 = None
        self.AUTOCORR2D_161 = None
        self.AUTOCORR2D_162 = None
        self.AUTOCORR2D_163 = None
        self.AUTOCORR2D_164 = None
        self.AUTOCORR2D_165 = None
        self.AUTOCORR2D_166 = None
        self.AUTOCORR2D_167 = None
        self.AUTOCORR2D_168 = None
        self.AUTOCORR2D_169 = None
        self.AUTOCORR2D_170 = None
        self.AUTOCORR2D_171 = None
        self.AUTOCORR2D_172 = None
        self.AUTOCORR2D_173 = None
        self.AUTOCORR2D_174 = None
        self.AUTOCORR2D_175 = None
        self.AUTOCORR2D_176 = None
        self.AUTOCORR2D_177 = None
        self.AUTOCORR2D_178 = None
        self.AUTOCORR2D_179 = None
        self.AUTOCORR2D_180 = None
        self.AUTOCORR2D_181 = None
        self.AUTOCORR2D_182 = None
        self.AUTOCORR2D_183 = None
        self.AUTOCORR2D_184 = None
        self.AUTOCORR2D_185 = None
        self.AUTOCORR2D_186 = None
        self.AUTOCORR2D_187 = None
        self.AUTOCORR2D_188 = None
        self.AUTOCORR2D_189 = None
        self.AUTOCORR2D_190 = None
        self.AUTOCORR2D_191 = None
        self.AUTOCORR2D_192 = None
        #endregion

        #region BCUT2D descriptors
        self.BCUT2D_CHGHI = None
        self.BCUT2D_CHGLO = None
        self.BCUT2D_LOGPHI = None
        self.BCUT2D_LOGPLOW = None
        self.BCUT2D_MRHI = None
        self.BCUT2D_MRLOW = None
        self.BCUT2D_MWHI = None
        self.BCUT2D_MWLOW = None
        #endregion

        self.BalabanJ = None
        self.BertzCT = None

        #region Chi descriptors
        self.Chi0 = None
        self.Chi0n = None
        self.Chi0v = None
        self.Chi1 = None
        self.Chi1n = None
        self.Chi1v = None
        self.Chi2n = None
        self.Chi2v = None
        self.Chi3n = None
        self.Chi3v = None
        self.Chi4n = None
        self.Chi4v = None
        #endregion

        #region EState descriptors
        self.EState_VSA1 = None
        self.EState_VSA2 = None
        self.EState_VSA3 = None
        self.EState_VSA4 = None
        self.EState_VSA5 = None
        self.EState_VSA6 = None
        self.EState_VSA7 = None
        self.EState_VSA8 = None
        self.EState_VSA9 = None
        self.EState_VSA10 = None
        self.EState_VSA11 = None

        self.MaxAbsEStateIndex = None
        self.MaxEStateIndex = None
        self.MinAbsEStateIndex = None
        self.MinEStateIndex = None
        #endregion

        self.ExactMolWt = None
        self.FpDensityMorgan1 = None
        self.FpDensityMorgan2 = None
        self.FpDensityMorgan3 = None

        #region fr_ descriptors
        self.fr_Al_COO = None
        self.fr_Al_OH = None
        self.fr_Al_OH_noTert = None
        self.fr_ArN = None
        self.fr_Ar_COO = None
        self.fr_Ar_N = None
        self.fr_Ar_NH = None
        self.fr_Ar_OH = None
        self.fr_COO = None
        self.fr_COO2 = None
        self.fr_C_O = None
        self.fr_C_O_noCOO = None
        self.fr_C_S = None
        self.fr_HOCCN = None
        self.fr_Imine = None
        self.fr_NH0 = None
        self.fr_NH1 = None
        self.fr_NH2 = None
        self.fr_N_O = None
        self.fr_Ndealkylation1 = None
        self.fr_Ndealkylation2 = None
        self.fr_Nhpyrrole = None
        self.fr_SH = None
        self.fr_aldehyde = None
        self.fr_alkyl_carbamate = None
        self.fr_alkyl_halide = None
        self.fr_allylic_oxid = None
        self.fr_amide = None
        self.fr_amidine = None
        self.fr_aniline = None
        self.fr_aryl_methyl = None
        self.fr_azide = None
        self.fr_azo = None
        self.fr_barbitur = None
        self.fr_benzene = None
        self.fr_benzodiazepine = None
        self.fr_bicyclic = None
        self.fr_diazo = None
        self.fr_dihydropyridine = None
        self.fr_epoxide = None
        self.fr_ester = None
        self.fr_ether = None
        self.fr_furan = None
        self.fr_guanido = None
        self.fr_halogen = None
        self.fr_hdrzine = None
        self.fr_hdrzone = None
        self.fr_imidazole = None
        self.fr_imide = None
        self.fr_isocyan = None
        self.fr_isothiocyan = None
        self.fr_ketone = None
        self.fr_ketone_Topliss = None
        self.fr_lactam = None
        self.fr_lactone = None
        self.fr_methoxy = None
        self.fr_morpholine = None
        self.fr_nitrile = None
        self.fr_nitro = None
        self.fr_nitro_arom = None
        self.fr_nitro_arom_nonortho = None
        self.fr_nitroso = None
        self.fr_oxazole = None
        self.fr_oxime = None
        self.fr_para_hydroxylation = None
        self.fr_phenol = None
        self.fr_phenol_noOrthoHbond = None
        self.fr_phos_acid = None
        self.fr_phos_ester = None
        self.fr_piperdine = None
        self.fr_piperzine = None
        self.fr_priamide = None
        self.fr_prisulfonamd = None
        self.fr_pyridine = None
        self.fr_quatN = None
        self.fr_sulfide = None
        self.fr_sulfonamd = None
        self.fr_sulfone = None
        self.fr_term_acetylene = None
        self.fr_tetrazole = None
        self.fr_thiazole = None
        self.fr_thiocyan = None
        self.fr_thiophene = None
        self.fr_unbrch_alkane = None
        self.fr_urea = None
        #endregion

        self.FractionCSP3 = None
        self.HallKierAlpha = None
        self.HeavyAtomMolWt = None
        self.HeavyAtomCount = None
        self.Ipc = None

        #region Kappa descriptors
        self.Kappa1 = None
        self.Kappa2 = None
        self.Kappa3 = None
        #endregion

        self.LabuteASA = None
        self.MaxAbsPartialCharge = None
        self.MaxPartialCharge = None
        self.MinAbsPartialCharge = None
        self.MinPartialCharge = None
        self.MolLogP = None
        self.MolMR = None
        self.MolWt = None
        self.NHOHCount = None
        self.NOCount = None

        #region 'count' descriptors
        self.NumAliphaticCarbocycles = None
        self.NumAliphaticHeterocycles = None
        self.NumAliphaticRings = None
        self.NumAromaticCarbocycles = None
        self.NumAromaticHeterocycles = None
        self.NumAromaticRings = None
        self.NumHAcceptors = None
        self.NumHDonors = None
        self.NumHeteroatoms = None
        self.NumRadicalElectrons = None
        self.NumRotatableBonds = None
        self.NumSaturatedCarbocycles = None
        self.NumSaturatedHeterocycles = None
        self.NumSaturatedRings = None
        self.NumValenceElectrons = None
        #endregion

        #region PEOE_VSA descriptors
        self.PEOE_VSA1 = None
        self.PEOE_VSA2 = None
        self.PEOE_VSA3 = None
        self.PEOE_VSA4 = None
        self.PEOE_VSA5 = None
        self.PEOE_VSA6 = None
        self.PEOE_VSA7 = None
        self.PEOE_VSA8 = None
        self.PEOE_VSA9 = None
        self.PEOE_VSA10 = None
        self.PEOE_VSA11 = None
        self.PEOE_VSA12 = None
        self.PEOE_VSA13 = None
        self.PEOE_VSA14 = None
        #endregion

        self.qed = None
        self.RingCount = None

        #region SMR_VSA descriptors
        self.SMR_VSA1 = None
        self.SMR_VSA2 = None
        self.SMR_VSA3 = None
        self.SMR_VSA4 = None
        self.SMR_VSA5 = None
        self.SMR_VSA6 = None
        self.SMR_VSA7 = None
        self.SMR_VSA8 = None
        self.SMR_VSA9 = None
        self.SMR_VSA10 = None
        #endregion

        #region SlogP_VSA descriptors
        self.SlogP_VSA1 = None
        self.SlogP_VSA2 = None
        self.SlogP_VSA3 = None
        self.SlogP_VSA4 = None
        self.SlogP_VSA5 = None
        self.SlogP_VSA6 = None
        self.SlogP_VSA7 = None
        self.SlogP_VSA8 = None
        self.SlogP_VSA9 = None
        self.SlogP_VSA10 = None
        self.SlogP_VSA11 = None
        self.SlogP_VSA12 = None
        #endregion

        self.TPSA = None

        #region VSA_EState descriptors
        self.VSA_EState1 = None
        self.VSA_EState2 = None
        self.VSA_EState3 = None
        self.VSA_EState4 = None
        self.VSA_EState5 = None
        self.VSA_EState6 = None
        self.VSA_EState7 = None
        self.VSA_EState8 = None
        self.VSA_EState9 = None
        self.VSA_EState10 = None
        #endregion

        #region 3D descriptors
        self.AUTOCORR3D_1 = None
        self.AUTOCORR3D_2 = None
        self.AUTOCORR3D_3 = None
        self.AUTOCORR3D_4 = None
        self.AUTOCORR3D_5 = None
        self.AUTOCORR3D_6 = None
        self.AUTOCORR3D_7 = None
        self.AUTOCORR3D_8 = None
        self.AUTOCORR3D_9 = None
        self.AUTOCORR3D_10 = None
        self.AUTOCORR3D_11 = None
        self.AUTOCORR3D_12 = None
        self.AUTOCORR3D_13 = None
        self.AUTOCORR3D_14 = None
        self.AUTOCORR3D_15 = None
        self.AUTOCORR3D_16 = None
        self.AUTOCORR3D_17 = None
        self.AUTOCORR3D_18 = None
        self.AUTOCORR3D_19 = None
        self.AUTOCORR3D_20 = None
        self.AUTOCORR3D_21 = None
        self.AUTOCORR3D_22 = None
        self.AUTOCORR3D_23 = None
        self.AUTOCORR3D_24 = None
        self.AUTOCORR3D_25 = None
        self.AUTOCORR3D_26 = None
        self.AUTOCORR3D_27 = None
        self.AUTOCORR3D_28 = None
        self.AUTOCORR3D_29 = None
        self.AUTOCORR3D_30 = None
        self.AUTOCORR3D_31 = None
        self.AUTOCORR3D_32 = None
        self.AUTOCORR3D_33 = None
        self.AUTOCORR3D_34 = None
        self.AUTOCORR3D_35 = None
        self.AUTOCORR3D_36 = None
        self.AUTOCORR3D_37 = None
        self.AUTOCORR3D_38 = None
        self.AUTOCORR3D_39 = None
        self.AUTOCORR3D_40 = None
        self.AUTOCORR3D_41 = None
        self.AUTOCORR3D_42 = None
        self.AUTOCORR3D_43 = None
        self.AUTOCORR3D_44 = None
        self.AUTOCORR3D_45 = None
        self.AUTOCORR3D_46 = None
        self.AUTOCORR3D_47 = None
        self.AUTOCORR3D_48 = None
        self.AUTOCORR3D_49 = None
        self.AUTOCORR3D_50 = None
        self.AUTOCORR3D_51 = None
        self.AUTOCORR3D_52 = None
        self.AUTOCORR3D_53 = None
        self.AUTOCORR3D_54 = None
        self.AUTOCORR3D_55 = None
        self.AUTOCORR3D_56 = None
        self.AUTOCORR3D_57 = None
        self.AUTOCORR3D_58 = None
        self.AUTOCORR3D_59 = None
        self.AUTOCORR3D_60 = None
        self.AUTOCORR3D_61 = None
        self.AUTOCORR3D_62 = None
        self.AUTOCORR3D_63 = None
        self.AUTOCORR3D_64 = None
        self.AUTOCORR3D_65 = None
        self.AUTOCORR3D_66 = None
        self.AUTOCORR3D_67 = None
        self.AUTOCORR3D_68 = None
        self.AUTOCORR3D_69 = None
        self.AUTOCORR3D_70 = None
        self.AUTOCORR3D_71 = None
        self.AUTOCORR3D_72 = None
        self.AUTOCORR3D_73 = None
        self.AUTOCORR3D_74 = None
        self.AUTOCORR3D_75 = None
        self.AUTOCORR3D_76 = None
        self.AUTOCORR3D_77 = None
        self.AUTOCORR3D_78 = None
        self.AUTOCORR3D_79 = None
        self.AUTOCORR3D_80 = None

        self.Asphericity = None
        self.Eccentricity = None
        self.InertialShapeFactor = None
        self.NPR1 = None
        self.NPR2 = None
        self.PMI1 = None
        self.PMI2 = None
        self.PMI3 = None
        self.RadiusOfGyration = None
        self.SpherocityIndex = None

        #endregion

        # If user pass a json
        if from_json_descriptors:
            # Read the descriptors from it
            data = read_descriptors_from_json(from_json_descriptors)
            # If data is None, a problem occurred while reading the json file
            if not data:
                ocprint.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None
            
            #region assign
            self.name, self.AUTOCORR2D_1, self.AUTOCORR2D_2, self.AUTOCORR2D_3, self.AUTOCORR2D_4, self.AUTOCORR2D_5, self.AUTOCORR2D_6, self.AUTOCORR2D_7, self.AUTOCORR2D_8, self.AUTOCORR2D_9, self.AUTOCORR2D_10, self.AUTOCORR2D_11, self.AUTOCORR2D_12, self.AUTOCORR2D_13, self.AUTOCORR2D_14, self.AUTOCORR2D_15, self.AUTOCORR2D_16, self.AUTOCORR2D_17, self.AUTOCORR2D_18, self.AUTOCORR2D_19, self.AUTOCORR2D_20, self.AUTOCORR2D_21, self.AUTOCORR2D_22, self.AUTOCORR2D_23, self.AUTOCORR2D_24, self.AUTOCORR2D_25, self.AUTOCORR2D_26, self.AUTOCORR2D_27, self.AUTOCORR2D_28, self.AUTOCORR2D_29, self.AUTOCORR2D_30, self.AUTOCORR2D_31, self.AUTOCORR2D_32, self.AUTOCORR2D_33, self.AUTOCORR2D_34, self.AUTOCORR2D_35, self.AUTOCORR2D_36, self.AUTOCORR2D_37, self.AUTOCORR2D_38, self.AUTOCORR2D_39, self.AUTOCORR2D_40, self.AUTOCORR2D_41, self.AUTOCORR2D_42, self.AUTOCORR2D_43, self.AUTOCORR2D_44, self.AUTOCORR2D_45, self.AUTOCORR2D_46, self.AUTOCORR2D_47, self.AUTOCORR2D_48, self.AUTOCORR2D_49, self.AUTOCORR2D_50, self.AUTOCORR2D_51, self.AUTOCORR2D_52, self.AUTOCORR2D_53, self.AUTOCORR2D_54, self.AUTOCORR2D_55, self.AUTOCORR2D_56, self.AUTOCORR2D_57, self.AUTOCORR2D_58, self.AUTOCORR2D_59, self.AUTOCORR2D_60, self.AUTOCORR2D_61, self.AUTOCORR2D_62, self.AUTOCORR2D_63, self.AUTOCORR2D_64, self.AUTOCORR2D_65, self.AUTOCORR2D_66, self.AUTOCORR2D_67, self.AUTOCORR2D_68, self.AUTOCORR2D_69, self.AUTOCORR2D_70, self.AUTOCORR2D_71, self.AUTOCORR2D_72, self.AUTOCORR2D_73, self.AUTOCORR2D_74, self.AUTOCORR2D_75, self.AUTOCORR2D_76, self.AUTOCORR2D_77, self.AUTOCORR2D_78, self.AUTOCORR2D_79, self.AUTOCORR2D_80, self.AUTOCORR2D_81, self.AUTOCORR2D_82, self.AUTOCORR2D_83, self.AUTOCORR2D_84, self.AUTOCORR2D_85, self.AUTOCORR2D_86, self.AUTOCORR2D_87, self.AUTOCORR2D_88, self.AUTOCORR2D_89, self.AUTOCORR2D_90, self.AUTOCORR2D_91, self.AUTOCORR2D_92, self.AUTOCORR2D_93, self.AUTOCORR2D_94, self.AUTOCORR2D_95, self.AUTOCORR2D_96, self.AUTOCORR2D_97, self.AUTOCORR2D_98, self.AUTOCORR2D_99, self.AUTOCORR2D_100, self.AUTOCORR2D_101, self.AUTOCORR2D_102, self.AUTOCORR2D_103, self.AUTOCORR2D_104, self.AUTOCORR2D_105, self.AUTOCORR2D_106, self.AUTOCORR2D_107, self.AUTOCORR2D_108, self.AUTOCORR2D_109, self.AUTOCORR2D_110, self.AUTOCORR2D_111, self.AUTOCORR2D_112, self.AUTOCORR2D_113, self.AUTOCORR2D_114, self.AUTOCORR2D_115, self.AUTOCORR2D_116, self.AUTOCORR2D_117, self.AUTOCORR2D_118, self.AUTOCORR2D_119, self.AUTOCORR2D_120, self.AUTOCORR2D_121, self.AUTOCORR2D_122, self.AUTOCORR2D_123, self.AUTOCORR2D_124, self.AUTOCORR2D_125, self.AUTOCORR2D_126, self.AUTOCORR2D_127, self.AUTOCORR2D_128, self.AUTOCORR2D_129, self.AUTOCORR2D_130, self.AUTOCORR2D_131, self.AUTOCORR2D_132, self.AUTOCORR2D_133, self.AUTOCORR2D_134, self.AUTOCORR2D_135, self.AUTOCORR2D_136, self.AUTOCORR2D_137, self.AUTOCORR2D_138, self.AUTOCORR2D_139, self.AUTOCORR2D_140, self.AUTOCORR2D_141, self.AUTOCORR2D_142, self.AUTOCORR2D_143, self.AUTOCORR2D_144, self.AUTOCORR2D_145, self.AUTOCORR2D_146, self.AUTOCORR2D_147, self.AUTOCORR2D_148, self.AUTOCORR2D_149, self.AUTOCORR2D_150, self.AUTOCORR2D_151, self.AUTOCORR2D_152, self.AUTOCORR2D_153, self.AUTOCORR2D_154, self.AUTOCORR2D_155, self.AUTOCORR2D_156, self.AUTOCORR2D_157, self.AUTOCORR2D_158, self.AUTOCORR2D_159, self.AUTOCORR2D_160, self.AUTOCORR2D_161, self.AUTOCORR2D_162, self.AUTOCORR2D_163, self.AUTOCORR2D_164, self.AUTOCORR2D_165, self.AUTOCORR2D_166, self.AUTOCORR2D_167, self.AUTOCORR2D_168, self.AUTOCORR2D_169, self.AUTOCORR2D_170, self.AUTOCORR2D_171, self.AUTOCORR2D_172, self.AUTOCORR2D_173, self.AUTOCORR2D_174, self.AUTOCORR2D_175, self.AUTOCORR2D_176, self.AUTOCORR2D_177, self.AUTOCORR2D_178, self.AUTOCORR2D_179, self.AUTOCORR2D_180, self.AUTOCORR2D_181, self.AUTOCORR2D_182, self.AUTOCORR2D_183, self.AUTOCORR2D_184, self.AUTOCORR2D_185, self.AUTOCORR2D_186, self.AUTOCORR2D_187, self.AUTOCORR2D_188, self.AUTOCORR2D_189, self.AUTOCORR2D_190, self.AUTOCORR2D_191, self.AUTOCORR2D_192, self.BCUT2D_CHGHI, self.BCUT2D_CHGLO, self.BCUT2D_LOGPHI, self.BCUT2D_LOGPLOW, self.BCUT2D_MRHI, self.BCUT2D_MRLOW, self.BCUT2D_MWHI, self.BCUT2D_MWLOW, self.BalabanJ, self.BertzCT, self.Chi0, self.Chi0n, self.Chi0v, self.Chi1, self.Chi1n, self.Chi1v, self.Chi2n, self.Chi2v, self.Chi3n, self.Chi3v, self.Chi4n, self.Chi4v, self.EState_VSA1, self.EState_VSA2, self.EState_VSA3, self.EState_VSA4, self.EState_VSA5, self.EState_VSA6, self.EState_VSA7, self.EState_VSA8, self.EState_VSA9, self.EState_VSA10, self.EState_VSA11, self.MaxAbsEStateIndex, self.MaxEStateIndex, self.MinAbsEStateIndex, self.MinEStateIndex, self.ExactMolWt, self.FpDensityMorgan1, self.FpDensityMorgan2, self.FpDensityMorgan3, self.fr_Al_COO, self.fr_Al_OH, self.fr_Al_OH_noTert, self.fr_ArN, self.fr_Ar_COO, self.fr_Ar_N, self.fr_Ar_NH, self.fr_Ar_OH, self.fr_COO, self.fr_COO2, self.fr_C_O, self.fr_C_O_noCOO, self.fr_C_S, self.fr_HOCCN, self.fr_Imine, self.fr_NH0, self.fr_NH1, self.fr_NH2, self.fr_N_O, self.fr_Ndealkylation1, self.fr_Ndealkylation2, self.fr_Nhpyrrole, self.fr_SH, self.fr_aldehyde, self.fr_alkyl_carbamate, self.fr_alkyl_halide, self.fr_allylic_oxid, self.fr_amide, self.fr_amidine, self.fr_aniline, self.fr_aryl_methyl, self.fr_azide, self.fr_azo, self.fr_barbitur, self.fr_benzene, self.fr_benzodiazepine, self.fr_bicyclic, self.fr_diazo, self.fr_dihydropyridine, self.fr_epoxide, self.fr_ester, self.fr_ether, self.fr_furan, self.fr_guanido, self.fr_halogen, self.fr_hdrzine, self.fr_hdrzone, self.fr_imidazole, self.fr_imide, self.fr_isocyan, self.fr_isothiocyan, self.fr_ketone, self.fr_ketone_Topliss, self.fr_lactam, self.fr_lactone, self.fr_methoxy, self.fr_morpholine, self.fr_nitrile, self.fr_nitro, self.fr_nitro_arom, self.fr_nitro_arom_nonortho, self.fr_nitroso, self.fr_oxazole, self.fr_oxime, self.fr_para_hydroxylation, self.fr_phenol, self.fr_phenol_noOrthoHbond, self.fr_phos_acid, self.fr_phos_ester, self.fr_piperdine, self.fr_piperzine, self.fr_priamide, self.fr_prisulfonamd, self.fr_pyridine, self.fr_quatN, self.fr_sulfide, self.fr_sulfonamd, self.fr_sulfone, self.fr_term_acetylene, self.fr_tetrazole, self.fr_thiazole, self.fr_thiocyan, self.fr_thiophene, self.fr_unbrch_alkane, self.fr_urea, self.FractionCSP3, self.HallKierAlpha, self.HeavyAtomMolWt, self.HeavyAtomCount, self.Ipc, self.Kappa1, self.Kappa2, self.Kappa3, self.LabuteASA, self.MaxAbsPartialCharge, self.MaxPartialCharge, self.MinAbsPartialCharge, self.MinPartialCharge, self.MolLogP, self.MolMR, self.MolWt, self.NHOHCount, self.NOCount, self.NumAliphaticCarbocycles, self.NumAliphaticHeterocycles, self.NumAliphaticRings, self.NumAromaticCarbocycles, self.NumAromaticHeterocycles, self.NumAromaticRings, self.NumHAcceptors, self.NumHDonors, self.NumHeteroatoms, self.NumRadicalElectrons, self.NumRotatableBonds, self.NumSaturatedCarbocycles, self.NumSaturatedHeterocycles, self.NumSaturatedRings, self.NumValenceElectrons, self.PEOE_VSA1, self.PEOE_VSA2, self.PEOE_VSA3, self.PEOE_VSA4, self.PEOE_VSA5, self.PEOE_VSA6, self.PEOE_VSA7, self.PEOE_VSA8, self.PEOE_VSA9, self.PEOE_VSA10, self.PEOE_VSA11, self.PEOE_VSA12, self.PEOE_VSA13, self.PEOE_VSA14, self.qed, self.RingCount, self.SMR_VSA1, self.SMR_VSA2, self.SMR_VSA3, self.SMR_VSA4, self.SMR_VSA5, self.SMR_VSA6, self.SMR_VSA7, self.SMR_VSA8, self.SMR_VSA9, self.SMR_VSA10, self.SlogP_VSA1, self.SlogP_VSA2, self.SlogP_VSA3, self.SlogP_VSA4, self.SlogP_VSA5, self.SlogP_VSA6, self.SlogP_VSA7, self.SlogP_VSA8, self.SlogP_VSA9, self.SlogP_VSA10, self.SlogP_VSA11, self.SlogP_VSA12, self.TPSA, self.VSA_EState1, self.VSA_EState2, self.VSA_EState3, self.VSA_EState4, self.VSA_EState5, self.VSA_EState6, self.VSA_EState7, self.VSA_EState8, self.VSA_EState9, self.VSA_EState10, self.AUTOCORR3D_1, self.AUTOCORR3D_2, self.AUTOCORR3D_3, self.AUTOCORR3D_4, self.AUTOCORR3D_5, self.AUTOCORR3D_6, self.AUTOCORR3D_7, self.AUTOCORR3D_8, self.AUTOCORR3D_9, self.AUTOCORR3D_10, self.AUTOCORR3D_11, self.AUTOCORR3D_12, self.AUTOCORR3D_13, self.AUTOCORR3D_14, self.AUTOCORR3D_15, self.AUTOCORR3D_16, self.AUTOCORR3D_17, self.AUTOCORR3D_18, self.AUTOCORR3D_19, self.AUTOCORR3D_20, self.AUTOCORR3D_21, self.AUTOCORR3D_22, self.AUTOCORR3D_23, self.AUTOCORR3D_24, self.AUTOCORR3D_25, self.AUTOCORR3D_26, self.AUTOCORR3D_27, self.AUTOCORR3D_28, self.AUTOCORR3D_29, self.AUTOCORR3D_30, self.AUTOCORR3D_31, self.AUTOCORR3D_32, self.AUTOCORR3D_33, self.AUTOCORR3D_34, self.AUTOCORR3D_35, self.AUTOCORR3D_36, self.AUTOCORR3D_37, self.AUTOCORR3D_38, self.AUTOCORR3D_39, self.AUTOCORR3D_40, self.AUTOCORR3D_41, self.AUTOCORR3D_42, self.AUTOCORR3D_43, self.AUTOCORR3D_44, self.AUTOCORR3D_45, self.AUTOCORR3D_46, self.AUTOCORR3D_47, self.AUTOCORR3D_48, self.AUTOCORR3D_49, self.AUTOCORR3D_50, self.AUTOCORR3D_51, self.AUTOCORR3D_52, self.AUTOCORR3D_53, self.AUTOCORR3D_54, self.AUTOCORR3D_55, self.AUTOCORR3D_56, self.AUTOCORR3D_57, self.AUTOCORR3D_58, self.AUTOCORR3D_59, self.AUTOCORR3D_60, self.AUTOCORR3D_61, self.AUTOCORR3D_62, self.AUTOCORR3D_63, self.AUTOCORR3D_64, self.AUTOCORR3D_65, self.AUTOCORR3D_66, self.AUTOCORR3D_67, self.AUTOCORR3D_68, self.AUTOCORR3D_69, self.AUTOCORR3D_70, self.AUTOCORR3D_71, self.AUTOCORR3D_72, self.AUTOCORR3D_73, self.AUTOCORR3D_74, self.AUTOCORR3D_75, self.AUTOCORR3D_76, self.AUTOCORR3D_77, self.AUTOCORR3D_78, self.AUTOCORR3D_79, self.AUTOCORR3D_80, self.Asphericity, self.Eccentricity, self.InertialShapeFactor, self.NPR1, self.NPR2, self.PMI1, self.PMI2, self.PMI3, self.RadiusOfGyration, self.SpherocityIndex = data # type: ignore
            #endregion

        else:
            # Check if the name is empty
            if not name:
                ocprint.print_error("The Ligand name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            #region AUTOCORR descriptors
            self.AUTOCORR2D_1 = findAUTOCORR2D_1(self.molecule)
            self.AUTOCORR2D_2 = findAUTOCORR2D_2(self.molecule)
            self.AUTOCORR2D_3 = findAUTOCORR2D_3(self.molecule)
            self.AUTOCORR2D_4 = findAUTOCORR2D_4(self.molecule)
            self.AUTOCORR2D_5 = findAUTOCORR2D_5(self.molecule)
            self.AUTOCORR2D_6 = findAUTOCORR2D_6(self.molecule)
            self.AUTOCORR2D_7 = findAUTOCORR2D_7(self.molecule)
            self.AUTOCORR2D_8 = findAUTOCORR2D_8(self.molecule)
            self.AUTOCORR2D_9 = findAUTOCORR2D_9(self.molecule)
            self.AUTOCORR2D_10 = findAUTOCORR2D_10(self.molecule)
            self.AUTOCORR2D_11 = findAUTOCORR2D_11(self.molecule)
            self.AUTOCORR2D_12 = findAUTOCORR2D_12(self.molecule)
            self.AUTOCORR2D_13 = findAUTOCORR2D_13(self.molecule)
            self.AUTOCORR2D_14 = findAUTOCORR2D_14(self.molecule)
            self.AUTOCORR2D_15 = findAUTOCORR2D_15(self.molecule)
            self.AUTOCORR2D_16 = findAUTOCORR2D_16(self.molecule)
            self.AUTOCORR2D_17 = findAUTOCORR2D_17(self.molecule)
            self.AUTOCORR2D_18 = findAUTOCORR2D_18(self.molecule)
            self.AUTOCORR2D_19 = findAUTOCORR2D_19(self.molecule)
            self.AUTOCORR2D_20 = findAUTOCORR2D_20(self.molecule)
            self.AUTOCORR2D_21 = findAUTOCORR2D_21(self.molecule)
            self.AUTOCORR2D_22 = findAUTOCORR2D_22(self.molecule)
            self.AUTOCORR2D_23 = findAUTOCORR2D_23(self.molecule)
            self.AUTOCORR2D_24 = findAUTOCORR2D_24(self.molecule)
            self.AUTOCORR2D_25 = findAUTOCORR2D_25(self.molecule)
            self.AUTOCORR2D_26 = findAUTOCORR2D_26(self.molecule)
            self.AUTOCORR2D_27 = findAUTOCORR2D_27(self.molecule)
            self.AUTOCORR2D_28 = findAUTOCORR2D_28(self.molecule)
            self.AUTOCORR2D_29 = findAUTOCORR2D_29(self.molecule)
            self.AUTOCORR2D_30 = findAUTOCORR2D_30(self.molecule)
            self.AUTOCORR2D_31 = findAUTOCORR2D_31(self.molecule)
            self.AUTOCORR2D_32 = findAUTOCORR2D_32(self.molecule)
            self.AUTOCORR2D_33 = findAUTOCORR2D_33(self.molecule)
            self.AUTOCORR2D_34 = findAUTOCORR2D_34(self.molecule)
            self.AUTOCORR2D_35 = findAUTOCORR2D_35(self.molecule)
            self.AUTOCORR2D_36 = findAUTOCORR2D_36(self.molecule)
            self.AUTOCORR2D_37 = findAUTOCORR2D_37(self.molecule)
            self.AUTOCORR2D_38 = findAUTOCORR2D_38(self.molecule)
            self.AUTOCORR2D_39 = findAUTOCORR2D_39(self.molecule)
            self.AUTOCORR2D_40 = findAUTOCORR2D_40(self.molecule)
            self.AUTOCORR2D_41 = findAUTOCORR2D_41(self.molecule)
            self.AUTOCORR2D_42 = findAUTOCORR2D_42(self.molecule)
            self.AUTOCORR2D_43 = findAUTOCORR2D_43(self.molecule)
            self.AUTOCORR2D_44 = findAUTOCORR2D_44(self.molecule)
            self.AUTOCORR2D_45 = findAUTOCORR2D_45(self.molecule)
            self.AUTOCORR2D_46 = findAUTOCORR2D_46(self.molecule)
            self.AUTOCORR2D_47 = findAUTOCORR2D_47(self.molecule)
            self.AUTOCORR2D_48 = findAUTOCORR2D_48(self.molecule)
            self.AUTOCORR2D_49 = findAUTOCORR2D_49(self.molecule)
            self.AUTOCORR2D_50 = findAUTOCORR2D_50(self.molecule)
            self.AUTOCORR2D_51 = findAUTOCORR2D_51(self.molecule)
            self.AUTOCORR2D_52 = findAUTOCORR2D_52(self.molecule)
            self.AUTOCORR2D_53 = findAUTOCORR2D_53(self.molecule)
            self.AUTOCORR2D_54 = findAUTOCORR2D_54(self.molecule)
            self.AUTOCORR2D_55 = findAUTOCORR2D_55(self.molecule)
            self.AUTOCORR2D_56 = findAUTOCORR2D_56(self.molecule)
            self.AUTOCORR2D_57 = findAUTOCORR2D_57(self.molecule)
            self.AUTOCORR2D_58 = findAUTOCORR2D_58(self.molecule)
            self.AUTOCORR2D_59 = findAUTOCORR2D_59(self.molecule)
            self.AUTOCORR2D_60 = findAUTOCORR2D_60(self.molecule)
            self.AUTOCORR2D_61 = findAUTOCORR2D_61(self.molecule)
            self.AUTOCORR2D_62 = findAUTOCORR2D_62(self.molecule)
            self.AUTOCORR2D_63 = findAUTOCORR2D_63(self.molecule)
            self.AUTOCORR2D_64 = findAUTOCORR2D_64(self.molecule)
            self.AUTOCORR2D_65 = findAUTOCORR2D_65(self.molecule)
            self.AUTOCORR2D_66 = findAUTOCORR2D_66(self.molecule)
            self.AUTOCORR2D_67 = findAUTOCORR2D_67(self.molecule)
            self.AUTOCORR2D_68 = findAUTOCORR2D_68(self.molecule)
            self.AUTOCORR2D_69 = findAUTOCORR2D_69(self.molecule)
            self.AUTOCORR2D_70 = findAUTOCORR2D_70(self.molecule)
            self.AUTOCORR2D_71 = findAUTOCORR2D_71(self.molecule)
            self.AUTOCORR2D_72 = findAUTOCORR2D_72(self.molecule)
            self.AUTOCORR2D_73 = findAUTOCORR2D_73(self.molecule)
            self.AUTOCORR2D_74 = findAUTOCORR2D_74(self.molecule)
            self.AUTOCORR2D_75 = findAUTOCORR2D_75(self.molecule)
            self.AUTOCORR2D_76 = findAUTOCORR2D_76(self.molecule)
            self.AUTOCORR2D_77 = findAUTOCORR2D_77(self.molecule)
            self.AUTOCORR2D_78 = findAUTOCORR2D_78(self.molecule)
            self.AUTOCORR2D_79 = findAUTOCORR2D_79(self.molecule)
            self.AUTOCORR2D_80 = findAUTOCORR2D_80(self.molecule)
            self.AUTOCORR2D_81 = findAUTOCORR2D_81(self.molecule)
            self.AUTOCORR2D_82 = findAUTOCORR2D_82(self.molecule)
            self.AUTOCORR2D_83 = findAUTOCORR2D_83(self.molecule)
            self.AUTOCORR2D_84 = findAUTOCORR2D_84(self.molecule)
            self.AUTOCORR2D_85 = findAUTOCORR2D_85(self.molecule)
            self.AUTOCORR2D_86 = findAUTOCORR2D_86(self.molecule)
            self.AUTOCORR2D_87 = findAUTOCORR2D_87(self.molecule)
            self.AUTOCORR2D_88 = findAUTOCORR2D_88(self.molecule)
            self.AUTOCORR2D_89 = findAUTOCORR2D_89(self.molecule)
            self.AUTOCORR2D_90 = findAUTOCORR2D_90(self.molecule)
            self.AUTOCORR2D_91 = findAUTOCORR2D_91(self.molecule)
            self.AUTOCORR2D_92 = findAUTOCORR2D_92(self.molecule)
            self.AUTOCORR2D_93 = findAUTOCORR2D_93(self.molecule)
            self.AUTOCORR2D_94 = findAUTOCORR2D_94(self.molecule)
            self.AUTOCORR2D_95 = findAUTOCORR2D_95(self.molecule)
            self.AUTOCORR2D_96 = findAUTOCORR2D_96(self.molecule)
            self.AUTOCORR2D_97 = findAUTOCORR2D_97(self.molecule)
            self.AUTOCORR2D_98 = findAUTOCORR2D_98(self.molecule)
            self.AUTOCORR2D_99 = findAUTOCORR2D_99(self.molecule)
            self.AUTOCORR2D_100 = findAUTOCORR2D_100(self.molecule)
            self.AUTOCORR2D_101 = findAUTOCORR2D_101(self.molecule)
            self.AUTOCORR2D_102 = findAUTOCORR2D_102(self.molecule)
            self.AUTOCORR2D_103 = findAUTOCORR2D_103(self.molecule)
            self.AUTOCORR2D_104 = findAUTOCORR2D_104(self.molecule)
            self.AUTOCORR2D_105 = findAUTOCORR2D_105(self.molecule)
            self.AUTOCORR2D_106 = findAUTOCORR2D_106(self.molecule)
            self.AUTOCORR2D_107 = findAUTOCORR2D_107(self.molecule)
            self.AUTOCORR2D_108 = findAUTOCORR2D_108(self.molecule)
            self.AUTOCORR2D_109 = findAUTOCORR2D_109(self.molecule)
            self.AUTOCORR2D_110 = findAUTOCORR2D_110(self.molecule)
            self.AUTOCORR2D_111 = findAUTOCORR2D_111(self.molecule)
            self.AUTOCORR2D_112 = findAUTOCORR2D_112(self.molecule)
            self.AUTOCORR2D_113 = findAUTOCORR2D_113(self.molecule)
            self.AUTOCORR2D_114 = findAUTOCORR2D_114(self.molecule)
            self.AUTOCORR2D_115 = findAUTOCORR2D_115(self.molecule)
            self.AUTOCORR2D_116 = findAUTOCORR2D_116(self.molecule)
            self.AUTOCORR2D_117 = findAUTOCORR2D_117(self.molecule)
            self.AUTOCORR2D_118 = findAUTOCORR2D_118(self.molecule)
            self.AUTOCORR2D_119 = findAUTOCORR2D_119(self.molecule)
            self.AUTOCORR2D_120 = findAUTOCORR2D_120(self.molecule)
            self.AUTOCORR2D_121 = findAUTOCORR2D_121(self.molecule)
            self.AUTOCORR2D_122 = findAUTOCORR2D_122(self.molecule)
            self.AUTOCORR2D_123 = findAUTOCORR2D_123(self.molecule)
            self.AUTOCORR2D_124 = findAUTOCORR2D_124(self.molecule)
            self.AUTOCORR2D_125 = findAUTOCORR2D_125(self.molecule)
            self.AUTOCORR2D_126 = findAUTOCORR2D_126(self.molecule)
            self.AUTOCORR2D_127 = findAUTOCORR2D_127(self.molecule)
            self.AUTOCORR2D_128 = findAUTOCORR2D_128(self.molecule)
            self.AUTOCORR2D_129 = findAUTOCORR2D_129(self.molecule)
            self.AUTOCORR2D_130 = findAUTOCORR2D_130(self.molecule)
            self.AUTOCORR2D_131 = findAUTOCORR2D_131(self.molecule)
            self.AUTOCORR2D_132 = findAUTOCORR2D_132(self.molecule)
            self.AUTOCORR2D_133 = findAUTOCORR2D_133(self.molecule)
            self.AUTOCORR2D_134 = findAUTOCORR2D_134(self.molecule)
            self.AUTOCORR2D_135 = findAUTOCORR2D_135(self.molecule)
            self.AUTOCORR2D_136 = findAUTOCORR2D_136(self.molecule)
            self.AUTOCORR2D_137 = findAUTOCORR2D_137(self.molecule)
            self.AUTOCORR2D_138 = findAUTOCORR2D_138(self.molecule)
            self.AUTOCORR2D_139 = findAUTOCORR2D_139(self.molecule)
            self.AUTOCORR2D_140 = findAUTOCORR2D_140(self.molecule)
            self.AUTOCORR2D_141 = findAUTOCORR2D_141(self.molecule)
            self.AUTOCORR2D_142 = findAUTOCORR2D_142(self.molecule)
            self.AUTOCORR2D_143 = findAUTOCORR2D_143(self.molecule)
            self.AUTOCORR2D_144 = findAUTOCORR2D_144(self.molecule)
            self.AUTOCORR2D_145 = findAUTOCORR2D_145(self.molecule)
            self.AUTOCORR2D_146 = findAUTOCORR2D_146(self.molecule)
            self.AUTOCORR2D_147 = findAUTOCORR2D_147(self.molecule)
            self.AUTOCORR2D_148 = findAUTOCORR2D_148(self.molecule)
            self.AUTOCORR2D_149 = findAUTOCORR2D_149(self.molecule)
            self.AUTOCORR2D_150 = findAUTOCORR2D_150(self.molecule)
            self.AUTOCORR2D_151 = findAUTOCORR2D_151(self.molecule)
            self.AUTOCORR2D_152 = findAUTOCORR2D_152(self.molecule)
            self.AUTOCORR2D_153 = findAUTOCORR2D_153(self.molecule)
            self.AUTOCORR2D_154 = findAUTOCORR2D_154(self.molecule)
            self.AUTOCORR2D_155 = findAUTOCORR2D_155(self.molecule)
            self.AUTOCORR2D_156 = findAUTOCORR2D_156(self.molecule)
            self.AUTOCORR2D_157 = findAUTOCORR2D_157(self.molecule)
            self.AUTOCORR2D_158 = findAUTOCORR2D_158(self.molecule)
            self.AUTOCORR2D_159 = findAUTOCORR2D_159(self.molecule)
            self.AUTOCORR2D_160 = findAUTOCORR2D_160(self.molecule)
            self.AUTOCORR2D_161 = findAUTOCORR2D_161(self.molecule)
            self.AUTOCORR2D_162 = findAUTOCORR2D_162(self.molecule)
            self.AUTOCORR2D_163 = findAUTOCORR2D_163(self.molecule)
            self.AUTOCORR2D_164 = findAUTOCORR2D_164(self.molecule)
            self.AUTOCORR2D_165 = findAUTOCORR2D_165(self.molecule)
            self.AUTOCORR2D_166 = findAUTOCORR2D_166(self.molecule)
            self.AUTOCORR2D_167 = findAUTOCORR2D_167(self.molecule)
            self.AUTOCORR2D_168 = findAUTOCORR2D_168(self.molecule)
            self.AUTOCORR2D_169 = findAUTOCORR2D_169(self.molecule)
            self.AUTOCORR2D_170 = findAUTOCORR2D_170(self.molecule)
            self.AUTOCORR2D_171 = findAUTOCORR2D_171(self.molecule)
            self.AUTOCORR2D_172 = findAUTOCORR2D_172(self.molecule)
            self.AUTOCORR2D_173 = findAUTOCORR2D_173(self.molecule)
            self.AUTOCORR2D_174 = findAUTOCORR2D_174(self.molecule)
            self.AUTOCORR2D_175 = findAUTOCORR2D_175(self.molecule)
            self.AUTOCORR2D_176 = findAUTOCORR2D_176(self.molecule)
            self.AUTOCORR2D_177 = findAUTOCORR2D_177(self.molecule)
            self.AUTOCORR2D_178 = findAUTOCORR2D_178(self.molecule)
            self.AUTOCORR2D_179 = findAUTOCORR2D_179(self.molecule)
            self.AUTOCORR2D_180 = findAUTOCORR2D_180(self.molecule)
            self.AUTOCORR2D_181 = findAUTOCORR2D_181(self.molecule)
            self.AUTOCORR2D_182 = findAUTOCORR2D_182(self.molecule)
            self.AUTOCORR2D_183 = findAUTOCORR2D_183(self.molecule)
            self.AUTOCORR2D_184 = findAUTOCORR2D_184(self.molecule)
            self.AUTOCORR2D_185 = findAUTOCORR2D_185(self.molecule)
            self.AUTOCORR2D_186 = findAUTOCORR2D_186(self.molecule)
            self.AUTOCORR2D_187 = findAUTOCORR2D_187(self.molecule)
            self.AUTOCORR2D_188 = findAUTOCORR2D_188(self.molecule)
            self.AUTOCORR2D_189 = findAUTOCORR2D_189(self.molecule)
            self.AUTOCORR2D_190 = findAUTOCORR2D_190(self.molecule)
            self.AUTOCORR2D_191 = findAUTOCORR2D_191(self.molecule)
            self.AUTOCORR2D_192 = findAUTOCORR2D_192(self.molecule)
            #endregion

            #region BCUT2D descriptors
            self.BCUT2D_CHGHI = findBCUT2D_CHGHI(self.molecule)
            self.BCUT2D_CHGLO = findBCUT2D_CHGLO(self.molecule)
            self.BCUT2D_LOGPHI = findBCUT2D_LOGPHI(self.molecule)
            self.BCUT2D_LOGPLOW = findBCUT2D_LOGPLOW(self.molecule)
            self.BCUT2D_MRHI = findBCUT2D_MRHI(self.molecule)
            self.BCUT2D_MRLOW = findBCUT2D_MRLOW(self.molecule)
            self.BCUT2D_MWHI = findBCUT2D_MWHI(self.molecule)
            self.BCUT2D_MWLOW = findBCUT2D_MWLOW(self.molecule)
            #endregion

            self.BalabanJ = findBalabanJ(self.molecule)
            self.BertzCT = findBertzCT(self.molecule)

            #region Chi descriptors
            self.Chi0 = findChi0(self.molecule)
            self.Chi0n = findChi0n(self.molecule)
            self.Chi0v = findChi0v(self.molecule)
            self.Chi1 = findChi1(self.molecule)
            self.Chi1n = findChi1n(self.molecule)
            self.Chi1v = findChi1v(self.molecule)
            self.Chi2n = findChi2n(self.molecule)
            self.Chi2v = findChi2v(self.molecule)
            self.Chi3n = findChi3n(self.molecule)
            self.Chi3v = findChi3v(self.molecule)
            self.Chi4n = findChi4n(self.molecule)
            self.Chi4v = findChi4v(self.molecule)
            #endregion

            #region EState descriptors
            self.EState_VSA1 = findEState_VSA1(self.molecule)
            self.EState_VSA2 = findEState_VSA2(self.molecule)
            self.EState_VSA3 = findEState_VSA3(self.molecule)
            self.EState_VSA4 = findEState_VSA4(self.molecule)
            self.EState_VSA5 = findEState_VSA5(self.molecule)
            self.EState_VSA6 = findEState_VSA6(self.molecule)
            self.EState_VSA7 = findEState_VSA7(self.molecule)
            self.EState_VSA8 = findEState_VSA8(self.molecule)
            self.EState_VSA9 = findEState_VSA9(self.molecule)
            self.EState_VSA10 = findEState_VSA10(self.molecule)
            self.EState_VSA11 = findEState_VSA11(self.molecule)

            self.MaxAbsEStateIndex = findMaxAbsEStateIndex(self.molecule)
            self.MaxEStateIndex = findMaxEStateIndex(self.molecule)
            self.MinAbsEStateIndex = findMinAbsEStateIndex(self.molecule)
            self.MinEStateIndex = findMinEStateIndex(self.molecule)
            #endregion

            self.ExactMolWt = findExactMolWt(self.molecule)
            self.FpDensityMorgan1 = findFpDensityMorgan1(self.molecule)
            self.FpDensityMorgan2 = findFpDensityMorgan2(self.molecule)
            self.FpDensityMorgan3 = findFpDensityMorgan3(self.molecule)

            #region fr_ descriptors
            self.fr_Al_COO = findfr_Al_COO(self.molecule)
            self.fr_Al_OH = findfr_Al_OH(self.molecule)
            self.fr_Al_OH_noTert = findfr_Al_OH_noTert(self.molecule)
            self.fr_ArN = findfr_ArN(self.molecule)
            self.fr_Ar_COO = findfr_Ar_COO(self.molecule)
            self.fr_Ar_N = findfr_Ar_N(self.molecule)
            self.fr_Ar_NH = findfr_Ar_NH(self.molecule)
            self.fr_Ar_OH = findfr_Ar_OH(self.molecule)
            self.fr_COO = findfr_COO(self.molecule)
            self.fr_COO2 = findfr_COO2(self.molecule)
            self.fr_C_O = findfr_C_O(self.molecule)
            self.fr_C_O_noCOO = findfr_C_O_noCOO(self.molecule)
            self.fr_C_S = findfr_C_S(self.molecule)
            self.fr_HOCCN = findfr_HOCCN(self.molecule)
            self.fr_Imine = findfr_Imine(self.molecule)
            self.fr_NH0 = findfr_NH0(self.molecule)
            self.fr_NH1 = findfr_NH1(self.molecule)
            self.fr_NH2 = findfr_NH2(self.molecule)
            self.fr_N_O = findfr_N_O(self.molecule)
            self.fr_Ndealkylation1 = findfr_Ndealkylation1(self.molecule)
            self.fr_Ndealkylation2 = findfr_Ndealkylation2(self.molecule)
            self.fr_Nhpyrrole = findfr_Nhpyrrole(self.molecule)
            self.fr_SH = findfr_SH(self.molecule)
            self.fr_aldehyde = findfr_aldehyde(self.molecule)
            self.fr_alkyl_carbamate = findfr_alkyl_carbamate(self.molecule)
            self.fr_alkyl_halide = findfr_alkyl_halide(self.molecule)
            self.fr_allylic_oxid = findfr_allylic_oxid(self.molecule)
            self.fr_amide = findfr_amide(self.molecule)
            self.fr_amidine = findfr_amidine(self.molecule)
            self.fr_aniline = findfr_aniline(self.molecule)
            self.fr_aryl_methyl = findfr_aryl_methyl(self.molecule)
            self.fr_azide = findfr_azide(self.molecule)
            self.fr_azo = findfr_azo(self.molecule)
            self.fr_barbitur = findfr_barbitur(self.molecule)
            self.fr_benzene = findfr_benzene(self.molecule)
            self.fr_benzodiazepine = findfr_benzodiazepine(self.molecule)
            self.fr_bicyclic = findfr_bicyclic(self.molecule)
            self.fr_diazo = findfr_diazo(self.molecule)
            self.fr_dihydropyridine = findfr_dihydropyridine(self.molecule)
            self.fr_epoxide = findfr_epoxide(self.molecule)
            self.fr_ester = findfr_ester(self.molecule)
            self.fr_ether = findfr_ether(self.molecule)
            self.fr_furan = findfr_furan(self.molecule)
            self.fr_guanido = findfr_guanido(self.molecule)
            self.fr_halogen = findfr_halogen(self.molecule)
            self.fr_hdrzine = findfr_hdrzine(self.molecule)
            self.fr_hdrzone = findfr_hdrzone(self.molecule)
            self.fr_imidazole = findfr_imidazole(self.molecule)
            self.fr_imide = findfr_imide(self.molecule)
            self.fr_isocyan = findfr_isocyan(self.molecule)
            self.fr_isothiocyan = findfr_isothiocyan(self.molecule)
            self.fr_ketone = findfr_ketone(self.molecule)
            self.fr_ketone_Topliss = findfr_ketone_Topliss(self.molecule)
            self.fr_lactam = findfr_lactam(self.molecule)
            self.fr_lactone = findfr_lactone(self.molecule)
            self.fr_methoxy = findfr_methoxy(self.molecule)
            self.fr_morpholine = findfr_morpholine(self.molecule)
            self.fr_nitrile = findfr_nitrile(self.molecule)
            self.fr_nitro = findfr_nitro(self.molecule)
            self.fr_nitro_arom = findfr_nitro_arom(self.molecule)
            self.fr_nitro_arom_nonortho = findfr_nitro_arom_nonortho(self.molecule)
            self.fr_nitroso = findfr_nitroso(self.molecule)
            self.fr_oxazole = findfr_oxazole(self.molecule)
            self.fr_oxime = findfr_oxime(self.molecule)
            self.fr_para_hydroxylation = findfr_para_hydroxylation(self.molecule)
            self.fr_phenol = findfr_phenol(self.molecule)
            self.fr_phenol_noOrthoHbond = findfr_phenol_noOrthoHbond(self.molecule)
            self.fr_phos_acid = findfr_phos_acid(self.molecule)
            self.fr_phos_ester = findfr_phos_ester(self.molecule)
            self.fr_piperdine = findfr_piperdine(self.molecule)
            self.fr_piperzine = findfr_piperzine(self.molecule)
            self.fr_priamide = findfr_priamide(self.molecule)
            self.fr_prisulfonamd = findfr_prisulfonamd(self.molecule)
            self.fr_pyridine = findfr_pyridine(self.molecule)
            self.fr_quatN = findfr_quatN(self.molecule)
            self.fr_sulfide = findfr_sulfide(self.molecule)
            self.fr_sulfonamd = findfr_sulfonamd(self.molecule)
            self.fr_sulfone = findfr_sulfone(self.molecule)
            self.fr_term_acetylene = findfr_term_acetylene(self.molecule)
            self.fr_tetrazole = findfr_tetrazole(self.molecule)
            self.fr_thiazole = findfr_thiazole(self.molecule)
            self.fr_thiocyan = findfr_thiocyan(self.molecule)
            self.fr_thiophene = findfr_thiophene(self.molecule)
            self.fr_unbrch_alkane = findfr_unbrch_alkane(self.molecule)
            self.fr_urea = findfr_urea(self.molecule)
            #endregion

            self.FractionCSP3 = findFractionCSP3(self.molecule)
            self.HallKierAlpha = findHallKierAlpha(self.molecule)
            self.HeavyAtomMolWt = findHeavyAtomMolWt(self.molecule)
            self.HeavyAtomCount = findHeavyAtomCount(self.molecule)
            self.Ipc = findIpc(self.molecule)

            #region Kappa descriptors
            self.Kappa1 = findKappa1(self.molecule)
            self.Kappa2 = findKappa2(self.molecule)
            self.Kappa3 = findKappa3(self.molecule)
            #endregion

            self.LabuteASA = findLabuteASA(self.molecule)
            self.MaxAbsPartialCharge = findMaxAbsPartialCharge(self.molecule)
            self.MaxPartialCharge = findMaxPartialCharge(self.molecule)
            self.MinAbsPartialCharge = findMinAbsPartialCharge(self.molecule)
            self.MinPartialCharge = findMinPartialCharge(self.molecule)
            self.MolLogP = findMolWt(self.molecule)
            self.MolMR = findMolMR(self.molecule)
            self.MolWt = findMolWt(self.molecule)

            #region 'count' descriptors
            self.NHOHCount = findNHOHCount(self.molecule)
            self.NOCount = findNOCount(self.molecule)
            self.NumAliphaticHeterocycles = findNumAliphaticHeterocycles(self.molecule)
            self.NumAliphaticRings = findNumAliphaticRings(self.molecule)
            self.NumAromaticCarbocycles = findNumAromaticCarbocycles(self.molecule)
            self.NumAromaticHeterocycles = findNumAromaticHeterocycles(self.molecule)
            self.NumAromaticRings = findNumAromaticRings(self.molecule)
            self.NumHAcceptors = findNumHAcceptors(self.molecule)
            self.NumHDonors = findNumHDonors(self.molecule)
            self.NumHeteroatoms = findNumHeteroatoms(self.molecule)
            self.NumRadicalElectrons = findNumRadicalElectrons(self.molecule)
            self.NumRotatableBonds = findNumRotatableBonds(self.molecule)
            self.NumSaturatedCarbocycles = findNumSaturatedCarbocycles(self.molecule)
            self.NumSaturatedHeterocycles = findNumSaturatedHeterocycles(self.molecule)
            self.NumSaturatedRings = findNumSaturatedRings(self.molecule)
            self.NumValenceElectrons = findNumValenceElectrons(self.molecule)
            self.NumAliphaticCarbocycles = findNumAliphaticCarbocycles(self.molecule)
            self.RingCount = findRingCount(self.molecule)
            #endregion

            #region PEOE_VSA descriptors
            self.PEOE_VSA1 = findPEOE_VSA1(self.molecule)
            self.PEOE_VSA2 = findPEOE_VSA2(self.molecule)
            self.PEOE_VSA3 = findPEOE_VSA3(self.molecule)
            self.PEOE_VSA4 = findPEOE_VSA4(self.molecule)
            self.PEOE_VSA5 = findPEOE_VSA5(self.molecule)
            self.PEOE_VSA6 = findPEOE_VSA6(self.molecule)
            self.PEOE_VSA7 = findPEOE_VSA7(self.molecule)
            self.PEOE_VSA8 = findPEOE_VSA8(self.molecule)
            self.PEOE_VSA9 = findPEOE_VSA9(self.molecule)
            self.PEOE_VSA10 = findPEOE_VSA10(self.molecule)
            self.PEOE_VSA11 = findPEOE_VSA11(self.molecule)
            self.PEOE_VSA12 = findPEOE_VSA12(self.molecule)
            self.PEOE_VSA13 = findPEOE_VSA13(self.molecule)
            self.PEOE_VSA14 = findPEOE_VSA14(self.molecule)
            #endregion

            self.qed = findqed(self.molecule)

            #region SMR_VSA descriptors
            self.SMR_VSA1 = findSMR_VSA1(self.molecule)
            self.SMR_VSA2 = findSMR_VSA2(self.molecule)
            self.SMR_VSA3 = findSMR_VSA3(self.molecule)
            self.SMR_VSA4 = findSMR_VSA4(self.molecule)
            self.SMR_VSA5 = findSMR_VSA5(self.molecule)
            self.SMR_VSA6 = findSMR_VSA6(self.molecule)
            self.SMR_VSA7 = findSMR_VSA7(self.molecule)
            self.SMR_VSA8 = findSMR_VSA8(self.molecule)
            self.SMR_VSA9 = findSMR_VSA9(self.molecule)
            self.SMR_VSA10 = findSMR_VSA10(self.molecule)
            #endregion

            #region SlogP_VSA descriptors
            self.SlogP_VSA1 = findSlogP_VSA1(self.molecule)
            self.SlogP_VSA2 = findSlogP_VSA2(self.molecule)
            self.SlogP_VSA3 = findSlogP_VSA3(self.molecule)
            self.SlogP_VSA4 = findSlogP_VSA4(self.molecule)
            self.SlogP_VSA5 = findSlogP_VSA5(self.molecule)
            self.SlogP_VSA6 = findSlogP_VSA6(self.molecule)
            self.SlogP_VSA7 = findSlogP_VSA7(self.molecule)
            self.SlogP_VSA8 = findSlogP_VSA8(self.molecule)
            self.SlogP_VSA9 = findSlogP_VSA9(self.molecule)
            self.SlogP_VSA10 = findSlogP_VSA10(self.molecule)
            self.SlogP_VSA11 = findSlogP_VSA11(self.molecule)
            self.SlogP_VSA12 = findSlogP_VSA12(self.molecule)
            #endregion

            self.TPSA = findTPSA(self.molecule)

            #region VSA_EState descriptors
            self.VSA_EState1 = findVSA_EState1(self.molecule)
            self.VSA_EState2 = findVSA_EState2(self.molecule)
            self.VSA_EState3 = findVSA_EState3(self.molecule)
            self.VSA_EState4 = findVSA_EState4(self.molecule)
            self.VSA_EState5 = findVSA_EState5(self.molecule)
            self.VSA_EState6 = findVSA_EState6(self.molecule)
            self.VSA_EState7 = findVSA_EState7(self.molecule)
            self.VSA_EState8 = findVSA_EState8(self.molecule)
            self.VSA_EState9 = findVSA_EState9(self.molecule)
            self.VSA_EState10 = findVSA_EState10(self.molecule)
            #endregion

            #region 3D descriptors
            self._AUTOCORR3D = findAUTOCORR3D(self.molecule)

            if self._AUTOCORR3D:
                self.AUTOCORR3D_1,  self.AUTOCORR3D_2,  self.AUTOCORR3D_3,  self.AUTOCORR3D_4,  self.AUTOCORR3D_5,  self.AUTOCORR3D_6,  self.AUTOCORR3D_7,  self.AUTOCORR3D_8,  self.AUTOCORR3D_9,  self.AUTOCORR3D_10, self.AUTOCORR3D_11, self.AUTOCORR3D_12, self.AUTOCORR3D_13, self.AUTOCORR3D_14, self.AUTOCORR3D_15, self.AUTOCORR3D_16, self.AUTOCORR3D_17, self.AUTOCORR3D_18, self.AUTOCORR3D_19, self.AUTOCORR3D_20, self.AUTOCORR3D_21, self.AUTOCORR3D_22, self.AUTOCORR3D_23, self.AUTOCORR3D_24, self.AUTOCORR3D_25, self.AUTOCORR3D_26, self.AUTOCORR3D_27, self.AUTOCORR3D_28, self.AUTOCORR3D_29, self.AUTOCORR3D_30, self.AUTOCORR3D_31, self.AUTOCORR3D_32, self.AUTOCORR3D_33, self.AUTOCORR3D_34, self.AUTOCORR3D_35, self.AUTOCORR3D_36, self.AUTOCORR3D_37, self.AUTOCORR3D_38, self.AUTOCORR3D_39, self.AUTOCORR3D_40, self.AUTOCORR3D_41, self.AUTOCORR3D_42, self.AUTOCORR3D_43, self.AUTOCORR3D_44, self.AUTOCORR3D_45, self.AUTOCORR3D_46, self.AUTOCORR3D_47, self.AUTOCORR3D_48, self.AUTOCORR3D_49, self.AUTOCORR3D_50, self.AUTOCORR3D_51, self.AUTOCORR3D_52, self.AUTOCORR3D_53, self.AUTOCORR3D_54, self.AUTOCORR3D_55, self.AUTOCORR3D_56, self.AUTOCORR3D_57, self.AUTOCORR3D_58, self.AUTOCORR3D_59, self.AUTOCORR3D_60, self.AUTOCORR3D_61, self.AUTOCORR3D_62, self.AUTOCORR3D_63, self.AUTOCORR3D_64, self.AUTOCORR3D_65, self.AUTOCORR3D_66, self.AUTOCORR3D_67, self.AUTOCORR3D_68, self.AUTOCORR3D_69, self.AUTOCORR3D_70, self.AUTOCORR3D_71, self.AUTOCORR3D_72, self.AUTOCORR3D_73, self.AUTOCORR3D_74, self.AUTOCORR3D_75, self.AUTOCORR3D_76, self.AUTOCORR3D_77, self.AUTOCORR3D_78, self.AUTOCORR3D_79, self.AUTOCORR3D_80 = self._AUTOCORR3D

            self.Asphericity = findAsphericity(self.molecule)
            self.Eccentricity = findEccentricity(self.molecule)
            self.InertialShapeFactor = findInertialShapeFactor(self.molecule)
            self.NPR1 = findNPR1(self.molecule)
            self.NPR2 = findNPR2(self.molecule)
            self.PMI1 = findPMI1(self.molecule)
            self.PMI2 = findPMI2(self.molecule)
            self.PMI3 = findPMI3(self.molecule)
            self.RadiusOfGyration = findRadiusOfGyration(self.molecule)
            self.SpherocityIndex = findSpherocityIndex(self.molecule)
            #endregion

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

    ## Public ##
    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        #region prints
        print(f"Name:                     '{self.name if self.name is not None else '-' }'")
        print(f"Molecule:                 '{self.molecule if self.molecule is not None else '-' }'")
        print(f"AUTOCORR2D_1:             '{self.AUTOCORR2D_1 if self.AUTOCORR2D_1 is not None else '-' }'")
        print(f"AUTOCORR2D_2:             '{self.AUTOCORR2D_2 if self.AUTOCORR2D_2 is not None else '-' }'")
        print(f"AUTOCORR2D_3:             '{self.AUTOCORR2D_3 if self.AUTOCORR2D_3 is not None else '-' }'")
        print(f"AUTOCORR2D_4:             '{self.AUTOCORR2D_4 if self.AUTOCORR2D_4 is not None else '-' }'")
        print(f"AUTOCORR2D_5:             '{self.AUTOCORR2D_5 if self.AUTOCORR2D_5 is not None else '-' }'")
        print(f"AUTOCORR2D_6:             '{self.AUTOCORR2D_6 if self.AUTOCORR2D_6 is not None else '-' }'")
        print(f"AUTOCORR2D_7:             '{self.AUTOCORR2D_7 if self.AUTOCORR2D_7 is not None else '-' }'")
        print(f"AUTOCORR2D_8:             '{self.AUTOCORR2D_8 if self.AUTOCORR2D_8 is not None else '-' }'")
        print(f"AUTOCORR2D_9:             '{self.AUTOCORR2D_9 if self.AUTOCORR2D_9 is not None else '-' }'")
        print(f"AUTOCORR2D_10:            '{self.AUTOCORR2D_10 if self.AUTOCORR2D_10 is not None else '-' }'")
        print(f"AUTOCORR2D_11:            '{self.AUTOCORR2D_11 if self.AUTOCORR2D_11 is not None else '-' }'")
        print(f"AUTOCORR2D_12:            '{self.AUTOCORR2D_12 if self.AUTOCORR2D_12 is not None else '-' }'")
        print(f"AUTOCORR2D_13:            '{self.AUTOCORR2D_13 if self.AUTOCORR2D_13 is not None else '-' }'")
        print(f"AUTOCORR2D_14:            '{self.AUTOCORR2D_14 if self.AUTOCORR2D_14 is not None else '-' }'")
        print(f"AUTOCORR2D_15:            '{self.AUTOCORR2D_15 if self.AUTOCORR2D_15 is not None else '-' }'")
        print(f"AUTOCORR2D_16:            '{self.AUTOCORR2D_16 if self.AUTOCORR2D_16 is not None else '-' }'")
        print(f"AUTOCORR2D_17:            '{self.AUTOCORR2D_17 if self.AUTOCORR2D_17 is not None else '-' }'")
        print(f"AUTOCORR2D_18:            '{self.AUTOCORR2D_18 if self.AUTOCORR2D_18 is not None else '-' }'")
        print(f"AUTOCORR2D_19:            '{self.AUTOCORR2D_19 if self.AUTOCORR2D_19 is not None else '-' }'")
        print(f"AUTOCORR2D_20:            '{self.AUTOCORR2D_20 if self.AUTOCORR2D_20 is not None else '-' }'")
        print(f"AUTOCORR2D_21:            '{self.AUTOCORR2D_21 if self.AUTOCORR2D_21 is not None else '-' }'")
        print(f"AUTOCORR2D_22:            '{self.AUTOCORR2D_22 if self.AUTOCORR2D_22 is not None else '-' }'")
        print(f"AUTOCORR2D_23:            '{self.AUTOCORR2D_23 if self.AUTOCORR2D_23 is not None else '-' }'")
        print(f"AUTOCORR2D_24:            '{self.AUTOCORR2D_24 if self.AUTOCORR2D_24 is not None else '-' }'")
        print(f"AUTOCORR2D_25:            '{self.AUTOCORR2D_25 if self.AUTOCORR2D_25 is not None else '-' }'")
        print(f"AUTOCORR2D_26:            '{self.AUTOCORR2D_26 if self.AUTOCORR2D_26 is not None else '-' }'")
        print(f"AUTOCORR2D_27:            '{self.AUTOCORR2D_27 if self.AUTOCORR2D_27 is not None else '-' }'")
        print(f"AUTOCORR2D_28:            '{self.AUTOCORR2D_28 if self.AUTOCORR2D_28 is not None else '-' }'")
        print(f"AUTOCORR2D_29:            '{self.AUTOCORR2D_29 if self.AUTOCORR2D_29 is not None else '-' }'")
        print(f"AUTOCORR2D_30:            '{self.AUTOCORR2D_30 if self.AUTOCORR2D_30 is not None else '-' }'")
        print(f"AUTOCORR2D_31:            '{self.AUTOCORR2D_31 if self.AUTOCORR2D_31 is not None else '-' }'")
        print(f"AUTOCORR2D_32:            '{self.AUTOCORR2D_32 if self.AUTOCORR2D_32 is not None else '-' }'")
        print(f"AUTOCORR2D_33:            '{self.AUTOCORR2D_33 if self.AUTOCORR2D_33 is not None else '-' }'")
        print(f"AUTOCORR2D_34:            '{self.AUTOCORR2D_34 if self.AUTOCORR2D_34 is not None else '-' }'")
        print(f"AUTOCORR2D_35:            '{self.AUTOCORR2D_35 if self.AUTOCORR2D_35 is not None else '-' }'")
        print(f"AUTOCORR2D_36:            '{self.AUTOCORR2D_36 if self.AUTOCORR2D_36 is not None else '-' }'")
        print(f"AUTOCORR2D_37:            '{self.AUTOCORR2D_37 if self.AUTOCORR2D_37 is not None else '-' }'")
        print(f"AUTOCORR2D_38:            '{self.AUTOCORR2D_38 if self.AUTOCORR2D_38 is not None else '-' }'")
        print(f"AUTOCORR2D_39:            '{self.AUTOCORR2D_39 if self.AUTOCORR2D_39 is not None else '-' }'")
        print(f"AUTOCORR2D_40:            '{self.AUTOCORR2D_40 if self.AUTOCORR2D_40 is not None else '-' }'")
        print(f"AUTOCORR2D_41:            '{self.AUTOCORR2D_41 if self.AUTOCORR2D_41 is not None else '-' }'")
        print(f"AUTOCORR2D_42:            '{self.AUTOCORR2D_42 if self.AUTOCORR2D_42 is not None else '-' }'")
        print(f"AUTOCORR2D_43:            '{self.AUTOCORR2D_43 if self.AUTOCORR2D_43 is not None else '-' }'")
        print(f"AUTOCORR2D_44:            '{self.AUTOCORR2D_44 if self.AUTOCORR2D_44 is not None else '-' }'")
        print(f"AUTOCORR2D_45:            '{self.AUTOCORR2D_45 if self.AUTOCORR2D_45 is not None else '-' }'")
        print(f"AUTOCORR2D_46:            '{self.AUTOCORR2D_46 if self.AUTOCORR2D_46 is not None else '-' }'")
        print(f"AUTOCORR2D_47:            '{self.AUTOCORR2D_47 if self.AUTOCORR2D_47 is not None else '-' }'")
        print(f"AUTOCORR2D_48:            '{self.AUTOCORR2D_48 if self.AUTOCORR2D_48 is not None else '-' }'")
        print(f"AUTOCORR2D_49:            '{self.AUTOCORR2D_49 if self.AUTOCORR2D_49 is not None else '-' }'")
        print(f"AUTOCORR2D_50:            '{self.AUTOCORR2D_50 if self.AUTOCORR2D_50 is not None else '-' }'")
        print(f"AUTOCORR2D_51:            '{self.AUTOCORR2D_51 if self.AUTOCORR2D_51 is not None else '-' }'")
        print(f"AUTOCORR2D_52:            '{self.AUTOCORR2D_52 if self.AUTOCORR2D_52 is not None else '-' }'")
        print(f"AUTOCORR2D_53:            '{self.AUTOCORR2D_53 if self.AUTOCORR2D_53 is not None else '-' }'")
        print(f"AUTOCORR2D_54:            '{self.AUTOCORR2D_54 if self.AUTOCORR2D_54 is not None else '-' }'")
        print(f"AUTOCORR2D_55:            '{self.AUTOCORR2D_55 if self.AUTOCORR2D_55 is not None else '-' }'")
        print(f"AUTOCORR2D_56:            '{self.AUTOCORR2D_56 if self.AUTOCORR2D_56 is not None else '-' }'")
        print(f"AUTOCORR2D_57:            '{self.AUTOCORR2D_57 if self.AUTOCORR2D_57 is not None else '-' }'")
        print(f"AUTOCORR2D_58:            '{self.AUTOCORR2D_58 if self.AUTOCORR2D_58 is not None else '-' }'")
        print(f"AUTOCORR2D_59:            '{self.AUTOCORR2D_59 if self.AUTOCORR2D_59 is not None else '-' }'")
        print(f"AUTOCORR2D_60:            '{self.AUTOCORR2D_60 if self.AUTOCORR2D_60 is not None else '-' }'")
        print(f"AUTOCORR2D_61:            '{self.AUTOCORR2D_61 if self.AUTOCORR2D_61 is not None else '-' }'")
        print(f"AUTOCORR2D_62:            '{self.AUTOCORR2D_62 if self.AUTOCORR2D_62 is not None else '-' }'")
        print(f"AUTOCORR2D_63:            '{self.AUTOCORR2D_63 if self.AUTOCORR2D_63 is not None else '-' }'")
        print(f"AUTOCORR2D_64:            '{self.AUTOCORR2D_64 if self.AUTOCORR2D_64 is not None else '-' }'")
        print(f"AUTOCORR2D_65:            '{self.AUTOCORR2D_65 if self.AUTOCORR2D_65 is not None else '-' }'")
        print(f"AUTOCORR2D_66:            '{self.AUTOCORR2D_66 if self.AUTOCORR2D_66 is not None else '-' }'")
        print(f"AUTOCORR2D_67:            '{self.AUTOCORR2D_67 if self.AUTOCORR2D_67 is not None else '-' }'")
        print(f"AUTOCORR2D_68:            '{self.AUTOCORR2D_68 if self.AUTOCORR2D_68 is not None else '-' }'")
        print(f"AUTOCORR2D_69:            '{self.AUTOCORR2D_69 if self.AUTOCORR2D_69 is not None else '-' }'")
        print(f"AUTOCORR2D_70:            '{self.AUTOCORR2D_70 if self.AUTOCORR2D_70 is not None else '-' }'")
        print(f"AUTOCORR2D_71:            '{self.AUTOCORR2D_71 if self.AUTOCORR2D_71 is not None else '-' }'")
        print(f"AUTOCORR2D_72:            '{self.AUTOCORR2D_72 if self.AUTOCORR2D_72 is not None else '-' }'")
        print(f"AUTOCORR2D_73:            '{self.AUTOCORR2D_73 if self.AUTOCORR2D_73 is not None else '-' }'")
        print(f"AUTOCORR2D_74:            '{self.AUTOCORR2D_74 if self.AUTOCORR2D_74 is not None else '-' }'")
        print(f"AUTOCORR2D_75:            '{self.AUTOCORR2D_75 if self.AUTOCORR2D_75 is not None else '-' }'")
        print(f"AUTOCORR2D_76:            '{self.AUTOCORR2D_76 if self.AUTOCORR2D_76 is not None else '-' }'")
        print(f"AUTOCORR2D_77:            '{self.AUTOCORR2D_77 if self.AUTOCORR2D_77 is not None else '-' }'")
        print(f"AUTOCORR2D_78:            '{self.AUTOCORR2D_78 if self.AUTOCORR2D_78 is not None else '-' }'")
        print(f"AUTOCORR2D_79:            '{self.AUTOCORR2D_79 if self.AUTOCORR2D_79 is not None else '-' }'")
        print(f"AUTOCORR2D_80:            '{self.AUTOCORR2D_80 if self.AUTOCORR2D_80 is not None else '-' }'")
        print(f"AUTOCORR2D_81:            '{self.AUTOCORR2D_81 if self.AUTOCORR2D_81 is not None else '-' }'")
        print(f"AUTOCORR2D_82:            '{self.AUTOCORR2D_82 if self.AUTOCORR2D_82 is not None else '-' }'")
        print(f"AUTOCORR2D_83:            '{self.AUTOCORR2D_83 if self.AUTOCORR2D_83 is not None else '-' }'")
        print(f"AUTOCORR2D_84:            '{self.AUTOCORR2D_84 if self.AUTOCORR2D_84 is not None else '-' }'")
        print(f"AUTOCORR2D_85:            '{self.AUTOCORR2D_85 if self.AUTOCORR2D_85 is not None else '-' }'")
        print(f"AUTOCORR2D_86:            '{self.AUTOCORR2D_86 if self.AUTOCORR2D_86 is not None else '-' }'")
        print(f"AUTOCORR2D_87:            '{self.AUTOCORR2D_87 if self.AUTOCORR2D_87 is not None else '-' }'")
        print(f"AUTOCORR2D_88:            '{self.AUTOCORR2D_88 if self.AUTOCORR2D_88 is not None else '-' }'")
        print(f"AUTOCORR2D_89:            '{self.AUTOCORR2D_89 if self.AUTOCORR2D_89 is not None else '-' }'")
        print(f"AUTOCORR2D_90:            '{self.AUTOCORR2D_90 if self.AUTOCORR2D_90 is not None else '-' }'")
        print(f"AUTOCORR2D_91:            '{self.AUTOCORR2D_91 if self.AUTOCORR2D_91 is not None else '-' }'")
        print(f"AUTOCORR2D_92:            '{self.AUTOCORR2D_92 if self.AUTOCORR2D_92 is not None else '-' }'")
        print(f"AUTOCORR2D_93:            '{self.AUTOCORR2D_93 if self.AUTOCORR2D_93 is not None else '-' }'")
        print(f"AUTOCORR2D_94:            '{self.AUTOCORR2D_94 if self.AUTOCORR2D_94 is not None else '-' }'")
        print(f"AUTOCORR2D_95:            '{self.AUTOCORR2D_95 if self.AUTOCORR2D_95 is not None else '-' }'")
        print(f"AUTOCORR2D_96:            '{self.AUTOCORR2D_96 if self.AUTOCORR2D_96 is not None else '-' }'")
        print(f"AUTOCORR2D_97:            '{self.AUTOCORR2D_97 if self.AUTOCORR2D_97 is not None else '-' }'")
        print(f"AUTOCORR2D_98:            '{self.AUTOCORR2D_98 if self.AUTOCORR2D_98 is not None else '-' }'")
        print(f"AUTOCORR2D_99:            '{self.AUTOCORR2D_99 if self.AUTOCORR2D_99 is not None else '-' }'")
        print(f"AUTOCORR2D_100:           '{self.AUTOCORR2D_100 if self.AUTOCORR2D_100 is not None else '-' }'")
        print(f"AUTOCORR2D_101:           '{self.AUTOCORR2D_101 if self.AUTOCORR2D_101 is not None else '-' }'")
        print(f"AUTOCORR2D_102:           '{self.AUTOCORR2D_102 if self.AUTOCORR2D_102 is not None else '-' }'")
        print(f"AUTOCORR2D_103:           '{self.AUTOCORR2D_103 if self.AUTOCORR2D_103 is not None else '-' }'")
        print(f"AUTOCORR2D_104:           '{self.AUTOCORR2D_104 if self.AUTOCORR2D_104 is not None else '-' }'")
        print(f"AUTOCORR2D_105:           '{self.AUTOCORR2D_105 if self.AUTOCORR2D_105 is not None else '-' }'")
        print(f"AUTOCORR2D_106:           '{self.AUTOCORR2D_106 if self.AUTOCORR2D_106 is not None else '-' }'")
        print(f"AUTOCORR2D_107:           '{self.AUTOCORR2D_107 if self.AUTOCORR2D_107 is not None else '-' }'")
        print(f"AUTOCORR2D_108:           '{self.AUTOCORR2D_108 if self.AUTOCORR2D_108 is not None else '-' }'")
        print(f"AUTOCORR2D_109:           '{self.AUTOCORR2D_109 if self.AUTOCORR2D_109 is not None else '-' }'")
        print(f"AUTOCORR2D_110:           '{self.AUTOCORR2D_110 if self.AUTOCORR2D_110 is not None else '-' }'")
        print(f"AUTOCORR2D_111:           '{self.AUTOCORR2D_111 if self.AUTOCORR2D_111 is not None else '-' }'")
        print(f"AUTOCORR2D_112:           '{self.AUTOCORR2D_112 if self.AUTOCORR2D_112 is not None else '-' }'")
        print(f"AUTOCORR2D_113:           '{self.AUTOCORR2D_113 if self.AUTOCORR2D_113 is not None else '-' }'")
        print(f"AUTOCORR2D_114:           '{self.AUTOCORR2D_114 if self.AUTOCORR2D_114 is not None else '-' }'")
        print(f"AUTOCORR2D_115:           '{self.AUTOCORR2D_115 if self.AUTOCORR2D_115 is not None else '-' }'")
        print(f"AUTOCORR2D_116:           '{self.AUTOCORR2D_116 if self.AUTOCORR2D_116 is not None else '-' }'")
        print(f"AUTOCORR2D_117:           '{self.AUTOCORR2D_117 if self.AUTOCORR2D_117 is not None else '-' }'")
        print(f"AUTOCORR2D_118:           '{self.AUTOCORR2D_118 if self.AUTOCORR2D_118 is not None else '-' }'")
        print(f"AUTOCORR2D_119:           '{self.AUTOCORR2D_119 if self.AUTOCORR2D_119 is not None else '-' }'")
        print(f"AUTOCORR2D_120:           '{self.AUTOCORR2D_120 if self.AUTOCORR2D_120 is not None else '-' }'")
        print(f"AUTOCORR2D_121:           '{self.AUTOCORR2D_121 if self.AUTOCORR2D_121 is not None else '-' }'")
        print(f"AUTOCORR2D_122:           '{self.AUTOCORR2D_122 if self.AUTOCORR2D_122 is not None else '-' }'")
        print(f"AUTOCORR2D_123:           '{self.AUTOCORR2D_123 if self.AUTOCORR2D_123 is not None else '-' }'")
        print(f"AUTOCORR2D_124:           '{self.AUTOCORR2D_124 if self.AUTOCORR2D_124 is not None else '-' }'")
        print(f"AUTOCORR2D_125:           '{self.AUTOCORR2D_125 if self.AUTOCORR2D_125 is not None else '-' }'")
        print(f"AUTOCORR2D_126:           '{self.AUTOCORR2D_126 if self.AUTOCORR2D_126 is not None else '-' }'")
        print(f"AUTOCORR2D_127:           '{self.AUTOCORR2D_127 if self.AUTOCORR2D_127 is not None else '-' }'")
        print(f"AUTOCORR2D_128:           '{self.AUTOCORR2D_128 if self.AUTOCORR2D_128 is not None else '-' }'")
        print(f"AUTOCORR2D_129:           '{self.AUTOCORR2D_129 if self.AUTOCORR2D_129 is not None else '-' }'")
        print(f"AUTOCORR2D_130:           '{self.AUTOCORR2D_130 if self.AUTOCORR2D_130 is not None else '-' }'")
        print(f"AUTOCORR2D_131:           '{self.AUTOCORR2D_131 if self.AUTOCORR2D_131 is not None else '-' }'")
        print(f"AUTOCORR2D_132:           '{self.AUTOCORR2D_132 if self.AUTOCORR2D_132 is not None else '-' }'")
        print(f"AUTOCORR2D_133:           '{self.AUTOCORR2D_133 if self.AUTOCORR2D_133 is not None else '-' }'")
        print(f"AUTOCORR2D_134:           '{self.AUTOCORR2D_134 if self.AUTOCORR2D_134 is not None else '-' }'")
        print(f"AUTOCORR2D_135:           '{self.AUTOCORR2D_135 if self.AUTOCORR2D_135 is not None else '-' }'")
        print(f"AUTOCORR2D_136:           '{self.AUTOCORR2D_136 if self.AUTOCORR2D_136 is not None else '-' }'")
        print(f"AUTOCORR2D_137:           '{self.AUTOCORR2D_137 if self.AUTOCORR2D_137 is not None else '-' }'")
        print(f"AUTOCORR2D_138:           '{self.AUTOCORR2D_138 if self.AUTOCORR2D_138 is not None else '-' }'")
        print(f"AUTOCORR2D_139:           '{self.AUTOCORR2D_139 if self.AUTOCORR2D_139 is not None else '-' }'")
        print(f"AUTOCORR2D_140:           '{self.AUTOCORR2D_140 if self.AUTOCORR2D_140 is not None else '-' }'")
        print(f"AUTOCORR2D_141:           '{self.AUTOCORR2D_141 if self.AUTOCORR2D_141 is not None else '-' }'")
        print(f"AUTOCORR2D_142:           '{self.AUTOCORR2D_142 if self.AUTOCORR2D_142 is not None else '-' }'")
        print(f"AUTOCORR2D_143:           '{self.AUTOCORR2D_143 if self.AUTOCORR2D_143 is not None else '-' }'")
        print(f"AUTOCORR2D_144:           '{self.AUTOCORR2D_144 if self.AUTOCORR2D_144 is not None else '-' }'")
        print(f"AUTOCORR2D_145:           '{self.AUTOCORR2D_145 if self.AUTOCORR2D_145 is not None else '-' }'")
        print(f"AUTOCORR2D_146:           '{self.AUTOCORR2D_146 if self.AUTOCORR2D_146 is not None else '-' }'")
        print(f"AUTOCORR2D_147:           '{self.AUTOCORR2D_147 if self.AUTOCORR2D_147 is not None else '-' }'")
        print(f"AUTOCORR2D_148:           '{self.AUTOCORR2D_148 if self.AUTOCORR2D_148 is not None else '-' }'")
        print(f"AUTOCORR2D_149:           '{self.AUTOCORR2D_149 if self.AUTOCORR2D_149 is not None else '-' }'")
        print(f"AUTOCORR2D_150:           '{self.AUTOCORR2D_150 if self.AUTOCORR2D_150 is not None else '-' }'")
        print(f"AUTOCORR2D_151:           '{self.AUTOCORR2D_151 if self.AUTOCORR2D_151 is not None else '-' }'")
        print(f"AUTOCORR2D_152:           '{self.AUTOCORR2D_152 if self.AUTOCORR2D_152 is not None else '-' }'")
        print(f"AUTOCORR2D_153:           '{self.AUTOCORR2D_153 if self.AUTOCORR2D_153 is not None else '-' }'")
        print(f"AUTOCORR2D_154:           '{self.AUTOCORR2D_154 if self.AUTOCORR2D_154 is not None else '-' }'")
        print(f"AUTOCORR2D_155:           '{self.AUTOCORR2D_155 if self.AUTOCORR2D_155 is not None else '-' }'")
        print(f"AUTOCORR2D_156:           '{self.AUTOCORR2D_156 if self.AUTOCORR2D_156 is not None else '-' }'")
        print(f"AUTOCORR2D_157:           '{self.AUTOCORR2D_157 if self.AUTOCORR2D_157 is not None else '-' }'")
        print(f"AUTOCORR2D_158:           '{self.AUTOCORR2D_158 if self.AUTOCORR2D_158 is not None else '-' }'")
        print(f"AUTOCORR2D_159:           '{self.AUTOCORR2D_159 if self.AUTOCORR2D_159 is not None else '-' }'")
        print(f"AUTOCORR2D_160:           '{self.AUTOCORR2D_160 if self.AUTOCORR2D_160 is not None else '-' }'")
        print(f"AUTOCORR2D_161:           '{self.AUTOCORR2D_161 if self.AUTOCORR2D_161 is not None else '-' }'")
        print(f"AUTOCORR2D_162:           '{self.AUTOCORR2D_162 if self.AUTOCORR2D_162 is not None else '-' }'")
        print(f"AUTOCORR2D_163:           '{self.AUTOCORR2D_163 if self.AUTOCORR2D_163 is not None else '-' }'")
        print(f"AUTOCORR2D_164:           '{self.AUTOCORR2D_164 if self.AUTOCORR2D_164 is not None else '-' }'")
        print(f"AUTOCORR2D_165:           '{self.AUTOCORR2D_165 if self.AUTOCORR2D_165 is not None else '-' }'")
        print(f"AUTOCORR2D_166:           '{self.AUTOCORR2D_166 if self.AUTOCORR2D_166 is not None else '-' }'")
        print(f"AUTOCORR2D_167:           '{self.AUTOCORR2D_167 if self.AUTOCORR2D_167 is not None else '-' }'")
        print(f"AUTOCORR2D_168:           '{self.AUTOCORR2D_168 if self.AUTOCORR2D_168 is not None else '-' }'")
        print(f"AUTOCORR2D_169:           '{self.AUTOCORR2D_169 if self.AUTOCORR2D_169 is not None else '-' }'")
        print(f"AUTOCORR2D_170:           '{self.AUTOCORR2D_170 if self.AUTOCORR2D_170 is not None else '-' }'")
        print(f"AUTOCORR2D_171:           '{self.AUTOCORR2D_171 if self.AUTOCORR2D_171 is not None else '-' }'")
        print(f"AUTOCORR2D_172:           '{self.AUTOCORR2D_172 if self.AUTOCORR2D_172 is not None else '-' }'")
        print(f"AUTOCORR2D_173:           '{self.AUTOCORR2D_173 if self.AUTOCORR2D_173 is not None else '-' }'")
        print(f"AUTOCORR2D_174:           '{self.AUTOCORR2D_174 if self.AUTOCORR2D_174 is not None else '-' }'")
        print(f"AUTOCORR2D_175:           '{self.AUTOCORR2D_175 if self.AUTOCORR2D_175 is not None else '-' }'")
        print(f"AUTOCORR2D_176:           '{self.AUTOCORR2D_176 if self.AUTOCORR2D_176 is not None else '-' }'")
        print(f"AUTOCORR2D_177:           '{self.AUTOCORR2D_177 if self.AUTOCORR2D_177 is not None else '-' }'")
        print(f"AUTOCORR2D_178:           '{self.AUTOCORR2D_178 if self.AUTOCORR2D_178 is not None else '-' }'")
        print(f"AUTOCORR2D_179:           '{self.AUTOCORR2D_179 if self.AUTOCORR2D_179 is not None else '-' }'")
        print(f"AUTOCORR2D_180:           '{self.AUTOCORR2D_180 if self.AUTOCORR2D_180 is not None else '-' }'")
        print(f"AUTOCORR2D_181:           '{self.AUTOCORR2D_181 if self.AUTOCORR2D_181 is not None else '-' }'")
        print(f"AUTOCORR2D_182:           '{self.AUTOCORR2D_182 if self.AUTOCORR2D_182 is not None else '-' }'")
        print(f"AUTOCORR2D_183:           '{self.AUTOCORR2D_183 if self.AUTOCORR2D_183 is not None else '-' }'")
        print(f"AUTOCORR2D_184:           '{self.AUTOCORR2D_184 if self.AUTOCORR2D_184 is not None else '-' }'")
        print(f"AUTOCORR2D_185:           '{self.AUTOCORR2D_185 if self.AUTOCORR2D_185 is not None else '-' }'")
        print(f"AUTOCORR2D_186:           '{self.AUTOCORR2D_186 if self.AUTOCORR2D_186 is not None else '-' }'")
        print(f"AUTOCORR2D_187:           '{self.AUTOCORR2D_187 if self.AUTOCORR2D_187 is not None else '-' }'")
        print(f"AUTOCORR2D_188:           '{self.AUTOCORR2D_188 if self.AUTOCORR2D_188 is not None else '-' }'")
        print(f"AUTOCORR2D_189:           '{self.AUTOCORR2D_189 if self.AUTOCORR2D_189 is not None else '-' }'")
        print(f"AUTOCORR2D_190:           '{self.AUTOCORR2D_190 if self.AUTOCORR2D_190 is not None else '-' }'")
        print(f"AUTOCORR2D_191:           '{self.AUTOCORR2D_191 if self.AUTOCORR2D_191 is not None else '-' }'")
        print(f"AUTOCORR2D_192:           '{self.AUTOCORR2D_192 if self.AUTOCORR2D_192 is not None else '-' }'")
        print(f"BCUT2D_CHGHI:             '{self.BCUT2D_CHGHI if self.BCUT2D_CHGHI is not None else '-' }'")
        print(f"BCUT2D_CHGLO:             '{self.BCUT2D_CHGLO if self.BCUT2D_CHGLO is not None else '-' }'")
        print(f"BCUT2D_LOGPHI:            '{self.BCUT2D_LOGPHI if self.BCUT2D_LOGPHI is not None else '-' }'")
        print(f"BCUT2D_LOGPLOW:           '{self.BCUT2D_LOGPLOW if self.BCUT2D_LOGPLOW is not None else '-' }'")
        print(f"BCUT2D_MRHI:              '{self.BCUT2D_MRHI if self.BCUT2D_MRHI is not None else '-' }'")
        print(f"BCUT2D_MRLOW:             '{self.BCUT2D_MRLOW if self.BCUT2D_MRLOW is not None else '-' }'")
        print(f"BCUT2D_MWHI:              '{self.BCUT2D_MWHI if self.BCUT2D_MWHI is not None else '-' }'")
        print(f"BCUT2D_MWLOW:             '{self.BCUT2D_MWLOW if self.BCUT2D_MWLOW is not None else '-' }'")
        print(f"BalabanJ:                 '{self.BalabanJ if self.BalabanJ is not None else '-' }'")
        print(f"BertzCT:                  '{self.BertzCT if self.BertzCT is not None else '-' }'")
        print(f"Chi0:                     '{self.Chi0 if self.Chi0 is not None else '-' }'")
        print(f"Chi0n:                    '{self.Chi0n if self.Chi0n is not None else '-' }'")
        print(f"Chi0v:                    '{self.Chi0v if self.Chi0v is not None else '-' }'")
        print(f"Chi1:                     '{self.Chi1 if self.Chi1 is not None else '-' }'")
        print(f"Chi1n:                    '{self.Chi1n if self.Chi1n is not None else '-' }'")
        print(f"Chi1v:                    '{self.Chi1v if self.Chi1v is not None else '-' }'")
        print(f"Chi2n:                    '{self.Chi2n if self.Chi2n is not None else '-' }'")
        print(f"Chi2v:                    '{self.Chi2v if self.Chi2v is not None else '-' }'")
        print(f"Chi3n:                    '{self.Chi3n if self.Chi3n is not None else '-' }'")
        print(f"Chi3v:                    '{self.Chi3v if self.Chi3v is not None else '-' }'")
        print(f"Chi4n:                    '{self.Chi4n if self.Chi4n is not None else '-' }'")
        print(f"Chi4v:                    '{self.Chi4v if self.Chi4v is not None else '-' }'")
        print(f"EState_VSA1:              '{self.EState_VSA1 if self.EState_VSA1 is not None else '-' }'")
        print(f"EState_VSA2:              '{self.EState_VSA2 if self.EState_VSA2 is not None else '-' }'")
        print(f"EState_VSA3:              '{self.EState_VSA3 if self.EState_VSA3 is not None else '-' }'")
        print(f"EState_VSA4:              '{self.EState_VSA4 if self.EState_VSA4 is not None else '-' }'")
        print(f"EState_VSA5:              '{self.EState_VSA5 if self.EState_VSA5 is not None else '-' }'")
        print(f"EState_VSA6:              '{self.EState_VSA6 if self.EState_VSA6 is not None else '-' }'")
        print(f"EState_VSA7:              '{self.EState_VSA7 if self.EState_VSA7 is not None else '-' }'")
        print(f"EState_VSA8:              '{self.EState_VSA8 if self.EState_VSA8 is not None else '-' }'")
        print(f"EState_VSA9:              '{self.EState_VSA9 if self.EState_VSA9 is not None else '-' }'")
        print(f"EState_VSA10:             '{self.EState_VSA10 if self.EState_VSA10 is not None else '-' }'")
        print(f"EState_VSA11:             '{self.EState_VSA11 if self.EState_VSA11 is not None else '-' }'")
        print(f"MaxAbsEStateIndex:        '{self.MaxAbsEStateIndex if self.MaxAbsEStateIndex is not None else '-' }'")
        print(f"MaxEStateIndex:           '{self.MaxEStateIndex if self.MaxEStateIndex is not None else '-' }'")
        print(f"MinAbsEStateIndex:        '{self.MinAbsEStateIndex if self.MinAbsEStateIndex is not None else '-' }'")
        print(f"MinEStateIndex:           '{self.MinEStateIndex if self.MinEStateIndex is not None else '-' }'")
        print(f"ExactMolWt:               '{self.ExactMolWt if self.ExactMolWt is not None else '-' }'")
        print(f"FpDensityMorgan1:         '{self.FpDensityMorgan1 if self.FpDensityMorgan1 is not None else '-' }'")
        print(f"FpDensityMorgan2:         '{self.FpDensityMorgan2 if self.FpDensityMorgan2 is not None else '-' }'")
        print(f"FpDensityMorgan3:         '{self.FpDensityMorgan3 if self.FpDensityMorgan3 is not None else '-' }'")
        print(f"fr_Al_COO:                '{self.fr_Al_COO if self.fr_Al_COO is not None else '-' }'")
        print(f"fr_Al_OH:                 '{self.fr_Al_OH if self.fr_Al_OH is not None else '-' }'")
        print(f"fr_Al_OH_noTert:          '{self.fr_Al_OH_noTert if self.fr_Al_OH_noTert is not None else '-' }'")
        print(f"fr_ArN:                   '{self.fr_ArN if self.fr_ArN is not None else '-' }'")
        print(f"fr_Ar_COO:                '{self.fr_Ar_COO if self.fr_Ar_COO is not None else '-' }'")
        print(f"fr_Ar_N:                  '{self.fr_Ar_N if self.fr_Ar_N is not None else '-' }'")
        print(f"fr_Ar_NH:                 '{self.fr_Ar_NH if self.fr_Ar_NH is not None else '-' }'")
        print(f"fr_Ar_OH:                 '{self.fr_Ar_OH if self.fr_Ar_OH is not None else '-' }'")
        print(f"fr_COO:                   '{self.fr_COO if self.fr_COO is not None else '-' }'")
        print(f"fr_COO2:                  '{self.fr_COO2 if self.fr_COO2 is not None else '-' }'")
        print(f"fr_C_O:                   '{self.fr_C_O if self.fr_C_O is not None else '-' }'")
        print(f"fr_C_O_noCOO:             '{self.fr_C_O_noCOO if self.fr_C_O_noCOO is not None else '-' }'")
        print(f"fr_C_S:                   '{self.fr_C_S if self.fr_C_S is not None else '-' }'")
        print(f"fr_HOCCN:                 '{self.fr_HOCCN if self.fr_HOCCN is not None else '-' }'")
        print(f"fr_Imine:                 '{self.fr_Imine if self.fr_Imine is not None else '-' }'")
        print(f"fr_NH0:                   '{self.fr_NH0 if self.fr_NH0 is not None else '-' }'")
        print(f"fr_NH1:                   '{self.fr_NH1 if self.fr_NH1 is not None else '-' }'")
        print(f"fr_NH2:                   '{self.fr_NH2 if self.fr_NH2 is not None else '-' }'")
        print(f"fr_N_O:                   '{self.fr_N_O if self.fr_N_O is not None else '-' }'")
        print(f"fr_Ndealkylation1:        '{self.fr_Ndealkylation1 if self.fr_Ndealkylation1 is not None else '-' }'")
        print(f"fr_Ndealkylation2:        '{self.fr_Ndealkylation2 if self.fr_Ndealkylation2 is not None else '-' }'")
        print(f"fr_Nhpyrrole:             '{self.fr_Nhpyrrole if self.fr_Nhpyrrole is not None else '-' }'")
        print(f"fr_SH:                    '{self.fr_SH if self.fr_SH is not None else '-' }'")
        print(f"fr_aldehyde:              '{self.fr_aldehyde if self.fr_aldehyde is not None else '-' }'")
        print(f"fr_alkyl_carbamate:       '{self.fr_alkyl_carbamate if self.fr_alkyl_carbamate is not None else '-' }'")
        print(f"fr_alkyl_halide:          '{self.fr_alkyl_halide if self.fr_alkyl_halide is not None else '-' }'")
        print(f"fr_allylic_oxid:          '{self.fr_allylic_oxid if self.fr_allylic_oxid is not None else '-' }'")
        print(f"fr_amide:                 '{self.fr_amide if self.fr_amide is not None else '-' }'")
        print(f"fr_amidine:               '{self.fr_amidine if self.fr_amidine is not None else '-' }'")
        print(f"fr_aniline:               '{self.fr_aniline if self.fr_aniline is not None else '-' }'")
        print(f"fr_aryl_methyl:           '{self.fr_aryl_methyl if self.fr_aryl_methyl is not None else '-' }'")
        print(f"fr_azide:                 '{self.fr_azide if self.fr_azide is not None else '-' }'")
        print(f"fr_azo:                   '{self.fr_azo if self.fr_azo is not None else '-' }'")
        print(f"fr_barbitur:              '{self.fr_barbitur if self.fr_barbitur is not None else '-' }'")
        print(f"fr_benzene:               '{self.fr_benzene if self.fr_benzene is not None else '-' }'")
        print(f"fr_benzodiazepine:        '{self.fr_benzodiazepine if self.fr_benzodiazepine is not None else '-' }'")
        print(f"fr_bicyclic:              '{self.fr_bicyclic if self.fr_bicyclic is not None else '-' }'")
        print(f"fr_diazo:                 '{self.fr_diazo if self.fr_diazo is not None else '-' }'")
        print(f"fr_dihydropyridine:       '{self.fr_dihydropyridine if self.fr_dihydropyridine is not None else '-' }'")
        print(f"fr_epoxide:               '{self.fr_epoxide if self.fr_epoxide is not None else '-' }'")
        print(f"fr_ester:                 '{self.fr_ester if self.fr_ester is not None else '-' }'")
        print(f"fr_ether:                 '{self.fr_ether if self.fr_ether is not None else '-' }'")
        print(f"fr_furan:                 '{self.fr_furan if self.fr_furan is not None else '-' }'")
        print(f"fr_guanido:               '{self.fr_guanido if self.fr_guanido is not None else '-' }'")
        print(f"fr_halogen:               '{self.fr_halogen if self.fr_halogen is not None else '-' }'")
        print(f"fr_hdrzine:               '{self.fr_hdrzine if self.fr_hdrzine is not None else '-' }'")
        print(f"fr_hdrzone:               '{self.fr_hdrzone if self.fr_hdrzone is not None else '-' }'")
        print(f"fr_imidazole:             '{self.fr_imidazole if self.fr_imidazole is not None else '-' }'")
        print(f"fr_imide:                 '{self.fr_imide if self.fr_imide is not None else '-' }'")
        print(f"fr_isocyan:               '{self.fr_isocyan if self.fr_isocyan is not None else '-' }'")
        print(f"fr_isothiocyan:           '{self.fr_isothiocyan if self.fr_isothiocyan is not None else '-' }'")
        print(f"fr_ketone:                '{self.fr_ketone if self.fr_ketone is not None else '-' }'")
        print(f"fr_ketone_Topliss:        '{self.fr_ketone_Topliss if self.fr_ketone_Topliss is not None else '-' }'")
        print(f"fr_lactam:                '{self.fr_lactam if self.fr_lactam is not None else '-' }'")
        print(f"fr_lactone:               '{self.fr_lactone if self.fr_lactone is not None else '-' }'")
        print(f"fr_methoxy:               '{self.fr_methoxy if self.fr_methoxy is not None else '-' }'")
        print(f"fr_morpholine:            '{self.fr_morpholine if self.fr_morpholine is not None else '-' }'")
        print(f"fr_nitrile:               '{self.fr_nitrile if self.fr_nitrile is not None else '-' }'")
        print(f"fr_nitro:                 '{self.fr_nitro if self.fr_nitro is not None else '-' }'")
        print(f"fr_nitro_arom:            '{self.fr_nitro_arom if self.fr_nitro_arom is not None else '-' }'")
        print(f"fr_nitro_arom_nonortho:   '{self.fr_nitro_arom_nonortho if self.fr_nitro_arom_nonortho is not None else '-' }'")
        print(f"fr_nitroso:               '{self.fr_nitroso if self.fr_nitroso is not None else '-' }'")
        print(f"fr_oxazole:               '{self.fr_oxazole if self.fr_oxazole is not None else '-' }'")
        print(f"fr_oxime:                 '{self.fr_oxime if self.fr_oxime is not None else '-' }'")
        print(f"fr_para_hydroxylation:    '{self.fr_para_hydroxylation if self.fr_para_hydroxylation is not None else '-' }'")
        print(f"fr_phenol:                '{self.fr_phenol if self.fr_phenol is not None else '-' }'")
        print(f"fr_phenol_noOrthoHbond:   '{self.fr_phenol_noOrthoHbond if self.fr_phenol_noOrthoHbond is not None else '-' }'")
        print(f"fr_phos_acid:             '{self.fr_phos_acid if self.fr_phos_acid is not None else '-' }'")
        print(f"fr_phos_ester:            '{self.fr_phos_ester if self.fr_phos_ester is not None else '-' }'")
        print(f"fr_piperdine:             '{self.fr_piperdine if self.fr_piperdine is not None else '-' }'")
        print(f"fr_piperzine:             '{self.fr_piperzine if self.fr_piperzine is not None else '-' }'")
        print(f"fr_priamide:              '{self.fr_priamide if self.fr_priamide is not None else '-' }'")
        print(f"fr_prisulfonamd:          '{self.fr_prisulfonamd if self.fr_prisulfonamd is not None else '-' }'")
        print(f"fr_pyridine:              '{self.fr_pyridine if self.fr_pyridine is not None else '-' }'")
        print(f"fr_quatN:                 '{self.fr_quatN if self.fr_quatN is not None else '-' }'")
        print(f"fr_sulfide:               '{self.fr_sulfide if self.fr_sulfide is not None else '-' }'")
        print(f"fr_sulfonamd:             '{self.fr_sulfonamd if self.fr_sulfonamd is not None else '-' }'")
        print(f"fr_sulfone:               '{self.fr_sulfone if self.fr_sulfone is not None else '-' }'")
        print(f"fr_term_acetylene:        '{self.fr_term_acetylene if self.fr_term_acetylene is not None else '-' }'")
        print(f"fr_tetrazole:             '{self.fr_tetrazole if self.fr_tetrazole is not None else '-' }'")
        print(f"fr_thiazole:              '{self.fr_thiazole if self.fr_thiazole is not None else '-' }'")
        print(f"fr_thiocyan:              '{self.fr_thiocyan if self.fr_thiocyan is not None else '-' }'")
        print(f"fr_thiophene:             '{self.fr_thiophene if self.fr_thiophene is not None else '-' }'")
        print(f"fr_unbrch_alkane:         '{self.fr_unbrch_alkane if self.fr_unbrch_alkane is not None else '-' }'")
        print(f"fr_urea:                  '{self.fr_urea if self.fr_urea is not None else '-' }'")
        print(f"FractionCSP3:             '{self.FractionCSP3 if self.FractionCSP3 is not None else '-' }'")
        print(f"HallKierAlpha:            '{self.HallKierAlpha if self.HallKierAlpha is not None else '-' }'")
        print(f"HeavyAtomMolWt:           '{self.HeavyAtomMolWt if self.HeavyAtomMolWt is not None else '-' }'")
        print(f"HeavyAtomCount:           '{self.HeavyAtomCount if self.HeavyAtomCount is not None else '-' }'")
        print(f"Ipc:                      '{self.Ipc if self.Ipc is not None else '-' }'")
        print(f"Kappa1:                   '{self.Kappa1 if self.Kappa1 is not None else '-' }'")
        print(f"Kappa2:                   '{self.Kappa2 if self.Kappa2 is not None else '-' }'")
        print(f"Kappa3:                   '{self.Kappa3 if self.Kappa3 is not None else '-' }'")
        print(f"LabuteASA:                '{self.LabuteASA if self.LabuteASA is not None else '-' }'")
        print(f"MaxAbsPartialCharge:      '{self.MaxAbsPartialCharge if self.MaxAbsPartialCharge is not None else '-' }'")
        print(f"MaxPartialCharge:         '{self.MaxPartialCharge if self.MaxPartialCharge is not None else '-' }'")
        print(f"MinAbsPartialCharge:      '{self.MinAbsPartialCharge if self.MinAbsPartialCharge is not None else '-' }'")
        print(f"MinPartialCharge:         '{self.MinPartialCharge if self.MinPartialCharge is not None else '-' }'")
        print(f"MolLogP:                  '{self.MolLogP if self.MolLogP is not None else '-' }'")
        print(f"MolMR:                    '{self.MolMR if self.MolMR is not None else '-' }'")
        print(f"MolWt:                    '{self.MolWt if self.MolWt is not None else '-' }'")
        print(f"NHOHCount:                '{self.NHOHCount if self.NHOHCount is not None else '-' }'")
        print(f"NOCount:                  '{self.NOCount if self.NOCount is not None else '-' }'")
        print(f"NumAliphaticCarbocycles:  '{self.NumAliphaticCarbocycles if self.NumAliphaticCarbocycles is not None else '-' }'")
        print(f"NumAliphaticHeterocycles: '{self.NumAliphaticHeterocycles if self.NumAliphaticHeterocycles is not None else '-' }'")
        print(f"NumAliphaticRings:        '{self.NumAliphaticRings if self.NumAliphaticRings is not None else '-' }'")
        print(f"NumAromaticCarbocycles:   '{self.NumAromaticCarbocycles if self.NumAromaticCarbocycles is not None else '-' }'")
        print(f"NumAromaticHeterocycles:  '{self.NumAromaticHeterocycles if self.NumAromaticHeterocycles is not None else '-' }'")
        print(f"NumAromaticRings:         '{self.NumAromaticRings if self.NumAromaticRings is not None else '-' }'")
        print(f"NumHAcceptors:            '{self.NumHAcceptors if self.NumHAcceptors is not None else '-' }'")
        print(f"NumHDonors:               '{self.NumHDonors if self.NumHDonors is not None else '-' }'")
        print(f"NumHeteroatoms:           '{self.NumHeteroatoms if self.NumHeteroatoms is not None else '-' }'")
        print(f"NumRadicalElectrons:      '{self.NumRadicalElectrons if self.NumRadicalElectrons is not None else '-' }'")
        print(f"NumRotatableBonds:        '{self.NumRotatableBonds if self.NumRotatableBonds is not None else '-' }'")
        print(f"NumSaturatedCarbocycles:  '{self.NumSaturatedCarbocycles if self.NumSaturatedCarbocycles is not None else '-' }'")
        print(f"NumSaturatedHeterocycles: '{self.NumSaturatedHeterocycles if self.NumSaturatedHeterocycles is not None else '-' }'")
        print(f"NumSaturatedRings:        '{self.NumSaturatedRings if self.NumSaturatedRings is not None else '-' }'")
        print(f"NumValenceElectrons:      '{self.NumValenceElectrons if self.NumValenceElectrons is not None else '-' }'")
        print(f"PEOE_VSA1:                '{self.PEOE_VSA1 if self.PEOE_VSA1 is not None else '-' }'")
        print(f"PEOE_VSA2:                '{self.PEOE_VSA2 if self.PEOE_VSA2 is not None else '-' }'")
        print(f"PEOE_VSA3:                '{self.PEOE_VSA3 if self.PEOE_VSA3 is not None else '-' }'")
        print(f"PEOE_VSA4:                '{self.PEOE_VSA4 if self.PEOE_VSA4 is not None else '-' }'")
        print(f"PEOE_VSA5:                '{self.PEOE_VSA5 if self.PEOE_VSA5 is not None else '-' }'")
        print(f"PEOE_VSA6:                '{self.PEOE_VSA6 if self.PEOE_VSA6 is not None else '-' }'")
        print(f"PEOE_VSA7:                '{self.PEOE_VSA7 if self.PEOE_VSA7 is not None else '-' }'")
        print(f"PEOE_VSA8:                '{self.PEOE_VSA8 if self.PEOE_VSA8 is not None else '-' }'")
        print(f"PEOE_VSA9:                '{self.PEOE_VSA9 if self.PEOE_VSA9 is not None else '-' }'")
        print(f"PEOE_VSA10:               '{self.PEOE_VSA10 if self.PEOE_VSA10 is not None else '-' }'")
        print(f"PEOE_VSA11:               '{self.PEOE_VSA11 if self.PEOE_VSA11 is not None else '-' }'")
        print(f"PEOE_VSA12:               '{self.PEOE_VSA12 if self.PEOE_VSA12 is not None else '-' }'")
        print(f"PEOE_VSA13:               '{self.PEOE_VSA13 if self.PEOE_VSA13 is not None else '-' }'")
        print(f"PEOE_VSA14:               '{self.PEOE_VSA14 if self.PEOE_VSA14 is not None else '-' }'")
        print(f"qed:                      '{self.qed if self.qed is not None else '-' }'")
        print(f"RingCount:                '{self.RingCount if self.RingCount is not None else '-' }'")
        print(f"SMR_VSA1:                 '{self.SMR_VSA1 if self.SMR_VSA1 is not None else '-' }'")
        print(f"SMR_VSA2:                 '{self.SMR_VSA2 if self.SMR_VSA2 is not None else '-' }'")
        print(f"SMR_VSA3:                 '{self.SMR_VSA3 if self.SMR_VSA3 is not None else '-' }'")
        print(f"SMR_VSA4:                 '{self.SMR_VSA4 if self.SMR_VSA4 is not None else '-' }'")
        print(f"SMR_VSA5:                 '{self.SMR_VSA5 if self.SMR_VSA5 is not None else '-' }'")
        print(f"SMR_VSA6:                 '{self.SMR_VSA6 if self.SMR_VSA6 is not None else '-' }'")
        print(f"SMR_VSA7:                 '{self.SMR_VSA7 if self.SMR_VSA7 is not None else '-' }'")
        print(f"SMR_VSA8:                 '{self.SMR_VSA8 if self.SMR_VSA8 is not None else '-' }'")
        print(f"SMR_VSA9:                 '{self.SMR_VSA9 if self.SMR_VSA9 is not None else '-' }'")
        print(f"SMR_VSA10:                '{self.SMR_VSA10 if self.SMR_VSA10 is not None else '-' }'")
        print(f"SlogP_VSA1:               '{self.SlogP_VSA1 if self.SlogP_VSA1 is not None else '-' }'")
        print(f"SlogP_VSA2:               '{self.SlogP_VSA2 if self.SlogP_VSA2 is not None else '-' }'")
        print(f"SlogP_VSA3:               '{self.SlogP_VSA3 if self.SlogP_VSA3 is not None else '-' }'")
        print(f"SlogP_VSA4:               '{self.SlogP_VSA4 if self.SlogP_VSA4 is not None else '-' }'")
        print(f"SlogP_VSA5:               '{self.SlogP_VSA5 if self.SlogP_VSA5 is not None else '-' }'")
        print(f"SlogP_VSA6:               '{self.SlogP_VSA6 if self.SlogP_VSA6 is not None else '-' }'")
        print(f"SlogP_VSA7:               '{self.SlogP_VSA7 if self.SlogP_VSA7 is not None else '-' }'")
        print(f"SlogP_VSA8:               '{self.SlogP_VSA8 if self.SlogP_VSA8 is not None else '-' }'")
        print(f"SlogP_VSA9:               '{self.SlogP_VSA9 if self.SlogP_VSA9 is not None else '-' }'")
        print(f"SlogP_VSA10:              '{self.SlogP_VSA10 if self.SlogP_VSA10 is not None else '-' }'")
        print(f"SlogP_VSA11:              '{self.SlogP_VSA11 if self.SlogP_VSA11 is not None else '-' }'")
        print(f"SlogP_VSA12:              '{self.SlogP_VSA12 if self.SlogP_VSA12 is not None else '-' }'")
        print(f"TPSA:                     '{self.TPSA if self.TPSA is not None else '-' }'")
        print(f"VSA_EState1:              '{self.VSA_EState1 if self.VSA_EState1 is not None else '-' }'")
        print(f"VSA_EState2:              '{self.VSA_EState2 if self.VSA_EState2 is not None else '-' }'")
        print(f"VSA_EState3:              '{self.VSA_EState3 if self.VSA_EState3 is not None else '-' }'")
        print(f"VSA_EState4:              '{self.VSA_EState4 if self.VSA_EState4 is not None else '-' }'")
        print(f"VSA_EState5:              '{self.VSA_EState5 if self.VSA_EState5 is not None else '-' }'")
        print(f"VSA_EState6:              '{self.VSA_EState6 if self.VSA_EState6 is not None else '-' }'")
        print(f"VSA_EState7:              '{self.VSA_EState7 if self.VSA_EState7 is not None else '-' }'")
        print(f"VSA_EState8:              '{self.VSA_EState8 if self.VSA_EState8 is not None else '-' }'")
        print(f"VSA_EState9:              '{self.VSA_EState9 if self.VSA_EState9 is not None else '-' }'")
        print(f"VSA_EState10:             '{self.VSA_EState10 if self.VSA_EState10 is not None else '-' }'")
        print(f"AUTOCORR3D_1:             '{self.AUTOCORR3D_1 if self.AUTOCORR3D_1 is not None else 0.0 }'")
        print(f"AUTOCORR3D_2:             '{self.AUTOCORR3D_2 if self.AUTOCORR3D_2 is not None else 0.0 }'")
        print(f"AUTOCORR3D_3:             '{self.AUTOCORR3D_3 if self.AUTOCORR3D_3 is not None else 0.0 }'")
        print(f"AUTOCORR3D_4:             '{self.AUTOCORR3D_4 if self.AUTOCORR3D_4 is not None else 0.0 }'")
        print(f"AUTOCORR3D_5:             '{self.AUTOCORR3D_5 if self.AUTOCORR3D_5 is not None else 0.0 }'")
        print(f"AUTOCORR3D_6:             '{self.AUTOCORR3D_6 if self.AUTOCORR3D_6 is not None else 0.0 }'")
        print(f"AUTOCORR3D_7:             '{self.AUTOCORR3D_7 if self.AUTOCORR3D_7 is not None else 0.0 }'")
        print(f"AUTOCORR3D_8:             '{self.AUTOCORR3D_8 if self.AUTOCORR3D_8 is not None else 0.0 }'")
        print(f"AUTOCORR3D_9:             '{self.AUTOCORR3D_9 if self.AUTOCORR3D_9 is not None else 0.0 }'")
        print(f"AUTOCORR3D_10:            '{self.AUTOCORR3D_10 if self.AUTOCORR3D_10 is not None else 0.0 }'")
        print(f"AUTOCORR3D_11:            '{self.AUTOCORR3D_11 if self.AUTOCORR3D_11 is not None else 0.0 }'")
        print(f"AUTOCORR3D_12:            '{self.AUTOCORR3D_12 if self.AUTOCORR3D_12 is not None else 0.0 }'")
        print(f"AUTOCORR3D_13:            '{self.AUTOCORR3D_13 if self.AUTOCORR3D_13 is not None else 0.0 }'")
        print(f"AUTOCORR3D_14:            '{self.AUTOCORR3D_14 if self.AUTOCORR3D_14 is not None else 0.0 }'")
        print(f"AUTOCORR3D_15:            '{self.AUTOCORR3D_15 if self.AUTOCORR3D_15 is not None else 0.0 }'")
        print(f"AUTOCORR3D_16:            '{self.AUTOCORR3D_16 if self.AUTOCORR3D_16 is not None else 0.0 }'")
        print(f"AUTOCORR3D_17:            '{self.AUTOCORR3D_17 if self.AUTOCORR3D_17 is not None else 0.0 }'")
        print(f"AUTOCORR3D_18:            '{self.AUTOCORR3D_18 if self.AUTOCORR3D_18 is not None else 0.0 }'")
        print(f"AUTOCORR3D_19:            '{self.AUTOCORR3D_19 if self.AUTOCORR3D_19 is not None else 0.0 }'")
        print(f"AUTOCORR3D_20:            '{self.AUTOCORR3D_20 if self.AUTOCORR3D_20 is not None else 0.0 }'")
        print(f"AUTOCORR3D_21:            '{self.AUTOCORR3D_21 if self.AUTOCORR3D_21 is not None else 0.0 }'")
        print(f"AUTOCORR3D_22:            '{self.AUTOCORR3D_22 if self.AUTOCORR3D_22 is not None else 0.0 }'")
        print(f"AUTOCORR3D_23:            '{self.AUTOCORR3D_23 if self.AUTOCORR3D_23 is not None else 0.0 }'")
        print(f"AUTOCORR3D_24:            '{self.AUTOCORR3D_24 if self.AUTOCORR3D_24 is not None else 0.0 }'")
        print(f"AUTOCORR3D_25:            '{self.AUTOCORR3D_25 if self.AUTOCORR3D_25 is not None else 0.0 }'")
        print(f"AUTOCORR3D_26:            '{self.AUTOCORR3D_26 if self.AUTOCORR3D_26 is not None else 0.0 }'")
        print(f"AUTOCORR3D_27:            '{self.AUTOCORR3D_27 if self.AUTOCORR3D_27 is not None else 0.0 }'")
        print(f"AUTOCORR3D_28:            '{self.AUTOCORR3D_28 if self.AUTOCORR3D_28 is not None else 0.0 }'")
        print(f"AUTOCORR3D_29:            '{self.AUTOCORR3D_29 if self.AUTOCORR3D_29 is not None else 0.0 }'")
        print(f"AUTOCORR3D_30:            '{self.AUTOCORR3D_30 if self.AUTOCORR3D_30 is not None else 0.0 }'")
        print(f"AUTOCORR3D_31:            '{self.AUTOCORR3D_31 if self.AUTOCORR3D_31 is not None else 0.0 }'")
        print(f"AUTOCORR3D_32:            '{self.AUTOCORR3D_32 if self.AUTOCORR3D_32 is not None else 0.0 }'")
        print(f"AUTOCORR3D_33:            '{self.AUTOCORR3D_33 if self.AUTOCORR3D_33 is not None else 0.0 }'")
        print(f"AUTOCORR3D_34:            '{self.AUTOCORR3D_34 if self.AUTOCORR3D_34 is not None else 0.0 }'")
        print(f"AUTOCORR3D_35:            '{self.AUTOCORR3D_35 if self.AUTOCORR3D_35 is not None else 0.0 }'")
        print(f"AUTOCORR3D_36:            '{self.AUTOCORR3D_36 if self.AUTOCORR3D_36 is not None else 0.0 }'")
        print(f"AUTOCORR3D_37:            '{self.AUTOCORR3D_37 if self.AUTOCORR3D_37 is not None else 0.0 }'")
        print(f"AUTOCORR3D_38:            '{self.AUTOCORR3D_38 if self.AUTOCORR3D_38 is not None else 0.0 }'")
        print(f"AUTOCORR3D_39:            '{self.AUTOCORR3D_39 if self.AUTOCORR3D_39 is not None else 0.0 }'")
        print(f"AUTOCORR3D_40:            '{self.AUTOCORR3D_40 if self.AUTOCORR3D_40 is not None else 0.0 }'")
        print(f"AUTOCORR3D_41:            '{self.AUTOCORR3D_41 if self.AUTOCORR3D_41 is not None else 0.0 }'")
        print(f"AUTOCORR3D_42:            '{self.AUTOCORR3D_42 if self.AUTOCORR3D_42 is not None else 0.0 }'")
        print(f"AUTOCORR3D_43:            '{self.AUTOCORR3D_43 if self.AUTOCORR3D_43 is not None else 0.0 }'")
        print(f"AUTOCORR3D_44:            '{self.AUTOCORR3D_44 if self.AUTOCORR3D_44 is not None else 0.0 }'")
        print(f"AUTOCORR3D_45:            '{self.AUTOCORR3D_45 if self.AUTOCORR3D_45 is not None else 0.0 }'")
        print(f"AUTOCORR3D_46:            '{self.AUTOCORR3D_46 if self.AUTOCORR3D_46 is not None else 0.0 }'")
        print(f"AUTOCORR3D_47:            '{self.AUTOCORR3D_47 if self.AUTOCORR3D_47 is not None else 0.0 }'")
        print(f"AUTOCORR3D_48:            '{self.AUTOCORR3D_48 if self.AUTOCORR3D_48 is not None else 0.0 }'")
        print(f"AUTOCORR3D_49:            '{self.AUTOCORR3D_49 if self.AUTOCORR3D_49 is not None else 0.0 }'")
        print(f"AUTOCORR3D_50:            '{self.AUTOCORR3D_50 if self.AUTOCORR3D_50 is not None else 0.0 }'")
        print(f"AUTOCORR3D_51:            '{self.AUTOCORR3D_51 if self.AUTOCORR3D_51 is not None else 0.0 }'")
        print(f"AUTOCORR3D_52:            '{self.AUTOCORR3D_52 if self.AUTOCORR3D_52 is not None else 0.0 }'")
        print(f"AUTOCORR3D_53:            '{self.AUTOCORR3D_53 if self.AUTOCORR3D_53 is not None else 0.0 }'")
        print(f"AUTOCORR3D_54:            '{self.AUTOCORR3D_54 if self.AUTOCORR3D_54 is not None else 0.0 }'")
        print(f"AUTOCORR3D_55:            '{self.AUTOCORR3D_55 if self.AUTOCORR3D_55 is not None else 0.0 }'")
        print(f"AUTOCORR3D_56:            '{self.AUTOCORR3D_56 if self.AUTOCORR3D_56 is not None else 0.0 }'")
        print(f"AUTOCORR3D_57:            '{self.AUTOCORR3D_57 if self.AUTOCORR3D_57 is not None else 0.0 }'")
        print(f"AUTOCORR3D_58:            '{self.AUTOCORR3D_58 if self.AUTOCORR3D_58 is not None else 0.0 }'")
        print(f"AUTOCORR3D_59:            '{self.AUTOCORR3D_59 if self.AUTOCORR3D_59 is not None else 0.0 }'")
        print(f"AUTOCORR3D_60:            '{self.AUTOCORR3D_60 if self.AUTOCORR3D_60 is not None else 0.0 }'")
        print(f"AUTOCORR3D_61:            '{self.AUTOCORR3D_61 if self.AUTOCORR3D_61 is not None else 0.0 }'")
        print(f"AUTOCORR3D_62:            '{self.AUTOCORR3D_62 if self.AUTOCORR3D_62 is not None else 0.0 }'")
        print(f"AUTOCORR3D_63:            '{self.AUTOCORR3D_63 if self.AUTOCORR3D_63 is not None else 0.0 }'")
        print(f"AUTOCORR3D_64:            '{self.AUTOCORR3D_64 if self.AUTOCORR3D_64 is not None else 0.0 }'")
        print(f"AUTOCORR3D_65:            '{self.AUTOCORR3D_65 if self.AUTOCORR3D_65 is not None else 0.0 }'")
        print(f"AUTOCORR3D_66:            '{self.AUTOCORR3D_66 if self.AUTOCORR3D_66 is not None else 0.0 }'")
        print(f"AUTOCORR3D_67:            '{self.AUTOCORR3D_67 if self.AUTOCORR3D_67 is not None else 0.0 }'")
        print(f"AUTOCORR3D_68:            '{self.AUTOCORR3D_68 if self.AUTOCORR3D_68 is not None else 0.0 }'")
        print(f"AUTOCORR3D_69:            '{self.AUTOCORR3D_69 if self.AUTOCORR3D_69 is not None else 0.0 }'")
        print(f"AUTOCORR3D_70:            '{self.AUTOCORR3D_70 if self.AUTOCORR3D_70 is not None else 0.0 }'")
        print(f"AUTOCORR3D_71:            '{self.AUTOCORR3D_71 if self.AUTOCORR3D_71 is not None else 0.0 }'")
        print(f"AUTOCORR3D_72:            '{self.AUTOCORR3D_72 if self.AUTOCORR3D_72 is not None else 0.0 }'")
        print(f"AUTOCORR3D_73:            '{self.AUTOCORR3D_73 if self.AUTOCORR3D_73 is not None else 0.0 }'")
        print(f"AUTOCORR3D_74:            '{self.AUTOCORR3D_74 if self.AUTOCORR3D_74 is not None else 0.0 }'")
        print(f"AUTOCORR3D_75:            '{self.AUTOCORR3D_75 if self.AUTOCORR3D_75 is not None else 0.0 }'")
        print(f"AUTOCORR3D_76:            '{self.AUTOCORR3D_76 if self.AUTOCORR3D_76 is not None else 0.0 }'")
        print(f"AUTOCORR3D_77:            '{self.AUTOCORR3D_77 if self.AUTOCORR3D_77 is not None else 0.0 }'")
        print(f"AUTOCORR3D_78:            '{self.AUTOCORR3D_78 if self.AUTOCORR3D_78 is not None else 0.0 }'")
        print(f"AUTOCORR3D_79:            '{self.AUTOCORR3D_79 if self.AUTOCORR3D_79 is not None else 0.0 }'")
        print(f"AUTOCORR3D_80:            '{self.AUTOCORR3D_80 if self.AUTOCORR3D_80 is not None else 0.0 }'")
        print(f"Asphericity:              '{self.Asphericity if self.Asphericity is not None else 0.0 }'")
        print(f"Eccentricity:             '{self.Eccentricity if self.Eccentricity is not None else 0.0 }'")
        print(f"InertialShapeFactor:      '{self.InertialShapeFactor if self.InertialShapeFactor is not None else 0.0 }'")
        print(f"NPR1:                     '{self.NPR1 if self.NPR1 is not None else 0.0 }'")
        print(f"NPR2:                     '{self.NPR2 if self.NPR2 is not None else 0.0 }'")
        print(f"PMI1:                     '{self.PMI1 if self.PMI1 is not None else 0.0 }'")
        print(f"PMI2:                     '{self.PMI2 if self.PMI2 is not None else 0.0 }'")
        print(f"PMI3:                     '{self.PMI3 if self.PMI3 is not None else 0.0 }'")
        print(f"RadiusOfGyration:         '{self.RadiusOfGyration if self.RadiusOfGyration is not None else 0.0 }'")
        print(f"SpherocityIndex:          '{self.SpherocityIndex if self.SpherocityIndex is not None else 0.0 }'")
        #endregion

        return

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

        descriptors = {
          "AUTOCORR2D_1": self.AUTOCORR2D_1 if self.AUTOCORR2D_1 is not None else 0.0,
          "AUTOCORR2D_2": self.AUTOCORR2D_2 if self.AUTOCORR2D_2 is not None else 0.0,
          "AUTOCORR2D_3": self.AUTOCORR2D_3 if self.AUTOCORR2D_3 is not None else 0.0,
          "AUTOCORR2D_4": self.AUTOCORR2D_4 if self.AUTOCORR2D_4 is not None else 0.0,
          "AUTOCORR2D_5": self.AUTOCORR2D_5 if self.AUTOCORR2D_5 is not None else 0.0,
          "AUTOCORR2D_6": self.AUTOCORR2D_6 if self.AUTOCORR2D_6 is not None else 0.0,
          "AUTOCORR2D_7": self.AUTOCORR2D_7 if self.AUTOCORR2D_7 is not None else 0.0,
          "AUTOCORR2D_8": self.AUTOCORR2D_8 if self.AUTOCORR2D_8 is not None else 0.0,
          "AUTOCORR2D_9": self.AUTOCORR2D_9 if self.AUTOCORR2D_9 is not None else 0.0,
          "AUTOCORR2D_10": self.AUTOCORR2D_10 if self.AUTOCORR2D_10 is not None else 0.0,
          "AUTOCORR2D_11": self.AUTOCORR2D_11 if self.AUTOCORR2D_11 is not None else 0.0,
          "AUTOCORR2D_12": self.AUTOCORR2D_12 if self.AUTOCORR2D_12 is not None else 0.0,
          "AUTOCORR2D_13": self.AUTOCORR2D_13 if self.AUTOCORR2D_13 is not None else 0.0,
          "AUTOCORR2D_14": self.AUTOCORR2D_14 if self.AUTOCORR2D_14 is not None else 0.0,
          "AUTOCORR2D_15": self.AUTOCORR2D_15 if self.AUTOCORR2D_15 is not None else 0.0,
          "AUTOCORR2D_16": self.AUTOCORR2D_16 if self.AUTOCORR2D_16 is not None else 0.0,
          "AUTOCORR2D_17": self.AUTOCORR2D_17 if self.AUTOCORR2D_17 is not None else 0.0,
          "AUTOCORR2D_18": self.AUTOCORR2D_18 if self.AUTOCORR2D_18 is not None else 0.0,
          "AUTOCORR2D_19": self.AUTOCORR2D_19 if self.AUTOCORR2D_19 is not None else 0.0,
          "AUTOCORR2D_20": self.AUTOCORR2D_20 if self.AUTOCORR2D_20 is not None else 0.0,
          "AUTOCORR2D_21": self.AUTOCORR2D_21 if self.AUTOCORR2D_21 is not None else 0.0,
          "AUTOCORR2D_22": self.AUTOCORR2D_22 if self.AUTOCORR2D_22 is not None else 0.0,
          "AUTOCORR2D_23": self.AUTOCORR2D_23 if self.AUTOCORR2D_23 is not None else 0.0,
          "AUTOCORR2D_24": self.AUTOCORR2D_24 if self.AUTOCORR2D_24 is not None else 0.0,
          "AUTOCORR2D_25": self.AUTOCORR2D_25 if self.AUTOCORR2D_25 is not None else 0.0,
          "AUTOCORR2D_26": self.AUTOCORR2D_26 if self.AUTOCORR2D_26 is not None else 0.0,
          "AUTOCORR2D_27": self.AUTOCORR2D_27 if self.AUTOCORR2D_27 is not None else 0.0,
          "AUTOCORR2D_28": self.AUTOCORR2D_28 if self.AUTOCORR2D_28 is not None else 0.0,
          "AUTOCORR2D_29": self.AUTOCORR2D_29 if self.AUTOCORR2D_29 is not None else 0.0,
          "AUTOCORR2D_30": self.AUTOCORR2D_30 if self.AUTOCORR2D_30 is not None else 0.0,
          "AUTOCORR2D_31": self.AUTOCORR2D_31 if self.AUTOCORR2D_31 is not None else 0.0,
          "AUTOCORR2D_32": self.AUTOCORR2D_32 if self.AUTOCORR2D_32 is not None else 0.0,
          "AUTOCORR2D_33": self.AUTOCORR2D_33 if self.AUTOCORR2D_33 is not None else 0.0,
          "AUTOCORR2D_34": self.AUTOCORR2D_34 if self.AUTOCORR2D_34 is not None else 0.0,
          "AUTOCORR2D_35": self.AUTOCORR2D_35 if self.AUTOCORR2D_35 is not None else 0.0,
          "AUTOCORR2D_36": self.AUTOCORR2D_36 if self.AUTOCORR2D_36 is not None else 0.0,
          "AUTOCORR2D_37": self.AUTOCORR2D_37 if self.AUTOCORR2D_37 is not None else 0.0,
          "AUTOCORR2D_38": self.AUTOCORR2D_38 if self.AUTOCORR2D_38 is not None else 0.0,
          "AUTOCORR2D_39": self.AUTOCORR2D_39 if self.AUTOCORR2D_39 is not None else 0.0,
          "AUTOCORR2D_40": self.AUTOCORR2D_40 if self.AUTOCORR2D_40 is not None else 0.0,
          "AUTOCORR2D_41": self.AUTOCORR2D_41 if self.AUTOCORR2D_41 is not None else 0.0,
          "AUTOCORR2D_42": self.AUTOCORR2D_42 if self.AUTOCORR2D_42 is not None else 0.0,
          "AUTOCORR2D_43": self.AUTOCORR2D_43 if self.AUTOCORR2D_43 is not None else 0.0,
          "AUTOCORR2D_44": self.AUTOCORR2D_44 if self.AUTOCORR2D_44 is not None else 0.0,
          "AUTOCORR2D_45": self.AUTOCORR2D_45 if self.AUTOCORR2D_45 is not None else 0.0,
          "AUTOCORR2D_46": self.AUTOCORR2D_46 if self.AUTOCORR2D_46 is not None else 0.0,
          "AUTOCORR2D_47": self.AUTOCORR2D_47 if self.AUTOCORR2D_47 is not None else 0.0,
          "AUTOCORR2D_48": self.AUTOCORR2D_48 if self.AUTOCORR2D_48 is not None else 0.0,
          "AUTOCORR2D_49": self.AUTOCORR2D_49 if self.AUTOCORR2D_49 is not None else 0.0,
          "AUTOCORR2D_50": self.AUTOCORR2D_50 if self.AUTOCORR2D_50 is not None else 0.0,
          "AUTOCORR2D_51": self.AUTOCORR2D_51 if self.AUTOCORR2D_51 is not None else 0.0,
          "AUTOCORR2D_52": self.AUTOCORR2D_52 if self.AUTOCORR2D_52 is not None else 0.0,
          "AUTOCORR2D_53": self.AUTOCORR2D_53 if self.AUTOCORR2D_53 is not None else 0.0,
          "AUTOCORR2D_54": self.AUTOCORR2D_54 if self.AUTOCORR2D_54 is not None else 0.0,
          "AUTOCORR2D_55": self.AUTOCORR2D_55 if self.AUTOCORR2D_55 is not None else 0.0,
          "AUTOCORR2D_56": self.AUTOCORR2D_56 if self.AUTOCORR2D_56 is not None else 0.0,
          "AUTOCORR2D_57": self.AUTOCORR2D_57 if self.AUTOCORR2D_57 is not None else 0.0,
          "AUTOCORR2D_58": self.AUTOCORR2D_58 if self.AUTOCORR2D_58 is not None else 0.0,
          "AUTOCORR2D_59": self.AUTOCORR2D_59 if self.AUTOCORR2D_59 is not None else 0.0,
          "AUTOCORR2D_60": self.AUTOCORR2D_60 if self.AUTOCORR2D_60 is not None else 0.0,
          "AUTOCORR2D_61": self.AUTOCORR2D_61 if self.AUTOCORR2D_61 is not None else 0.0,
          "AUTOCORR2D_62": self.AUTOCORR2D_62 if self.AUTOCORR2D_62 is not None else 0.0,
          "AUTOCORR2D_63": self.AUTOCORR2D_63 if self.AUTOCORR2D_63 is not None else 0.0,
          "AUTOCORR2D_64": self.AUTOCORR2D_64 if self.AUTOCORR2D_64 is not None else 0.0,
          "AUTOCORR2D_65": self.AUTOCORR2D_65 if self.AUTOCORR2D_65 is not None else 0.0,
          "AUTOCORR2D_66": self.AUTOCORR2D_66 if self.AUTOCORR2D_66 is not None else 0.0,
          "AUTOCORR2D_67": self.AUTOCORR2D_67 if self.AUTOCORR2D_67 is not None else 0.0,
          "AUTOCORR2D_68": self.AUTOCORR2D_68 if self.AUTOCORR2D_68 is not None else 0.0,
          "AUTOCORR2D_69": self.AUTOCORR2D_69 if self.AUTOCORR2D_69 is not None else 0.0,
          "AUTOCORR2D_70": self.AUTOCORR2D_70 if self.AUTOCORR2D_70 is not None else 0.0,
          "AUTOCORR2D_71": self.AUTOCORR2D_71 if self.AUTOCORR2D_71 is not None else 0.0,
          "AUTOCORR2D_72": self.AUTOCORR2D_72 if self.AUTOCORR2D_72 is not None else 0.0,
          "AUTOCORR2D_73": self.AUTOCORR2D_73 if self.AUTOCORR2D_73 is not None else 0.0,
          "AUTOCORR2D_74": self.AUTOCORR2D_74 if self.AUTOCORR2D_74 is not None else 0.0,
          "AUTOCORR2D_75": self.AUTOCORR2D_75 if self.AUTOCORR2D_75 is not None else 0.0,
          "AUTOCORR2D_76": self.AUTOCORR2D_76 if self.AUTOCORR2D_76 is not None else 0.0,
          "AUTOCORR2D_77": self.AUTOCORR2D_77 if self.AUTOCORR2D_77 is not None else 0.0,
          "AUTOCORR2D_78": self.AUTOCORR2D_78 if self.AUTOCORR2D_78 is not None else 0.0,
          "AUTOCORR2D_79": self.AUTOCORR2D_79 if self.AUTOCORR2D_79 is not None else 0.0,
          "AUTOCORR2D_80": self.AUTOCORR2D_80 if self.AUTOCORR2D_80 is not None else 0.0,
          "AUTOCORR2D_81": self.AUTOCORR2D_81 if self.AUTOCORR2D_81 is not None else 0.0,
          "AUTOCORR2D_82": self.AUTOCORR2D_82 if self.AUTOCORR2D_82 is not None else 0.0,
          "AUTOCORR2D_83": self.AUTOCORR2D_83 if self.AUTOCORR2D_83 is not None else 0.0,
          "AUTOCORR2D_84": self.AUTOCORR2D_84 if self.AUTOCORR2D_84 is not None else 0.0,
          "AUTOCORR2D_85": self.AUTOCORR2D_85 if self.AUTOCORR2D_85 is not None else 0.0,
          "AUTOCORR2D_86": self.AUTOCORR2D_86 if self.AUTOCORR2D_86 is not None else 0.0,
          "AUTOCORR2D_87": self.AUTOCORR2D_87 if self.AUTOCORR2D_87 is not None else 0.0,
          "AUTOCORR2D_88": self.AUTOCORR2D_88 if self.AUTOCORR2D_88 is not None else 0.0,
          "AUTOCORR2D_89": self.AUTOCORR2D_89 if self.AUTOCORR2D_89 is not None else 0.0,
          "AUTOCORR2D_90": self.AUTOCORR2D_90 if self.AUTOCORR2D_90 is not None else 0.0,
          "AUTOCORR2D_91": self.AUTOCORR2D_91 if self.AUTOCORR2D_91 is not None else 0.0,
          "AUTOCORR2D_92": self.AUTOCORR2D_92 if self.AUTOCORR2D_92 is not None else 0.0,
          "AUTOCORR2D_93": self.AUTOCORR2D_93 if self.AUTOCORR2D_93 is not None else 0.0,
          "AUTOCORR2D_94": self.AUTOCORR2D_94 if self.AUTOCORR2D_94 is not None else 0.0,
          "AUTOCORR2D_95": self.AUTOCORR2D_95 if self.AUTOCORR2D_95 is not None else 0.0,
          "AUTOCORR2D_96": self.AUTOCORR2D_96 if self.AUTOCORR2D_96 is not None else 0.0,
          "AUTOCORR2D_97": self.AUTOCORR2D_97 if self.AUTOCORR2D_97 is not None else 0.0,
          "AUTOCORR2D_98": self.AUTOCORR2D_98 if self.AUTOCORR2D_98 is not None else 0.0,
          "AUTOCORR2D_99": self.AUTOCORR2D_99 if self.AUTOCORR2D_99 is not None else 0.0,
          "AUTOCORR2D_100": self.AUTOCORR2D_100 if self.AUTOCORR2D_100 is not None else 0.0,
          "AUTOCORR2D_101": self.AUTOCORR2D_101 if self.AUTOCORR2D_101 is not None else 0.0,
          "AUTOCORR2D_102": self.AUTOCORR2D_102 if self.AUTOCORR2D_102 is not None else 0.0,
          "AUTOCORR2D_103": self.AUTOCORR2D_103 if self.AUTOCORR2D_103 is not None else 0.0,
          "AUTOCORR2D_104": self.AUTOCORR2D_104 if self.AUTOCORR2D_104 is not None else 0.0,
          "AUTOCORR2D_105": self.AUTOCORR2D_105 if self.AUTOCORR2D_105 is not None else 0.0,
          "AUTOCORR2D_106": self.AUTOCORR2D_106 if self.AUTOCORR2D_106 is not None else 0.0,
          "AUTOCORR2D_107": self.AUTOCORR2D_107 if self.AUTOCORR2D_107 is not None else 0.0,
          "AUTOCORR2D_108": self.AUTOCORR2D_108 if self.AUTOCORR2D_108 is not None else 0.0,
          "AUTOCORR2D_109": self.AUTOCORR2D_109 if self.AUTOCORR2D_109 is not None else 0.0,
          "AUTOCORR2D_110": self.AUTOCORR2D_110 if self.AUTOCORR2D_110 is not None else 0.0,
          "AUTOCORR2D_111": self.AUTOCORR2D_111 if self.AUTOCORR2D_111 is not None else 0.0,
          "AUTOCORR2D_112": self.AUTOCORR2D_112 if self.AUTOCORR2D_112 is not None else 0.0,
          "AUTOCORR2D_113": self.AUTOCORR2D_113 if self.AUTOCORR2D_113 is not None else 0.0,
          "AUTOCORR2D_114": self.AUTOCORR2D_114 if self.AUTOCORR2D_114 is not None else 0.0,
          "AUTOCORR2D_115": self.AUTOCORR2D_115 if self.AUTOCORR2D_115 is not None else 0.0,
          "AUTOCORR2D_116": self.AUTOCORR2D_116 if self.AUTOCORR2D_116 is not None else 0.0,
          "AUTOCORR2D_117": self.AUTOCORR2D_117 if self.AUTOCORR2D_117 is not None else 0.0,
          "AUTOCORR2D_118": self.AUTOCORR2D_118 if self.AUTOCORR2D_118 is not None else 0.0,
          "AUTOCORR2D_119": self.AUTOCORR2D_119 if self.AUTOCORR2D_119 is not None else 0.0,
          "AUTOCORR2D_120": self.AUTOCORR2D_120 if self.AUTOCORR2D_120 is not None else 0.0,
          "AUTOCORR2D_121": self.AUTOCORR2D_121 if self.AUTOCORR2D_121 is not None else 0.0,
          "AUTOCORR2D_122": self.AUTOCORR2D_122 if self.AUTOCORR2D_122 is not None else 0.0,
          "AUTOCORR2D_123": self.AUTOCORR2D_123 if self.AUTOCORR2D_123 is not None else 0.0,
          "AUTOCORR2D_124": self.AUTOCORR2D_124 if self.AUTOCORR2D_124 is not None else 0.0,
          "AUTOCORR2D_125": self.AUTOCORR2D_125 if self.AUTOCORR2D_125 is not None else 0.0,
          "AUTOCORR2D_126": self.AUTOCORR2D_126 if self.AUTOCORR2D_126 is not None else 0.0,
          "AUTOCORR2D_127": self.AUTOCORR2D_127 if self.AUTOCORR2D_127 is not None else 0.0,
          "AUTOCORR2D_128": self.AUTOCORR2D_128 if self.AUTOCORR2D_128 is not None else 0.0,
          "AUTOCORR2D_129": self.AUTOCORR2D_129 if self.AUTOCORR2D_129 is not None else 0.0,
          "AUTOCORR2D_130": self.AUTOCORR2D_130 if self.AUTOCORR2D_130 is not None else 0.0,
          "AUTOCORR2D_131": self.AUTOCORR2D_131 if self.AUTOCORR2D_131 is not None else 0.0,
          "AUTOCORR2D_132": self.AUTOCORR2D_132 if self.AUTOCORR2D_132 is not None else 0.0,
          "AUTOCORR2D_133": self.AUTOCORR2D_133 if self.AUTOCORR2D_133 is not None else 0.0,
          "AUTOCORR2D_134": self.AUTOCORR2D_134 if self.AUTOCORR2D_134 is not None else 0.0,
          "AUTOCORR2D_135": self.AUTOCORR2D_135 if self.AUTOCORR2D_135 is not None else 0.0,
          "AUTOCORR2D_136": self.AUTOCORR2D_136 if self.AUTOCORR2D_136 is not None else 0.0,
          "AUTOCORR2D_137": self.AUTOCORR2D_137 if self.AUTOCORR2D_137 is not None else 0.0,
          "AUTOCORR2D_138": self.AUTOCORR2D_138 if self.AUTOCORR2D_138 is not None else 0.0,
          "AUTOCORR2D_139": self.AUTOCORR2D_139 if self.AUTOCORR2D_139 is not None else 0.0,
          "AUTOCORR2D_140": self.AUTOCORR2D_140 if self.AUTOCORR2D_140 is not None else 0.0,
          "AUTOCORR2D_141": self.AUTOCORR2D_141 if self.AUTOCORR2D_141 is not None else 0.0,
          "AUTOCORR2D_142": self.AUTOCORR2D_142 if self.AUTOCORR2D_142 is not None else 0.0,
          "AUTOCORR2D_143": self.AUTOCORR2D_143 if self.AUTOCORR2D_143 is not None else 0.0,
          "AUTOCORR2D_144": self.AUTOCORR2D_144 if self.AUTOCORR2D_144 is not None else 0.0,
          "AUTOCORR2D_145": self.AUTOCORR2D_145 if self.AUTOCORR2D_145 is not None else 0.0,
          "AUTOCORR2D_146": self.AUTOCORR2D_146 if self.AUTOCORR2D_146 is not None else 0.0,
          "AUTOCORR2D_147": self.AUTOCORR2D_147 if self.AUTOCORR2D_147 is not None else 0.0,
          "AUTOCORR2D_148": self.AUTOCORR2D_148 if self.AUTOCORR2D_148 is not None else 0.0,
          "AUTOCORR2D_149": self.AUTOCORR2D_149 if self.AUTOCORR2D_149 is not None else 0.0,
          "AUTOCORR2D_150": self.AUTOCORR2D_150 if self.AUTOCORR2D_150 is not None else 0.0,
          "AUTOCORR2D_151": self.AUTOCORR2D_151 if self.AUTOCORR2D_151 is not None else 0.0,
          "AUTOCORR2D_152": self.AUTOCORR2D_152 if self.AUTOCORR2D_152 is not None else 0.0,
          "AUTOCORR2D_153": self.AUTOCORR2D_153 if self.AUTOCORR2D_153 is not None else 0.0,
          "AUTOCORR2D_154": self.AUTOCORR2D_154 if self.AUTOCORR2D_154 is not None else 0.0,
          "AUTOCORR2D_155": self.AUTOCORR2D_155 if self.AUTOCORR2D_155 is not None else 0.0,
          "AUTOCORR2D_156": self.AUTOCORR2D_156 if self.AUTOCORR2D_156 is not None else 0.0,
          "AUTOCORR2D_157": self.AUTOCORR2D_157 if self.AUTOCORR2D_157 is not None else 0.0,
          "AUTOCORR2D_158": self.AUTOCORR2D_158 if self.AUTOCORR2D_158 is not None else 0.0,
          "AUTOCORR2D_159": self.AUTOCORR2D_159 if self.AUTOCORR2D_159 is not None else 0.0,
          "AUTOCORR2D_160": self.AUTOCORR2D_160 if self.AUTOCORR2D_160 is not None else 0.0,
          "AUTOCORR2D_161": self.AUTOCORR2D_161 if self.AUTOCORR2D_161 is not None else 0.0,
          "AUTOCORR2D_162": self.AUTOCORR2D_162 if self.AUTOCORR2D_162 is not None else 0.0,
          "AUTOCORR2D_163": self.AUTOCORR2D_163 if self.AUTOCORR2D_163 is not None else 0.0,
          "AUTOCORR2D_164": self.AUTOCORR2D_164 if self.AUTOCORR2D_164 is not None else 0.0,
          "AUTOCORR2D_165": self.AUTOCORR2D_165 if self.AUTOCORR2D_165 is not None else 0.0,
          "AUTOCORR2D_166": self.AUTOCORR2D_166 if self.AUTOCORR2D_166 is not None else 0.0,
          "AUTOCORR2D_167": self.AUTOCORR2D_167 if self.AUTOCORR2D_167 is not None else 0.0,
          "AUTOCORR2D_168": self.AUTOCORR2D_168 if self.AUTOCORR2D_168 is not None else 0.0,
          "AUTOCORR2D_169": self.AUTOCORR2D_169 if self.AUTOCORR2D_169 is not None else 0.0,
          "AUTOCORR2D_170": self.AUTOCORR2D_170 if self.AUTOCORR2D_170 is not None else 0.0,
          "AUTOCORR2D_171": self.AUTOCORR2D_171 if self.AUTOCORR2D_171 is not None else 0.0,
          "AUTOCORR2D_172": self.AUTOCORR2D_172 if self.AUTOCORR2D_172 is not None else 0.0,
          "AUTOCORR2D_173": self.AUTOCORR2D_173 if self.AUTOCORR2D_173 is not None else 0.0,
          "AUTOCORR2D_174": self.AUTOCORR2D_174 if self.AUTOCORR2D_174 is not None else 0.0,
          "AUTOCORR2D_175": self.AUTOCORR2D_175 if self.AUTOCORR2D_175 is not None else 0.0,
          "AUTOCORR2D_176": self.AUTOCORR2D_176 if self.AUTOCORR2D_176 is not None else 0.0,
          "AUTOCORR2D_177": self.AUTOCORR2D_177 if self.AUTOCORR2D_177 is not None else 0.0,
          "AUTOCORR2D_178": self.AUTOCORR2D_178 if self.AUTOCORR2D_178 is not None else 0.0,
          "AUTOCORR2D_179": self.AUTOCORR2D_179 if self.AUTOCORR2D_179 is not None else 0.0,
          "AUTOCORR2D_180": self.AUTOCORR2D_180 if self.AUTOCORR2D_180 is not None else 0.0,
          "AUTOCORR2D_181": self.AUTOCORR2D_181 if self.AUTOCORR2D_181 is not None else 0.0,
          "AUTOCORR2D_182": self.AUTOCORR2D_182 if self.AUTOCORR2D_182 is not None else 0.0,
          "AUTOCORR2D_183": self.AUTOCORR2D_183 if self.AUTOCORR2D_183 is not None else 0.0,
          "AUTOCORR2D_184": self.AUTOCORR2D_184 if self.AUTOCORR2D_184 is not None else 0.0,
          "AUTOCORR2D_185": self.AUTOCORR2D_185 if self.AUTOCORR2D_185 is not None else 0.0,
          "AUTOCORR2D_186": self.AUTOCORR2D_186 if self.AUTOCORR2D_186 is not None else 0.0,
          "AUTOCORR2D_187": self.AUTOCORR2D_187 if self.AUTOCORR2D_187 is not None else 0.0,
          "AUTOCORR2D_188": self.AUTOCORR2D_188 if self.AUTOCORR2D_188 is not None else 0.0,
          "AUTOCORR2D_189": self.AUTOCORR2D_189 if self.AUTOCORR2D_189 is not None else 0.0,
          "AUTOCORR2D_190": self.AUTOCORR2D_190 if self.AUTOCORR2D_190 is not None else 0.0,
          "AUTOCORR2D_191": self.AUTOCORR2D_191 if self.AUTOCORR2D_191 is not None else 0.0,
          "AUTOCORR2D_192": self.AUTOCORR2D_192 if self.AUTOCORR2D_192 is not None else 0.0,
          "BCUT2D_CHGHI": self.BCUT2D_CHGHI if self.BCUT2D_CHGHI is not None else 0.0,
          "BCUT2D_CHGLO": self.BCUT2D_CHGLO if self.BCUT2D_CHGLO is not None else 0.0,
          "BCUT2D_LOGPHI": self.BCUT2D_LOGPHI if self.BCUT2D_LOGPHI is not None else 0.0,
          "BCUT2D_LOGPLOW": self.BCUT2D_LOGPLOW if self.BCUT2D_LOGPLOW is not None else 0.0,
          "BCUT2D_MRHI": self.BCUT2D_MRHI if self.BCUT2D_MRHI is not None else 0.0,
          "BCUT2D_MRLOW": self.BCUT2D_MRLOW if self.BCUT2D_MRLOW is not None else 0.0,
          "BCUT2D_MWHI": self.BCUT2D_MWHI if self.BCUT2D_MWHI is not None else 0.0,
          "BCUT2D_MWLOW": self.BCUT2D_MWLOW if self.BCUT2D_MWLOW is not None else 0.0,
          "BalabanJ": self.BalabanJ if self.BalabanJ is not None else 0.0,
          "BertzCT": self.BertzCT if self.BertzCT is not None else 0.0,
          "Chi0": self.Chi0 if self.Chi0 is not None else 0.0,
          "Chi0n": self.Chi0n if self.Chi0n is not None else 0.0,
          "Chi0v": self.Chi0v if self.Chi0v is not None else 0.0,
          "Chi1": self.Chi1 if self.Chi1 is not None else 0.0,
          "Chi1n": self.Chi1n if self.Chi1n is not None else 0.0,
          "Chi1v": self.Chi1v if self.Chi1v is not None else 0.0,
          "Chi2n": self.Chi2n if self.Chi2n is not None else 0.0,
          "Chi2v": self.Chi2v if self.Chi2v is not None else 0.0,
          "Chi3n": self.Chi3n if self.Chi3n is not None else 0.0,
          "Chi3v": self.Chi3v if self.Chi3v is not None else 0.0,
          "Chi4n": self.Chi4n if self.Chi4n is not None else 0.0,
          "Chi4v": self.Chi4v if self.Chi4v is not None else 0.0,
          "EState_VSA1": self.EState_VSA1 if self.EState_VSA1 is not None else 0.0,
          "EState_VSA2": self.EState_VSA2 if self.EState_VSA2 is not None else 0.0,
          "EState_VSA3": self.EState_VSA3 if self.EState_VSA3 is not None else 0.0,
          "EState_VSA4": self.EState_VSA4 if self.EState_VSA4 is not None else 0.0,
          "EState_VSA5": self.EState_VSA5 if self.EState_VSA5 is not None else 0.0,
          "EState_VSA6": self.EState_VSA6 if self.EState_VSA6 is not None else 0.0,
          "EState_VSA7": self.EState_VSA7 if self.EState_VSA7 is not None else 0.0,
          "EState_VSA8": self.EState_VSA8 if self.EState_VSA8 is not None else 0.0,
          "EState_VSA9": self.EState_VSA9 if self.EState_VSA9 is not None else 0.0,
          "EState_VSA10": self.EState_VSA10 if self.EState_VSA10 is not None else 0.0,
          "EState_VSA11": self.EState_VSA11 if self.EState_VSA11 is not None else 0.0,
          "MaxAbsEStateIndex": self.MaxAbsEStateIndex if self.MaxAbsEStateIndex is not None else 0.0,
          "MaxEStateIndex": self.MaxEStateIndex if self.MaxEStateIndex is not None else 0.0,
          "MinAbsEStateIndex": self.MinAbsEStateIndex if self.MinAbsEStateIndex is not None else 0.0,
          "MinEStateIndex": self.MinEStateIndex if self.MinEStateIndex is not None else 0.0,
          "ExactMolWt": self.ExactMolWt if self.ExactMolWt is not None else 0.0,
          "FpDensityMorgan1": self.FpDensityMorgan1 if self.FpDensityMorgan1 is not None else 0,
          "FpDensityMorgan2": self.FpDensityMorgan2 if self.FpDensityMorgan2 is not None else 0,
          "FpDensityMorgan3": self.FpDensityMorgan3 if self.FpDensityMorgan3 is not None else 0,
          "fr_Al_COO": self.fr_Al_COO if self.fr_Al_COO is not None else 0,
          "fr_Al_OH": self.fr_Al_OH if self.fr_Al_OH is not None else 0,
          "fr_Al_OH_noTert": self.fr_Al_OH_noTert if self.fr_Al_OH_noTert is not None else 0,
          "fr_ArN": self.fr_ArN if self.fr_ArN is not None else 0,
          "fr_Ar_COO": self.fr_Ar_COO if self.fr_Ar_COO is not None else 0,
          "fr_Ar_N": self.fr_Ar_N if self.fr_Ar_N is not None else 0,
          "fr_Ar_NH": self.fr_Ar_NH if self.fr_Ar_NH is not None else 0,
          "fr_Ar_OH": self.fr_Ar_OH if self.fr_Ar_OH is not None else 0,
          "fr_COO": self.fr_COO if self.fr_COO is not None else 0,
          "fr_COO2": self.fr_COO2 if self.fr_COO2 is not None else 0,
          "fr_C_O": self.fr_C_O if self.fr_C_O is not None else 0,
          "fr_C_O_noCOO": self.fr_C_O_noCOO if self.fr_C_O_noCOO is not None else 0,
          "fr_C_S": self.fr_C_S if self.fr_C_S is not None else 0,
          "fr_HOCCN": self.fr_HOCCN if self.fr_HOCCN is not None else 0,
          "fr_Imine": self.fr_Imine if self.fr_Imine is not None else 0,
          "fr_NH0": self.fr_NH0 if self.fr_NH0 is not None else 0,
          "fr_NH1": self.fr_NH1 if self.fr_NH1 is not None else 0,
          "fr_NH2": self.fr_NH2 if self.fr_NH2 is not None else 0,
          "fr_N_O": self.fr_N_O if self.fr_N_O is not None else 0,
          "fr_Ndealkylation1": self.fr_Ndealkylation1 if self.fr_Ndealkylation1 is not None else 0,
          "fr_Ndealkylation2": self.fr_Ndealkylation2 if self.fr_Ndealkylation2 is not None else 0,
          "fr_Nhpyrrole": self.fr_Nhpyrrole if self.fr_Nhpyrrole is not None else 0,
          "fr_SH": self.fr_SH if self.fr_SH is not None else 0,
          "fr_aldehyde": self.fr_aldehyde if self.fr_aldehyde is not None else 0,
          "fr_alkyl_carbamate": self.fr_alkyl_carbamate if self.fr_alkyl_carbamate is not None else 0,
          "fr_alkyl_halide": self.fr_alkyl_halide if self.fr_alkyl_halide is not None else 0,
          "fr_allylic_oxid": self.fr_allylic_oxid if self.fr_allylic_oxid is not None else 0,
          "fr_amide": self.fr_amide if self.fr_amide is not None else 0,
          "fr_amidine": self.fr_amidine if self.fr_amidine is not None else 0,
          "fr_aniline": self.fr_aniline if self.fr_aniline is not None else 0,
          "fr_aryl_methyl": self.fr_aryl_methyl if self.fr_aryl_methyl is not None else 0,
          "fr_azide": self.fr_azide if self.fr_azide is not None else 0,
          "fr_azo": self.fr_azo if self.fr_azo is not None else 0,
          "fr_barbitur": self.fr_barbitur if self.fr_barbitur is not None else 0,
          "fr_benzene": self.fr_benzene if self.fr_benzene is not None else 0,
          "fr_benzodiazepine": self.fr_benzodiazepine if self.fr_benzodiazepine is not None else 0,
          "fr_bicyclic": self.fr_bicyclic if self.fr_bicyclic is not None else 0,
          "fr_diazo": self.fr_diazo if self.fr_diazo is not None else 0,
          "fr_dihydropyridine": self.fr_dihydropyridine if self.fr_dihydropyridine is not None else 0,
          "fr_epoxide": self.fr_epoxide if self.fr_epoxide is not None else 0,
          "fr_ester": self.fr_ester if self.fr_ester is not None else 0,
          "fr_ether": self.fr_ether if self.fr_ether is not None else 0,
          "fr_furan": self.fr_furan if self.fr_furan is not None else 0,
          "fr_guanido": self.fr_guanido if self.fr_guanido is not None else 0,
          "fr_halogen": self.fr_halogen if self.fr_halogen is not None else 0,
          "fr_hdrzine": self.fr_hdrzine if self.fr_hdrzine is not None else 0,
          "fr_hdrzone": self.fr_hdrzone if self.fr_hdrzone is not None else 0,
          "fr_imidazole": self.fr_imidazole if self.fr_imidazole is not None else 0,
          "fr_imide": self.fr_imide if self.fr_imide is not None else 0,
          "fr_isocyan": self.fr_isocyan if self.fr_isocyan is not None else 0,
          "fr_isothiocyan": self.fr_isothiocyan if self.fr_isothiocyan is not None else 0,
          "fr_ketone": self.fr_ketone if self.fr_ketone is not None else 0,
          "fr_ketone_Topliss": self.fr_ketone_Topliss if self.fr_ketone_Topliss is not None else 0,
          "fr_lactam": self.fr_lactam if self.fr_lactam is not None else 0,
          "fr_lactone": self.fr_lactone if self.fr_lactone is not None else 0,
          "fr_methoxy": self.fr_methoxy if self.fr_methoxy is not None else 0,
          "fr_morpholine": self.fr_morpholine if self.fr_morpholine is not None else 0,
          "fr_nitrile": self.fr_nitrile if self.fr_nitrile is not None else 0,
          "fr_nitro": self.fr_nitro if self.fr_nitro is not None else 0,
          "fr_nitro_arom": self.fr_nitro_arom if self.fr_nitro_arom is not None else 0,
          "fr_nitro_arom_nonortho": self.fr_nitro_arom_nonortho if self.fr_nitro_arom_nonortho is not None else 0,
          "fr_nitroso": self.fr_nitroso if self.fr_nitroso is not None else 0,
          "fr_oxazole": self.fr_oxazole if self.fr_oxazole is not None else 0,
          "fr_oxime": self.fr_oxime if self.fr_oxime is not None else 0,
          "fr_para_hydroxylation": self.fr_para_hydroxylation if self.fr_para_hydroxylation is not None else 0,
          "fr_phenol": self.fr_phenol if self.fr_phenol is not None else 0,
          "fr_phenol_noOrthoHbond": self.fr_phenol_noOrthoHbond if self.fr_phenol_noOrthoHbond is not None else 0,
          "fr_phos_acid": self.fr_phos_acid if self.fr_phos_acid is not None else 0,
          "fr_phos_ester": self.fr_phos_ester if self.fr_phos_ester is not None else 0,
          "fr_piperdine": self.fr_piperdine if self.fr_piperdine is not None else 0,
          "fr_piperzine": self.fr_piperzine if self.fr_piperzine is not None else 0,
          "fr_priamide": self.fr_priamide if self.fr_priamide is not None else 0,
          "fr_prisulfonamd": self.fr_prisulfonamd if self.fr_prisulfonamd is not None else 0,
          "fr_pyridine": self.fr_pyridine if self.fr_pyridine is not None else 0,
          "fr_quatN": self.fr_quatN if self.fr_quatN is not None else 0,
          "fr_sulfide": self.fr_sulfide if self.fr_sulfide is not None else 0,
          "fr_sulfonamd": self.fr_sulfonamd if self.fr_sulfonamd is not None else 0,
          "fr_sulfone": self.fr_sulfone if self.fr_sulfone is not None else 0,
          "fr_term_acetylene": self.fr_term_acetylene if self.fr_term_acetylene is not None else 0,
          "fr_tetrazole": self.fr_tetrazole if self.fr_tetrazole is not None else 0,
          "fr_thiazole": self.fr_thiazole if self.fr_thiazole is not None else 0,
          "fr_thiocyan": self.fr_thiocyan if self.fr_thiocyan is not None else 0,
          "fr_thiophene": self.fr_thiophene if self.fr_thiophene is not None else 0,
          "fr_unbrch_alkane": self.fr_unbrch_alkane if self.fr_unbrch_alkane is not None else 0,
          "fr_urea": self.fr_urea if self.fr_urea is not None else 0,
          "FractionCSP3": self.FractionCSP3 if self.FractionCSP3 is not None else 0.0,
          "HallKierAlpha": self.HallKierAlpha if self.HallKierAlpha is not None else 0.0,
          "HeavyAtomMolWt": self.HeavyAtomMolWt if self.HeavyAtomMolWt is not None else 0.0,
          "HeavyAtomCount": self.HeavyAtomCount if self.HeavyAtomCount is not None else 0,
          "Ipc": self.Ipc if self.Ipc is not None else 0.0,
          "Kappa1": self.Kappa1 if self.Kappa1 is not None else 0.0,
          "Kappa2": self.Kappa2 if self.Kappa2 is not None else 0.0,
          "Kappa3": self.Kappa3 if self.Kappa3 is not None else 0.0,
          "LabuteASA": self.LabuteASA if self.LabuteASA is not None else 0.0,
          "MaxAbsPartialCharge": self.MaxAbsPartialCharge if self.MaxAbsPartialCharge is not None else 0.0,
          "MaxPartialCharge": self.MaxPartialCharge if self.MaxPartialCharge is not None else 0.0,
          "MinAbsPartialCharge": self.MinAbsPartialCharge if self.MinAbsPartialCharge is not None else 0.0,
          "MinPartialCharge": self.MinPartialCharge if self.MinPartialCharge is not None else 0.0,
          "MolLogP": self.MolLogP if self.MolLogP is not None else 0.0,
          "MolMR": self.MolMR if self.MolMR is not None else 0.0,
          "MolWt": self.MolWt if self.MolWt is not None else 0.0,
          "NHOHCount": self.NHOHCount if self.NHOHCount is not None else 0,
          "NOCount": self.NOCount if self.NOCount is not None else 0,
          "NumAliphaticCarbocycles": self.NumAliphaticCarbocycles if self.NumAliphaticCarbocycles is not None else 0,
          "NumAliphaticHeterocycles": self.NumAliphaticHeterocycles if self.NumAliphaticHeterocycles is not None else 0,
          "NumAliphaticRings": self.NumAliphaticRings if self.NumAliphaticRings is not None else 0,
          "NumAromaticCarbocycles": self.NumAromaticCarbocycles if self.NumAromaticCarbocycles is not None else 0,
          "NumAromaticHeterocycles": self.NumAromaticHeterocycles if self.NumAromaticHeterocycles is not None else 0,
          "NumAromaticRings": self.NumAromaticRings if self.NumAromaticRings is not None else 0,
          "NumHAcceptors": self.NumHAcceptors if self.NumHAcceptors is not None else 0,
          "NumHDonors": self.NumHDonors if self.NumHDonors is not None else 0,
          "NumHeteroatoms": self.NumHeteroatoms if self.NumHeteroatoms is not None else 0,
          "NumRadicalElectrons": self.NumRadicalElectrons if self.NumRadicalElectrons is not None else 0,
          "NumRotatableBonds": self.NumRotatableBonds if self.NumRotatableBonds is not None else 0,
          "NumSaturatedCarbocycles": self.NumSaturatedCarbocycles if self.NumSaturatedCarbocycles is not None else 0,
          "NumSaturatedHeterocycles": self.NumSaturatedHeterocycles if self.NumSaturatedHeterocycles is not None else 0,
          "NumSaturatedRings": self.NumSaturatedRings if self.NumSaturatedRings is not None else 0,
          "NumValenceElectrons": self.NumValenceElectrons if self.NumValenceElectrons is not None else 0,
          "PEOE_VSA1": self.PEOE_VSA1 if self.PEOE_VSA1 is not None else 0.0,
          "PEOE_VSA2": self.PEOE_VSA2 if self.PEOE_VSA2 is not None else 0.0,
          "PEOE_VSA3": self.PEOE_VSA3 if self.PEOE_VSA3 is not None else 0.0,
          "PEOE_VSA4": self.PEOE_VSA4 if self.PEOE_VSA4 is not None else 0.0,
          "PEOE_VSA5": self.PEOE_VSA5 if self.PEOE_VSA5 is not None else 0.0,
          "PEOE_VSA6": self.PEOE_VSA6 if self.PEOE_VSA6 is not None else 0.0,
          "PEOE_VSA7": self.PEOE_VSA7 if self.PEOE_VSA7 is not None else 0.0,
          "PEOE_VSA8": self.PEOE_VSA8 if self.PEOE_VSA8 is not None else 0.0,
          "PEOE_VSA9": self.PEOE_VSA9 if self.PEOE_VSA9 is not None else 0.0,
          "PEOE_VSA10": self.PEOE_VSA10 if self.PEOE_VSA10 is not None else 0.0,
          "PEOE_VSA11": self.PEOE_VSA11 if self.PEOE_VSA11 is not None else 0.0,
          "PEOE_VSA12": self.PEOE_VSA12 if self.PEOE_VSA12 is not None else 0.0,
          "PEOE_VSA13": self.PEOE_VSA13 if self.PEOE_VSA13 is not None else 0.0,
          "PEOE_VSA14": self.PEOE_VSA14 if self.PEOE_VSA14 is not None else 0.0,
          "qed": self.qed if self.qed is not None else 0.0,
          "RingCount": self.RingCount if self.RingCount is not None else 0,
          "SMR_VSA1": self.SMR_VSA1 if self.SMR_VSA1 is not None else 0.0,
          "SMR_VSA2": self.SMR_VSA2 if self.SMR_VSA2 is not None else 0.0,
          "SMR_VSA3": self.SMR_VSA3 if self.SMR_VSA3 is not None else 0.0,
          "SMR_VSA4": self.SMR_VSA4 if self.SMR_VSA4 is not None else 0.0,
          "SMR_VSA5": self.SMR_VSA5 if self.SMR_VSA5 is not None else 0.0,
          "SMR_VSA6": self.SMR_VSA6 if self.SMR_VSA6 is not None else 0.0,
          "SMR_VSA7": self.SMR_VSA7 if self.SMR_VSA7 is not None else 0.0,
          "SMR_VSA8": self.SMR_VSA8 if self.SMR_VSA8 is not None else 0.0,
          "SMR_VSA9": self.SMR_VSA9 if self.SMR_VSA9 is not None else 0.0,
          "SMR_VSA10": self.SMR_VSA10 if self.SMR_VSA10 is not None else 0.0,
          "SlogP_VSA1": self.SlogP_VSA1 if self.SlogP_VSA1 is not None else 0.0,
          "SlogP_VSA2": self.SlogP_VSA2 if self.SlogP_VSA2 is not None else 0.0,
          "SlogP_VSA3": self.SlogP_VSA3 if self.SlogP_VSA3 is not None else 0.0,
          "SlogP_VSA4": self.SlogP_VSA4 if self.SlogP_VSA4 is not None else 0.0,
          "SlogP_VSA5": self.SlogP_VSA5 if self.SlogP_VSA5 is not None else 0.0,
          "SlogP_VSA6": self.SlogP_VSA6 if self.SlogP_VSA6 is not None else 0.0,
          "SlogP_VSA7": self.SlogP_VSA7 if self.SlogP_VSA7 is not None else 0.0,
          "SlogP_VSA8": self.SlogP_VSA8 if self.SlogP_VSA8 is not None else 0.0,
          "SlogP_VSA9": self.SlogP_VSA9 if self.SlogP_VSA9 is not None else 0.0,
          "SlogP_VSA10": self.SlogP_VSA10 if self.SlogP_VSA10 is not None else 0.0,
          "SlogP_VSA11": self.SlogP_VSA11 if self.SlogP_VSA11 is not None else 0.0,
          "SlogP_VSA12": self.SlogP_VSA12 if self.SlogP_VSA12 is not None else 0.0,
          "TPSA": self.TPSA if self.TPSA is not None else 0.0,
          "VSA_EState1": self.VSA_EState1 if self.VSA_EState1 is not None else 0.0,
          "VSA_EState2": self.VSA_EState2 if self.VSA_EState2 is not None else 0.0,
          "VSA_EState3": self.VSA_EState3 if self.VSA_EState3 is not None else 0.0,
          "VSA_EState4": self.VSA_EState4 if self.VSA_EState4 is not None else 0.0,
          "VSA_EState5": self.VSA_EState5 if self.VSA_EState5 is not None else 0.0,
          "VSA_EState6": self.VSA_EState6 if self.VSA_EState6 is not None else 0.0,
          "VSA_EState7": self.VSA_EState7 if self.VSA_EState7 is not None else 0.0,
          "VSA_EState8": self.VSA_EState8 if self.VSA_EState8 is not None else 0.0,
          "VSA_EState9": self.VSA_EState9 if self.VSA_EState9 is not None else 0.0,
          "VSA_EState10": self.VSA_EState10 if self.VSA_EState10 is not None else 0.0,
          "AUTOCORR3D_1": self.AUTOCORR3D_1 if self.AUTOCORR3D_1 is not None else 0.0,
          "AUTOCORR3D_2": self.AUTOCORR3D_2 if self.AUTOCORR3D_2 is not None else 0.0,
          "AUTOCORR3D_3": self.AUTOCORR3D_3 if self.AUTOCORR3D_3 is not None else 0.0,
          "AUTOCORR3D_4": self.AUTOCORR3D_4 if self.AUTOCORR3D_4 is not None else 0.0,
          "AUTOCORR3D_5": self.AUTOCORR3D_5 if self.AUTOCORR3D_5 is not None else 0.0,
          "AUTOCORR3D_6": self.AUTOCORR3D_6 if self.AUTOCORR3D_6 is not None else 0.0,
          "AUTOCORR3D_7": self.AUTOCORR3D_7 if self.AUTOCORR3D_7 is not None else 0.0,
          "AUTOCORR3D_8": self.AUTOCORR3D_8 if self.AUTOCORR3D_8 is not None else 0.0,
          "AUTOCORR3D_9": self.AUTOCORR3D_9 if self.AUTOCORR3D_9 is not None else 0.0,
          "AUTOCORR3D_10": self.AUTOCORR3D_10 if self.AUTOCORR3D_10 is not None else 0.0,
          "AUTOCORR3D_11": self.AUTOCORR3D_11 if self.AUTOCORR3D_11 is not None else 0.0,
          "AUTOCORR3D_12": self.AUTOCORR3D_12 if self.AUTOCORR3D_12 is not None else 0.0,
          "AUTOCORR3D_13": self.AUTOCORR3D_13 if self.AUTOCORR3D_13 is not None else 0.0,
          "AUTOCORR3D_14": self.AUTOCORR3D_14 if self.AUTOCORR3D_14 is not None else 0.0,
          "AUTOCORR3D_15": self.AUTOCORR3D_15 if self.AUTOCORR3D_15 is not None else 0.0,
          "AUTOCORR3D_16": self.AUTOCORR3D_16 if self.AUTOCORR3D_16 is not None else 0.0,
          "AUTOCORR3D_17": self.AUTOCORR3D_17 if self.AUTOCORR3D_17 is not None else 0.0,
          "AUTOCORR3D_18": self.AUTOCORR3D_18 if self.AUTOCORR3D_18 is not None else 0.0,
          "AUTOCORR3D_19": self.AUTOCORR3D_19 if self.AUTOCORR3D_19 is not None else 0.0,
          "AUTOCORR3D_20": self.AUTOCORR3D_20 if self.AUTOCORR3D_20 is not None else 0.0,
          "AUTOCORR3D_21": self.AUTOCORR3D_21 if self.AUTOCORR3D_21 is not None else 0.0,
          "AUTOCORR3D_22": self.AUTOCORR3D_22 if self.AUTOCORR3D_22 is not None else 0.0,
          "AUTOCORR3D_23": self.AUTOCORR3D_23 if self.AUTOCORR3D_23 is not None else 0.0,
          "AUTOCORR3D_24": self.AUTOCORR3D_24 if self.AUTOCORR3D_24 is not None else 0.0,
          "AUTOCORR3D_25": self.AUTOCORR3D_25 if self.AUTOCORR3D_25 is not None else 0.0,
          "AUTOCORR3D_26": self.AUTOCORR3D_26 if self.AUTOCORR3D_26 is not None else 0.0,
          "AUTOCORR3D_27": self.AUTOCORR3D_27 if self.AUTOCORR3D_27 is not None else 0.0,
          "AUTOCORR3D_28": self.AUTOCORR3D_28 if self.AUTOCORR3D_28 is not None else 0.0,
          "AUTOCORR3D_29": self.AUTOCORR3D_29 if self.AUTOCORR3D_29 is not None else 0.0,
          "AUTOCORR3D_30": self.AUTOCORR3D_30 if self.AUTOCORR3D_30 is not None else 0.0,
          "AUTOCORR3D_31": self.AUTOCORR3D_31 if self.AUTOCORR3D_31 is not None else 0.0,
          "AUTOCORR3D_32": self.AUTOCORR3D_32 if self.AUTOCORR3D_32 is not None else 0.0,
          "AUTOCORR3D_33": self.AUTOCORR3D_33 if self.AUTOCORR3D_33 is not None else 0.0,
          "AUTOCORR3D_34": self.AUTOCORR3D_34 if self.AUTOCORR3D_34 is not None else 0.0,
          "AUTOCORR3D_35": self.AUTOCORR3D_35 if self.AUTOCORR3D_35 is not None else 0.0,
          "AUTOCORR3D_36": self.AUTOCORR3D_36 if self.AUTOCORR3D_36 is not None else 0.0,
          "AUTOCORR3D_37": self.AUTOCORR3D_37 if self.AUTOCORR3D_37 is not None else 0.0,
          "AUTOCORR3D_38": self.AUTOCORR3D_38 if self.AUTOCORR3D_38 is not None else 0.0,
          "AUTOCORR3D_39": self.AUTOCORR3D_39 if self.AUTOCORR3D_39 is not None else 0.0,
          "AUTOCORR3D_40": self.AUTOCORR3D_40 if self.AUTOCORR3D_40 is not None else 0.0,
          "AUTOCORR3D_41": self.AUTOCORR3D_41 if self.AUTOCORR3D_41 is not None else 0.0,
          "AUTOCORR3D_42": self.AUTOCORR3D_42 if self.AUTOCORR3D_42 is not None else 0.0,
          "AUTOCORR3D_43": self.AUTOCORR3D_43 if self.AUTOCORR3D_43 is not None else 0.0,
          "AUTOCORR3D_44": self.AUTOCORR3D_44 if self.AUTOCORR3D_44 is not None else 0.0,
          "AUTOCORR3D_45": self.AUTOCORR3D_45 if self.AUTOCORR3D_45 is not None else 0.0,
          "AUTOCORR3D_46": self.AUTOCORR3D_46 if self.AUTOCORR3D_46 is not None else 0.0,
          "AUTOCORR3D_47": self.AUTOCORR3D_47 if self.AUTOCORR3D_47 is not None else 0.0,
          "AUTOCORR3D_48": self.AUTOCORR3D_48 if self.AUTOCORR3D_48 is not None else 0.0,
          "AUTOCORR3D_49": self.AUTOCORR3D_49 if self.AUTOCORR3D_49 is not None else 0.0,
          "AUTOCORR3D_50": self.AUTOCORR3D_50 if self.AUTOCORR3D_50 is not None else 0.0,
          "AUTOCORR3D_51": self.AUTOCORR3D_51 if self.AUTOCORR3D_51 is not None else 0.0,
          "AUTOCORR3D_52": self.AUTOCORR3D_52 if self.AUTOCORR3D_52 is not None else 0.0,
          "AUTOCORR3D_53": self.AUTOCORR3D_53 if self.AUTOCORR3D_53 is not None else 0.0,
          "AUTOCORR3D_54": self.AUTOCORR3D_54 if self.AUTOCORR3D_54 is not None else 0.0,
          "AUTOCORR3D_55": self.AUTOCORR3D_55 if self.AUTOCORR3D_55 is not None else 0.0,
          "AUTOCORR3D_56": self.AUTOCORR3D_56 if self.AUTOCORR3D_56 is not None else 0.0,
          "AUTOCORR3D_57": self.AUTOCORR3D_57 if self.AUTOCORR3D_57 is not None else 0.0,
          "AUTOCORR3D_58": self.AUTOCORR3D_58 if self.AUTOCORR3D_58 is not None else 0.0,
          "AUTOCORR3D_59": self.AUTOCORR3D_59 if self.AUTOCORR3D_59 is not None else 0.0,
          "AUTOCORR3D_60": self.AUTOCORR3D_60 if self.AUTOCORR3D_60 is not None else 0.0,
          "AUTOCORR3D_61": self.AUTOCORR3D_61 if self.AUTOCORR3D_61 is not None else 0.0,
          "AUTOCORR3D_62": self.AUTOCORR3D_62 if self.AUTOCORR3D_62 is not None else 0.0,
          "AUTOCORR3D_63": self.AUTOCORR3D_63 if self.AUTOCORR3D_63 is not None else 0.0,
          "AUTOCORR3D_64": self.AUTOCORR3D_64 if self.AUTOCORR3D_64 is not None else 0.0,
          "AUTOCORR3D_65": self.AUTOCORR3D_65 if self.AUTOCORR3D_65 is not None else 0.0,
          "AUTOCORR3D_66": self.AUTOCORR3D_66 if self.AUTOCORR3D_66 is not None else 0.0,
          "AUTOCORR3D_67": self.AUTOCORR3D_67 if self.AUTOCORR3D_67 is not None else 0.0,
          "AUTOCORR3D_68": self.AUTOCORR3D_68 if self.AUTOCORR3D_68 is not None else 0.0,
          "AUTOCORR3D_69": self.AUTOCORR3D_69 if self.AUTOCORR3D_69 is not None else 0.0,
          "AUTOCORR3D_70": self.AUTOCORR3D_70 if self.AUTOCORR3D_70 is not None else 0.0,
          "AUTOCORR3D_71": self.AUTOCORR3D_71 if self.AUTOCORR3D_71 is not None else 0.0,
          "AUTOCORR3D_72": self.AUTOCORR3D_72 if self.AUTOCORR3D_72 is not None else 0.0,
          "AUTOCORR3D_73": self.AUTOCORR3D_73 if self.AUTOCORR3D_73 is not None else 0.0,
          "AUTOCORR3D_74": self.AUTOCORR3D_74 if self.AUTOCORR3D_74 is not None else 0.0,
          "AUTOCORR3D_75": self.AUTOCORR3D_75 if self.AUTOCORR3D_75 is not None else 0.0,
          "AUTOCORR3D_76": self.AUTOCORR3D_76 if self.AUTOCORR3D_76 is not None else 0.0,
          "AUTOCORR3D_77": self.AUTOCORR3D_77 if self.AUTOCORR3D_77 is not None else 0.0,
          "AUTOCORR3D_78": self.AUTOCORR3D_78 if self.AUTOCORR3D_78 is not None else 0.0,
          "AUTOCORR3D_79": self.AUTOCORR3D_79 if self.AUTOCORR3D_79 is not None else 0.0,
          "AUTOCORR3D_80": self.AUTOCORR3D_80 if self.AUTOCORR3D_80 is not None else 0.0,
          "Asphericity": self.Asphericity if self.Asphericity is not None else 0.0,
          "Eccentricity": self.Eccentricity if self.Eccentricity is not None else 0.0,
          "InertialShapeFactor": self.InertialShapeFactor if self.InertialShapeFactor is not None else 0.0,
          "NPR1": self.NPR1 if self.NPR1 is not None else 0.0,
          "NPR2": self.NPR2 if self.NPR2 is not None else 0.0,
          "PMI1": self.PMI1 if self.PMI1 is not None else 0.0,
          "PMI2": self.PMI2 if self.PMI2 is not None else 0.0,
          "PMI3": self.PMI3 if self.PMI3 is not None else 0.0,
          "RadiusOfGyration": self.RadiusOfGyration if self.RadiusOfGyration is not None else 0.0,
          "SpherocityIndex": self.SpherocityIndex if self.SpherocityIndex is not None else 0.0
        }
        
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
                    return errors.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", "warn")
                # Warns the user that the file will be overwritten
                _ = errors.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!")

            try:
                # Create a lock for multithreading
                lock = Lock()
                with lock:
                    # Open the file for writing
                    with open(outputJson, 'w') as outfile:
                        # Write the json file
                        json.dump(self.__safe_to_dict(), outfile)
                return errors.ok()
            except Exception as e:
                return errors.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.")
        except Exception as e:
            return errors.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", "error")

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
        if None in [self.name, self.path, self.AUTOCORR2D_1, self.AUTOCORR2D_2, self.AUTOCORR2D_3, self.AUTOCORR2D_4, self.AUTOCORR2D_5, self.AUTOCORR2D_6, self.AUTOCORR2D_7, self.AUTOCORR2D_8, self.AUTOCORR2D_9, self.AUTOCORR2D_10, self.AUTOCORR2D_11, self.AUTOCORR2D_12, self.AUTOCORR2D_13, self.AUTOCORR2D_14, self.AUTOCORR2D_15, self.AUTOCORR2D_16, self.AUTOCORR2D_17, self.AUTOCORR2D_18, self.AUTOCORR2D_19, self.AUTOCORR2D_20, self.AUTOCORR2D_21, self.AUTOCORR2D_22, self.AUTOCORR2D_23, self.AUTOCORR2D_24, self.AUTOCORR2D_25, self.AUTOCORR2D_26, self.AUTOCORR2D_27, self.AUTOCORR2D_28, self.AUTOCORR2D_29, self.AUTOCORR2D_30, self.AUTOCORR2D_31, self.AUTOCORR2D_32, self.AUTOCORR2D_33, self.AUTOCORR2D_34, self.AUTOCORR2D_35, self.AUTOCORR2D_36, self.AUTOCORR2D_37, self.AUTOCORR2D_38, self.AUTOCORR2D_39, self.AUTOCORR2D_40, self.AUTOCORR2D_41, self.AUTOCORR2D_42, self.AUTOCORR2D_43, self.AUTOCORR2D_44, self.AUTOCORR2D_45, self.AUTOCORR2D_46, self.AUTOCORR2D_47, self.AUTOCORR2D_48, self.AUTOCORR2D_49, self.AUTOCORR2D_50, self.AUTOCORR2D_51, self.AUTOCORR2D_52, self.AUTOCORR2D_53, self.AUTOCORR2D_54, self.AUTOCORR2D_55, self.AUTOCORR2D_56, self.AUTOCORR2D_57, self.AUTOCORR2D_58, self.AUTOCORR2D_59, self.AUTOCORR2D_60, self.AUTOCORR2D_61, self.AUTOCORR2D_62, self.AUTOCORR2D_63, self.AUTOCORR2D_64, self.AUTOCORR2D_65, self.AUTOCORR2D_66, self.AUTOCORR2D_67, self.AUTOCORR2D_68, self.AUTOCORR2D_69, self.AUTOCORR2D_70, self.AUTOCORR2D_71, self.AUTOCORR2D_72, self.AUTOCORR2D_73, self.AUTOCORR2D_74, self.AUTOCORR2D_75, self.AUTOCORR2D_76, self.AUTOCORR2D_77, self.AUTOCORR2D_78, self.AUTOCORR2D_79, self.AUTOCORR2D_80, self.AUTOCORR2D_81, self.AUTOCORR2D_82, self.AUTOCORR2D_83, self.AUTOCORR2D_84, self.AUTOCORR2D_85, self.AUTOCORR2D_86, self.AUTOCORR2D_87, self.AUTOCORR2D_88, self.AUTOCORR2D_89, self.AUTOCORR2D_90, self.AUTOCORR2D_91, self.AUTOCORR2D_92, self.AUTOCORR2D_93, self.AUTOCORR2D_94, self.AUTOCORR2D_95, self.AUTOCORR2D_96, self.AUTOCORR2D_97, self.AUTOCORR2D_98, self.AUTOCORR2D_99, self.AUTOCORR2D_100, self.AUTOCORR2D_101, self.AUTOCORR2D_102, self.AUTOCORR2D_103, self.AUTOCORR2D_104, self.AUTOCORR2D_105, self.AUTOCORR2D_106, self.AUTOCORR2D_107, self.AUTOCORR2D_108, self.AUTOCORR2D_109, self.AUTOCORR2D_110, self.AUTOCORR2D_111, self.AUTOCORR2D_112, self.AUTOCORR2D_113, self.AUTOCORR2D_114, self.AUTOCORR2D_115, self.AUTOCORR2D_116, self.AUTOCORR2D_117, self.AUTOCORR2D_118, self.AUTOCORR2D_119, self.AUTOCORR2D_120, self.AUTOCORR2D_121, self.AUTOCORR2D_122, self.AUTOCORR2D_123, self.AUTOCORR2D_124, self.AUTOCORR2D_125, self.AUTOCORR2D_126, self.AUTOCORR2D_127, self.AUTOCORR2D_128, self.AUTOCORR2D_129, self.AUTOCORR2D_130, self.AUTOCORR2D_131, self.AUTOCORR2D_132, self.AUTOCORR2D_133, self.AUTOCORR2D_134, self.AUTOCORR2D_135, self.AUTOCORR2D_136, self.AUTOCORR2D_137, self.AUTOCORR2D_138, self.AUTOCORR2D_139, self.AUTOCORR2D_140, self.AUTOCORR2D_141, self.AUTOCORR2D_142, self.AUTOCORR2D_143, self.AUTOCORR2D_144, self.AUTOCORR2D_145, self.AUTOCORR2D_146, self.AUTOCORR2D_147, self.AUTOCORR2D_148, self.AUTOCORR2D_149, self.AUTOCORR2D_150, self.AUTOCORR2D_151, self.AUTOCORR2D_152, self.AUTOCORR2D_153, self.AUTOCORR2D_154, self.AUTOCORR2D_155, self.AUTOCORR2D_156, self.AUTOCORR2D_157, self.AUTOCORR2D_158, self.AUTOCORR2D_159, self.AUTOCORR2D_160, self.AUTOCORR2D_161, self.AUTOCORR2D_162, self.AUTOCORR2D_163, self.AUTOCORR2D_164, self.AUTOCORR2D_165, self.AUTOCORR2D_166, self.AUTOCORR2D_167, self.AUTOCORR2D_168, self.AUTOCORR2D_169, self.AUTOCORR2D_170, self.AUTOCORR2D_171, self.AUTOCORR2D_172, self.AUTOCORR2D_173, self.AUTOCORR2D_174, self.AUTOCORR2D_175, self.AUTOCORR2D_176, self.AUTOCORR2D_177, self.AUTOCORR2D_178, self.AUTOCORR2D_179, self.AUTOCORR2D_180, self.AUTOCORR2D_181, self.AUTOCORR2D_182, self.AUTOCORR2D_183, self.AUTOCORR2D_184, self.AUTOCORR2D_185, self.AUTOCORR2D_186, self.AUTOCORR2D_187, self.AUTOCORR2D_188, self.AUTOCORR2D_189, self.AUTOCORR2D_190, self.AUTOCORR2D_191, self.AUTOCORR2D_192, self.BCUT2D_CHGHI, self.BCUT2D_CHGLO, self.BCUT2D_LOGPHI, self.BCUT2D_LOGPLOW, self.BCUT2D_MRHI, self.BCUT2D_MRLOW, self.BCUT2D_MWHI, self.BCUT2D_MWLOW, self.BalabanJ, self.BertzCT, self.Chi0, self.Chi0n, self.Chi0v, self.Chi1, self.Chi1n, self.Chi1v, self.Chi2n, self.Chi2v, self.Chi3n, self.Chi3v, self.Chi4n, self.Chi4v, self.EState_VSA1, self.EState_VSA2, self.EState_VSA3, self.EState_VSA4, self.EState_VSA5, self.EState_VSA6, self.EState_VSA7, self.EState_VSA8, self.EState_VSA9, self.EState_VSA10, self.EState_VSA11, self.MaxAbsEStateIndex, self.MaxEStateIndex, self.MinAbsEStateIndex, self.MinEStateIndex, self.ExactMolWt, self.FpDensityMorgan1, self.FpDensityMorgan2, self.FpDensityMorgan3, self.fr_Al_COO, self.fr_Al_OH, self.fr_Al_OH_noTert, self.fr_ArN, self.fr_Ar_COO, self.fr_Ar_N, self.fr_Ar_NH, self.fr_Ar_OH, self.fr_COO, self.fr_COO2, self.fr_C_O, self.fr_C_O_noCOO, self.fr_C_S, self.fr_HOCCN, self.fr_Imine, self.fr_NH0, self.fr_NH1, self.fr_NH2, self.fr_N_O, self.fr_Ndealkylation1, self.fr_Ndealkylation2, self.fr_Nhpyrrole, self.fr_SH, self.fr_aldehyde, self.fr_alkyl_carbamate, self.fr_alkyl_halide, self.fr_allylic_oxid, self.fr_amide, self.fr_amidine, self.fr_aniline, self.fr_aryl_methyl, self.fr_azide, self.fr_azo, self.fr_barbitur, self.fr_benzene, self.fr_benzodiazepine, self.fr_bicyclic, self.fr_diazo, self.fr_dihydropyridine, self.fr_epoxide, self.fr_ester, self.fr_ether, self.fr_furan, self.fr_guanido, self.fr_halogen, self.fr_hdrzine, self.fr_hdrzone, self.fr_imidazole, self.fr_imide, self.fr_isocyan, self.fr_isothiocyan, self.fr_ketone, self.fr_ketone_Topliss, self.fr_lactam, self.fr_lactone, self.fr_methoxy, self.fr_morpholine, self.fr_nitrile, self.fr_nitro, self.fr_nitro_arom, self.fr_nitro_arom_nonortho, self.fr_nitroso, self.fr_oxazole, self.fr_oxime, self.fr_para_hydroxylation, self.fr_phenol, self.fr_phenol_noOrthoHbond, self.fr_phos_acid, self.fr_phos_ester, self.fr_piperdine, self.fr_piperzine, self.fr_priamide, self.fr_prisulfonamd, self.fr_pyridine, self.fr_quatN, self.fr_sulfide, self.fr_sulfonamd, self.fr_sulfone, self.fr_term_acetylene, self.fr_tetrazole, self.fr_thiazole, self.fr_thiocyan, self.fr_thiophene, self.fr_unbrch_alkane, self.fr_urea, self.FractionCSP3, self.HallKierAlpha, self.HeavyAtomMolWt, self.HeavyAtomCount, self.Ipc, self.Kappa1, self.Kappa2, self.Kappa3, self.LabuteASA, self.MaxAbsPartialCharge, self.MaxPartialCharge, self.MinAbsPartialCharge, self.MinPartialCharge, self.MolLogP, self.MolMR, self.MolWt, self.NHOHCount, self.NOCount, self.NumAliphaticCarbocycles, self.NumAliphaticHeterocycles, self.NumAliphaticRings, self.NumAromaticCarbocycles, self.NumAromaticHeterocycles, self.NumAromaticRings, self.NumHAcceptors, self.NumHDonors, self.NumHeteroatoms, self.NumRadicalElectrons, self.NumRotatableBonds, self.NumSaturatedCarbocycles, self.NumSaturatedHeterocycles, self.NumSaturatedRings, self.NumValenceElectrons, self.PEOE_VSA1, self.PEOE_VSA2, self.PEOE_VSA3, self.PEOE_VSA4, self.PEOE_VSA5, self.PEOE_VSA6, self.PEOE_VSA7, self.PEOE_VSA8, self.PEOE_VSA9, self.PEOE_VSA10, self.PEOE_VSA11, self.PEOE_VSA12, self.PEOE_VSA13, self.PEOE_VSA14, self.qed, self.RingCount, self.SMR_VSA1, self.SMR_VSA2, self.SMR_VSA3, self.SMR_VSA4, self.SMR_VSA5, self.SMR_VSA6, self.SMR_VSA7, self.SMR_VSA8, self.SMR_VSA9, self.SMR_VSA10, self.SlogP_VSA1, self.SlogP_VSA2, self.SlogP_VSA3, self.SlogP_VSA4, self.SlogP_VSA5, self.SlogP_VSA6, self.SlogP_VSA7, self.SlogP_VSA8, self.SlogP_VSA9, self.SlogP_VSA10, self.SlogP_VSA11, self.SlogP_VSA12, self.TPSA, self.VSA_EState1, self.VSA_EState2, self.VSA_EState3, self.VSA_EState4, self.VSA_EState5, self.VSA_EState6, self.VSA_EState7, self.VSA_EState8, self.VSA_EState9, self.VSA_EState10]:
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

    def is_same_molecule(self, molecule: Union[rdkit.Chem.rdchem.Mol, Ligand], sanitize: bool = True) -> Union[bool, int]: # type: ignore
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
        ligandMACCSSKeys = MACCSkeys.GenMACCSKeys(self.molecule)
        # Check if the type of the molecule is a Ligand
        if type(molecule) == Ligand:
            # If yes, get its MACCSKeys
            targetMACCSSKeys = MACCSkeys.GenMACCSKeys(molecule)
        # Otherwise check if it is a Chem.rdchem.Mol object
        elif type(molecule) == Chem.rdchem.Mol:
            # If it is, get its smiles using the Ligand public function, get_smiles()
            mol = loadMol(molecule, sanitize = sanitize)
            targetMACCSSKeys = MACCSkeys.GenMACCSKeys(molecule)
        # If is neither both types above
        else:
            # Return an error
            return errors.wrong_type(f"The provided variable is a '{type(molecule)}' and was expected a 'rdkit.Chem.rdchem.Mol' or 'ocl.Ligand'.")
        # Check if the Fingerprints are the same using the Tanimoto similarity
        if DataStructs.FingerprintSimilarity(ligandMACCSSKeys, targetMACCSSKeys) == 1.0:
            # If they are the same, return True
            return True
        # Otherwise (they are not the same)
        else:
            # Return False
            return False

    def is_same_molecule_SMILES(self, molecule: Union[rdkit.Chem.rdchem.Mol, Ligand], sanitize: bool = True) -> Union[bool, int]: # type: ignore
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

        # Get the smiles for the ligand object
        molSmiles = self.to_smiles()
        # Check if the type of the molecule is a Ligand
        if type(molecule) == Ligand:
            # If yes, use the to_smiles Ligand method
            targetMolSmiles = molecule.to_smiles()
            targetMolMorganFp1 = molecule.FpDensityMorgan1
            targetMolMorganFp2 = molecule.FpDensityMorgan2
            targetMolMorganFp3 = molecule.FpDensityMorgan3

        # Otherwise check if it is a Chem.rdchem.Mol object
        elif type(molecule) == Chem.rdchem.Mol:
            # If it is, get its smiles using the Ligand public function, get_smiles()
            mol = loadMol(molecule, sanitize = sanitize)
            targetMolSmiles = get_smiles(mol)
            targetMolMorganFp1 = findFpDensityMorgan1(mol)
            targetMolMorganFp2 = findFpDensityMorgan2(mol)
            targetMolMorganFp3 = findFpDensityMorgan3(mol)

        # If is neither both types above
        else:
            # Return an error
            return errors.wrong_type(f"The provided variable is a '{type(molecule)}' and was expected a 'rdkit.Chem.rdchem.Mol' or 'ocl.Ligand'.")
        # Check if both smiles and MorganFp 1, 2 and 3 are the same
        if molSmiles == targetMolSmiles and self.FpDensityMorgan1 == targetMolMorganFp1 and self.FpDensityMorgan2 == targetMolMorganFp2 and self.FpDensityMorgan3 == targetMolMorganFp3:
            # If they are the same, return True
            return True
        # Otherwise (they are not the same)
        else:
            # Return False
            return False

    def get_centroid(self, sanitize: bool = True) -> rdkit.Geometry.rdGeometry.Point3D: # type: ignore
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

    def create_box(self, centroid: Union[Tuple[float, float, float], None] = None, savePath: str = "", boxLength: float = 2.9, overwrite: bool = False) -> Union[int, None]:
        '''Create a box file to be used by docking software.

        Parameters
        ----------
        centroid : tuple | None, optional
            The centroid of the box. If not provided, the centroid of the molecule will be used. (default is None)
        savePath : str, optional
            The path to save the box file. If not provided, the box file will be saved in the same path as the molecule. (default is "", which turns into self.boxPath)
        boxLength : float, optional
            The length of the box. (default is 2.9)
        overwrite : bool, optional
            Flag to allow, or not, the overwrite of the box file. (default is False)

        Returns
        -------
        int | None
            If the box file was created, return None. If fails, return the exit code of the command (based on the Error.py code table).
        '''

        # Check if the box file already exists
        if os.path.isfile(savePath) and not overwrite:
            # If it exists and the overwrite flag is False, return an error
            return errors.file_exists(f"The box file '{savePath}' already exists. If you want to overwrite it, set the 'overwrite' flag to True.")
            
        # If the centroid is not defined
        if not centroid:
            # Compute it
            centroid = self.get_centroid()

        # Check if the centroid is the type rdkit.Geometry.rdGeometry.Point3D
        if type(centroid) == rdkit.Geometry.rdGeometry.Point3D: # type: ignore
            centroid = (centroid.x, centroid.y, centroid.z) # type: ignore

        # Get the partial size for each axis (to determine how much should be expanded in each direction)
        partialSize = (boxLength * self.RadiusOfGyration) / 2 # type: ignore

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

        # If the savePath is not defined
        if not savePath:
            # Set it as the same dir as the ligand
            savePath = os.path.join(os.path.split(self.path)[0], 'boxes')
            # If the savePath does not exist
            if not os.path.exists(savePath):
                # Create it
                _ = ocff.safe_create_dir(savePath)
        else:
            # If the savePath does not exist, warn the user
            if not os.path.exists(savePath):
                _ =  errors.dir_does_not_exist(f"The savePath '{savePath}' does not exist. Creating it.", level = "error")
                os.mkdir(savePath)

        # Write out the box file (following the one given in the DUD-E database)
        with open(f"{savePath}/box0.pdb", 'w') as f:
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
def splitMolecules(molecule: str, outputDir: str = "", prefix: str = "ligand") -> List[str]:
    '''Given a molecule file, checks if it has more than one ligand, if positive, splits the file into multiple single molecule files. Uses openbabel python library. TODO: Make this function work better with the new database structure.

    Parameters
    ----------
    molecule : str
        The path to the molecule file.
    outputDir : str, optional
        The path to the output directory, by default ""
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
    extension = ocvalidation.validate_obabel_extension(molecule)

    # If the outputDir is not defined
    if not outputDir:
        # Set it as the same dir as the ligand
        outputDir = f"{os.path.split(os.path.abspath(molecule))[0]}/compounds"

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

def multipleMoleculesSDF(molecule: rdkit.Chem.rdchem.Mol) -> List[Ligand]: # type: ignore
    '''Parse a .sdf or .mol2 file with multiple molecules returning a list of ligands.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule object.

    Returns
    -------
    List[Ligand]
        A list of ligands.
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
            else:
                # This case the return code is suppressed because it is needed to return None in case of failure
                _ = errors.wrong_type(message=f"The molecule file MUST be the .sdf format!", level="error")
        else:
            # File does not exist
            _ = errors.file_do_not_exist(message=f"The file '{molecule}' does not exist!", level="error")
    else:
        # This case the return code is suppressed because it is needed to return None in case of failure
        _ = errors.wrong_type(message=f"The molecule file path MUST be a string!", level="error")

    return ligands

def loadMol(molecule: Union[str, rdkit.Chem.rdchem.Mol], sanitize: bool = True) -> rdkit.Chem.rdchem.Mol: # type: ignore
    '''Load a molecule pdb/sdf/mol/mol2 if a path is provided or just assign the Mol object to the molecule.

    Parameters
    ----------
    molecule : str/rdkit.Chem.rdchem.Mol
        The molecule path or the Mol object.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The molecule object.
    '''

    # Check if the type of the variable molecule is a string or a rdkit.Chem.rdchem.Mol
    if type(molecule) == rdkit.Chem.rdchem.Mol: # type: ignore
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
                    m = rdkit.Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = False) # type: ignore
                    try:
                        # Turn off the property cachea
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True) # type: ignore
                        # Return the sanitized molecule
                        return molecule, m
                    except Exception as e:
                        _ = errors.parse_molecule(f"The molecule '{molecule}' could not be parsed. Error: {e}", "error")
                        return molecule, None

                return molecule, rdkit.Chem.rdmolfiles.MolFromMol2File(molecule, sanitize = True) # type: ignore
            else:
                # Since is needed to convert the ligand, create the output path
                outputMoleculePath = f"{os.path.dirname(molecule)}/ligand.mol2"

                # Only process if is not smiles format, because it demands a different approach
                if extension not in [".smi", ".smiles"]:
                    # Process the ligand
                    occonversion.convertMols(molecule, outputMoleculePath)

                if extension == ".pdb":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule
                        m = rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule, sanitize = False) # type: ignore
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True) # type: ignore
                        # Return the sanitized molecule
                        return molecule, m

                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromPDBFile(molecule, sanitize = True) # type: ignore
                elif extension == ".sdf":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule (Since the sdf file can hold more than one molecule...)
                        mol = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = False) # type: ignore
                        if len(mol) > 1:
                            ocprint.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function splitMolecules to split the ligand into multiple ligand files. Otherwise just the first molecule will be processed.")
                        # Get the first molecule
                        m = mol[0]
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True) # type: ignore
                        # Return the sanitized molecule
                        return molecule, m

                    # Since the sdf file can hold more than one molecule...
                    mols = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = True) # type: ignore
                    # If has multiple molecules, indicate the user to use the right function
                    if len(mols) > 1:
                        ocprint.print_warning("This sdf has more than one molecule!! If you want to parse all the molecules within this file use the function splitMolecules to split the ligand into multiple ligand files. Otherwise just the first molecule will be processed.")

                    # Return just the first molecule
                    return outputMoleculePath, mols[0]
                elif extension == ".mol":
                    # If sanitize is off
                    if not sanitize:
                        # Load the molecule
                        m = rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = False) # type: ignore
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True) # type: ignore
                        # Return the sanitized molecule
                        return molecule, m

                    return outputMoleculePath, rdkit.Chem.rdmolfiles.MolFromMolFile(molecule, sanitize = True) # type: ignore
                elif extension in [".smi", ".smiles"]:
                    # Read the smiles file into a string
                    with open(molecule, 'r') as file:
                      smiles = file.read().strip()
                    # Initialise the salt remover
                    remover = SaltRemover()
                    # Load the molecule
                    m = rdkit.Chem.rdmolfiles.MolFromSmiles(smiles, sanitize = sanitize) # type: ignore
                    # Remove the salts
                    m = remover.StripMol(m)
                    # Add the hydrogens
                    m = Chem.AddHs(m) # type: ignore
                    # Embed the molecule
                    _ = AllChem.EmbedMolecule(m, AllChem.ETKDG()) # type: ignore
                    # Optimize the molecule
                    _ = AllChem.UFFOptimizeMolecule(m) # type: ignore

                    occonversion.convertMolsFromString("", outputMoleculePath, mol = m)
                    
                    # Find its name (without extension)
                    name = os.path.splitext(os.path.basename(molecule))[0]
                    # Set its name
                    m.SetProp("_Name", name)
                    # If sanitize is off
                    if not sanitize:
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True) # type: ignore
                    # Return the molecule
                    return outputMoleculePath, m
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

def read_descriptors_from_json(path: str, returnData: bool = False, returnVaex: bool = False) -> Union[Dict[str, Union[str, float, int]], Tuple[Union[str, float, int]], vdf.DataFrameLocal, None]:
    '''Read the descriptors from a json file.

    Parameters
    ----------
    path : str
        The path to the json file.
    returnData : bool, optional
        If True, returns a dictionary with the descriptors. It only works when returnVaex is set to False, by default False.
    returnVaex : bool, optional
        If True, returns a vaex DataFrame with the descriptors. Will also behave like when returnData is set to True, by default False.
    
    Returns
    -------
    Dict[str, str | float | int] | Tuple[str | float | int] | vdf.DataFrameLocal | None
        The descriptors.

    Raises
    ------
    KeyError
    '''

    # Try to read the file
    try:
        # Open the json file in read mode
        with open(path, 'r') as f:
            # Load the data
            data = json.load(f)

        # Missing keys list
        missing = []
        # Expected keys to have in the json file
        #region keys
        keys = ["Name", "AUTOCORR2D_1", "AUTOCORR2D_2", "AUTOCORR2D_3", "AUTOCORR2D_4", "AUTOCORR2D_5", "AUTOCORR2D_6", "AUTOCORR2D_7", "AUTOCORR2D_8", "AUTOCORR2D_9", "AUTOCORR2D_10", "AUTOCORR2D_11", "AUTOCORR2D_12", "AUTOCORR2D_13", "AUTOCORR2D_14", "AUTOCORR2D_15", "AUTOCORR2D_16", "AUTOCORR2D_17", "AUTOCORR2D_18", "AUTOCORR2D_19", "AUTOCORR2D_20", "AUTOCORR2D_21", "AUTOCORR2D_22", "AUTOCORR2D_23", "AUTOCORR2D_24", "AUTOCORR2D_25", "AUTOCORR2D_26", "AUTOCORR2D_27", "AUTOCORR2D_28", "AUTOCORR2D_29", "AUTOCORR2D_30", "AUTOCORR2D_31", "AUTOCORR2D_32", "AUTOCORR2D_33", "AUTOCORR2D_34", "AUTOCORR2D_35", "AUTOCORR2D_36", "AUTOCORR2D_37", "AUTOCORR2D_38", "AUTOCORR2D_39", "AUTOCORR2D_40", "AUTOCORR2D_41", "AUTOCORR2D_42", "AUTOCORR2D_43", "AUTOCORR2D_44", "AUTOCORR2D_45", "AUTOCORR2D_46", "AUTOCORR2D_47", "AUTOCORR2D_48", "AUTOCORR2D_49", "AUTOCORR2D_50", "AUTOCORR2D_51", "AUTOCORR2D_52", "AUTOCORR2D_53", "AUTOCORR2D_54", "AUTOCORR2D_55", "AUTOCORR2D_56", "AUTOCORR2D_57", "AUTOCORR2D_58", "AUTOCORR2D_59", "AUTOCORR2D_60", "AUTOCORR2D_61", "AUTOCORR2D_62", "AUTOCORR2D_63", "AUTOCORR2D_64", "AUTOCORR2D_65", "AUTOCORR2D_66", "AUTOCORR2D_67", "AUTOCORR2D_68", "AUTOCORR2D_69", "AUTOCORR2D_70", "AUTOCORR2D_71", "AUTOCORR2D_72", "AUTOCORR2D_73", "AUTOCORR2D_74", "AUTOCORR2D_75", "AUTOCORR2D_76", "AUTOCORR2D_77", "AUTOCORR2D_78", "AUTOCORR2D_79", "AUTOCORR2D_80", "AUTOCORR2D_81", "AUTOCORR2D_82", "AUTOCORR2D_83", "AUTOCORR2D_84", "AUTOCORR2D_85", "AUTOCORR2D_86", "AUTOCORR2D_87", "AUTOCORR2D_88", "AUTOCORR2D_89", "AUTOCORR2D_90", "AUTOCORR2D_91", "AUTOCORR2D_92", "AUTOCORR2D_93", "AUTOCORR2D_94", "AUTOCORR2D_95", "AUTOCORR2D_96", "AUTOCORR2D_97", "AUTOCORR2D_98", "AUTOCORR2D_99", "AUTOCORR2D_100", "AUTOCORR2D_101", "AUTOCORR2D_102", "AUTOCORR2D_103", "AUTOCORR2D_104", "AUTOCORR2D_105", "AUTOCORR2D_106", "AUTOCORR2D_107", "AUTOCORR2D_108", "AUTOCORR2D_109", "AUTOCORR2D_110", "AUTOCORR2D_111", "AUTOCORR2D_112", "AUTOCORR2D_113", "AUTOCORR2D_114", "AUTOCORR2D_115", "AUTOCORR2D_116", "AUTOCORR2D_117", "AUTOCORR2D_118", "AUTOCORR2D_119", "AUTOCORR2D_120", "AUTOCORR2D_121", "AUTOCORR2D_122", "AUTOCORR2D_123", "AUTOCORR2D_124", "AUTOCORR2D_125", "AUTOCORR2D_126", "AUTOCORR2D_127", "AUTOCORR2D_128", "AUTOCORR2D_129", "AUTOCORR2D_130", "AUTOCORR2D_131", "AUTOCORR2D_132", "AUTOCORR2D_133", "AUTOCORR2D_134", "AUTOCORR2D_135", "AUTOCORR2D_136", "AUTOCORR2D_137", "AUTOCORR2D_138", "AUTOCORR2D_139", "AUTOCORR2D_140", "AUTOCORR2D_141", "AUTOCORR2D_142", "AUTOCORR2D_143", "AUTOCORR2D_144", "AUTOCORR2D_145", "AUTOCORR2D_146", "AUTOCORR2D_147", "AUTOCORR2D_148", "AUTOCORR2D_149", "AUTOCORR2D_150", "AUTOCORR2D_151", "AUTOCORR2D_152", "AUTOCORR2D_153", "AUTOCORR2D_154", "AUTOCORR2D_155", "AUTOCORR2D_156", "AUTOCORR2D_157", "AUTOCORR2D_158", "AUTOCORR2D_159", "AUTOCORR2D_160", "AUTOCORR2D_161", "AUTOCORR2D_162", "AUTOCORR2D_163", "AUTOCORR2D_164", "AUTOCORR2D_165", "AUTOCORR2D_166", "AUTOCORR2D_167", "AUTOCORR2D_168", "AUTOCORR2D_169", "AUTOCORR2D_170", "AUTOCORR2D_171", "AUTOCORR2D_172", "AUTOCORR2D_173", "AUTOCORR2D_174", "AUTOCORR2D_175", "AUTOCORR2D_176", "AUTOCORR2D_177", "AUTOCORR2D_178", "AUTOCORR2D_179", "AUTOCORR2D_180", "AUTOCORR2D_181", "AUTOCORR2D_182", "AUTOCORR2D_183", "AUTOCORR2D_184", "AUTOCORR2D_185", "AUTOCORR2D_186", "AUTOCORR2D_187", "AUTOCORR2D_188", "AUTOCORR2D_189", "AUTOCORR2D_190", "AUTOCORR2D_191", "AUTOCORR2D_192", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11", "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "FractionCSP3", "HallKierAlpha", "HeavyAtomMolWt", "HeavyAtomCount", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "qed", "RingCount", "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "TPSA", "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10", "AUTOCORR3D_1", "AUTOCORR3D_2", "AUTOCORR3D_3", "AUTOCORR3D_4", "AUTOCORR3D_5", "AUTOCORR3D_6", "AUTOCORR3D_7", "AUTOCORR3D_8", "AUTOCORR3D_9", "AUTOCORR3D_10", "AUTOCORR3D_11", "AUTOCORR3D_12", "AUTOCORR3D_13", "AUTOCORR3D_14", "AUTOCORR3D_15", "AUTOCORR3D_16", "AUTOCORR3D_17", "AUTOCORR3D_18", "AUTOCORR3D_19", "AUTOCORR3D_20", "AUTOCORR3D_21", "AUTOCORR3D_22", "AUTOCORR3D_23", "AUTOCORR3D_24", "AUTOCORR3D_25", "AUTOCORR3D_26", "AUTOCORR3D_27", "AUTOCORR3D_28", "AUTOCORR3D_29", "AUTOCORR3D_30", "AUTOCORR3D_31", "AUTOCORR3D_32", "AUTOCORR3D_33", "AUTOCORR3D_34", "AUTOCORR3D_35", "AUTOCORR3D_36", "AUTOCORR3D_37", "AUTOCORR3D_38", "AUTOCORR3D_39", "AUTOCORR3D_40", "AUTOCORR3D_41", "AUTOCORR3D_42", "AUTOCORR3D_43", "AUTOCORR3D_44", "AUTOCORR3D_45", "AUTOCORR3D_46", "AUTOCORR3D_47", "AUTOCORR3D_48", "AUTOCORR3D_49", "AUTOCORR3D_50", "AUTOCORR3D_51", "AUTOCORR3D_52", "AUTOCORR3D_53", "AUTOCORR3D_54", "AUTOCORR3D_55", "AUTOCORR3D_56", "AUTOCORR3D_57", "AUTOCORR3D_58", "AUTOCORR3D_59", "AUTOCORR3D_60", "AUTOCORR3D_61", "AUTOCORR3D_62", "AUTOCORR3D_63", "AUTOCORR3D_64", "AUTOCORR3D_65", "AUTOCORR3D_66", "AUTOCORR3D_67", "AUTOCORR3D_68", "AUTOCORR3D_69", "AUTOCORR3D_70", "AUTOCORR3D_71", "AUTOCORR3D_72", "AUTOCORR3D_73", "AUTOCORR3D_74", "AUTOCORR3D_75", "AUTOCORR3D_76", "AUTOCORR3D_77", "AUTOCORR3D_78", "AUTOCORR3D_79", "AUTOCORR3D_80", "Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"]
        #endregion

        # Validate the data
        for key in keys:
            # If key is lacking in data read from json (means malformed json!)
            if not key in data:
                # Add the missing key to the missing list
                missing.append(key)
        
        # If missing list is not empty
        if missing:
            # Raise a Key error passing the file and the missing keys joined with ', '
            raise KeyError((path, ", ".join(missing)))

        # If the returnVaex is set
        if returnVaex:
            # Check if data has a 'Name' key
            if "Name" in data:
                # Temporary fix
                if data["Name"] == "molecule":
                    # Get the last part of the folder path
                    data["Ligand"] = os.path.dirname(data["Path"]).split("/")[-1]
                    _ = data.pop("Name")
                else:
                    # If it has, change the key name to 'Ligand'
                    data["Ligand"] = data.pop("Name")

            # Check if data has a 'Path' key
            if "Path" in data:
                # Remove the entry
                _ = data.pop("Path")

            # For each key, element in data
            for key, element in data.items():
                # Make the element for key be a list with only the element
                data[key] = [element]
            
            # Convert the data to a vaex DataFrame
            return vaex.from_dict(data)

        # If the returnData flag is on
        if returnData:
            # Return the entire dict
            return data

        # Since we have all keys, read them and return their values
        #region Return the data
        return data["Name"], data["AUTOCORR2D_1"], data["AUTOCORR2D_2"], data["AUTOCORR2D_3"], data["AUTOCORR2D_4"], data["AUTOCORR2D_5"], data["AUTOCORR2D_6"], data["AUTOCORR2D_7"], data["AUTOCORR2D_8"], data["AUTOCORR2D_9"], data["AUTOCORR2D_10"], data["AUTOCORR2D_11"], data["AUTOCORR2D_12"], data["AUTOCORR2D_13"], data["AUTOCORR2D_14"], data["AUTOCORR2D_15"], data["AUTOCORR2D_16"], data["AUTOCORR2D_17"], data["AUTOCORR2D_18"], data["AUTOCORR2D_19"], data["AUTOCORR2D_20"], data["AUTOCORR2D_21"], data["AUTOCORR2D_22"], data["AUTOCORR2D_23"], data["AUTOCORR2D_24"], data["AUTOCORR2D_25"], data["AUTOCORR2D_26"], data["AUTOCORR2D_27"], data["AUTOCORR2D_28"], data["AUTOCORR2D_29"], data["AUTOCORR2D_30"], data["AUTOCORR2D_31"], data["AUTOCORR2D_32"], data["AUTOCORR2D_33"], data["AUTOCORR2D_34"], data["AUTOCORR2D_35"], data["AUTOCORR2D_36"], data["AUTOCORR2D_37"], data["AUTOCORR2D_38"], data["AUTOCORR2D_39"], data["AUTOCORR2D_40"], data["AUTOCORR2D_41"], data["AUTOCORR2D_42"], data["AUTOCORR2D_43"], data["AUTOCORR2D_44"], data["AUTOCORR2D_45"], data["AUTOCORR2D_46"], data["AUTOCORR2D_47"], data["AUTOCORR2D_48"], data["AUTOCORR2D_49"], data["AUTOCORR2D_50"], data["AUTOCORR2D_51"], data["AUTOCORR2D_52"], data["AUTOCORR2D_53"], data["AUTOCORR2D_54"], data["AUTOCORR2D_55"], data["AUTOCORR2D_56"], data["AUTOCORR2D_57"], data["AUTOCORR2D_58"], data["AUTOCORR2D_59"], data["AUTOCORR2D_60"], data["AUTOCORR2D_61"], data["AUTOCORR2D_62"], data["AUTOCORR2D_63"], data["AUTOCORR2D_64"], data["AUTOCORR2D_65"], data["AUTOCORR2D_66"], data["AUTOCORR2D_67"], data["AUTOCORR2D_68"], data["AUTOCORR2D_69"], data["AUTOCORR2D_70"], data["AUTOCORR2D_71"], data["AUTOCORR2D_72"], data["AUTOCORR2D_73"], data["AUTOCORR2D_74"], data["AUTOCORR2D_75"], data["AUTOCORR2D_76"], data["AUTOCORR2D_77"], data["AUTOCORR2D_78"], data["AUTOCORR2D_79"], data["AUTOCORR2D_80"], data["AUTOCORR2D_81"], data["AUTOCORR2D_82"], data["AUTOCORR2D_83"], data["AUTOCORR2D_84"], data["AUTOCORR2D_85"], data["AUTOCORR2D_86"], data["AUTOCORR2D_87"], data["AUTOCORR2D_88"], data["AUTOCORR2D_89"], data["AUTOCORR2D_90"], data["AUTOCORR2D_91"], data["AUTOCORR2D_92"], data["AUTOCORR2D_93"], data["AUTOCORR2D_94"], data["AUTOCORR2D_95"], data["AUTOCORR2D_96"], data["AUTOCORR2D_97"], data["AUTOCORR2D_98"], data["AUTOCORR2D_99"], data["AUTOCORR2D_100"], data["AUTOCORR2D_101"], data["AUTOCORR2D_102"], data["AUTOCORR2D_103"], data["AUTOCORR2D_104"], data["AUTOCORR2D_105"], data["AUTOCORR2D_106"], data["AUTOCORR2D_107"], data["AUTOCORR2D_108"], data["AUTOCORR2D_109"], data["AUTOCORR2D_110"], data["AUTOCORR2D_111"], data["AUTOCORR2D_112"], data["AUTOCORR2D_113"], data["AUTOCORR2D_114"], data["AUTOCORR2D_115"], data["AUTOCORR2D_116"], data["AUTOCORR2D_117"], data["AUTOCORR2D_118"], data["AUTOCORR2D_119"], data["AUTOCORR2D_120"], data["AUTOCORR2D_121"], data["AUTOCORR2D_122"], data["AUTOCORR2D_123"], data["AUTOCORR2D_124"], data["AUTOCORR2D_125"], data["AUTOCORR2D_126"], data["AUTOCORR2D_127"], data["AUTOCORR2D_128"], data["AUTOCORR2D_129"], data["AUTOCORR2D_130"], data["AUTOCORR2D_131"], data["AUTOCORR2D_132"], data["AUTOCORR2D_133"], data["AUTOCORR2D_134"], data["AUTOCORR2D_135"], data["AUTOCORR2D_136"], data["AUTOCORR2D_137"], data["AUTOCORR2D_138"], data["AUTOCORR2D_139"], data["AUTOCORR2D_140"], data["AUTOCORR2D_141"], data["AUTOCORR2D_142"], data["AUTOCORR2D_143"], data["AUTOCORR2D_144"], data["AUTOCORR2D_145"], data["AUTOCORR2D_146"], data["AUTOCORR2D_147"], data["AUTOCORR2D_148"], data["AUTOCORR2D_149"], data["AUTOCORR2D_150"], data["AUTOCORR2D_151"], data["AUTOCORR2D_152"], data["AUTOCORR2D_153"], data["AUTOCORR2D_154"], data["AUTOCORR2D_155"], data["AUTOCORR2D_156"], data["AUTOCORR2D_157"], data["AUTOCORR2D_158"], data["AUTOCORR2D_159"], data["AUTOCORR2D_160"], data["AUTOCORR2D_161"], data["AUTOCORR2D_162"], data["AUTOCORR2D_163"], data["AUTOCORR2D_164"], data["AUTOCORR2D_165"], data["AUTOCORR2D_166"], data["AUTOCORR2D_167"], data["AUTOCORR2D_168"], data["AUTOCORR2D_169"], data["AUTOCORR2D_170"], data["AUTOCORR2D_171"], data["AUTOCORR2D_172"], data["AUTOCORR2D_173"], data["AUTOCORR2D_174"], data["AUTOCORR2D_175"], data["AUTOCORR2D_176"], data["AUTOCORR2D_177"], data["AUTOCORR2D_178"], data["AUTOCORR2D_179"], data["AUTOCORR2D_180"], data["AUTOCORR2D_181"], data["AUTOCORR2D_182"], data["AUTOCORR2D_183"], data["AUTOCORR2D_184"], data["AUTOCORR2D_185"], data["AUTOCORR2D_186"], data["AUTOCORR2D_187"], data["AUTOCORR2D_188"], data["AUTOCORR2D_189"], data["AUTOCORR2D_190"], data["AUTOCORR2D_191"], data["AUTOCORR2D_192"], data["BCUT2D_CHGHI"], data["BCUT2D_CHGLO"], data["BCUT2D_LOGPHI"], data["BCUT2D_LOGPLOW"], data["BCUT2D_MRHI"], data["BCUT2D_MRLOW"], data["BCUT2D_MWHI"], data["BCUT2D_MWLOW"], data["BalabanJ"], data["BertzCT"], data["Chi0"], data["Chi0n"], data["Chi0v"], data["Chi1"], data["Chi1n"], data["Chi1v"], data["Chi2n"], data["Chi2v"], data["Chi3n"], data["Chi3v"], data["Chi4n"], data["Chi4v"], data["EState_VSA1"], data["EState_VSA2"], data["EState_VSA3"], data["EState_VSA4"], data["EState_VSA5"], data["EState_VSA6"], data["EState_VSA7"], data["EState_VSA8"], data["EState_VSA9"], data["EState_VSA10"], data["EState_VSA11"], data["MaxAbsEStateIndex"], data["MaxEStateIndex"], data["MinAbsEStateIndex"], data["MinEStateIndex"], data["ExactMolWt"], data["FpDensityMorgan1"], data["FpDensityMorgan2"], data["FpDensityMorgan3"], data["fr_Al_COO"], data["fr_Al_OH"], data["fr_Al_OH_noTert"], data["fr_ArN"], data["fr_Ar_COO"], data["fr_Ar_N"], data["fr_Ar_NH"], data["fr_Ar_OH"], data["fr_COO"], data["fr_COO2"], data["fr_C_O"], data["fr_C_O_noCOO"], data["fr_C_S"], data["fr_HOCCN"], data["fr_Imine"], data["fr_NH0"], data["fr_NH1"], data["fr_NH2"], data["fr_N_O"], data["fr_Ndealkylation1"], data["fr_Ndealkylation2"], data["fr_Nhpyrrole"], data["fr_SH"], data["fr_aldehyde"], data["fr_alkyl_carbamate"], data["fr_alkyl_halide"], data["fr_allylic_oxid"], data["fr_amide"], data["fr_amidine"], data["fr_aniline"], data["fr_aryl_methyl"], data["fr_azide"], data["fr_azo"], data["fr_barbitur"], data["fr_benzene"], data["fr_benzodiazepine"], data["fr_bicyclic"], data["fr_diazo"], data["fr_dihydropyridine"], data["fr_epoxide"], data["fr_ester"], data["fr_ether"], data["fr_furan"], data["fr_guanido"], data["fr_halogen"], data["fr_hdrzine"], data["fr_hdrzone"], data["fr_imidazole"], data["fr_imide"], data["fr_isocyan"], data["fr_isothiocyan"], data["fr_ketone"], data["fr_ketone_Topliss"], data["fr_lactam"], data["fr_lactone"], data["fr_methoxy"], data["fr_morpholine"], data["fr_nitrile"], data["fr_nitro"], data["fr_nitro_arom"], data["fr_nitro_arom_nonortho"], data["fr_nitroso"], data["fr_oxazole"], data["fr_oxime"], data["fr_para_hydroxylation"], data["fr_phenol"], data["fr_phenol_noOrthoHbond"], data["fr_phos_acid"], data["fr_phos_ester"], data["fr_piperdine"], data["fr_piperzine"], data["fr_priamide"], data["fr_prisulfonamd"], data["fr_pyridine"], data["fr_quatN"], data["fr_sulfide"], data["fr_sulfonamd"], data["fr_sulfone"], data["fr_term_acetylene"], data["fr_tetrazole"], data["fr_thiazole"], data["fr_thiocyan"], data["fr_thiophene"], data["fr_unbrch_alkane"], data["fr_urea"], data["FractionCSP3"], data["HallKierAlpha"], data["HeavyAtomMolWt"], data["HeavyAtomCount"], data["Ipc"], data["Kappa1"], data["Kappa2"], data["Kappa3"], data["LabuteASA"], data["MaxAbsPartialCharge"], data["MaxPartialCharge"], data["MinAbsPartialCharge"], data["MinPartialCharge"], data["MolLogP"], data["MolMR"], data["MolWt"], data["NHOHCount"], data["NOCount"], data["NumAliphaticCarbocycles"], data["NumAliphaticHeterocycles"], data["NumAliphaticRings"], data["NumAromaticCarbocycles"], data["NumAromaticHeterocycles"], data["NumAromaticRings"], data["NumHAcceptors"], data["NumHDonors"], data["NumHeteroatoms"], data["NumRadicalElectrons"], data["NumRotatableBonds"], data["NumSaturatedCarbocycles"], data["NumSaturatedHeterocycles"], data["NumSaturatedRings"], data["NumValenceElectrons"], data["PEOE_VSA1"], data["PEOE_VSA2"], data["PEOE_VSA3"], data["PEOE_VSA4"], data["PEOE_VSA5"], data["PEOE_VSA6"], data["PEOE_VSA7"], data["PEOE_VSA8"], data["PEOE_VSA9"], data["PEOE_VSA10"], data["PEOE_VSA11"], data["PEOE_VSA12"], data["PEOE_VSA13"], data["PEOE_VSA14"], data["qed"], data["RingCount"], data["SMR_VSA1"], data["SMR_VSA2"], data["SMR_VSA3"], data["SMR_VSA4"], data["SMR_VSA5"], data["SMR_VSA6"], data["SMR_VSA7"], data["SMR_VSA8"], data["SMR_VSA9"], data["SMR_VSA10"], data["SlogP_VSA1"], data["SlogP_VSA2"], data["SlogP_VSA3"], data["SlogP_VSA4"], data["SlogP_VSA5"], data["SlogP_VSA6"], data["SlogP_VSA7"], data["SlogP_VSA8"], data["SlogP_VSA9"], data["SlogP_VSA10"], data["SlogP_VSA11"], data["SlogP_VSA12"], data["TPSA"], data["VSA_EState1"], data["VSA_EState2"], data["VSA_EState3"], data["VSA_EState4"], data["VSA_EState5"], data["VSA_EState6"], data["VSA_EState7"], data["VSA_EState8"], data["VSA_EState9"], data["VSA_EState10"], data["AUTOCORR3D_1"], data["AUTOCORR3D_2"], data["AUTOCORR3D_3"], data["AUTOCORR3D_4"], data["AUTOCORR3D_5"], data["AUTOCORR3D_6"], data["AUTOCORR3D_7"], data["AUTOCORR3D_8"], data["AUTOCORR3D_9"], data["AUTOCORR3D_10"], data["AUTOCORR3D_11"], data["AUTOCORR3D_12"], data["AUTOCORR3D_13"], data["AUTOCORR3D_14"], data["AUTOCORR3D_15"], data["AUTOCORR3D_16"], data["AUTOCORR3D_17"], data["AUTOCORR3D_18"], data["AUTOCORR3D_19"], data["AUTOCORR3D_20"], data["AUTOCORR3D_21"], data["AUTOCORR3D_22"], data["AUTOCORR3D_23"], data["AUTOCORR3D_24"], data["AUTOCORR3D_25"], data["AUTOCORR3D_26"], data["AUTOCORR3D_27"], data["AUTOCORR3D_28"], data["AUTOCORR3D_29"], data["AUTOCORR3D_30"], data["AUTOCORR3D_31"], data["AUTOCORR3D_32"], data["AUTOCORR3D_33"], data["AUTOCORR3D_34"], data["AUTOCORR3D_35"], data["AUTOCORR3D_36"], data["AUTOCORR3D_37"], data["AUTOCORR3D_38"], data["AUTOCORR3D_39"], data["AUTOCORR3D_40"], data["AUTOCORR3D_41"], data["AUTOCORR3D_42"], data["AUTOCORR3D_43"], data["AUTOCORR3D_44"], data["AUTOCORR3D_45"], data["AUTOCORR3D_46"], data["AUTOCORR3D_47"], data["AUTOCORR3D_48"], data["AUTOCORR3D_49"], data["AUTOCORR3D_50"], data["AUTOCORR3D_51"], data["AUTOCORR3D_52"], data["AUTOCORR3D_53"], data["AUTOCORR3D_54"], data["AUTOCORR3D_55"], data["AUTOCORR3D_56"], data["AUTOCORR3D_57"], data["AUTOCORR3D_58"], data["AUTOCORR3D_59"], data["AUTOCORR3D_60"], data["AUTOCORR3D_61"], data["AUTOCORR3D_62"], data["AUTOCORR3D_63"], data["AUTOCORR3D_64"], data["AUTOCORR3D_65"], data["AUTOCORR3D_66"], data["AUTOCORR3D_67"], data["AUTOCORR3D_68"], data["AUTOCORR3D_69"], data["AUTOCORR3D_70"], data["AUTOCORR3D_71"], data["AUTOCORR3D_72"], data["AUTOCORR3D_73"], data["AUTOCORR3D_74"], data["AUTOCORR3D_75"], data["AUTOCORR3D_76"], data["AUTOCORR3D_77"], data["AUTOCORR3D_78"], data["AUTOCORR3D_79"], data["AUTOCORR3D_80"], data["Asphericity"], data["Eccentricity"], data["InertialShapeFactor"], data["NPR1"], data["NPR2"], data["PMI1"], data["PMI2"], data["PMI3"], data["RadiusOfGyration"], data["SpherocityIndex"] # type: ignore
        #endregion
        
    # Key error (when there is a missing key)
    except KeyError as missed:
        ocprint.print_error(f"The following keys were not found in the json file '{missed[0]}': {missed[1]}.") # type: ignore
    # General error (call it as problem to read file)
    except Exception as e:
        ocprint.print_error(f"Could not read the file '{path}'. Error: {e}")

    return None

def get_smiles(molecule: rdkit.Chem.rdchem.Mol) -> Union[str, int]: # type: ignore
    '''Return the smiles of the molecule

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
        return errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")

    return errors.not_set(f"The variable is not set.")

def get_centroid(molecule: Union[str, rdkit.Chem.rdchem.Mol], sanitize = True) -> rdkit.Geometry.rdGeometry.Point3D: # type: ignore
    '''Get the centroid of the molecule.

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
    '''

    # Check if the molecule is a string (means that it is a path)
    if type(molecule) == str:
        # Load it
        _, molecule = loadMol(molecule, sanitize = sanitize)

    # Get the molecule conformer
    conf = molecule.GetConformer() # type: ignore

    # Compute the centroid of the molecule and return it
    return ComputeCentroid(conf)


# Descriptors functions #

#region AUTOCORR descriptors
def findAUTOCORR2D_1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_1 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")
    
    return None

def findAUTOCORR2D_2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_2 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_4 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_5 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_5 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_6 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_7 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_7 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_8 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_8 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_9 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_9 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_10 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_10 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_11(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_11 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_11 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_11(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_12(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_12 descriptor.
        
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_12 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_12(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_13(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_13 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_13 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_13(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")


    return None

def findAUTOCORR2D_14(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_14 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_14 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_14(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_15(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_15 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_15 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_15(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_16(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_16 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_16 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_16(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_17(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_17 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_17 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_17(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_18(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_18 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_18 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_18(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")
    
    return None

def findAUTOCORR2D_19(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_19 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_19 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_19(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_20(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_20 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_20 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_20(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_21(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_21 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_21 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_21(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_22(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_22 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_22 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_22(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_23(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_23 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_23 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_23(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_24(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_24 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_24 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_24(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_25(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_25 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_25 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_25(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_26(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_26 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_26 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_26(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_27(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_27 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_27 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_27(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_28(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_28 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_28 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_28(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_29(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_29 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_29 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_29(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_30(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_30 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_30 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_30(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_31(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_31 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_31 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_31(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_32(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_32 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_32 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_32(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_33(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_33 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_33 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_33(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_34(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_34 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_34 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_34(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_35(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_35 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_35 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_35(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_36(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_36 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_36 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_36(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_37(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_37 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_37 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_37(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_38(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_38 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_38 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_38(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_39(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_39 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_39 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_39(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_40(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_40 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_40 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_40(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_41(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_41 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_41 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_41(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_42(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_42 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_42 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_42(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_43(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_43 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_43 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_43(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_44(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_44 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_44 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_44(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_45(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_45 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_45 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_45(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_46(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_46 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_46 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_46(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_47(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_47 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_47 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_47(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_48(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_48 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_48 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_48(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_49(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_49 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_49 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_49(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_50(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_50 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_50 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_50(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_51(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_51 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_51 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_51(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_52(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_52 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_52 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_52(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_53(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_53 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_53 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_53(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_54(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_54 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_54 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_54(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_55(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_55 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_55 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_55(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_56(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_56 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_56 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_56(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_57(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_57 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_57 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_57(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_58(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_58 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_58 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_58(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_59(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_59 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_59 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_59(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_60(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_60 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_60 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_60(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_61(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_61 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_61 value or None if parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_61(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_62(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_62 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_62 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_62(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_63(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_63 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_63 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_63(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_64(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_64 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_64 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_64(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_65(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_65 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_65 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_65(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_66(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_66 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_66 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_66(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_67(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_67 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_67 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_67(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_68(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_68 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_68 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_68(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_69(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_69 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_69 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_69(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_70(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_70 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_70 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_70(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_71(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_71 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_71 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_71(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_72(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_72 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_72 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_72(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_73(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_73 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_73 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_73(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_74(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_74 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_74 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_74(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_75(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_75 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_75 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_75(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_76(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_76 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_76 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_76(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_77(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_77 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_77 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_77(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_78(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_78 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_78 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_78(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_79(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_79 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_79 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_79(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_80(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_80 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_80 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_80(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_81(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_81 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_81 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_81(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_82(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_82 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_82 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_82(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_83(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_83 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_83 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_83(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_84(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_84 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_84 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_84(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_85(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_85 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_85 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_85(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_86(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_86 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_86 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_86(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_87(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_87 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_87 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_87(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_88(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_88 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_88 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_88(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_89(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_89 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_89 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_89(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_90(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_90 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_90 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_90(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_91(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_91 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_91 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_91(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_92(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_92 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_92 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_92(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_93(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_93 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_93 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_93(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_94(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_94 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_94 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_94(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_95(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_95 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_95 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_95(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_96(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_96 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_96 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_96(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_97(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_97 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_97 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_97(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_98(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_98 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_98 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_98(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_99(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_99 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_99 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_99(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_100(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_100 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_100 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_100(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_101(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_101 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_101 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_101(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_102(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_102 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_102 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_102(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_103(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_103 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_103 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_103(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_104(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_104 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_104 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_104(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_105(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_105 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_105 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_105(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_106(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_106 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_106 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_106(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_107(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_107 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_107 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_107(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_108(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_108 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_108 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_108(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_109(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_109 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_109 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_109(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_110(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_110 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_110 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_110(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_111(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_111 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_111 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_111(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_112(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_112 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_112 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_112(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_113(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_113 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_113 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_113(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_114(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_114 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_114 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_114(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_115(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_115 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_115 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_115(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_116(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_116 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_116 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_116(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_117(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_117 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_117 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_117(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_118(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_118 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_118 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_118(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_119(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_119 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_119 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_119(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_120(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_120 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_120 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_120(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_121(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_121 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_121 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_121(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_122(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_122 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_122 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_122(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_123(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_123 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_123 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_123(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_124(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_124 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_124 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_124(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_125(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_125 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_125 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_125(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_126(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_126 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_126 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_126(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_127(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_127 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_127 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_127(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_128(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_128 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_128 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_128(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_129(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_129 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_129 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_129(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_130(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_130 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_130 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_130(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_131(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_131 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_131 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_131(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_132(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_132 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_132 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_132(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_133(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_133 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_133 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_133(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_134(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_134 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_134 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_134(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_135(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_135 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_135 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_135(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_136(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_136 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_136 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_136(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_137(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_137 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_137 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_137(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_138(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_138 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_138 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_138(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_139(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_139 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_139 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_139(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_140(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_140 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_140 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_140(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_141(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_141 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_141 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_141(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_142(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_142 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_142 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_142(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_143(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_143 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_143 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_143(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_144(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_144 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_144 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_144(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_145(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_145 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_145 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_145(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_146(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_146 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_146 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_146(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_147(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_147 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_147 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_147(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_148(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_148 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_148 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_148(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_149(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_149 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_149 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_149(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_150(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_150 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_150 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_150(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_151(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_151 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_151 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_151(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_152(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_152 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_152 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_152(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_153(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_153 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_153 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_153(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_154(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_154 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_154 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_154(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_155(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_155 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_155 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_155(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_156(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_156 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_156 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_156(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_157(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_157 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_157 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_157(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_158(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_158 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_158 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_158(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_159(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_159 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_159 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_159(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_160(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_160 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_160 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_160(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_161(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_161 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_161 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_161(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_162(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_162 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_162 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_162(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_163(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_163 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_163 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_163(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_164(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_164 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_164 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_164(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_165(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_165 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_165 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_165(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_166(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_166 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_166 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_166(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_167(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_167 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_167 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_167(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_168(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_168 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_168 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_168(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_169(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_169 descriptor.
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_169 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_169(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_170(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_170 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_170 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_170(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_171(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_171 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_171 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_171(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_172(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_172 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_172 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_172(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_173(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_173 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_173 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_173(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_174(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_174 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_174 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_174(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_175(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_175 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_175 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_175(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_176(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_176 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_176 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_176(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_177(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_177 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_177 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_177(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_178(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_178 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_178 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_178(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_179(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_179 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_179 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_179(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_180(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_180 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_180 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_180(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_181(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_181 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_181 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_181(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_182(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_182 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_182 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_182(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_183(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_183 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_183 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_183(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_184(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_184 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_184 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_184(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_185(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_185 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_185 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_185(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_186(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_186 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_186 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_186(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_187(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_187 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_187 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_187(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_188(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_188 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_188 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_188(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_189(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_189 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_189 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_189(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_190(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_190 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_190 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_190(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_191(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_191 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_191 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_191(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findAUTOCORR2D_192(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the autocorrelation2D_192 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The autocorrelation2D_192 value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.AUTOCORR2D_192(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

#region BCUT2D descriptors
def findBCUT2D_CHGHI(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_CHGHI descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_CHGHI value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_CHGHI(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_CHGLO(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_CHGLO descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_CHGLO value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_CHGLO(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_LOGPHI(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_LOGPHI descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_LOGPHI value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_LOGPHI(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_LOGPLOW(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_LOGPLOW descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_LOGPLOW value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_LOGPLOW(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_MRHI(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''
    Compute the BCUT2D_MRHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_MRHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_MRHI(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_MRLOW(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_MRLOW descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_MRLOW value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_MRLOW(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_MWHI(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_MWHI descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_MWHI value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_MWHI(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBCUT2D_MWLOW(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BCUT2D_MWLOW descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BCUT2D_MWLOW value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BCUT2D_MWLOW(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

def findBalabanJ(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BalabanJ descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BalabanJ value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BalabanJ(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findBertzCT(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the BertzCT descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The BertzCT value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.BertzCT(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region Chi descriptors
def findChi0(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi0 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi0 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi0(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi0n(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi0n descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi0n value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi0n(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi0v(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi0v descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi0v value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi0v(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi1n(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi1n descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi1n value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi1n(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi1v(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi1v descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi1v value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi1v(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi2n(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi2n descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi2n value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi2n(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi2v(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi2v descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi2v value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi2v(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi3n(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi3n descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi3n value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi3n(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi3v(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi3v descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi3v value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi3v(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi4n(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi4n descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi4n value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi4n(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findChi4v(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Chi4v descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Chi4v value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Chi4v(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

#region EState descriptors
def findEState_VSA1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA4 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA5 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA5 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA6 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA7 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA7 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA8 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA8 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA9 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA9 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA10 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA10 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEState_VSA11(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the EState_VSA11 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The EState_VSA11 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.EState_VSA11(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMaxAbsEStateIndex(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MaxAbsEStateIndex descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MaxAbsEStateIndex value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MaxAbsEStateIndex(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMaxEStateIndex(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MaxEStateIndex descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MaxEStateIndex value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MaxEStateIndex(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMinAbsEStateIndex(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MinAbsEStateIndex descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MinAbsEStateIndex value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MinAbsEStateIndex(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMinEStateIndex(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MinEStateIndex descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MinEStateIndex value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MinEStateIndex(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

def findExactMolWt(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the exact molecular weight of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The exact molecular weight of the molecule or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.ExactMolWt(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findFpDensityMorgan1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Morgan fingerprint, radius 1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Morgan fingerprint, radius 1 descriptor or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.FpDensityMorgan1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findFpDensityMorgan2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Morgan fingerprint, radius 2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Morgan fingerprint, radius 2 descriptor or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.FpDensityMorgan2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findFpDensityMorgan3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Morgan fingerprint, radius 3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Morgan fingerprint, radius 3 descriptor or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.FpDensityMorgan3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region fr_ descriptors
def findfr_Al_COO(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Al_COO descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Al_COO value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Al_COO(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Al_OH(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Al_OH descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Al_OH value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Al_OH(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Al_OH_noTert(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Al_OH_noTert descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Al_OH_noTert value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Al_OH_noTert(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_ArN(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_ArN descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_ArN value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_ArN(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ar_COO(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ar_COO descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ar_COO value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ar_COO(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ar_N(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ar_N descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ar_N value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ar_N(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ar_NH(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ar_NH descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ar_NH value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ar_NH(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ar_OH(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ar_OH descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ar_OH value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ar_OH(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_COO(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_COO descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_COO value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_COO(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_COO2(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_COO2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_COO2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_COO2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_C_O(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_C_O descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_C_O value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_C_O(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_C_O_noCOO(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_C_O_noCOO descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_C_O_noCOO value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_C_O_noCOO(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_C_S(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_C_S descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_C_S value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_C_S(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_HOCCN(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_HOCCN descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_HOCCN value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_HOCCN(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Imine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Imine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Imine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Imine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_NH0(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_NH0 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_NH0 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_NH0(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_NH1(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_NH1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_NH1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_NH1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_NH2(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_NH2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_NH2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_NH2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_N_O(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_N_O descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_N_O value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_N_O(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ndealkylation1(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ndealkylation1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ndealkylation1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ndealkylation1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Ndealkylation2(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Ndealkylation2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Ndealkylation2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Ndealkylation2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_Nhpyrrole(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_Nhpyrrole descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_Nhpyrrole value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_Nhpyrrole(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_SH(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_SH descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_SH value or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_SH(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_aldehyde(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_aldehyde descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_aldehyde value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_aldehyde(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_alkyl_carbamate(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_alkyl_carbamate descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_alkyl_carbamate value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_alkyl_carbamate(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_alkyl_halide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_alkyl_halide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_alkyl_halide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_alkyl_halide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_allylic_oxid(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_allylic_oxid descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_allylic_oxid value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_allylic_oxid(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_amide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_amide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_amide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_amide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_amidine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_amidine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_amidine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_amidine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_aniline(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_aniline descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_aniline value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_aniline(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_aryl_methyl(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_aryl_methyl descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_aryl_methyl value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_aryl_methyl(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_azide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_azide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_azide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_azide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_azo(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_azo descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_azo value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_azo(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_barbitur(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_barbitur descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_barbitur value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_barbitur(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_benzene(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_benzene descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_benzene value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_benzene(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_benzodiazepine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_benzodiazepine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_benzodiazepine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_benzodiazepine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_bicyclic(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_bicyclic descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_bicyclic value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_bicyclic(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_diazo(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_diazo descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_diazo value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_diazo(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_dihydropyridine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_dihydropyridine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_dihydropyridine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_dihydropyridine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_epoxide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_epoxide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_epoxide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_epoxide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_ester(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_ester descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_ester value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_ester(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_ether(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_ether descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_ether value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_ether(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_furan(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_furan descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_furan value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_furan(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_guanido(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_guanido descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_guanido value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_guanido(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_halogen(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_halogen descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_halogen value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_halogen(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_hdrzine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_hdrzine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_hdrzine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_hdrzine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_hdrzone(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_hdrzone descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_hdrzone value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_hdrzone(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_imidazole(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_imidazole descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_imidazole value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_imidazole(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_imide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_imide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_imide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_imide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_isocyan(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_isocyan descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_isocyan value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_isocyan(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_isothiocyan(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_isothiocyan descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_isothiocyan value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_isothiocyan(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_ketone(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_ketone descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_ketone value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_ketone(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_ketone_Topliss(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_ketone_Topliss descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_ketone_Topliss value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_ketone_Topliss(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_lactam(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_lactam descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_lactam value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_lactam(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_lactone(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_lactone descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_lactone value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_lactone(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_methoxy(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_methoxy descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_methoxy value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_methoxy(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_morpholine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_morpholine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_morpholine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_morpholine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_nitrile(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_nitrile descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_nitrile value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_nitrile(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_nitro(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_nitro descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_nitro value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_nitro(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_nitro_arom(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_nitro_arom descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_nitro_arom value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_nitro_arom(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_nitro_arom_nonortho(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_nitro_arom_nonortho descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_nitro_arom_nonortho value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_nitro_arom_nonortho(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_nitroso(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_nitroso descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_nitroso value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_nitroso(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_oxazole(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_oxazole descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_oxazole value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_oxazole(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_oxime(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_oxime descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_oxime value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_oxime(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_para_hydroxylation(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_para_hydroxylation descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_para_hydroxylation value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_para_hydroxylation(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_phenol(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_phenol descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_phenol value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_phenol(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_phenol_noOrthoHbond(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_phenol_noOrthoHbond descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_phenol_noOrthoHbond value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_phenol_noOrthoHbond(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_phos_acid(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_phos_acid descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_phos_acid value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_phos_acid(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_phos_ester(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_phos_ester descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_phos_ester value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_phos_ester(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_piperdine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_piperdine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_piperdine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_piperdine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_piperzine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_piperzine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_piperzine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_piperzine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_priamide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_priamide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_priamide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_priamide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_prisulfonamd(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_prisulfonamd descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_prisulfonamd value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_prisulfonamd(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_pyridine(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_pyridine descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_pyridine value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_pyridine(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_quatN(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_quatN descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_quatN value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_quatN(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_sulfide(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_sulfide descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_sulfide value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_sulfide(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_sulfonamd(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_sulfonamd descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_sulfonamd value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_sulfonamd(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_sulfone(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_sulfone descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_sulfone value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_sulfone(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_term_acetylene(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_term_acetylene descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_term_acetylene value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_term_acetylene(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_tetrazole(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_tetrazole descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_tetrazole value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_tetrazole(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_thiazole(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_thiazole descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_thiazole value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_thiazole(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_thiocyan(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_thiocyan descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_thiocyan value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_thiocyan(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_thiophene(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_thiophene descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_thiophene value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_thiophene(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_unbrch_alkane(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_unbrch_alkane descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_unbrch_alkane value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_unbrch_alkane(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findfr_urea(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the fr_urea descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The fr_urea value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.fr_urea(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

def findFractionCSP3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the FractionCSP3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The FractionCSP3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.FractionCSP3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findHallKierAlpha(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the HallKierAlpha descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The HallKierAlpha value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.HallKierAlpha(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findHeavyAtomMolWt(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the heavy atom molecular weight of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The heavy atom molecular weight of the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.HeavyAtomMolWt(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findHeavyAtomCount(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the HeavyAtomCount descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The HeavyAtomCount value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.HeavyAtomCount(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findIpc(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Ipc descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Ipc value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Ipc(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region Kappa descriptors
def findKappa1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Kappa1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Kappa1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Kappa1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findKappa2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Kappa2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Kappa2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Kappa2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findKappa3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Kappa3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Kappa3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.Kappa3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#endregion

def findLabuteASA(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the LabuteASA descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The LabuteASA value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.LabuteASA(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMaxAbsPartialCharge(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the maximum absolute partial charge of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The maximum absolute partial charge of the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MaxAbsPartialCharge(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMaxPartialCharge(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the absolute partial charge of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The absolute partial charge of the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MaxPartialCharge(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMinAbsPartialCharge(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the minimum absolute partial charge of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The minimum absolute partial charge of the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MinAbsPartialCharge(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMinPartialCharge(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the minimum partial charge of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The minimum partial charge of the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MinPartialCharge(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMolLogP(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MolLogP descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MolLogP value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MolLogP(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMolMR(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the MolMR descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The MolMR value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MolMR(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findMolWt(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the molecular weight of the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The molecular weight of the molecule or None if parsing the descriptor fails.
    '''

    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.MolWt(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region 'count' descriptors
def findNHOHCount(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NHOHCount descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NHOHCount value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NHOHCount(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNOCount(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NOCount descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NOCount value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NOCount(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAliphaticCarbocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAliphaticCarbocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAliphaticCarbocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAliphaticCarbocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAliphaticHeterocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAliphaticHeterocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAliphaticHeterocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAliphaticHeterocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAliphaticRings(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAliphaticRings descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAliphaticRings value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAliphaticRings(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAromaticCarbocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAromaticCarbocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAromaticCarbocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAromaticCarbocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAromaticHeterocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAromaticHeterocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAromaticHeterocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAromaticHeterocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumAromaticRings(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumAromaticRings descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumAromaticRings value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumAromaticRings(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumHAcceptors(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumHAcceptors descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumHAcceptors value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumHAcceptors(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumHDonors(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumHDonors descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumHDonors value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumHDonors(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumHeteroatoms(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumHeteroatoms descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumHeteroatoms value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumHeteroatoms(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumRadicalElectrons(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the number of radical electrons in the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The number of radical electrons in the molecule or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumRadicalElectrons(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumRotatableBonds(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumRotatableBonds descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumRotatableBonds value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumRotatableBonds(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumSaturatedCarbocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumSaturatedCarbocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumSaturatedCarbocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumSaturatedCarbocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumSaturatedHeterocycles(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumSaturatedHeterocycles descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumSaturatedHeterocycles value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumSaturatedHeterocycles(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumSaturatedRings(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the NumSaturatedRings descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The NumSaturatedRings value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumSaturatedRings(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNumValenceElectrons(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the number of valence electrons in the molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The number of valence electrons in the moleculeor None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.NumValenceElectrons(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findRingCount(molecule: rdkit.Chem.rdchem.Mol) -> Union[int, None]: # type: ignore
    '''Compute the RingCount descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    int | None
        The RingCount value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.RingCount(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion

#region PEOE_VSA descriptors
def findPEOE_VSA1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA4 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA5 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA5 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA6 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA7 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA7 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA8 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA8 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA9 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA9 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA10 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA10 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA11(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA11 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA11 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA11(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA12(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA12 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA12 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA12(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA13(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA13 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA13 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA13(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPEOE_VSA14(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PEOE_VSA14 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PEOE_VSA14 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.PEOE_VSA14(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion

def findqed(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the qed descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The qed value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.qed(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region SMR_VSA descriptors
def findSMR_VSA1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA4 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA5 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA5 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA6 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA7 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA7 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA8 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA8 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA9 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA9 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSMR_VSA10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SMR_VSA10 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SMR_VSA10 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SMR_VSA10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion

#region SlogP_VSA descriptors
def findSlogP_VSA1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA4 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA5 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA5 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA6 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA7 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA7 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA8 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA8 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA9 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA9 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA10 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA10 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA11(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA11 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA11 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA11(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSlogP_VSA12(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SlogP_VSA12 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SlogP_VSA12 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.SlogP_VSA12(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion

def findTPSA(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the TPSA descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The TPSA value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.TPSA(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

#region VSA_EState descriptors
def findVSA_EState1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState4(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState4 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState4 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState4(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState5(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState5 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState5 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState5(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState6(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState6 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState6 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState6(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState7(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState7 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState7 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState7(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState8(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState8 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState8 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState8(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState9(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState9 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState9 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState9(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findVSA_EState10(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the VSA_EState10 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The VSA_EState10 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors.VSA_EState10(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion

#region 3D descriptors
def findAUTOCORR3D(molecule: rdkit.Chem.rdchem.Mol) -> Union[List[float], None]: # type: ignore
    '''Compute the AUTOCORR3D descriptors.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    List[float] | None
        The AUTOCORR3D values or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.rdMolDescriptors.CalcAUTOCORR3D(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None

    _ = errors.not_set(f"The variable is not set.")

    return None

def findAsphericity(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Asphericity descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Asphericity value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.Asphericity(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findEccentricity(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the Eccentricity descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The Eccentricity value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.Eccentricity(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findInertialShapeFactor(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the InertialShapeFactor descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The InertialShapeFactor value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.InertialShapeFactor(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNPR1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the NPR1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The NPR1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.NPR1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findNPR2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the NPR2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The NPR2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.NPR2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPMI1(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PMI1 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PMI1 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.PMI1(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPMI2(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PMI2 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PMI2 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.PMI2(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findPMI3(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the PMI3 descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The PMI3 value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.PMI3(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findRadiusOfGyration(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the RadiusOfGyration descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The RadiusOfGyration value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.RadiusOfGyration(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None

def findSpherocityIndex(molecule: rdkit.Chem.rdchem.Mol) -> Union[float, None]: # type: ignore
    '''Compute the SpherocityIndex descriptor.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to be evaluated.

    Returns
    -------
    float | None
        The SpherocityIndex value or None if parsing the descriptor fails.
    '''
    
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return Descriptors3D.SpherocityIndex(molecule) # type: ignore
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    else:
        _ = errors.not_set(f"The variable is not set.")

    return None
#endregion
