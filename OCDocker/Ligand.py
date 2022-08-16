#!/usr/lib/python3

# Imports
###############################################################################
import os
import json
import rdkit
from glob import glob

from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors

from openbabel import openbabel

from OCDocker.Initialise import *
import OCDocker.Toolbox as octools

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
    def __init__(self, molecule, name, sanitize = True, from_json_descriptors = ""):
        # Set the path and structure (NEVER SHOUD BE NONE)
        self.path, self.molecule = self.__loadMol(molecule, sanitize)
        # Define everything as None
        self.name = None

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
            data = self.__read_descriptors_from_json(from_json_descriptors)
            # If data is None, a problem occurred while reading the json file
            if not data:
                octools.print_error(f"Problems while parsing json file: '{from_json_descriptors}'")
                return None
            #region assign
            self.name, self.AUTOCORR2D_1, self.AUTOCORR2D_2, self.AUTOCORR2D_3, self.AUTOCORR2D_4, self.AUTOCORR2D_5, self.AUTOCORR2D_6, self.AUTOCORR2D_7, self.AUTOCORR2D_8, self.AUTOCORR2D_9, self.AUTOCORR2D_10, self.AUTOCORR2D_11, self.AUTOCORR2D_12, self.AUTOCORR2D_13, self.AUTOCORR2D_14, self.AUTOCORR2D_15, self.AUTOCORR2D_16, self.AUTOCORR2D_17, self.AUTOCORR2D_18, self.AUTOCORR2D_19, self.AUTOCORR2D_20, self.AUTOCORR2D_21, self.AUTOCORR2D_22, self.AUTOCORR2D_23, self.AUTOCORR2D_24, self.AUTOCORR2D_25, self.AUTOCORR2D_26, self.AUTOCORR2D_27, self.AUTOCORR2D_28, self.AUTOCORR2D_29, self.AUTOCORR2D_30, self.AUTOCORR2D_31, self.AUTOCORR2D_32, self.AUTOCORR2D_33, self.AUTOCORR2D_34, self.AUTOCORR2D_35, self.AUTOCORR2D_36, self.AUTOCORR2D_37, self.AUTOCORR2D_38, self.AUTOCORR2D_39, self.AUTOCORR2D_40, self.AUTOCORR2D_41, self.AUTOCORR2D_42, self.AUTOCORR2D_43, self.AUTOCORR2D_44, self.AUTOCORR2D_45, self.AUTOCORR2D_46, self.AUTOCORR2D_47, self.AUTOCORR2D_48, self.AUTOCORR2D_49, self.AUTOCORR2D_50, self.AUTOCORR2D_51, self.AUTOCORR2D_52, self.AUTOCORR2D_53, self.AUTOCORR2D_54, self.AUTOCORR2D_55, self.AUTOCORR2D_56, self.AUTOCORR2D_57, self.AUTOCORR2D_58, self.AUTOCORR2D_59, self.AUTOCORR2D_60, self.AUTOCORR2D_61, self.AUTOCORR2D_62, self.AUTOCORR2D_63, self.AUTOCORR2D_64, self.AUTOCORR2D_65, self.AUTOCORR2D_66, self.AUTOCORR2D_67, self.AUTOCORR2D_68, self.AUTOCORR2D_69, self.AUTOCORR2D_70, self.AUTOCORR2D_71, self.AUTOCORR2D_72, self.AUTOCORR2D_73, self.AUTOCORR2D_74, self.AUTOCORR2D_75, self.AUTOCORR2D_76, self.AUTOCORR2D_77, self.AUTOCORR2D_78, self.AUTOCORR2D_79, self.AUTOCORR2D_80, self.AUTOCORR2D_81, self.AUTOCORR2D_82, self.AUTOCORR2D_83, self.AUTOCORR2D_84, self.AUTOCORR2D_85, self.AUTOCORR2D_86, self.AUTOCORR2D_87, self.AUTOCORR2D_88, self.AUTOCORR2D_89, self.AUTOCORR2D_90, self.AUTOCORR2D_91, self.AUTOCORR2D_92, self.AUTOCORR2D_93, self.AUTOCORR2D_94, self.AUTOCORR2D_95, self.AUTOCORR2D_96, self.AUTOCORR2D_97, self.AUTOCORR2D_98, self.AUTOCORR2D_99, self.AUTOCORR2D_100, self.AUTOCORR2D_101, self.AUTOCORR2D_102, self.AUTOCORR2D_103, self.AUTOCORR2D_104, self.AUTOCORR2D_105, self.AUTOCORR2D_106, self.AUTOCORR2D_107, self.AUTOCORR2D_108, self.AUTOCORR2D_109, self.AUTOCORR2D_110, self.AUTOCORR2D_111, self.AUTOCORR2D_112, self.AUTOCORR2D_113, self.AUTOCORR2D_114, self.AUTOCORR2D_115, self.AUTOCORR2D_116, self.AUTOCORR2D_117, self.AUTOCORR2D_118, self.AUTOCORR2D_119, self.AUTOCORR2D_120, self.AUTOCORR2D_121, self.AUTOCORR2D_122, self.AUTOCORR2D_123, self.AUTOCORR2D_124, self.AUTOCORR2D_125, self.AUTOCORR2D_126, self.AUTOCORR2D_127, self.AUTOCORR2D_128, self.AUTOCORR2D_129, self.AUTOCORR2D_130, self.AUTOCORR2D_131, self.AUTOCORR2D_132, self.AUTOCORR2D_133, self.AUTOCORR2D_134, self.AUTOCORR2D_135, self.AUTOCORR2D_136, self.AUTOCORR2D_137, self.AUTOCORR2D_138, self.AUTOCORR2D_139, self.AUTOCORR2D_140, self.AUTOCORR2D_141, self.AUTOCORR2D_142, self.AUTOCORR2D_143, self.AUTOCORR2D_144, self.AUTOCORR2D_145, self.AUTOCORR2D_146, self.AUTOCORR2D_147, self.AUTOCORR2D_148, self.AUTOCORR2D_149, self.AUTOCORR2D_150, self.AUTOCORR2D_151, self.AUTOCORR2D_152, self.AUTOCORR2D_153, self.AUTOCORR2D_154, self.AUTOCORR2D_155, self.AUTOCORR2D_156, self.AUTOCORR2D_157, self.AUTOCORR2D_158, self.AUTOCORR2D_159, self.AUTOCORR2D_160, self.AUTOCORR2D_161, self.AUTOCORR2D_162, self.AUTOCORR2D_163, self.AUTOCORR2D_164, self.AUTOCORR2D_165, self.AUTOCORR2D_166, self.AUTOCORR2D_167, self.AUTOCORR2D_168, self.AUTOCORR2D_169, self.AUTOCORR2D_170, self.AUTOCORR2D_171, self.AUTOCORR2D_172, self.AUTOCORR2D_173, self.AUTOCORR2D_174, self.AUTOCORR2D_175, self.AUTOCORR2D_176, self.AUTOCORR2D_177, self.AUTOCORR2D_178, self.AUTOCORR2D_179, self.AUTOCORR2D_180, self.AUTOCORR2D_181, self.AUTOCORR2D_182, self.AUTOCORR2D_183, self.AUTOCORR2D_184, self.AUTOCORR2D_185, self.AUTOCORR2D_186, self.AUTOCORR2D_187, self.AUTOCORR2D_188, self.AUTOCORR2D_189, self.AUTOCORR2D_190, self.AUTOCORR2D_191, self.AUTOCORR2D_192, self.BCUT2D_CHGHI, self.BCUT2D_CHGLO, self.BCUT2D_LOGPHI, self.BCUT2D_LOGPLOW, self.BCUT2D_MRHI, self.BCUT2D_MRLOW, self.BCUT2D_MWHI, self.BCUT2D_MWLOW, self.BalabanJ, self.BertzCT, self.Chi0, self.Chi0n, self.Chi0v, self.Chi1, self.Chi1n, self.Chi1v, self.Chi2n, self.Chi2v, self.Chi3n, self.Chi3v, self.Chi4n, self.Chi4v, self.EState_VSA1, self.EState_VSA2, self.EState_VSA3, self.EState_VSA4, self.EState_VSA5, self.EState_VSA6, self.EState_VSA7, self.EState_VSA8, self.EState_VSA9, self.EState_VSA10, self.EState_VSA11, self.MaxAbsEStateIndex, self.MaxEStateIndex, self.MinAbsEStateIndex, self.MinEStateIndex, self.ExactMolWt, self.FpDensityMorgan1, self.FpDensityMorgan2, self.FpDensityMorgan3, self.fr_Al_COO, self.fr_Al_OH, self.fr_Al_OH_noTert, self.fr_ArN, self.fr_Ar_COO, self.fr_Ar_N, self.fr_Ar_NH, self.fr_Ar_OH, self.fr_COO, self.fr_COO2, self.fr_C_O, self.fr_C_O_noCOO, self.fr_C_S, self.fr_HOCCN, self.fr_Imine, self.fr_NH0, self.fr_NH1, self.fr_NH2, self.fr_N_O, self.fr_Ndealkylation1, self.fr_Ndealkylation2, self.fr_Nhpyrrole, self.fr_SH, self.fr_aldehyde, self.fr_alkyl_carbamate, self.fr_alkyl_halide, self.fr_allylic_oxid, self.fr_amide, self.fr_amidine, self.fr_aniline, self.fr_aryl_methyl, self.fr_azide, self.fr_azo, self.fr_barbitur, self.fr_benzene, self.fr_benzodiazepine, self.fr_bicyclic, self.fr_diazo, self.fr_dihydropyridine, self.fr_epoxide, self.fr_ester, self.fr_ether, self.fr_furan, self.fr_guanido, self.fr_halogen, self.fr_hdrzine, self.fr_hdrzone, self.fr_imidazole, self.fr_imide, self.fr_isocyan, self.fr_isothiocyan, self.fr_ketone, self.fr_ketone_Topliss, self.fr_lactam, self.fr_lactone, self.fr_methoxy, self.fr_morpholine, self.fr_nitrile, self.fr_nitro, self.fr_nitro_arom, self.fr_nitro_arom_nonortho, self.fr_nitroso, self.fr_oxazole, self.fr_oxime, self.fr_para_hydroxylation, self.fr_phenol, self.fr_phenol_noOrthoHbond, self.fr_phos_acid, self.fr_phos_ester, self.fr_piperdine, self.fr_piperzine, self.fr_priamide, self.fr_prisulfonamd, self.fr_pyridine, self.fr_quatN, self.fr_sulfide, self.fr_sulfonamd, self.fr_sulfone, self.fr_term_acetylene, self.fr_tetrazole, self.fr_thiazole, self.fr_thiocyan, self.fr_thiophene, self.fr_unbrch_alkane, self.fr_urea, self.FractionCSP3, self.HallKierAlpha, self.HeavyAtomMolWt, self.HeavyAtomCount, self.Ipc, self.Kappa1, self.Kappa2, self.Kappa3, self.LabuteASA, self.MaxAbsPartialCharge, self.MaxPartialCharge, self.MinAbsPartialCharge, self.MinPartialCharge, self.MolLogP, self.MolMR, self.MolWt, self.NHOHCount, self.NOCount, self.NumAliphaticCarbocycles, self.NumAliphaticHeterocycles, self.NumAliphaticRings, self.NumAromaticCarbocycles, self.NumAromaticHeterocycles, self.NumAromaticRings, self.NumHAcceptors, self.NumHDonors, self.NumHeteroatoms, self.NumRadicalElectrons, self.NumRotatableBonds, self.NumSaturatedCarbocycles, self.NumSaturatedHeterocycles, self.NumSaturatedRings, self.NumValenceElectrons, self.PEOE_VSA1, self.PEOE_VSA2, self.PEOE_VSA3, self.PEOE_VSA4, self.PEOE_VSA5, self.PEOE_VSA6, self.PEOE_VSA7, self.PEOE_VSA8, self.PEOE_VSA9, self.PEOE_VSA10, self.PEOE_VSA11, self.PEOE_VSA12, self.PEOE_VSA13, self.PEOE_VSA14, self.qed, self.RingCount, self.SMR_VSA1, self.SMR_VSA2, self.SMR_VSA3, self.SMR_VSA4, self.SMR_VSA5, self.SMR_VSA6, self.SMR_VSA7, self.SMR_VSA8, self.SMR_VSA9, self.SMR_VSA10, self.SlogP_VSA1, self.SlogP_VSA2, self.SlogP_VSA3, self.SlogP_VSA4, self.SlogP_VSA5, self.SlogP_VSA6, self.SlogP_VSA7, self.SlogP_VSA8, self.SlogP_VSA9, self.SlogP_VSA10, self.SlogP_VSA11, self.SlogP_VSA12, self.TPSA, self.VSA_EState1, self.VSA_EState2, self.VSA_EState3, self.VSA_EState4, self.VSA_EState5, self.VSA_EState6, self.VSA_EState7, self.VSA_EState8, self.VSA_EState9, self.VSA_EState10, self.AUTOCORR3D_1, self.AUTOCORR3D_2, self.AUTOCORR3D_3, self.AUTOCORR3D_4, self.AUTOCORR3D_5, self.AUTOCORR3D_6, self.AUTOCORR3D_7, self.AUTOCORR3D_8, self.AUTOCORR3D_9, self.AUTOCORR3D_10, self.AUTOCORR3D_11, self.AUTOCORR3D_12, self.AUTOCORR3D_13, self.AUTOCORR3D_14, self.AUTOCORR3D_15, self.AUTOCORR3D_16, self.AUTOCORR3D_17, self.AUTOCORR3D_18, self.AUTOCORR3D_19, self.AUTOCORR3D_20, self.AUTOCORR3D_21, self.AUTOCORR3D_22, self.AUTOCORR3D_23, self.AUTOCORR3D_24, self.AUTOCORR3D_25, self.AUTOCORR3D_26, self.AUTOCORR3D_27, self.AUTOCORR3D_28, self.AUTOCORR3D_29, self.AUTOCORR3D_30, self.AUTOCORR3D_31, self.AUTOCORR3D_32, self.AUTOCORR3D_33, self.AUTOCORR3D_34, self.AUTOCORR3D_35, self.AUTOCORR3D_36, self.AUTOCORR3D_37, self.AUTOCORR3D_38, self.AUTOCORR3D_39, self.AUTOCORR3D_40, self.AUTOCORR3D_41, self.AUTOCORR3D_42, self.AUTOCORR3D_43, self.AUTOCORR3D_44, self.AUTOCORR3D_45, self.AUTOCORR3D_46, self.AUTOCORR3D_47, self.AUTOCORR3D_48, self.AUTOCORR3D_49, self.AUTOCORR3D_50, self.AUTOCORR3D_51, self.AUTOCORR3D_52, self.AUTOCORR3D_53, self.AUTOCORR3D_54, self.AUTOCORR3D_55, self.AUTOCORR3D_56, self.AUTOCORR3D_57, self.AUTOCORR3D_58, self.AUTOCORR3D_59, self.AUTOCORR3D_60, self.AUTOCORR3D_61, self.AUTOCORR3D_62, self.AUTOCORR3D_63, self.AUTOCORR3D_64, self.AUTOCORR3D_65, self.AUTOCORR3D_66, self.AUTOCORR3D_67, self.AUTOCORR3D_68, self.AUTOCORR3D_69, self.AUTOCORR3D_70, self.AUTOCORR3D_71, self.AUTOCORR3D_72, self.AUTOCORR3D_73, self.AUTOCORR3D_74, self.AUTOCORR3D_75, self.AUTOCORR3D_76, self.AUTOCORR3D_77, self.AUTOCORR3D_78, self.AUTOCORR3D_79, self.AUTOCORR3D_80, self.Asphericity, self.Eccentricity, self.InertialShapeFactor, self.NPR1, self.NPR2, self.PMI1, self.PMI2, self.PMI3, self.RadiusOfGyration, self.SpherocityIndex = data

            #endregion
        else:
            # Check if the name is empty
            if not name:
                octools.print_error("The Ligand name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            #region AUTOCORR descriptors
            self.AUTOCORR2D_1 = self.__findAUTOCORR2D_1()
            self.AUTOCORR2D_2 = self.__findAUTOCORR2D_2()
            self.AUTOCORR2D_3 = self.__findAUTOCORR2D_3()
            self.AUTOCORR2D_4 = self.__findAUTOCORR2D_4()
            self.AUTOCORR2D_5 = self.__findAUTOCORR2D_5()
            self.AUTOCORR2D_6 = self.__findAUTOCORR2D_6()
            self.AUTOCORR2D_7 = self.__findAUTOCORR2D_7()
            self.AUTOCORR2D_8 = self.__findAUTOCORR2D_8()
            self.AUTOCORR2D_9 = self.__findAUTOCORR2D_9()
            self.AUTOCORR2D_10 = self.__findAUTOCORR2D_10()
            self.AUTOCORR2D_11 = self.__findAUTOCORR2D_11()
            self.AUTOCORR2D_12 = self.__findAUTOCORR2D_12()
            self.AUTOCORR2D_13 = self.__findAUTOCORR2D_13()
            self.AUTOCORR2D_14 = self.__findAUTOCORR2D_14()
            self.AUTOCORR2D_15 = self.__findAUTOCORR2D_15()
            self.AUTOCORR2D_16 = self.__findAUTOCORR2D_16()
            self.AUTOCORR2D_17 = self.__findAUTOCORR2D_17()
            self.AUTOCORR2D_18 = self.__findAUTOCORR2D_18()
            self.AUTOCORR2D_19 = self.__findAUTOCORR2D_19()
            self.AUTOCORR2D_20 = self.__findAUTOCORR2D_20()
            self.AUTOCORR2D_21 = self.__findAUTOCORR2D_21()
            self.AUTOCORR2D_22 = self.__findAUTOCORR2D_22()
            self.AUTOCORR2D_23 = self.__findAUTOCORR2D_23()
            self.AUTOCORR2D_24 = self.__findAUTOCORR2D_24()
            self.AUTOCORR2D_25 = self.__findAUTOCORR2D_25()
            self.AUTOCORR2D_26 = self.__findAUTOCORR2D_26()
            self.AUTOCORR2D_27 = self.__findAUTOCORR2D_27()
            self.AUTOCORR2D_28 = self.__findAUTOCORR2D_28()
            self.AUTOCORR2D_29 = self.__findAUTOCORR2D_29()
            self.AUTOCORR2D_30 = self.__findAUTOCORR2D_30()
            self.AUTOCORR2D_31 = self.__findAUTOCORR2D_31()
            self.AUTOCORR2D_32 = self.__findAUTOCORR2D_32()
            self.AUTOCORR2D_33 = self.__findAUTOCORR2D_33()
            self.AUTOCORR2D_34 = self.__findAUTOCORR2D_34()
            self.AUTOCORR2D_35 = self.__findAUTOCORR2D_35()
            self.AUTOCORR2D_36 = self.__findAUTOCORR2D_36()
            self.AUTOCORR2D_37 = self.__findAUTOCORR2D_37()
            self.AUTOCORR2D_38 = self.__findAUTOCORR2D_38()
            self.AUTOCORR2D_39 = self.__findAUTOCORR2D_39()
            self.AUTOCORR2D_40 = self.__findAUTOCORR2D_40()
            self.AUTOCORR2D_41 = self.__findAUTOCORR2D_41()
            self.AUTOCORR2D_42 = self.__findAUTOCORR2D_42()
            self.AUTOCORR2D_43 = self.__findAUTOCORR2D_43()
            self.AUTOCORR2D_44 = self.__findAUTOCORR2D_44()
            self.AUTOCORR2D_45 = self.__findAUTOCORR2D_45()
            self.AUTOCORR2D_46 = self.__findAUTOCORR2D_46()
            self.AUTOCORR2D_47 = self.__findAUTOCORR2D_47()
            self.AUTOCORR2D_48 = self.__findAUTOCORR2D_48()
            self.AUTOCORR2D_49 = self.__findAUTOCORR2D_49()
            self.AUTOCORR2D_50 = self.__findAUTOCORR2D_50()
            self.AUTOCORR2D_51 = self.__findAUTOCORR2D_51()
            self.AUTOCORR2D_52 = self.__findAUTOCORR2D_52()
            self.AUTOCORR2D_53 = self.__findAUTOCORR2D_53()
            self.AUTOCORR2D_54 = self.__findAUTOCORR2D_54()
            self.AUTOCORR2D_55 = self.__findAUTOCORR2D_55()
            self.AUTOCORR2D_56 = self.__findAUTOCORR2D_56()
            self.AUTOCORR2D_57 = self.__findAUTOCORR2D_57()
            self.AUTOCORR2D_58 = self.__findAUTOCORR2D_58()
            self.AUTOCORR2D_59 = self.__findAUTOCORR2D_59()
            self.AUTOCORR2D_60 = self.__findAUTOCORR2D_60()
            self.AUTOCORR2D_61 = self.__findAUTOCORR2D_61()
            self.AUTOCORR2D_62 = self.__findAUTOCORR2D_62()
            self.AUTOCORR2D_63 = self.__findAUTOCORR2D_63()
            self.AUTOCORR2D_64 = self.__findAUTOCORR2D_64()
            self.AUTOCORR2D_65 = self.__findAUTOCORR2D_65()
            self.AUTOCORR2D_66 = self.__findAUTOCORR2D_66()
            self.AUTOCORR2D_67 = self.__findAUTOCORR2D_67()
            self.AUTOCORR2D_68 = self.__findAUTOCORR2D_68()
            self.AUTOCORR2D_69 = self.__findAUTOCORR2D_69()
            self.AUTOCORR2D_70 = self.__findAUTOCORR2D_70()
            self.AUTOCORR2D_71 = self.__findAUTOCORR2D_71()
            self.AUTOCORR2D_72 = self.__findAUTOCORR2D_72()
            self.AUTOCORR2D_73 = self.__findAUTOCORR2D_73()
            self.AUTOCORR2D_74 = self.__findAUTOCORR2D_74()
            self.AUTOCORR2D_75 = self.__findAUTOCORR2D_75()
            self.AUTOCORR2D_76 = self.__findAUTOCORR2D_76()
            self.AUTOCORR2D_77 = self.__findAUTOCORR2D_77()
            self.AUTOCORR2D_78 = self.__findAUTOCORR2D_78()
            self.AUTOCORR2D_79 = self.__findAUTOCORR2D_79()
            self.AUTOCORR2D_80 = self.__findAUTOCORR2D_80()
            self.AUTOCORR2D_81 = self.__findAUTOCORR2D_81()
            self.AUTOCORR2D_82 = self.__findAUTOCORR2D_82()
            self.AUTOCORR2D_83 = self.__findAUTOCORR2D_83()
            self.AUTOCORR2D_84 = self.__findAUTOCORR2D_84()
            self.AUTOCORR2D_85 = self.__findAUTOCORR2D_85()
            self.AUTOCORR2D_86 = self.__findAUTOCORR2D_86()
            self.AUTOCORR2D_87 = self.__findAUTOCORR2D_87()
            self.AUTOCORR2D_88 = self.__findAUTOCORR2D_88()
            self.AUTOCORR2D_89 = self.__findAUTOCORR2D_89()
            self.AUTOCORR2D_90 = self.__findAUTOCORR2D_90()
            self.AUTOCORR2D_91 = self.__findAUTOCORR2D_91()
            self.AUTOCORR2D_92 = self.__findAUTOCORR2D_92()
            self.AUTOCORR2D_93 = self.__findAUTOCORR2D_93()
            self.AUTOCORR2D_94 = self.__findAUTOCORR2D_94()
            self.AUTOCORR2D_95 = self.__findAUTOCORR2D_95()
            self.AUTOCORR2D_96 = self.__findAUTOCORR2D_96()
            self.AUTOCORR2D_97 = self.__findAUTOCORR2D_97()
            self.AUTOCORR2D_98 = self.__findAUTOCORR2D_98()
            self.AUTOCORR2D_99 = self.__findAUTOCORR2D_99()
            self.AUTOCORR2D_100 = self.__findAUTOCORR2D_100()
            self.AUTOCORR2D_101 = self.__findAUTOCORR2D_101()
            self.AUTOCORR2D_102 = self.__findAUTOCORR2D_102()
            self.AUTOCORR2D_103 = self.__findAUTOCORR2D_103()
            self.AUTOCORR2D_104 = self.__findAUTOCORR2D_104()
            self.AUTOCORR2D_105 = self.__findAUTOCORR2D_105()
            self.AUTOCORR2D_106 = self.__findAUTOCORR2D_106()
            self.AUTOCORR2D_107 = self.__findAUTOCORR2D_107()
            self.AUTOCORR2D_108 = self.__findAUTOCORR2D_108()
            self.AUTOCORR2D_109 = self.__findAUTOCORR2D_109()
            self.AUTOCORR2D_110 = self.__findAUTOCORR2D_110()
            self.AUTOCORR2D_111 = self.__findAUTOCORR2D_111()
            self.AUTOCORR2D_112 = self.__findAUTOCORR2D_112()
            self.AUTOCORR2D_113 = self.__findAUTOCORR2D_113()
            self.AUTOCORR2D_114 = self.__findAUTOCORR2D_114()
            self.AUTOCORR2D_115 = self.__findAUTOCORR2D_115()
            self.AUTOCORR2D_116 = self.__findAUTOCORR2D_116()
            self.AUTOCORR2D_117 = self.__findAUTOCORR2D_117()
            self.AUTOCORR2D_118 = self.__findAUTOCORR2D_118()
            self.AUTOCORR2D_119 = self.__findAUTOCORR2D_119()
            self.AUTOCORR2D_120 = self.__findAUTOCORR2D_120()
            self.AUTOCORR2D_121 = self.__findAUTOCORR2D_121()
            self.AUTOCORR2D_122 = self.__findAUTOCORR2D_122()
            self.AUTOCORR2D_123 = self.__findAUTOCORR2D_123()
            self.AUTOCORR2D_124 = self.__findAUTOCORR2D_124()
            self.AUTOCORR2D_125 = self.__findAUTOCORR2D_125()
            self.AUTOCORR2D_126 = self.__findAUTOCORR2D_126()
            self.AUTOCORR2D_127 = self.__findAUTOCORR2D_127()
            self.AUTOCORR2D_128 = self.__findAUTOCORR2D_128()
            self.AUTOCORR2D_129 = self.__findAUTOCORR2D_129()
            self.AUTOCORR2D_130 = self.__findAUTOCORR2D_130()
            self.AUTOCORR2D_131 = self.__findAUTOCORR2D_131()
            self.AUTOCORR2D_132 = self.__findAUTOCORR2D_132()
            self.AUTOCORR2D_133 = self.__findAUTOCORR2D_133()
            self.AUTOCORR2D_134 = self.__findAUTOCORR2D_134()
            self.AUTOCORR2D_135 = self.__findAUTOCORR2D_135()
            self.AUTOCORR2D_136 = self.__findAUTOCORR2D_136()
            self.AUTOCORR2D_137 = self.__findAUTOCORR2D_137()
            self.AUTOCORR2D_138 = self.__findAUTOCORR2D_138()
            self.AUTOCORR2D_139 = self.__findAUTOCORR2D_139()
            self.AUTOCORR2D_140 = self.__findAUTOCORR2D_140()
            self.AUTOCORR2D_141 = self.__findAUTOCORR2D_141()
            self.AUTOCORR2D_142 = self.__findAUTOCORR2D_142()
            self.AUTOCORR2D_143 = self.__findAUTOCORR2D_143()
            self.AUTOCORR2D_144 = self.__findAUTOCORR2D_144()
            self.AUTOCORR2D_145 = self.__findAUTOCORR2D_145()
            self.AUTOCORR2D_146 = self.__findAUTOCORR2D_146()
            self.AUTOCORR2D_147 = self.__findAUTOCORR2D_147()
            self.AUTOCORR2D_148 = self.__findAUTOCORR2D_148()
            self.AUTOCORR2D_149 = self.__findAUTOCORR2D_149()
            self.AUTOCORR2D_150 = self.__findAUTOCORR2D_150()
            self.AUTOCORR2D_151 = self.__findAUTOCORR2D_151()
            self.AUTOCORR2D_152 = self.__findAUTOCORR2D_152()
            self.AUTOCORR2D_153 = self.__findAUTOCORR2D_153()
            self.AUTOCORR2D_154 = self.__findAUTOCORR2D_154()
            self.AUTOCORR2D_155 = self.__findAUTOCORR2D_155()
            self.AUTOCORR2D_156 = self.__findAUTOCORR2D_156()
            self.AUTOCORR2D_157 = self.__findAUTOCORR2D_157()
            self.AUTOCORR2D_158 = self.__findAUTOCORR2D_158()
            self.AUTOCORR2D_159 = self.__findAUTOCORR2D_159()
            self.AUTOCORR2D_160 = self.__findAUTOCORR2D_160()
            self.AUTOCORR2D_161 = self.__findAUTOCORR2D_161()
            self.AUTOCORR2D_162 = self.__findAUTOCORR2D_162()
            self.AUTOCORR2D_163 = self.__findAUTOCORR2D_163()
            self.AUTOCORR2D_164 = self.__findAUTOCORR2D_164()
            self.AUTOCORR2D_165 = self.__findAUTOCORR2D_165()
            self.AUTOCORR2D_166 = self.__findAUTOCORR2D_166()
            self.AUTOCORR2D_167 = self.__findAUTOCORR2D_167()
            self.AUTOCORR2D_168 = self.__findAUTOCORR2D_168()
            self.AUTOCORR2D_169 = self.__findAUTOCORR2D_169()
            self.AUTOCORR2D_170 = self.__findAUTOCORR2D_170()
            self.AUTOCORR2D_171 = self.__findAUTOCORR2D_171()
            self.AUTOCORR2D_172 = self.__findAUTOCORR2D_172()
            self.AUTOCORR2D_173 = self.__findAUTOCORR2D_173()
            self.AUTOCORR2D_174 = self.__findAUTOCORR2D_174()
            self.AUTOCORR2D_175 = self.__findAUTOCORR2D_175()
            self.AUTOCORR2D_176 = self.__findAUTOCORR2D_176()
            self.AUTOCORR2D_177 = self.__findAUTOCORR2D_177()
            self.AUTOCORR2D_178 = self.__findAUTOCORR2D_178()
            self.AUTOCORR2D_179 = self.__findAUTOCORR2D_179()
            self.AUTOCORR2D_180 = self.__findAUTOCORR2D_180()
            self.AUTOCORR2D_181 = self.__findAUTOCORR2D_181()
            self.AUTOCORR2D_182 = self.__findAUTOCORR2D_182()
            self.AUTOCORR2D_183 = self.__findAUTOCORR2D_183()
            self.AUTOCORR2D_184 = self.__findAUTOCORR2D_184()
            self.AUTOCORR2D_185 = self.__findAUTOCORR2D_185()
            self.AUTOCORR2D_186 = self.__findAUTOCORR2D_186()
            self.AUTOCORR2D_187 = self.__findAUTOCORR2D_187()
            self.AUTOCORR2D_188 = self.__findAUTOCORR2D_188()
            self.AUTOCORR2D_189 = self.__findAUTOCORR2D_189()
            self.AUTOCORR2D_190 = self.__findAUTOCORR2D_190()
            self.AUTOCORR2D_191 = self.__findAUTOCORR2D_191()
            self.AUTOCORR2D_192 = self.__findAUTOCORR2D_192()
            #endregion

            #region BCUT2D descriptors
            self.BCUT2D_CHGHI = self.__findBCUT2D_CHGHI()
            self.BCUT2D_CHGLO = self.__findBCUT2D_CHGLO()
            self.BCUT2D_LOGPHI = self.__findBCUT2D_LOGPHI()
            self.BCUT2D_LOGPLOW = self.__findBCUT2D_LOGPLOW()
            self.BCUT2D_MRHI = self.__findBCUT2D_MRHI()
            self.BCUT2D_MRLOW = self.__findBCUT2D_MRLOW()
            self.BCUT2D_MWHI = self.__findBCUT2D_MWHI()
            self.BCUT2D_MWLOW = self.__findBCUT2D_MWLOW()
            #endregion

            self.BalabanJ = self.__findBalabanJ()
            self.BertzCT = self.__findBertzCT()

            #region Chi descriptors
            self.Chi0 = self.__findChi0()
            self.Chi0n = self.__findChi0n()
            self.Chi0v = self.__findChi0v()
            self.Chi1 = self.__findChi1()
            self.Chi1n = self.__findChi1n()
            self.Chi1v = self.__findChi1v()
            self.Chi2n = self.__findChi2n()
            self.Chi2v = self.__findChi2v()
            self.Chi3n = self.__findChi3n()
            self.Chi3v = self.__findChi3v()
            self.Chi4n = self.__findChi4n()
            self.Chi4v = self.__findChi4v()
            #endregion

            #region EState descriptors
            self.EState_VSA1 = self.__findEState_VSA1()
            self.EState_VSA2 = self.__findEState_VSA2()
            self.EState_VSA3 = self.__findEState_VSA3()
            self.EState_VSA4 = self.__findEState_VSA4()
            self.EState_VSA5 = self.__findEState_VSA5()
            self.EState_VSA6 = self.__findEState_VSA6()
            self.EState_VSA7 = self.__findEState_VSA7()
            self.EState_VSA8 = self.__findEState_VSA8()
            self.EState_VSA9 = self.__findEState_VSA9()
            self.EState_VSA10 = self.__findEState_VSA10()
            self.EState_VSA11 = self.__findEState_VSA11()

            self.MaxAbsEStateIndex = self.__findMaxAbsEStateIndex()
            self.MaxEStateIndex = self.__findMaxEStateIndex()
            self.MinAbsEStateIndex = self.__findMinAbsEStateIndex()
            self.MinEStateIndex = self.__findMinEStateIndex()
            #endregion

            self.ExactMolWt = self.__findExactMolWt()
            self.FpDensityMorgan1 = self.__findFpDensityMorgan1()
            self.FpDensityMorgan2 = self.__findFpDensityMorgan2()
            self.FpDensityMorgan3 = self.__findFpDensityMorgan3()

            #region fr_ descriptors
            self.fr_Al_COO = self.__findfr_Al_COO()
            self.fr_Al_OH = self.__findfr_Al_OH()
            self.fr_Al_OH_noTert = self.__findfr_Al_OH_noTert()
            self.fr_ArN = self.__findfr_ArN()
            self.fr_Ar_COO = self.__findfr_Ar_COO()
            self.fr_Ar_N = self.__findfr_Ar_N()
            self.fr_Ar_NH = self.__findfr_Ar_NH()
            self.fr_Ar_OH = self.__findfr_Ar_OH()
            self.fr_COO = self.__findfr_COO()
            self.fr_COO2 = self.__findfr_COO2()
            self.fr_C_O = self.__findfr_C_O()
            self.fr_C_O_noCOO = self.__findfr_C_O_noCOO()
            self.fr_C_S = self.__findfr_C_S()
            self.fr_HOCCN = self.__findfr_HOCCN()
            self.fr_Imine = self.__findfr_Imine()
            self.fr_NH0 = self.__findfr_NH0()
            self.fr_NH1 = self.__findfr_NH1()
            self.fr_NH2 = self.__findfr_NH2()
            self.fr_N_O = self.__findfr_N_O()
            self.fr_Ndealkylation1 = self.__findfr_Ndealkylation1()
            self.fr_Ndealkylation2 = self.__findfr_Ndealkylation2()
            self.fr_Nhpyrrole = self.__findfr_Nhpyrrole()
            self.fr_SH = self.__findfr_SH()
            self.fr_aldehyde = self.__findfr_aldehyde()
            self.fr_alkyl_carbamate = self.__findfr_alkyl_carbamate()
            self.fr_alkyl_halide = self.__findfr_alkyl_halide()
            self.fr_allylic_oxid = self.__findfr_allylic_oxid()
            self.fr_amide = self.__findfr_amide()
            self.fr_amidine = self.__findfr_amidine()
            self.fr_aniline = self.__findfr_aniline()
            self.fr_aryl_methyl = self.__findfr_aryl_methyl()
            self.fr_azide = self.__findfr_azide()
            self.fr_azo = self.__findfr_azo()
            self.fr_barbitur = self.__findfr_barbitur()
            self.fr_benzene = self.__findfr_benzene()
            self.fr_benzodiazepine = self.__findfr_benzodiazepine()
            self.fr_bicyclic = self.__findfr_bicyclic()
            self.fr_diazo = self.__findfr_diazo()
            self.fr_dihydropyridine = self.__findfr_dihydropyridine()
            self.fr_epoxide = self.__findfr_epoxide()
            self.fr_ester = self.__findfr_ester()
            self.fr_ether = self.__findfr_ether()
            self.fr_furan = self.__findfr_furan()
            self.fr_guanido = self.__findfr_guanido()
            self.fr_halogen = self.__findfr_halogen()
            self.fr_hdrzine = self.__findfr_hdrzine()
            self.fr_hdrzone = self.__findfr_hdrzone()
            self.fr_imidazole = self.__findfr_imidazole()
            self.fr_imide = self.__findfr_imide()
            self.fr_isocyan = self.__findfr_isocyan()
            self.fr_isothiocyan = self.__findfr_isothiocyan()
            self.fr_ketone = self.__findfr_ketone()
            self.fr_ketone_Topliss = self.__findfr_ketone_Topliss()
            self.fr_lactam = self.__findfr_lactam()
            self.fr_lactone = self.__findfr_lactone()
            self.fr_methoxy = self.__findfr_methoxy()
            self.fr_morpholine = self.__findfr_morpholine()
            self.fr_nitrile = self.__findfr_nitrile()
            self.fr_nitro = self.__findfr_nitro()
            self.fr_nitro_arom = self.__findfr_nitro_arom()
            self.fr_nitro_arom_nonortho = self.__findfr_nitro_arom_nonortho()
            self.fr_nitroso = self.__findfr_nitroso()
            self.fr_oxazole = self.__findfr_oxazole()
            self.fr_oxime = self.__findfr_oxime()
            self.fr_para_hydroxylation = self.__findfr_para_hydroxylation()
            self.fr_phenol = self.__findfr_phenol()
            self.fr_phenol_noOrthoHbond = self.__findfr_phenol_noOrthoHbond()
            self.fr_phos_acid = self.__findfr_phos_acid()
            self.fr_phos_ester = self.__findfr_phos_ester()
            self.fr_piperdine = self.__findfr_piperdine()
            self.fr_piperzine = self.__findfr_piperzine()
            self.fr_priamide = self.__findfr_priamide()
            self.fr_prisulfonamd = self.__findfr_prisulfonamd()
            self.fr_pyridine = self.__findfr_pyridine()
            self.fr_quatN = self.__findfr_quatN()
            self.fr_sulfide = self.__findfr_sulfide()
            self.fr_sulfonamd = self.__findfr_sulfonamd()
            self.fr_sulfone = self.__findfr_sulfone()
            self.fr_term_acetylene = self.__findfr_term_acetylene()
            self.fr_tetrazole = self.__findfr_tetrazole()
            self.fr_thiazole = self.__findfr_thiazole()
            self.fr_thiocyan = self.__findfr_thiocyan()
            self.fr_thiophene = self.__findfr_thiophene()
            self.fr_unbrch_alkane = self.__findfr_unbrch_alkane()
            self.fr_urea = self.__findfr_urea()

            #endregion

            self.FractionCSP3 = self.__findFractionCSP3()
            self.HallKierAlpha = self.__findHallKierAlpha()
            self.HeavyAtomMolWt = self.__findHeavyAtomMolWt()
            self.HeavyAtomCount = self.__findHeavyAtomCount()
            self.Ipc = self.__findIpc()

            #region Kappa descriptors
            self.Kappa1 = self.__findKappa1()
            self.Kappa2 = self.__findKappa2()
            self.Kappa3 = self.__findKappa3()

            #endregion

            self.LabuteASA = self.__findLabuteASA()
            self.MaxAbsPartialCharge = self.__findMaxAbsPartialCharge()
            self.MaxPartialCharge = self.__findMaxPartialCharge()
            self.MinAbsPartialCharge = self.__findMinAbsPartialCharge()
            self.MinPartialCharge = self.__findMinPartialCharge()
            self.MolLogP = self.__findMolWt()
            self.MolMR = self.__findMolMR()
            self.MolWt = self.__findMolWt()

            #region 'count' descriptors
            self.NHOHCount = self.__findNHOHCount()
            self.NOCount = self.__findNOCount()
            self.NumAliphaticHeterocycles = self.__findNumAliphaticHeterocycles()
            self.NumAliphaticRings = self.__findNumAliphaticRings()
            self.NumAromaticCarbocycles = self.__findNumAromaticCarbocycles()
            self.NumAromaticHeterocycles = self.__findNumAromaticHeterocycles()
            self.NumAromaticRings = self.__findNumAromaticRings()
            self.NumHAcceptors = self.__findNumHAcceptors()
            self.NumHDonors = self.__findNumHDonors()
            self.NumHeteroatoms = self.__findNumHeteroatoms()
            self.NumRadicalElectrons = self.__findNumRadicalElectrons()
            self.NumRotatableBonds = self.__findNumRotatableBonds()
            self.NumSaturatedCarbocycles = self.__findNumSaturatedCarbocycles()
            self.NumSaturatedHeterocycles = self.__findNumSaturatedHeterocycles()
            self.NumSaturatedRings = self.__findNumSaturatedRings()
            self.NumValenceElectrons = self.__findNumValenceElectrons()
            self.NumAliphaticCarbocycles = self.__findNumAliphaticCarbocycles()
            self.RingCount = self.__findRingCount()

            #endregion

            #region PEOE_VSA descriptors
            self.PEOE_VSA1 = self.__findPEOE_VSA1()
            self.PEOE_VSA2 = self.__findPEOE_VSA2()
            self.PEOE_VSA3 = self.__findPEOE_VSA3()
            self.PEOE_VSA4 = self.__findPEOE_VSA4()
            self.PEOE_VSA5 = self.__findPEOE_VSA5()
            self.PEOE_VSA6 = self.__findPEOE_VSA6()
            self.PEOE_VSA7 = self.__findPEOE_VSA7()
            self.PEOE_VSA8 = self.__findPEOE_VSA8()
            self.PEOE_VSA9 = self.__findPEOE_VSA9()
            self.PEOE_VSA10 = self.__findPEOE_VSA10()
            self.PEOE_VSA11 = self.__findPEOE_VSA11()
            self.PEOE_VSA12 = self.__findPEOE_VSA12()
            self.PEOE_VSA13 = self.__findPEOE_VSA13()
            self.PEOE_VSA14 = self.__findPEOE_VSA14()

            #endregion

            self.qed = self.__findqed()

            #region SMR_VSA descriptors
            self.SMR_VSA1 = self.__findSMR_VSA1()
            self.SMR_VSA2 = self.__findSMR_VSA2()
            self.SMR_VSA3 = self.__findSMR_VSA3()
            self.SMR_VSA4 = self.__findSMR_VSA4()
            self.SMR_VSA5 = self.__findSMR_VSA5()
            self.SMR_VSA6 = self.__findSMR_VSA6()
            self.SMR_VSA7 = self.__findSMR_VSA7()
            self.SMR_VSA8 = self.__findSMR_VSA8()
            self.SMR_VSA9 = self.__findSMR_VSA9()
            self.SMR_VSA10 = self.__findSMR_VSA10()

            #endregion

            #region SlogP_VSA descriptors
            self.SlogP_VSA1 = self.__findSlogP_VSA1()
            self.SlogP_VSA2 = self.__findSlogP_VSA2()
            self.SlogP_VSA3 = self.__findSlogP_VSA3()
            self.SlogP_VSA4 = self.__findSlogP_VSA4()
            self.SlogP_VSA5 = self.__findSlogP_VSA5()
            self.SlogP_VSA6 = self.__findSlogP_VSA6()
            self.SlogP_VSA7 = self.__findSlogP_VSA7()
            self.SlogP_VSA8 = self.__findSlogP_VSA8()
            self.SlogP_VSA9 = self.__findSlogP_VSA9()
            self.SlogP_VSA10 = self.__findSlogP_VSA10()
            self.SlogP_VSA11 = self.__findSlogP_VSA11()
            self.SlogP_VSA12 = self.__findSlogP_VSA12()

            #endregion

            self.TPSA = self.__findTPSA()

            #region VSA_EState descriptors
            self.VSA_EState1 = self.__findVSA_EState1()
            self.VSA_EState2 = self.__findVSA_EState2()
            self.VSA_EState3 = self.__findVSA_EState3()
            self.VSA_EState4 = self.__findVSA_EState4()
            self.VSA_EState5 = self.__findVSA_EState5()
            self.VSA_EState6 = self.__findVSA_EState6()
            self.VSA_EState7 = self.__findVSA_EState7()
            self.VSA_EState8 = self.__findVSA_EState8()
            self.VSA_EState9 = self.__findVSA_EState9()
            self.VSA_EState10 = self.__findVSA_EState10()

            #endregion

            #region 3D descriptors
            self.AUTOCORR3D_1,  self.AUTOCORR3D_2,  self.AUTOCORR3D_3,  self.AUTOCORR3D_4,  self.AUTOCORR3D_5,  self.AUTOCORR3D_6,  self.AUTOCORR3D_7,  self.AUTOCORR3D_8,  self.AUTOCORR3D_9,  self.AUTOCORR3D_10, self.AUTOCORR3D_11, self.AUTOCORR3D_12, self.AUTOCORR3D_13, self.AUTOCORR3D_14, self.AUTOCORR3D_15, self.AUTOCORR3D_16, self.AUTOCORR3D_17, self.AUTOCORR3D_18, self.AUTOCORR3D_19, self.AUTOCORR3D_20, self.AUTOCORR3D_21, self.AUTOCORR3D_22, self.AUTOCORR3D_23, self.AUTOCORR3D_24, self.AUTOCORR3D_25, self.AUTOCORR3D_26, self.AUTOCORR3D_27, self.AUTOCORR3D_28, self.AUTOCORR3D_29, self.AUTOCORR3D_30, self.AUTOCORR3D_31, self.AUTOCORR3D_32, self.AUTOCORR3D_33, self.AUTOCORR3D_34, self.AUTOCORR3D_35, self.AUTOCORR3D_36, self.AUTOCORR3D_37, self.AUTOCORR3D_38, self.AUTOCORR3D_39, self.AUTOCORR3D_40, self.AUTOCORR3D_41, self.AUTOCORR3D_42, self.AUTOCORR3D_43, self.AUTOCORR3D_44, self.AUTOCORR3D_45, self.AUTOCORR3D_46, self.AUTOCORR3D_47, self.AUTOCORR3D_48, self.AUTOCORR3D_49, self.AUTOCORR3D_50, self.AUTOCORR3D_51, self.AUTOCORR3D_52, self.AUTOCORR3D_53, self.AUTOCORR3D_54, self.AUTOCORR3D_55, self.AUTOCORR3D_56, self.AUTOCORR3D_57, self.AUTOCORR3D_58, self.AUTOCORR3D_59, self.AUTOCORR3D_60, self.AUTOCORR3D_61, self.AUTOCORR3D_62, self.AUTOCORR3D_63, self.AUTOCORR3D_64, self.AUTOCORR3D_65, self.AUTOCORR3D_66, self.AUTOCORR3D_67, self.AUTOCORR3D_68, self.AUTOCORR3D_69, self.AUTOCORR3D_70, self.AUTOCORR3D_71, self.AUTOCORR3D_72, self.AUTOCORR3D_73, self.AUTOCORR3D_74, self.AUTOCORR3D_75, self.AUTOCORR3D_76, self.AUTOCORR3D_77, self.AUTOCORR3D_78, self.AUTOCORR3D_79, self.AUTOCORR3D_80 = self.__findAUTOCORR3D()

            self.Asphericity = self.__findAsphericity()
            self.Eccentricity = self.__findEccentricity()
            self.InertialShapeFactor = self.__findInertialShapeFactor()
            self.NPR1 = self.__findNPR1()
            self.NPR2 = self.__findNPR2()
            self.PMI1 = self.__findPMI1()
            self.PMI2 = self.__findPMI2()
            self.PMI3 = self.__findPMI3()
            self.RadiusOfGyration = self.__findRadiusOfGyration()
            self.SpherocityIndex = self.__findSpherocityIndex()

            #endregion

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

    def __read_descriptors_from_json(self, path):
        '''
        Read the descriptors from a json file.
        Input:
          -
        Return:
          [list(mixed)] - Descriptors read from the json file. If fails, returns null.
        '''
        return read_descriptors_from_json(path)

    def __safe_to_dict(self):
        '''
        Return all the properties (except the molecule object) for the Ligand object.
        Input:
          -
        Return:
          [dict of mixed]
        '''
        # Create new dict
        properties = dict()
        # Set Name and Path
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        # Combine both in one dict and return them
        return {**properties, **self.get_descriptors()}

        return properties

    #region AUTOCORR descriptors
    def __findAUTOCORR2D_1(self):
        '''
        Compute the autocorrelation2D_1 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_1(self.molecule)

    def __findAUTOCORR2D_2(self):
        '''
        Compute the autocorrelation2D_2 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_2(self.molecule)

    def __findAUTOCORR2D_3(self):
        '''
        Compute the autocorrelation2D_3 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_3(self.molecule)

    def __findAUTOCORR2D_4(self):
        '''
        Compute the autocorrelation2D_4 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_4(self.molecule)

    def __findAUTOCORR2D_5(self):
        '''
        Compute the autocorrelation2D_5 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_5(self.molecule)

    def __findAUTOCORR2D_6(self):
        '''
        Compute the autocorrelation2D_6 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_6(self.molecule)

    def __findAUTOCORR2D_7(self):
        '''
        Compute the autocorrelation2D_7 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_7(self.molecule)

    def __findAUTOCORR2D_8(self):
        '''
        Compute the autocorrelation2D_8 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_8(self.molecule)

    def __findAUTOCORR2D_9(self):
        '''
        Compute the autocorrelation2D_9 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_9(self.molecule)

    def __findAUTOCORR2D_10(self):
        '''
        Compute the autocorrelation2D_10 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_10(self.molecule)

    def __findAUTOCORR2D_11(self):
        '''
        Compute the autocorrelation2D_11 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_11 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_11(self.molecule)

    def __findAUTOCORR2D_12(self):
        '''
        Compute the autocorrelation2D_12 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_12 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_12(self.molecule)

    def __findAUTOCORR2D_13(self):
        '''
        Compute the autocorrelation2D_13 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_13 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_13(self.molecule)

    def __findAUTOCORR2D_14(self):
        '''
        Compute the autocorrelation2D_14 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_14 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_14(self.molecule)

    def __findAUTOCORR2D_15(self):
        '''
        Compute the autocorrelation2D_15 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_15 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_15(self.molecule)

    def __findAUTOCORR2D_16(self):
        '''
        Compute the autocorrelation2D_16 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_16 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_16(self.molecule)

    def __findAUTOCORR2D_17(self):
        '''
        Compute the autocorrelation2D_17 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_17 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_17(self.molecule)

    def __findAUTOCORR2D_18(self):
        '''
        Compute the autocorrelation2D_18 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_18 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_18(self.molecule)

    def __findAUTOCORR2D_19(self):
        '''
        Compute the autocorrelation2D_19 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_19 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_19(self.molecule)

    def __findAUTOCORR2D_20(self):
        '''
        Compute the autocorrelation2D_20 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_20 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_20(self.molecule)

    def __findAUTOCORR2D_21(self):
        '''
        Compute the autocorrelation2D_21 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_21 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_21(self.molecule)

    def __findAUTOCORR2D_22(self):
        '''
        Compute the autocorrelation2D_22 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_22 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_22(self.molecule)

    def __findAUTOCORR2D_23(self):
        '''
        Compute the autocorrelation2D_23 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_23 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_23(self.molecule)

    def __findAUTOCORR2D_24(self):
        '''
        Compute the autocorrelation2D_24 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_24 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_24(self.molecule)

    def __findAUTOCORR2D_25(self):
        '''
        Compute the autocorrelation2D_25 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_25 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_25(self.molecule)

    def __findAUTOCORR2D_26(self):
        '''
        Compute the autocorrelation2D_26 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_26 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_26(self.molecule)

    def __findAUTOCORR2D_27(self):
        '''
        Compute the autocorrelation2D_27 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_27 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_27(self.molecule)

    def __findAUTOCORR2D_28(self):
        '''
        Compute the autocorrelation2D_28 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_28 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_28(self.molecule)

    def __findAUTOCORR2D_29(self):
        '''
        Compute the autocorrelation2D_29 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_29 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_29(self.molecule)

    def __findAUTOCORR2D_30(self):
        '''
        Compute the autocorrelation2D_30 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_30 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_30(self.molecule)

    def __findAUTOCORR2D_31(self):
        '''
        Compute the autocorrelation2D_31 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_31 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_31(self.molecule)

    def __findAUTOCORR2D_32(self):
        '''
        Compute the autocorrelation2D_32 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_32 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_32(self.molecule)

    def __findAUTOCORR2D_33(self):
        '''
        Compute the autocorrelation2D_33 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_33 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_33(self.molecule)

    def __findAUTOCORR2D_34(self):
        '''
        Compute the autocorrelation2D_34 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_34 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_34(self.molecule)

    def __findAUTOCORR2D_35(self):
        '''
        Compute the autocorrelation2D_35 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_35 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_35(self.molecule)

    def __findAUTOCORR2D_36(self):
        '''
        Compute the autocorrelation2D_36 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_36 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_36(self.molecule)

    def __findAUTOCORR2D_37(self):
        '''
        Compute the autocorrelation2D_37 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_37 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_37(self.molecule)

    def __findAUTOCORR2D_38(self):
        '''
        Compute the autocorrelation2D_38 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_38 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_38(self.molecule)

    def __findAUTOCORR2D_39(self):
        '''
        Compute the autocorrelation2D_39 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_39 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_39(self.molecule)

    def __findAUTOCORR2D_40(self):
        '''
        Compute the autocorrelation2D_40 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_40 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_40(self.molecule)

    def __findAUTOCORR2D_41(self):
        '''
        Compute the autocorrelation2D_41 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_41 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_41(self.molecule)

    def __findAUTOCORR2D_42(self):
        '''
        Compute the autocorrelation2D_42 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_42 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_42(self.molecule)

    def __findAUTOCORR2D_43(self):
        '''
        Compute the autocorrelation2D_43 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_43 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_43(self.molecule)

    def __findAUTOCORR2D_44(self):
        '''
        Compute the autocorrelation2D_44 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_44 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_44(self.molecule)

    def __findAUTOCORR2D_45(self):
        '''
        Compute the autocorrelation2D_45 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_45 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_45(self.molecule)

    def __findAUTOCORR2D_46(self):
        '''
        Compute the autocorrelation2D_46 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_46 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_46(self.molecule)

    def __findAUTOCORR2D_47(self):
        '''
        Compute the autocorrelation2D_47 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_47 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_47(self.molecule)

    def __findAUTOCORR2D_48(self):
        '''
        Compute the autocorrelation2D_48 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_48 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_48(self.molecule)

    def __findAUTOCORR2D_49(self):
        '''
        Compute the autocorrelation2D_49 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_49 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_49(self.molecule)

    def __findAUTOCORR2D_50(self):
        '''
        Compute the autocorrelation2D_50 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_50 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_50(self.molecule)

    def __findAUTOCORR2D_51(self):
        '''
        Compute the autocorrelation2D_51 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_51 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_51(self.molecule)

    def __findAUTOCORR2D_52(self):
        '''
        Compute the autocorrelation2D_52 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_52 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_52(self.molecule)

    def __findAUTOCORR2D_53(self):
        '''
        Compute the autocorrelation2D_53 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_53 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_53(self.molecule)

    def __findAUTOCORR2D_54(self):
        '''
        Compute the autocorrelation2D_54 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_54 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_54(self.molecule)

    def __findAUTOCORR2D_55(self):
        '''
        Compute the autocorrelation2D_55 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_55 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_55(self.molecule)

    def __findAUTOCORR2D_56(self):
        '''
        Compute the autocorrelation2D_56 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_56 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_56(self.molecule)

    def __findAUTOCORR2D_57(self):
        '''
        Compute the autocorrelation2D_57 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_57 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_57(self.molecule)

    def __findAUTOCORR2D_58(self):
        '''
        Compute the autocorrelation2D_58 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_58 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_58(self.molecule)

    def __findAUTOCORR2D_59(self):
        '''
        Compute the autocorrelation2D_59 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_59 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_59(self.molecule)

    def __findAUTOCORR2D_60(self):
        '''
        Compute the autocorrelation2D_60 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_60 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_60(self.molecule)

    def __findAUTOCORR2D_61(self):
        '''
        Compute the autocorrelation2D_61 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_61 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_61(self.molecule)

    def __findAUTOCORR2D_62(self):
        '''
        Compute the autocorrelation2D_62 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_62 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_62(self.molecule)

    def __findAUTOCORR2D_63(self):
        '''
        Compute the autocorrelation2D_63 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_63 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_63(self.molecule)

    def __findAUTOCORR2D_64(self):
        '''
        Compute the autocorrelation2D_64 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_64 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_64(self.molecule)

    def __findAUTOCORR2D_65(self):
        '''
        Compute the autocorrelation2D_65 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_65 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_65(self.molecule)

    def __findAUTOCORR2D_66(self):
        '''
        Compute the autocorrelation2D_66 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_66 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_66(self.molecule)

    def __findAUTOCORR2D_67(self):
        '''
        Compute the autocorrelation2D_67 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_67 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_67(self.molecule)

    def __findAUTOCORR2D_68(self):
        '''
        Compute the autocorrelation2D_68 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_68 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_68(self.molecule)

    def __findAUTOCORR2D_69(self):
        '''
        Compute the autocorrelation2D_69 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_69 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_69(self.molecule)

    def __findAUTOCORR2D_70(self):
        '''
        Compute the autocorrelation2D_70 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_70 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_70(self.molecule)

    def __findAUTOCORR2D_71(self):
        '''
        Compute the autocorrelation2D_71 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_71 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_71(self.molecule)

    def __findAUTOCORR2D_72(self):
        '''
        Compute the autocorrelation2D_72 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_72 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_72(self.molecule)

    def __findAUTOCORR2D_73(self):
        '''
        Compute the autocorrelation2D_73 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_73 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_73(self.molecule)

    def __findAUTOCORR2D_74(self):
        '''
        Compute the autocorrelation2D_74 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_74 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_74(self.molecule)

    def __findAUTOCORR2D_75(self):
        '''
        Compute the autocorrelation2D_75 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_75 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_75(self.molecule)

    def __findAUTOCORR2D_76(self):
        '''
        Compute the autocorrelation2D_76 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_76 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_76(self.molecule)

    def __findAUTOCORR2D_77(self):
        '''
        Compute the autocorrelation2D_77 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_77 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_77(self.molecule)

    def __findAUTOCORR2D_78(self):
        '''
        Compute the autocorrelation2D_78 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_78 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_78(self.molecule)

    def __findAUTOCORR2D_79(self):
        '''
        Compute the autocorrelation2D_79 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_79 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_79(self.molecule)

    def __findAUTOCORR2D_80(self):
        '''
        Compute the autocorrelation2D_80 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_80 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_80(self.molecule)

    def __findAUTOCORR2D_81(self):
        '''
        Compute the autocorrelation2D_81 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_81 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_81(self.molecule)

    def __findAUTOCORR2D_82(self):
        '''
        Compute the autocorrelation2D_82 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_82 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_82(self.molecule)

    def __findAUTOCORR2D_83(self):
        '''
        Compute the autocorrelation2D_83 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_83 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_83(self.molecule)

    def __findAUTOCORR2D_84(self):
        '''
        Compute the autocorrelation2D_84 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_84 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_84(self.molecule)

    def __findAUTOCORR2D_85(self):
        '''
        Compute the autocorrelation2D_85 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_85 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_85(self.molecule)

    def __findAUTOCORR2D_86(self):
        '''
        Compute the autocorrelation2D_86 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_86 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_86(self.molecule)

    def __findAUTOCORR2D_87(self):
        '''
        Compute the autocorrelation2D_87 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_87 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_87(self.molecule)

    def __findAUTOCORR2D_88(self):
        '''
        Compute the autocorrelation2D_88 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_88 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_88(self.molecule)

    def __findAUTOCORR2D_89(self):
        '''
        Compute the autocorrelation2D_89 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_89 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_89(self.molecule)

    def __findAUTOCORR2D_90(self):
        '''
        Compute the autocorrelation2D_90 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_90 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_90(self.molecule)

    def __findAUTOCORR2D_91(self):
        '''
        Compute the autocorrelation2D_91 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_91 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_91(self.molecule)

    def __findAUTOCORR2D_92(self):
        '''
        Compute the autocorrelation2D_92 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_92 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_92(self.molecule)

    def __findAUTOCORR2D_93(self):
        '''
        Compute the autocorrelation2D_93 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_93 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_93(self.molecule)

    def __findAUTOCORR2D_94(self):
        '''
        Compute the autocorrelation2D_94 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_94 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_94(self.molecule)

    def __findAUTOCORR2D_95(self):
        '''
        Compute the autocorrelation2D_95 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_95 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_95(self.molecule)

    def __findAUTOCORR2D_96(self):
        '''
        Compute the autocorrelation2D_96 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_96 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_96(self.molecule)

    def __findAUTOCORR2D_97(self):
        '''
        Compute the autocorrelation2D_97 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_97 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_97(self.molecule)

    def __findAUTOCORR2D_98(self):
        '''
        Compute the autocorrelation2D_98 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_98 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_98(self.molecule)

    def __findAUTOCORR2D_99(self):
        '''
        Compute the autocorrelation2D_99 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_99 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_99(self.molecule)

    def __findAUTOCORR2D_100(self):
        '''
        Compute the autocorrelation2D_100 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_100 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_100(self.molecule)

    def __findAUTOCORR2D_101(self):
        '''
        Compute the autocorrelation2D_101 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_101 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_101(self.molecule)

    def __findAUTOCORR2D_102(self):
        '''
        Compute the autocorrelation2D_102 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_102 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_102(self.molecule)

    def __findAUTOCORR2D_103(self):
        '''
        Compute the autocorrelation2D_103 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_103 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_103(self.molecule)

    def __findAUTOCORR2D_104(self):
        '''
        Compute the autocorrelation2D_104 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_104 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_104(self.molecule)

    def __findAUTOCORR2D_105(self):
        '''
        Compute the autocorrelation2D_105 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_105 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_105(self.molecule)

    def __findAUTOCORR2D_106(self):
        '''
        Compute the autocorrelation2D_106 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_106 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_106(self.molecule)

    def __findAUTOCORR2D_107(self):
        '''
        Compute the autocorrelation2D_107 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_107 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_107(self.molecule)

    def __findAUTOCORR2D_108(self):
        '''
        Compute the autocorrelation2D_108 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_108 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_108(self.molecule)

    def __findAUTOCORR2D_109(self):
        '''
        Compute the autocorrelation2D_109 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_109 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_109(self.molecule)

    def __findAUTOCORR2D_110(self):
        '''
        Compute the autocorrelation2D_110 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_110 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_110(self.molecule)

    def __findAUTOCORR2D_111(self):
        '''
        Compute the autocorrelation2D_111 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_111 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_111(self.molecule)

    def __findAUTOCORR2D_112(self):
        '''
        Compute the autocorrelation2D_112 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_112 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_112(self.molecule)

    def __findAUTOCORR2D_113(self):
        '''
        Compute the autocorrelation2D_113 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_113 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_113(self.molecule)

    def __findAUTOCORR2D_114(self):
        '''
        Compute the autocorrelation2D_114 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_114 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_114(self.molecule)

    def __findAUTOCORR2D_115(self):
        '''
        Compute the autocorrelation2D_115 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_115 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_115(self.molecule)

    def __findAUTOCORR2D_116(self):
        '''
        Compute the autocorrelation2D_116 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_116 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_116(self.molecule)

    def __findAUTOCORR2D_117(self):
        '''
        Compute the autocorrelation2D_117 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_117 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_117(self.molecule)

    def __findAUTOCORR2D_118(self):
        '''
        Compute the autocorrelation2D_118 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_118 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_118(self.molecule)

    def __findAUTOCORR2D_119(self):
        '''
        Compute the autocorrelation2D_119 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_119 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_119(self.molecule)

    def __findAUTOCORR2D_120(self):
        '''
        Compute the autocorrelation2D_120 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_120 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_120(self.molecule)

    def __findAUTOCORR2D_121(self):
        '''
        Compute the autocorrelation2D_121 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_121 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_121(self.molecule)

    def __findAUTOCORR2D_122(self):
        '''
        Compute the autocorrelation2D_122 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_122 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_122(self.molecule)

    def __findAUTOCORR2D_123(self):
        '''
        Compute the autocorrelation2D_123 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_123 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_123(self.molecule)

    def __findAUTOCORR2D_124(self):
        '''
        Compute the autocorrelation2D_124 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_124 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_124(self.molecule)

    def __findAUTOCORR2D_125(self):
        '''
        Compute the autocorrelation2D_125 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_125 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_125(self.molecule)

    def __findAUTOCORR2D_126(self):
        '''
        Compute the autocorrelation2D_126 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_126 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_126(self.molecule)

    def __findAUTOCORR2D_127(self):
        '''
        Compute the autocorrelation2D_127 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_127 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_127(self.molecule)

    def __findAUTOCORR2D_128(self):
        '''
        Compute the autocorrelation2D_128 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_128 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_128(self.molecule)

    def __findAUTOCORR2D_129(self):
        '''
        Compute the autocorrelation2D_129 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_129 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_129(self.molecule)

    def __findAUTOCORR2D_130(self):
        '''
        Compute the autocorrelation2D_130 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_130 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_130(self.molecule)

    def __findAUTOCORR2D_131(self):
        '''
        Compute the autocorrelation2D_131 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_131 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_131(self.molecule)

    def __findAUTOCORR2D_132(self):
        '''
        Compute the autocorrelation2D_132 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_132 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_132(self.molecule)

    def __findAUTOCORR2D_133(self):
        '''
        Compute the autocorrelation2D_133 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_133 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_133(self.molecule)

    def __findAUTOCORR2D_134(self):
        '''
        Compute the autocorrelation2D_134 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_134 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_134(self.molecule)

    def __findAUTOCORR2D_135(self):
        '''
        Compute the autocorrelation2D_135 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_135 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_135(self.molecule)

    def __findAUTOCORR2D_136(self):
        '''
        Compute the autocorrelation2D_136 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_136 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_136(self.molecule)

    def __findAUTOCORR2D_137(self):
        '''
        Compute the autocorrelation2D_137 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_137 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_137(self.molecule)

    def __findAUTOCORR2D_138(self):
        '''
        Compute the autocorrelation2D_138 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_138 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_138(self.molecule)

    def __findAUTOCORR2D_139(self):
        '''
        Compute the autocorrelation2D_139 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_139 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_139(self.molecule)

    def __findAUTOCORR2D_140(self):
        '''
        Compute the autocorrelation2D_140 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_140 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_140(self.molecule)

    def __findAUTOCORR2D_141(self):
        '''
        Compute the autocorrelation2D_141 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_141 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_141(self.molecule)

    def __findAUTOCORR2D_142(self):
        '''
        Compute the autocorrelation2D_142 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_142 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_142(self.molecule)

    def __findAUTOCORR2D_143(self):
        '''
        Compute the autocorrelation2D_143 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_143 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_143(self.molecule)

    def __findAUTOCORR2D_144(self):
        '''
        Compute the autocorrelation2D_144 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_144 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_144(self.molecule)

    def __findAUTOCORR2D_145(self):
        '''
        Compute the autocorrelation2D_145 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_145 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_145(self.molecule)

    def __findAUTOCORR2D_146(self):
        '''
        Compute the autocorrelation2D_146 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_146 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_146(self.molecule)

    def __findAUTOCORR2D_147(self):
        '''
        Compute the autocorrelation2D_147 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_147 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_147(self.molecule)

    def __findAUTOCORR2D_148(self):
        '''
        Compute the autocorrelation2D_148 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_148 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_148(self.molecule)

    def __findAUTOCORR2D_149(self):
        '''
        Compute the autocorrelation2D_149 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_149 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_149(self.molecule)

    def __findAUTOCORR2D_150(self):
        '''
        Compute the autocorrelation2D_150 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_150 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_150(self.molecule)

    def __findAUTOCORR2D_151(self):
        '''
        Compute the autocorrelation2D_151 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_151 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_151(self.molecule)

    def __findAUTOCORR2D_152(self):
        '''
        Compute the autocorrelation2D_152 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_152 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_152(self.molecule)

    def __findAUTOCORR2D_153(self):
        '''
        Compute the autocorrelation2D_153 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_153 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_153(self.molecule)

    def __findAUTOCORR2D_154(self):
        '''
        Compute the autocorrelation2D_154 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_154 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_154(self.molecule)

    def __findAUTOCORR2D_155(self):
        '''
        Compute the autocorrelation2D_155 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_155 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_155(self.molecule)

    def __findAUTOCORR2D_156(self):
        '''
        Compute the autocorrelation2D_156 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_156 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_156(self.molecule)

    def __findAUTOCORR2D_157(self):
        '''
        Compute the autocorrelation2D_157 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_157 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_157(self.molecule)

    def __findAUTOCORR2D_158(self):
        '''
        Compute the autocorrelation2D_158 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_158 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_158(self.molecule)

    def __findAUTOCORR2D_159(self):
        '''
        Compute the autocorrelation2D_159 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_159 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_159(self.molecule)

    def __findAUTOCORR2D_160(self):
        '''
        Compute the autocorrelation2D_160 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_160 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_160(self.molecule)

    def __findAUTOCORR2D_161(self):
        '''
        Compute the autocorrelation2D_161 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_161 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_161(self.molecule)

    def __findAUTOCORR2D_162(self):
        '''
        Compute the autocorrelation2D_162 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_162 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_162(self.molecule)

    def __findAUTOCORR2D_163(self):
        '''
        Compute the autocorrelation2D_163 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_163 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_163(self.molecule)

    def __findAUTOCORR2D_164(self):
        '''
        Compute the autocorrelation2D_164 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_164 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_164(self.molecule)

    def __findAUTOCORR2D_165(self):
        '''
        Compute the autocorrelation2D_165 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_165 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_165(self.molecule)

    def __findAUTOCORR2D_166(self):
        '''
        Compute the autocorrelation2D_166 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_166 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_166(self.molecule)

    def __findAUTOCORR2D_167(self):
        '''
        Compute the autocorrelation2D_167 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_167 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_167(self.molecule)

    def __findAUTOCORR2D_168(self):
        '''
        Compute the autocorrelation2D_168 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_168 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_168(self.molecule)

    def __findAUTOCORR2D_169(self):
        '''
        Compute the autocorrelation2D_169 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_169 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_169(self.molecule)

    def __findAUTOCORR2D_170(self):
        '''
        Compute the autocorrelation2D_170 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_170 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_170(self.molecule)

    def __findAUTOCORR2D_171(self):
        '''
        Compute the autocorrelation2D_171 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_171 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_171(self.molecule)

    def __findAUTOCORR2D_172(self):
        '''
        Compute the autocorrelation2D_172 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_172 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_172(self.molecule)

    def __findAUTOCORR2D_173(self):
        '''
        Compute the autocorrelation2D_173 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_173 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_173(self.molecule)

    def __findAUTOCORR2D_174(self):
        '''
        Compute the autocorrelation2D_174 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_174 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_174(self.molecule)

    def __findAUTOCORR2D_175(self):
        '''
        Compute the autocorrelation2D_175 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_175 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_175(self.molecule)

    def __findAUTOCORR2D_176(self):
        '''
        Compute the autocorrelation2D_176 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_176 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_176(self.molecule)

    def __findAUTOCORR2D_177(self):
        '''
        Compute the autocorrelation2D_177 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_177 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_177(self.molecule)

    def __findAUTOCORR2D_178(self):
        '''
        Compute the autocorrelation2D_178 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_178 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_178(self.molecule)

    def __findAUTOCORR2D_179(self):
        '''
        Compute the autocorrelation2D_179 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_179 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_179(self.molecule)

    def __findAUTOCORR2D_180(self):
        '''
        Compute the autocorrelation2D_180 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_180 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_180(self.molecule)

    def __findAUTOCORR2D_181(self):
        '''
        Compute the autocorrelation2D_181 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_181 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_181(self.molecule)

    def __findAUTOCORR2D_182(self):
        '''
        Compute the autocorrelation2D_182 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_182 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_182(self.molecule)

    def __findAUTOCORR2D_183(self):
        '''
        Compute the autocorrelation2D_183 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_183 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_183(self.molecule)

    def __findAUTOCORR2D_184(self):
        '''
        Compute the autocorrelation2D_184 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_184 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_184(self.molecule)

    def __findAUTOCORR2D_185(self):
        '''
        Compute the autocorrelation2D_185 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_185 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_185(self.molecule)

    def __findAUTOCORR2D_186(self):
        '''
        Compute the autocorrelation2D_186 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_186 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_186(self.molecule)

    def __findAUTOCORR2D_187(self):
        '''
        Compute the autocorrelation2D_187 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_187 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_187(self.molecule)

    def __findAUTOCORR2D_188(self):
        '''
        Compute the autocorrelation2D_188 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_188 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_188(self.molecule)

    def __findAUTOCORR2D_189(self):
        '''
        Compute the autocorrelation2D_189 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_189 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_189(self.molecule)

    def __findAUTOCORR2D_190(self):
        '''
        Compute the autocorrelation2D_190 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_190 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_190(self.molecule)

    def __findAUTOCORR2D_191(self):
        '''
        Compute the autocorrelation2D_191 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_191 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_191(self.molecule)

    def __findAUTOCORR2D_192(self):
        '''
        Compute the autocorrelation2D_192 descriptor.
        Input:
          -
        Return:
          [float] - The autocorrelation2D_192 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_192(self.molecule)

    #endregion

    #region BCUT2D descriptors
    def __findBCUT2D_CHGHI(self):
        '''
        Compute the BCUT2D_CHGHI descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_CHGHI descriptor.
        '''
        return findBCUT2D_CHGHI(self.molecule)

    def __findBCUT2D_CHGLO(self):
        '''
        Compute the BCUT2D_CHGLO descriptor.
        Input:
          -
        Return:
          [float] - The the BCUT2D_CHGLO descriptor.
        '''
        return findBCUT2D_CHGLO(self.molecule)

    def __findBCUT2D_LOGPHI(self):
        '''
        Compute the BCUT2D_LOGPHI descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_LOGPHI descriptor.
        '''
        return findBCUT2D_LOGPHI(self.molecule)

    def __findBCUT2D_LOGPLOW(self):
        '''
        Compute the BCUT2D_LOGPLOW descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_LOGPLOW descriptor.
        '''
        return findBCUT2D_LOGPLOW(self.molecule)

    def __findBCUT2D_MRHI(self):
        '''
        Compute the BCUT2D_MRHI descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_MRHI descriptor.
        '''
        return findBCUT2D_MRHI(self.molecule)

    def __findBCUT2D_MRLOW(self):
        '''
        Compute the BCUT2D_MRLOW descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_MRLOW descriptor.
        '''
        return findBCUT2D_MRLOW(self.molecule)

    def __findBCUT2D_MWHI(self):
        '''
        Compute the BCUT2D_MWHI descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_MWHI descriptor.
        '''
        return findBCUT2D_MWHI(self.molecule)

    def __findBCUT2D_MWLOW(self):
        '''
        Compute the BCUT2D_MWLOW descriptor.
        Input:
          -
        Return:
          [float] - The BCUT2D_MWLOW descriptor.
        '''
        return findBCUT2D_MWLOW(self.molecule)

    #endregion

    def __findBalabanJ(self):
        '''
        Compute the BalabanJ descriptor.
        Input:
          -
        Return:
          [float] - The BalabanJ descriptor.
        '''
        return findBalabanJ(self.molecule)

    def __findBertzCT(self):
        '''
        Compute the BertzCT descriptor.
        Input:
          -
        Return:
          [float] - The BertzCT descriptor.
        '''
        return findBertzCT(self.molecule)

    #region Chi descriptors
    def __findChi0(self):
        '''
        Compute the Chi0 descriptor.
        Input:
          -
        Return:
          [float] - The Chi0 descriptor.
        '''
        return findChi0(self.molecule)

    def __findChi0n(self):
        '''
        Compute the Chi0n descriptor.
        Input:
          -
        Return:
          [float] - The Chi0n descriptor.
        '''
        return findChi0n(self.molecule)

    def __findChi0v(self):
        '''
        Compute the Chi0v descriptor.
        Input:
          -
        Return:
          [float] - The Chi0v descriptor.
        '''
        return findChi0v(self.molecule)

    def __findChi1(self):
        '''
        Compute the Chi1 descriptor.
        Input:
          -
        Return:
          [float] - The Chi1 descriptor.
        '''
        return findChi1(self.molecule)

    def __findChi1n(self):
        '''
        Compute the Chi1n descriptor.
        Input:
          -
        Return:
          [float] - The Chi1n descriptor.
        '''
        return findChi1n(self.molecule)

    def __findChi1v(self):
        '''
        Compute the Chi1v descriptor.
        Input:
          -
        Return:
          [float] - The Chi1v descriptor.
        '''
        return findChi1v(self.molecule)

    def __findChi2n(self):
        '''
        Compute the Chi2n descriptor.
        Input:
          -
        Return:
          [float] - The Chi2n descriptor.
        '''
        return findChi2n(self.molecule)

    def __findChi2v(self):
        '''
        Compute the Chi2v descriptor.
        Input:
          -
        Return:
          [float] - The Chi2v descriptor.
        '''
        return findChi2v(self.molecule)

    def __findChi3n(self):
        '''
        Compute the Chi3n descriptor.
        Input:
          -
        Return:
          [float] - The Chi3n descriptor.
        '''
        return findChi3n(self.molecule)

    def __findChi3v(self):
        '''
        Compute the Chi3v descriptor.
        Input:
          -
        Return:
          [float] - The Chi3v descriptor.
        '''
        return findChi3v(self.molecule)

    def __findChi4n(self):
        '''
        Compute the Chi4n descriptor.
        Input:
          -
        Return:
          [float] - The Chi4n descriptor.
        '''
        return findChi4n(self.molecule)

    def __findChi4v(self):
        '''
        Compute the Chi4v descriptor.
        Input:
          -
        Return:
          [float] - The Chi4v descriptor.
        '''
        return findChi4v(self.molecule)

    #endregion

    #region EState descriptors
    def __findEState_VSA1(self):
        '''
        Compute the EState_VSA1 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA1 descriptor.
        '''
        return findEState_VSA1(self.molecule)

    def __findEState_VSA2(self):
        '''
        Compute the EState_VSA2 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA2 descriptor.
        '''
        return findEState_VSA2(self.molecule)

    def __findEState_VSA3(self):
        '''
        Compute the EState_VSA3 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA3 descriptor.
        '''
        return findEState_VSA3(self.molecule)

    def __findEState_VSA4(self):
        '''
        Compute the EState_VSA4 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA4 descriptor.
        '''
        return findEState_VSA4(self.molecule)

    def __findEState_VSA5(self):
        '''
        Compute the EState_VSA5 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA5 descriptor.
        '''
        return findEState_VSA5(self.molecule)

    def __findEState_VSA6(self):
        '''
        Compute the EState_VSA6 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA6 descriptor.
        '''
        return findEState_VSA6(self.molecule)

    def __findEState_VSA7(self):
        '''
        Compute the EState_VSA7 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA7 descriptor.
        '''
        return findEState_VSA7(self.molecule)

    def __findEState_VSA8(self):
        '''
        Compute the EState_VSA8 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA8 descriptor.
        '''
        return findEState_VSA8(self.molecule)

    def __findEState_VSA9(self):
        '''
        Compute the EState_VSA9 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA9 descriptor.
        '''
        return findEState_VSA9(self.molecule)

    def __findEState_VSA10(self):
        '''
        Compute the EState_VSA10 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA10 descriptor.
        '''
        return findEState_VSA10(self.molecule)

    def __findEState_VSA11(self):
        '''
        Compute the EState_VSA11 descriptor.
        Input:
          -
        Return:
          [float] - The EState_VSA11 descriptor.
        '''
        return findEState_VSA11(self.molecule)

    def __findMaxAbsEStateIndex(self):
        '''
        Compute the MaxAbsEStateIndex descriptor.
        Input:
          -
        Return:
          [float] - The MaxAbsEStateIndex descriptor.
        '''
        return findMaxAbsEStateIndex(self.molecule)

    def __findMaxEStateIndex(self):
        '''
        Compute the MaxEStateIndex descriptor.
        Input:
          -
        Return:
          [float] - The MaxEStateIndex descriptor.
        '''
        return findMaxEStateIndex(self.molecule)

    def __findMinAbsEStateIndex(self):
        '''
        Compute the MinAbsEStateIndex descriptor.
        Input:
          -
        Return:
          [float] - The MinAbsEStateIndex descriptor.
        '''
        return findMinAbsEStateIndex(self.molecule)

    def __findMinEStateIndex(self):
        '''
        Compute the MinEStateIndex descriptor.
        Input:
          -
        Return:
          [float] - The MinEStateIndex descriptor.
        '''
        return findMinEStateIndex(self.molecule)

    #endregion

    def __findExactMolWt(self):
        '''
        Compute the exact molecular weight of the molecule.
        Input:
          -
        Return:
          [float] - The exact molecular weight.
        '''
        return findExactMolWt(self.molecule)

    def __findFpDensityMorgan1(self):
        '''
        Compute the Morgan fingerprint, radius 1 descriptor of the molecule.
        Input:
          -
        Return:
          [float] - The Morgan fingerprint, radius 1.
        '''
        return findFpDensityMorgan1(self.molecule)

    def __findFpDensityMorgan2(self):
        '''
        Compute the Morgan fingerprint, radius 2 descriptor of the molecule.
        Input:
          -
        Return:
          [float] - The Morgan fingerprint, radius 2.
        '''
        return findFpDensityMorgan2(self.molecule)

    def __findFpDensityMorgan3(self):
        '''
        Compute the Morgan fingerprint, radius 3 descriptor of the molecule.
        Input:
          -
        Return:
          [float] - The Morgan fingerprint, radius 3.
        '''
        return findFpDensityMorgan3(self.molecule)

    #region fr_ descriptors
    def __findfr_Al_COO(self):
        '''
        Compute the fr_Al_COO descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Al_COO value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Al_COO(self.molecule)

    def __findfr_Al_OH(self):
        '''
        Compute the fr_Al_OH descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Al_OH value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Al_OH(self.molecule)

    def __findfr_Al_OH_noTert(self):
        '''
        Compute the fr_Al_OH_noTert descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Al_OH_noTert value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Al_OH_noTert(self.molecule)

    def __findfr_ArN(self):
        '''
        Compute the fr_ArN descriptor.
        Input:
          -
        Return:
          [int]  - The fr_ArN value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_ArN(self.molecule)

    def __findfr_Ar_COO(self):
        '''
        Compute the fr_Ar_COO descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ar_COO value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ar_COO(self.molecule)

    def __findfr_Ar_N(self):
        '''
        Compute the fr_Ar_N descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ar_N value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ar_N(self.molecule)

    def __findfr_Ar_NH(self):
        '''
        Compute the fr_Ar_NH descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ar_NH value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ar_NH(self.molecule)

    def __findfr_Ar_OH(self):
        '''
        Compute the fr_Ar_OH descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ar_OH value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ar_OH(self.molecule)

    def __findfr_COO(self):
        '''
        Compute the fr_COO descriptor.
        Input:
          -
        Return:
          [int]  - The fr_COO value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_COO(self.molecule)

    def __findfr_COO2(self):
        '''
        Compute the fr_COO2 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_COO2 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_COO2(self.molecule)

    def __findfr_C_O(self):
        '''
        Compute the fr_C_O descriptor.
        Input:
          -
        Return:
          [int]  - The fr_C_O value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_C_O(self.molecule)

    def __findfr_C_O_noCOO(self):
        '''
        Compute the fr_C_O_noCOO descriptor.
        Input:
          -
        Return:
          [int]  - The fr_C_O_noCOO value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_C_O_noCOO(self.molecule)

    def __findfr_C_S(self):
        '''
        Compute the fr_C_S descriptor.
        Input:
          -
        Return:
          [int]  - The fr_C_S value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_C_S(self.molecule)

    def __findfr_HOCCN(self):
        '''
        Compute the fr_HOCCN descriptor.
        Input:
          -
        Return:
          [int]  - The fr_HOCCN value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_HOCCN(self.molecule)

    def __findfr_Imine(self):
        '''
        Compute the fr_Imine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Imine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Imine(self.molecule)

    def __findfr_NH0(self):
        '''
        Compute the fr_NH0 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_NH0 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_NH0(self.molecule)

    def __findfr_NH1(self):
        '''
        Compute the fr_NH1 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_NH1 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_NH1(self.molecule)

    def __findfr_NH2(self):
        '''
        Compute the fr_NH2 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_NH2 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_NH2(self.molecule)

    def __findfr_N_O(self):
        '''
        Compute the fr_N_O descriptor.
        Input:
          -
        Return:
          [int]  - The fr_N_O value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_N_O(self.molecule)

    def __findfr_Ndealkylation1(self):
        '''
        Compute the fr_Ndealkylation1 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ndealkylation1 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ndealkylation1(self.molecule)

    def __findfr_Ndealkylation2(self):
        '''
        Compute the fr_Ndealkylation2 descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Ndealkylation2 value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Ndealkylation2(self.molecule)

    def __findfr_Nhpyrrole(self):
        '''
        Compute the fr_Nhpyrrole descriptor.
        Input:
          -
        Return:
          [int]  - The fr_Nhpyrrole value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_Nhpyrrole(self.molecule)

    def __findfr_SH(self):
        '''
        Compute the fr_SH descriptor.
        Input:
          -
        Return:
          [int]  - The fr_SH value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_SH(self.molecule)

    def __findfr_aldehyde(self):
        '''
        Compute the fr_aldehyde descriptor.
        Input:
          -
        Return:
          [int]  - The fr_aldehyde value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_aldehyde(self.molecule)

    def __findfr_alkyl_carbamate(self):
        '''
        Compute the fr_alkyl_carbamate descriptor.
        Input:
          -
        Return:
          [int]  - The fr_alkyl_carbamate value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_alkyl_carbamate(self.molecule)

    def __findfr_alkyl_halide(self):
        '''
        Compute the fr_alkyl_halide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_alkyl_halide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_alkyl_halide(self.molecule)

    def __findfr_allylic_oxid(self):
        '''
        Compute the fr_allylic_oxid descriptor.
        Input:
          -
        Return:
          [int]  - The fr_allylic_oxid value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_allylic_oxid(self.molecule)

    def __findfr_amide(self):
        '''
        Compute the fr_amide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_amide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_amide(self.molecule)

    def __findfr_amidine(self):
        '''
        Compute the fr_amidine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_amidine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_amidine(self.molecule)

    def __findfr_aniline(self):
        '''
        Compute the fr_aniline descriptor.
        Input:
          -
        Return:
          [int]  - The fr_aniline value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_aniline(self.molecule)

    def __findfr_aryl_methyl(self):
        '''
        Compute the fr_aryl_methyl descriptor.
        Input:
          -
        Return:
          [int]  - The fr_aryl_methyl value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_aryl_methyl(self.molecule)

    def __findfr_azide(self):
        '''
        Compute the fr_azide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_azide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_azide(self.molecule)

    def __findfr_azo(self):
        '''
        Compute the fr_azo descriptor.
        Input:
          -
        Return:
          [int]  - The fr_azo value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_azo(self.molecule)

    def __findfr_barbitur(self):
        '''
        Compute the fr_barbitur descriptor.
        Input:
          -
        Return:
          [int]  - The fr_barbitur value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_barbitur(self.molecule)

    def __findfr_benzene(self):
        '''
        Compute the fr_benzene descriptor.
        Input:
          -
        Return:
          [int]  - The fr_benzene value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_benzene(self.molecule)

    def __findfr_benzodiazepine(self):
        '''
        Compute the fr_benzodiazepine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_benzodiazepine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_benzodiazepine(self.molecule)

    def __findfr_bicyclic(self):
        '''
        Compute the fr_bicyclic descriptor.
        Input:
          -
        Return:
          [int]  - The fr_bicyclic value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_bicyclic(self.molecule)

    def __findfr_diazo(self):
        '''
        Compute the fr_diazo descriptor.
        Input:
          -
        Return:
          [int]  - The fr_diazo value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_diazo(self.molecule)

    def __findfr_dihydropyridine(self):
        '''
        Compute the fr_dihydropyridine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_dihydropyridine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_dihydropyridine(self.molecule)

    def __findfr_epoxide(self):
        '''
        Compute the fr_epoxide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_epoxide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_epoxide(self.molecule)

    def __findfr_ester(self):
        '''
        Compute the fr_ester descriptor.
        Input:
          -
        Return:
          [int]  - The fr_ester value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_ester(self.molecule)

    def __findfr_ether(self):
        '''
        Compute the fr_ether descriptor.
        Input:
          -
        Return:
          [int]  - The fr_ether value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_ether(self.molecule)

    def __findfr_furan(self):
        '''
        Compute the fr_furan descriptor.
        Input:
          -
        Return:
          [int]  - The fr_furan value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_furan(self.molecule)

    def __findfr_guanido(self):
        '''
        Compute the fr_guanido descriptor.
        Input:
          -
        Return:
          [int]  - The fr_guanido value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_guanido(self.molecule)

    def __findfr_halogen(self):
        '''
        Compute the fr_halogen descriptor.
        Input:
          -
        Return:
          [int]  - The fr_halogen value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_halogen(self.molecule)

    def __findfr_hdrzine(self):
        '''
        Compute the fr_hdrzine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_hdrzine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_hdrzine(self.molecule)

    def __findfr_hdrzone(self):
        '''
        Compute the fr_hdrzone descriptor.
        Input:
          -
        Return:
          [int]  - The fr_hdrzone value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_hdrzone(self.molecule)

    def __findfr_imidazole(self):
        '''
        Compute the fr_imidazole descriptor.
        Input:
          -
        Return:
          [int]  - The fr_imidazole value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_imidazole(self.molecule)

    def __findfr_imide(self):
        '''
        Compute the fr_imide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_imide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_imide(self.molecule)

    def __findfr_isocyan(self):
        '''
        Compute the fr_isocyan descriptor.
        Input:
          -
        Return:
          [int]  - The fr_isocyan value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_isocyan(self.molecule)

    def __findfr_isothiocyan(self):
        '''
        Compute the fr_isothiocyan descriptor.
        Input:
          -
        Return:
          [int]  - The fr_isothiocyan value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_isothiocyan(self.molecule)

    def __findfr_ketone(self):
        '''
        Compute the fr_ketone descriptor.
        Input:
          -
        Return:
          [int]  - The fr_ketone value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_ketone(self.molecule)

    def __findfr_ketone_Topliss(self):
        '''
        Compute the fr_ketone_Topliss descriptor.
        Input:
          -
        Return:
          [int]  - The fr_ketone_Topliss value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_ketone_Topliss(self.molecule)

    def __findfr_lactam(self):
        '''
        Compute the fr_lactam descriptor.
        Input:
          -
        Return:
          [int]  - The fr_lactam value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_lactam(self.molecule)

    def __findfr_lactone(self):
        '''
        Compute the fr_lactone descriptor.
        Input:
          -
        Return:
          [int]  - The fr_lactone value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_lactone(self.molecule)

    def __findfr_methoxy(self):
        '''
        Compute the fr_methoxy descriptor.
        Input:
          -
        Return:
          [int]  - The fr_methoxy value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_methoxy(self.molecule)

    def __findfr_morpholine(self):
        '''
        Compute the fr_morpholine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_morpholine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_morpholine(self.molecule)

    def __findfr_nitrile(self):
        '''
        Compute the fr_nitrile descriptor.
        Input:
          -
        Return:
          [int]  - The fr_nitrile value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_nitrile(self.molecule)

    def __findfr_nitro(self):
        '''
        Compute the fr_nitro descriptor.
        Input:
          -
        Return:
          [int]  - The fr_nitro value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_nitro(self.molecule)

    def __findfr_nitro_arom(self):
        '''
        Compute the fr_nitro_arom descriptor.
        Input:
          -
        Return:
          [int]  - The fr_nitro_arom value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_nitro_arom(self.molecule)

    def __findfr_nitro_arom_nonortho(self):
        '''
        Compute the fr_nitro_arom_nonortho descriptor.
        Input:
          -
        Return:
          [int]  - The fr_nitro_arom_nonortho value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_nitro_arom_nonortho(self.molecule)

    def __findfr_nitroso(self):
        '''
        Compute the fr_nitroso descriptor.
        Input:
          -
        Return:
          [int]  - The fr_nitroso value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_nitroso(self.molecule)

    def __findfr_oxazole(self):
        '''
        Compute the fr_oxazole descriptor.
        Input:
          -
        Return:
          [int]  - The fr_oxazole value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_oxazole(self.molecule)

    def __findfr_oxime(self):
        '''
        Compute the fr_oxime descriptor.
        Input:
          -
        Return:
          [int]  - The fr_oxime value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_oxime(self.molecule)

    def __findfr_para_hydroxylation(self):
        '''
        Compute the fr_para_hydroxylation descriptor.
        Input:
          -
        Return:
          [int]  - The fr_para_hydroxylation value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_para_hydroxylation(self.molecule)

    def __findfr_phenol(self):
        '''
        Compute the fr_phenol descriptor.
        Input:
          -
        Return:
          [int]  - The fr_phenol value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_phenol(self.molecule)

    def __findfr_phenol_noOrthoHbond(self):
        '''
        Compute the fr_phenol_noOrthoHbond descriptor.
        Input:
          -
        Return:
          [int]  - The fr_phenol_noOrthoHbond value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_phenol_noOrthoHbond(self.molecule)

    def __findfr_phos_acid(self):
        '''
        Compute the fr_phos_acid descriptor.
        Input:
          -
        Return:
          [int]  - The fr_phos_acid value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_phos_acid(self.molecule)

    def __findfr_phos_ester(self):
        '''
        Compute the fr_phos_ester descriptor.
        Input:
          -
        Return:
          [int]  - The fr_phos_ester value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_phos_ester(self.molecule)

    def __findfr_piperdine(self):
        '''
        Compute the fr_piperdine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_piperdine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_piperdine(self.molecule)

    def __findfr_piperzine(self):
        '''
        Compute the fr_piperzine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_piperzine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_piperzine(self.molecule)

    def __findfr_priamide(self):
        '''
        Compute the fr_priamide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_priamide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_priamide(self.molecule)

    def __findfr_prisulfonamd(self):
        '''
        Compute the fr_prisulfonamd descriptor.
        Input:
          -
        Return:
          [int]  - The fr_prisulfonamd value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_prisulfonamd(self.molecule)

    def __findfr_pyridine(self):
        '''
        Compute the fr_pyridine descriptor.
        Input:
          -
        Return:
          [int]  - The fr_pyridine value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_pyridine(self.molecule)

    def __findfr_quatN(self):
        '''
        Compute the fr_quatN descriptor.
        Input:
          -
        Return:
          [int]  - The fr_quatN value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_quatN(self.molecule)

    def __findfr_sulfide(self):
        '''
        Compute the fr_sulfide descriptor.
        Input:
          -
        Return:
          [int]  - The fr_sulfide value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_sulfide(self.molecule)

    def __findfr_sulfonamd(self):
        '''
        Compute the fr_sulfonamd descriptor.
        Input:
          -
        Return:
          [int]  - The fr_sulfonamd value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_sulfonamd(self.molecule)

    def __findfr_sulfone(self):
        '''
        Compute the fr_sulfone descriptor.
        Input:
          -
        Return:
          [int]  - The fr_sulfone value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_sulfone(self.molecule)

    def __findfr_term_acetylene(self):
        '''
        Compute the fr_term_acetylene descriptor.
        Input:
          -
        Return:
          [int]  - The fr_term_acetylene value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_term_acetylene(self.molecule)

    def __findfr_tetrazole(self):
        '''
        Compute the fr_tetrazole descriptor.
        Input:
          -
        Return:
          [int]  - The fr_tetrazole value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_tetrazole(self.molecule)

    def __findfr_thiazole(self):
        '''
        Compute the fr_thiazole descriptor.
        Input:
          -
        Return:
          [int]  - The fr_thiazole value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_thiazole(self.molecule)

    def __findfr_thiocyan(self):
        '''
        Compute the fr_thiocyan descriptor.
        Input:
          -
        Return:
          [int]  - The fr_thiocyan value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_thiocyan(self.molecule)

    def __findfr_thiophene(self):
        '''
        Compute the fr_thiophene descriptor.
        Input:
          -
        Return:
          [int]  - The fr_thiophene value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_thiophene(self.molecule)

    def __findfr_unbrch_alkane(self):
        '''
        Compute the fr_unbrch_alkane descriptor.
        Input:
          -
        Return:
          [int]  - The fr_unbrch_alkane value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_unbrch_alkane(self.molecule)

    def __findfr_urea(self):
        '''
        Compute the fr_urea descriptor.
        Input:
          -
        Return:
          [int]  - The fr_urea value.
          [None] - If parsing the descriptor fails.
        '''
        return findfr_urea(self.molecule)

    #endregion

    def __findFractionCSP3(self):
        '''
        Compute the FractionCSP3 descriptor.
        Input:
          -
        Return:
          [float] - The FractionCSP3 descriptor.
        '''
        return findFractionCSP3(self.molecule)

    def __findHallKierAlpha(self):
        '''
        Compute the HallKierAlpha descriptor.
        Input:
          -
        Return:
          [float] - The HallKierAlpha descriptor.
        '''
        return findHallKierAlpha(self.molecule)

    def __findHeavyAtomMolWt(self):
        '''
        Compute the heavy atom molecular weight of the molecule.
        Input:
          -
        Return:
          [float] - The heavy atom molecular weight.
        '''
        return findHeavyAtomMolWt(self.molecule)

    def __findHeavyAtomCount(self):
        '''
        Compute the HeavyAtomCount descriptor.
        Input:
          -
        Return:
          [int] - The HeavyAtomCount descriptor.
        '''
        return findHeavyAtomCount(self.molecule)

    def __findIpc(self):
        '''
        Compute the Ipc descriptor.
        Input:
          -
        Return:
          [float] - The Ipc descriptor.
        '''
        return findIpc(self.molecule)

    #region Kappa descriptors
    def __findKappa1(self):
        '''
        Compute the Kappa1 descriptor.
        Input:
          -
        Return:
          [float] - The Kappa1 descriptor.
        '''
        return findKappa1(self.molecule)

    def __findKappa2(self):
        '''
        Compute the Kappa2 descriptor.
        Input:
          -
        Return:
          [float] - The Kappa2 descriptor.
        '''
        return findKappa2(self.molecule)

    def __findKappa3(self):
        '''
        Compute the Kappa3 descriptor.
        Input:
          -
        Return:
          [float] - The Kappa3 descriptor.
        '''
        return findKappa3(self.molecule)

    #endregion

    def __findLabuteASA(self):
        '''
        Compute the LabuteASA descriptor.
        Input:
          -
        Return:
          [float] - The LabuteASA descriptor.
        '''
        return findLabuteASA(self.molecule)

    def __findMaxAbsPartialCharge(self):
        '''
        Compute the maximum absolute partial charge of the molecule.
        Input:
          -
        Return:
          [float] - The maximum absolute partial charge.
        '''
        return findMaxAbsPartialCharge(self.molecule)

    def __findMaxPartialCharge(self):
        '''
        Compute the absolute partial charge of the molecule.
        Input:
          -
        Return:
          [float] - The absolute partial partial charge.
        '''
        return findMaxPartialCharge(self.molecule)

    def __findMinAbsPartialCharge(self):
        '''
        Compute the minimum absolute partial charge of the molecule.
        Input:
          -
        Return:
          [float] - The minimum absolute partial partial charge.
        '''
        return findMinAbsPartialCharge(self.molecule)

    def __findMinPartialCharge(self):
        '''
        Compute the minimum partial charge of the molecule.
        Input:
          -
        Return:
          [float] - The minimum partial partial charge.
        '''
        return findMinPartialCharge(self.molecule)

    def __findMolLogP(self):
        '''
        Compute the MolLogP descriptor.
        Input:
          -
        Return:
          [float] - The MolLogP descriptor.
        '''
        return findMolLogP(self.molecule)

    def __findMolMR(self):
        '''
        Compute the MolMR descriptor.
        Input:
          -
        Return:
          [float] - The MolMR descriptor.
        '''
        return findMolMR(self.molecule)

    def __findMolWt(self):
        '''
        Compute the molecular weight of the molecule.
        Input:
          -
        Return:
          [float] - The molecular weight.
        '''
        return findMolWt(self.molecule)

    #region 'count' descriptors
    def __findNHOHCount(self):
        '''
        Compute the NHOHCount descriptor.
        Input:
          -
        Return:
          [int] - The NHOHCount descriptor.
        '''
        return findNHOHCount(self.molecule)

    def __findNOCount(self):
        '''
        Compute the NOCount descriptor.
        Input:
          -
        Return:
          [int] - The NOCount descriptor.
        '''
        return findNOCount(self.molecule)

    def __findNumAliphaticCarbocycles(self):
        '''
        Compute the NumAliphaticCarbocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumAliphaticCarbocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAliphaticCarbocycles(self.molecule)

    def __findNumAliphaticHeterocycles(self):
        '''
        Compute the NumAliphaticHeterocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumAliphaticHeterocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAliphaticHeterocycles(self.molecule)

    def __findNumAliphaticRings(self):
        '''
        Compute the NumAliphaticRings descriptor.
        Input:
          -
        Return:
          [int]  - The NumAliphaticRings value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAliphaticRings(self.molecule)

    def __findNumAromaticCarbocycles(self):
        '''
        Compute the NumAromaticCarbocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumAromaticCarbocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAromaticCarbocycles(self.molecule)

    def __findNumAromaticHeterocycles(self):
        '''
        Compute the NumAromaticHeterocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumAromaticHeterocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAromaticHeterocycles(self.molecule)

    def __findNumAromaticRings(self):
        '''
        Compute the NumAromaticRings descriptor.
        Input:
          -
        Return:
          [int]  - The NumAromaticRings value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumAromaticRings(self.molecule)

    def __findNumHAcceptors(self):
        '''
        Compute the NumHAcceptors descriptor.
        Input:
          -
        Return:
          [int]  - The NumHAcceptors value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumHAcceptors(self.molecule)

    def __findNumHDonors(self):
        '''
        Compute the NumHDonors descriptor.
        Input:
          -
        Return:
          [int]  - The NumHDonors value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumHDonors(self.molecule)

    def __findNumHeteroatoms(self):
        '''
        Compute the NumHeteroatoms descriptor.
        Input:
          -
        Return:
          [int]  - The NumHeteroatoms value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumHeteroatoms(self.molecule)

    def __findNumRotatableBonds(self):
        '''
        Compute the NumRotatableBonds descriptor.
        Input:
          -
        Return:
          [int]  - The NumRotatableBonds value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumRotatableBonds(self.molecule)

    def __findNumSaturatedCarbocycles(self):
        '''
        Compute the NumSaturatedCarbocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumSaturatedCarbocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumSaturatedCarbocycles(self.molecule)

    def __findNumSaturatedHeterocycles(self):
        '''
        Compute the NumSaturatedHeterocycles descriptor.
        Input:
          -
        Return:
          [int]  - The NumSaturatedHeterocycles value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumSaturatedHeterocycles(self.molecule)

    def __findNumSaturatedRings(self):
        '''
        Compute the NumSaturatedRings descriptor.
        Input:
          -
        Return:
          [int]  - The NumSaturatedRings value.
          [None] - If parsing the descriptor fails.
        '''
        return findNumSaturatedRings(self.molecule)

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

    def __findRingCount(self):
        '''
        Compute the RingCount descriptor.
        Input:
          -
        Return:
          [float] - The RingCount value.
          [None]   - If parsing the descriptor fails.
        '''
        return findRingCount(self.molecule)

    #endregion

    #region PEOE_VSA descriptors
    def __findPEOE_VSA1(self):
        '''
        Compute the PEOE_VSA1 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA1(self.molecule)

    def __findPEOE_VSA2(self):
        '''
        Compute the PEOE_VSA2 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA2(self.molecule)

    def __findPEOE_VSA3(self):
        '''
        Compute the PEOE_VSA3 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA3(self.molecule)

    def __findPEOE_VSA4(self):
        '''
        Compute the PEOE_VSA4 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA4(self.molecule)

    def __findPEOE_VSA5(self):
        '''
        Compute the PEOE_VSA5 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA5(self.molecule)

    def __findPEOE_VSA6(self):
        '''
        Compute the PEOE_VSA6 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA6(self.molecule)

    def __findPEOE_VSA7(self):
        '''
        Compute the PEOE_VSA7 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA7(self.molecule)

    def __findPEOE_VSA8(self):
        '''
        Compute the PEOE_VSA8 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA8(self.molecule)

    def __findPEOE_VSA9(self):
        '''
        Compute the PEOE_VSA9 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA9(self.molecule)

    def __findPEOE_VSA10(self):
        '''
        Compute the PEOE_VSA10 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA10(self.molecule)

    def __findPEOE_VSA11(self):
        '''
        Compute the PEOE_VSA11 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA11 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA11(self.molecule)

    def __findPEOE_VSA12(self):
        '''
        Compute the PEOE_VSA12 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA12 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA12(self.molecule)

    def __findPEOE_VSA13(self):
        '''
        Compute the PEOE_VSA13 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA13 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA13(self.molecule)

    def __findPEOE_VSA14(self):
        '''
        Compute the PEOE_VSA14 descriptor.
        Input:
          -
        Return:
          [float] - The PEOE_VSA14 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPEOE_VSA14(self.molecule)

    #endregion

    def __findqed(self):
        '''
        Compute the qed descriptor.
        Input:
          -
        Return:
          [float] - The qed value.
          [None]   - If parsing the descriptor fails.
        '''
        return findqed(self.molecule)

    #region SMR_VSA1 descriptors

    def __findSMR_VSA1(self):
        '''
        Compute the SMR_VSA1 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA1(self.molecule)

    def __findSMR_VSA2(self):
        '''
        Compute the SMR_VSA2 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA2(self.molecule)

    def __findSMR_VSA3(self):
        '''
        Compute the SMR_VSA3 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA3(self.molecule)

    def __findSMR_VSA4(self):
        '''
        Compute the SMR_VSA4 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA4(self.molecule)

    def __findSMR_VSA5(self):
        '''
        Compute the SMR_VSA5 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA5(self.molecule)

    def __findSMR_VSA6(self):
        '''
        Compute the SMR_VSA6 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA6(self.molecule)

    def __findSMR_VSA7(self):
        '''
        Compute the SMR_VSA7 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA7(self.molecule)

    def __findSMR_VSA8(self):
        '''
        Compute the SMR_VSA8 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA8(self.molecule)

    def __findSMR_VSA9(self):
        '''
        Compute the SMR_VSA9 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA9(self.molecule)

    def __findSMR_VSA10(self):
        '''
        Compute the SMR_VSA10 descriptor.
        Input:
          -
        Return:
          [float] - The SMR_VSA10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSMR_VSA10(self.molecule)

    #endregion

    #region SlogP_VSA1 descriptors
    def __findSlogP_VSA1(self):
        '''
        Compute the SlogP_VSA1 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA1(self.molecule)

    def __findSlogP_VSA2(self):
        '''
        Compute the SlogP_VSA2 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA2(self.molecule)

    def __findSlogP_VSA3(self):
        '''
        Compute the SlogP_VSA3 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA3(self.molecule)

    def __findSlogP_VSA4(self):
        '''
        Compute the SlogP_VSA4 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA4(self.molecule)

    def __findSlogP_VSA5(self):
        '''
        Compute the SlogP_VSA5 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA5(self.molecule)

    def __findSlogP_VSA6(self):
        '''
        Compute the SlogP_VSA6 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA6(self.molecule)

    def __findSlogP_VSA7(self):
        '''
        Compute the SlogP_VSA7 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA7(self.molecule)

    def __findSlogP_VSA8(self):
        '''
        Compute the SlogP_VSA8 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA8(self.molecule)

    def __findSlogP_VSA9(self):
        '''
        Compute the SlogP_VSA9 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA9(self.molecule)

    def __findSlogP_VSA10(self):
        '''
        Compute the SlogP_VSA10 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA10(self.molecule)

    def __findSlogP_VSA11(self):
        '''
        Compute the SlogP_VSA11 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA11 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA11(self.molecule)

    def __findSlogP_VSA12(self):
        '''
        Compute the SlogP_VSA12 descriptor.
        Input:
          -
        Return:
          [float] - The SlogP_VSA12 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSlogP_VSA12(self.molecule)

    #endregion

    #region 3D descriptors
    def __findAUTOCORR3D(self):
        '''
        Compute the AUTOCORR3D descriptors.
        Input:
          -
        Return:
          [float] - The AUTOCORR3D_1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR3D(self.molecule)

    def __findAsphericity(self):
        '''
        Compute the Asphericity descriptor.
        Input:
          -
        Return:
          [float] - The Asphericity value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAsphericity(self.molecule)

    def __findEccentricity(self):
        '''
        Compute the Eccentricity descriptor.
        Input:
          -
        Return:
          [float] - The Eccentricity value.
          [None]   - If parsing the descriptor fails.
        '''
        return findEccentricity(self.molecule)

    def __findInertialShapeFactor(self):
        '''
        Compute the InertialShapeFactor descriptor.
        Input:
          -
        Return:
          [float] - The InertialShapeFactor value.
          [None]   - If parsing the descriptor fails.
        '''
        return findInertialShapeFactor(self.molecule)

    def __findNPR1(self):
        '''
        Compute the NPR1 descriptor.
        Input:
          -
        Return:
          [float] - The NPR1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findNPR1(self.molecule)

    def __findNPR2(self):
        '''
        Compute the NPR2 descriptor.
        Input:
          -
        Return:
          [float] - The NPR2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findNPR2(self.molecule)

    def __findPMI1(self):
        '''
        Compute the PMI1 descriptor.
        Input:
          -
        Return:
          [float] - The PMI1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPMI1(self.molecule)

    def __findPMI2(self):
        '''
        Compute the PMI2 descriptor.
        Input:
          -
        Return:
          [float] - The PMI2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPMI2(self.molecule)

    def __findPMI3(self):
        '''
        Compute the PMI3 descriptor.
        Input:
          -
        Return:
          [float] - The PMI3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findPMI3(self.molecule)

    def __findRadiusOfGyration(self):
        '''
        Compute the RadiusOfGyration descriptor.
        Input:
          -
        Return:
          [float] - The RadiusOfGyration value.
          [None]   - If parsing the descriptor fails.
        '''
        return findRadiusOfGyration(self.molecule)

    def __findSpherocityIndex(self):
        '''
        Compute the SpherocityIndex descriptor.
        Input:
          -
        Return:
          [float] - The SpherocityIndex value.
          [None]   - If parsing the descriptor fails.
        '''
        return findSpherocityIndex(self.molecule)

    #endregion

    def __findTPSA(self):
        '''
        Compute the TPSA descriptor.
        Input:
          -
        Return:
          [float] - The TPSA value.
          [None]   - If parsing the descriptor fails.
        '''
        return findTPSA(self.molecule)

    #region VSA_EState descriptors
    def __findVSA_EState1(self):
        '''
        Compute the VSA_EState1 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState1(self.molecule)

    def __findVSA_EState2(self):
        '''
        Compute the VSA_EState2 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState2(self.molecule)

    def __findVSA_EState3(self):
        '''
        Compute the VSA_EState3 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState3(self.molecule)

    def __findVSA_EState4(self):
        '''
        Compute the VSA_EState4 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState4(self.molecule)

    def __findVSA_EState5(self):
        '''
        Compute the VSA_EState5 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState5(self.molecule)

    def __findVSA_EState6(self):
        '''
        Compute the VSA_EState6 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState6(self.molecule)

    def __findVSA_EState7(self):
        '''
        Compute the VSA_EState7 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState7(self.molecule)

    def __findVSA_EState8(self):
        '''
        Compute the VSA_EState8 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState8(self.molecule)

    def __findVSA_EState9(self):
        '''
        Compute the VSA_EState9 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState9(self.molecule)

    def __findVSA_EState10(self):
        '''
        Compute the VSA_EState10 descriptor.
        Input:
          -
        Return:
          [float] - The VSA_EState10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findVSA_EState10(self.molecule)

    #endregion

    ## Public ##
    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
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

    def get_descriptors(self):
        '''
        Return the descriptors for the Ligand object.
        Input:
          -
        Return:
          [dict] - Dictionary of descriptors for the recpetor.
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
        return descriptors

    def to_dict(self):
        '''
        Return all the properties for the Ligand object.
        Input:
          -
        Return:
          -
        '''
        # Create new dict
        properties = dict()
        # Set Name, Path and molecule
        properties["Name"] = self.name if self.name is not None else "-"
        properties["Path"] = self.path if self.path is not None else "-"
        properties["Molecule"] = self.molecule if self.molecule is not None else "-"
        # Combine both in one dict and return them
        return {**properties, **self.get_descriptors()}

    def to_json(self, overwrite = False):
        '''
        Stores the descriptors as json to avoid the necessity of evaluate them many times.
        Input:
          overwrite [bool] DEFAULT: False - Flag to allow overwriting the target file.
        Return:
          [int]
          See Error.py for all return codes.
        '''
        try:
            outputJson = f"{os.path.dirname(self.path)}/{self.name}_descriptors.json"
            if not overwrite and os.path.isfile(outputJson):
                return errors.file_exists(f"The file {outputJson} already exists and the overwrite flag is set to False, no file will be generated or overwrited.", "warn")
            if os.path.isfile(outputJson):
                _ = errors.file_exists(f"The file '{outputJson}' already exists. It will be OVERWRITED!!!")
            try:
                with open(outputJson, "w") as outfile:
                    json.dump(self.__safe_to_dict(), outfile)
                return errors.ok()
            except Exception as e:
                return errors.write_file(f"Problems while writing the file '{outputJson}' Error: {e}.")
        except Exception as e:
            return errors.unknown(f"Unknown error while converting the ligand {self.name} to json.\nError: {e}", "error")

    def is_valid(self):
        '''
        Check if a Ligand object is valid.
        Input:
          -
        Return:
          [bool]
            True  - if valid
            False - if not valid
        '''
        #region if any attribute is None
        if self.name is None or self.path is None or self.AUTOCORR2D_1 is None or self.AUTOCORR2D_2 is None or self.AUTOCORR2D_3 is None or self.AUTOCORR2D_4 is None or self.AUTOCORR2D_5 is None or self.AUTOCORR2D_6 is None or self.AUTOCORR2D_7 is None or self.AUTOCORR2D_8 is None or self.AUTOCORR2D_9 is None or self.AUTOCORR2D_10 is None or self.AUTOCORR2D_11 is None or self.AUTOCORR2D_12 is None or self.AUTOCORR2D_13 is None or self.AUTOCORR2D_14 is None or self.AUTOCORR2D_15 is None or self.AUTOCORR2D_16 is None or self.AUTOCORR2D_17 is None or self.AUTOCORR2D_18 is None or self.AUTOCORR2D_19 is None or self.AUTOCORR2D_20 is None or self.AUTOCORR2D_21 is None or self.AUTOCORR2D_22 is None or self.AUTOCORR2D_23 is None or self.AUTOCORR2D_24 is None or self.AUTOCORR2D_25 is None or self.AUTOCORR2D_26 is None or self.AUTOCORR2D_27 is None or self.AUTOCORR2D_28 is None or self.AUTOCORR2D_29 is None or self.AUTOCORR2D_30 is None or self.AUTOCORR2D_31 is None or self.AUTOCORR2D_32 is None or self.AUTOCORR2D_33 is None or self.AUTOCORR2D_34 is None or self.AUTOCORR2D_35 is None or self.AUTOCORR2D_36 is None or self.AUTOCORR2D_37 is None or self.AUTOCORR2D_38 is None or self.AUTOCORR2D_39 is None or self.AUTOCORR2D_40 is None or self.AUTOCORR2D_41 is None or self.AUTOCORR2D_42 is None or self.AUTOCORR2D_43 is None or self.AUTOCORR2D_44 is None or self.AUTOCORR2D_45 is None or self.AUTOCORR2D_46 is None or self.AUTOCORR2D_47 is None or self.AUTOCORR2D_48 is None or self.AUTOCORR2D_49 is None or self.AUTOCORR2D_50 is None or self.AUTOCORR2D_51 is None or self.AUTOCORR2D_52 is None or self.AUTOCORR2D_53 is None or self.AUTOCORR2D_54 is None or self.AUTOCORR2D_55 is None or self.AUTOCORR2D_56 is None or self.AUTOCORR2D_57 is None or self.AUTOCORR2D_58 is None or self.AUTOCORR2D_59 is None or self.AUTOCORR2D_60 is None or self.AUTOCORR2D_61 is None or self.AUTOCORR2D_62 is None or self.AUTOCORR2D_63 is None or self.AUTOCORR2D_64 is None or self.AUTOCORR2D_65 is None or self.AUTOCORR2D_66 is None or self.AUTOCORR2D_67 is None or self.AUTOCORR2D_68 is None or self.AUTOCORR2D_69 is None or self.AUTOCORR2D_70 is None or self.AUTOCORR2D_71 is None or self.AUTOCORR2D_72 is None or self.AUTOCORR2D_73 is None or self.AUTOCORR2D_74 is None or self.AUTOCORR2D_75 is None or self.AUTOCORR2D_76 is None or self.AUTOCORR2D_77 is None or self.AUTOCORR2D_78 is None or self.AUTOCORR2D_79 is None or self.AUTOCORR2D_80 is None or self.AUTOCORR2D_81 is None or self.AUTOCORR2D_82 is None or self.AUTOCORR2D_83 is None or self.AUTOCORR2D_84 is None or self.AUTOCORR2D_85 is None or self.AUTOCORR2D_86 is None or self.AUTOCORR2D_87 is None or self.AUTOCORR2D_88 is None or self.AUTOCORR2D_89 is None or self.AUTOCORR2D_90 is None or self.AUTOCORR2D_91 is None or self.AUTOCORR2D_92 is None or self.AUTOCORR2D_93 is None or self.AUTOCORR2D_94 is None or self.AUTOCORR2D_95 is None or self.AUTOCORR2D_96 is None or self.AUTOCORR2D_97 is None or self.AUTOCORR2D_98 is None or self.AUTOCORR2D_99 is None or self.AUTOCORR2D_100 is None or self.AUTOCORR2D_101 is None or self.AUTOCORR2D_102 is None or self.AUTOCORR2D_103 is None or self.AUTOCORR2D_104 is None or self.AUTOCORR2D_105 is None or self.AUTOCORR2D_106 is None or self.AUTOCORR2D_107 is None or self.AUTOCORR2D_108 is None or self.AUTOCORR2D_109 is None or self.AUTOCORR2D_110 is None or self.AUTOCORR2D_111 is None or self.AUTOCORR2D_112 is None or self.AUTOCORR2D_113 is None or self.AUTOCORR2D_114 is None or self.AUTOCORR2D_115 is None or self.AUTOCORR2D_116 is None or self.AUTOCORR2D_117 is None or self.AUTOCORR2D_118 is None or self.AUTOCORR2D_119 is None or self.AUTOCORR2D_120 is None or self.AUTOCORR2D_121 is None or self.AUTOCORR2D_122 is None or self.AUTOCORR2D_123 is None or self.AUTOCORR2D_124 is None or self.AUTOCORR2D_125 is None or self.AUTOCORR2D_126 is None or self.AUTOCORR2D_127 is None or self.AUTOCORR2D_128 is None or self.AUTOCORR2D_129 is None or self.AUTOCORR2D_130 is None or self.AUTOCORR2D_131 is None or self.AUTOCORR2D_132 is None or self.AUTOCORR2D_133 is None or self.AUTOCORR2D_134 is None or self.AUTOCORR2D_135 is None or self.AUTOCORR2D_136 is None or self.AUTOCORR2D_137 is None or self.AUTOCORR2D_138 is None or self.AUTOCORR2D_139 is None or self.AUTOCORR2D_140 is None or self.AUTOCORR2D_141 is None or self.AUTOCORR2D_142 is None or self.AUTOCORR2D_143 is None or self.AUTOCORR2D_144 is None or self.AUTOCORR2D_145 is None or self.AUTOCORR2D_146 is None or self.AUTOCORR2D_147 is None or self.AUTOCORR2D_148 is None or self.AUTOCORR2D_149 is None or self.AUTOCORR2D_150 is None or self.AUTOCORR2D_151 is None or self.AUTOCORR2D_152 is None or self.AUTOCORR2D_153 is None or self.AUTOCORR2D_154 is None or self.AUTOCORR2D_155 is None or self.AUTOCORR2D_156 is None or self.AUTOCORR2D_157 is None or self.AUTOCORR2D_158 is None or self.AUTOCORR2D_159 is None or self.AUTOCORR2D_160 is None or self.AUTOCORR2D_161 is None or self.AUTOCORR2D_162 is None or self.AUTOCORR2D_163 is None or self.AUTOCORR2D_164 is None or self.AUTOCORR2D_165 is None or self.AUTOCORR2D_166 is None or self.AUTOCORR2D_167 is None or self.AUTOCORR2D_168 is None or self.AUTOCORR2D_169 is None or self.AUTOCORR2D_170 is None or self.AUTOCORR2D_171 is None or self.AUTOCORR2D_172 is None or self.AUTOCORR2D_173 is None or self.AUTOCORR2D_174 is None or self.AUTOCORR2D_175 is None or self.AUTOCORR2D_176 is None or self.AUTOCORR2D_177 is None or self.AUTOCORR2D_178 is None or self.AUTOCORR2D_179 is None or self.AUTOCORR2D_180 is None or self.AUTOCORR2D_181 is None or self.AUTOCORR2D_182 is None or self.AUTOCORR2D_183 is None or self.AUTOCORR2D_184 is None or self.AUTOCORR2D_185 is None or self.AUTOCORR2D_186 is None or self.AUTOCORR2D_187 is None or self.AUTOCORR2D_188 is None or self.AUTOCORR2D_189 is None or self.AUTOCORR2D_190 is None or self.AUTOCORR2D_191 is None or self.AUTOCORR2D_192 is None or self.BCUT2D_CHGHI is None or self.BCUT2D_CHGLO is None or self.BCUT2D_LOGPHI is None or self.BCUT2D_LOGPLOW is None or self.BCUT2D_MRHI is None or self.BCUT2D_MRLOW is None or self.BCUT2D_MWHI is None or self.BCUT2D_MWLOW is None or self.BalabanJ is None or self.BertzCT is None or self.Chi0 is None or self.Chi0n is None or self.Chi0v is None or self.Chi1 is None or self.Chi1n is None or self.Chi1v is None or self.Chi2n is None or self.Chi2v is None or self.Chi3n is None or self.Chi3v is None or self.Chi4n is None or self.Chi4v is None or self.EState_VSA1 is None or self.EState_VSA2 is None or self.EState_VSA3 is None or self.EState_VSA4 is None or self.EState_VSA5 is None or self.EState_VSA6 is None or self.EState_VSA7 is None or self.EState_VSA8 is None or self.EState_VSA9 is None or self.EState_VSA10 is None or self.EState_VSA11 is None or self.MaxAbsEStateIndex is None or self.MaxEStateIndex is None or self.MinAbsEStateIndex is None or self.MinEStateIndex is None or self.ExactMolWt is None or self.FpDensityMorgan1 is None or self.FpDensityMorgan2 is None or self.FpDensityMorgan3 is None or self.fr_Al_COO is None or self.fr_Al_OH is None or self.fr_Al_OH_noTert is None or self.fr_ArN is None or self.fr_Ar_COO is None or self.fr_Ar_N is None or self.fr_Ar_NH is None or self.fr_Ar_OH is None or self.fr_COO is None or self.fr_COO2 is None or self.fr_C_O is None or self.fr_C_O_noCOO is None or self.fr_C_S is None or self.fr_HOCCN is None or self.fr_Imine is None or self.fr_NH0 is None or self.fr_NH1 is None or self.fr_NH2 is None or self.fr_N_O is None or self.fr_Ndealkylation1 is None or self.fr_Ndealkylation2 is None or self.fr_Nhpyrrole is None or self.fr_SH is None or self.fr_aldehyde is None or self.fr_alkyl_carbamate is None or self.fr_alkyl_halide is None or self.fr_allylic_oxid is None or self.fr_amide is None or self.fr_amidine is None or self.fr_aniline is None or self.fr_aryl_methyl is None or self.fr_azide is None or self.fr_azo is None or self.fr_barbitur is None or self.fr_benzene is None or self.fr_benzodiazepine is None or self.fr_bicyclic is None or self.fr_diazo is None or self.fr_dihydropyridine is None or self.fr_epoxide is None or self.fr_ester is None or self.fr_ether is None or self.fr_furan is None or self.fr_guanido is None or self.fr_halogen is None or self.fr_hdrzine is None or self.fr_hdrzone is None or self.fr_imidazole is None or self.fr_imide is None or self.fr_isocyan is None or self.fr_isothiocyan is None or self.fr_ketone is None or self.fr_ketone_Topliss is None or self.fr_lactam is None or self.fr_lactone is None or self.fr_methoxy is None or self.fr_morpholine is None or self.fr_nitrile is None or self.fr_nitro is None or self.fr_nitro_arom is None or self.fr_nitro_arom_nonortho is None or self.fr_nitroso is None or self.fr_oxazole is None or self.fr_oxime is None or self.fr_para_hydroxylation is None or self.fr_phenol is None or self.fr_phenol_noOrthoHbond is None or self.fr_phos_acid is None or self.fr_phos_ester is None or self.fr_piperdine is None or self.fr_piperzine is None or self.fr_priamide is None or self.fr_prisulfonamd is None or self.fr_pyridine is None or self.fr_quatN is None or self.fr_sulfide is None or self.fr_sulfonamd is None or self.fr_sulfone is None or self.fr_term_acetylene is None or self.fr_tetrazole is None or self.fr_thiazole is None or self.fr_thiocyan is None or self.fr_thiophene is None or self.fr_unbrch_alkane is None or self.fr_urea is None or self.FractionCSP3 is None or self.HallKierAlpha is None or self.HeavyAtomMolWt is None or self.HeavyAtomCount is None or self.Ipc is None or self.Kappa1 is None or self.Kappa2 is None or self.Kappa3 is None or self.LabuteASA is None or self.MaxAbsPartialCharge is None or self.MaxPartialCharge is None or self.MinAbsPartialCharge is None or self.MinPartialCharge is None or self.MolLogP is None or self.MolMR is None or self.MolWt is None or self.NHOHCount is None or self.NOCount is None or self.NumAliphaticCarbocycles is None or self.NumAliphaticHeterocycles is None or self.NumAliphaticRings is None or self.NumAromaticCarbocycles is None or self.NumAromaticHeterocycles is None or self.NumAromaticRings is None or self.NumHAcceptors is None or self.NumHDonors is None or self.NumHeteroatoms is None or self.NumRadicalElectrons is None or self.NumRotatableBonds is None or self.NumSaturatedCarbocycles is None or self.NumSaturatedHeterocycles is None or self.NumSaturatedRings is None or self.NumValenceElectrons is None or self.PEOE_VSA1 is None or self.PEOE_VSA2 is None or self.PEOE_VSA3 is None or self.PEOE_VSA4 is None or self.PEOE_VSA5 is None or self.PEOE_VSA6 is None or self.PEOE_VSA7 is None or self.PEOE_VSA8 is None or self.PEOE_VSA9 is None or self.PEOE_VSA10 is None or self.PEOE_VSA11 is None or self.PEOE_VSA12 is None or self.PEOE_VSA13 is None or self.PEOE_VSA14 is None or self.qed is None or self.RingCount is None or self.SMR_VSA1 is None or self.SMR_VSA2 is None or self.SMR_VSA3 is None or self.SMR_VSA4 is None or self.SMR_VSA5 is None or self.SMR_VSA6 is None or self.SMR_VSA7 is None or self.SMR_VSA8 is None or self.SMR_VSA9 is None or self.SMR_VSA10 is None or self.SlogP_VSA1 is None or self.SlogP_VSA2 is None or self.SlogP_VSA3 is None or self.SlogP_VSA4 is None or self.SlogP_VSA5 is None or self.SlogP_VSA6 is None or self.SlogP_VSA7 is None or self.SlogP_VSA8 is None or self.SlogP_VSA9 is None or self.SlogP_VSA10 is None or self.SlogP_VSA11 is None or self.SlogP_VSA12 is None or self.TPSA is None or self.VSA_EState1 is None or self.VSA_EState2 is None or self.VSA_EState3 is None or self.VSA_EState4 is None or self.VSA_EState5 is None or self.VSA_EState6 is None or self.VSA_EState7 is None or self.VSA_EState8 is None or self.VSA_EState9 is None or self.VSA_EState10 is None:
            return False
        #endregion
        return True

    def to_smiles(self):
        '''
        Return the smiles of the molecule
        Input:
          -
        Return:
          [string] The smiles of given molecule
        '''
        return get_smiles(self.molecule)

    def is_same_molecule(self, molecule, sanitize = False):
        '''
        Compare two molecules to check if they are the same using their MACCSkeys.
        Input:
          [rdkit.Chem.rdchem.Mol/ocl.Ligand] molecule               - The molecule to compare with.
          [bool]                             sanitize DEFAULT: True - Flag to allow, or not, molecules sanitization.
        Return:
          [bool]
            True  - If both molecules are the same.
            False - If both molecules are not the same.
          [int] If fails
            Check Error.py for error codes
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

    def is_same_molecule_SMILES(self, molecule, sanitize = False):
        '''
        Compare two molecules to check if they are the same using their SMILES and FpDensityMorgan 1 2 and 3.
        Input:
          [rdkit.Chem.rdchem.Mol/ocl.Ligand] molecule               - The molecule to compare with.
          [bool]                             sanitize DEFAULT: True - Flag to allow, or not, molecules sanitization.
        Return:
          [bool]
            True  - If both molecules are the same.
            False - If both molecules are not the same.
          [int] If fails
            Check Error.py for error codes
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
                    try:
                        # Turn off the property cache
                        m.UpdatePropertyCache(strict = False)
                        # Perform a partial sanitization (THIS IS VERY IMPORTANT!!!!)
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True)
                        # Return the sanitized molecule
                        return molecule, m
                    except Exception as e:
                        _ = errors.parse_molecule(f"The molecule '{molecule}' could not be parsed.", "error")
                        return molecule, None

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
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True)
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
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True)
                        # Return the sanitized molecule
                        return molecule, m

                    # Since the sdf file can hold more than one molecule...
                    mols = rdkit.Chem.rdmolfiles.SDMolSupplier(molecule, sanitize = True)
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
                        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catcherrors=True)
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

def read_descriptors_from_json(path, returnDict = False):
    '''
    Read the descriptors from a json file.
    Input:
      path       [string]                - Path to the json file
      returnDict [bool]   DEFAULT: False - If true forces the function to return the entire dict rather than each element separately.
    Return:
      [list(mixed)] - Descriptors read from the json file. If fails, returns null.
    '''
    # Try to read the file
    try:
        # Open the json file in read mode
        with open(path, "r") as f:
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
        # If the returnDict flag is on
        if returnDict:
            # Return the entire dict
            return data
        # Since we have all keys, read them and return their values
        #region Return data
        return data["Name"], data["AUTOCORR2D_1"], data["AUTOCORR2D_2"], data["AUTOCORR2D_3"], data["AUTOCORR2D_4"], data["AUTOCORR2D_5"], data["AUTOCORR2D_6"], data["AUTOCORR2D_7"], data["AUTOCORR2D_8"], data["AUTOCORR2D_9"], data["AUTOCORR2D_10"], data["AUTOCORR2D_11"], data["AUTOCORR2D_12"], data["AUTOCORR2D_13"], data["AUTOCORR2D_14"], data["AUTOCORR2D_15"], data["AUTOCORR2D_16"], data["AUTOCORR2D_17"], data["AUTOCORR2D_18"], data["AUTOCORR2D_19"], data["AUTOCORR2D_20"], data["AUTOCORR2D_21"], data["AUTOCORR2D_22"], data["AUTOCORR2D_23"], data["AUTOCORR2D_24"], data["AUTOCORR2D_25"], data["AUTOCORR2D_26"], data["AUTOCORR2D_27"], data["AUTOCORR2D_28"], data["AUTOCORR2D_29"], data["AUTOCORR2D_30"], data["AUTOCORR2D_31"], data["AUTOCORR2D_32"], data["AUTOCORR2D_33"], data["AUTOCORR2D_34"], data["AUTOCORR2D_35"], data["AUTOCORR2D_36"], data["AUTOCORR2D_37"], data["AUTOCORR2D_38"], data["AUTOCORR2D_39"], data["AUTOCORR2D_40"], data["AUTOCORR2D_41"], data["AUTOCORR2D_42"], data["AUTOCORR2D_43"], data["AUTOCORR2D_44"], data["AUTOCORR2D_45"], data["AUTOCORR2D_46"], data["AUTOCORR2D_47"], data["AUTOCORR2D_48"], data["AUTOCORR2D_49"], data["AUTOCORR2D_50"], data["AUTOCORR2D_51"], data["AUTOCORR2D_52"], data["AUTOCORR2D_53"], data["AUTOCORR2D_54"], data["AUTOCORR2D_55"], data["AUTOCORR2D_56"], data["AUTOCORR2D_57"], data["AUTOCORR2D_58"], data["AUTOCORR2D_59"], data["AUTOCORR2D_60"], data["AUTOCORR2D_61"], data["AUTOCORR2D_62"], data["AUTOCORR2D_63"], data["AUTOCORR2D_64"], data["AUTOCORR2D_65"], data["AUTOCORR2D_66"], data["AUTOCORR2D_67"], data["AUTOCORR2D_68"], data["AUTOCORR2D_69"], data["AUTOCORR2D_70"], data["AUTOCORR2D_71"], data["AUTOCORR2D_72"], data["AUTOCORR2D_73"], data["AUTOCORR2D_74"], data["AUTOCORR2D_75"], data["AUTOCORR2D_76"], data["AUTOCORR2D_77"], data["AUTOCORR2D_78"], data["AUTOCORR2D_79"], data["AUTOCORR2D_80"], data["AUTOCORR2D_81"], data["AUTOCORR2D_82"], data["AUTOCORR2D_83"], data["AUTOCORR2D_84"], data["AUTOCORR2D_85"], data["AUTOCORR2D_86"], data["AUTOCORR2D_87"], data["AUTOCORR2D_88"], data["AUTOCORR2D_89"], data["AUTOCORR2D_90"], data["AUTOCORR2D_91"], data["AUTOCORR2D_92"], data["AUTOCORR2D_93"], data["AUTOCORR2D_94"], data["AUTOCORR2D_95"], data["AUTOCORR2D_96"], data["AUTOCORR2D_97"], data["AUTOCORR2D_98"], data["AUTOCORR2D_99"], data["AUTOCORR2D_100"], data["AUTOCORR2D_101"], data["AUTOCORR2D_102"], data["AUTOCORR2D_103"], data["AUTOCORR2D_104"], data["AUTOCORR2D_105"], data["AUTOCORR2D_106"], data["AUTOCORR2D_107"], data["AUTOCORR2D_108"], data["AUTOCORR2D_109"], data["AUTOCORR2D_110"], data["AUTOCORR2D_111"], data["AUTOCORR2D_112"], data["AUTOCORR2D_113"], data["AUTOCORR2D_114"], data["AUTOCORR2D_115"], data["AUTOCORR2D_116"], data["AUTOCORR2D_117"], data["AUTOCORR2D_118"], data["AUTOCORR2D_119"], data["AUTOCORR2D_120"], data["AUTOCORR2D_121"], data["AUTOCORR2D_122"], data["AUTOCORR2D_123"], data["AUTOCORR2D_124"], data["AUTOCORR2D_125"], data["AUTOCORR2D_126"], data["AUTOCORR2D_127"], data["AUTOCORR2D_128"], data["AUTOCORR2D_129"], data["AUTOCORR2D_130"], data["AUTOCORR2D_131"], data["AUTOCORR2D_132"], data["AUTOCORR2D_133"], data["AUTOCORR2D_134"], data["AUTOCORR2D_135"], data["AUTOCORR2D_136"], data["AUTOCORR2D_137"], data["AUTOCORR2D_138"], data["AUTOCORR2D_139"], data["AUTOCORR2D_140"], data["AUTOCORR2D_141"], data["AUTOCORR2D_142"], data["AUTOCORR2D_143"], data["AUTOCORR2D_144"], data["AUTOCORR2D_145"], data["AUTOCORR2D_146"], data["AUTOCORR2D_147"], data["AUTOCORR2D_148"], data["AUTOCORR2D_149"], data["AUTOCORR2D_150"], data["AUTOCORR2D_151"], data["AUTOCORR2D_152"], data["AUTOCORR2D_153"], data["AUTOCORR2D_154"], data["AUTOCORR2D_155"], data["AUTOCORR2D_156"], data["AUTOCORR2D_157"], data["AUTOCORR2D_158"], data["AUTOCORR2D_159"], data["AUTOCORR2D_160"], data["AUTOCORR2D_161"], data["AUTOCORR2D_162"], data["AUTOCORR2D_163"], data["AUTOCORR2D_164"], data["AUTOCORR2D_165"], data["AUTOCORR2D_166"], data["AUTOCORR2D_167"], data["AUTOCORR2D_168"], data["AUTOCORR2D_169"], data["AUTOCORR2D_170"], data["AUTOCORR2D_171"], data["AUTOCORR2D_172"], data["AUTOCORR2D_173"], data["AUTOCORR2D_174"], data["AUTOCORR2D_175"], data["AUTOCORR2D_176"], data["AUTOCORR2D_177"], data["AUTOCORR2D_178"], data["AUTOCORR2D_179"], data["AUTOCORR2D_180"], data["AUTOCORR2D_181"], data["AUTOCORR2D_182"], data["AUTOCORR2D_183"], data["AUTOCORR2D_184"], data["AUTOCORR2D_185"], data["AUTOCORR2D_186"], data["AUTOCORR2D_187"], data["AUTOCORR2D_188"], data["AUTOCORR2D_189"], data["AUTOCORR2D_190"], data["AUTOCORR2D_191"], data["AUTOCORR2D_192"], data["BCUT2D_CHGHI"], data["BCUT2D_CHGLO"], data["BCUT2D_LOGPHI"], data["BCUT2D_LOGPLOW"], data["BCUT2D_MRHI"], data["BCUT2D_MRLOW"], data["BCUT2D_MWHI"], data["BCUT2D_MWLOW"], data["BalabanJ"], data["BertzCT"], data["Chi0"], data["Chi0n"], data["Chi0v"], data["Chi1"], data["Chi1n"], data["Chi1v"], data["Chi2n"], data["Chi2v"], data["Chi3n"], data["Chi3v"], data["Chi4n"], data["Chi4v"], data["EState_VSA1"], data["EState_VSA2"], data["EState_VSA3"], data["EState_VSA4"], data["EState_VSA5"], data["EState_VSA6"], data["EState_VSA7"], data["EState_VSA8"], data["EState_VSA9"], data["EState_VSA10"], data["EState_VSA11"], data["MaxAbsEStateIndex"], data["MaxEStateIndex"], data["MinAbsEStateIndex"], data["MinEStateIndex"], data["ExactMolWt"], data["FpDensityMorgan1"], data["FpDensityMorgan2"], data["FpDensityMorgan3"], data["fr_Al_COO"], data["fr_Al_OH"], data["fr_Al_OH_noTert"], data["fr_ArN"], data["fr_Ar_COO"], data["fr_Ar_N"], data["fr_Ar_NH"], data["fr_Ar_OH"], data["fr_COO"], data["fr_COO2"], data["fr_C_O"], data["fr_C_O_noCOO"], data["fr_C_S"], data["fr_HOCCN"], data["fr_Imine"], data["fr_NH0"], data["fr_NH1"], data["fr_NH2"], data["fr_N_O"], data["fr_Ndealkylation1"], data["fr_Ndealkylation2"], data["fr_Nhpyrrole"], data["fr_SH"], data["fr_aldehyde"], data["fr_alkyl_carbamate"], data["fr_alkyl_halide"], data["fr_allylic_oxid"], data["fr_amide"], data["fr_amidine"], data["fr_aniline"], data["fr_aryl_methyl"], data["fr_azide"], data["fr_azo"], data["fr_barbitur"], data["fr_benzene"], data["fr_benzodiazepine"], data["fr_bicyclic"], data["fr_diazo"], data["fr_dihydropyridine"], data["fr_epoxide"], data["fr_ester"], data["fr_ether"], data["fr_furan"], data["fr_guanido"], data["fr_halogen"], data["fr_hdrzine"], data["fr_hdrzone"], data["fr_imidazole"], data["fr_imide"], data["fr_isocyan"], data["fr_isothiocyan"], data["fr_ketone"], data["fr_ketone_Topliss"], data["fr_lactam"], data["fr_lactone"], data["fr_methoxy"], data["fr_morpholine"], data["fr_nitrile"], data["fr_nitro"], data["fr_nitro_arom"], data["fr_nitro_arom_nonortho"], data["fr_nitroso"], data["fr_oxazole"], data["fr_oxime"], data["fr_para_hydroxylation"], data["fr_phenol"], data["fr_phenol_noOrthoHbond"], data["fr_phos_acid"], data["fr_phos_ester"], data["fr_piperdine"], data["fr_piperzine"], data["fr_priamide"], data["fr_prisulfonamd"], data["fr_pyridine"], data["fr_quatN"], data["fr_sulfide"], data["fr_sulfonamd"], data["fr_sulfone"], data["fr_term_acetylene"], data["fr_tetrazole"], data["fr_thiazole"], data["fr_thiocyan"], data["fr_thiophene"], data["fr_unbrch_alkane"], data["fr_urea"], data["FractionCSP3"], data["HallKierAlpha"], data["HeavyAtomMolWt"], data["HeavyAtomCount"], data["Ipc"], data["Kappa1"], data["Kappa2"], data["Kappa3"], data["LabuteASA"], data["MaxAbsPartialCharge"], data["MaxPartialCharge"], data["MinAbsPartialCharge"], data["MinPartialCharge"], data["MolLogP"], data["MolMR"], data["MolWt"], data["NHOHCount"], data["NOCount"], data["NumAliphaticCarbocycles"], data["NumAliphaticHeterocycles"], data["NumAliphaticRings"], data["NumAromaticCarbocycles"], data["NumAromaticHeterocycles"], data["NumAromaticRings"], data["NumHAcceptors"], data["NumHDonors"], data["NumHeteroatoms"], data["NumRadicalElectrons"], data["NumRotatableBonds"], data["NumSaturatedCarbocycles"], data["NumSaturatedHeterocycles"], data["NumSaturatedRings"], data["NumValenceElectrons"], data["PEOE_VSA1"], data["PEOE_VSA2"], data["PEOE_VSA3"], data["PEOE_VSA4"], data["PEOE_VSA5"], data["PEOE_VSA6"], data["PEOE_VSA7"], data["PEOE_VSA8"], data["PEOE_VSA9"], data["PEOE_VSA10"], data["PEOE_VSA11"], data["PEOE_VSA12"], data["PEOE_VSA13"], data["PEOE_VSA14"], data["qed"], data["RingCount"], data["SMR_VSA1"], data["SMR_VSA2"], data["SMR_VSA3"], data["SMR_VSA4"], data["SMR_VSA5"], data["SMR_VSA6"], data["SMR_VSA7"], data["SMR_VSA8"], data["SMR_VSA9"], data["SMR_VSA10"], data["SlogP_VSA1"], data["SlogP_VSA2"], data["SlogP_VSA3"], data["SlogP_VSA4"], data["SlogP_VSA5"], data["SlogP_VSA6"], data["SlogP_VSA7"], data["SlogP_VSA8"], data["SlogP_VSA9"], data["SlogP_VSA10"], data["SlogP_VSA11"], data["SlogP_VSA12"], data["TPSA"], data["VSA_EState1"], data["VSA_EState2"], data["VSA_EState3"], data["VSA_EState4"], data["VSA_EState5"], data["VSA_EState6"], data["VSA_EState7"], data["VSA_EState8"], data["VSA_EState9"], data["VSA_EState10"], data["AUTOCORR3D_1"], data["AUTOCORR3D_2"], data["AUTOCORR3D_3"], data["AUTOCORR3D_4"], data["AUTOCORR3D_5"], data["AUTOCORR3D_6"], data["AUTOCORR3D_7"], data["AUTOCORR3D_8"], data["AUTOCORR3D_9"], data["AUTOCORR3D_10"], data["AUTOCORR3D_11"], data["AUTOCORR3D_12"], data["AUTOCORR3D_13"], data["AUTOCORR3D_14"], data["AUTOCORR3D_15"], data["AUTOCORR3D_16"], data["AUTOCORR3D_17"], data["AUTOCORR3D_18"], data["AUTOCORR3D_19"], data["AUTOCORR3D_20"], data["AUTOCORR3D_21"], data["AUTOCORR3D_22"], data["AUTOCORR3D_23"], data["AUTOCORR3D_24"], data["AUTOCORR3D_25"], data["AUTOCORR3D_26"], data["AUTOCORR3D_27"], data["AUTOCORR3D_28"], data["AUTOCORR3D_29"], data["AUTOCORR3D_30"], data["AUTOCORR3D_31"], data["AUTOCORR3D_32"], data["AUTOCORR3D_33"], data["AUTOCORR3D_34"], data["AUTOCORR3D_35"], data["AUTOCORR3D_36"], data["AUTOCORR3D_37"], data["AUTOCORR3D_38"], data["AUTOCORR3D_39"], data["AUTOCORR3D_40"], data["AUTOCORR3D_41"], data["AUTOCORR3D_42"], data["AUTOCORR3D_43"], data["AUTOCORR3D_44"], data["AUTOCORR3D_45"], data["AUTOCORR3D_46"], data["AUTOCORR3D_47"], data["AUTOCORR3D_48"], data["AUTOCORR3D_49"], data["AUTOCORR3D_50"], data["AUTOCORR3D_51"], data["AUTOCORR3D_52"], data["AUTOCORR3D_53"], data["AUTOCORR3D_54"], data["AUTOCORR3D_55"], data["AUTOCORR3D_56"], data["AUTOCORR3D_57"], data["AUTOCORR3D_58"], data["AUTOCORR3D_59"], data["AUTOCORR3D_60"], data["AUTOCORR3D_61"], data["AUTOCORR3D_62"], data["AUTOCORR3D_63"], data["AUTOCORR3D_64"], data["AUTOCORR3D_65"], data["AUTOCORR3D_66"], data["AUTOCORR3D_67"], data["AUTOCORR3D_68"], data["AUTOCORR3D_69"], data["AUTOCORR3D_70"], data["AUTOCORR3D_71"], data["AUTOCORR3D_72"], data["AUTOCORR3D_73"], data["AUTOCORR3D_74"], data["AUTOCORR3D_75"], data["AUTOCORR3D_76"], data["AUTOCORR3D_77"], data["AUTOCORR3D_78"], data["AUTOCORR3D_79"], data["AUTOCORR3D_80"], data["Asphericity"], data["Eccentricity"], data["InertialShapeFactor"], data["NPR1"], data["NPR2"], data["PMI1"], data["PMI2"], data["PMI3"], data["RadiusOfGyration"], data["SpherocityIndex"]
        #endregion
    # Key error (when there is a missing key)
    except KeyError as k:
        octools.print_error(f"The following keys were not found in the json file '{k[0]}': {k[1]}.")
    # General error (call it as problem to read file)
    except Exception as e:
        octools.print_error(f"Could not read the file '{path}'. Error: {e}")
    return None

def get_smiles(molecule):
    '''
    Return the smiles of the molecule
    Input:
      [rdkit.Chem.rdchem.Mol] molecule - The molecule to retrive the smiles
    Return:
      [string] The smiles of given molecule
    '''
    if molecule:
        if type(molecule) == rdkit.Chem.rdchem.Mol:
            return Chem.MolToSmiles(molecule)
        return errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    return errors.not_set(f"The variable is not set.")

# Descriptors functions #

#region AUTOCORR descriptors
def findAUTOCORR2D_1(molecule):
    '''
    Compute the autocorrelation2D_1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_2(molecule):
    '''
    Compute the autocorrelation2D_2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_3(molecule):
    '''
    Compute the autocorrelation2D_3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_4(molecule):
    '''
    Compute the autocorrelation2D_4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_5(molecule):
    '''
    Compute the autocorrelation2D_5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_6(molecule):
    '''
    Compute the autocorrelation2D_6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_7(molecule):
    '''
    Compute the autocorrelation2D_7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_8(molecule):
    '''
    Compute the autocorrelation2D_8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_9(molecule):
    '''
    Compute the autocorrelation2D_9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_10(molecule):
    '''
    Compute the autocorrelation2D_10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_11(molecule):
    '''
    Compute the autocorrelation2D_11 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_11 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_11(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_12(molecule):
    '''
    Compute the autocorrelation2D_12 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_12 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_12(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_13(molecule):
    '''
    Compute the autocorrelation2D_13 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_13 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_13(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_14(molecule):
    '''
    Compute the autocorrelation2D_14 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_14 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_14(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_15(molecule):
    '''
    Compute the autocorrelation2D_15 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_15 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_15(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_16(molecule):
    '''
    Compute the autocorrelation2D_16 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_16 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_16(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_17(molecule):
    '''
    Compute the autocorrelation2D_17 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_17 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_17(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_18(molecule):
    '''
    Compute the autocorrelation2D_18 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_18 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_18(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_19(molecule):
    '''
    Compute the autocorrelation2D_19 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_19 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_19(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_20(molecule):
    '''
    Compute the autocorrelation2D_20 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_20 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_20(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_21(molecule):
    '''
    Compute the autocorrelation2D_21 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_21 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_21(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_22(molecule):
    '''
    Compute the autocorrelation2D_22 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_22 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_22(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_23(molecule):
    '''
    Compute the autocorrelation2D_23 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_23 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_23(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_24(molecule):
    '''
    Compute the autocorrelation2D_24 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_24 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_24(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_25(molecule):
    '''
    Compute the autocorrelation2D_25 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_25 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_25(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_26(molecule):
    '''
    Compute the autocorrelation2D_26 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_26 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_26(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_27(molecule):
    '''
    Compute the autocorrelation2D_27 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_27 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_27(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_28(molecule):
    '''
    Compute the autocorrelation2D_28 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_28 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_28(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_29(molecule):
    '''
    Compute the autocorrelation2D_29 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_29 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_29(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_30(molecule):
    '''
    Compute the autocorrelation2D_30 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_30 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_30(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_31(molecule):
    '''
    Compute the autocorrelation2D_31 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_31 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_31(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_32(molecule):
    '''
    Compute the autocorrelation2D_32 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_32 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_32(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_33(molecule):
    '''
    Compute the autocorrelation2D_33 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_33 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_33(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_34(molecule):
    '''
    Compute the autocorrelation2D_34 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_34 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_34(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_35(molecule):
    '''
    Compute the autocorrelation2D_35 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_35 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_35(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_36(molecule):
    '''
    Compute the autocorrelation2D_36 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_36 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_36(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_37(molecule):
    '''
    Compute the autocorrelation2D_37 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_37 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_37(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_38(molecule):
    '''
    Compute the autocorrelation2D_38 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_38 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_38(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_39(molecule):
    '''
    Compute the autocorrelation2D_39 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_39 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_39(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_40(molecule):
    '''
    Compute the autocorrelation2D_40 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_40 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_40(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_41(molecule):
    '''
    Compute the autocorrelation2D_41 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_41 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_41(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_42(molecule):
    '''
    Compute the autocorrelation2D_42 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_42 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_42(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_43(molecule):
    '''
    Compute the autocorrelation2D_43 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_43 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_43(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_44(molecule):
    '''
    Compute the autocorrelation2D_44 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_44 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_44(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_45(molecule):
    '''
    Compute the autocorrelation2D_45 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_45 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_45(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_46(molecule):
    '''
    Compute the autocorrelation2D_46 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_46 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_46(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_47(molecule):
    '''
    Compute the autocorrelation2D_47 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_47 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_47(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_48(molecule):
    '''
    Compute the autocorrelation2D_48 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_48 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_48(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_49(molecule):
    '''
    Compute the autocorrelation2D_49 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_49 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_49(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_50(molecule):
    '''
    Compute the autocorrelation2D_50 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_50 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_50(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_51(molecule):
    '''
    Compute the autocorrelation2D_51 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_51 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_51(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_52(molecule):
    '''
    Compute the autocorrelation2D_52 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_52 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_52(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_53(molecule):
    '''
    Compute the autocorrelation2D_53 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_53 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_53(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_54(molecule):
    '''
    Compute the autocorrelation2D_54 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_54 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_54(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_55(molecule):
    '''
    Compute the autocorrelation2D_55 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_55 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_55(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_56(molecule):
    '''
    Compute the autocorrelation2D_56 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_56 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_56(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_57(molecule):
    '''
    Compute the autocorrelation2D_57 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_57 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_57(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_58(molecule):
    '''
    Compute the autocorrelation2D_58 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_58 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_58(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_59(molecule):
    '''
    Compute the autocorrelation2D_59 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_59 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_59(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_60(molecule):
    '''
    Compute the autocorrelation2D_60 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_60 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_60(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_61(molecule):
    '''
    Compute the autocorrelation2D_61 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_61 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_61(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_62(molecule):
    '''
    Compute the autocorrelation2D_62 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_62 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_62(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_63(molecule):
    '''
    Compute the autocorrelation2D_63 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_63 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_63(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_64(molecule):
    '''
    Compute the autocorrelation2D_64 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_64 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_64(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_65(molecule):
    '''
    Compute the autocorrelation2D_65 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_65 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_65(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_66(molecule):
    '''
    Compute the autocorrelation2D_66 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_66 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_66(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_67(molecule):
    '''
    Compute the autocorrelation2D_67 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_67 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_67(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_68(molecule):
    '''
    Compute the autocorrelation2D_68 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_68 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_68(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_69(molecule):
    '''
    Compute the autocorrelation2D_69 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_69 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_69(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_70(molecule):
    '''
    Compute the autocorrelation2D_70 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_70 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_70(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_71(molecule):
    '''
    Compute the autocorrelation2D_71 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_71 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_71(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_72(molecule):
    '''
    Compute the autocorrelation2D_72 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_72 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_72(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_73(molecule):
    '''
    Compute the autocorrelation2D_73 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_73 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_73(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_74(molecule):
    '''
    Compute the autocorrelation2D_74 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_74 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_74(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_75(molecule):
    '''
    Compute the autocorrelation2D_75 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_75 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_75(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_76(molecule):
    '''
    Compute the autocorrelation2D_76 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_76 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_76(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_77(molecule):
    '''
    Compute the autocorrelation2D_77 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_77 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_77(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_78(molecule):
    '''
    Compute the autocorrelation2D_78 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_78 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_78(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_79(molecule):
    '''
    Compute the autocorrelation2D_79 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_79 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_79(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_80(molecule):
    '''
    Compute the autocorrelation2D_80 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_80 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_80(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_81(molecule):
    '''
    Compute the autocorrelation2D_81 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_81 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_81(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_82(molecule):
    '''
    Compute the autocorrelation2D_82 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_82 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_82(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_83(molecule):
    '''
    Compute the autocorrelation2D_83 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_83 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_83(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_84(molecule):
    '''
    Compute the autocorrelation2D_84 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_84 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_84(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_85(molecule):
    '''
    Compute the autocorrelation2D_85 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_85 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_85(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_86(molecule):
    '''
    Compute the autocorrelation2D_86 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_86 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_86(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_87(molecule):
    '''
    Compute the autocorrelation2D_87 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_87 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_87(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_88(molecule):
    '''
    Compute the autocorrelation2D_88 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_88 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_88(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_89(molecule):
    '''
    Compute the autocorrelation2D_89 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_89 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_89(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_90(molecule):
    '''
    Compute the autocorrelation2D_90 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_90 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_90(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_91(molecule):
    '''
    Compute the autocorrelation2D_91 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_91 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_91(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_92(molecule):
    '''
    Compute the autocorrelation2D_92 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_92 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_92(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_93(molecule):
    '''
    Compute the autocorrelation2D_93 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_93 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_93(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_94(molecule):
    '''
    Compute the autocorrelation2D_94 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_94 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_94(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_95(molecule):
    '''
    Compute the autocorrelation2D_95 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_95 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_95(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_96(molecule):
    '''
    Compute the autocorrelation2D_96 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_96 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_96(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_97(molecule):
    '''
    Compute the autocorrelation2D_97 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_97 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_97(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_98(molecule):
    '''
    Compute the autocorrelation2D_98 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_98 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_98(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_99(molecule):
    '''
    Compute the autocorrelation2D_99 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_99 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_99(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_100(molecule):
    '''
    Compute the autocorrelation2D_100 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_100 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_100(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_101(molecule):
    '''
    Compute the autocorrelation2D_101 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_101 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_101(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_102(molecule):
    '''
    Compute the autocorrelation2D_102 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_102 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_102(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_103(molecule):
    '''
    Compute the autocorrelation2D_103 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_103 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_103(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_104(molecule):
    '''
    Compute the autocorrelation2D_104 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_104 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_104(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_105(molecule):
    '''
    Compute the autocorrelation2D_105 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_105 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_105(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_106(molecule):
    '''
    Compute the autocorrelation2D_106 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_106 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_106(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_107(molecule):
    '''
    Compute the autocorrelation2D_107 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_107 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_107(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_108(molecule):
    '''
    Compute the autocorrelation2D_108 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_108 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_108(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_109(molecule):
    '''
    Compute the autocorrelation2D_109 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_109 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_109(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_110(molecule):
    '''
    Compute the autocorrelation2D_110 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_110 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_110(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_111(molecule):
    '''
    Compute the autocorrelation2D_111 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_111 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_111(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_112(molecule):
    '''
    Compute the autocorrelation2D_112 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_112 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_112(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_113(molecule):
    '''
    Compute the autocorrelation2D_113 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_113 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_113(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_114(molecule):
    '''
    Compute the autocorrelation2D_114 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_114 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_114(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_115(molecule):
    '''
    Compute the autocorrelation2D_115 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_115 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_115(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_116(molecule):
    '''
    Compute the autocorrelation2D_116 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_116 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_116(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_117(molecule):
    '''
    Compute the autocorrelation2D_117 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_117 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_117(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_118(molecule):
    '''
    Compute the autocorrelation2D_118 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_118 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_118(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_119(molecule):
    '''
    Compute the autocorrelation2D_119 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_119 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_119(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_120(molecule):
    '''
    Compute the autocorrelation2D_120 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_120 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_120(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_121(molecule):
    '''
    Compute the autocorrelation2D_121 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_121 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_121(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_122(molecule):
    '''
    Compute the autocorrelation2D_122 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_122 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_122(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_123(molecule):
    '''
    Compute the autocorrelation2D_123 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_123 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_123(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_124(molecule):
    '''
    Compute the autocorrelation2D_124 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_124 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_124(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_125(molecule):
    '''
    Compute the autocorrelation2D_125 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_125 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_125(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_126(molecule):
    '''
    Compute the autocorrelation2D_126 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_126 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_126(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_127(molecule):
    '''
    Compute the autocorrelation2D_127 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_127 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_127(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_128(molecule):
    '''
    Compute the autocorrelation2D_128 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_128 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_128(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_129(molecule):
    '''
    Compute the autocorrelation2D_129 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_129 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_129(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_130(molecule):
    '''
    Compute the autocorrelation2D_130 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_130 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_130(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_131(molecule):
    '''
    Compute the autocorrelation2D_131 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_131 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_131(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_132(molecule):
    '''
    Compute the autocorrelation2D_132 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_132 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_132(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_133(molecule):
    '''
    Compute the autocorrelation2D_133 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_133 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_133(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_134(molecule):
    '''
    Compute the autocorrelation2D_134 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_134 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_134(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_135(molecule):
    '''
    Compute the autocorrelation2D_135 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_135 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_135(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_136(molecule):
    '''
    Compute the autocorrelation2D_136 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_136 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_136(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_137(molecule):
    '''
    Compute the autocorrelation2D_137 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_137 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_137(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_138(molecule):
    '''
    Compute the autocorrelation2D_138 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_138 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_138(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_139(molecule):
    '''
    Compute the autocorrelation2D_139 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_139 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_139(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_140(molecule):
    '''
    Compute the autocorrelation2D_140 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_140 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_140(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_141(molecule):
    '''
    Compute the autocorrelation2D_141 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_141 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_141(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_142(molecule):
    '''
    Compute the autocorrelation2D_142 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_142 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_142(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_143(molecule):
    '''
    Compute the autocorrelation2D_143 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_143 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_143(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_144(molecule):
    '''
    Compute the autocorrelation2D_144 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_144 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_144(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_145(molecule):
    '''
    Compute the autocorrelation2D_145 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_145 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_145(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_146(molecule):
    '''
    Compute the autocorrelation2D_146 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_146 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_146(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_147(molecule):
    '''
    Compute the autocorrelation2D_147 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_147 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_147(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_148(molecule):
    '''
    Compute the autocorrelation2D_148 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_148 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_148(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_149(molecule):
    '''
    Compute the autocorrelation2D_149 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_149 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_149(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_150(molecule):
    '''
    Compute the autocorrelation2D_150 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_150 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_150(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_151(molecule):
    '''
    Compute the autocorrelation2D_151 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_151 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_151(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_152(molecule):
    '''
    Compute the autocorrelation2D_152 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_152 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_152(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_153(molecule):
    '''
    Compute the autocorrelation2D_153 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_153 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_153(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_154(molecule):
    '''
    Compute the autocorrelation2D_154 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_154 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_154(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_155(molecule):
    '''
    Compute the autocorrelation2D_155 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_155 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_155(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_156(molecule):
    '''
    Compute the autocorrelation2D_156 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_156 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_156(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_157(molecule):
    '''
    Compute the autocorrelation2D_157 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_157 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_157(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_158(molecule):
    '''
    Compute the autocorrelation2D_158 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_158 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_158(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_159(molecule):
    '''
    Compute the autocorrelation2D_159 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_159 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_159(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_160(molecule):
    '''
    Compute the autocorrelation2D_160 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_160 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_160(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_161(molecule):
    '''
    Compute the autocorrelation2D_161 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_161 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_161(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_162(molecule):
    '''
    Compute the autocorrelation2D_162 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_162 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_162(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_163(molecule):
    '''
    Compute the autocorrelation2D_163 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_163 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_163(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_164(molecule):
    '''
    Compute the autocorrelation2D_164 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_164 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_164(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_165(molecule):
    '''
    Compute the autocorrelation2D_165 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_165 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_165(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_166(molecule):
    '''
    Compute the autocorrelation2D_166 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_166 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_166(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_167(molecule):
    '''
    Compute the autocorrelation2D_167 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_167 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_167(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_168(molecule):
    '''
    Compute the autocorrelation2D_168 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_168 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_168(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_169(molecule):
    '''
    Compute the autocorrelation2D_169 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_169 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_169(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_170(molecule):
    '''
    Compute the autocorrelation2D_170 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_170 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_170(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_171(molecule):
    '''
    Compute the autocorrelation2D_171 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_171 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_171(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_172(molecule):
    '''
    Compute the autocorrelation2D_172 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_172 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_172(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_173(molecule):
    '''
    Compute the autocorrelation2D_173 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_173 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_173(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_174(molecule):
    '''
    Compute the autocorrelation2D_174 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_174 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_174(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_175(molecule):
    '''
    Compute the autocorrelation2D_175 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_175 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_175(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_176(molecule):
    '''
    Compute the autocorrelation2D_176 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_176 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_176(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_177(molecule):
    '''
    Compute the autocorrelation2D_177 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_177 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_177(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_178(molecule):
    '''
    Compute the autocorrelation2D_178 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_178 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_178(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_179(molecule):
    '''
    Compute the autocorrelation2D_179 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_179 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_179(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_180(molecule):
    '''
    Compute the autocorrelation2D_180 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_180 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_180(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_181(molecule):
    '''
    Compute the autocorrelation2D_181 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_181 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_181(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_182(molecule):
    '''
    Compute the autocorrelation2D_182 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_182 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_182(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_183(molecule):
    '''
    Compute the autocorrelation2D_183 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_183 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_183(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_184(molecule):
    '''
    Compute the autocorrelation2D_184 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_184 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_184(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_185(molecule):
    '''
    Compute the autocorrelation2D_185 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_185 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_185(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_186(molecule):
    '''
    Compute the autocorrelation2D_186 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_186 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_186(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_187(molecule):
    '''
    Compute the autocorrelation2D_187 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_187 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_187(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_188(molecule):
    '''
    Compute the autocorrelation2D_188 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_188 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_188(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_189(molecule):
    '''
    Compute the autocorrelation2D_189 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_189 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_189(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_190(molecule):
    '''
    Compute the autocorrelation2D_190 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_190 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_190(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_191(molecule):
    '''
    Compute the autocorrelation2D_191 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_191 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_191(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_192(molecule):
    '''
    Compute the autocorrelation2D_192 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The autocorrelation2D_192 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_192(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

#region BCUT2D descriptors
def findBCUT2D_CHGHI(molecule):
    '''
    Compute the BCUT2D_CHGHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_CHGHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_CHGHI(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_CHGLO(molecule):
    '''
    Compute the BCUT2D_CHGLO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The the BCUT2D_CHGLO descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_CHGLO(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_LOGPHI(molecule):
    '''
    Compute the BCUT2D_LOGPHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_LOGPHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_LOGPHI(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_LOGPLOW(molecule):
    '''
    Compute the BCUT2D_LOGPLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_LOGPLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_LOGPLOW(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MRHI(molecule):
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
            return rdkit.Chem.Descriptors.BCUT2D_MRHI(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MRLOW(molecule):
    '''
    Compute the BCUT2D_MRLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_MRLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MRLOW(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MWHI(molecule):
    '''
    Compute the BCUT2D_MWHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_MWHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MWHI(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MWLOW(molecule):
    '''
    Compute the BCUT2D_MWLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BCUT2D_MWLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MWLOW(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findBalabanJ(molecule):
    '''
    Compute the BalabanJ descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BalabanJ descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BalabanJ(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findBertzCT(molecule):
    '''
    Compute the BertzCT descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The BertzCT descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BertzCT(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region Chi descriptors
def findChi0(molecule):
    '''
    Compute the Chi0 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi0 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi0n(molecule):
    '''
    Compute the Chi0n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi0n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0n(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi0v(molecule):
    '''
    Compute the Chi0v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi0v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0v(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi1(molecule):
    '''
    Compute the Chi1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi1 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi1n(molecule):
    '''
    Compute the Chi1n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi1n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1n(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi1v(molecule):
    '''
    Compute the Chi1v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi1v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1v(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi2n(molecule):
    '''
    Compute the Chi2n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi2n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi2n(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi2v(molecule):
    '''
    Compute the Chi2v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi2v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi2v(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi3n(molecule):
    '''
    Compute the Chi3n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi3n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi3n(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi3v(molecule):
    '''
    Compute the Chi3v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi3v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi3v(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi4n(molecule):
    '''
    Compute the Chi4n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi4n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi4n(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findChi4v(molecule):
    '''
    Compute the Chi4v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Chi4v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi4v(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

#region EState descriptors
def findEState_VSA1(molecule):
    '''
    Compute the EState_VSA1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA1 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA2(molecule):
    '''
    Compute the EState_VSA2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA2 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA3(molecule):
    '''
    Compute the EState_VSA3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA3 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA4(molecule):
    '''
    Compute the EState_VSA4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA4 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA5(molecule):
    '''
    Compute the EState_VSA5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA5 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA6(molecule):
    '''
    Compute the EState_VSA6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA6 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA7(molecule):
    '''
    Compute the EState_VSA7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA7 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA8(molecule):
    '''
    Compute the EState_VSA8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA8 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA9(molecule):
    '''
    Compute the EState_VSA9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA9 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA10(molecule):
    '''
    Compute the EState_VSA10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA10 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEState_VSA11(molecule):
    '''
    Compute the EState_VSA11 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The EState_VSA11 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.EState_VSA11(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMaxAbsEStateIndex(molecule):
    '''
    Compute the MaxAbsEStateIndex descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MaxAbsEStateIndex descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxAbsEStateIndex(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMaxEStateIndex(molecule):
    '''
    Compute the MaxEStateIndex descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MaxEStateIndex descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxEStateIndex(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMinAbsEStateIndex(molecule):
    '''
    Compute the MinAbsEStateIndex descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MinAbsEStateIndex descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinAbsEStateIndex(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMinEStateIndex(molecule):
    '''
    Compute the MinEStateIndex descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MinEStateIndex descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinEStateIndex(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findExactMolWt(molecule):
    '''
    Compute the exact molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The exact molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.ExactMolWt(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findFpDensityMorgan1(molecule):
    '''
    Compute the Morgan fingerprint, radius 1 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Morgan fingerprint, radius 1.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findFpDensityMorgan2(molecule):
    '''
    Compute the Morgan fingerprint, radius 2 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Morgan fingerprint, radius 2.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findFpDensityMorgan3(molecule):
    '''
    Compute the Morgan fingerprint, radius 3 descriptor of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Morgan fingerprint, radius 3.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region fr_ descriptors
def findfr_Al_COO(molecule):
    '''
    Compute the fr_Al_COO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Al_COO value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Al_COO(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Al_OH(molecule):
    '''
    Compute the fr_Al_OH descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Al_OH value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Al_OH(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Al_OH_noTert(molecule):
    '''
    Compute the fr_Al_OH_noTert descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Al_OH_noTert value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Al_OH_noTert(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_ArN(molecule):
    '''
    Compute the fr_ArN descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_ArN value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_ArN(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ar_COO(molecule):
    '''
    Compute the fr_Ar_COO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ar_COO value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ar_COO(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ar_N(molecule):
    '''
    Compute the fr_Ar_N descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ar_N value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ar_N(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ar_NH(molecule):
    '''
    Compute the fr_Ar_NH descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ar_NH value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ar_NH(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ar_OH(molecule):
    '''
    Compute the fr_Ar_OH descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ar_OH value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ar_OH(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_COO(molecule):
    '''
    Compute the fr_COO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_COO value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_COO(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_COO2(molecule):
    '''
    Compute the fr_COO2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_COO2 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_COO2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_C_O(molecule):
    '''
    Compute the fr_C_O descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_C_O value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_C_O(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_C_O_noCOO(molecule):
    '''
    Compute the fr_C_O_noCOO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_C_O_noCOO value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_C_O_noCOO(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_C_S(molecule):
    '''
    Compute the fr_C_S descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_C_S value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_C_S(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_HOCCN(molecule):
    '''
    Compute the fr_HOCCN descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_HOCCN value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_HOCCN(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Imine(molecule):
    '''
    Compute the fr_Imine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Imine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Imine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_NH0(molecule):
    '''
    Compute the fr_NH0 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_NH0 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_NH0(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_NH1(molecule):
    '''
    Compute the fr_NH1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_NH1 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_NH1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_NH2(molecule):
    '''
    Compute the fr_NH2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_NH2 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_NH2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_N_O(molecule):
    '''
    Compute the fr_N_O descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_N_O value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_N_O(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ndealkylation1(molecule):
    '''
    Compute the fr_Ndealkylation1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ndealkylation1 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ndealkylation1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Ndealkylation2(molecule):
    '''
    Compute the fr_Ndealkylation2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Ndealkylation2 value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Ndealkylation2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_Nhpyrrole(molecule):
    '''
    Compute the fr_Nhpyrrole descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_Nhpyrrole value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_Nhpyrrole(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_SH(molecule):
    '''
    Compute the fr_SH descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_SH value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_SH(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_aldehyde(molecule):
    '''
    Compute the fr_aldehyde descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_aldehyde value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_aldehyde(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_alkyl_carbamate(molecule):
    '''
    Compute the fr_alkyl_carbamate descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_alkyl_carbamate value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_alkyl_carbamate(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_alkyl_halide(molecule):
    '''
    Compute the fr_alkyl_halide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_alkyl_halide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_alkyl_halide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_allylic_oxid(molecule):
    '''
    Compute the fr_allylic_oxid descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_allylic_oxid value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_allylic_oxid(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_amide(molecule):
    '''
    Compute the fr_amide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_amide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_amide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_amidine(molecule):
    '''
    Compute the fr_amidine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_amidine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_amidine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_aniline(molecule):
    '''
    Compute the fr_aniline descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_aniline value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_aniline(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_aryl_methyl(molecule):
    '''
    Compute the fr_aryl_methyl descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_aryl_methyl value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_aryl_methyl(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_azide(molecule):
    '''
    Compute the fr_azide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_azide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_azide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_azo(molecule):
    '''
    Compute the fr_azo descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_azo value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_azo(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_barbitur(molecule):
    '''
    Compute the fr_barbitur descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_barbitur value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_barbitur(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_benzene(molecule):
    '''
    Compute the fr_benzene descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_benzene value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_benzene(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_benzodiazepine(molecule):
    '''
    Compute the fr_benzodiazepine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_benzodiazepine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_benzodiazepine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_bicyclic(molecule):
    '''
    Compute the fr_bicyclic descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_bicyclic value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_bicyclic(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_diazo(molecule):
    '''
    Compute the fr_diazo descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_diazo value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_diazo(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_dihydropyridine(molecule):
    '''
    Compute the fr_dihydropyridine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_dihydropyridine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_dihydropyridine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_epoxide(molecule):
    '''
    Compute the fr_epoxide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_epoxide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_epoxide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_ester(molecule):
    '''
    Compute the fr_ester descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_ester value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_ester(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_ether(molecule):
    '''
    Compute the fr_ether descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_ether value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_ether(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_furan(molecule):
    '''
    Compute the fr_furan descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_furan value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_furan(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_guanido(molecule):
    '''
    Compute the fr_guanido descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_guanido value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_guanido(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_halogen(molecule):
    '''
    Compute the fr_halogen descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_halogen value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_halogen(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_hdrzine(molecule):
    '''
    Compute the fr_hdrzine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_hdrzine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_hdrzine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_hdrzone(molecule):
    '''
    Compute the fr_hdrzone descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_hdrzone value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_hdrzone(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_imidazole(molecule):
    '''
    Compute the fr_imidazole descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_imidazole value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_imidazole(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_imide(molecule):
    '''
    Compute the fr_imide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_imide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_imide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_isocyan(molecule):
    '''
    Compute the fr_isocyan descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_isocyan value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_isocyan(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_isothiocyan(molecule):
    '''
    Compute the fr_isothiocyan descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_isothiocyan value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_isothiocyan(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_ketone(molecule):
    '''
    Compute the fr_ketone descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_ketone value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_ketone(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_ketone_Topliss(molecule):
    '''
    Compute the fr_ketone_Topliss descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_ketone_Topliss value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_ketone_Topliss(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_lactam(molecule):
    '''
    Compute the fr_lactam descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_lactam value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_lactam(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_lactone(molecule):
    '''
    Compute the fr_lactone descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_lactone value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_lactone(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_methoxy(molecule):
    '''
    Compute the fr_methoxy descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_methoxy value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_methoxy(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_morpholine(molecule):
    '''
    Compute the fr_morpholine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_morpholine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_morpholine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_nitrile(molecule):
    '''
    Compute the fr_nitrile descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_nitrile value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_nitrile(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_nitro(molecule):
    '''
    Compute the fr_nitro descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_nitro value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_nitro(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_nitro_arom(molecule):
    '''
    Compute the fr_nitro_arom descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_nitro_arom value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_nitro_arom(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_nitro_arom_nonortho(molecule):
    '''
    Compute the fr_nitro_arom_nonortho descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_nitro_arom_nonortho value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_nitro_arom_nonortho(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_nitroso(molecule):
    '''
    Compute the fr_nitroso descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_nitroso value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_nitroso(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_oxazole(molecule):
    '''
    Compute the fr_oxazole descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_oxazole value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_oxazole(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_oxime(molecule):
    '''
    Compute the fr_oxime descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_oxime value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_oxime(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_para_hydroxylation(molecule):
    '''
    Compute the fr_para_hydroxylation descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_para_hydroxylation value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_para_hydroxylation(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_phenol(molecule):
    '''
    Compute the fr_phenol descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_phenol value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_phenol(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_phenol_noOrthoHbond(molecule):
    '''
    Compute the fr_phenol_noOrthoHbond descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_phenol_noOrthoHbond value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_phenol_noOrthoHbond(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_phos_acid(molecule):
    '''
    Compute the fr_phos_acid descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_phos_acid value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_phos_acid(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_phos_ester(molecule):
    '''
    Compute the fr_phos_ester descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_phos_ester value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_phos_ester(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_piperdine(molecule):
    '''
    Compute the fr_piperdine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_piperdine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_piperdine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_piperzine(molecule):
    '''
    Compute the fr_piperzine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_piperzine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_piperzine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_priamide(molecule):
    '''
    Compute the fr_priamide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_priamide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_priamide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_prisulfonamd(molecule):
    '''
    Compute the fr_prisulfonamd descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_prisulfonamd value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_prisulfonamd(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_pyridine(molecule):
    '''
    Compute the fr_pyridine descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_pyridine value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_pyridine(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_quatN(molecule):
    '''
    Compute the fr_quatN descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_quatN value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_quatN(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_sulfide(molecule):
    '''
    Compute the fr_sulfide descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_sulfide value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_sulfide(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_sulfonamd(molecule):
    '''
    Compute the fr_sulfonamd descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_sulfonamd value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_sulfonamd(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_sulfone(molecule):
    '''
    Compute the fr_sulfone descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_sulfone value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_sulfone(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_term_acetylene(molecule):
    '''
    Compute the fr_term_acetylene descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_term_acetylene value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_term_acetylene(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_tetrazole(molecule):
    '''
    Compute the fr_tetrazole descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_tetrazole value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_tetrazole(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_thiazole(molecule):
    '''
    Compute the fr_thiazole descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_thiazole value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_thiazole(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_thiocyan(molecule):
    '''
    Compute the fr_thiocyan descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The fr_thiocyan value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_thiocyan(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_thiophene(molecule):
    '''
    Compute the fr_thiophene descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The fr_thiophene value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_thiophene(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_unbrch_alkane(molecule):
    '''
    Compute the fr_unbrch_alkane descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The fr_unbrch_alkane value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_unbrch_alkane(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findfr_urea(molecule):
    '''
    Compute the fr_urea descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The fr_urea value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.fr_urea(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findFractionCSP3(molecule):
    '''
    Compute the FractionCSP3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The FractionCSP3 descriptor.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FractionCSP3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findHallKierAlpha(molecule):
    '''
    Compute the HallKierAlpha descriptor.
    Input:
      -
    Return:
      [float] - The HallKierAlpha descriptor.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.HallKierAlpha(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findHeavyAtomMolWt(molecule):
    '''
    Compute the heavy atom molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The heavy atom molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.HeavyAtomMolWt(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findHeavyAtomCount(molecule):
    '''
    Compute the HeavyAtomCount descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The HeavyAtomCount descriptor.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.HeavyAtomCount(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findIpc(molecule):
    '''
    Compute the Ipc descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Ipc descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Ipc(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region Kappa descriptors
def findKappa1(molecule):
    '''
    Compute the Kappa1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Kappa1 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Kappa1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findKappa2(molecule):
    '''
    Compute the Kappa2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Kappa2 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Kappa2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findKappa3(molecule):
    '''
    Compute the Kappa3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Kappa3 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Kappa3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findLabuteASA(molecule):
    '''
    Compute the LabuteASA descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The LabuteASA descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.LabuteASA(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMaxAbsPartialCharge(molecule):
    '''
    Compute the maximum absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The maximum absolute partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxAbsPartialCharge(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMaxPartialCharge(molecule):
    '''
    Compute the absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The absolute partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxPartialCharge(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMinAbsPartialCharge(molecule):
    '''
    Compute the minimum absolute partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The minimum absolute partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinAbsPartialCharge(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMinPartialCharge(molecule):
    '''
    Compute the minimum partial charge of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The minimum partial partial charge.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinPartialCharge(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMolLogP(molecule):
    '''
    Compute the MolLogP descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MolLogP descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MolLogP(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMolMR(molecule):
    '''
    Compute the MolMR descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The MolMR descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MolMR(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findMolWt(molecule):
    '''
    Compute the molecular weight of the molecule.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The molecular weight.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MolWt(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region 'count' descriptors
def findNHOHCount(molecule):
    '''
    Compute the NHOHCount descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The NHOHCount descriptor.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NHOHCount(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNOCount(molecule):
    '''
    Compute the NOCount descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The NOCount descriptor.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NOCount(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAliphaticCarbocycles(molecule):
    '''
    Compute the NumAliphaticCarbocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAliphaticCarbocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAliphaticCarbocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAliphaticHeterocycles(molecule):
    '''
    Compute the NumAliphaticHeterocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAliphaticHeterocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAliphaticHeterocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAliphaticRings(molecule):
    '''
    Compute the NumAliphaticRings descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAliphaticRings value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAliphaticRings(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAromaticCarbocycles(molecule):
    '''
    Compute the NumAromaticCarbocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAromaticCarbocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAromaticCarbocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAromaticHeterocycles(molecule):
    '''
    Compute the NumAromaticHeterocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAromaticHeterocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAromaticHeterocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumAromaticRings(molecule):
    '''
    Compute the NumAromaticRings descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumAromaticRings value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumAromaticRings(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumHAcceptors(molecule):
    '''
    Compute the NumHAcceptors descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumHAcceptors value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumHAcceptors(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumHDonors(molecule):
    '''
    Compute the NumHDonors descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumHDonors value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumHDonors(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumHeteroatoms(molecule):
    '''
    Compute the NumHeteroatoms descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumHeteroatoms value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumHeteroatoms(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumRadicalElectrons(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumRotatableBonds(molecule):
    '''
    Compute the NumRotatableBonds descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumRotatableBonds value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumRotatableBonds(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumSaturatedCarbocycles(molecule):
    '''
    Compute the NumSaturatedCarbocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumSaturatedCarbocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumSaturatedCarbocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumSaturatedHeterocycles(molecule):
    '''
    Compute the NumSaturatedHeterocycles descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumSaturatedHeterocycles value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumSaturatedHeterocycles(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNumSaturatedRings(molecule):
    '''
    Compute the NumSaturatedRings descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NumSaturatedRings value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumSaturatedRings(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.NumValenceElectrons(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findRingCount(molecule):
    '''
    Compute the RingCount descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [int]  - The RingCount value.
      [None] - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.RingCount(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

#region PEOE_VSA descriptors
def findPEOE_VSA1(molecule):
    '''
    Compute the PEOE_VSA1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA2(molecule):
    '''
    Compute the PEOE_VSA2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA3(molecule):
    '''
    Compute the PEOE_VSA3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA4(molecule):
    '''
    Compute the PEOE_VSA4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA5(molecule):
    '''
    Compute the PEOE_VSA5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA6(molecule):
    '''
    Compute the PEOE_VSA6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA7(molecule):
    '''
    Compute the PEOE_VSA7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA8(molecule):
    '''
    Compute the PEOE_VSA8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA9(molecule):
    '''
    Compute the PEOE_VSA9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA10(molecule):
    '''
    Compute the PEOE_VSA10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA11(molecule):
    '''
    Compute the PEOE_VSA11 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA11 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA11(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA12(molecule):
    '''
    Compute the PEOE_VSA12 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA12 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA12(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA13(molecule):
    '''
    Compute the PEOE_VSA13 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA13 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA13(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPEOE_VSA14(molecule):
    '''
    Compute the PEOE_VSA14 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PEOE_VSA14 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.PEOE_VSA14(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findqed(molecule):
    '''
    Compute the qed descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The qed value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.qed(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region SMR_VSA descriptors
def findSMR_VSA1(molecule):
    '''
    Compute the SMR_VSA1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA2(molecule):
    '''
    Compute the SMR_VSA2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA3(molecule):
    '''
    Compute the SMR_VSA3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA4(molecule):
    '''
    Compute the SMR_VSA4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA5(molecule):
    '''
    Compute the SMR_VSA5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA6(molecule):
    '''
    Compute the SMR_VSA6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA7(molecule):
    '''
    Compute the SMR_VSA7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA8(molecule):
    '''
    Compute the SMR_VSA8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA9(molecule):
    '''
    Compute the SMR_VSA9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSMR_VSA10(molecule):
    '''
    Compute the SMR_VSA10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SMR_VSA10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SMR_VSA10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

#region SlogP_VSA descriptors
def findSlogP_VSA1(molecule):
    '''
    Compute the SlogP_VSA1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA2(molecule):
    '''
    Compute the SlogP_VSA2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA3(molecule):
    '''
    Compute the SlogP_VSA3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA4(molecule):
    '''
    Compute the SlogP_VSA4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA5(molecule):
    '''
    Compute the SlogP_VSA5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA6(molecule):
    '''
    Compute the SlogP_VSA6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA7(molecule):
    '''
    Compute the SlogP_VSA7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA8(molecule):
    '''
    Compute the SlogP_VSA8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA9(molecule):
    '''
    Compute the SlogP_VSA9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA10(molecule):
    '''
    Compute the SlogP_VSA10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA11(molecule):
    '''
    Compute the SlogP_VSA11 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA11 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA11(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSlogP_VSA12(molecule):
    '''
    Compute the SlogP_VSA12 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SlogP_VSA12 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.SlogP_VSA12(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

def findTPSA(molecule):
    '''
    Compute the TPSA descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The TPSA value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.TPSA(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#region VSA_EState descriptors
def findVSA_EState1(molecule):
    '''
    Compute the VSA_EState1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState2(molecule):
    '''
    Compute the VSA_EState2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState3(molecule):
    '''
    Compute the VSA_EState3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState4(molecule):
    '''
    Compute the VSA_EState4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState4(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState5(molecule):
    '''
    Compute the VSA_EState5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState5(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState6(molecule):
    '''
    Compute the VSA_EState6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState6(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState7(molecule):
    '''
    Compute the VSA_EState7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState7(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState8(molecule):
    '''
    Compute the VSA_EState8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState8(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState9(molecule):
    '''
    Compute the VSA_EState9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState9(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findVSA_EState10(molecule):
    '''
    Compute the VSA_EState10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The VSA_EState10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.VSA_EState10(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion

#region 3D descriptors
def findAUTOCORR3D(molecule):
    '''
    Compute the AUTOCORR3D descriptors.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The AUTOCORR3D values.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.rdMolDescriptors.CalcAUTOCORR3D(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findAsphericity(molecule):
    '''
    Compute the Asphericity descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Asphericity value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.Asphericity(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findEccentricity(molecule):
    '''
    Compute the Eccentricity descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The Eccentricity value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.Eccentricity(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findInertialShapeFactor(molecule):
    '''
    Compute the InertialShapeFactor descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The InertialShapeFactor value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.InertialShapeFactor(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNPR1(molecule):
    '''
    Compute the NPR1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NPR1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.NPR1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findNPR2(molecule):
    '''
    Compute the NPR2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The NPR2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.NPR2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPMI1(molecule):
    '''
    Compute the PMI1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PMI1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.PMI1(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPMI2(molecule):
    '''
    Compute the PMI2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PMI2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.PMI2(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findPMI3(molecule):
    '''
    Compute the PMI3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The PMI3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.PMI3(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findRadiusOfGyration(molecule):
    '''
    Compute the RadiusOfGyration descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The RadiusOfGyration value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.RadiusOfGyration(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

def findSpherocityIndex(molecule):
    '''
    Compute the SpherocityIndex descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [float] - The SpherocityIndex value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors3D.SpherocityIndex(molecule)
        _ = errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = errors.not_set(f"The variable is not set.")
    return None

#endregion
