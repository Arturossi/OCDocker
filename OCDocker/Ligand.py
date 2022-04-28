#!/usr/lib/python3

# Imports
###############################################################################
import os
import json
import rdkit
from glob import glob

from rdkit import Chem
from rdkit.Chem import Descriptors
from openbabel import openbabel
from rdkit import RDLogger

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
        self.path, self.molecule = self.__loadMol(molecule, sanitize)
        # Define everything as None
        self.name = None

        # <editor-fold> AUTOCORR descriptors
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
        # </editor-fold>

        # <editor-fold> BCUT2D descriptors
        self.BCUT2D_CHGHI = None
        self.BCUT2D_CHGLO = None
        self.BCUT2D_LOGPHI = None
        self.BCUT2D_LOGPLOW = None
        self.BCUT2D_MRHI = None
        self.BCUT2D_MRLOW = None
        self.BCUT2D_MWHI = None
        self.BCUT2D_MWLOW = None
        # </editor-fold>

        self.BalabanJ = None
        self.BertzCT = None

        # <editor-fold> Chi descriptors
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
        # </editor-fold>

        # <editor-fold> EState descriptors
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
        # </editor-fold>

        self.ExactMolWt = None

        # <editor-fold> fr_ descriptors
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
        # </editor-fold>

        self.FractionCSP3 = None
        self.FpDensityMorgan1 = None
        self.FpDensityMorgan2 = None
        self.FpDensityMorgan3 = None
        self.HallKierAlpha = None
        self.HeavyAtomMolWt = None
        self.HeavyAtomCount = None
        self.Ipc = None

        # <editor-fold> Kappa descriptors
        self.Kappa1 = None
        self.Kappa2 = None
        self.Kappa3 = None
        # </editor-fold>

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

        # <editor-fold> 'count' descriptors
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
        # </editor-fold>

        # <editor-fold> PEOE_VSA descriptors
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
        # </editor-fold>

        self.PropertyFunctor = None
        self.qed = None
        self.RingCount = None

        # <editor-fold> SMR_VSA descriptors
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
        # </editor-fold>

        # <editor-fold> SlogP_VSA descriptors
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
        # </editor-fold>

        self.TPSA = None

        # <editor-fold> VSA_EState descriptors
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
        # </editor-fold>

        # If user pass a json
        if from_json_descriptors:
            # Read the descriptors from it
            data = self.__read_descriptors_from_json(from_json_descriptors)
            # If data is None, a problem occurred while reading the json file
            if not data:
                octools.print_error(f"Problems while parsin json file: '{from_json_descriptors}'")
                return None
            self.name, self.ExactMolWt, self.FpDensityMorgan1, self.FpDensityMorgan2, self.FpDensityMorgan3, self.HeavyAtomMolWt, self.MaxAbsPartialCharge, self.MaxPartialCharge, self.MinAbsPartialCharge, self.MinPartialCharge, self.MolWt, self.NumRadicalElectrons, self.NumValenceElectrons = data
        else:
            # Check if the name is empty
            if not name:
                octools.print_error("The Ligand name should not be empty!")
                return None
            self.name = name.replace(" ", "_")

            # <editor-fold> AUTOCORR descriptors
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
            # </editor-fold>

            # <editor-fold> BCUT2D descriptors
            self.BCUT2D_CHGHI = self.__findBCUT2D_CHGHI()
            self.BCUT2D_CHGLO = self.__findBCUT2D_CHGLO()
            self.BCUT2D_LOGPHI = self.__findBCUT2D_LOGPHI()
            self.BCUT2D_LOGPLOW = self.__findBCUT2D_LOGPLOW()
            self.BCUT2D_MRHI = self.__findBCUT2D_MRHI()
            self.BCUT2D_MRLOW = self.__findBCUT2D_MRLOW()
            self.BCUT2D_MWHI = self.__findBCUT2D_MWHI()
            self.BCUT2D_MWLOW = self.__findBCUT2D_MWLOW()
            # </editor-fold>

            self.BalabanJ = self.__findBalabanJ()
            self.BertzCT = self.__findBertzCT()

            # <editor-fold> Chi descriptors
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
            # </editor-fold>

            # <editor-fold> EState descriptors
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
            # </editor-fold>

            self.ExactMolWt = self.__findExactMolWt()

            # <editor-fold> fr_ descriptors
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
            # </editor-fold>

            self.FractionCSP3 = None
            self.FpDensityMorgan1 = self.__findFpDensityMorgan1()
            self.FpDensityMorgan2 = self.__findFpDensityMorgan2()
            self.FpDensityMorgan3 = self.__findFpDensityMorgan3()
            self.HallKierAlpha = None
            self.HeavyAtomMolWt = self.__findHeavyAtomMolWt()
            self.HeavyAtomCount = None
            self.Ipc = None

            # <editor-fold> Kappa descriptors
            self.Kappa1 = None
            self.Kappa2 = None
            self.Kappa3 = None
            # </editor-fold>

            self.LabuteASA = None
            self.MaxAbsPartialCharge = self.__findMaxAbsPartialCharge()
            self.MaxPartialCharge = self.__findMaxPartialCharge()
            self.MinAbsPartialCharge = self.__findMinAbsPartialCharge()
            self.MinPartialCharge = self.__findMinPartialCharge()
            self.MolLogP = None
            self.MolMR = None
            self.MolWt = self.__findMolWt()
            self.NHOHCount = None
            self.NOCount = None

            # <editor-fold> 'count' descriptors
            self.NumAliphaticCarbocycles = None
            self.NumAliphaticHeterocycles = None
            self.NumAliphaticRings = None
            self.NumAromaticCarbocycles = None
            self.NumAromaticHeterocycles = None
            self.NumAromaticRings = None
            self.NumHAcceptors = None
            self.NumHDonors = None
            self.NumHeteroatoms = None
            self.NumRadicalElectrons = self.__findNumRadicalElectrons()
            self.NumRotatableBonds = None
            self.NumSaturatedCarbocycles = None
            self.NumSaturatedHeterocycles = None
            self.NumSaturatedRings = None
            self.NumValenceElectrons = self.__findNumValenceElectrons()
            # </editor-fold>

            # <editor-fold> PEOE_VSA descriptors
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
            # </editor-fold>

            self.PropertyFunctor = None
            self.qed = None
            self.RingCount = None

            # <editor-fold> SMR_VSA descriptors
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
            # </editor-fold>

            # <editor-fold> SlogP_VSA descriptors
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
            # </editor-fold>

            self.TPSA = None

            # <editor-fold> VSA_EState descriptors
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
            # </editor-fold>

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

    # <editor-fold> AUTOCORR descriptors
    def __findAUTOCORR2D_1(self):
        '''
        Compute the autocorrelation2D_1 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_1 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_1(self.molecule)

    def __findAUTOCORR2D_2(self):
        '''
        Compute the autocorrelation2D_2 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_2 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_2(self.molecule)

    def __findAUTOCORR2D_3(self):
        '''
        Compute the autocorrelation2D_3 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_3 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_3(self.molecule)

    def __findAUTOCORR2D_4(self):
        '''
        Compute the autocorrelation2D_4 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_4 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_4(self.molecule)

    def __findAUTOCORR2D_5(self):
        '''
        Compute the autocorrelation2D_5 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_5 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_5(self.molecule)

    def __findAUTOCORR2D_6(self):
        '''
        Compute the autocorrelation2D_6 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_6 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_6(self.molecule)

    def __findAUTOCORR2D_7(self):
        '''
        Compute the autocorrelation2D_7 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_7 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_7(self.molecule)

    def __findAUTOCORR2D_8(self):
        '''
        Compute the autocorrelation2D_8 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_8 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_8(self.molecule)

    def __findAUTOCORR2D_9(self):
        '''
        Compute the autocorrelation2D_9 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_9 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_9(self.molecule)

    def __findAUTOCORR2D_10(self):
        '''
        Compute the autocorrelation2D_10 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_10 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_10(self.molecule)

    def __findAUTOCORR2D_11(self):
        '''
        Compute the autocorrelation2D_11 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_11 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_11(self.molecule)

    def __findAUTOCORR2D_12(self):
        '''
        Compute the autocorrelation2D_12 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_12 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_12(self.molecule)

    def __findAUTOCORR2D_13(self):
        '''
        Compute the autocorrelation2D_13 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_13 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_13(self.molecule)

    def __findAUTOCORR2D_14(self):
        '''
        Compute the autocorrelation2D_14 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_14 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_14(self.molecule)

    def __findAUTOCORR2D_15(self):
        '''
        Compute the autocorrelation2D_15 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_15 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_15(self.molecule)

    def __findAUTOCORR2D_16(self):
        '''
        Compute the autocorrelation2D_16 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_16 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_16(self.molecule)

    def __findAUTOCORR2D_17(self):
        '''
        Compute the autocorrelation2D_17 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_17 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_17(self.molecule)

    def __findAUTOCORR2D_18(self):
        '''
        Compute the autocorrelation2D_18 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_18 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_18(self.molecule)

    def __findAUTOCORR2D_19(self):
        '''
        Compute the autocorrelation2D_19 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_19 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_19(self.molecule)

    def __findAUTOCORR2D_20(self):
        '''
        Compute the autocorrelation2D_20 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_20 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_20(self.molecule)

    def __findAUTOCORR2D_21(self):
        '''
        Compute the autocorrelation2D_21 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_21 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_21(self.molecule)

    def __findAUTOCORR2D_22(self):
        '''
        Compute the autocorrelation2D_22 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_22 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_22(self.molecule)

    def __findAUTOCORR2D_23(self):
        '''
        Compute the autocorrelation2D_23 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_23 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_23(self.molecule)

    def __findAUTOCORR2D_24(self):
        '''
        Compute the autocorrelation2D_24 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_24 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_24(self.molecule)

    def __findAUTOCORR2D_25(self):
        '''
        Compute the autocorrelation2D_25 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_25 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_25(self.molecule)

    def __findAUTOCORR2D_26(self):
        '''
        Compute the autocorrelation2D_26 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_26 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_26(self.molecule)

    def __findAUTOCORR2D_27(self):
        '''
        Compute the autocorrelation2D_27 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_27 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_27(self.molecule)

    def __findAUTOCORR2D_28(self):
        '''
        Compute the autocorrelation2D_28 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_28 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_28(self.molecule)

    def __findAUTOCORR2D_29(self):
        '''
        Compute the autocorrelation2D_29 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_29 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_29(self.molecule)

    def __findAUTOCORR2D_30(self):
        '''
        Compute the autocorrelation2D_30 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_30 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_30(self.molecule)

    def __findAUTOCORR2D_31(self):
        '''
        Compute the autocorrelation2D_31 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_31 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_31(self.molecule)

    def __findAUTOCORR2D_32(self):
        '''
        Compute the autocorrelation2D_32 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_32 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_32(self.molecule)

    def __findAUTOCORR2D_33(self):
        '''
        Compute the autocorrelation2D_33 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_33 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_33(self.molecule)

    def __findAUTOCORR2D_34(self):
        '''
        Compute the autocorrelation2D_34 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_34 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_34(self.molecule)

    def __findAUTOCORR2D_35(self):
        '''
        Compute the autocorrelation2D_35 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_35 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_35(self.molecule)

    def __findAUTOCORR2D_36(self):
        '''
        Compute the autocorrelation2D_36 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_36 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_36(self.molecule)

    def __findAUTOCORR2D_37(self):
        '''
        Compute the autocorrelation2D_37 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_37 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_37(self.molecule)

    def __findAUTOCORR2D_38(self):
        '''
        Compute the autocorrelation2D_38 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_38 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_38(self.molecule)

    def __findAUTOCORR2D_39(self):
        '''
        Compute the autocorrelation2D_39 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_39 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_39(self.molecule)

    def __findAUTOCORR2D_40(self):
        '''
        Compute the autocorrelation2D_40 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_40 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_40(self.molecule)

    def __findAUTOCORR2D_41(self):
        '''
        Compute the autocorrelation2D_41 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_41 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_41(self.molecule)

    def __findAUTOCORR2D_42(self):
        '''
        Compute the autocorrelation2D_42 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_42 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_42(self.molecule)

    def __findAUTOCORR2D_43(self):
        '''
        Compute the autocorrelation2D_43 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_43 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_43(self.molecule)

    def __findAUTOCORR2D_44(self):
        '''
        Compute the autocorrelation2D_44 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_44 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_44(self.molecule)

    def __findAUTOCORR2D_45(self):
        '''
        Compute the autocorrelation2D_45 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_45 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_45(self.molecule)

    def __findAUTOCORR2D_46(self):
        '''
        Compute the autocorrelation2D_46 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_46 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_46(self.molecule)

    def __findAUTOCORR2D_47(self):
        '''
        Compute the autocorrelation2D_47 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_47 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_47(self.molecule)

    def __findAUTOCORR2D_48(self):
        '''
        Compute the autocorrelation2D_48 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_48 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_48(self.molecule)

    def __findAUTOCORR2D_49(self):
        '''
        Compute the autocorrelation2D_49 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_49 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_49(self.molecule)

    def __findAUTOCORR2D_50(self):
        '''
        Compute the autocorrelation2D_50 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_50 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_50(self.molecule)

    def __findAUTOCORR2D_51(self):
        '''
        Compute the autocorrelation2D_51 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_51 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_51(self.molecule)

    def __findAUTOCORR2D_52(self):
        '''
        Compute the autocorrelation2D_52 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_52 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_52(self.molecule)

    def __findAUTOCORR2D_53(self):
        '''
        Compute the autocorrelation2D_53 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_53 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_53(self.molecule)

    def __findAUTOCORR2D_54(self):
        '''
        Compute the autocorrelation2D_54 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_54 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_54(self.molecule)

    def __findAUTOCORR2D_55(self):
        '''
        Compute the autocorrelation2D_55 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_55 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_55(self.molecule)

    def __findAUTOCORR2D_56(self):
        '''
        Compute the autocorrelation2D_56 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_56 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_56(self.molecule)

    def __findAUTOCORR2D_57(self):
        '''
        Compute the autocorrelation2D_57 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_57 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_57(self.molecule)

    def __findAUTOCORR2D_58(self):
        '''
        Compute the autocorrelation2D_58 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_58 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_58(self.molecule)

    def __findAUTOCORR2D_59(self):
        '''
        Compute the autocorrelation2D_59 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_59 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_59(self.molecule)

    def __findAUTOCORR2D_60(self):
        '''
        Compute the autocorrelation2D_60 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_60 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_60(self.molecule)

    def __findAUTOCORR2D_61(self):
        '''
        Compute the autocorrelation2D_61 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_61 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_61(self.molecule)

    def __findAUTOCORR2D_62(self):
        '''
        Compute the autocorrelation2D_62 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_62 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_62(self.molecule)

    def __findAUTOCORR2D_63(self):
        '''
        Compute the autocorrelation2D_63 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_63 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_63(self.molecule)

    def __findAUTOCORR2D_64(self):
        '''
        Compute the autocorrelation2D_64 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_64 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_64(self.molecule)

    def __findAUTOCORR2D_65(self):
        '''
        Compute the autocorrelation2D_65 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_65 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_65(self.molecule)

    def __findAUTOCORR2D_66(self):
        '''
        Compute the autocorrelation2D_66 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_66 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_66(self.molecule)

    def __findAUTOCORR2D_67(self):
        '''
        Compute the autocorrelation2D_67 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_67 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_67(self.molecule)

    def __findAUTOCORR2D_68(self):
        '''
        Compute the autocorrelation2D_68 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_68 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_68(self.molecule)

    def __findAUTOCORR2D_69(self):
        '''
        Compute the autocorrelation2D_69 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_69 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_69(self.molecule)

    def __findAUTOCORR2D_70(self):
        '''
        Compute the autocorrelation2D_70 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_70 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_70(self.molecule)

    def __findAUTOCORR2D_71(self):
        '''
        Compute the autocorrelation2D_71 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_71 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_71(self.molecule)

    def __findAUTOCORR2D_72(self):
        '''
        Compute the autocorrelation2D_72 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_72 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_72(self.molecule)

    def __findAUTOCORR2D_73(self):
        '''
        Compute the autocorrelation2D_73 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_73 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_73(self.molecule)

    def __findAUTOCORR2D_74(self):
        '''
        Compute the autocorrelation2D_74 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_74 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_74(self.molecule)

    def __findAUTOCORR2D_75(self):
        '''
        Compute the autocorrelation2D_75 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_75 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_75(self.molecule)

    def __findAUTOCORR2D_76(self):
        '''
        Compute the autocorrelation2D_76 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_76 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_76(self.molecule)

    def __findAUTOCORR2D_77(self):
        '''
        Compute the autocorrelation2D_77 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_77 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_77(self.molecule)

    def __findAUTOCORR2D_78(self):
        '''
        Compute the autocorrelation2D_78 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_78 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_78(self.molecule)

    def __findAUTOCORR2D_79(self):
        '''
        Compute the autocorrelation2D_79 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_79 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_79(self.molecule)

    def __findAUTOCORR2D_80(self):
        '''
        Compute the autocorrelation2D_80 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_80 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_80(self.molecule)

    def __findAUTOCORR2D_81(self):
        '''
        Compute the autocorrelation2D_81 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_81 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_81(self.molecule)

    def __findAUTOCORR2D_82(self):
        '''
        Compute the autocorrelation2D_82 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_82 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_82(self.molecule)

    def __findAUTOCORR2D_83(self):
        '''
        Compute the autocorrelation2D_83 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_83 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_83(self.molecule)

    def __findAUTOCORR2D_84(self):
        '''
        Compute the autocorrelation2D_84 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_84 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_84(self.molecule)

    def __findAUTOCORR2D_85(self):
        '''
        Compute the autocorrelation2D_85 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_85 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_85(self.molecule)

    def __findAUTOCORR2D_86(self):
        '''
        Compute the autocorrelation2D_86 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_86 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_86(self.molecule)

    def __findAUTOCORR2D_87(self):
        '''
        Compute the autocorrelation2D_87 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_87 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_87(self.molecule)

    def __findAUTOCORR2D_88(self):
        '''
        Compute the autocorrelation2D_88 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_88 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_88(self.molecule)

    def __findAUTOCORR2D_89(self):
        '''
        Compute the autocorrelation2D_89 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_89 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_89(self.molecule)

    def __findAUTOCORR2D_90(self):
        '''
        Compute the autocorrelation2D_90 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_90 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_90(self.molecule)

    def __findAUTOCORR2D_91(self):
        '''
        Compute the autocorrelation2D_91 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_91 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_91(self.molecule)

    def __findAUTOCORR2D_92(self):
        '''
        Compute the autocorrelation2D_92 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_92 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_92(self.molecule)

    def __findAUTOCORR2D_93(self):
        '''
        Compute the autocorrelation2D_93 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_93 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_93(self.molecule)

    def __findAUTOCORR2D_94(self):
        '''
        Compute the autocorrelation2D_94 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_94 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_94(self.molecule)

    def __findAUTOCORR2D_95(self):
        '''
        Compute the autocorrelation2D_95 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_95 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_95(self.molecule)

    def __findAUTOCORR2D_96(self):
        '''
        Compute the autocorrelation2D_96 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_96 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_96(self.molecule)

    def __findAUTOCORR2D_97(self):
        '''
        Compute the autocorrelation2D_97 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_97 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_97(self.molecule)

    def __findAUTOCORR2D_98(self):
        '''
        Compute the autocorrelation2D_98 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_98 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_98(self.molecule)

    def __findAUTOCORR2D_99(self):
        '''
        Compute the autocorrelation2D_99 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_99 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_99(self.molecule)

    def __findAUTOCORR2D_100(self):
        '''
        Compute the autocorrelation2D_100 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_100 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_100(self.molecule)

    def __findAUTOCORR2D_101(self):
        '''
        Compute the autocorrelation2D_101 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_101 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_101(self.molecule)

    def __findAUTOCORR2D_102(self):
        '''
        Compute the autocorrelation2D_102 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_102 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_102(self.molecule)

    def __findAUTOCORR2D_103(self):
        '''
        Compute the autocorrelation2D_103 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_103 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_103(self.molecule)

    def __findAUTOCORR2D_104(self):
        '''
        Compute the autocorrelation2D_104 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_104 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_104(self.molecule)

    def __findAUTOCORR2D_105(self):
        '''
        Compute the autocorrelation2D_105 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_105 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_105(self.molecule)

    def __findAUTOCORR2D_106(self):
        '''
        Compute the autocorrelation2D_106 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_106 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_106(self.molecule)

    def __findAUTOCORR2D_107(self):
        '''
        Compute the autocorrelation2D_107 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_107 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_107(self.molecule)

    def __findAUTOCORR2D_108(self):
        '''
        Compute the autocorrelation2D_108 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_108 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_108(self.molecule)

    def __findAUTOCORR2D_109(self):
        '''
        Compute the autocorrelation2D_109 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_109 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_109(self.molecule)

    def __findAUTOCORR2D_110(self):
        '''
        Compute the autocorrelation2D_110 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_110 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_110(self.molecule)

    def __findAUTOCORR2D_111(self):
        '''
        Compute the autocorrelation2D_111 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_111 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_111(self.molecule)

    def __findAUTOCORR2D_112(self):
        '''
        Compute the autocorrelation2D_112 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_112 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_112(self.molecule)

    def __findAUTOCORR2D_113(self):
        '''
        Compute the autocorrelation2D_113 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_113 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_113(self.molecule)

    def __findAUTOCORR2D_114(self):
        '''
        Compute the autocorrelation2D_114 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_114 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_114(self.molecule)

    def __findAUTOCORR2D_115(self):
        '''
        Compute the autocorrelation2D_115 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_115 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_115(self.molecule)

    def __findAUTOCORR2D_116(self):
        '''
        Compute the autocorrelation2D_116 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_116 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_116(self.molecule)

    def __findAUTOCORR2D_117(self):
        '''
        Compute the autocorrelation2D_117 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_117 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_117(self.molecule)

    def __findAUTOCORR2D_118(self):
        '''
        Compute the autocorrelation2D_118 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_118 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_118(self.molecule)

    def __findAUTOCORR2D_119(self):
        '''
        Compute the autocorrelation2D_119 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_119 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_119(self.molecule)

    def __findAUTOCORR2D_120(self):
        '''
        Compute the autocorrelation2D_120 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_120 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_120(self.molecule)

    def __findAUTOCORR2D_121(self):
        '''
        Compute the autocorrelation2D_121 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_121 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_121(self.molecule)

    def __findAUTOCORR2D_122(self):
        '''
        Compute the autocorrelation2D_122 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_122 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_122(self.molecule)

    def __findAUTOCORR2D_123(self):
        '''
        Compute the autocorrelation2D_123 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_123 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_123(self.molecule)

    def __findAUTOCORR2D_124(self):
        '''
        Compute the autocorrelation2D_124 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_124 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_124(self.molecule)

    def __findAUTOCORR2D_125(self):
        '''
        Compute the autocorrelation2D_125 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_125 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_125(self.molecule)

    def __findAUTOCORR2D_126(self):
        '''
        Compute the autocorrelation2D_126 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_126 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_126(self.molecule)

    def __findAUTOCORR2D_127(self):
        '''
        Compute the autocorrelation2D_127 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_127 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_127(self.molecule)

    def __findAUTOCORR2D_128(self):
        '''
        Compute the autocorrelation2D_128 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_128 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_128(self.molecule)

    def __findAUTOCORR2D_129(self):
        '''
        Compute the autocorrelation2D_129 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_129 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_129(self.molecule)

    def __findAUTOCORR2D_130(self):
        '''
        Compute the autocorrelation2D_130 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_130 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_130(self.molecule)

    def __findAUTOCORR2D_131(self):
        '''
        Compute the autocorrelation2D_131 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_131 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_131(self.molecule)

    def __findAUTOCORR2D_132(self):
        '''
        Compute the autocorrelation2D_132 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_132 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_132(self.molecule)

    def __findAUTOCORR2D_133(self):
        '''
        Compute the autocorrelation2D_133 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_133 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_133(self.molecule)

    def __findAUTOCORR2D_134(self):
        '''
        Compute the autocorrelation2D_134 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_134 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_134(self.molecule)

    def __findAUTOCORR2D_135(self):
        '''
        Compute the autocorrelation2D_135 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_135 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_135(self.molecule)

    def __findAUTOCORR2D_136(self):
        '''
        Compute the autocorrelation2D_136 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_136 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_136(self.molecule)

    def __findAUTOCORR2D_137(self):
        '''
        Compute the autocorrelation2D_137 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_137 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_137(self.molecule)

    def __findAUTOCORR2D_138(self):
        '''
        Compute the autocorrelation2D_138 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_138 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_138(self.molecule)

    def __findAUTOCORR2D_139(self):
        '''
        Compute the autocorrelation2D_139 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_139 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_139(self.molecule)

    def __findAUTOCORR2D_140(self):
        '''
        Compute the autocorrelation2D_140 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_140 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_140(self.molecule)

    def __findAUTOCORR2D_141(self):
        '''
        Compute the autocorrelation2D_141 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_141 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_141(self.molecule)

    def __findAUTOCORR2D_142(self):
        '''
        Compute the autocorrelation2D_142 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_142 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_142(self.molecule)

    def __findAUTOCORR2D_143(self):
        '''
        Compute the autocorrelation2D_143 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_143 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_143(self.molecule)

    def __findAUTOCORR2D_144(self):
        '''
        Compute the autocorrelation2D_144 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_144 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_144(self.molecule)

    def __findAUTOCORR2D_145(self):
        '''
        Compute the autocorrelation2D_145 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_145 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_145(self.molecule)

    def __findAUTOCORR2D_146(self):
        '''
        Compute the autocorrelation2D_146 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_146 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_146(self.molecule)

    def __findAUTOCORR2D_147(self):
        '''
        Compute the autocorrelation2D_147 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_147 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_147(self.molecule)

    def __findAUTOCORR2D_148(self):
        '''
        Compute the autocorrelation2D_148 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_148 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_148(self.molecule)

    def __findAUTOCORR2D_149(self):
        '''
        Compute the autocorrelation2D_149 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_149 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_149(self.molecule)

    def __findAUTOCORR2D_150(self):
        '''
        Compute the autocorrelation2D_150 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_150 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_150(self.molecule)

    def __findAUTOCORR2D_151(self):
        '''
        Compute the autocorrelation2D_151 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_151 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_151(self.molecule)

    def __findAUTOCORR2D_152(self):
        '''
        Compute the autocorrelation2D_152 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_152 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_152(self.molecule)

    def __findAUTOCORR2D_153(self):
        '''
        Compute the autocorrelation2D_153 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_153 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_153(self.molecule)

    def __findAUTOCORR2D_154(self):
        '''
        Compute the autocorrelation2D_154 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_154 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_154(self.molecule)

    def __findAUTOCORR2D_155(self):
        '''
        Compute the autocorrelation2D_155 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_155 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_155(self.molecule)

    def __findAUTOCORR2D_156(self):
        '''
        Compute the autocorrelation2D_156 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_156 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_156(self.molecule)

    def __findAUTOCORR2D_157(self):
        '''
        Compute the autocorrelation2D_157 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_157 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_157(self.molecule)

    def __findAUTOCORR2D_158(self):
        '''
        Compute the autocorrelation2D_158 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_158 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_158(self.molecule)

    def __findAUTOCORR2D_159(self):
        '''
        Compute the autocorrelation2D_159 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_159 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_159(self.molecule)

    def __findAUTOCORR2D_160(self):
        '''
        Compute the autocorrelation2D_160 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_160 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_160(self.molecule)

    def __findAUTOCORR2D_161(self):
        '''
        Compute the autocorrelation2D_161 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_161 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_161(self.molecule)

    def __findAUTOCORR2D_162(self):
        '''
        Compute the autocorrelation2D_162 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_162 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_162(self.molecule)

    def __findAUTOCORR2D_163(self):
        '''
        Compute the autocorrelation2D_163 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_163 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_163(self.molecule)

    def __findAUTOCORR2D_164(self):
        '''
        Compute the autocorrelation2D_164 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_164 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_164(self.molecule)

    def __findAUTOCORR2D_165(self):
        '''
        Compute the autocorrelation2D_165 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_165 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_165(self.molecule)

    def __findAUTOCORR2D_166(self):
        '''
        Compute the autocorrelation2D_166 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_166 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_166(self.molecule)

    def __findAUTOCORR2D_167(self):
        '''
        Compute the autocorrelation2D_167 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_167 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_167(self.molecule)

    def __findAUTOCORR2D_168(self):
        '''
        Compute the autocorrelation2D_168 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_168 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_168(self.molecule)

    def __findAUTOCORR2D_169(self):
        '''
        Compute the autocorrelation2D_169 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_169 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_169(self.molecule)

    def __findAUTOCORR2D_170(self):
        '''
        Compute the autocorrelation2D_170 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_170 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_170(self.molecule)

    def __findAUTOCORR2D_171(self):
        '''
        Compute the autocorrelation2D_171 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_171 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_171(self.molecule)

    def __findAUTOCORR2D_172(self):
        '''
        Compute the autocorrelation2D_172 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_172 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_172(self.molecule)

    def __findAUTOCORR2D_173(self):
        '''
        Compute the autocorrelation2D_173 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_173 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_173(self.molecule)

    def __findAUTOCORR2D_174(self):
        '''
        Compute the autocorrelation2D_174 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_174 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_174(self.molecule)

    def __findAUTOCORR2D_175(self):
        '''
        Compute the autocorrelation2D_175 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_175 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_175(self.molecule)

    def __findAUTOCORR2D_176(self):
        '''
        Compute the autocorrelation2D_176 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_176 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_176(self.molecule)

    def __findAUTOCORR2D_177(self):
        '''
        Compute the autocorrelation2D_177 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_177 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_177(self.molecule)

    def __findAUTOCORR2D_178(self):
        '''
        Compute the autocorrelation2D_178 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_178 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_178(self.molecule)

    def __findAUTOCORR2D_179(self):
        '''
        Compute the autocorrelation2D_179 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_179 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_179(self.molecule)

    def __findAUTOCORR2D_180(self):
        '''
        Compute the autocorrelation2D_180 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_180 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_180(self.molecule)

    def __findAUTOCORR2D_181(self):
        '''
        Compute the autocorrelation2D_181 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_181 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_181(self.molecule)

    def __findAUTOCORR2D_182(self):
        '''
        Compute the autocorrelation2D_182 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_182 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_182(self.molecule)

    def __findAUTOCORR2D_183(self):
        '''
        Compute the autocorrelation2D_183 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_183 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_183(self.molecule)

    def __findAUTOCORR2D_184(self):
        '''
        Compute the autocorrelation2D_184 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_184 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_184(self.molecule)

    def __findAUTOCORR2D_185(self):
        '''
        Compute the autocorrelation2D_185 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_185 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_185(self.molecule)

    def __findAUTOCORR2D_186(self):
        '''
        Compute the autocorrelation2D_186 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_186 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_186(self.molecule)

    def __findAUTOCORR2D_187(self):
        '''
        Compute the autocorrelation2D_187 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_187 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_187(self.molecule)

    def __findAUTOCORR2D_188(self):
        '''
        Compute the autocorrelation2D_188 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_188 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_188(self.molecule)

    def __findAUTOCORR2D_189(self):
        '''
        Compute the autocorrelation2D_189 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_189 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_189(self.molecule)

    def __findAUTOCORR2D_190(self):
        '''
        Compute the autocorrelation2D_190 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_190 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_190(self.molecule)

    def __findAUTOCORR2D_191(self):
        '''
        Compute the autocorrelation2D_191 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_191 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_191(self.molecule)

    def __findAUTOCORR2D_192(self):
        '''
        Compute the autocorrelation2D_192 descriptor.
        Input:
          -
        Return:
          [double] - The autocorrelation2D_192 value.
          [None]   - If parsing the descriptor fails.
        '''
        return findAUTOCORR2D_192(self.molecule)

    # </editor-fold>

    # <editor-fold> BCUT2D descriptors
    def __findBCUT2D_CHGHI(self):
        '''
        Compute the BCUT2D_CHGHI descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_CHGHI descriptor.
        '''
        return findBCUT2D_CHGHI(self.molecule)

    def __findBCUT2D_CHGLO(self):
        '''
        Compute the BCUT2D_CHGLO descriptor.
        Input:
          -
        Return:
          [double] - The the BCUT2D_CHGLO descriptor.
        '''
        return findBCUT2D_CHGLO(self.molecule)

    def __findBCUT2D_LOGPHI(self):
        '''
        Compute the BCUT2D_LOGPHI descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_LOGPHI descriptor.
        '''
        return findBCUT2D_LOGPHI(self.molecule)

    def __findBCUT2D_LOGPLOW(self):
        '''
        Compute the BCUT2D_LOGPLOW descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_LOGPLOW descriptor.
        '''
        return findBCUT2D_LOGPLOW(self.molecule)

    def __findBCUT2D_MRHI(self):
        '''
        Compute the BCUT2D_MRHI descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_MRHI descriptor.
        '''
        return findBCUT2D_MRHI(self.molecule)

    def __findBCUT2D_MRLOW(self):
        '''
        Compute the BCUT2D_MRLOW descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_MRLOW descriptor.
        '''
        return findBCUT2D_MRLOW(self.molecule)

    def __findBCUT2D_MWHI(self):
        '''
        Compute the BCUT2D_MWHI descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_MWHI descriptor.
        '''
        return findBCUT2D_MWHI(self.molecule)

    def __findBCUT2D_MWLOW(self):
        '''
        Compute the BCUT2D_MWLOW descriptor.
        Input:
          -
        Return:
          [double] - The BCUT2D_MWLOW descriptor.
        '''
        return findBCUT2D_MWLOW(self.molecule)

    # </editor-fold>

    def __findBalabanJ(self):
        '''
        Compute the BalabanJ descriptor.
        Input:
          -
        Return:
          [double] - The BalabanJ descriptor.
        '''
        return findBalabanJ(self.molecule)

    def __findBertzCT(self):
        '''
        Compute the BertzCT descriptor.
        Input:
          -
        Return:
          [double] - The BertzCT descriptor.
        '''
        return findBertzCT(self.molecule)

    # <editor-fold> Chi descriptors
    def __findChi0(self):
        '''
        Compute the Chi0 descriptor.
        Input:
          -
        Return:
          [double] - The Chi0 descriptor.
        '''
        return findChi0(self.molecule)

    def __findChi0n(self):
        '''
        Compute the Chi0n descriptor.
        Input:
          -
        Return:
          [double] - The Chi0n descriptor.
        '''
        return findChi0n(self.molecule)

    def __findChi0v(self):
        '''
        Compute the Chi0v descriptor.
        Input:
          -
        Return:
          [double] - The Chi0v descriptor.
        '''
        return findChi0v(self.molecule)

    def __findChi1(self):
        '''
        Compute the Chi1 descriptor.
        Input:
          -
        Return:
          [double] - The Chi1 descriptor.
        '''
        return findChi1(self.molecule)

    def __findChi1n(self):
        '''
        Compute the Chi1n descriptor.
        Input:
          -
        Return:
          [double] - The Chi1n descriptor.
        '''
        return findChi1n(self.molecule)

    def __findChi1v(self):
        '''
        Compute the Chi1v descriptor.
        Input:
          -
        Return:
          [double] - The Chi1v descriptor.
        '''
        return findChi1v(self.molecule)

    def __findChi2n(self):
        '''
        Compute the Chi2n descriptor.
        Input:
          -
        Return:
          [double] - The Chi2n descriptor.
        '''
        return findChi2n(self.molecule)

    def __findChi2v(self):
        '''
        Compute the Chi2v descriptor.
        Input:
          -
        Return:
          [double] - The Chi2v descriptor.
        '''
        return findChi2v(self.molecule)

    def __findChi3n(self):
        '''
        Compute the Chi3n descriptor.
        Input:
          -
        Return:
          [double] - The Chi3n descriptor.
        '''
        return findChi3n(self.molecule)

    def __findChi3v(self):
        '''
        Compute the Chi3v descriptor.
        Input:
          -
        Return:
          [double] - The Chi3v descriptor.
        '''
        return findChi3v(self.molecule)

    def __findChi4n(self):
        '''
        Compute the Chi4n descriptor.
        Input:
          -
        Return:
          [double] - The Chi4n descriptor.
        '''
        return findChi4n(self.molecule)

    def __findChi4v(self):
        '''
        Compute the Chi4v descriptor.
        Input:
          -
        Return:
          [double] - The Chi4v descriptor.
        '''
        return findChi4v(self.molecule)

    # </editor-fold>

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
        if self.path is None or self.molecule is None or self.name is None or self.ExactMolWt is None or self.FpDensityMorgan1 is None or self.FpDensityMorgan2 is None or self.FpDensityMorgan3 is None or self.HeavyAtomMolWt is None or self.MaxAbsPartialCharge is None or self.MaxPartialCharge is None or self.MinAbsPartialCharge is None or self.MinPartialCharge is None or self.MolWt is None or self.NumRadicalElectrons is None or self.NumValenceElectrons is None:
            return False
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
        Compare two molecules to check if they are the same using their SMILES.
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
                        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                        # Return the sanitized molecule
                        return molecule, m
                    except Exception as e:
                        _ = errors.parseMolecule(f"The molecule '{molecule}' could not be parsed.", "error")
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

def read_descriptors_from_json(path):
    '''
    Read the descriptors from a json file.
    Input:
      -
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
        keys = ["Name", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "HeavyAtomMolWt", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "MolWt", "NumRadicalElectrons", "NumValenceElectrons"]
        # Validate the data
        for key in keys:
            # If key is lacking in data read from json (means malformed json!)
            if not key in data:
                # Add the missing key to the missing list
                missing.Append(key)
        # If missing list is not empty
        if missing:
            # Raise a Key error passing the file and the missing keys joined with ', '
            raise KeyError((path, ", ".join(missing)))
        # Since we have all keys, read them and return their values
        return data["Name"], data["ExactMolWt"], data["FpDensityMorgan1"], data["FpDensityMorgan2"], data["FpDensityMorgan3"], data["HeavyAtomMolWt"], data["MaxAbsPartialCharge"], data["MaxPartialCharge"], data["MinAbsPartialCharge"], data["MinPartialCharge"], data["MolWt"], data["NumRadicalElectrons"], data["NumValenceElectrons"]
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
        return Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
    return Errors.not_set(f"The variable is not set.")

# Descriptors functions #

# <editor-fold> AUTOCORR descriptors
def findAUTOCORR2D_1(molecule):
    '''
    Compute the autocorrelation2D_1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_1 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_1(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_2(molecule):
    '''
    Compute the autocorrelation2D_2 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_2 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_2(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_3(molecule):
    '''
    Compute the autocorrelation2D_3 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_3 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_3(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_4(molecule):
    '''
    Compute the autocorrelation2D_4 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_4 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_4(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_5(molecule):
    '''
    Compute the autocorrelation2D_5 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_5 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_5(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_6(molecule):
    '''
    Compute the autocorrelation2D_6 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_6 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_6(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_7(molecule):
    '''
    Compute the autocorrelation2D_7 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_7 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_7(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_8(molecule):
    '''
    Compute the autocorrelation2D_8 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_8 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_8(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_9(molecule):
    '''
    Compute the autocorrelation2D_9 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_9 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_9(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_10(molecule):
    '''
    Compute the autocorrelation2D_10 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_10 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_10(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_11(molecule):
    '''
    Compute the autocorrelation2D_11 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_11 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_11(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_12(molecule):
    '''
    Compute the autocorrelation2D_12 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_12 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_12(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_13(molecule):
    '''
    Compute the autocorrelation2D_13 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_13 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_13(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_14(molecule):
    '''
    Compute the autocorrelation2D_14 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_14 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_14(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_15(molecule):
    '''
    Compute the autocorrelation2D_15 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_15 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_15(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_16(molecule):
    '''
    Compute the autocorrelation2D_16 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_16 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_16(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_17(molecule):
    '''
    Compute the autocorrelation2D_17 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_17 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_17(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_18(molecule):
    '''
    Compute the autocorrelation2D_18 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_18 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_18(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_19(molecule):
    '''
    Compute the autocorrelation2D_19 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_19 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_19(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_20(molecule):
    '''
    Compute the autocorrelation2D_20 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_20 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_20(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_21(molecule):
    '''
    Compute the autocorrelation2D_21 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_21 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_21(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_22(molecule):
    '''
    Compute the autocorrelation2D_22 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_22 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_22(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_23(molecule):
    '''
    Compute the autocorrelation2D_23 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_23 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_23(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_24(molecule):
    '''
    Compute the autocorrelation2D_24 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_24 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_24(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_25(molecule):
    '''
    Compute the autocorrelation2D_25 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_25 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_25(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_26(molecule):
    '''
    Compute the autocorrelation2D_26 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_26 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_26(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_27(molecule):
    '''
    Compute the autocorrelation2D_27 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_27 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_27(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_28(molecule):
    '''
    Compute the autocorrelation2D_28 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_28 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_28(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_29(molecule):
    '''
    Compute the autocorrelation2D_29 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_29 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_29(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_30(molecule):
    '''
    Compute the autocorrelation2D_30 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_30 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_30(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_31(molecule):
    '''
    Compute the autocorrelation2D_31 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_31 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_31(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_32(molecule):
    '''
    Compute the autocorrelation2D_32 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_32 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_32(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_33(molecule):
    '''
    Compute the autocorrelation2D_33 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_33 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_33(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_34(molecule):
    '''
    Compute the autocorrelation2D_34 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_34 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_34(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_35(molecule):
    '''
    Compute the autocorrelation2D_35 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_35 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_35(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_36(molecule):
    '''
    Compute the autocorrelation2D_36 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_36 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_36(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_37(molecule):
    '''
    Compute the autocorrelation2D_37 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_37 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_37(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_38(molecule):
    '''
    Compute the autocorrelation2D_38 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_38 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_38(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_39(molecule):
    '''
    Compute the autocorrelation2D_39 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_39 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_39(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_40(molecule):
    '''
    Compute the autocorrelation2D_40 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_40 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_40(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_41(molecule):
    '''
    Compute the autocorrelation2D_41 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_41 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_41(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_42(molecule):
    '''
    Compute the autocorrelation2D_42 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_42 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_42(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_43(molecule):
    '''
    Compute the autocorrelation2D_43 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_43 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_43(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_44(molecule):
    '''
    Compute the autocorrelation2D_44 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_44 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_44(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_45(molecule):
    '''
    Compute the autocorrelation2D_45 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_45 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_45(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_46(molecule):
    '''
    Compute the autocorrelation2D_46 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_46 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_46(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_47(molecule):
    '''
    Compute the autocorrelation2D_47 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_47 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_47(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_48(molecule):
    '''
    Compute the autocorrelation2D_48 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_48 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_48(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_49(molecule):
    '''
    Compute the autocorrelation2D_49 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_49 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_49(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_50(molecule):
    '''
    Compute the autocorrelation2D_50 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_50 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_50(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_51(molecule):
    '''
    Compute the autocorrelation2D_51 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_51 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_51(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_52(molecule):
    '''
    Compute the autocorrelation2D_52 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_52 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_52(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_53(molecule):
    '''
    Compute the autocorrelation2D_53 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_53 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_53(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_54(molecule):
    '''
    Compute the autocorrelation2D_54 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_54 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_54(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_55(molecule):
    '''
    Compute the autocorrelation2D_55 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_55 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_55(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_56(molecule):
    '''
    Compute the autocorrelation2D_56 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_56 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_56(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_57(molecule):
    '''
    Compute the autocorrelation2D_57 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_57 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_57(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_58(molecule):
    '''
    Compute the autocorrelation2D_58 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_58 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_58(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_59(molecule):
    '''
    Compute the autocorrelation2D_59 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_59 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_59(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_60(molecule):
    '''
    Compute the autocorrelation2D_60 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_60 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_60(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_61(molecule):
    '''
    Compute the autocorrelation2D_61 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_61 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_61(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_62(molecule):
    '''
    Compute the autocorrelation2D_62 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_62 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_62(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_63(molecule):
    '''
    Compute the autocorrelation2D_63 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_63 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_63(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_64(molecule):
    '''
    Compute the autocorrelation2D_64 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_64 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_64(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_65(molecule):
    '''
    Compute the autocorrelation2D_65 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_65 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_65(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_66(molecule):
    '''
    Compute the autocorrelation2D_66 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_66 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_66(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_67(molecule):
    '''
    Compute the autocorrelation2D_67 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_67 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_67(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_68(molecule):
    '''
    Compute the autocorrelation2D_68 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_68 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_68(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_69(molecule):
    '''
    Compute the autocorrelation2D_69 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_69 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_69(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_70(molecule):
    '''
    Compute the autocorrelation2D_70 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_70 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_70(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_71(molecule):
    '''
    Compute the autocorrelation2D_71 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_71 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_71(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_72(molecule):
    '''
    Compute the autocorrelation2D_72 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_72 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_72(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_73(molecule):
    '''
    Compute the autocorrelation2D_73 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_73 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_73(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_74(molecule):
    '''
    Compute the autocorrelation2D_74 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_74 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_74(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_75(molecule):
    '''
    Compute the autocorrelation2D_75 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_75 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_75(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_76(molecule):
    '''
    Compute the autocorrelation2D_76 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_76 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_76(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_77(molecule):
    '''
    Compute the autocorrelation2D_77 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_77 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_77(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_78(molecule):
    '''
    Compute the autocorrelation2D_78 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_78 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_78(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_79(molecule):
    '''
    Compute the autocorrelation2D_79 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_79 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_79(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_80(molecule):
    '''
    Compute the autocorrelation2D_80 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_80 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_80(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_81(molecule):
    '''
    Compute the autocorrelation2D_81 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_81 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_81(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_82(molecule):
    '''
    Compute the autocorrelation2D_82 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_82 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_82(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_83(molecule):
    '''
    Compute the autocorrelation2D_83 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_83 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_83(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_84(molecule):
    '''
    Compute the autocorrelation2D_84 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_84 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_84(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_85(molecule):
    '''
    Compute the autocorrelation2D_85 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_85 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_85(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_86(molecule):
    '''
    Compute the autocorrelation2D_86 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_86 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_86(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_87(molecule):
    '''
    Compute the autocorrelation2D_87 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_87 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_87(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_88(molecule):
    '''
    Compute the autocorrelation2D_88 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_88 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_88(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_89(molecule):
    '''
    Compute the autocorrelation2D_89 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_89 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_89(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_90(molecule):
    '''
    Compute the autocorrelation2D_90 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_90 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_90(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_91(molecule):
    '''
    Compute the autocorrelation2D_91 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_91 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_91(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_92(molecule):
    '''
    Compute the autocorrelation2D_92 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_92 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_92(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_93(molecule):
    '''
    Compute the autocorrelation2D_93 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_93 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_93(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_94(molecule):
    '''
    Compute the autocorrelation2D_94 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_94 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_94(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_95(molecule):
    '''
    Compute the autocorrelation2D_95 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_95 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_95(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_96(molecule):
    '''
    Compute the autocorrelation2D_96 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_96 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_96(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_97(molecule):
    '''
    Compute the autocorrelation2D_97 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_97 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_97(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_98(molecule):
    '''
    Compute the autocorrelation2D_98 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_98 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_98(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_99(molecule):
    '''
    Compute the autocorrelation2D_99 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_99 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_99(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_100(molecule):
    '''
    Compute the autocorrelation2D_100 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_100 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_100(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_101(molecule):
    '''
    Compute the autocorrelation2D_101 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_101 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_101(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_102(molecule):
    '''
    Compute the autocorrelation2D_102 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_102 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_102(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_103(molecule):
    '''
    Compute the autocorrelation2D_103 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_103 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_103(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_104(molecule):
    '''
    Compute the autocorrelation2D_104 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_104 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_104(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_105(molecule):
    '''
    Compute the autocorrelation2D_105 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_105 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_105(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_106(molecule):
    '''
    Compute the autocorrelation2D_106 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_106 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_106(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_107(molecule):
    '''
    Compute the autocorrelation2D_107 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_107 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_107(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_108(molecule):
    '''
    Compute the autocorrelation2D_108 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_108 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_108(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_109(molecule):
    '''
    Compute the autocorrelation2D_109 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_109 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_109(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_110(molecule):
    '''
    Compute the autocorrelation2D_110 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_110 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_110(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_111(molecule):
    '''
    Compute the autocorrelation2D_111 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_111 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_111(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_112(molecule):
    '''
    Compute the autocorrelation2D_112 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_112 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_112(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_113(molecule):
    '''
    Compute the autocorrelation2D_113 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_113 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_113(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_114(molecule):
    '''
    Compute the autocorrelation2D_114 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_114 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_114(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_115(molecule):
    '''
    Compute the autocorrelation2D_115 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_115 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_115(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_116(molecule):
    '''
    Compute the autocorrelation2D_116 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_116 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_116(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_117(molecule):
    '''
    Compute the autocorrelation2D_117 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_117 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_117(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_118(molecule):
    '''
    Compute the autocorrelation2D_118 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_118 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_118(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_119(molecule):
    '''
    Compute the autocorrelation2D_119 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_119 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_119(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_120(molecule):
    '''
    Compute the autocorrelation2D_120 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_120 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_120(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_121(molecule):
    '''
    Compute the autocorrelation2D_121 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_121 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_121(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_122(molecule):
    '''
    Compute the autocorrelation2D_122 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_122 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_122(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_123(molecule):
    '''
    Compute the autocorrelation2D_123 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_123 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_123(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_124(molecule):
    '''
    Compute the autocorrelation2D_124 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_124 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_124(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_125(molecule):
    '''
    Compute the autocorrelation2D_125 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_125 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_125(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_126(molecule):
    '''
    Compute the autocorrelation2D_126 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_126 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_126(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_127(molecule):
    '''
    Compute the autocorrelation2D_127 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_127 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_127(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_128(molecule):
    '''
    Compute the autocorrelation2D_128 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_128 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_128(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_129(molecule):
    '''
    Compute the autocorrelation2D_129 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_129 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_129(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_130(molecule):
    '''
    Compute the autocorrelation2D_130 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_130 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_130(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_131(molecule):
    '''
    Compute the autocorrelation2D_131 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_131 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_131(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_132(molecule):
    '''
    Compute the autocorrelation2D_132 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_132 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_132(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_133(molecule):
    '''
    Compute the autocorrelation2D_133 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_133 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_133(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_134(molecule):
    '''
    Compute the autocorrelation2D_134 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_134 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_134(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_135(molecule):
    '''
    Compute the autocorrelation2D_135 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_135 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_135(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_136(molecule):
    '''
    Compute the autocorrelation2D_136 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_136 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_136(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_137(molecule):
    '''
    Compute the autocorrelation2D_137 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_137 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_137(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_138(molecule):
    '''
    Compute the autocorrelation2D_138 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_138 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_138(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_139(molecule):
    '''
    Compute the autocorrelation2D_139 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_139 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_139(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_140(molecule):
    '''
    Compute the autocorrelation2D_140 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_140 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_140(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_141(molecule):
    '''
    Compute the autocorrelation2D_141 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_141 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_141(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_142(molecule):
    '''
    Compute the autocorrelation2D_142 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_142 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_142(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_143(molecule):
    '''
    Compute the autocorrelation2D_143 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_143 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_143(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_144(molecule):
    '''
    Compute the autocorrelation2D_144 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_144 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_144(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_145(molecule):
    '''
    Compute the autocorrelation2D_145 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_145 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_145(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_146(molecule):
    '''
    Compute the autocorrelation2D_146 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_146 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_146(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_147(molecule):
    '''
    Compute the autocorrelation2D_147 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_147 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_147(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_148(molecule):
    '''
    Compute the autocorrelation2D_148 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_148 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_148(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_149(molecule):
    '''
    Compute the autocorrelation2D_149 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_149 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_149(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_150(molecule):
    '''
    Compute the autocorrelation2D_150 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_150 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_150(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_151(molecule):
    '''
    Compute the autocorrelation2D_151 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_151 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_151(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_152(molecule):
    '''
    Compute the autocorrelation2D_152 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_152 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_152(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_153(molecule):
    '''
    Compute the autocorrelation2D_153 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_153 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_153(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_154(molecule):
    '''
    Compute the autocorrelation2D_154 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_154 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_154(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_155(molecule):
    '''
    Compute the autocorrelation2D_155 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_155 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_155(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_156(molecule):
    '''
    Compute the autocorrelation2D_156 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_156 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_156(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_157(molecule):
    '''
    Compute the autocorrelation2D_157 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_157 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_157(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_158(molecule):
    '''
    Compute the autocorrelation2D_158 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_158 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_158(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_159(molecule):
    '''
    Compute the autocorrelation2D_159 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_159 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_159(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_160(molecule):
    '''
    Compute the autocorrelation2D_160 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_160 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_160(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_161(molecule):
    '''
    Compute the autocorrelation2D_161 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_161 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_161(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_162(molecule):
    '''
    Compute the autocorrelation2D_162 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_162 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_162(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_163(molecule):
    '''
    Compute the autocorrelation2D_163 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_163 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_163(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_164(molecule):
    '''
    Compute the autocorrelation2D_164 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_164 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_164(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_165(molecule):
    '''
    Compute the autocorrelation2D_165 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_165 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_165(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_166(molecule):
    '''
    Compute the autocorrelation2D_166 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_166 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_166(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_167(molecule):
    '''
    Compute the autocorrelation2D_167 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_167 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_167(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_168(molecule):
    '''
    Compute the autocorrelation2D_168 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_168 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_168(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_169(molecule):
    '''
    Compute the autocorrelation2D_169 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_169 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_169(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_170(molecule):
    '''
    Compute the autocorrelation2D_170 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_170 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_170(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_171(molecule):
    '''
    Compute the autocorrelation2D_171 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_171 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_171(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_172(molecule):
    '''
    Compute the autocorrelation2D_172 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_172 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_172(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_173(molecule):
    '''
    Compute the autocorrelation2D_173 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_173 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_173(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_174(molecule):
    '''
    Compute the autocorrelation2D_174 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_174 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_174(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_175(molecule):
    '''
    Compute the autocorrelation2D_175 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_175 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_175(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_176(molecule):
    '''
    Compute the autocorrelation2D_176 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_176 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_176(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_177(molecule):
    '''
    Compute the autocorrelation2D_177 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_177 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_177(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_178(molecule):
    '''
    Compute the autocorrelation2D_178 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_178 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_178(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_179(molecule):
    '''
    Compute the autocorrelation2D_179 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_179 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_179(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_180(molecule):
    '''
    Compute the autocorrelation2D_180 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_180 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_180(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_181(molecule):
    '''
    Compute the autocorrelation2D_181 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_181 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_181(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_182(molecule):
    '''
    Compute the autocorrelation2D_182 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_182 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_182(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_183(molecule):
    '''
    Compute the autocorrelation2D_183 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_183 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_183(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_184(molecule):
    '''
    Compute the autocorrelation2D_184 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_184 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_184(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_185(molecule):
    '''
    Compute the autocorrelation2D_185 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_185 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_185(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_186(molecule):
    '''
    Compute the autocorrelation2D_186 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_186 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_186(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_187(molecule):
    '''
    Compute the autocorrelation2D_187 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_187 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_187(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_188(molecule):
    '''
    Compute the autocorrelation2D_188 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_188 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_188(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_189(molecule):
    '''
    Compute the autocorrelation2D_189 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_189 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_189(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_190(molecule):
    '''
    Compute the autocorrelation2D_190 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_190 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_190(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_191(molecule):
    '''
    Compute the autocorrelation2D_191 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_191 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_191(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findAUTOCORR2D_192(molecule):
    '''
    Compute the autocorrelation2D_192 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The autocorrelation2D_192 value.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.AUTOCORR2D_192(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

# </editor-fold>

# <editor-fold> BCUT2D descriptors
def findBCUT2D_CHGHI(molecule):
    '''
    Compute the BCUT2D_CHGHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_CHGHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_CHGHI(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_CHGLO(molecule):
    '''
    Compute the BCUT2D_CHGLO descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The the BCUT2D_CHGLO descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_CHGLO(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_LOGPHI(molecule):
    '''
    Compute the BCUT2D_LOGPHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_LOGPHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_LOGPHI(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_LOGPLOW(molecule):
    '''
    Compute the BCUT2D_LOGPLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_LOGPLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_LOGPLOW(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MRHI(molecule):
    '''
    Compute the BCUT2D_MRHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_MRHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MRHI(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MRLOW(molecule):
    '''
    Compute the BCUT2D_MRLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_MRLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MRLOW(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MWHI(molecule):
    '''
    Compute the BCUT2D_MWHI descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_MWHI descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MWHI(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBCUT2D_MWLOW(molecule):
    '''
    Compute the BCUT2D_MWLOW descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BCUT2D_MWLOW descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BCUT2D_MWLOW(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

# </editor-fold>

def findBalabanJ(molecule):
    '''
    Compute the BalabanJ descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BalabanJ descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BalabanJ(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findBertzCT(molecule):
    '''
    Compute the BertzCT descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The BertzCT descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.BertzCT(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

# <editor-fold> Chi descriptors
def findChi0(molecule):
    '''
    Compute the Chi0 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi0 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi0n(molecule):
    '''
    Compute the Chi0n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi0n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0n(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi0v(molecule):
    '''
    Compute the Chi0v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi0v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi0v(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi1(molecule):
    '''
    Compute the Chi1 descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi1 descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi1n(molecule):
    '''
    Compute the Chi1n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi1n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1n(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi1v(molecule):
    '''
    Compute the Chi1v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi1v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi1v(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi2n(molecule):
    '''
    Compute the Chi2n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi2n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi2n(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi2v(molecule):
    '''
    Compute the Chi2v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi2v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi2v(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi3n(molecule):
    '''
    Compute the Chi3n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi3n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi3n(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi3v(molecule):
    '''
    Compute the Chi3v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi3v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi3v(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi4n(molecule):
    '''
    Compute the Chi4n descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi4n descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi4n(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

def findChi4v(molecule):
    '''
    Compute the Chi4v descriptor.
    Input:
      molecule [rdkit.Chem.rdchem.Mol] - The molecule to be evaluated.
    Return:
      [double] - The Chi4v descriptor.
      [None]   - If parsing the descriptor fails.
    '''
    if molecule:
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.Chi4v(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None

# </editor-fold>

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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.ExactMolWt(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan1(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan2(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.FpDensityMorgan3(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.HeavyAtomMolWt(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxAbsPartialCharge(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MaxPartialCharge(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinAbsPartialCharge(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MinPartialCharge(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        if type(molecule) == Chem.rdchem.Mol:
            return rdkit.Chem.Descriptors.MolWt(molecule)
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
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
        _ = Errors.wrong_type(f"The molecule '{molecule}' has wrong type! Expected 'rdkit.Chem.rdchem.Mol' and got '{type(molecule)}'")
        return None
    _ = Errors.not_set(f"The variable is not set.")
    return None
