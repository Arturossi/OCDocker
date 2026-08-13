#!/usr/bin/env python3

# Description
###############################################################################
"""
Configuration management for OCDocker using dataclasses and singleton pattern.

This module provides a structured way to manage OCDocker configuration,
replacing the global variables in Initialise.py with type-safe dataclasses.

Usage:

from OCDocker.Config import get_config, OCDockerConfig
"""

# Imports
###############################################################################
import configparser
import os
import threading

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import OCDocker.Error as ocerror

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################
# Configuration Dataclasses


GNINA_DEFAULT_SCORING_FUNCTIONS = [
    "ad4_scoring",
    "default",
    "dkoes_fast",
    "dkoes_scoring",
    "dkoes_scoring_old",
    "vina",
    "vinardo",
]


GNINA_DEFAULT_CNN_MODELS = [
    "all_default_to_default_1_3_1",
    "all_default_to_default_1_3_2",
    "all_default_to_default_1_3_3",
    "crossdock_default2018",
    "crossdock_default2018_1",
    "crossdock_default2018_1_3",
    "crossdock_default2018_1_3_1",
    "crossdock_default2018_1_3_2",
    "crossdock_default2018_1_3_3",
    "crossdock_default2018_1_3_4",
    "crossdock_default2018_2",
    "crossdock_default2018_3",
    "crossdock_default2018_4",
    "crossdock_default2018_KD_1",
    "crossdock_default2018_KD_2",
    "crossdock_default2018_KD_3",
    "crossdock_default2018_KD_4",
    "crossdock_default2018_KD_5",
    "default1.0",
    "default2017",
    "dense",
    "dense_1",
    "dense_1_3",
    "dense_1_3_1",
    "dense_1_3_2",
    "dense_1_3_3",
    "dense_1_3_4",
    "dense_1_3_PT_KD",
    "dense_1_3_PT_KD_1",
    "dense_1_3_PT_KD_2",
    "dense_1_3_PT_KD_3",
    "dense_1_3_PT_KD_4",
    "dense_1_3_PT_KD_def2018",
    "dense_1_3_PT_KD_def2018_1",
    "dense_1_3_PT_KD_def2018_2",
    "dense_1_3_PT_KD_def2018_3",
    "dense_1_3_PT_KD_def2018_4",
    "dense_2",
    "dense_3",
    "dense_4",
    "fast",
    "general_default2018",
    "general_default2018_1",
    "general_default2018_2",
    "general_default2018_3",
    "general_default2018_4",
    "general_default2018_KD_1",
    "general_default2018_KD_2",
    "general_default2018_KD_3",
    "general_default2018_KD_4",
    "general_default2018_KD_5",
    "redock_default2018",
    "redock_default2018_1",
    "redock_default2018_1_3",
    "redock_default2018_1_3_1",
    "redock_default2018_1_3_2",
    "redock_default2018_1_3_3",
    "redock_default2018_1_3_4",
    "redock_default2018_2",
    "redock_default2018_3",
    "redock_default2018_4",
    "redock_default2018_KD_1",
    "redock_default2018_KD_2",
    "redock_default2018_KD_3",
    "redock_default2018_KD_4",
    "redock_default2018_KD_5",
]


DEFAULT_REFERENCE_COLUMN_ORDER = [
    "name",
    "receptor",
    "ligand",
    "SMINA_VINA",
    "SMINA_SCORING_DKOES",
    "SMINA_VINARDO",
    "SMINA_OLD_SCORING_DKOES",
    "SMINA_FAST_DKOES",
    "SMINA_SCORING_AD4",
    "VINA_VINA",
    "VINA_VINARDO",
    "PLANTS_CHEMPLP",
    "PLANTS_PLP",
    "PLANTS_PLP95",
    "ODDT_RFSCORE_V1",
    "ODDT_RFSCORE_V2",
    "ODDT_RFSCORE_V3",
    "ODDT_PLECRF_P5_L1_S65536",
    "ODDT_NNSCORE",
    "countA",
    "countR",
    "countN",
    "countD",
    "countC",
    "countQ",
    "countE",
    "countG",
    "countH",
    "countI",
    "countL",
    "countK",
    "countM",
    "countF",
    "countP",
    "countS",
    "countT",
    "countW",
    "countY",
    "countV",
    "TotalAALength",
    "AvgAALength",
    "countChain",
    "SASA",
    "DipoleMoment",
    "IsoelectricPoint",
    "GRAVY",
    "Aromaticity",
    "InstabilityIndex",
    "AUTOCORR2D_1",
    "AUTOCORR2D_2",
    "AUTOCORR2D_3",
    "AUTOCORR2D_4",
    "AUTOCORR2D_5",
    "AUTOCORR2D_6",
    "AUTOCORR2D_7",
    "AUTOCORR2D_8",
    "AUTOCORR2D_9",
    "AUTOCORR2D_10",
    "AUTOCORR2D_11",
    "AUTOCORR2D_12",
    "AUTOCORR2D_13",
    "AUTOCORR2D_14",
    "AUTOCORR2D_15",
    "AUTOCORR2D_16",
    "AUTOCORR2D_17",
    "AUTOCORR2D_18",
    "AUTOCORR2D_19",
    "AUTOCORR2D_20",
    "AUTOCORR2D_21",
    "AUTOCORR2D_22",
    "AUTOCORR2D_23",
    "AUTOCORR2D_24",
    "AUTOCORR2D_25",
    "AUTOCORR2D_26",
    "AUTOCORR2D_27",
    "AUTOCORR2D_28",
    "AUTOCORR2D_29",
    "AUTOCORR2D_30",
    "AUTOCORR2D_31",
    "AUTOCORR2D_32",
    "AUTOCORR2D_33",
    "AUTOCORR2D_34",
    "AUTOCORR2D_35",
    "AUTOCORR2D_36",
    "AUTOCORR2D_37",
    "AUTOCORR2D_38",
    "AUTOCORR2D_39",
    "AUTOCORR2D_40",
    "AUTOCORR2D_41",
    "AUTOCORR2D_42",
    "AUTOCORR2D_43",
    "AUTOCORR2D_44",
    "AUTOCORR2D_45",
    "AUTOCORR2D_46",
    "AUTOCORR2D_47",
    "AUTOCORR2D_48",
    "AUTOCORR2D_49",
    "AUTOCORR2D_50",
    "AUTOCORR2D_51",
    "AUTOCORR2D_52",
    "AUTOCORR2D_53",
    "AUTOCORR2D_54",
    "AUTOCORR2D_55",
    "AUTOCORR2D_56",
    "AUTOCORR2D_57",
    "AUTOCORR2D_58",
    "AUTOCORR2D_59",
    "AUTOCORR2D_60",
    "AUTOCORR2D_61",
    "AUTOCORR2D_62",
    "AUTOCORR2D_63",
    "AUTOCORR2D_64",
    "AUTOCORR2D_65",
    "AUTOCORR2D_66",
    "AUTOCORR2D_67",
    "AUTOCORR2D_68",
    "AUTOCORR2D_69",
    "AUTOCORR2D_70",
    "AUTOCORR2D_71",
    "AUTOCORR2D_72",
    "AUTOCORR2D_73",
    "AUTOCORR2D_74",
    "AUTOCORR2D_75",
    "AUTOCORR2D_76",
    "AUTOCORR2D_77",
    "AUTOCORR2D_78",
    "AUTOCORR2D_79",
    "AUTOCORR2D_80",
    "AUTOCORR2D_81",
    "AUTOCORR2D_82",
    "AUTOCORR2D_83",
    "AUTOCORR2D_84",
    "AUTOCORR2D_85",
    "AUTOCORR2D_86",
    "AUTOCORR2D_87",
    "AUTOCORR2D_88",
    "AUTOCORR2D_89",
    "AUTOCORR2D_90",
    "AUTOCORR2D_91",
    "AUTOCORR2D_92",
    "AUTOCORR2D_93",
    "AUTOCORR2D_94",
    "AUTOCORR2D_95",
    "AUTOCORR2D_96",
    "AUTOCORR2D_97",
    "AUTOCORR2D_98",
    "AUTOCORR2D_99",
    "AUTOCORR2D_100",
    "AUTOCORR2D_101",
    "AUTOCORR2D_102",
    "AUTOCORR2D_103",
    "AUTOCORR2D_104",
    "AUTOCORR2D_105",
    "AUTOCORR2D_106",
    "AUTOCORR2D_107",
    "AUTOCORR2D_108",
    "AUTOCORR2D_109",
    "AUTOCORR2D_110",
    "AUTOCORR2D_111",
    "AUTOCORR2D_112",
    "AUTOCORR2D_113",
    "AUTOCORR2D_114",
    "AUTOCORR2D_115",
    "AUTOCORR2D_116",
    "AUTOCORR2D_117",
    "AUTOCORR2D_118",
    "AUTOCORR2D_119",
    "AUTOCORR2D_120",
    "AUTOCORR2D_121",
    "AUTOCORR2D_122",
    "AUTOCORR2D_123",
    "AUTOCORR2D_124",
    "AUTOCORR2D_125",
    "AUTOCORR2D_126",
    "AUTOCORR2D_127",
    "AUTOCORR2D_128",
    "AUTOCORR2D_129",
    "AUTOCORR2D_130",
    "AUTOCORR2D_131",
    "AUTOCORR2D_132",
    "AUTOCORR2D_133",
    "AUTOCORR2D_134",
    "AUTOCORR2D_135",
    "AUTOCORR2D_136",
    "AUTOCORR2D_137",
    "AUTOCORR2D_138",
    "AUTOCORR2D_139",
    "AUTOCORR2D_140",
    "AUTOCORR2D_141",
    "AUTOCORR2D_142",
    "AUTOCORR2D_143",
    "AUTOCORR2D_144",
    "AUTOCORR2D_145",
    "AUTOCORR2D_146",
    "AUTOCORR2D_147",
    "AUTOCORR2D_148",
    "AUTOCORR2D_149",
    "AUTOCORR2D_150",
    "AUTOCORR2D_151",
    "AUTOCORR2D_152",
    "AUTOCORR2D_153",
    "AUTOCORR2D_154",
    "AUTOCORR2D_155",
    "AUTOCORR2D_156",
    "AUTOCORR2D_157",
    "AUTOCORR2D_158",
    "AUTOCORR2D_159",
    "AUTOCORR2D_160",
    "AUTOCORR2D_161",
    "AUTOCORR2D_162",
    "AUTOCORR2D_163",
    "AUTOCORR2D_164",
    "AUTOCORR2D_165",
    "AUTOCORR2D_166",
    "AUTOCORR2D_167",
    "AUTOCORR2D_168",
    "AUTOCORR2D_169",
    "AUTOCORR2D_170",
    "AUTOCORR2D_171",
    "AUTOCORR2D_172",
    "AUTOCORR2D_173",
    "AUTOCORR2D_174",
    "AUTOCORR2D_175",
    "AUTOCORR2D_176",
    "AUTOCORR2D_177",
    "AUTOCORR2D_178",
    "AUTOCORR2D_179",
    "AUTOCORR2D_180",
    "AUTOCORR2D_181",
    "AUTOCORR2D_182",
    "AUTOCORR2D_183",
    "AUTOCORR2D_184",
    "AUTOCORR2D_185",
    "AUTOCORR2D_186",
    "AUTOCORR2D_187",
    "AUTOCORR2D_188",
    "AUTOCORR2D_189",
    "AUTOCORR2D_190",
    "AUTOCORR2D_191",
    "AUTOCORR2D_192",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
    "Chi0",
    "Chi0v",
    "Chi0n",
    "Chi1",
    "Chi1v",
    "Chi1n",
    "Chi2v",
    "Chi2n",
    "Chi3v",
    "Chi3n",
    "Chi4v",
    "Chi4n",
    "EState_VSA1",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "EState_VSA10",
    "EState_VSA11",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "MolLogP",
    "MolMR",
    "MolWt",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "PEOE_VSA1",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "SMR_VSA1",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SMR_VSA10",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "VSA_EState1",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "VSA_EState10",
    "BalabanJ",
    "BertzCT",
    "ExactMolWt",
    "FractionCSP3",
    "HallKierAlpha",
    "HeavyAtomMolWt",
    "HeavyAtomCount",
    "LabuteASA",
    "TPSA",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "MaxAbsPartialCharge",
    "MaxPartialCharge",
    "MinAbsPartialCharge",
    "MinPartialCharge",
    "qed",
    "RingCount",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "RadiusOfGyration",
    "SpherocityIndex",
    "NHOHCount",
    "NOCount",
]


@dataclass
class VinaConfig:
    """Configuration for the AutoDock Vina docking engine.

    Parameters
    ----------
    executable : str, optional
        Path or name of the Vina executable, by default ``"vina"``.
    split_executable : str, optional
        Path or name of the Vina split executable, by default ``"vina_split"``.
    energy_range : str, optional
        Maximum energy difference between best and worst binding mode (kcal/mol),
        by default ``"10"``.
    exhaustiveness : int | str, optional
        Search exhaustiveness, by default ``5``.
    num_modes : str, optional
        Maximum number of binding modes to generate, by default ``"3"``.
    scoring : str, optional
        Active scoring function name, by default ``"vina"``.
    scoring_functions : list of str, optional
        Scoring functions available for rescoring, by default ``["vina"]``.
    """

    executable: str = "vina"
    split_executable: str = "vina_split"
    energy_range: str = "10"
    exhaustiveness: Any = 5  # Can be int or str depending on config file
    num_modes: str = "3"
    scoring: str = "vina"
    scoring_functions: List[str] = field(default_factory=lambda: ["vina"])


@dataclass
class SminaConfig:
    """Configuration for the Smina docking engine.

    Parameters
    ----------
    executable : str, optional
        Path or name of the Smina executable, by default ``"smina"``.
    energy_range : str, optional
        Maximum energy difference between best and worst binding mode, by default ``"10"``.
    exhaustiveness : str, optional
        Search exhaustiveness, by default ``"5"``.
    num_modes : str, optional
        Maximum number of binding modes to generate, by default ``"3"``.
    scoring : str, optional
        Active scoring function name, by default ``"vinardo"``.
    scoring_functions : list of str, optional
        Scoring functions available for rescoring, by default ``["vinardo"]``.
    custom_scoring : str, optional
        Custom scoring file path or ``"no"``, by default ``"no"``.
    custom_atoms : str, optional
        Custom atom types file path or ``"no"``, by default ``"no"``.
    local_only : str, optional
        Local optimization only flag, by default ``"no"``.
    minimize : str, optional
        Minimize final poses flag, by default ``"no"``.
    randomize_only : str, optional
        Randomize coordinates only flag, by default ``"no"``.
    minimize_iters : str, optional
        Number of minimization iterations, by default ``"0"``.
    accurate_line : str, optional
        Accurate line search flag, by default ``"no"``.
    minimize_early_term : str, optional
        Early termination during minimization flag, by default ``"no"``.
    approximation : str, optional
        Approximation method for scoring, by default ``"spline"``.
    factor : str, optional
        Approximation factor, by default ``"32"``.
    force_cap : str, optional
        Force cap during minimization, by default ``"10"``.
    user_grid : str, optional
        User grid file path or ``"no"``, by default ``"no"``.
    user_grid_lambda : str, optional
        User grid lambda parameter, by default ``"no"``.
    """

    executable: str = "smina"
    energy_range: str = "10"
    exhaustiveness: str = "5"
    num_modes: str = "3"
    scoring: str = "vinardo"
    scoring_functions: List[str] = field(default_factory=lambda: ["vinardo"])
    custom_scoring: str = "no"
    custom_atoms: str = "no"
    local_only: str = "no"
    minimize: str = "no"
    randomize_only: str = "no"
    minimize_iters: str = "0"
    accurate_line: str = "no"
    minimize_early_term: str = "no"
    approximation: str = "spline"
    factor: str = "32"
    force_cap: str = "10"
    user_grid: str = "no"
    user_grid_lambda: str = "no"


@dataclass
class GninaConfig:
    """Configuration for the Gnina docking engine.

    String fields use ``"no"`` where a Gnina CLI flag is disabled unless noted.
    List fields hold allowed scoring or CNN model identifiers.

    Parameters
    ----------
    executable : str, optional
        Path or name of the Gnina executable, by default ``"gnina"``.
    flex, flexres, flexdist_ligand, flexdist, flex_limit, flex_max : str, optional
        Flexible receptor and distance restraints, by default ``"no"``.
    autobox_ligand : str, optional
        Ligand file for autobox centering, by default ``"no"``.
    autobox_add, autobox_extend : str, optional
        Autobox padding (Å), by default ``"4"`` and ``"1"``.
    no_lig : str, optional
        Disable ligand input flag, by default ``"no"``.
    covalent_rec_atom, covalent_lig_atom_pattern, covalent_lig_atom_position, covalent_fix_lig_atom_position, covalent_bond_order, covalent_optimize_lig : str, optional
        Covalent docking options, by default ``"no"`` except ``covalent_bond_order`` (``"1"``).
    exhaustiveness : str, optional
        Search exhaustiveness, by default ``"8"``.
    num_modes : str, optional
        Maximum number of output poses, by default ``"9"``.
    scoring : str, optional
        Primary scoring function, by default ``"default"``.
    scoring_functions : list of str, optional
        Allowed Gnina scoring function names, by default GNINA default list.
    custom_scoring, custom_atoms : str, optional
        Custom scoring or atom-type files, by default ``"no"``.
    score_only, local_only, minimize, randomize_only : str, optional
        Pose processing modes, by default ``"no"``.
    num_mc_steps, max_mc_steps, num_mc_saved, temperature : str, optional
        Monte Carlo minimization controls, by default ``"no"``.
    minimize_iters : str, optional
        Minimization iteration count, by default ``"0"``.
    accurate_line, simple_ascent, minimize_early_term, minimize_single_full : str, optional
        Minimization behavior flags, by default ``"no"``.
    approximation : str, optional
        Scoring approximation method, by default ``"spline"``.
    factor, force_cap : str, optional
        Approximation factor and force cap, by default ``"32"`` and ``"10"``.
    user_grid : str, optional
        User grid file, by default ``"no"``.
    user_grid_lambda : str, optional
        User grid lambda, by default ``"-1"``.
    print_terms, print_atom_types : str, optional
        Debug output flags, by default ``"no"``.
    cnn_scoring : str, optional
        CNN rescoring mode, by default ``"rescore"``.
    cnn : str, optional
        Default CNN model set, by default ``"default"``.
    cnn_models : list of str, optional
        Available CNN model identifiers, by default GNINA default list.
    cnn_model : str, optional
        Explicit CNN model override, by default ``"no"``.
    cnn_rotation : str, optional
        CNN rotation augmentation count, by default ``"0"``.
    cnn_mix_emp_force, cnn_mix_emp_energy, cnn_empirical_weight : str, optional
        CNN / empirical score mixing weights, by default ``"no"``, ``"no"``, ``"1"``.
    cnn_center_x, cnn_center_y, cnn_center_z : str, optional
        CNN grid center overrides, by default ``"no"``.
    cnn_verbose : str, optional
        Verbose CNN output, by default ``"no"``.
    out_flex, atom_terms, atom_term_data : str, optional
        Output formatting options, by default ``"no"``.
    pose_sort_order : str, optional
        Pose sorting key, by default ``"CNNscore"``.
    full_flex_output : str, optional
        Write full flexible receptor output, by default ``"no"``.
    cpu : str, optional
        CPU thread count, by default ``"auto"``.
    seed : str, optional
        Random seed, by default ``"no"``.
    min_rmsd_filter : str, optional
        Minimum RMSD filter between poses, by default ``"1"``.
    quiet : str, optional
        Suppress Gnina stdout, by default ``"no"``.
    addH, stripH : str, optional
        Protonation controls, by default ``"yes"`` and ``"no"``.
    device : str, optional
        CUDA device index, by default ``"0"``.
    no_gpu : str, optional
        Disable GPU acceleration, by default ``"no"``.
    """

    executable: str = "gnina"
    # Input/flexible receptor
    flex: str = "no"
    flexres: str = "no"
    flexdist_ligand: str = "no"
    flexdist: str = "no"
    flex_limit: str = "no"
    flex_max: str = "no"
    # Search space/covalent
    autobox_ligand: str = "no"
    autobox_add: str = "4"
    autobox_extend: str = "1"
    no_lig: str = "no"
    covalent_rec_atom: str = "no"
    covalent_lig_atom_pattern: str = "no"
    covalent_lig_atom_position: str = "no"
    covalent_fix_lig_atom_position: str = "no"
    covalent_bond_order: str = "1"
    covalent_optimize_lig: str = "no"
    # Scoring/minimization
    exhaustiveness: str = "8"
    num_modes: str = "9"
    scoring: str = "default"
    scoring_functions: List[str] = field(default_factory=lambda: GNINA_DEFAULT_SCORING_FUNCTIONS.copy())
    custom_scoring: str = "no"
    custom_atoms: str = "no"
    score_only: str = "no"
    local_only: str = "no"
    minimize: str = "no"
    randomize_only: str = "no"
    num_mc_steps: str = "no"
    max_mc_steps: str = "no"
    num_mc_saved: str = "no"
    temperature: str = "no"
    minimize_iters: str = "0"
    accurate_line: str = "no"
    simple_ascent: str = "no"
    minimize_early_term: str = "no"
    minimize_single_full: str = "no"
    approximation: str = "spline"
    factor: str = "32"
    force_cap: str = "10"
    user_grid: str = "no"
    user_grid_lambda: str = "-1"
    print_terms: str = "no"
    print_atom_types: str = "no"
    # CNN
    cnn_scoring: str = "rescore"
    cnn: str = "default"
    cnn_models: List[str] = field(default_factory=lambda: GNINA_DEFAULT_CNN_MODELS.copy())
    cnn_model: str = "no"
    cnn_rotation: str = "0"
    cnn_mix_emp_force: str = "no"
    cnn_mix_emp_energy: str = "no"
    cnn_empirical_weight: str = "1"
    cnn_center_x: str = "no"
    cnn_center_y: str = "no"
    cnn_center_z: str = "no"
    cnn_verbose: str = "no"
    # Output extras
    out_flex: str = "no"
    atom_terms: str = "no"
    atom_term_data: str = "no"
    pose_sort_order: str = "CNNscore"
    full_flex_output: str = "no"
    # Misc
    cpu: str = "auto"
    seed: str = "no"
    min_rmsd_filter: str = "1"
    quiet: str = "no"
    addH: str = "yes"
    stripH: str = "no"
    device: str = "0"
    no_gpu: str = "no"


@dataclass
class PLANTSConfig:
    """Configuration for the PLANTS docking engine.

    Parameters
    ----------
    executable : str, optional
        Path or name of the PLANTS executable, by default ``"plants"``.
    cluster_structures : int, optional
        Number of cluster structures to retain, by default ``3``.
    cluster_rmsd : str, optional
        RMSD threshold for clustering (Å), by default ``"2.0"``.
    search_speed : str, optional
        PLANTS search speed preset, by default ``"speed1"``.
    scoring : str, optional
        Primary scoring function, by default ``"chemplp"``.
    scoring_functions : list of str, optional
        Available scoring functions, by default ``["chemplp", "plp", "plp95"]``.
    rescoring_mode : str, optional
        Rescoring mode after docking, by default ``"simplex"``.
    """

    executable: str = "plants"
    cluster_structures: int = 3
    cluster_rmsd: str = "2.0"
    search_speed: str = "speed1"
    scoring: str = "chemplp"
    scoring_functions: List[str] = field(default_factory=lambda: ["chemplp", "plp", "plp95"])
    rescoring_mode: str = "simplex"


@dataclass
class Dock6Config:
    """Configuration for the Dock6 docking engine.

    Parameters
    ----------
    executable : str, optional
        Path to the Dock6 executable, by default ``""``.
    vdw_defn_file : str, optional
        Path to the van der Waals parameter file, by default ``""``.
    flex_defn_file : str, optional
        Path to the flexible receptor definition file, by default ``""``.
    flex_drive_file : str, optional
        Path to the flexible receptor drive file, by default ``""``.
    """

    executable: str = ""
    vdw_defn_file: str = ""
    flex_defn_file: str = ""
    flex_drive_file: str = ""


@dataclass
class LeDockConfig:
    """Configuration for the LeDock docking engine.

    Parameters
    ----------
    executable : str, optional
        Path to the LeDock executable, by default ``""``.
    lepro : str, optional
        Path to the LePro executable, by default ``""``.
    rmsd : str, optional
        RMSD clustering threshold, by default ``""``.
    num_poses : str, optional
        Number of poses to output, by default ``""``.
    """

    executable: str = ""
    lepro: str = ""
    rmsd: str = ""
    num_poses: str = ""


@dataclass
class ODDTConfig:
    """Configuration for ODDT rescoring functions.

    Parameters
    ----------
    seed : str, optional
        Random seed for ODDT scoring, by default ``""``.
    chunk_size : str, optional
        Batch chunk size for ODDT inference, by default ``""``.
    scoring_functions : list of str, optional
        Enabled ODDT scoring function names, by default empty list.
    """

    seed: str = ""
    chunk_size: str = ""
    scoring_functions: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Database connection configuration.

    Parameters
    ----------
    backend : str, optional
        Database backend (``postgresql``, ``mysql``, ``sqlite``), by default ``"postgresql"``.
    host : str, optional
        Database host, by default ``""``.
    user : str, optional
        Database user name, by default ``""``.
    password : str, optional
        Database password, by default ``""``.
    database : str, optional
        Database name, by default ``""``.
    optimizedb : str, optional
        Optimized database identifier, by default ``""``.
    port : int, optional
        Database port, by default ``None``.
    sqlite_path : str, optional
        Filesystem path for SQLite backend, by default ``""``.
    """

    backend: str = "postgresql"
    host: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    optimizedb: str = ""
    port: Optional[int] = None
    sqlite_path: str = ""


@dataclass
class ToolsConfig:
    """Configuration for external helper tools.

    Parameters
    ----------
    pythonsh : str, optional
        Python interpreter used by MGLTools scripts, by default ``"pythonsh"``.
    prepare_ligand : str, optional
        MGLTools ligand preparation script, by default ``"prepare_ligand4.py"``.
    prepare_receptor : str, optional
        MGLTools receptor preparation script, by default ``"prepare_receptor4.py"``.
    chimera : str, optional
        UCSF Chimera executable path, by default ``""``.
    dssp : str, optional
        DSSP executable for secondary structure, by default ``"dssp"``.
    obabel : str, optional
        Open Babel executable, by default ``"obabel"``.
    spores : str, optional
        SPORES executable for receptor preparation, by default ``"spores"``.
    dudez_download : str, optional
        DUDE-Z download helper path, by default ``""``.
    """

    pythonsh: str = "pythonsh"
    prepare_ligand: str = "prepare_ligand4.py"
    prepare_receptor: str = "prepare_receptor4.py"
    chimera: str = ""
    dssp: str = "dssp"
    obabel: str = "obabel"
    spores: str = "spores"
    dudez_download: str = ""


@dataclass
class PathsConfig:
    """Path configuration for datasets and reference files.

    Parameters
    ----------
    ocdb_path : str, optional
        Root path to the OCDocker database bundle, by default ``""``.
    pca_path : str, optional
        Path to PCA model artifacts, by default ``""``.
    pdbbind_kdki_order : str, optional
        PDBbind Kd/Ki ordering flag, by default ``"u"``.
    reference_column_order : list of str, optional
        Reference column order for mask application, by default empty list.
    """

    ocdb_path: str = ""
    pca_path: str = ""
    pdbbind_kdki_order: str = "u"
    reference_column_order: List[str] = field(default_factory=list)  # Column order list for mask application


@dataclass
class OCDockerConfig:
    """Main configuration object for OCDocker.

    Encapsulates docking engines, database, tools, paths, and runtime settings.
    Prefer :func:`get_config` for the process-wide singleton instance.

    Parameters
    ----------
    vina : VinaConfig, optional
        AutoDock Vina settings, by default factory ``VinaConfig``.
    smina : SminaConfig, optional
        Smina settings, by default factory ``SminaConfig``.
    gnina : GninaConfig, optional
        Gnina settings, by default factory ``GninaConfig``.
    plants : PLANTSConfig, optional
        PLANTS settings, by default factory ``PLANTSConfig``.
    dock6 : Dock6Config, optional
        Dock6 settings, by default factory ``Dock6Config``.
    ledock : LeDockConfig, optional
        LeDock settings, by default factory ``LeDockConfig``.
    oddt : ODDTConfig, optional
        ODDT rescoring settings, by default factory ``ODDTConfig``.
    database : DatabaseConfig, optional
        Database connection settings, by default factory ``DatabaseConfig``.
    tools : ToolsConfig, optional
        External tool paths, by default factory ``ToolsConfig``.
    paths : PathsConfig, optional
        Dataset and reference paths, by default factory ``PathsConfig``.
    output_level : ReportLevel, optional
        Global logging/report level, by default ``ReportLevel.WARNING``.
    multiprocess : bool, optional
        Enable multiprocessing where supported, by default ``True``.
    overwrite : bool, optional
        Overwrite existing output files, by default ``False``.
    tmp_dir : str, optional
        Temporary working directory, by default ``""``.
    """

    # Docking engines
    vina: VinaConfig = field(default_factory=VinaConfig)
    smina: SminaConfig = field(default_factory=SminaConfig)
    gnina: GninaConfig = field(default_factory=GninaConfig)
    plants: PLANTSConfig = field(default_factory=PLANTSConfig)
    dock6: Dock6Config = field(default_factory=Dock6Config)
    ledock: LeDockConfig = field(default_factory=LeDockConfig)
    oddt: ODDTConfig = field(default_factory=ODDTConfig)

    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Tools
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    # Paths
    paths: PathsConfig = field(default_factory=PathsConfig)

    # General settings
    output_level: ocerror.ReportLevel = ocerror.ReportLevel.WARNING
    multiprocess: bool = True
    overwrite: bool = False
    tmp_dir: str = ""

    # Runtime paths (computed during bootstrap)
    ocdocker_path: str = ""
    dudez_archive: str = ""
    pdbbind_archive: str = ""
    litpcba_archive: str = ""
    parsed_archive: str = ""
    logdir: str = ""
    oddt_models_dir: str = ""
    available_cores: int = 1

    @classmethod
    def from_config_file(cls, config_file: str) -> "OCDockerConfig":
        '''Load configuration from config file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file

        Returns
        -------
        OCDockerConfig
            Configured instance
        '''
        # Import here to avoid circular dependency
        import os
        from OCDocker.Initialise import _parse_config_file

        try:
            from OCDocker.Initialise import _resolve_config_file_path as resolve_config_file_path
        except ImportError:

            def resolve_config_file_path(
                requested_config: Optional[str],
                *,
                include_package_locations: bool = True,
            ) -> str:
                '''Fallback config path resolver used when Initialise cannot be imported.

                Parameters
                ----------
                requested_config : str, optional
                    Caller-requested config path, if any.
                include_package_locations : bool, optional
                    Unused; kept for signature parity with
                    :func:`OCDocker.Initialise._resolve_config_file_path`.

                Returns
                -------
                str
                    Absolute path to the resolved configuration file.

                Raises
                ------
                FileNotFoundError
                    If no configuration file can be found.
                '''

                del include_package_locations  # unused in fallback path
                requested = str(requested_config or "").strip()
                if requested and os.path.isfile(requested):
                    return os.path.abspath(requested)
                if os.path.isfile("OCDocker.cfg"):
                    return os.path.abspath("OCDocker.cfg")
                if os.path.isfile("OCDocker.yml"):
                    return os.path.abspath("OCDocker.yml")
                requested_hint = requested or "<not provided>"
                raise FileNotFoundError(
                    f"No configuration file found. Requested: {requested_hint}. " "Searched: OCDocker.cfg, OCDocker.yml"
                )

        # Resolve config file path if not provided or doesn't exist
        # Bootstrap already resolves the path, so if provided and exists, use it as-is
        if config_file and os.path.isfile(config_file):
            # File exists, use it directly (bootstrap already resolved it).
            config_file = os.path.abspath(config_file)
        else:
            requested = str(config_file or os.getenv("OCDOCKER_CONFIG", "")).strip()
            try:
                config_file = resolve_config_file_path(requested, include_package_locations=False)
            except FileNotFoundError as exc:
                raise FileNotFoundError(str(exc)) from exc

        cfg = _parse_config_file(config_file)

        # Verify cfg is populated - if empty, something went wrong
        if not cfg:
            raise ValueError(f"Configuration file '{config_file}' was parsed but returned an empty dictionary")

        # Build configuration
        config = cls(
            # Vina
            vina=VinaConfig(
                executable=cfg.get("vina", "vina"),
                split_executable=cfg.get("vina_split", "vina_split"),
                energy_range=cfg.get("vina_energy_range", "10"),
                exhaustiveness=_get_exhaustiveness(cfg, "vina_exhaustiveness", 5),
                num_modes=cfg.get("vina_num_modes", "3"),
                scoring=cfg.get("vina_scoring", "vina"),
                scoring_functions=cfg.get("vina_scoring_functions", ["vina"]),
            ),
            # Smina
            smina=SminaConfig(
                executable=cfg.get("smina", "smina"),
                energy_range=cfg.get("smina_energy_range", "10"),
                exhaustiveness=cfg.get("smina_exhaustiveness", "5"),
                num_modes=cfg.get("smina_num_modes", "3"),
                scoring=cfg.get("smina_scoring", "vinardo"),
                scoring_functions=cfg.get("smina_scoring_functions", ["vinardo"]),
                custom_scoring=cfg.get("smina_custom_scoring", "no"),
                custom_atoms=cfg.get("smina_custom_atoms", "no"),
                local_only=cfg.get("smina_local_only", "no"),
                minimize=cfg.get("smina_minimize", "no"),
                randomize_only=cfg.get("smina_randomize_only", "no"),
                minimize_iters=cfg.get("smina_minimize_iters", "0"),
                accurate_line=cfg.get("smina_accurate_line", "no"),
                minimize_early_term=cfg.get("smina_minimize_early_term", "no"),
                approximation=cfg.get("smina_approximation", "spline"),
                factor=cfg.get("smina_factor", "32"),
                force_cap=cfg.get("smina_force_cap", "10"),
                user_grid=cfg.get("smina_user_grid", "no"),
                user_grid_lambda=cfg.get("smina_user_grid_lambda", "no"),
            ),
            # Gnina
            gnina=GninaConfig(
                executable=cfg.get("gnina", "gnina"),
                flex=cfg.get("gnina_flex", "no"),
                flexres=cfg.get("gnina_flexres", "no"),
                flexdist_ligand=cfg.get("gnina_flexdist_ligand", "no"),
                flexdist=cfg.get("gnina_flexdist", "no"),
                flex_limit=cfg.get("gnina_flex_limit", "no"),
                flex_max=cfg.get("gnina_flex_max", "no"),
                autobox_ligand=cfg.get("gnina_autobox_ligand", "no"),
                autobox_add=cfg.get("gnina_autobox_add", "4"),
                autobox_extend=cfg.get("gnina_autobox_extend", "1"),
                no_lig=cfg.get("gnina_no_lig", "no"),
                covalent_rec_atom=cfg.get("gnina_covalent_rec_atom", "no"),
                covalent_lig_atom_pattern=cfg.get("gnina_covalent_lig_atom_pattern", "no"),
                covalent_lig_atom_position=cfg.get("gnina_covalent_lig_atom_position", "no"),
                covalent_fix_lig_atom_position=cfg.get("gnina_covalent_fix_lig_atom_position", "no"),
                covalent_bond_order=cfg.get("gnina_covalent_bond_order", "1"),
                covalent_optimize_lig=cfg.get("gnina_covalent_optimize_lig", "no"),
                exhaustiveness=cfg.get("gnina_exhaustiveness", "8"),
                num_modes=cfg.get("gnina_num_modes", "9"),
                scoring=cfg.get("gnina_scoring", "default"),
                scoring_functions=cfg.get("gnina_scoring_functions", GNINA_DEFAULT_SCORING_FUNCTIONS.copy()),
                custom_scoring=cfg.get("gnina_custom_scoring", "no"),
                custom_atoms=cfg.get("gnina_custom_atoms", "no"),
                score_only=cfg.get("gnina_score_only", "no"),
                local_only=cfg.get("gnina_local_only", "no"),
                minimize=cfg.get("gnina_minimize", "no"),
                randomize_only=cfg.get("gnina_randomize_only", "no"),
                num_mc_steps=cfg.get("gnina_num_mc_steps", "no"),
                max_mc_steps=cfg.get("gnina_max_mc_steps", "no"),
                num_mc_saved=cfg.get("gnina_num_mc_saved", "no"),
                temperature=cfg.get("gnina_temperature", "no"),
                minimize_iters=cfg.get("gnina_minimize_iters", "0"),
                accurate_line=cfg.get("gnina_accurate_line", "no"),
                simple_ascent=cfg.get("gnina_simple_ascent", "no"),
                minimize_early_term=cfg.get("gnina_minimize_early_term", "no"),
                minimize_single_full=cfg.get("gnina_minimize_single_full", "no"),
                approximation=cfg.get("gnina_approximation", "spline"),
                factor=cfg.get("gnina_factor", "32"),
                force_cap=cfg.get("gnina_force_cap", "10"),
                user_grid=cfg.get("gnina_user_grid", "no"),
                user_grid_lambda=cfg.get("gnina_user_grid_lambda", "-1"),
                print_terms=cfg.get("gnina_print_terms", "no"),
                print_atom_types=cfg.get("gnina_print_atom_types", "no"),
                cnn_scoring=cfg.get("gnina_cnn_scoring", "rescore"),
                cnn=cfg.get("gnina_cnn", "default"),
                cnn_models=cfg.get("gnina_cnn_models", GNINA_DEFAULT_CNN_MODELS.copy()),
                cnn_model=cfg.get("gnina_cnn_model", "no"),
                cnn_rotation=cfg.get("gnina_cnn_rotation", "0"),
                cnn_mix_emp_force=cfg.get("gnina_cnn_mix_emp_force", "no"),
                cnn_mix_emp_energy=cfg.get("gnina_cnn_mix_emp_energy", "no"),
                cnn_empirical_weight=cfg.get("gnina_cnn_empirical_weight", "1"),
                cnn_center_x=cfg.get("gnina_cnn_center_x", "no"),
                cnn_center_y=cfg.get("gnina_cnn_center_y", "no"),
                cnn_center_z=cfg.get("gnina_cnn_center_z", "no"),
                cnn_verbose=cfg.get("gnina_cnn_verbose", "no"),
                out_flex=cfg.get("gnina_out_flex", "no"),
                atom_terms=cfg.get("gnina_atom_terms", "no"),
                atom_term_data=cfg.get("gnina_atom_term_data", "no"),
                pose_sort_order=cfg.get("gnina_pose_sort_order", "CNNscore"),
                full_flex_output=cfg.get("gnina_full_flex_output", "no"),
                cpu=cfg.get("gnina_cpu", "auto"),
                seed=cfg.get("gnina_seed", "no"),
                min_rmsd_filter=cfg.get("gnina_min_rmsd_filter", "1"),
                quiet=cfg.get("gnina_quiet", "no"),
                addH=cfg.get("gnina_addH", "yes"),
                stripH=cfg.get("gnina_stripH", "no"),
                device=cfg.get("gnina_device", "0"),
                no_gpu=cfg.get("gnina_no_gpu", "no"),
            ),
            # PLANTS
            plants=PLANTSConfig(
                executable=cfg.get("plants", "plants"),
                cluster_structures=cfg.get("plants_cluster_structures", 3),
                cluster_rmsd=cfg.get("plants_cluster_rmsd", "2.0"),
                search_speed=cfg.get("plants_search_speed", "speed1"),
                scoring=cfg.get("plants_scoring", "chemplp"),
                scoring_functions=cfg.get("plants_scoring_functions", ["chemplp", "plp", "plp95"]),
                rescoring_mode=cfg.get("plants_rescoring_mode", "simplex"),
            ),
            # Dock6
            dock6=Dock6Config(
                executable=cfg.get("dock6", ""),
                vdw_defn_file=cfg.get("dock6_vdw_defn_file", ""),
                flex_defn_file=cfg.get("dock6_flex_defn_file", ""),
                flex_drive_file=cfg.get("dock6_flex_drive_file", ""),
            ),
            # LeDock
            ledock=LeDockConfig(
                executable=cfg.get("ledock", ""),
                lepro=cfg.get("lepro", ""),
                rmsd=cfg.get("ledock_rmsd", ""),
                num_poses=cfg.get("ledock_num_poses", ""),
            ),
            # ODDT
            oddt=ODDTConfig(
                seed=cfg.get("oddt_seed", ""),
                chunk_size=cfg.get("oddt_chunk_size", ""),
                scoring_functions=cfg.get("oddt_scoring_functions", []),
            ),
            # Database
            database=DatabaseConfig(
                backend=cfg.get("DB_BACKEND", "postgresql"),
                host=cfg.get("HOST", ""),
                user=cfg.get("USER", ""),
                password=cfg.get("PASSWORD", ""),
                database=cfg.get("DATABASE", ""),
                optimizedb=cfg.get("OPTIMIZEDB", ""),
                port=cfg.get("PORT", None),
                sqlite_path=cfg.get("SQLITE_PATH", ""),
            ),
            # Tools
            tools=ToolsConfig(
                pythonsh=cfg.get("pythonsh", "pythonsh"),
                prepare_ligand=cfg.get("prepare_ligand", "prepare_ligand4.py"),
                prepare_receptor=cfg.get("prepare_receptor", "prepare_receptor4.py"),
                chimera=cfg.get("chimera", ""),
                dssp=cfg.get("dssp", "dssp"),
                obabel=cfg.get("obabel", "obabel"),
                spores=cfg.get("spores", "spores"),
                dudez_download=cfg.get("DUDEz", ""),
            ),
            # Paths
            paths=PathsConfig(
                ocdb_path=cfg.get("ocdb", ""),
                pca_path=cfg.get("pca", ""),
                pdbbind_kdki_order=cfg.get("pdbbind_KdKi_order", "u"),
                reference_column_order=cfg.get("reference_column_order", DEFAULT_REFERENCE_COLUMN_ORDER.copy()),
            ),
        )

        # Direct attributes (optional)
        if "oddt_models_dir" in cfg:
            config.oddt_models_dir = cfg.get("oddt_models_dir", "")

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OCDockerConfig":
        '''Create configuration from dictionary.

        Useful for testing and programmatic configuration.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration values

        Returns
        -------
        OCDockerConfig
            Configured instance
        '''

        # This is a simplified version - can be expanded as needed
        config = cls()

        # Update from dict if provided
        if "vina" in config_dict:
            config.vina = VinaConfig(**config_dict["vina"])
        if "smina" in config_dict:
            config.smina = SminaConfig(**config_dict["smina"])
        if "gnina" in config_dict:
            config.gnina = GninaConfig(**config_dict["gnina"])
        if "plants" in config_dict:
            config.plants = PLANTSConfig(**config_dict["plants"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])
        if "tools" in config_dict:
            config.tools = ToolsConfig(**config_dict["tools"])
        if "paths" in config_dict:
            config.paths = PathsConfig(**config_dict["paths"])

        # Direct attributes
        if "output_level" in config_dict:
            config.output_level = config_dict["output_level"]
        if "multiprocess" in config_dict:
            config.multiprocess = config_dict["multiprocess"]
        if "overwrite" in config_dict:
            config.overwrite = config_dict["overwrite"]
        if "tmp_dir" in config_dict:
            config.tmp_dir = config_dict["tmp_dir"]
        if "oddt_models_dir" in config_dict:
            config.oddt_models_dir = config_dict["oddt_models_dir"]

        return config


# Functions
###############################################################################
## Private ##


def _get_exhaustiveness(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Get exhaustiveness value from configuration, handling both int and str types.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed configuration dictionary.
    key : str
        The configuration key to retrieve.
    default : Any
        The default value to return if the key is not found or conversion fails.

    Returns
    -------
    Any
        The exhaustiveness value as int if convertible, otherwise as str. Returns default if key not found.
    """

    val = cfg.get(key, default)
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (ValueError, TypeError):
        return str(val)


# Singleton Pattern
_config_lock = threading.Lock()
_config_instance: Optional[OCDockerConfig] = None


## Public ##
def get_config() -> OCDockerConfig:
    '''Get the global configuration instance (singleton pattern).

    Returns
    -------
    OCDockerConfig
        The global configuration instance

    Note
    ----
    If no configuration has been set, returns a default configuration.
    For proper initialization, call set_config() or bootstrap from Initialise.
    '''

    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                # Return default config if not initialized
                # This allows the Config module to be imported before bootstrap
                _config_instance = OCDockerConfig()
    return _config_instance


def reset_config() -> None:
    '''Reset the global configuration to None.

    Useful for testing to ensure clean state.
    '''

    global _config_instance
    with _config_lock:
        _config_instance = None


def set_config(config: OCDockerConfig) -> None:
    '''Set the global configuration (useful for testing).

    Parameters
    ----------
    config : OCDockerConfig
        Configuration instance to set as global

    Note
    ----
    This function is thread-safe and can be used to override
    the global configuration, particularly useful in tests.
    '''

    global _config_instance
    with _config_lock:
        _config_instance = config
