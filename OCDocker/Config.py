#!/usr/bin/env python3

# Description
###############################################################################
'''
Configuration management for OCDocker using dataclasses and singleton pattern.

This module provides a structured way to manage OCDocker configuration,
replacing the global variables in Initialise.py with type-safe dataclasses.

They are imported as:

from OCDocker.Config import get_config, OCDockerConfig
'''

# Imports
###############################################################################
import os
import threading
import configparser
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

import OCDocker.Error as ocerror

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

# Configuration Dataclasses
###############################################################################


@dataclass
class VinaConfig:
    """Configuration for Vina docking engine."""
    executable: str = "vina"
    split_executable: str = "vina_split"
    energy_range: str = "10"
    exhaustiveness: Any = 5  # Can be int or str depending on config file
    num_modes: str = "3"
    scoring: str = "vina"
    scoring_functions: List[str] = field(default_factory=lambda: ["vina"])


@dataclass
class SminaConfig:
    """Configuration for Smina docking engine."""
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
    """Configuration for Gnina docking engine."""
    executable: str = "gnina"
    exhaustiveness: str = ""
    num_modes: str = ""
    scoring: str = ""
    custom_scoring: str = ""
    custom_atoms: str = ""
    local_only: str = ""
    minimize: str = ""
    randomize_only: str = ""
    num_mc_steps: str = ""
    max_mc_steps: str = ""
    num_mc_saved: str = ""
    minimize_iters: str = ""
    simple_ascent: str = ""
    accurate_line: str = ""
    minimize_early_term: str = ""
    approximation: str = ""
    factor: str = ""
    force_cap: str = ""
    user_grid: str = ""
    user_grid_lambda: str = ""
    no_gpu: str = ""


@dataclass
class PLANTSConfig:
    """Configuration for PLANTS docking engine."""
    executable: str = "plants"
    cluster_structures: int = 3
    cluster_rmsd: str = "2.0"
    search_speed: str = "speed1"
    scoring: str = "chemplp"
    scoring_functions: List[str] = field(default_factory=lambda: ["chemplp", "plp", "plp95"])
    rescoring_mode: str = "simplex"


@dataclass
class Dock6Config:
    """Configuration for Dock6 docking engine."""
    executable: str = ""
    vdw_defn_file: str = ""
    flex_defn_file: str = ""
    flex_drive_file: str = ""


@dataclass
class LeDockConfig:
    """Configuration for LeDock docking engine."""
    executable: str = ""
    lepro: str = ""
    rmsd: str = ""
    num_poses: str = ""


@dataclass
class ODDTConfig:
    """Configuration for ODDT scoring functions."""
    executable: str = ""
    seed: str = ""
    chunk_size: str = ""
    scoring_functions: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    optimizedb: str = ""
    port: Optional[int] = 3306
    use_sqlite: str = ""
    sqlite_path: str = ""


@dataclass
class ToolsConfig:
    """Configuration for external tools."""
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
    """Path configuration."""
    ocdb_path: str = ""
    pca_path: str = ""
    pdbbind_kdki_order: str = "u"
    reference_column_order: List[str] = field(default_factory=list)  # Column order list for mask application


@dataclass
class OCDockerConfig:
    """Main configuration object for OCDocker.
    
    This class encapsulates all configuration settings for OCDocker,
    replacing the global variables in Initialise.py.
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
    parsed_archive: str = ""
    logdir: str = ""
    oddt_models_dir: str = ""
    available_cores: int = 1
    
    @classmethod
    def from_config_file(cls, config_file: str) -> 'OCDockerConfig':
        """Load configuration from config file.
        
        Parameters
        ----------
        config_file : str
            Path to the configuration file
            
        Returns
        -------
        OCDockerConfig
            Configured instance
        """
        # Import here to avoid circular dependency
        import os
        from OCDocker.Initialise import _parse_config_file
        
        # Resolve config file path if not provided or doesn't exist
        # Bootstrap already resolves the path, so if provided and exists, use it as-is
        if config_file and os.path.isfile(config_file):
            # File exists, use it directly (bootstrap already resolved it)
            pass
        else:
            # File doesn't exist or not provided, try to find it
            if not config_file:
                config_file = os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg')
            
            # Try to find the file
            if not os.path.isfile(config_file):
                if os.path.isfile("OCDocker.cfg"):
                    config_file = os.path.abspath("OCDocker.cfg")
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_file}")
            else:
                # Convert to absolute path for consistency
                config_file = os.path.abspath(config_file)
        
        cfg = _parse_config_file(config_file)
        
        # Verify cfg is populated - if empty, something went wrong
        if not cfg:
            raise ValueError(f"Configuration file '{config_file}' was parsed but returned an empty dictionary")
        
        # Helper to convert string to bool
        def str_to_bool(val: str) -> bool:
            '''Convert a string value to a boolean.
            
            Parameters
            ----------
            val : str
                The string value to convert. Accepts '1', 'true', 'yes', 'y', 'on' (case-insensitive).
            
            Returns
            -------
            bool
                True if the value is a recognized truthy string, False otherwise.
            '''

            return str(val).lower() in ('1', 'true', 'yes', 'y', 'on')
        
        # Helper to convert exhaustiveness (can be int or str)
        def get_exhaustiveness(key: str, default: Any) -> Any:
            '''Get exhaustiveness value from configuration, handling both int and str types.
            
            Parameters
            ----------
            key : str
                The configuration key to retrieve.
            default : Any
                The default value to return if the key is not found or conversion fails.
            
            Returns
            -------
            Any
                The exhaustiveness value as int if convertible, otherwise as str. Returns default if key not found.
            '''
            
            val = cfg.get(key, default)
            if isinstance(val, int):
                return val
            try:
                return int(val)
            except (ValueError, TypeError):
                return str(val)
        
        # Build configuration
        config = cls(
            # Vina
            vina=VinaConfig(
                executable=cfg.get('vina', 'vina'),
                split_executable=cfg.get('vina_split', 'vina_split'),
                energy_range=cfg.get('vina_energy_range', '10'),
                exhaustiveness=get_exhaustiveness('vina_exhaustiveness', 5),
                num_modes=cfg.get('vina_num_modes', '3'),
                scoring=cfg.get('vina_scoring', 'vina'),
                scoring_functions=cfg.get('vina_scoring_functions', ['vina']),
            ),
            
            # Smina
            smina=SminaConfig(
                executable=cfg.get('smina', 'smina'),
                energy_range=cfg.get('smina_energy_range', '10'),
                exhaustiveness=cfg.get('smina_exhaustiveness', '5'),
                num_modes=cfg.get('smina_num_modes', '3'),
                scoring=cfg.get('smina_scoring', 'vinardo'),
                scoring_functions=cfg.get('smina_scoring_functions', ['vinardo']),
                custom_scoring=cfg.get('smina_custom_scoring', 'no'),
                custom_atoms=cfg.get('smina_custom_atoms', 'no'),
                local_only=cfg.get('smina_local_only', 'no'),
                minimize=cfg.get('smina_minimize', 'no'),
                randomize_only=cfg.get('smina_randomize_only', 'no'),
                minimize_iters=cfg.get('smina_minimize_iters', '0'),
                accurate_line=cfg.get('smina_accurate_line', 'no'),
                minimize_early_term=cfg.get('smina_minimize_early_term', 'no'),
                approximation=cfg.get('smina_approximation', 'spline'),
                factor=cfg.get('smina_factor', '32'),
                force_cap=cfg.get('smina_force_cap', '10'),
                user_grid=cfg.get('smina_user_grid', 'no'),
                user_grid_lambda=cfg.get('smina_user_grid_lambda', 'no'),
            ),
            
            # Gnina
            gnina=GninaConfig(
                executable=cfg.get('gnina', 'gnina'),
                exhaustiveness=cfg.get('gnina_exhaustiveness', ''),
                num_modes=cfg.get('gnina_num_modes', ''),
                scoring=cfg.get('gnina_scoring', ''),
                custom_scoring=cfg.get('gnina_custom_scoring', ''),
                custom_atoms=cfg.get('gnina_custom_atoms', ''),
                local_only=cfg.get('gnina_local_only', ''),
                minimize=cfg.get('gnina_minimize', ''),
                randomize_only=cfg.get('gnina_randomize_only', ''),
                num_mc_steps=cfg.get('gnina_num_mc_steps', ''),
                max_mc_steps=cfg.get('gnina_max_mc_steps', ''),
                num_mc_saved=cfg.get('gnina_num_mc_saved', ''),
                minimize_iters=cfg.get('gnina_minimize_iters', ''),
                simple_ascent=cfg.get('gnina_simple_ascent', ''),
                accurate_line=cfg.get('gnina_accurate_line', ''),
                minimize_early_term=cfg.get('gnina_minimize_early_term', ''),
                approximation=cfg.get('gnina_approximation', ''),
                factor=cfg.get('gnina_factor', ''),
                force_cap=cfg.get('gnina_force_cap', ''),
                user_grid=cfg.get('gnina_user_grid', ''),
                user_grid_lambda=cfg.get('gnina_user_grid_lambda', ''),
                no_gpu=cfg.get('gnina_no_gpu', ''),
            ),
            
            # PLANTS
            plants=PLANTSConfig(
                executable=cfg.get('plants', 'plants'),
                cluster_structures=cfg.get('plants_cluster_structures', 3),
                cluster_rmsd=cfg.get('plants_cluster_rmsd', '2.0'),
                search_speed=cfg.get('plants_search_speed', 'speed1'),
                scoring=cfg.get('plants_scoring', 'chemplp'),
                scoring_functions=cfg.get('plants_scoring_functions', ['chemplp', 'plp', 'plp95']),
                rescoring_mode=cfg.get('plants_rescoring_mode', 'simplex'),
            ),
            
            # Dock6
            dock6=Dock6Config(
                executable=cfg.get('dock6', ''),
                vdw_defn_file=cfg.get('dock6_vdw_defn_file', ''),
                flex_defn_file=cfg.get('dock6_flex_defn_file', ''),
                flex_drive_file=cfg.get('dock6_flex_drive_file', ''),
            ),
            
            # LeDock
            ledock=LeDockConfig(
                executable=cfg.get('ledock', ''),
                lepro=cfg.get('lepro', ''),
                rmsd=cfg.get('ledock_rmsd', ''),
                num_poses=cfg.get('ledock_num_poses', ''),
            ),
            
            # ODDT
            oddt=ODDTConfig(
                executable=cfg.get('oddt', ''),
                seed=cfg.get('oddt_seed', ''),
                chunk_size=cfg.get('oddt_chunk_size', ''),
                scoring_functions=cfg.get('oddt_scoring_functions', []),
            ),
            
            # Database
            database=DatabaseConfig(
                host=cfg.get('HOST', ''),
                user=cfg.get('USER', ''),
                password=cfg.get('PASSWORD', ''),
                database=cfg.get('DATABASE', ''),
                optimizedb=cfg.get('OPTIMIZEDB', ''),
                port=cfg.get('PORT', 3306),
                use_sqlite=cfg.get('USE_SQLITE', ''),
                sqlite_path=cfg.get('SQLITE_PATH', ''),
            ),
            
            # Tools
            tools=ToolsConfig(
                pythonsh=cfg.get('pythonsh', 'pythonsh'),
                prepare_ligand=cfg.get('prepare_ligand', 'prepare_ligand4.py'),
                prepare_receptor=cfg.get('prepare_receptor', 'prepare_receptor4.py'),
                chimera=cfg.get('chimera', ''),
                dssp=cfg.get('dssp', 'dssp'),
                obabel=cfg.get('obabel', 'obabel'),
                spores=cfg.get('spores', 'spores'),
                dudez_download=cfg.get('DUDEz', ''),
            ),
            
            # Paths
            paths=PathsConfig(
                ocdb_path=cfg.get('ocdb', ''),
                pca_path=cfg.get('pca', ''),
                pdbbind_kdki_order=cfg.get('pdbbind_KdKi_order', 'u'),
                reference_column_order=cfg.get('reference_column_order', ['name','receptor','ligand','SMINA_VINA','SMINA_SCORING_DKOES','SMINA_VINARDO','SMINA_OLD_SCORING_DKOES','SMINA_FAST_DKOES','SMINA_SCORING_AD4','VINA_VINA','VINA_VINARDO','PLANTS_CHEMPLP','PLANTS_PLP','PLANTS_PLP95','ODDT_RFSCORE_V1','ODDT_RFSCORE_V2','ODDT_RFSCORE_V3','ODDT_PLECRF_P5_L1_S65536','ODDT_NNSCORE','countA','countR','countN','countD','countC','countQ','countE','countG','countH','countI','countL','countK','countM','countF','countP','countS','countT','countW','countY','countV','TotalAALength','AvgAALength','countChain','SASA','DipoleMoment','IsoelectricPoint','GRAVY','Aromaticity','InstabilityIndex','AUTOCORR2D_1','AUTOCORR2D_2','AUTOCORR2D_3','AUTOCORR2D_4','AUTOCORR2D_5','AUTOCORR2D_6','AUTOCORR2D_7','AUTOCORR2D_8','AUTOCORR2D_9','AUTOCORR2D_10','AUTOCORR2D_11','AUTOCORR2D_12','AUTOCORR2D_13','AUTOCORR2D_14','AUTOCORR2D_15','AUTOCORR2D_16','AUTOCORR2D_17','AUTOCORR2D_18','AUTOCORR2D_19','AUTOCORR2D_20','AUTOCORR2D_21','AUTOCORR2D_22','AUTOCORR2D_23','AUTOCORR2D_24','AUTOCORR2D_25','AUTOCORR2D_26','AUTOCORR2D_27','AUTOCORR2D_28','AUTOCORR2D_29','AUTOCORR2D_30','AUTOCORR2D_31','AUTOCORR2D_32','AUTOCORR2D_33','AUTOCORR2D_34','AUTOCORR2D_35','AUTOCORR2D_36','AUTOCORR2D_37','AUTOCORR2D_38','AUTOCORR2D_39','AUTOCORR2D_40','AUTOCORR2D_41','AUTOCORR2D_42','AUTOCORR2D_43','AUTOCORR2D_44','AUTOCORR2D_45','AUTOCORR2D_46','AUTOCORR2D_47','AUTOCORR2D_48','AUTOCORR2D_49','AUTOCORR2D_50','AUTOCORR2D_51','AUTOCORR2D_52','AUTOCORR2D_53','AUTOCORR2D_54','AUTOCORR2D_55','AUTOCORR2D_56','AUTOCORR2D_57','AUTOCORR2D_58','AUTOCORR2D_59','AUTOCORR2D_60','AUTOCORR2D_61','AUTOCORR2D_62','AUTOCORR2D_63','AUTOCORR2D_64','AUTOCORR2D_65','AUTOCORR2D_66','AUTOCORR2D_67','AUTOCORR2D_68','AUTOCORR2D_69','AUTOCORR2D_70','AUTOCORR2D_71','AUTOCORR2D_72','AUTOCORR2D_73','AUTOCORR2D_74','AUTOCORR2D_75','AUTOCORR2D_76','AUTOCORR2D_77','AUTOCORR2D_78','AUTOCORR2D_79','AUTOCORR2D_80','AUTOCORR2D_81','AUTOCORR2D_82','AUTOCORR2D_83','AUTOCORR2D_84','AUTOCORR2D_85','AUTOCORR2D_86','AUTOCORR2D_87','AUTOCORR2D_88','AUTOCORR2D_89','AUTOCORR2D_90','AUTOCORR2D_91','AUTOCORR2D_92','AUTOCORR2D_93','AUTOCORR2D_94','AUTOCORR2D_95','AUTOCORR2D_96','AUTOCORR2D_97','AUTOCORR2D_98','AUTOCORR2D_99','AUTOCORR2D_100','AUTOCORR2D_101','AUTOCORR2D_102','AUTOCORR2D_103','AUTOCORR2D_104','AUTOCORR2D_105','AUTOCORR2D_106','AUTOCORR2D_107','AUTOCORR2D_108','AUTOCORR2D_109','AUTOCORR2D_110','AUTOCORR2D_111','AUTOCORR2D_112','AUTOCORR2D_113','AUTOCORR2D_114','AUTOCORR2D_115','AUTOCORR2D_116','AUTOCORR2D_117','AUTOCORR2D_118','AUTOCORR2D_119','AUTOCORR2D_120','AUTOCORR2D_121','AUTOCORR2D_122','AUTOCORR2D_123','AUTOCORR2D_124','AUTOCORR2D_125','AUTOCORR2D_126','AUTOCORR2D_127','AUTOCORR2D_128','AUTOCORR2D_129','AUTOCORR2D_130','AUTOCORR2D_131','AUTOCORR2D_132','AUTOCORR2D_133','AUTOCORR2D_134','AUTOCORR2D_135','AUTOCORR2D_136','AUTOCORR2D_137','AUTOCORR2D_138','AUTOCORR2D_139','AUTOCORR2D_140','AUTOCORR2D_141','AUTOCORR2D_142','AUTOCORR2D_143','AUTOCORR2D_144','AUTOCORR2D_145','AUTOCORR2D_146','AUTOCORR2D_147','AUTOCORR2D_148','AUTOCORR2D_149','AUTOCORR2D_150','AUTOCORR2D_151','AUTOCORR2D_152','AUTOCORR2D_153','AUTOCORR2D_154','AUTOCORR2D_155','AUTOCORR2D_156','AUTOCORR2D_157','AUTOCORR2D_158','AUTOCORR2D_159','AUTOCORR2D_160','AUTOCORR2D_161','AUTOCORR2D_162','AUTOCORR2D_163','AUTOCORR2D_164','AUTOCORR2D_165','AUTOCORR2D_166','AUTOCORR2D_167','AUTOCORR2D_168','AUTOCORR2D_169','AUTOCORR2D_170','AUTOCORR2D_171','AUTOCORR2D_172','AUTOCORR2D_173','AUTOCORR2D_174','AUTOCORR2D_175','AUTOCORR2D_176','AUTOCORR2D_177','AUTOCORR2D_178','AUTOCORR2D_179','AUTOCORR2D_180','AUTOCORR2D_181','AUTOCORR2D_182','AUTOCORR2D_183','AUTOCORR2D_184','AUTOCORR2D_185','AUTOCORR2D_186','AUTOCORR2D_187','AUTOCORR2D_188','AUTOCORR2D_189','AUTOCORR2D_190','AUTOCORR2D_191','AUTOCORR2D_192','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW','BCUT2D_MWHI','BCUT2D_MWLOW','fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N','fr_Ar_NH','fr_Ar_OH','fr_COO','fr_COO2','fr_C_O','fr_C_O_noCOO','fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2','fr_N_O','fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_SH','fr_aldehyde','fr_alkyl_carbamate','fr_alkyl_halide','fr_allylic_oxid','fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl','fr_azide','fr_azo','fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic','fr_diazo','fr_dihydropyridine','fr_epoxide','fr_ester','fr_ether','fr_furan','fr_guanido','fr_halogen','fr_hdrzine','fr_hdrzone','fr_imidazole','fr_imide','fr_isocyan','fr_isothiocyan','fr_ketone','fr_ketone_Topliss','fr_lactam','fr_lactone','fr_methoxy','fr_morpholine','fr_nitrile','fr_nitro','fr_nitro_arom','fr_nitro_arom_nonortho','fr_nitroso','fr_oxazole','fr_oxime','fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond','fr_phos_acid','fr_phos_ester','fr_piperdine','fr_piperzine','fr_priamide','fr_prisulfonamd','fr_pyridine','fr_quatN','fr_sulfide','fr_sulfonamd','fr_sulfone','fr_term_acetylene','fr_tetrazole','fr_thiazole','fr_thiocyan','fr_thiophene','fr_unbrch_alkane','fr_urea','Chi0','Chi0v','Chi0n','Chi1','Chi1v','Chi1n','Chi2v','Chi2n','Chi3v','Chi3n','Chi4v','Chi4n','EState_VSA1','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9','EState_VSA10','EState_VSA11','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','Kappa1','Kappa2','Kappa3','MolLogP','MolMR','MolWt','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRadicalElectrons','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons','NPR1','NPR2','PMI1','PMI2','PMI3','PEOE_VSA1','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','SMR_VSA1','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SMR_VSA10','SlogP_VSA1','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','VSA_EState1','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9','VSA_EState10','BalabanJ','BertzCT','ExactMolWt','FractionCSP3','HallKierAlpha','HeavyAtomMolWt','HeavyAtomCount','LabuteASA','TPSA','MaxAbsEStateIndex','MaxEStateIndex','MinAbsEStateIndex','MinEStateIndex','MaxAbsPartialCharge','MaxPartialCharge','MinAbsPartialCharge','MinPartialCharge','qed','RingCount','Asphericity','Eccentricity','InertialShapeFactor','RadiusOfGyration','SpherocityIndex','NHOHCount','NOCount']),
            ),
        )
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OCDockerConfig':
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
        if 'vina' in config_dict:
            config.vina = VinaConfig(**config_dict['vina'])
        if 'smina' in config_dict:
            config.smina = SminaConfig(**config_dict['smina'])
        if 'gnina' in config_dict:
            config.gnina = GninaConfig(**config_dict['gnina'])
        if 'plants' in config_dict:
            config.plants = PLANTSConfig(**config_dict['plants'])
        if 'database' in config_dict:
            config.database = DatabaseConfig(**config_dict['database'])
        if 'tools' in config_dict:
            config.tools = ToolsConfig(**config_dict['tools'])
        if 'paths' in config_dict:
            config.paths = PathsConfig(**config_dict['paths'])
        
        # Direct attributes
        if 'output_level' in config_dict:
            config.output_level = config_dict['output_level']
        if 'multiprocess' in config_dict:
            config.multiprocess = config_dict['multiprocess']
        if 'overwrite' in config_dict:
            config.overwrite = config_dict['overwrite']
        if 'tmp_dir' in config_dict:
            config.tmp_dir = config_dict['tmp_dir']
        
        return config


# Singleton Pattern
###############################################################################

_config_lock = threading.Lock()
_config_instance: Optional[OCDockerConfig] = None


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


def reset_config() -> None:
    '''Reset the global configuration to None.
    
    Useful for testing to ensure clean state.
    '''
    
    global _config_instance
    with _config_lock:
        _config_instance = None
