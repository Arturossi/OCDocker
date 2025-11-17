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
        from OCDocker.Initialise import _parse_config_file
        
        cfg = _parse_config_file(config_file)
        
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

