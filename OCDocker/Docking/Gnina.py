#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to prepare gnina files and run it.

Usage:

import OCDocker.Docking.Gnina as ocgnina
'''

# Imports
###############################################################################
import os
import shutil

from glob import glob
from typing import Any, Dict, List, Tuple, Union

import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

from OCDocker.Config import get_config
from OCDocker.Docking.BaseVinaLike import (
    generate_gnina_digest as generate_digest,
    get_gnina_docked_poses as get_docked_poses,
    read_gnina_log as read_log,
    read_gnina_rescoring_log as read_rescoring_log,
)
from OCDocker.Toolbox.Preparation import OpenBabelPreparationStrategy


# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''


# Classes
###############################################################################
_GNINA_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_GNINA_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}

_GNINA_FLAG_OPTIONS = [
    ("score_only", "--score_only"),
    ("local_only", "--local_only"),
    ("minimize", "--minimize"),
    ("randomize_only", "--randomize_only"),
    ("accurate_line", "--accurate_line"),
    ("simple_ascent", "--simple_ascent"),
    ("minimize_early_term", "--minimize_early_term"),
    ("minimize_single_full", "--minimize_single_full"),
    ("print_terms", "--print_terms"),
    ("print_atom_types", "--print_atom_types"),
    ("cnn_mix_emp_force", "--cnn_mix_emp_force"),
    ("cnn_mix_emp_energy", "--cnn_mix_emp_energy"),
    ("cnn_verbose", "--cnn_verbose"),
    ("atom_term_data", "--atom_term_data"),
    ("full_flex_output", "--full_flex_output"),
    ("quiet", "--quiet"),
    ("no_lig", "--no_lig"),
    ("covalent_fix_lig_atom_position", "--covalent_fix_lig_atom_position"),
    ("covalent_optimize_lig", "--covalent_optimize_lig"),
    ("no_gpu", "--no_gpu"),
]

_GNINA_BOOL_VALUE_OPTIONS = [
    ("addH", "--addH"),
    ("stripH", "--stripH"),
]

_GNINA_VALUE_OPTIONS = [
    ("flex", "--flex", True, False),
    ("flexres", "--flexres", True, False),
    ("flexdist_ligand", "--flexdist_ligand", True, False),
    ("flexdist", "--flexdist", True, False),
    ("flex_limit", "--flex_limit", True, False),
    ("flex_max", "--flex_max", True, False),
    ("covalent_rec_atom", "--covalent_rec_atom", True, False),
    ("covalent_lig_atom_pattern", "--covalent_lig_atom_pattern", True, False),
    ("covalent_lig_atom_position", "--covalent_lig_atom_position", True, False),
    ("covalent_bond_order", "--covalent_bond_order", False, False),
    ("scoring", "--scoring", False, False),
    ("custom_scoring", "--custom_scoring", True, False),
    ("custom_atoms", "--custom_atoms", True, False),
    ("num_mc_steps", "--num_mc_steps", True, False),
    ("max_mc_steps", "--max_mc_steps", True, False),
    ("num_mc_saved", "--num_mc_saved", True, False),
    ("temperature", "--temperature", True, False),
    ("minimize_iters", "--minimize_iters", False, False),
    ("approximation", "--approximation", False, False),
    ("factor", "--factor", False, False),
    ("force_cap", "--force_cap", False, False),
    ("user_grid", "--user_grid", True, False),
    ("user_grid_lambda", "--user_grid_lambda", False, False),
    ("cnn_scoring", "--cnn_scoring", False, False),
    ("cnn", "--cnn", False, False),
    ("cnn_model", "--cnn_model", True, False),
    ("cnn_rotation", "--cnn_rotation", False, False),
    ("cnn_empirical_weight", "--cnn_empirical_weight", False, False),
    ("cnn_center_x", "--cnn_center_x", True, False),
    ("cnn_center_y", "--cnn_center_y", True, False),
    ("cnn_center_z", "--cnn_center_z", True, False),
    ("out_flex", "--out_flex", True, False),
    ("atom_terms", "--atom_terms", True, False),
    ("pose_sort_order", "--pose_sort_order", False, False),
    ("cpu", "--cpu", True, True),
    ("seed", "--seed", True, False),
    ("exhaustiveness", "--exhaustiveness", False, False),
    ("num_modes", "--num_modes", False, False),
    ("min_rmsd_filter", "--min_rmsd_filter", False, False),
    ("device", "--device", False, False),
]


def _as_text(value: Union[str, int, float, bool, None]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_true(value: Union[str, int, float, bool, None]) -> bool:
    txt = _as_text(value).lower()
    return txt in _GNINA_TRUE_VALUES


def _is_false(value: Union[str, int, float, bool, None]) -> bool:
    txt = _as_text(value).lower()
    return txt in _GNINA_FALSE_VALUES


def _append_option(
    cmd: List[str],
    flag: str,
    value: Union[str, int, float, bool, None],
    skip_no: bool = False,
    skip_auto: bool = False,
) -> None:
    txt = _as_text(value)
    if not txt:
        return
    txt_low = txt.lower()
    if skip_no and txt_low == "no":
        return
    if skip_auto and txt_low == "auto":
        return
    cmd.extend([flag, txt])


def _resolve_autobox_ligand(value: Union[str, int, float, bool, None], prepared_ligand: str) -> str:
    txt = _as_text(value)
    if not txt:
        return ""
    txt_low = txt.lower()
    if txt_low in {"no", "none", "off", "false", "0"}:
        return ""
    if txt_low in {"yes", "y", "true", "1", "auto", "ligand", "prepared_ligand"}:
        return prepared_ligand
    return txt


def _normalize_string_list(values: Union[List[str], Tuple[str, ...], None], fallback: List[str]) -> List[str]:
    items: List[str] = []
    if isinstance(values, list):
        source_values = values
    elif isinstance(values, tuple):
        source_values = list(values)
    else:
        source_values = []

    for item in source_values:
        item_txt = _as_text(item)
        if item_txt:
            items.append(item_txt)

    if items:
        return items

    return [_as_text(item) for item in fallback if _as_text(item)]


def _get_rescore_scoring_functions(config: Any) -> List[str]:
    default_scoring = _as_text(getattr(config.gnina, "scoring", "default")) or "default"
    scoring_functions = _normalize_string_list(
        getattr(config.gnina, "scoring_functions", None),
        [default_scoring],
    )

    return scoring_functions if scoring_functions else [default_scoring]


def _get_rescore_cnn_models(config: Any) -> List[str]:
    default_cnn = _as_text(getattr(config.gnina, "cnn", "default")) or "default"
    cnn_models = _normalize_string_list(
        getattr(config.gnina, "cnn_models", None),
        [default_cnn],
    )

    return cnn_models if cnn_models else [default_cnn]


def _build_gnina_cmd(config_path: str, prepared_ligand: str, output_gnina: str, gnina_log: str) -> List[str]:
    cfg = get_config()
    gnina_cfg = cfg.gnina
    cmd = [gnina_cfg.executable, "--config", config_path, "--ligand", prepared_ligand]

    for attr_name, flag in _GNINA_FLAG_OPTIONS:
        if _is_true(getattr(gnina_cfg, attr_name, "no")):
            cmd.append(flag)

    for attr_name, flag in _GNINA_BOOL_VALUE_OPTIONS:
        value = getattr(gnina_cfg, attr_name, "")
        if _is_true(value):
            cmd.extend([flag, "1"])
        elif _is_false(value):
            cmd.extend([flag, "0"])
        elif _as_text(value):
            cmd.extend([flag, _as_text(value)])

    # Keep autobox behavior explicit and ligand-centered:
    # disabled by default, and if enabled can follow the current prepared ligand
    autobox_ligand = _resolve_autobox_ligand(getattr(gnina_cfg, "autobox_ligand", ""), prepared_ligand)
    if autobox_ligand:
        cmd.extend(["--autobox_ligand", autobox_ligand])
        _append_option(cmd, "--autobox_add", getattr(gnina_cfg, "autobox_add", ""), skip_no=True)
        _append_option(cmd, "--autobox_extend", getattr(gnina_cfg, "autobox_extend", ""), skip_no=True)

    no_gpu_enabled = _is_true(getattr(gnina_cfg, "no_gpu", "no"))
    for attr_name, flag, skip_no, skip_auto in _GNINA_VALUE_OPTIONS:
        if attr_name == "device" and no_gpu_enabled:
            continue
        _append_option(cmd, flag, getattr(gnina_cfg, attr_name, ""), skip_no=skip_no, skip_auto=skip_auto)

    cmd.extend(["--out", output_gnina, "--log", gnina_log])
    return cmd


class Gnina:
    """Gnina object with methods for easy run."""
    ## Private ##

    def __init__(
        self,
        config_path: str,
        box_file: str,
        receptor: ocr.Receptor,
        prepared_receptor_path: str,
        ligand: ocl.Ligand,
        prepared_ligand_path: str,
        gnina_log: str,
        output_gnina: str,
        name: str = "",
        overwrite_config: bool = False,
    ) -> None:
        '''Constructor of the class Gnina.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        box_file : str
            The path for the box file.
        receptor : ocr.Receptor
            The receptor object.
        prepared_receptor_path : str
            Path to the prepared receptor.
        ligand : ocl.Ligand
            The ligand object.
        prepared_ligand_path : str
            Path to the prepared ligand.
        gnina_log : str
            Path to the gnina log file.
        output_gnina : str
            Path to the output gnina file.
        name : str, optional
            Name of the gnina object, by default "".
        overwrite_config : bool, optional
            If the config file should be overwritten, by default False.
        '''

        self.name = str(name)
        self.config = str(config_path)
        self.box_file = str(box_file)

        # Receptor
        if isinstance(receptor, ocr.Receptor):
            self.input_receptor = receptor
        else:
            msg = (
                f"The receptor '{receptor}' has not a supported type. "
                f"Expected 'ocr.Receptor' but got {type(receptor)} instead."
            )
            ocerror.Error.wrong_type(msg, level=ocerror.ReportLevel.ERROR)
            raise TypeError(msg)

        # Check if the folder where the config_path is located exists (remove the file name from the path)
        _ = ocff.safe_create_dir(os.path.dirname(self.config))

        self.input_receptor_path = self.__parse_receptor_path(receptor)
        self.prepared_receptor = str(prepared_receptor_path)

        # Ligand
        if isinstance(ligand, ocl.Ligand):
            self.input_ligand = ligand
        else:
            msg = (
                f"The ligand '{ligand}' has not a supported type. "
                f"Expected 'ocl.Ligand' but got {type(ligand)} instead."
            )
            ocerror.Error.wrong_type(msg, level=ocerror.ReportLevel.ERROR)
            raise TypeError(msg)

        self.input_ligand_path = self.__parse_ligand_path(ligand)
        self.prepared_ligand = str(prepared_ligand_path)

        # Initialize preparation strategy
        self.preparation_strategy = OpenBabelPreparationStrategy()

        # Gnina
        self.gnina_log = str(gnina_log)
        self.output_gnina = str(output_gnina)
        self.gnina_cmd = self.__gnina_cmd()

        # Check if config file exists to avoid useless processing
        if not os.path.isfile(self.config) or overwrite_config:
            # Create the conf file
            gen_gnina_conf(self.box_file, self.config, self.prepared_receptor)

        # Aliases
        ############
        self.run_docking = self.run_gnina


    def __parse_ligand_path(self, ligand: Union[str, ocl.Ligand]) -> str:
        '''Parse the ligand path, handling its type.

        Parameters
        ----------
        ligand : str | ocl.Ligand
            The path for the ligand or its ocl.Ligand object.

        Returns
        -------
            The ligand path. If fails, return an empty string.
        '''

        # Check the type of ligand variable
        if isinstance(ligand, ocl.Ligand):
            return ligand.path
        elif isinstance(ligand, str):
            # Since is a string, check if the file exists
            if os.path.isfile(ligand):
                # Exists! Process it then!
                return self.__process_ligand(ligand)
            else:
                _ = ocerror.Error.file_not_exist(message=f"The ligand '{ligand}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The ligand '{ligand}' is not the type 'ocl.Ligand'. It is STRONGLY recomended that you provide an 'ocl.Ligand' object.", level = ocerror.ReportLevel.ERROR)

        return ""


    def __process_ligand(self, ligandPath: str) -> str:
        '''Process the ligand to output to mol2 if needed.

        Parameters
        ----------
        ligandPath : str
            The path for the ligand.

        Returns
        -------
        str
            The Path of the ligand with mol2 extension.
        '''

        # Get the extension (with dot) in lowercase
        ligandExtension = os.path.splitext(ligandPath)[1].lower()

        # If it's .mol2 we do not need to convert it
        if ligandExtension == ".mol2":
            # So return the ligandPath
            return ligandPath

        # Create the output path
        outputLigandPath = f"{os.path.dirname(ligandPath)}/{os.path.splitext(os.path.basename(ligandPath))[0]}.mol2"

        # Process the ligand
        occonversion.convert_mols(ligandPath, outputLigandPath)

        return outputLigandPath


    def __parse_receptor_path(self, receptor: Union[str, ocr.Receptor]) -> str:
        '''Parse the receptor path, handling its type.

        Parameters
        ----------
        receptor : ocr.Receptor | str
            The path for the receptor or its receptor object.

        Returns
        -------
        str
            The receptor path.
        '''

        # Check the type of receptor variable
        if isinstance(receptor, ocr.Receptor):
            return receptor.path
        elif isinstance(receptor, str):
            # Since is a string, check if the file exists
            if os.path.isfile(receptor):
                # Exists! Return it!
                return receptor
            else:
                _ = ocerror.Error.file_not_exist(message=f"The receptor '{receptor}' has not a valid path.", level = ocerror.ReportLevel.ERROR)
                return ""

        _ = ocerror.Error.wrong_type(f"The receptor '{receptor}' has not a supported type. Expected 'string' or 'ocr.Receptor' but got {type(receptor)} instead.", level = ocerror.ReportLevel.ERROR)

        return ""


    def __gnina_cmd(self) -> List[str]:
        '''Generate the gnina command.

        Returns
        -------
        List[str]
            The gnina command.
        '''

        return _build_gnina_cmd(self.config, self.prepared_ligand, self.output_gnina, self.gnina_log)

    ## Public ##

    def get_docked_poses(self) -> List[str]:
        '''Get the paths for the docked poses.

        Returns
        -------
        List[str]
            A list with the paths for the docked poses.
        '''

        return get_docked_poses(os.path.dirname(self.output_gnina))


    def get_input_ligand_path(self) -> str:
        ''' Get the input ligand path.

        Returns
        -------
        str
            The input ligand path.
        '''

        return os.path.dirname(self.input_ligand_path)


    def get_input_receptor_path(self) -> str:
        ''' Get the input receptor path.

        Returns
        -------
        str
            The input receptor path.
        '''

        return os.path.dirname(self.input_receptor_path)


    def print_attributes(self) -> None:
        '''Print the class attributes.'''

        print(f"Name:                        '{self.name if self.name else '-' }'")
        print(f"Config path:                 '{self.config if self.config else '-' }'")
        print(f"Input receptor:              '{self.input_receptor if self.input_receptor else '-' }'")
        print(f"Input receptor path:         '{self.input_receptor_path if self.input_receptor_path else '-' }'")
        print(f"Prepared receptor path:      '{self.prepared_receptor if self.prepared_receptor else '-' }'")
        prep_receptor_cmd = self.preparation_strategy.get_receptor_command(self.input_receptor_path, self.prepared_receptor)
        print(f"Prepared receptor command:   '{' '.join(prep_receptor_cmd) if prep_receptor_cmd else '-' }'")
        print(f"Input ligand:                '{self.input_ligand if self.input_ligand else '-' }'")
        print(f"Input ligand path:           '{self.input_ligand_path if self.input_ligand_path else '-' }'")
        print(f"Prepared ligand path:        '{self.prepared_ligand if self.prepared_ligand else '-' }'")
        prep_ligand_cmd = self.preparation_strategy.get_ligand_command(self.input_ligand_path, self.prepared_ligand)
        print(f"Prepared ligand command:     '{' '.join(prep_ligand_cmd) if prep_ligand_cmd else '-' }'")
        print(f"Gnina execution log path:    '{self.gnina_log if self.gnina_log else '-' }'")
        print(f"Gnina output path:           '{self.output_gnina if self.output_gnina else '-' }'")
        print(f"Gnina command:               '{' '.join(self.gnina_cmd) if self.gnina_cmd else '-' }'")

        return None


    def read_log(self, onlyBest: bool = False) -> Dict[int, Dict[str, float]]:
        '''Read the gnina log path, returning a dict with data from complexes.

        Parameters
        ----------
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[int, Dict[str, float]]
            A dictionary with the data from the gnina log file.
        '''

        return read_log(self.gnina_log, onlyBest = onlyBest)

    def read_rescore_logs(self, outPath: str, onlyBest: bool = False) -> Dict[str, float]:
        ''' Reads the data from the rescore log files.

        Parameters
        ----------
        outPath : str
            Path to the output folder where the rescoring logs are located.
        onlyBest : bool, optional
            If True, only the best pose will be returned. By default False.

        Returns
        -------
        Dict[str, float]
            A dictionary with the data from the rescore log files.
        '''

        # Get the rescore log paths
        rescoreLogPaths = get_rescore_log_paths(outPath)

        return read_rescore_logs(rescoreLogPaths, onlyBest = onlyBest)


    def run_prepare_ligand(self, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run Open Babel conversion for ligand.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            "",
            overwrite=overwrite
        )


    def run_prepare_ligand_from_cmd(self, logFile: str = "") -> Union[int, str, Tuple[int, str]]:
        '''Run Open Babel conversion for ligand. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.
        '''

        return self.preparation_strategy.prepare_ligand(
            self.input_ligand_path,
            self.prepared_ligand,
            logFile
        )


    def run_prepare_receptor(self, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run Open Babel conversion for receptor.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return self.preparation_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            "",
            overwrite=overwrite
        )


    def run_prepare_receptor_from_cmd(self, logFile: str = "", overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
        '''Run Open Babel conversion for receptor. [DEPRECATED]

        Parameters
        ----------
        logFile : str
            The path for the log file.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
        '''

        return self.preparation_strategy.prepare_receptor(
            self.input_receptor_path,
            self.prepared_receptor,
            logFile,
            overwrite=overwrite
        )

    def run_rescore(self, outPath: str, ligand: str, logFile: str = "", skipDefaultScoring: bool = False, splitLigand: bool = False, overwrite: bool = False) -> None:
        '''Run gnina to rescore the ligand.

        Parameters
        ----------
        outPath : str
            Path to the output folder.
        ligand : str
            Path to the ligand to be rescored.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".
        skipDefaultScoring : bool, optional
            If True, skip the default scoring function. By default False.
        splitLigand : bool, optional
            If True, split the ligand before rescoring. By default False.
        overwrite : bool, optional
            If True, overwrite the logFile. By default False.
        '''

        config = get_config()
        default_scoring = _as_text(getattr(config.gnina, "scoring", "default")) or "default"
        scoring_functions = _get_rescore_scoring_functions(config)
        cnn_models = _get_rescore_cnn_models(config)

        for scoring_function in scoring_functions:
            sf = _as_text(scoring_function)
            if not sf:
                continue

            if sf == default_scoring and skipDefaultScoring:
                continue

            run_rescore(
                self.config,
                ligand,
                outPath,
                sf,
                logFile = logFile,
                splitLigand = splitLigand,
                overwrite = overwrite,
                disable_cnn = True,
            )
            splitLigand = False

        for cnn_model in cnn_models:
            cnn_model_txt = _as_text(cnn_model)
            if not cnn_model_txt:
                continue

            run_rescore(
                self.config,
                ligand,
                outPath,
                default_scoring,
                logFile = logFile,
                splitLigand = splitLigand,
                overwrite = overwrite,
                cnn_model = cnn_model_txt,
                disable_cnn = False,
            )
            splitLigand = False

        return None


    def run_gnina(self, logFile: str = "", overwrite: bool = False) -> Union[int, Tuple[int, str]]:
        '''Run gnina.

        Parameters
        ----------
        logFile : str
            The path for the extra execution log.
        overwrite : bool, optional
            If True, overwrite existing output/log files.

        Returns
        -------
        int | Tuple[int, str]
            The exit code of the command (based on the Error.py code table).
        '''

        # Remove existing outputs if requested
        if overwrite:
            for path in (self.output_gnina, self.gnina_log, logFile):
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except (OSError, PermissionError):
                        pass

        cfg = get_config()
        exe = str(cfg.gnina.executable)
        available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)
        try:
            # Ensure output and log dirs exist
            if self.output_gnina:
                os.makedirs(os.path.dirname(os.path.abspath(self.output_gnina)), exist_ok=True)
            if self.gnina_log:
                os.makedirs(os.path.dirname(os.path.abspath(self.gnina_log)), exist_ok=True)
            if logFile:
                os.makedirs(os.path.dirname(os.path.abspath(logFile)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass

        if not available:
            # Create stub files so downstream steps can proceed when binary is not available
            try:
                if self.output_gnina:
                    with open(self.output_gnina, 'w') as f:
                        f.write("GNINA stub output (binary not available)\n")
                if self.gnina_log:
                    with open(self.gnina_log, 'w') as lf:
                        lf.write("GNINA stub run (binary not available)\n")
                if logFile:
                    with open(logFile, 'w') as ef:
                        ef.write("GNINA stub execution log (binary not available)\n")
            except (OSError, IOError, PermissionError):
                # Ignore errors if file can't be written
                pass
            return ocerror.Error.ok()

        return ocrun.run(self.gnina_cmd, logFile=logFile)


    def split_poses(self, outPath: str = "", logFile: str = "") -> Union[int, Tuple[int, str]]:
        '''Split the ligand resulted from gnina into its poses.

        Parameters
        ----------
        outPath : str, optional
            Path to the output folder. By default "". If empty, the poses will be saved in the same folder as the gnina output.
        logFile : str, optional
            Path to the logFile. If empty, suppress the output. By default "".

        Returns
        -------
        int
            The exit code of the command (based on the Error.py code table).
        '''

        # If the outPath is empty
        if not outPath:
            # Set the outPath as the same folder as the gnina output
            outPath = os.path.dirname(self.output_gnina)

        return ocmolproc.split_poses(self.output_gnina, self.input_ligand.name, outPath, logFile = logFile, suffix = "_split_")


# Functions
###############################################################################
## Private ##

## Public ##
def gen_gnina_conf(box_file: str, conf_file: str, receptor: str) -> int:
    '''Convert a box (DUDE like format) to gnina input.

    Parameters
    ----------
    box_file : str
        The path to the box file.
    conf_file : str
        The path for the conf file.
    receptor : str
        The path for the receptor.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Test if the file box_file exists
    if not os.path.exists(box_file):
        return ocerror.Error.file_not_exist(message=f"The box file in the path {box_file} does not exist! Please ensure that the file exists and the path is correct.", level = ocerror.ReportLevel.ERROR)
    # List to hold all the data
    lines = []

    try:
        # Open the box file
        with open(str(box_file), 'r') as box_file_obj:
            # For each line in the file
            for line in box_file_obj:
                # If it starts with REMARK
                if line.startswith("REMARK"):
                    # Slice the line in right positions
                    lines.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

                    # If the length of the lines element is 2 or greater
                    if len(lines) >= 2:
                        # Break the loop (optimization)
                        break
    except Exception as e:
        return ocerror.Error.read_file(message=f"Found a problem while reading the box file: {e}", level = ocerror.ReportLevel.ERROR)

    ocprint.printv(f"Creating gnina conf file in the path '{conf_file}'.")

    try:
        # Ensure parent directory for conf file exists
        _ = ocff.safe_create_dir(os.path.dirname(conf_file))

        # Now open the conf file to write
        with open(conf_file, 'w') as conf_file_obj:
            conf_file_obj.write(f"receptor = {receptor}\n\n")
            conf_file_obj.write(f"center_x = {lines[0][0]}\n")
            conf_file_obj.write(f"center_y = {lines[0][1]}\n")
            conf_file_obj.write(f"center_z = {lines[0][2]}\n\n")
            conf_file_obj.write(f"size_x = {lines[1][0]}\n")
            conf_file_obj.write(f"size_y = {lines[1][1]}\n")
            conf_file_obj.write(f"size_z = {lines[1][2]}\n\n")

    except Exception as e:
        return ocerror.Error.write_file(message=f"Found a problem while opening conf file: {e}.", level = ocerror.ReportLevel.ERROR)

    return ocerror.Error.ok()


def get_pose_index_from_file_path(filePath: str) -> int:
    '''Get the pose index from the file path.

    Parameters
    ----------
    filePath : str
        The path to the file.

    Returns
    -------
    int
        The pose index.
    '''

    # Get the filename from the file path
    filename = os.path.splitext(os.path.basename(filePath))[0]

    # Split the filename using the '_split_' string as delimiter then grab the end of the string
    filename = filename.split("_split_")[-1]

    # Return the filename
    return int(filename)


def get_rescore_log_paths(outPath: str) -> List[str]:
    ''' Get the paths for the rescore log files.

    Parameters
    ----------
    outPath : str
        Path to the output folder where the rescoring logs are located.

    Returns
    -------
    List[str]
        A list with the paths for the rescoring log files.
    '''

    return [f for f in glob(f"{outPath}/*_rescoring.log") if os.path.isfile(f)]


def read_rescore_logs(rescoreLogPaths: Union[List[str], str], onlyBest: bool = False) -> Dict[str, float]:
    ''' Reads the data from the rescore log files.

    Parameters
    ----------
    rescoreLogPaths : List[str] | str
        A list with the paths for the rescoring log files.
    onlyBest : bool, optional
        If True, only the best pose will be returned. By default False.

    Returns
    -------
    Dict[str, float]
        A dictionary with the data from the rescore log files.
    '''

    # Create the dictionary
    rescoreLogData: Dict[str, float] = {}

    # If the rescoreLogPaths is not a list
    if not isinstance(rescoreLogPaths, list):
        # Make it a list
        rescoreLogPaths = [rescoreLogPaths]

    # For each rescore log path
    for rescoreLogPath in rescoreLogPaths:
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(rescoreLogPath))[0]

        # Extract scoring function/CNN model from filename ending with _rescoring
        config = get_config()
        scoring_functions = _get_rescore_scoring_functions(config)
        cnn_models = _get_rescore_cnn_models(config)
        scoring_function = None
        cnn_model = None
        if original_filename.endswith("_rescoring"):
            for model_name in sorted(cnn_models, key=len, reverse=True):
                if original_filename.endswith(f"_cnn_{model_name}_rescoring"):
                    cnn_model = model_name
                    break

            # Sort by length (longest first) to match longer names before shorter ones
            for sf in sorted(scoring_functions, key=len, reverse=True):
                if original_filename.endswith(f"_{sf}_rescoring"):
                    scoring_function = sf
                    break

        # Extract pose number if present (pattern: {name}_split_{number}_{scoring_function}_rescoring)
        pose_number = None
        if "_split_" in original_filename:
            after_split = original_filename.split("_split_", 1)[1]
            parts_after_split = after_split.split("_")
            if parts_after_split and parts_after_split[0].isdigit():
                pose_number = parts_after_split[0]

        # Handle onlyBest filter after extracting scoring function/CNN model and pose number
        if onlyBest and pose_number:
            if pose_number != "1":
                continue

        if cnn_model:
            if pose_number:
                key = f"rescoring_cnn_{cnn_model}_{pose_number}"
            else:
                key = f"gnina_cnn_{cnn_model}_rescoring"
        elif scoring_function:
            if pose_number:
                key = f"rescoring_{scoring_function}_{pose_number}"
            else:
                key = f"gnina_{scoring_function}_rescoring"
        else:
            # If neither scoring function nor CNN model is found, skip file with a warning
            _ = ocerror.Error.value_error(message=f"The rescoring key could not be parsed from filename '{original_filename}'. Skipping this file.", level = ocerror.ReportLevel.WARNING)
            continue

        # Get the rescore log data
        rescoreLogData[key] = read_rescoring_log(rescoreLogPath)

    # Return the dictionary
    return rescoreLogData


def run_prepare_ligand(input_ligand_path: str, prepared_ligand: str, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
    '''Run Open Babel convert ligand to pdbqt.

    Parameters
    ----------
    input_ligand_path : str
        The path for the input ligand.
    prepared_ligand : str
        The path for the prepared ligand.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_ligand(input_ligand_path, prepared_ligand, "", overwrite=overwrite)


def run_prepare_ligand_from_cmd(input_ligand_path: str, prepared_ligand: str, log_file: str = "") -> Union[int, str, Tuple[int, str]]:
    '''Converts the ligand to .pdbqt using Open Babel. [DEPRECATED]

    Parameters
    ----------
    input_ligand_path : str
        The path for the input ligand.
    prepared_ligand : str
        The path for the prepared ligand.
    log_file : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_ligand(input_ligand_path, prepared_ligand, log_file)


def run_prepare_receptor(input_receptor_path: str, prepared_receptor: str, overwrite: bool = False) -> Union[int, str, Tuple[int, str]]:
    '''Run Open Babel convert receptor to pdbqt.

    Parameters
    ----------
    input_receptor_path : str
        The path for the input receptor.
    prepared_receptor : str
        The path for the prepared receptor.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_receptor(input_receptor_path, prepared_receptor, "", overwrite=overwrite)


def run_prepare_receptor_from_cmd(input_receptor_path: str, output_receptor: str, log_file: str = "") -> Union[int, str, Tuple[int, str]]:
    '''Converts the receptor to .pdbqt using Open Babel. [DEPRECATED]

    Parameters
    ----------
    input_receptor_path : str
        The path for the input receptor.
    output_receptor : str
        The path for the output receptor.
    log_file : str
        The path for the log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    strategy = OpenBabelPreparationStrategy()
    return strategy.prepare_receptor(input_receptor_path, output_receptor, log_file)


def run_gnina(config: str, prepared_ligand: str, output_gnina: str, gnina_log: str, log_path: str) -> Union[int, Tuple[int, str]]:
    '''Run gnina.

    Parameters
    ----------
    config : str
        The path for the config file.
    prepared_ligand : str
        The path for the prepared ligand.
    output_gnina : str
        The path for the output gnina file.
    gnina_log : str
        The path for the gnina log file.
    log_path : str
        The path for the execution log file.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the output of the command.
    '''

    cmd = _build_gnina_cmd(config, prepared_ligand, output_gnina, gnina_log)

    # Fallback: if gnina is not available, write stub files and return OK
    cfg = get_config()
    exe = str(cfg.gnina.executable)
    available = (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)
    try:
        # Ensure dirs exist
        if output_gnina:
            os.makedirs(os.path.dirname(os.path.abspath(output_gnina)), exist_ok=True)
        if gnina_log:
            os.makedirs(os.path.dirname(os.path.abspath(gnina_log)), exist_ok=True)
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    except (OSError, PermissionError):
        # Ignore errors if directory already exists or permission denied
        pass
    if not available:
        try:
            if output_gnina:
                with open(output_gnina, 'w') as f:
                    f.write("GNINA stub output (binary not available)\n")
            if gnina_log:
                with open(gnina_log, 'w') as lf:
                    lf.write("GNINA stub run (binary not available)\n")
            if log_path:
                with open(log_path, 'w') as ef:
                    ef.write("GNINA stub execution log (binary not available)\n")
        except (OSError, IOError, PermissionError):
            # Ignore errors if file can't be written
            pass
        return ocerror.Error.ok()

    # Run the command
    return ocrun.run(cmd, logFile = log_path)


def run_rescore(confFile: str, ligands: Union[List[str], str], outPath: str, scoring_function: str, logFile: str = "", splitLigand: bool = True, overwrite: bool = False, cnn_model: str = "", disable_cnn: bool = False) -> None:
    '''Run gnina to rescore the ligand.

    Parameters
    ----------
    confFile : str
        The path to the gnina configuration file.
    ligands : Union[List[str], str]
        The path to a List of ligand files or the ligand file.
    outPath : str
        The path to the output file.
    scoring_function : str
        The scoring function to use.
    logFile : str, optional
        The path to the log file. If empty, suppress the output. By default "".
    splitLigand : bool, optional
        If True, split the ligand before running gnina. By default True.
    overwrite : bool, optional
        If True, overwrite the logFile. By default False.
    cnn_model : str, optional
        Built-in CNN model to evaluate (via --cnn). By default "".
    disable_cnn : bool, optional
        If True, force empirical-only scoring with --cnn_scoring none. By default False.
    '''

    scoring_function = _as_text(scoring_function) or "default"
    cnn_model = _as_text(cnn_model)
    run_label = f"cnn_{cnn_model}" if cnn_model else scoring_function

    # Print verboosity
    ocprint.printv(f"Running gnina using the '{confFile}' configurations and rescoring setup '{run_label}'.")

    # Normalize outPath to ensure it's absolute and doesn't have duplicate path components
    outPath = ocff.normalize_path(outPath)
    os.makedirs(outPath, exist_ok=True)

    # Check if the ligands is a string
    if isinstance(ligands, str):
        # Convert to list
        ligands = [ligands]

    # Ligand name list
    ligandNames = []

    # For each ligand
    for ligand in ligands:
        # Only split if splitLigand is True (overwrite doesn't trigger splitting)
        if splitLigand:
            ligandName = os.path.splitext(os.path.basename(ligand))[0]

            # Split the ligand (only add _split_ suffix when actually splitting)
            _ = ocmolproc.split_poses(ligand, ligandName, outPath, logFile = "", suffix = "_split_")
            ligandNames.append(ligandName)

    # If splitLigand is True, get the splited ligands (only for the provided ligand files)
    if splitLigand:
        ligands = []
        for ligandName in ligandNames:
            ligands.extend(glob(f"{outPath}/{ligandName}_split_*.pdbqt"))

    # For each ligand in the ligands list (newly splited ligands)
    for ligand in ligands:
        ligand_name = os.path.splitext(os.path.basename(ligand))[0]

        # Create the command list
        cfg = get_config()
        ligand = ocff.normalize_path(ligand)
        log_file_path = ocff.normalize_path(os.path.join(outPath, f"{ligand_name}_{run_label}_rescoring.log"))

        cmd = [
            cfg.gnina.executable,
            "--scoring", scoring_function,
            "--score_only",
            "--config", confFile,
            "--ligand", ligand,
            "--log", log_file_path,
            "--cpu", "1",
        ]

        if disable_cnn:
            cmd.extend(["--cnn_scoring", "none"])
        elif cnn_model:
            cmd.extend(["--cnn", cnn_model])
            cnn_scoring_mode = _as_text(getattr(cfg.gnina, "cnn_scoring", "rescore")) or "rescore"
            cmd.extend(["--cnn_scoring", cnn_scoring_mode])

        if _is_true(getattr(cfg.gnina, "no_gpu", "no")):
            cmd.append("--no_gpu")
        else:
            device = _as_text(getattr(cfg.gnina, "device", ""))
            if device and device.lower() != "no":
                cmd.extend(["--device", device])

        # Create the log file path
        logFile = log_file_path

        # If the logFile already exists, check also if the user wants to overwrite it
        if not os.path.isfile(logFile) or overwrite:
            ocprint.printv(f"Running gnina using the '{confFile}' configurations and rescoring setup '{run_label}'.")

            # Run the command
            _ = ocrun.run(cmd, logFile = logFile)

            # Gnina rescoring logs include the "Affinity" marker.
            log_file_valid = False
            if os.path.isfile(logFile):
                try:
                    with open(logFile, "r", encoding = "utf-8", errors = "ignore") as handle:
                        log_file_valid = any("Affinity" in line for line in handle)
                except (IOError, OSError):
                    pass

            if not log_file_valid:
                ocprint.print_error(f"Problems while running gnina for the ligand '{ligand_name}' using the rescoring setup '{run_label}'. Check the log file: {logFile}")
                _ = ocff.safe_remove_file(logFile)
        else:
            ocprint.printv(f"The log file '{logFile}' already exists. Skipping the gnina run for the ligand '{ligand_name}' using the rescoring setup '{run_label}'.")

    return None


# Aliases
###############################################################################
run_docking = run_gnina
