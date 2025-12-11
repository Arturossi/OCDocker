#!/usr/bin/env python3

'''
OCDocker CLI
============

Unified command-line interface for OCDocker tasks.

Main commands
- version: prints library version.
- init-config: creates a quick `OCDocker.cfg` from the example file.
- vs: runs docking and optional rescoring for one receptor/ligand/box using Vina, Smina, or PLANTS.
- shap: delegates to existing OCScore SHAP CLI.
- pipeline: full multi-engine flow — run docking across engines, cluster poses by RMSD,
            pick the representative pose (medoid of the largest cluster), rescore and export results.

Global options
- --conf, --multiprocess, --update-databases, --output-level, --overwrite:
  compatible with OCDocker.Initialise and used to bootstrap the environment.
'''

from __future__ import annotations

__all__ = ['main']

import argparse
import importlib
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def _preparse_global_args(argv: list[str]) -> argparse.Namespace:
    '''Extract global flags from anywhere in argv.

    Works around argparse limitation when global options appear after the subcommand.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed global arguments.
    '''
    
    ns = argparse.Namespace(
        version=False,
        multiprocess=True,
        update=False,
        config_file=None,
        output_level=1,
        overwrite=False,
        log_file=None,
        no_stdout_log=False,
    )

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--version":
            ns.version = True
            i += 1
            continue
        if tok == "--multiprocess":
            ns.multiprocess = True
            i += 1
            continue
        if tok in ("-u", "--update-databases"):
            ns.update = True
            i += 1
            continue
        if tok == "--conf" and i + 1 < len(argv):
            ns.config_file = argv[i + 1]
            i += 2
            continue
        if tok == "--output-level" and i + 1 < len(argv):
            try:
                ns.output_level = int(argv[i + 1])
            except (ValueError, TypeError):
                # Ignore invalid output level values
                pass
            i += 2
            continue
        if tok == "--overwrite":
            ns.overwrite = True
            i += 1
            continue
        if tok == "--log-file" and i + 1 < len(argv):
            ns.log_file = argv[i + 1]
            i += 2
            continue
        if tok == "--no-stdout-log":
            ns.no_stdout_log = True
            i += 1
            continue
        # skip token
        i += 1
    return ns

def _bootstrap_ocdocker_env(ns: argparse.Namespace) -> None:
    '''Bootstrap OCDocker.Initialise explicitly (no import-time side effects).

    - Set `OCDOCKER_CONFIG` env var if provided
    - Call `OCDocker.Initialise.bootstrap(ns)`

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed command-line arguments.
    '''
    
    if ns.config_file:
        os.environ["OCDOCKER_CONFIG"] = ns.config_file
    init_mod = importlib.import_module("OCDocker.Initialise")
    if hasattr(init_mod, "bootstrap"):
        init_mod.bootstrap(ns)  # type: ignore
    else:
        raise RuntimeError("OCDocker.Initialise.bootstrap not found")


def _require_file(p: str, label: str) -> Path:
    '''Ensure a file path exists. Print a helpful message and exit if not.

    Also warns if the path seems to contain a Unicode ellipsis (…)
    which is often a placeholder, not a real path.

    Parameters
    ----------
    p : str
        The file path to check.
    label : str
        A label for the file path (used in error messages).

    Returns
    -------
    Path
        The resolved file path.

    Raises
    ------
    SystemExit
        If the file path is invalid or not found.
    '''
    
    if "…" in p:
        print(f"Error: {label} contains an ellipsis character (…). Replace it with a real path.")
        raise SystemExit(2)
    path = Path(p).resolve()
    if not path.is_file():
        print(f"Error: {label} file not found: {p}")
        raise SystemExit(2)
    return path

def build_parser() -> argparse.ArgumentParser:
    '''Build the main argument parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        The constructed argument parser.
    '''

    parser = argparse.ArgumentParser(
        prog="ocdocker",
        description=(
            "OCDocker CLI: Unified command-line interface for molecular docking, virtual screening, and analysis.\n\n"
            "Main commands:\n"
            "  vs        - Single-engine docking with rescoring of all poses\n"
            "  pipeline  - Multi-engine consensus docking with clustering and representative pose selection\n"
            "  shap      - SHAP analysis for model interpretability\n"
            "  console   - Interactive Python console with OCDocker pre-loaded\n"
            "  script    - Run a Python script with OCDocker libraries pre-loaded\n"
            "  doctor    - Environment diagnostics and setup verification\n"
            "  init-config - Create starter configuration file\n"
            "  version   - Print version information\n\n"
            "Use 'ocdocker <command> --help' for detailed information about each command."
        ),
        epilog=(
            "Note: SQLite backend is intended for development/tests. "
            "For production workloads (performance/concurrency), a full MySQL installation is strongly recommended."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options (mirrors OCDocker.Initialise)
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=True,
        help="Enable multiprocessing for supported tasks. Allows parallel execution when possible. Default: enabled"
    )
    parser.add_argument(
        "-u",
        "--update-databases",
        dest="update",
        action="store_true",
        default=False,
        help="Update databases on startup. Runs database schema updates and migrations if needed."
    )
    parser.add_argument(
        "--conf",
        dest="config_file",
        type=str,
        help="Path to OCDocker.cfg configuration file. If not specified, uses default locations or OCDOCKER_CONFIG environment variable."
    )
    parser.add_argument(
        "--output-level",
        dest="output_level",
        type=int,
        default=1,
        help="Logging verbosity level (0-5). Higher numbers provide more detailed output. 0=silent, 1=normal, 2-5=increasing verbosity. Default: 1"
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting existing output files. By default, existing files are preserved to prevent accidental data loss."
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=str,
        default=None,
        help="Write log messages to this file in addition to stdout. Useful for saving detailed logs for later analysis."
    )
    parser.add_argument(
        "--no-stdout-log",
        dest="no_stdout_log",
        action="store_true",
        default=False,
        help="Disable logging to stdout. Only log to file if --log-file is specified. Useful for cleaner console output."
    )

    # Parent parser to allow repeating global options after subcommand
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--multiprocess", action="store_true", default=True, help="Enable multiprocessing for supported tasks")
    parent.add_argument("-u", "--update-databases", dest="update", action="store_true", default=False, help="Update databases on startup")
    parent.add_argument("--conf", dest="config_file", type=str, help="Path to OCDocker.cfg configuration file")
    parent.add_argument("--output-level", dest="output_level", type=int, default=1, help="Logging verbosity level (0-5)")
    parent.add_argument("--overwrite", dest="overwrite", action="store_true", default=False, help="Allow overwriting existing output files")
    parent.add_argument("--log-file", dest="log_file", type=str, default=None, help="Write log messages to this file")
    parent.add_argument("--no-stdout-log", dest="no_stdout_log", action="store_true", default=False, help="Disable logging to stdout")

    sub = parser.add_subparsers(dest="command", required=True)

    # init-config
    p_init = sub.add_parser(
        "init-config",
        description=(
            "Create a starter OCDocker.cfg configuration file from the example template.\n"
            "This command copies OCDocker.cfg.example to OCDocker.cfg in the current directory,\n"
            "allowing you to customize paths to docking binaries, databases, and other settings."
        ),
        help="Create a starter OCDocker.cfg configuration file",
        parents=[parent]
    )
    p_init.set_defaults(func=cmd_init_config)

    # version
    p_ver = sub.add_parser(
        "version",
        description="Print the installed version of OCDocker without bootstrapping the full environment.",
        help="Print OCDocker version",
        parents=[parent]
    )
    p_ver.set_defaults(func=cmd_version)

    # vs (single-entry virtual screening)
    p_vs = sub.add_parser(
        "vs",
        description=(
            "Run docking with a single engine (Vina, Smina, or PLANTS) and optionally rescore all poses.\n\n"
            "This command performs:\n"
            "  1. Receptor and ligand preparation\n"
            "  2. Docking with the selected engine\n"
            "  3. Pose splitting (for Vina/Smina) into individual files\n"
            "  4. Rescoring of all generated poses (unless --skip-rescore is used)\n\n"
            "Use this mode for quick single-engine docking runs where you want all poses rescored.\n"
            "For multi-engine consensus docking with clustering, use the 'pipeline' command instead."
        ),
        help="Run docking with one engine and rescore all poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_vs.add_argument(
        "--engine",
        choices=["vina", "smina", "plants"],
        default="vina",
        help="Docking engine to use. Options: 'vina' (AutoDock Vina), 'smina' (Vina with additional scoring functions), or 'plants' (PLANTS docking). Default: vina"
    )
    p_vs.add_argument(
        "--receptor",
        required=True,
        help="Path to the receptor structure file (e.g., PDB format). The receptor will be prepared automatically if needed."
    )
    p_vs.add_argument(
        "--ligand",
        required=True,
        help="Path to the ligand file. Supported formats: SMILES (.smi), SDF (.sdf), MOL2 (.mol2), or PDBQT (.pdbqt). The ligand will be prepared automatically if needed."
    )
    p_vs.add_argument(
        "--box",
        required=True,
        help="Path to the binding site box definition file (PDB format with REMARK records containing center coordinates and size). This defines the search space for docking."
    )
    p_vs.add_argument(
        "--name",
        help="Job name identifier. If not provided, defaults to the ligand filename (without extension). Used for output file naming."
    )
    p_vs.add_argument(
        "--outdir",
        default="./ocdocker_out",
        help="Output directory where all results will be saved. Default: ./ocdocker_out"
    )
    p_vs.add_argument(
        "--skip-rescore",
        action="store_true",
        help="Skip the rescoring phase. Only perform docking without applying additional scoring functions. Useful for faster runs when rescoring is not needed."
    )
    p_vs.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip pose splitting step (only applicable for Vina/Smina). By default, poses are split into individual files. Use this to keep all poses in a single file."
    )
    p_vs.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for external docking tools. Overrides the OCDOCKER_TIMEOUT environment variable. If a tool exceeds this time, the process will be terminated."
    )
    p_vs.add_argument(
        "--store-db",
        action="store_true",
        help="Store minimal metadata about this docking run in the database (Complexes table). Requires database to be configured and accessible."
    )
    p_vs.set_defaults(func=cmd_vs)

    # shap passthrough (reuses existing module)
    p_shap = sub.add_parser(
        "shap",
        description=(
            "Run SHAP (SHapley Additive exPlanations) analysis for OCScore models.\n\n"
            "SHAP analysis provides interpretability for machine learning models by explaining\n"
            "the contribution of each feature to the model's predictions. This command delegates\n"
            "to the OCScore SHAP analysis module for detailed feature importance analysis.\n\n"
            "This is an advanced analysis tool for understanding model behavior and feature contributions."
        ),
        help="Run SHAP analysis for OCScore model interpretability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_shap.add_argument(
        "--storage",
        required=True,
        help="Path to the storage directory containing study data and model files."
    )
    p_shap.add_argument(
        "--ao_study",
        required=True,
        help="Name or identifier of the atom order (AO) study to analyze."
    )
    p_shap.add_argument(
        "--nn_study",
        required=True,
        help="Name or identifier of the neural network (NN) study to analyze."
    )
    p_shap.add_argument(
        "--seed_study",
        required=True,
        help="Name or identifier of the seed study to analyze."
    )
    p_shap.add_argument(
        "--mask_study",
        required=True,
        help="Name or identifier of the mask study to analyze."
    )
    p_shap.add_argument(
        "--df_path",
        required=True,
        help="Path to the dataframe file (CSV or similar) containing the data to analyze."
    )
    p_shap.add_argument(
        "--base_models",
        required=True,
        help="Path or identifier for the base models to use for SHAP analysis."
    )
    p_shap.add_argument(
        "--study_number",
        type=int,
        required=True,
        help="Study number identifier for organizing the analysis results."
    )
    p_shap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory where SHAP analysis results will be saved."
    )
    p_shap.add_argument(
        "--explainer",
        default="deep",
        choices=["deep", "kernel"],
        help="SHAP explainer type to use. 'deep' uses DeepExplainer for neural networks, 'kernel' uses KernelExplainer (more general but slower). Default: deep"
    )
    p_shap.add_argument(
        "--background_size",
        type=int,
        help="Size of the background dataset used for SHAP value computation. Larger values provide more stable estimates but take longer. If not specified, uses a default size."
    )
    p_shap.add_argument(
        "--eval_size",
        type=int,
        help="Number of samples to evaluate and compute SHAP values for. If not specified, evaluates all samples in the dataset."
    )
    p_shap.add_argument(
        "--stratify_by",
        nargs="*",
        help="Column names to stratify the dataset by when creating background/evaluation sets. Ensures balanced representation across groups."
    )
    p_shap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default: 0"
    )
    p_shap.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not generate CSV output files. Only save other output formats."
    )
    p_shap.set_defaults(func=cmd_shap)

    # pipeline (multi‑engine + clustering + rescoring)
    p_pipe = sub.add_parser(
        "pipeline",
        description=(
            "Run multi-engine docking with RMSD clustering and representative pose selection.\n\n"
            "This command performs a complete workflow:\n"
            "  1. Runs docking with multiple engines (Vina, Smina, PLANTS, or any combination)\n"
            "  2. Collects all poses from all engines\n"
            "  3. Converts poses to MOL2 format\n"
            "  4. Clusters poses by RMSD similarity\n"
            "  5. Selects the representative pose (medoid of the largest cluster)\n"
            "  6. Rescores only the representative pose (not all poses)\n"
            "  7. Saves representative.mol2 and summary.json with rescoring results\n\n"
            "Use this mode for consensus docking where you want to combine results from multiple\n"
            "engines and identify the most representative binding pose. This is more computationally\n"
            "intensive but provides better confidence in the final pose selection.\n\n"
            "Note: Only the representative pose is rescored, unlike 'vs' which rescores all poses."
        ),
        help="Multi-engine docking with clustering and representative pose selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_pipe.add_argument(
        "--receptor",
        required=True,
        help="Path to the receptor structure file (e.g., PDB format). The receptor will be prepared automatically if needed."
    )
    p_pipe.add_argument(
        "--ligand",
        required=True,
        help="Path to the ligand file. Supported formats: SMILES (.smi), SDF (.sdf), MOL2 (.mol2), or PDBQT (.pdbqt). The ligand will be prepared automatically if needed."
    )
    p_pipe.add_argument(
        "--box",
        required=True,
        help="Path to the binding site box definition file (PDB format with REMARK records containing center coordinates and size). This defines the search space for docking."
    )
    p_pipe.add_argument(
        "--engines",
        default="vina,smina,plants",
        help="Comma-separated list of docking engines to use. Options: 'vina', 'smina', 'plants', or any combination (e.g., 'vina,smina' or 'vina,plants'). Default: vina,smina,plants (all engines)"
    )
    p_pipe.add_argument(
        "--rescoring-engines",
        "--rescore-engines",  # Alias for convenience
        dest="rescoring_engines",
        default=None,
        help="Comma-separated list of engines to use for rescoring. Options: 'vina', 'smina', 'plants', 'oddt', or any combination. If not specified, uses the same engines as --engines. Can be different from docking engines (e.g., dock with 'vina,plants' but rescore with 'vina,smina,oddt')."
    )
    p_pipe.add_argument(
        "--name",
        help="Job name identifier. If not provided, defaults to the ligand filename (without extension). Used for output file naming."
    )
    p_pipe.add_argument(
        "--outdir",
        default="./ocdocker_out",
        help="Output directory where all results will be saved. Default: ./ocdocker_out"
    )
    p_pipe.add_argument(
        "--cluster-min",
        type=float,
        default=10.0,
        help="Minimum RMSD threshold (in Angstroms) for clustering. The clustering algorithm searches between --cluster-min and --cluster-max to find optimal clusters. Default: 10.0"
    )
    p_pipe.add_argument(
        "--cluster-max",
        type=float,
        default=20.0,
        help="Maximum RMSD threshold (in Angstroms) for clustering. Poses within this distance are considered similar. Default: 20.0"
    )
    p_pipe.add_argument(
        "--cluster-step",
        type=float,
        default=0.1,
        help="Step size (in Angstroms) for searching the optimal clustering threshold between --cluster-min and --cluster-max. Smaller values provide finer search but take longer. Default: 0.1"
    )
    p_pipe.add_argument(
        "--store-db",
        action="store_true",
        help="Store minimal metadata about this docking run in the database (Complexes table). Requires database to be configured and accessible."
    )
    p_pipe.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for external docking tools. Overrides the OCDOCKER_TIMEOUT environment variable. If a tool exceeds this time, the process will be terminated."
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    # console (interactive mode)
    p_console = sub.add_parser(
        "console",
        description=(
            "Launch an interactive Python console with OCDocker pre-loaded.\n\n"
            "This provides an interactive environment with tab-completion and command history,\n"
            "allowing you to use OCDocker programmatically. All OCDocker modules are imported\n"
            "and ready to use. Useful for exploratory work, debugging, or custom workflows\n"
            "that don't fit the standard CLI commands."
        ),
        help="Open interactive OCDocker Python console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_console.set_defaults(func=cmd_console)

    # script (run Python script with OCDocker pre-loaded)
    p_script = sub.add_parser(
        "script",
        description=(
            "Run a Python script with OCDocker libraries pre-loaded.\n\n"
            "This command bootstraps the OCDocker environment, loads all OCDocker modules,\n"
            "and executes your script file. All OCDocker classes and functions are available\n"
            "in the script's namespace, just like in the interactive console.\n\n"
            "Useful for running custom workflows, batch processing, or automation scripts\n"
            "that use OCDocker functionality."
        ),
        help="Run a Python script with OCDocker libraries pre-loaded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_script.add_argument(
        "script_file",
        help="Path to the Python script file to execute. The script will have access to all OCDocker modules."
    )
    p_script.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script (accessible via sys.argv in the script)."
    )
    p_script.set_defaults(func=cmd_script)

    # doctor (environment diagnostics)
    p_doc = sub.add_parser(
        "doctor",
        description=(
            "Run diagnostics to check your OCDocker environment setup.\n\n"
            "This command verifies:\n"
            "  - Availability and accessibility of docking engine binaries (Vina, Smina, PLANTS)\n"
            "  - Python dependencies and package versions\n"
            "  - Database connectivity and configuration\n"
            "  - Configuration file validity\n\n"
            "Use this command to troubleshoot installation or configuration issues before\n"
            "running docking jobs. It provides detailed information about what's working\n"
            "and what needs to be fixed."
        ),
        help="Check environment setup and diagnose issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent]
    )
    p_doc.set_defaults(func=cmd_doctor)

    return parser

def cmd_init_config(args: argparse.Namespace) -> int:
    '''Create a base OCDocker.cfg from the example file.

    This avoids importing Initialise (which expects a ready config).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    # Look for example file in current directory or parent directories
    example = Path("OCDocker.cfg.example")
    if not example.exists():
        # Try looking in the OCDocker package directory
        import OCDocker
        pkg_dir = Path(OCDocker.__file__).parent.parent
        example = pkg_dir / "OCDocker.cfg.example"
        if not example.exists():
            print("OCDocker.cfg.example not found in current directory or package directory.")
            return 1

    target = Path(args.config_file or "OCDocker.cfg")
    if target.exists():
        print(f"Config already exists: {target}")
        return 0

    target.write_text(example.read_text())
    print(f"Config created at: {target}. Please review and adjust paths.")
    return 0

def cmd_version(args: argparse.Namespace) -> int:
    '''Print package version without bootstrapping the full environment.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    try:
        import OCDocker as _oc
        v = getattr(_oc, "__version__", None)
        if v:
            print(v)
            return 0
    except (ImportError, AttributeError):
        # Module import failed or version attribute missing
        pass
    # Fallback to importlib.metadata (may work when installed)
    try:
        from importlib.metadata import version as _pkg_version
        print(_pkg_version("OCDocker"))
        return 0
    except (ImportError, AttributeError):
        # importlib.metadata not available or package not found
        pass
    # Last resort: try legacy variable if available (avoid heavy import)
    print("unknown")
    return 0

def cmd_vs(args: argparse.Namespace) -> int:  # pragma: no cover - heavy integration path, exercised by engine-specific tests
    '''Run a simple docking with the selected engine.

    Flow: prepare receptor/ligand, run docking, split poses (when applicable),
    and optionally run rescoring.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    # Bootstrap environment before importing engines
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror  # type: ignore
        import OCDocker.Toolbox.Logging as oclogging  # type: ignore
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    # Optionally set timeout for external processes
    if args.timeout:
        os.environ["OCDOCKER_TIMEOUT"] = str(args.timeout)

    # Imports after env is ready
    import OCDocker.Ligand as ocl  # type: ignore
    import OCDocker.Receptor as ocr  # type: ignore
    if args.engine == "vina":
        import OCDocker.Docking.Vina as engine_mod  # type: ignore
        eng = "vina"
    elif args.engine == "smina":
        import OCDocker.Docking.Smina as engine_mod  # type: ignore
        eng = "smina"
    else:
        import OCDocker.Docking.PLANTS as engine_mod  # type: ignore
        eng = "plants"

    # Validate engine binary availability based on configuration
    try:
        from OCDocker.Config import get_config
        config = get_config()
        _vina_bin = config.vina.executable
        _smina_bin = config.smina.executable
        _plants_bin = config.plants.executable
    except (ImportError, AttributeError):
        # Fallback if binaries are not configured
        _vina_bin = _smina_bin = _plants_bin = None

    def _exists_exe(p: Optional[str]) -> bool:
        if not p:
            return False
        if os.path.isabs(p):
            return os.path.isfile(p) and os.access(p, os.X_OK)
        return shutil.which(p) is not None

    if eng == "vina" and not _exists_exe(_vina_bin):
        print("Error: Vina binary not found. Check 'vina' in OCDocker.cfg or PATH.")
        return 2
    if eng == "smina" and not _exists_exe(_smina_bin):
        print("Error: Smina binary not found. Check 'smina' in OCDocker.cfg or PATH.")
        return 2
    if eng == "plants" and not _exists_exe(_plants_bin):
        print("Error: PLANTS binary not found. Check 'plants' in OCDocker.cfg or PATH.")
        return 2

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    name = args.name or Path(args.ligand).stem

    # Validate inputs and derive default locations (mimic Console/tests)
    receptor_path = _require_file(str(args.receptor), "--receptor")
    ligand_path = _require_file(str(args.ligand), "--ligand")
    box_path = _require_file(str(args.box), "--box")

    ligand_dir = ligand_path.parent
    receptor_dir = receptor_path.parent

    if eng == "vina":
        files_dir = ligand_dir / "vinaFiles"
        conf_path = files_dir / "conf_vina.txt"
        prep_rec = receptor_dir / "prepared_receptor.pdbqt"
        prep_lig = ligand_dir / "prepared_ligand.pdbqt"
        log_path = files_dir / f"{name}.log"
        out_pose = files_dir / f"{name}.pdbqt"
    elif eng == "smina":
        files_dir = ligand_dir / "sminaFiles"
        conf_path = files_dir / "conf_smina.txt"
        prep_rec = receptor_dir / "prepared_receptor.pdbqt"
        prep_lig = ligand_dir / "prepared_ligand.pdbqt"
        log_path = files_dir / f"{name}.log"
        out_pose = files_dir / f"{name}.pdbqt"
    else:  # plants
        files_dir = ligand_dir / "plantsFiles"
        conf_path = files_dir / "conf_plants.txt"
        prep_rec = receptor_dir / "prepared_receptor.mol2"
        prep_lig = ligand_dir / "prepared_ligand.mol2"
        log_path = files_dir / f"{name}.log"
        out_pose = files_dir  # PLANTS output directory

    # Create domain objects
    receptor = ocr.Receptor(str(receptor_path), name=f"{name}_receptor")
    ligand = ocl.Ligand(str(ligand_path), name=f"{name}_ligand")
    if eng == "vina":
        dock = engine_mod.Vina
        runner = dock(
            str(conf_path), str(args.box), receptor, str(prep_rec), ligand,
            str(prep_lig), str(log_path), str(out_pose), name=f"VINA {name}", overwrite_config=True,
        )
    elif eng == "smina":
        dock = engine_mod.Smina
        runner = dock(
            str(conf_path), str(args.box), receptor, str(prep_rec), ligand,
            str(prep_lig), str(log_path), str(out_pose), name=f"SMINA {name}", overwrite_config=True,
        )
    else:
        dock = engine_mod.PLANTS
        runner = dock(
            str(conf_path), str(args.box), receptor, str(prep_rec), ligand,
            str(prep_lig), str(log_path), str(out_pose), name=f"PLANTS {name}", overwrite_config=True,
        )

    # Prepare and run
    import os as _os
    prep_rec_path = str(prep_rec)
    prep_lig_path = str(prep_lig)
    # Overwrite handling: remove existing prepared files to force regeneration
    if args.overwrite:
        try:
            if _os.path.isfile(prep_rec_path):
                _os.remove(prep_rec_path)
        except (OSError, FileNotFoundError, PermissionError):
            # Ignore if file doesn't exist or can't be removed
            pass
        try:
            if _os.path.isfile(prep_lig_path):
                _os.remove(prep_lig_path)
        except (OSError, FileNotFoundError, PermissionError):
            # Ignore if file doesn't exist or can't be removed
            pass

    # Logs for preparation
    prep_rec_log = files_dir / "prepare_receptor.log"
    prep_lig_log = files_dir / "prepare_ligand.log"

    # Receptor preparation
    if not (_os.path.isfile(prep_rec_path) and _os.path.getsize(prep_rec_path) > 0):
        rc = runner.run_prepare_receptor(logFile=str(prep_rec_log))
        if isinstance(rc, tuple):
            rc = rc[0]
        if rc != 0 and eng in ("vina", "smina"):
            # Fallback via OpenBabel
            rc_fb = runner.run_prepare_receptor(logFile=str(prep_rec_log), useOpenBabel=True)
            if isinstance(rc_fb, tuple):
                rc_fb = rc_fb[0]
            if rc_fb != 0:
                print(f"Error: receptor preparation failed. See {prep_rec_log}")
                return int(rc)
        elif rc != 0:
            print(f"Error: receptor preparation failed. See {prep_rec_log}")
            return int(rc)

    # Ligand preparation
    if not (_os.path.isfile(prep_lig_path) and _os.path.getsize(prep_lig_path) > 0):
        rc = runner.run_prepare_ligand(logFile=str(prep_lig_log))
        if isinstance(rc, tuple):
            rc = rc[0]
        if rc != 0 and eng in ("vina", "smina"):
            # Fallback via OpenBabel
            rc_fb = runner.run_prepare_ligand(logFile=str(prep_lig_log), useOpenBabel=True)
            if isinstance(rc_fb, tuple):
                rc_fb = rc_fb[0]
            if rc_fb != 0:
                print(f"Error: ligand preparation failed. See {prep_lig_log}")
                return int(rc)
        elif rc != 0:
            print(f"Error: ligand preparation failed. See {prep_lig_log}")
            return int(rc)

    rc = runner.run_docking()
    if isinstance(rc, tuple):
        rc = rc[0]
    if rc != 0:
        return int(rc)

    if not args.skip_split and eng in ("vina", "smina"):
        _ = runner.split_poses(str(files_dir))

    if not args.skip_rescore:
        if eng in ("vina", "smina"):
            runner.run_rescore(str(files_dir), skipDefaultScoring=True)
        else:
            pose_list = runner.write_pose_list(overwrite=True)
            if pose_list:
                runner.run_rescore(pose_list, overwrite=True)

    print(f"Completed {eng} for job '{name}'. Outputs in: {files_dir}")
    # Optional DB store
    if args.store_db:
        try:
            # Ensure tables exist
            from OCDocker.DB.DB import create_tables  # type: ignore
            create_tables()
            from OCDocker.DB.Models.Complexes import Complexes  # type: ignore
            Complexes.insert_or_update({"name": name})
        except Exception as e:
            print(f"Warning: failed to store to DB: {e}")
    return 0

def cmd_shap(args: argparse.Namespace) -> int:  # pragma: no cover - delegates to external OCScore CLI
    '''Run SHAP analysis.

    This function serves as a command-line interface for running SHAP analysis
    on the specified data.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    # No heavy OCDocker env needed for SHAP module, just dispatch
    from OCDocker.OCScore.Analysis.SHAP.Cli import main as shap_main  # type: ignore
    return int(shap_main([
        "--storage", args.storage,
        "--ao_study", args.ao_study,
        "--nn_study", args.nn_study,
        "--seed_study", args.seed_study,
        "--mask_study", args.mask_study,
        "--df_path", args.df_path,
        "--base_models", args.base_models,
        "--study_number", str(args.study_number),
        "--out_dir", args.out_dir,
        "--explainer", args.explainer,
        *( ["--background_size", str(args.background_size)] if args.background_size is not None else [] ),
        *( ["--eval_size", str(args.eval_size)] if args.eval_size is not None else [] ),
        *( ["--stratify_by", *args.stratify_by] if args.stratify_by else [] ),
        "--seed", str(args.seed),
        *( ["--no_csv"] if args.no_csv else [] ),
    ]))


def _ensure_mol2_poses(pose_paths: List[str], dest_dir: Path, pose_engine_map: Dict[str, str] = None) -> Tuple[List[str], Dict[str, str]]:
    '''Ensure a list of poses in MOL2 format, converting when needed.

    Returns a list of .mol2 paths and a mapping mol2->original path.
    Uses unique filenames based on engine source to avoid overwriting.

    Parameters
    ----------
    pose_paths : List[str]
        List of pose file paths to ensure are in MOL2 format.
    dest_dir : Path
        Destination directory for converted MOL2 files.
    pose_engine_map : Dict[str, str], optional
        Mapping from pose path to engine name (vina, smina, plants) to create unique filenames.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        A tuple containing a list of .mol2 paths and a mapping from mol2 paths to original paths.
    '''
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    mol2_paths: List[str] = []
    mapping: Dict[str, str] = {}

    import OCDocker.Toolbox.Conversion as occonversion  # type: ignore
    for p in pose_paths:
        src = Path(p)
        if src.suffix.lower() == ".mol2":
            # Already MOL2 - use as-is but track mapping
            mol2_paths.append(str(src))
            mapping[str(src)] = str(src)
            continue
        
        # Create unique filename based on engine and original filename
        engine = pose_engine_map.get(str(src), "unknown") if pose_engine_map else "unknown"
        # Include engine in filename to avoid collisions
        unique_name = f"{engine}_{src.stem}.mol2"
        out = dest_dir / unique_name
        _ = occonversion.convert_mols(str(src), str(out), overwrite=True)
        mol2_paths.append(str(out))
        mapping[str(out)] = str(src)
    return mol2_paths, mapping


def cmd_pipeline(args: argparse.Namespace) -> int:  # pragma: no cover - heavy integration path assembling multiple engines
    '''Full multi-engine flow with clustering, rescoring and export.

    1) Run docking on selected engines.
    2) Convert poses to MOL2, cluster by RMSD and pick the medoid of the largest cluster.
    3) Rescore only the representative pose.
    4) Save representative.mol2 and summary.json (rescoring results).
    5) (Optional) Store minimal metadata to DB.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    # Bootstrap env
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror  # type: ignore
        import OCDocker.Toolbox.Logging as oclogging  # type: ignore
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    # Optionally set timeout for external processes
    if args.timeout:
        os.environ["OCDOCKER_TIMEOUT"] = str(args.timeout)

    # Domain imports
    import OCDocker.Ligand as ocl  # type: ignore
    import OCDocker.Receptor as ocr  # type: ignore
    import OCDocker.Docking.Vina as ocvina  # type: ignore
    import OCDocker.Docking.Smina as ocsmina  # type: ignore
    import OCDocker.Docking.PLANTS as ocplants  # type: ignore
    import OCDocker.Toolbox.MoleculeProcessing as ocmolproc  # type: ignore
    import OCDocker.Toolbox.Printing as ocprint  # type: ignore
    import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsd  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import json

    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.ligand).stem

    # Validate input files
    receptor_path = _require_file(str(args.receptor), "--receptor")
    ligand_path = _require_file(str(args.ligand), "--ligand")
    box_path = _require_file(str(args.box), "--box")

    receptor = ocr.Receptor(str(receptor_path), name=f"{name}_receptor")
    # Use just the name for ligand to avoid "ligand_ligand" duplication when input file is already named "ligand"
    ligand_name = name if not name.endswith("_ligand") else name[:-7]  # Remove "_ligand" suffix if present
    ligand = ocl.Ligand(str(ligand_path), name=ligand_name)

    engines = [e.strip().lower() for e in args.engines.split(',') if e.strip()]
    engines = [e for e in engines if e in ("vina", "smina", "plants")]
    if not engines:
        print("No valid engine provided. Use --engines vina,smina,plants")
        return 1
    
    # Get rescoring engines (default to same as docking engines if not specified)
    rescoring_engines = engines
    if args.rescoring_engines:
        rescoring_engines = [e.strip().lower() for e in args.rescoring_engines.split(",") if e.strip()]
        # Validate rescoring engines
        valid_rescoring = {"vina", "smina", "plants", "oddt"}
        invalid_rescoring = [e for e in rescoring_engines if e not in valid_rescoring]
        if invalid_rescoring:
            print(f"Error: invalid rescoring engines: {', '.join(invalid_rescoring)}. Valid options: vina, smina, plants, oddt")
            return 2
        # Filter to only valid engines
        rescoring_engines = [e for e in rescoring_engines if e in valid_rescoring]
        if not rescoring_engines:
            print("Error: no valid rescoring engines specified. Valid options: vina, smina, plants, oddt")
            return 2

    # Validate required binaries are available
    try:
        from OCDocker.Config import get_config
        config = get_config()
        _vina_bin = config.vina.executable
        _smina_bin = config.smina.executable
        _plants_bin = config.plants.executable
    except (ImportError, AttributeError):
        # Fallback if binaries are not configured
        _vina_bin = _smina_bin = _plants_bin = None

    def _exists_exe(p: Optional[str]) -> bool:
        if not p:
            return False
        if os.path.isabs(p):
            return os.path.isfile(p) and os.access(p, os.X_OK)
        return shutil.which(p) is not None

    missing = []
    for e in engines:
        if e == "vina" and not _exists_exe(_vina_bin):
            missing.append("vina")
        elif e == "smina" and not _exists_exe(_smina_bin):
            missing.append("smina")
        elif e == "plants" and not _exists_exe(_plants_bin):
            missing.append("plants")
    if missing:
        print(f"Error: missing engine binaries: {', '.join(missing)}. Check paths in OCDocker.cfg or PATH.")
        return 2

    all_poses: List[str] = []
    pose_engine_map: Dict[str, str] = {}  # Map pose path to engine name
    ctx: Dict[str, Dict[str, str]] = {}
    engine_errors: Dict[str, str] = {}
    import os as _os

    for eng in engines:
        e_dir = outdir / f"{eng}Files"; e_dir.mkdir(parents=True, exist_ok=True)
        try:
            if eng == "vina":
                conf = e_dir / "conf_vina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
                log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
                r = ocvina.Vina(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"VINA {name}", overwrite_config=True)
                # Only prepare receptor/ligand if they don't exist
                if not (_os.path.isfile(str(prep_r)) and _os.path.getsize(str(prep_r)) > 0):
                    rc = r.run_prepare_receptor(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Receptor preparation failed with code {rc}"
                        ocprint.print_warning(f"Vina receptor preparation failed. Continuing with other engines...")
                        continue
                if not (_os.path.isfile(str(prep_l)) and _os.path.getsize(str(prep_l)) > 0):
                    rc = r.run_prepare_ligand(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Ligand preparation failed with code {rc}"
                        ocprint.print_warning(f"Vina ligand preparation failed. Continuing with other engines...")
                        continue
                rc = r.run_docking(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0:
                    engine_errors[eng] = f"Docking failed with code {rc}"
                    ocprint.print_warning(f"Vina docking failed. Continuing with other engines...")
                    continue
                _ = r.split_poses(str(e_dir))
                poses = r.get_docked_poses()
                all_poses.extend(poses)
                # Track which engine each pose came from
                for pose in poses:
                    pose_engine_map[pose] = eng
                ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
            elif eng == "smina":
                conf = e_dir / "conf_smina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
                log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
                r = ocsmina.Smina(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"SMINA {name}", overwrite_config=True)
                # Only prepare receptor/ligand if they don't exist
                if not (_os.path.isfile(str(prep_r)) and _os.path.getsize(str(prep_r)) > 0):
                    rc = r.run_prepare_receptor(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Receptor preparation failed with code {rc}"
                        ocprint.print_warning(f"Smina receptor preparation failed. Continuing with other engines...")
                        continue
                if not (_os.path.isfile(str(prep_l)) and _os.path.getsize(str(prep_l)) > 0):
                    rc = r.run_prepare_ligand(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Ligand preparation failed with code {rc}"
                        ocprint.print_warning(f"Smina ligand preparation failed. Continuing with other engines...")
                        continue
                rc = r.run_docking(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0:
                    engine_errors[eng] = f"Docking failed with code {rc}"
                    ocprint.print_warning(f"Smina docking failed. Continuing with other engines...")
                    continue
                _ = r.split_poses(str(e_dir))
                poses = r.get_docked_poses()
                all_poses.extend(poses)
                # Track which engine each pose came from
                for pose in poses:
                    pose_engine_map[pose] = eng
                ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
            else:
                conf = e_dir / "conf_plants.txt"; prep_r = outdir / "prepared_receptor.mol2"; prep_l = outdir / "prepared_ligand.mol2"
                log = e_dir / f"{name}.log"; outp = e_dir
                r = ocplants.PLANTS(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"PLANTS {name}", overwrite_config=True)
                # Only prepare receptor/ligand if they don't exist
                if not (_os.path.isfile(str(prep_r)) and _os.path.getsize(str(prep_r)) > 0):
                    rc = r.run_prepare_receptor(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Receptor preparation failed with code {rc}"
                        ocprint.print_warning(f"PLANTS receptor preparation failed. Continuing with other engines...")
                        continue
                if not (_os.path.isfile(str(prep_l)) and _os.path.getsize(str(prep_l)) > 0):
                    rc = r.run_prepare_ligand(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Ligand preparation failed with code {rc}"
                        ocprint.print_warning(f"PLANTS ligand preparation failed. Continuing with other engines...")
                        continue
                rc = r.run_docking(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0:
                    engine_errors[eng] = f"Docking failed with code {rc}"
                    ocprint.print_warning(f"PLANTS docking failed. Continuing with other engines...")
                    continue
                poses = r.get_docked_poses()
                all_poses.extend(poses)
                # Track which engine each pose came from
                for pose in poses:
                    pose_engine_map[pose] = eng
                ctx[eng] = {"conf": str(conf), "dir": str(e_dir), "prep_rec": str(prep_r)}
        except Exception as e:
            engine_errors[eng] = f"Exception: {str(e)}"
            ocprint.print_warning(f"{eng.capitalize()} failed with exception: {e}. Continuing with other engines...")
            continue

    # Report any engine errors
    if engine_errors:
        print("\n=== Engine Errors ===")
        for eng, error_msg in engine_errors.items():
            print(f"{eng.capitalize()}: {error_msg}")
        print("")
    
    if not all_poses:
        if engine_errors:
            print("No poses were generated from any engine. All engines failed.")
            return 2
        else:
            print("No poses were generated.")
            return 2

    # Convert to MOL2 and cluster by RMSD
    # Use unique filenames based on engine to avoid overwriting
    mol2_dir = outdir / "poses_mol2"
    mol2_list, mol2_map = _ensure_mol2_poses(all_poses, mol2_dir, pose_engine_map)
    rmsd = ocmolproc.get_rmsd_matrix(mol2_list)
    df = pd.DataFrame(rmsd).loc[mol2_list, mol2_list]
    
    # Save RMSD matrix for reference
    rmsd_matrix_file = outdir / "rmsd_matrix.csv"
    df.to_csv(rmsd_matrix_file)
    
    # Perform clustering with plot output
    cluster_plot = outdir / "clustering_dendrogram.png"
    clusters = ocrmsd.cluster_rmsd(
        df,
        min_distance_threshold=args.cluster_min,
        max_distance_threshold=args.cluster_max,
        threshold_step=args.cluster_step,
        outputPlot=str(cluster_plot),
        molecule_name=name,
    )
    
    # Determine representative pose and save clustering results
    clustering_info = {
        "method": "rmsd_based_clustering",
        "total_poses": len(mol2_list),
        "representative_selection": None,
        "clusters": None,
        "cluster_sizes": None,
        "medoids": None,
    }
    
    if isinstance(clusters, int) or getattr(clusters, "size", 0) == 0:
        ocprint.print_warning(
            "Clustering did not converge or returned no labels; using the first pose as representative."
        )
        rep_mol2 = mol2_list[0]
        clustering_info["representative_selection"] = "first_pose_fallback"
        clustering_info["reason"] = "clustering_failed_or_no_labels"
    else:
        # Save cluster assignments
        cluster_assignments = pd.DataFrame({
            "pose_path": mol2_list,
            "cluster_id": clusters
        })
        cluster_assignments_file = outdir / "cluster_assignments.csv"
        cluster_assignments.to_csv(cluster_assignments_file, index=False)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        for cluster_id, size in zip(unique_clusters, counts):
            cluster_sizes[int(cluster_id)] = int(size)
        
        clustering_info["clusters"] = int(len(unique_clusters))
        clustering_info["cluster_sizes"] = cluster_sizes
        
        meds = ocrmsd.get_medoids(df, clusters, onlyBiggest=True)
        if not meds:
            ocprint.print_warning(
                "No medoid found from clusters; using the first pose as representative."
            )
            rep_mol2 = mol2_list[0]
            clustering_info["representative_selection"] = "first_pose_fallback"
            clustering_info["reason"] = "no_medoid_found"
        else:
            rep_mol2 = meds[0]
            clustering_info["representative_selection"] = "medoid_of_largest_cluster"
            clustering_info["medoids"] = [str(m) for m in meds]
            clustering_info["representative_pose"] = str(rep_mol2)
            # Find which cluster the representative belongs to
            rep_idx = mol2_list.index(rep_mol2)
            rep_cluster_id = int(clusters[rep_idx])
            clustering_info["representative_cluster_id"] = rep_cluster_id
            clustering_info["representative_cluster_size"] = cluster_sizes.get(rep_cluster_id, 0)

    # Get the original pose path for the representative
    rep_original = mol2_map.get(rep_mol2, rep_mol2)
    rep_engine = pose_engine_map.get(rep_original, None)
    
    # Convert representative to appropriate format for each engine's rescoring
    # Vina/Smina need PDBQT, PLANTS needs MOL2
    rep_pdbqt = None
    rep_mol2_final = None
    
    import OCDocker.Toolbox.Conversion as occonversion  # type: ignore
    import shutil
    
    if rep_original.endswith('.pdbqt'):
        # Already PDBQT - use for vina/smina
        rep_pdbqt = rep_original
        # Convert to MOL2 for PLANTS if needed
        rep_mol2_final = outdir / "representative_for_plants.mol2"
        occonversion.convert_mols(rep_original, str(rep_mol2_final), overwrite=True)
    elif rep_original.endswith('.mol2'):
        # Already MOL2 - use for PLANTS
        rep_mol2_final = rep_original
        # Convert to PDBQT for vina/smina if needed
        rep_pdbqt = outdir / "representative_for_vina_smina.pdbqt"
        occonversion.convert_mols(rep_original, str(rep_pdbqt), overwrite=True)
    else:
        # Fallback: use the mol2 version we have
        rep_mol2_final = rep_mol2
        rep_pdbqt = outdir / "representative_for_vina_smina.pdbqt"
        occonversion.convert_mols(rep_mol2, str(rep_pdbqt), overwrite=True)
    
    # Save representative in MOL2 format (for general use)
    rep_path = outdir / "representative.mol2"
    if rep_mol2_final and Path(rep_mol2_final).exists():
        shutil.copyfile(rep_mol2_final, rep_path)
    else:
        shutil.copyfile(rep_mol2, rep_path)
    
    # Save clustering information
    clustering_info_file = outdir / "clustering_info.json"
    clustering_info_file.write_text(json.dumps(clustering_info, indent=2))

    # Rescoring (representative only)
    # Only rescore with engines specified in --rescoring-engines (or same as docking engines if not specified)
    rescoring: Dict[str, Dict[str, float]] = {}
    # Get config for scoring functions
    from OCDocker.Config import get_config
    config = get_config()
    
    # VINA
    if "vina" in ctx and "vina" in rescoring_engines:
        from OCDocker.Docking.Vina import run_rescore as v_rescore, get_rescore_log_paths as v_logs, read_rescore_logs as v_read  # type: ignore
        if rep_pdbqt and Path(rep_pdbqt).exists():
            # Get scoring functions from config
            vina_sfs = config.vina.scoring_functions if config.vina.scoring_functions else ["vina"]
            for sf in vina_sfs:
                try:
                    v_rescore(ctx["vina"]["conf"], str(rep_pdbqt), ctx["vina"]["dir"], sf, splitLigand=False, overwrite=True)
                except Exception as e:
                    ocprint.print_warning(f"Vina rescoring with {sf} failed: {e}. Continuing with other scoring functions...")
            try:
                # Wait a moment for files to be written (in case of async operations)
                import time
                time.sleep(0.5)
                log_paths = v_logs(ctx["vina"]["dir"])
                if not log_paths:
                    ocprint.print_warning(f"No Vina rescoring log files found in {ctx['vina']['dir']}. Check if rescoring completed successfully.")
                    # Debug: list files in directory
                    if Path(ctx["vina"]["dir"]).exists():
                        files = list(Path(ctx["vina"]["dir"]).glob("*"))
                        ocprint.print_warning(f"Files in Vina directory: {[f.name for f in files]}")
                else:
                    ocprint.printv(f"Found Vina rescoring log files: {log_paths}")
                    data = v_read(log_paths, onlyBest=True)
                    if not data:
                        ocprint.print_warning(f"Vina rescoring log files found but no data extracted. Log paths: {log_paths}")
                    else:
                        vals: Dict[str, float] = {}
                        # Data structure: Dict[str, List[Union[str, float]]] according to type hint, but actual return is Dict[str, float]
                        # Key format: "rescoring_{scoring_function}_{pose_number}" or "vina_{scoring_function}_rescoring"
                        for k, v in data.items():
                            try:
                                # v can be a float or a list - handle both cases
                                if isinstance(v, (int, float)):
                                    # Normalize key: extract scoring function and create clean key
                                    # Keys can be: "vina_vina_rescoring", "rescoring_vina_1", "rescoring_vinardo_1", etc.
                                    if k.startswith("vina_") and k.endswith("_rescoring"):
                                        # Format: "vina_{scoring_function}_rescoring"
                                        sf_name = k.replace("vina_", "").replace("_rescoring", "")
                                        clean_key = f"vina_{sf_name}"
                                    elif k.startswith("rescoring_"):
                                        # Format: "rescoring_{scoring_function}_{pose_number}"
                                        parts = k.replace("rescoring_", "").split("_")
                                        if len(parts) >= 1:
                                            sf_name = parts[0]
                                            clean_key = f"vina_{sf_name}"
                                        else:
                                            clean_key = k
                                    else:
                                        clean_key = k
                                    vals[clean_key] = float(v)
                                elif isinstance(v, list) and len(v) > 0:
                                    # Handle list case (type hint says List[Union[str, float]])
                                    # Extract the numeric value
                                    numeric_val = None
                                    for item in v:
                                        if isinstance(item, (int, float)):
                                            numeric_val = float(item)
                                            break
                                        elif isinstance(item, str):
                                            try:
                                                numeric_val = float(item)
                                                break
                                            except ValueError:
                                                continue
                                    if numeric_val is not None:
                                        # Normalize key
                                        if k.startswith("vina_") and k.endswith("_rescoring"):
                                            sf_name = k.replace("vina_", "").replace("_rescoring", "")
                                            clean_key = f"vina_{sf_name}"
                                        elif k.startswith("rescoring_"):
                                            parts = k.replace("rescoring_", "").split("_")
                                            if len(parts) >= 1:
                                                sf_name = parts[0]
                                                clean_key = f"vina_{sf_name}"
                                            else:
                                                clean_key = k
                                        else:
                                            clean_key = k
                                        vals[clean_key] = numeric_val
                            except (ValueError, TypeError, KeyError) as e:
                                ocprint.print_warning(f"Failed to parse Vina rescoring value for {k}: {e}. Value type: {type(v)}, value: {v}")
                        if vals:
                            rescoring["vina"] = vals
                        else:
                            ocprint.print_warning(f"Vina rescoring data found but no valid values extracted. Data structure: {data}")
            except Exception as e:
                ocprint.print_warning(f"Failed to read Vina rescoring results: {e}")
                import traceback
                ocprint.print_warning(f"Traceback: {traceback.format_exc()}")
    # SMINA
    if "smina" in rescoring_engines:
        from OCDocker.Docking.Smina import run_rescore as s_rescore, get_rescore_log_paths as s_logs, read_rescore_logs as s_read  # type: ignore
        if rep_pdbqt and Path(rep_pdbqt).exists():
            # If smina wasn't docked, we can still use vina's prepared files (they share PDBQT format)
            # Create smina context if it doesn't exist
            if "smina" not in ctx:
                # Use vina's config if available, otherwise create a new smina config
                if "vina" in ctx:
                    # Create smina directory and config
                    smina_dir = outdir / "sminaFiles"
                    smina_dir.mkdir(parents=True, exist_ok=True)
                    smina_conf = smina_dir / "conf_smina.txt"
                    # Create a Smina object just to generate the config file
                    import OCDocker.Docking.Smina as ocsmina  # type: ignore
                    prep_r = outdir / "prepared_receptor.pdbqt"
                    prep_l = outdir / "prepared_ligand.pdbqt"
                    smina_obj = ocsmina.Smina(str(smina_conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(smina_dir / f"{name}.log"), str(smina_dir / f"{name}.pdbqt"), name=f"SMINA {name}", overwrite_config=True)
                    ctx["smina"] = {"conf": str(smina_conf), "dir": str(smina_dir)}
                else:
                    ocprint.print_warning("Smina rescoring requested but neither Smina nor Vina was docked. Smina rescoring requires PDBQT format files.")
                    # Skip smina rescoring
                    pass
            if "smina" in ctx:
                # Get scoring functions from config
                smina_sfs = config.smina.scoring_functions if config.smina.scoring_functions else ["vinardo"]
                for sf in smina_sfs:
                    try:
                        s_rescore(ctx["smina"]["conf"], str(rep_pdbqt), ctx["smina"]["dir"], sf, splitLigand=False, overwrite=True)
                    except Exception as e:
                        ocprint.print_warning(f"Smina rescoring with {sf} failed: {e}. Continuing with other scoring functions...")
                try:
                    # Wait a moment for files to be written (in case of async operations)
                    import time
                    time.sleep(0.5)
                    log_paths = s_logs(ctx["smina"]["dir"])
                    if not log_paths:
                        ocprint.print_warning(f"No Smina rescoring log files found in {ctx['smina']['dir']}")
                        # Debug: list files in directory
                        if Path(ctx["smina"]["dir"]).exists():
                            files = list(Path(ctx["smina"]["dir"]).glob("*"))
                            ocprint.print_warning(f"Files in Smina directory: {[f.name for f in files]}")
                    else:
                        ocprint.printv(f"Found Smina rescoring log files: {log_paths}")
                        data = s_read(log_paths, onlyBest=True)
                        vals: Dict[str, float] = {}
                        # Data structure: Dict[str, float] (read_rescoring_log returns float, not list)
                        # Key format: "rescoring_{scoring_function}_{pose_number}" or "smina_{scoring_function}_rescoring"
                        for k, v in data.items():
                            try:
                                # v is a float (from read_rescoring_log)
                                if isinstance(v, (int, float)):
                                    # Normalize key: extract scoring function and create clean key
                                    # Keys can be: "smina_vinardo_rescoring", "rescoring_vina_1", "rescoring_dkoes_scoring_1", etc.
                                    if k.startswith("smina_") and k.endswith("_rescoring"):
                                        # Format: "smina_{scoring_function}_rescoring"
                                        sf_name = k.replace("smina_", "").replace("_rescoring", "")
                                        clean_key = f"smina_{sf_name}"
                                    elif k.startswith("rescoring_"):
                                        # Format: "rescoring_{scoring_function}_{pose_number}"
                                        parts = k.replace("rescoring_", "").split("_")
                                        if len(parts) >= 1:
                                            # Handle multi-part scoring function names like "dkoes_scoring"
                                            # Try to match against known scoring functions
                                            sf_name = None
                                            for known_sf in smina_sfs:
                                                # Check if the key starts with this scoring function
                                                if "_".join(parts[:len(known_sf.split("_"))]) == known_sf:
                                                    sf_name = known_sf
                                                    break
                                            if not sf_name and parts:
                                                # Fallback: use first part
                                                sf_name = parts[0]
                                            clean_key = f"smina_{sf_name}" if sf_name else k
                                        else:
                                            clean_key = k
                                    else:
                                        clean_key = k
                                    vals[clean_key] = float(v)
                                elif isinstance(v, list) and len(v) > 0:
                                    # Handle list case (shouldn't happen but just in case)
                                    vals[k] = float(v[0] if not isinstance(v[0], (list, tuple)) else v[0][0])
                            except (ValueError, TypeError, KeyError) as e:
                                ocprint.print_warning(f"Failed to parse Smina rescoring value for {k}: {e}")
                        if vals:
                            rescoring["smina"] = vals
                        else:
                            ocprint.print_warning(f"Smina rescoring data found but no valid values extracted. Data structure: {data}")
                except Exception as e:
                    ocprint.print_warning(f"Failed to read Smina rescoring results: {e}")
    # PLANTS
    if "plants" in ctx and "plants" in rescoring_engines:
        from OCDocker.Docking.PLANTS import write_rescoring_config_file, run_rescore as p_rescore, get_binding_site  # type: ignore
        pose_list = outdir / "pose_list_single.txt"
        # Use MOL2 format for PLANTS rescoring
        plants_rep = str(rep_mol2_final) if rep_mol2_final and Path(rep_mol2_final).exists() else str(rep_path)
        pose_list.write_text(plants_rep + "\n")
        # Extract center/radius from the box
        center, radius = get_binding_site(str(box_path))  # type: ignore
        # Get scoring functions from config
        plants_sfs = config.plants.scoring_functions if config.plants.scoring_functions else ["chemplp", "plp", "plp95"]
        for sf in plants_sfs:
            try:
                # Each scoring function must have its own output directory (PLANTS requirement)
                outPath_sf = Path(ctx["plants"]["dir"]) / f"run_{sf}"
                conf_sf = Path(ctx["plants"]["dir"]) / f"{name}_rescoring_{sf}.txt"
                write_rescoring_config_file(str(conf_sf), ctx["plants"]["prep_rec"], str(pose_list), str(outPath_sf), center[0], center[1], center[2], radius, scoringFunction=sf)
                p_rescore(str(conf_sf), str(pose_list), str(outPath_sf), ctx["plants"]["prep_rec"], sf, center[0], center[1], center[2], radius, overwrite=True)
            except Exception as e:
                ocprint.print_warning(f"PLANTS rescoring with {sf} failed: {e}. Continuing with other scoring functions...")
        # Read PLANTS rescoring results
        try:
            from OCDocker.Docking.PLANTS import read_log as plants_read_log  # type: ignore
            plants_rescoring_data: Dict[str, float] = {}
            for sf in plants_sfs:
                # Each scoring function has its own directory: run_{scoring_function}
                ranking_file = Path(ctx["plants"]["dir"]) / f"run_{sf}" / "bestranking.csv"
                if ranking_file.exists():
                    try:
                        log_data = plants_read_log(str(ranking_file), onlyBest=True)
                        if log_data:
                            # PLANTS returns Dict[int, Dict[int, float]] where first int is pose number, second is score type
                            # When onlyBest=True, typically only one pose (key 1)
                            for pose_num, scores in log_data.items():
                                # scores is Dict[int, float] where int is score type code
                                # We want TOTAL_SCORE which is typically the first or main score
                                # Extract all scores and use meaningful keys
                                for score_type_code, score_value in scores.items():
                                    # Use scoring function name and score type
                                    key = f"plants_{sf}"
                                    # Store the main score (TOTAL_SCORE is typically the first one)
                                    if key not in plants_rescoring_data or score_type_code == 0:
                                        plants_rescoring_data[key] = float(score_value) if isinstance(score_value, (int, float)) else float(score_value[0]) if isinstance(score_value, (list, tuple)) else 0.0
                                break  # Only take first pose when onlyBest=True
                    except Exception as e:
                        ocprint.print_warning(f"Failed to read PLANTS rescoring results for {sf}: {e}")
                else:
                    ocprint.print_warning(f"PLANTS rescoring ranking file not found: {ranking_file}")
            if plants_rescoring_data:
                rescoring["plants"] = plants_rescoring_data
            else:
                ocprint.print_warning("No PLANTS rescoring data found")
        except Exception as e:
            ocprint.print_warning(f"Failed to read PLANTS rescoring results: {e}")
    
    # ODDT (can rescore independently, doesn't require docking)
    if "oddt" in rescoring_engines:
        try:
            from OCDocker.Rescoring.ODDT import run_oddt, df_to_dict  # type: ignore
            # ODDT needs the prepared receptor - use from any available engine
            prepared_receptor = None
            if "vina" in ctx or "smina" in ctx:
                # Use PDBQT receptor from vina/smina
                prepared_receptor = str(outdir / "prepared_receptor.pdbqt")
            elif "plants" in ctx:
                # Use MOL2 receptor from PLANTS
                prepared_receptor = ctx["plants"]["prep_rec"]
            else:
                # Fallback: try to find any prepared receptor
                pdbqt_rec = outdir / "prepared_receptor.pdbqt"
                mol2_rec = outdir / "prepared_receptor.mol2"
                if pdbqt_rec.exists():
                    prepared_receptor = str(pdbqt_rec)
                elif mol2_rec.exists():
                    prepared_receptor = str(mol2_rec)
            
            if prepared_receptor and Path(prepared_receptor).exists():
                # ODDT needs MOL2 format for ligand
                oddt_ligand = str(rep_mol2_final) if rep_mol2_final and Path(rep_mol2_final).exists() else str(rep_path)
                oddt_output = outdir / "oddt_rescoring"
                oddt_output.mkdir(parents=True, exist_ok=True)
                
                # Run ODDT rescoring
                try:
                    df = run_oddt(
                        prepared_receptor,
                        oddt_ligand,
                        name,
                        str(oddt_output),
                        overwrite=True,
                        returnData=True
                    )
                    
                    # Check if run_oddt returned an error code (int) instead of DataFrame
                    if isinstance(df, int):
                        ocprint.print_warning(f"ODDT rescoring returned error code: {df}. Check ODDT configuration and logs.")
                    elif df is not None:
                        try:
                            oddt_dict = df_to_dict(df)
                            # Extract values from ODDT results
                            oddt_vals: Dict[str, float] = {}
                            if oddt_dict:
                                # Get the first (and typically only) entry (ligand name is the key)
                                first_key = list(oddt_dict.keys())[0]
                                for score_name, score_value in oddt_dict[first_key].items():
                                    try:
                                        # Skip non-numeric columns
                                        if score_name.lower() in ['ligand_name', 'name']:
                                            continue
                                        key = f"oddt_{score_name}"
                                        if isinstance(score_value, (int, float)):
                                            oddt_vals[key] = float(score_value)
                                        elif isinstance(score_value, (list, tuple)) and len(score_value) > 0:
                                            oddt_vals[key] = float(score_value[0])
                                        elif isinstance(score_value, str):
                                            # Try to convert string to float
                                            try:
                                                oddt_vals[key] = float(score_value)
                                            except ValueError:
                                                pass
                                    except (ValueError, TypeError) as e:
                                        ocprint.print_warning(f"Failed to parse ODDT score {score_name}: {e}")
                            if oddt_vals:
                                rescoring["oddt"] = oddt_vals
                            else:
                                ocprint.print_warning(f"ODDT rescoring completed but no valid scores extracted. Dict keys: {list(oddt_dict.keys()) if oddt_dict else 'None'}")
                        except Exception as e:
                            ocprint.print_warning(f"Failed to convert ODDT results to dictionary: {e}")
                            import traceback
                            ocprint.print_warning(f"Traceback: {traceback.format_exc()}")
                    else:
                        ocprint.print_warning("ODDT rescoring returned None. Check ODDT configuration and logs.")
                except Exception as e:
                    ocprint.print_warning(f"ODDT rescoring failed: {e}")
                    import traceback
                    ocprint.print_warning(f"Traceback: {traceback.format_exc()}")
            else:
                ocprint.print_warning("ODDT rescoring skipped: no prepared receptor found")
        except ImportError as e:
            ocprint.print_warning(f"ODDT rescoring not available (import error): {e}")
        except Exception as e:
            ocprint.print_warning(f"ODDT rescoring failed: {e}")
    
    # Write summary
    # Track which engines were actually used for rescoring (those with results)
    rescoring_engines_used = list(rescoring.keys())
    summ = {
        "job": name,
        "engines": engines,
        "rescoring_engines": rescoring_engines_used,  # Engines that actually produced rescoring results
        "representative_pose": str(rep_path),
        "clustering": clustering_info,
        "rescoring": rescoring,
    }
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))

    if args.store_db:
        try:
            from OCDocker.DB.DB import create_tables  # type: ignore
            create_tables()
            from OCDocker.DB.Models.Complexes import Complexes  # type: ignore
            Complexes.insert_or_update({"name": name})
        except Exception as e:
            print(f"Warning: failed to store to DB: {e}")

    print(f"Pipeline finished. Representative pose: {rep_path}")
    return 0

def cmd_console(args: argparse.Namespace) -> int:  # pragma: no cover - interactive console, unsuitable for automated coverage
    '''Open an interactive console with OCDockerConsole namespace.

    Respects global flags by bootstrapping environment first.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    # Bootstrap env to ensure Initialise is safe to import
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror  # type: ignore
        import OCDocker.Toolbox.Logging as oclogging  # type: ignore
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    # Import console module and open interactive session with its namespace
    try:
        # OCDockerConsole.py is at project root, add it to path if needed
        import OCDockerConsole as occ  # type: ignore
    except ImportError:
        # Try to find OCDockerConsole.py relative to the package
        try:
            ocdocker_pkg = Path(__file__).resolve().parent.parent  # OCDocker package dir
            project_root = ocdocker_pkg.parent  # Project root where OCDockerConsole.py lives
            if project_root not in sys.path:
                sys.path.insert(0, str(project_root))
            import OCDockerConsole as occ  # type: ignore
        except Exception as e:
            print(f"Failed to import OCDockerConsole: {e}")
            return 1
    except Exception as e:
        print(f"Failed to import OCDockerConsole: {e}")
        return 1

    print("Launching OCDocker Console. Press Ctrl-D to exit.")
    try:
        # Expose console namespace without dunders
        local_ns = {k: v for k, v in vars(occ).items() if not k.startswith('__')}
        
        # Try to use IPython if available, otherwise fallback to standard Python console
        try:
            from IPython import embed  # type: ignore
            # Determine if colors should be enabled (when called from a terminal like bash)
            colors = 'NoColor'
            if sys.stdout.isatty() and os.getenv('TERM') and 'dumb' not in os.getenv('TERM', ''):
                # Enable colors when running in a terminal (bash, etc.)
                # 'Linux' provides good color scheme for terminals
                colors = 'Linux'
            # Use IPython with the console namespace and colors enabled
            embed(user_ns=local_ns, banner1="", colors=colors, display_banner=False)
        except ImportError:
            # Fallback to standard Python console
            import code
            # Setup tab-completion with the console namespace, if readline is available
            try:
                import readline  # type: ignore
                import rlcompleter  # type: ignore
                completer = rlcompleter.Completer(local_ns)
                readline.set_completer(completer.complete)  # type: ignore
                readline.parse_and_bind('tab: complete')  # type: ignore
                # History file (optional)
                hist = os.path.expanduser('~/.ocdocker_console_history')
                try:
                    readline.read_history_file(hist)
                except (OSError, FileNotFoundError):
                    # Ignore if history file doesn't exist or can't be read
                    pass
            except (ImportError, AttributeError):
                # Ignore if readline is not available
                pass
            # Avoid printing the console banner twice: it's already printed on import
            code.interact(banner="", local=local_ns)
            # Save history (best-effort)
            try:
                if 'readline' in sys.modules:
                    sys.modules['readline'].write_history_file(os.path.expanduser('~/.ocdocker_console_history'))
            except (OSError, AttributeError):
                # Ignore if history file can't be written or readline not available
                pass
    except Exception as e:
        print(f"Interactive console exited with error: {e}")
        return 1
    return 0

def cmd_script(args: argparse.Namespace) -> int:  # pragma: no cover - script execution is user-provided code
    '''Run a Python script with OCDocker libraries pre-loaded.

    Bootstraps the environment, loads all OCDocker modules, and executes
    the provided script file with those modules available in the namespace.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments. Must contain 'script_file' and optionally 'script_args'.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    # Bootstrap env to ensure Initialise is safe to import
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror  # type: ignore
        import OCDocker.Toolbox.Logging as oclogging  # type: ignore
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    # Validate script file exists
    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        return 1
    
    if not script_path.is_file():
        print(f"Error: Path is not a file: {script_path}")
        return 1

    # Load OCDocker libraries into namespace (similar to OCDockerConsole)
    script_namespace = {}
    
    # Import all OCDocker modules
    try:
        # Import Initialise module and add all non-dunder symbols to namespace
        import OCDocker.Initialise as ocinit  # type: ignore
        for k, v in vars(ocinit).items():
            if not k.startswith('__'):
                script_namespace[k] = v
        
        import OCDocker.Toolbox as octools  # type: ignore
        script_namespace['octools'] = octools
        
        import OCDocker.Ligand as ocl  # type: ignore
        script_namespace['ocl'] = ocl
        
        import OCDocker.Receptor as ocr  # type: ignore
        script_namespace['ocr'] = ocr
        
        import OCDocker.Docking.Vina as ocvina  # type: ignore
        script_namespace['ocvina'] = ocvina
        
        import OCDocker.Docking.Smina as ocsmina  # type: ignore
        script_namespace['ocsmina'] = ocsmina
        
        import OCDocker.Docking.PLANTS as ocplants  # type: ignore
        script_namespace['ocplants'] = ocplants
        
        import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust  # type: ignore
        script_namespace['ocrmsdclust'] = ocrmsdclust
        
        import OCDocker.Rescoring.ODDT as ocoddt  # type: ignore
        script_namespace['ocoddt'] = ocoddt
        
        import OCDocker.Toolbox.Conversion as occonversion  # type: ignore
        script_namespace['occonversion'] = occonversion
        
        import OCDocker.Toolbox.MoleculeProcessing as ocmolproc  # type: ignore
        script_namespace['ocmolproc'] = ocmolproc
        
        # Add standard library modules that scripts commonly need
        from glob import glob
        from pprint import pprint
        script_namespace['os'] = os
        script_namespace['sys'] = sys
        script_namespace['Path'] = Path
        script_namespace['glob'] = glob
        script_namespace['pprint'] = pprint
        
    except Exception as e:
        print(f"Error loading OCDocker libraries: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Update sys.argv to include script args for the script's use
    original_argv = sys.argv[:]
    try:
        # Set sys.argv to: [script_file, ...script_args]
        sys.argv = [str(script_path)] + (args.script_args or [])
        
        # Read and execute the script
        script_content = script_path.read_text(encoding='utf-8')
        
        # Compile the script for better error messages
        try:
            compiled_script = compile(script_content, str(script_path), 'exec')
        except SyntaxError as e:
            print(f"Syntax error in script {script_path}:")
            print(f"  Line {e.lineno}: {e.text}")
            print(f"  {e.msg}")
            return 1
        
        # Execute the script with the loaded namespace
        exec(compiled_script, script_namespace)
        
        return 0
        
    except SystemExit as e:
        # Script called sys.exit(), respect its exit code
        return int(e.code) if e.code is not None else 0
    except KeyboardInterrupt:
        print("\nScript execution interrupted by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error executing script {script_path}:")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def cmd_doctor(args: argparse.Namespace) -> int:  # pragma: no cover - environment probing is platform-dependent
    '''Run diagnostics: config, binaries, Python deps, DB connectivity.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    # Bootstrap to load config and DB
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror  # type: ignore
        import OCDocker.Toolbox.Logging as oclogging  # type: ignore
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    report: Dict[str, Dict[str, str]] = {}

    # Config source
    try:
        import OCDocker.Initialise as OCI  # type: ignore
        cfg = getattr(OCI, 'config_file', None)
        report['config'] = {
            'path': str(cfg) if cfg else 'unknown',
        }
    except Exception as e:
        report['config'] = {'error': f'{e}'}

    # Engine binaries
    def _exists_exe(p: Optional[str]) -> bool:
        if not p:
            return False
        if os.path.isabs(p):
            return os.path.isfile(p) and os.access(p, os.X_OK)
        return shutil.which(p) is not None

    try:
        from OCDocker.Config import get_config
        config = get_config()
        v = config.vina.executable
        s = config.smina.executable
        p = config.plants.executable
    except Exception:
        # Fallback if config is not available
        v = s = p = None
    report['binaries'] = {
        'vina': 'OK' if _exists_exe(v) else 'MISSING',
        'smina': 'OK' if _exists_exe(s) else 'MISSING',
        'plants': 'OK' if _exists_exe(p) else 'MISSING',
    }

    # Python dependencies
    # SECURITY NOTE: Dynamic import is used here to check for optional dependencies.
    # The module names are hardcoded in a whitelist ('rdkit', 'Bio', 'oddt', 'sqlalchemy')
    # and never come from user input, making this safer from injection attacks.
    pydeps = {}
    
    # Whitelist of allowed module names for dependency checking
    ALLOWED_DEPENDENCY_MODULES = ('rdkit', 'Bio', 'oddt', 'sqlalchemy')
    for mod in ALLOWED_DEPENDENCY_MODULES:
        # Validate module name contains only safe characters (alphanumeric and underscore)
        if not isinstance(mod, str) or not mod.replace('_', '').isalnum():
            pydeps[mod] = 'INVALID_MODULE_NAME'
            continue
        try:
            __import__(mod)
            pydeps[mod] = 'OK'
        except Exception as e:
            pydeps[mod] = f'MISSING ({e.__class__.__name__})'
    report['python_deps'] = pydeps

    # DB connectivity
    try:
        eng = getattr(OCI, 'engine', None)
        if eng is None:
            report['database'] = {'status': 'MISSING ENGINE'}
        else:
            conn = eng.connect()
            conn.close()
            report['database'] = {'status': 'OK'}
    except Exception as e:
        report['database'] = {'status': f'ERROR ({e})'}

    # Summary printout
    print(json.dumps(report, indent=2))

    return 0

def main(argv: Optional[list[str]] = None) -> int:
    '''Main entry point for the CLI.
    
    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''
    
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
