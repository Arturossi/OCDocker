#!/usr/bin/env python3
"""Single-engine virtual screening (vs) CLI command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.CLI.common import (
    _bootstrap_ocdocker_env,
    _db_dependencies_available,
    _preparse_global_args,
    _print_optional_dependency_hint,
    _require_file,
)
from OCDocker.CLI import workflow as cli_workflow

LOGGER = oclogging.get_logger("cli")

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

    if args.store_db:
        db_ok, db_exc = _db_dependencies_available()
        if not db_ok and db_exc is not None:
            return _print_optional_dependency_hint(
                feature="database storage (--store-db)",
                extra="db",
                exc=db_exc,
            )

    # Bootstrap environment before importing engines
    globals_ns = _preparse_global_args(sys.argv[1:])
    store_db_requested = bool(getattr(args, "store_db", False))
    setattr(globals_ns, "_ocdocker_init_db", store_db_requested)
    setattr(globals_ns, "_ocdocker_create_db_if_missing", store_db_requested)
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror
        import OCDocker.Toolbox.Logging as oclogging
        oclogging.configure(
            level=ocerror.Error.get_output_level(),
            log_file=args.log_file,
            to_stdout=(not args.no_stdout_log),
        )
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    # Optionally set timeout for external processes
    if args.timeout:
        os.environ["OCDOCKER_TIMEOUT"] = str(args.timeout)

    # Imports after env is ready
    try:
        import OCDocker.Ligand as ocl
        import OCDocker.Receptor as ocr
        import importlib
    except ModuleNotFoundError as exc:
        extra = _suggest_extra_for_missing_module(getattr(exc, "name", ""))
        return _print_optional_dependency_hint(
            feature="single-engine docking workflow",
            extra=extra,
            exc=exc,
        )
    engine_mod: Any
    if args.engine == "vina":
        engine_mod = importlib.import_module("OCDocker.Docking.Vina")
        eng = "vina"
    elif args.engine == "smina":
        engine_mod = importlib.import_module("OCDocker.Docking.Smina")
        eng = "smina"
    elif args.engine == "gnina":
        engine_mod = importlib.import_module("OCDocker.Docking.Gnina")
        eng = "gnina"
    else:
        engine_mod = importlib.import_module("OCDocker.Docking.PLANTS")
        eng = "plants"

    # Validate engine binary availability based on configuration
    _vina_bin: Optional[str] = None
    _smina_bin: Optional[str] = None
    _gnina_bin: Optional[str] = None
    _plants_bin: Optional[str] = None
    try:
        from OCDocker.Config import get_config
        config = get_config()
        _vina_bin = config.vina.executable
        _smina_bin = config.smina.executable
        _gnina_bin = config.gnina.executable
        _plants_bin = config.plants.executable
    except (ImportError, AttributeError):
        # Fallback if binaries are not configured
        pass

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
    if eng == "gnina" and not _exists_exe(_gnina_bin):
        print("Error: Gnina binary not found. Check 'gnina' in OCDocker.cfg or PATH.")
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
    boxes = cli_workflow._list_boxes(ligand_dir, box_path, args.all_boxes)
    if args.all_boxes and not boxes:
        print("Warning: no box*.pdb files found. Skipping docking.")
        return 2
    use_multi_boxes = args.all_boxes and len(boxes) > 1

    if eng == "vina":
        base_files_dir = ligand_dir / "vinaFiles"
        conf_name = "conf_vina.txt"
        prep_rec = receptor_dir / "prepared_receptor.pdbqt"
        prep_lig = ligand_dir / "prepared_ligand.pdbqt"
        log_name = f"{name}.log"
        out_name = f"{name}.pdbqt"
    elif eng == "smina":
        base_files_dir = ligand_dir / "sminaFiles"
        conf_name = "conf_smina.txt"
        prep_rec = receptor_dir / "prepared_receptor.pdbqt"
        prep_lig = ligand_dir / "prepared_ligand.pdbqt"
        log_name = f"{name}.log"
        out_name = f"{name}.pdbqt"
    elif eng == "gnina":
        base_files_dir = ligand_dir / "gninaFiles"
        conf_name = "conf_gnina.conf"
        prep_rec = receptor_dir / "prepared_receptor.pdbqt"
        prep_lig = ligand_dir / "prepared_ligand.pdbqt"
        log_name = f"{name}.log"
        out_name = f"{name}.pdbqt"
    else:  # plants
        base_files_dir = ligand_dir / "plantsFiles"
        conf_name = "conf_plants.txt"
        prep_rec = receptor_dir / "prepared_receptor.mol2"
        prep_lig = ligand_dir / "prepared_ligand.mol2"
        log_name = f"{name}.log"
        out_name = None  # PLANTS output directory
    base_files_dir.mkdir(parents=True, exist_ok=True)

    # Create domain objects
    receptor = ocr.Receptor(str(receptor_path), name=f"{name}_receptor")
    ligand = ocl.Ligand(str(ligand_path), name=f"{name}_ligand")

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
    prep_rec_log = base_files_dir / "prepare_receptor.log"
    prep_lig_log = base_files_dir / "prepare_ligand.log"

    overall_rc = 0
    prep_done = False
    for box in boxes:
        box_id = box.stem
        box_files_dir = base_files_dir / box_id if use_multi_boxes else base_files_dir
        box_files_dir.mkdir(parents=True, exist_ok=True)
        conf_path = box_files_dir / conf_name
        log_path = box_files_dir / log_name
        out_pose = box_files_dir if out_name is None else box_files_dir / out_name

        if eng == "vina":
            dock = engine_mod.Vina
            runner = dock(
                str(conf_path), str(box), receptor, str(prep_rec), ligand,
                str(prep_lig), str(log_path), str(out_pose), name=f"VINA {name}", overwrite_config=True,
            )
        elif eng == "smina":
            dock = engine_mod.Smina
            runner = dock(
                str(conf_path), str(box), receptor, str(prep_rec), ligand,
                str(prep_lig), str(log_path), str(out_pose), name=f"SMINA {name}", overwrite_config=True,
            )
        elif eng == "gnina":
            dock = engine_mod.Gnina
            runner = dock(
                str(conf_path), str(box), receptor, str(prep_rec), ligand,
                str(prep_lig), str(log_path), str(out_pose), name=f"GNINA {name}", overwrite_config=True,
            )
        else:
            dock = engine_mod.PLANTS
            runner = dock(
                str(conf_path), str(box), receptor, str(prep_rec), ligand,
                str(prep_lig), str(log_path), str(out_pose), name=f"PLANTS {name}", overwrite_config=True,
            )

        if not prep_done:
            # Receptor preparation
            if not (_os.path.isfile(prep_rec_path) and _os.path.getsize(prep_rec_path) > 0):
                if eng in ("vina", "smina", "plants"):
                    rc = runner.run_prepare_receptor(logFile=str(prep_rec_log))
                else:
                    rc = runner.run_prepare_receptor(overwrite=args.overwrite)
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
                if eng in ("vina", "smina", "plants"):
                    rc = runner.run_prepare_ligand(logFile=str(prep_lig_log))
                else:
                    rc = runner.run_prepare_ligand(overwrite=args.overwrite)
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
            prep_done = True

        rc = runner.run_docking()
        if isinstance(rc, tuple):
            rc = rc[0]
        if rc != 0:
            overall_rc = int(rc)
            print(f"Warning: docking failed for box '{box_id}'.")
            continue

        if not args.skip_split and eng in ("vina", "smina", "gnina"):
            _ = runner.split_poses(str(box_files_dir))

        if not args.skip_rescore:
            if eng in ("vina", "smina", "gnina"):
                runner.run_rescore(str(box_files_dir), prep_lig_path, skipDefaultScoring=True)
            else:
                pose_list = runner.write_pose_list(overwrite=True)
                if pose_list:
                    runner.run_rescore(pose_list, overwrite=True)

        if use_multi_boxes:
            print(f"Completed {eng} for job '{name}' (box {box_id}). Outputs in: {box_files_dir}")
        else:
            print(f"Completed {eng} for job '{name}'. Outputs in: {box_files_dir}")
    # Optional DB store
    if args.store_db:
        try:
            stored, stored_name, _ = cli_workflow._store_pipeline_results_in_db(
                job_name = name,
                receptor = receptor,
                ligand = ligand,
                rescoring = {},
                box_label = None,
            )
            if stored:
                print(f"Stored docking data in database row '{stored_name}'.")
            else:
                print("Warning: failed to store docking data in DB (upsert returned False).")
        except Exception as e:
            print(f"Warning: failed to store to DB: {e}")
    return overall_rc


def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    p_vs = sub.add_parser(
        "vs",
        description=(
            "Run docking with a single engine (Vina, Smina, Gnina, or PLANTS) and optionally rescore all poses.\n\n"
            "This command performs:\n"
            "  1. Receptor and ligand preparation\n"
            "  2. Docking with the selected engine\n"
            "  3. Pose splitting (for Vina/Smina/Gnina) into individual files\n"
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
        choices=["vina", "smina", "gnina", "plants"],
        default="vina",
        help="Docking engine to use. Options: 'vina' (AutoDock Vina), 'smina' (Vina with additional scoring functions), 'gnina' (CNN-enabled Vina-like docking), or 'plants' (PLANTS docking). Default: vina"
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
        "--all-boxes",
        action="store_true",
        help="Use all box*.pdb files found in the ligand directory (and the --box directory). Outputs are placed under <engine>Files/boxN/."
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
        help="Skip pose splitting step (only applicable for Vina/Smina/Gnina). By default, poses are split into individual files. Use this to keep all poses in a single file."
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
        help="Store run data in the database (Receptors, Ligands, Complexes) including available descriptors and supported program scores. Requires database to be configured and accessible, and optional DB deps installed (`pip install \"ocdocker[db]\"`)."
    )
    p_vs.set_defaults(func=cmd_vs)

