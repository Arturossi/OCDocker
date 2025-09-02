#!/usr/bin/env python3

"""
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
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def _preparse_global_args(argv: list[str]) -> argparse.Namespace:
    """Extract global flags from anywhere in argv.

    Works around argparse limitation when global options appear after the subcommand.
    """
    ns = argparse.Namespace(
        version=False,
        multiprocess=True,
        update=False,
        config_file=None,
        output_level=1,
        overwrite=False,
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
            except Exception:
                pass
            i += 2
            continue
        if tok == "--overwrite":
            ns.overwrite = True
            i += 1
            continue
        # skip token
        i += 1
    return ns

def _bootstrap_ocdocker_env(ns: argparse.Namespace) -> None:
    """Prepare process state so OCDocker.Initialise can import safely.

    - Set OCDOCKER_CONFIG env var if provided
    - Temporarily replace sys.argv with only known flags
    - Import OCDocker.Initialise (which initialises environment)
    - Restore sys.argv
    """
    # Prepare argv for Initialise
    init_argv = [sys.argv[0]]
    if ns.multiprocess:
        init_argv.append("--multiprocess")
    if ns.update:
        init_argv.append("--update-databases")
    if ns.config_file:
        os.environ["OCDOCKER_CONFIG"] = ns.config_file
        init_argv.extend(["--conf", ns.config_file])
    if ns.output_level is not None:
        init_argv.extend(["--output-level", str(ns.output_level)])
    if ns.overwrite:
        init_argv.append("--overwrite")

    prev_argv = list(sys.argv)
    try:
        sys.argv = init_argv
        importlib.import_module("OCDocker.Initialise")
    finally:
        sys.argv = prev_argv


def _require_file(p: str, label: str) -> Path:
    """Ensure a file path exists. Print a helpful message and exit if not.

    Also warns if the path seems to contain a Unicode ellipsis (…)
    which is often a placeholder, not a real path.
    """
    if "…" in p:
        print(f"Error: {label} contains an ellipsis character (…). Replace it with a real path.")
        raise SystemExit(2)
    path = Path(p).resolve()
    if not path.is_file():
        print(f"Error: {label} file not found: {p}")
        raise SystemExit(2)
    return path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocdocker",
        description="OCDocker CLI: docking, screening and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options (mirrors OCDocker.Initialise)
    parser.add_argument("--multiprocess", action="store_true", default=True, help="Enable multiprocessing for supported tasks")
    parser.add_argument("-u", "--update-databases", dest="update", action="store_true", default=False, help="Update databases on startup")
    parser.add_argument("--conf", dest="config_file", type=str, help="Path to OCDocker.cfg")
    parser.add_argument("--output-level", dest="output_level", type=int, default=1, help="Log level (0-5)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False, help="Overwrite outputs when applicable")

    # Parent parser to allow repeating global options after subcommand
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--multiprocess", action="store_true", default=True)
    parent.add_argument("-u", "--update-databases", dest="update", action="store_true", default=False)
    parent.add_argument("--conf", dest="config_file", type=str)
    parent.add_argument("--output-level", dest="output_level", type=int, default=1)
    parent.add_argument("--overwrite", dest="overwrite", action="store_true", default=False)

    sub = parser.add_subparsers(dest="command", required=True)

    # init-config
    p_init = sub.add_parser("init-config", help="Interactive creation of OCDocker.cfg", parents=[parent])
    p_init.set_defaults(func=cmd_init_config)

    # version
    p_ver = sub.add_parser("version", help="Print OCDocker version", parents=[parent])
    p_ver.set_defaults(func=cmd_version)

    # vs (single-entry virtual screening)
    p_vs = sub.add_parser("vs", help="Run docking + rescoring for a single receptor/ligand/box", parents=[parent])
    p_vs.add_argument("--engine", choices=["vina", "smina", "plants"], default="vina", help="Docking engine")
    p_vs.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDB)")
    p_vs.add_argument("--ligand", required=True, help="Path to ligand file (smi/sdf/mol2/pdbqt)")
    p_vs.add_argument("--box", required=True, help="Path to box file (PDB with REMARK center/size)")
    p_vs.add_argument("--name", help="Job name (defaults to ligand stem)")
    p_vs.add_argument("--outdir", default="./ocdocker_out", help="Output directory")
    p_vs.add_argument("--skip-rescore", action="store_true", help="Skip rescoring phase")
    p_vs.add_argument("--skip-split", action="store_true", help="Skip pose splitting (when applicable)")
    p_vs.set_defaults(func=cmd_vs)

    # shap passthrough (reuses existing module)
    p_shap = sub.add_parser("shap", help="Run SHAP analysis (OCScore)", parents=[parent])
    p_shap.add_argument("--storage", required=True)
    p_shap.add_argument("--ao_study", required=True)
    p_shap.add_argument("--nn_study", required=True)
    p_shap.add_argument("--seed_study", required=True)
    p_shap.add_argument("--mask_study", required=True)
    p_shap.add_argument("--df_path", required=True)
    p_shap.add_argument("--base_models", required=True)
    p_shap.add_argument("--study_number", type=int, required=True)
    p_shap.add_argument("--out_dir", required=True)
    p_shap.add_argument("--explainer", default="deep", choices=["deep", "kernel"])
    p_shap.add_argument("--background_size", type=int)
    p_shap.add_argument("--eval_size", type=int)
    p_shap.add_argument("--stratify_by", nargs="*")
    p_shap.add_argument("--seed", type=int, default=0)
    p_shap.add_argument("--no_csv", action="store_true")
    p_shap.set_defaults(func=cmd_shap)

    # pipeline (multi‑engine + clustering + rescoring)
    p_pipe = sub.add_parser("pipeline", help="Run docking (vina/smina/plants), cluster by RMSD, select representative pose, and apply rescoring", parents=[parent])
    p_pipe.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDB)")
    p_pipe.add_argument("--ligand", required=True, help="Path to ligand file (smi/sdf/mol2/pdbqt)")
    p_pipe.add_argument("--box", required=True, help="Path to box file (PDB with REMARK center/size)")
    p_pipe.add_argument("--engines", default="vina,smina,plants", help="Comma-separated list: vina,smina,plants")
    p_pipe.add_argument("--name", help="Job name (defaults to ligand stem)")
    p_pipe.add_argument("--outdir", default="./ocdocker_out", help="Output directory")
    p_pipe.add_argument("--cluster-min", type=float, default=10.0, help="Minimum threshold for clustering")
    p_pipe.add_argument("--cluster-max", type=float, default=20.0, help="Maximum threshold for clustering")
    p_pipe.add_argument("--cluster-step", type=float, default=0.1, help="Search step for threshold")
    p_pipe.add_argument("--store-db", action="store_true", help="Store minimal metadata in DB (Complexes)")
    p_pipe.set_defaults(func=cmd_pipeline)

    # console (interactive mode)
    p_console = sub.add_parser("console", help="Open interactive OCDocker console", parents=[parent])
    p_console.set_defaults(func=cmd_console)

    return parser

def cmd_init_config(args: argparse.Namespace) -> int:
    """Create a base OCDocker.cfg from the example file.

    This avoids importing Initialise (which expects a ready config).
    """
    example = Path("OCDocker.cfg.example")
    if not example.exists():
        print("OCDocker.cfg.example not found in current directory.")
        return 1

    target = Path(args.config_file or "OCDocker.cfg")
    if target.exists():
        print(f"Config already exists: {target}")
        return 0

    target.write_text(example.read_text())
    print(f"Config created at: {target}. Please review and adjust paths.")
    return 0

def cmd_version(args: argparse.Namespace) -> int:
    """Print package version without bootstrapping the full environment."""
    try:
        import OCDocker as _oc
        v = getattr(_oc, "__version__", None)
        if v:
            print(v)
            return 0
    except Exception:
        pass
    # Fallback to importlib.metadata (may work when installed)
    try:
        from importlib.metadata import version as _pkg_version
        print(_pkg_version("OCDocker"))
        return 0
    except Exception:
        pass
    # Last resort: try legacy variable if available (avoid heavy import)
    print("unknown")
    return 0

def cmd_vs(args: argparse.Namespace) -> int:
    """Run a simple docking with the selected engine.

    Flow: prepare receptor/ligand, run docking, split poses (when applicable),
    and optionally run rescoring.
    """

    # Bootstrap environment before importing engines
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

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
            str(prep_lig), str(log_path), str(out_pose), name=f"VINA {name}", overwriteConfig=True,
        )
    elif eng == "smina":
        dock = engine_mod.Smina
        runner = dock(
            str(conf_path), str(args.box), receptor, str(prep_rec), ligand,
            str(prep_lig), str(log_path), str(out_pose), name=f"SMINA {name}", overwriteConfig=True,
        )
    else:
        dock = engine_mod.PLANTS
        runner = dock(
            str(conf_path), str(args.box), receptor, str(prep_rec), ligand,
            str(prep_lig), str(log_path), str(out_pose), name=f"PLANTS {name}", overwriteConfig=True,
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
        except Exception:
            pass
        try:
            if _os.path.isfile(prep_lig_path):
                _os.remove(prep_lig_path)
        except Exception:
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
    return 0

def cmd_shap(args: argparse.Namespace) -> int:
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


def _ensure_mol2_poses(pose_paths: List[str], dest_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    """Ensure a list of poses in MOL2 format, converting when needed.

    Returns a list of .mol2 paths and a mapping mol2->original path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    mol2_paths: List[str] = []
    mapping: Dict[str, str] = {}

    import OCDocker.Toolbox.Conversion as occonversion  # type: ignore
    for p in pose_paths:
        src = Path(p)
        if src.suffix.lower() == ".mol2":
            mol2_paths.append(str(src))
            mapping[str(src)] = str(src)
            continue
        out = dest_dir / (src.stem + ".mol2")
        _ = occonversion.convertMols(str(src), str(out), overwrite=True)
        mol2_paths.append(str(out))
        mapping[str(out)] = str(src)
    return mol2_paths, mapping


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Full multi-engine flow with clustering, rescoring and export.

    1) Run docking on selected engines.
    2) Convert poses to MOL2, cluster by RMSD and pick the medoid of the largest cluster.
    3) Rescore only the representative pose.
    4) Save representative.mol2 and summary.json (rescoring results).
    5) (Optional) Store minimal metadata to DB.
    """

    # Bootstrap env
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

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
    import json

    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.ligand).stem

    # Validate input files
    receptor_path = _require_file(str(args.receptor), "--receptor")
    ligand_path = _require_file(str(args.ligand), "--ligand")
    box_path = _require_file(str(args.box), "--box")

    receptor = ocr.Receptor(str(receptor_path), name=f"{name}_receptor")
    ligand = ocl.Ligand(str(ligand_path), name=f"{name}_ligand")

    engines = [e.strip().lower() for e in args.engines.split(',') if e.strip()]
    engines = [e for e in engines if e in ("vina", "smina", "plants")]
    if not engines:
        print("No valid engine provided. Use --engines vina,smina,plants")
        return 1

    all_poses: List[str] = []
    ctx: Dict[str, Dict[str, str]] = {}

    for eng in engines:
        e_dir = outdir / f"{eng}Files"; e_dir.mkdir(parents=True, exist_ok=True)
        if eng == "vina":
            conf = e_dir / "conf_vina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
            log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
            r = ocvina.Vina(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"VINA {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            _ = r.split_poses(str(e_dir))
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
        elif eng == "smina":
            conf = e_dir / "conf_smina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
            log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
            r = ocsmina.Smina(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"SMINA {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            _ = r.split_poses(str(e_dir))
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
        else:
            conf = e_dir / "conf_plants.txt"; prep_r = outdir / "prepared_receptor.mol2"; prep_l = outdir / "prepared_ligand.mol2"
            log = e_dir / f"{name}.log"; outp = e_dir
            r = ocplants.PLANTS(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"PLANTS {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir), "prep_rec": str(prep_r)}

    if not all_poses:
        print("No poses were generated.")
        return 2

    # Convert to MOL2 and cluster by RMSD
    mol2_dir = outdir / "poses_mol2"
    mol2_list, mol2_map = _ensure_mol2_poses(all_poses, mol2_dir)
    rmsd = ocmolproc.get_rmsd_matrix(mol2_list)
    df = pd.DataFrame(rmsd).loc[mol2_list, mol2_list]
    clusters = ocrmsd.cluster_rmsd(
        df,
        min_distance_threshold=args.cluster_min,
        max_distance_threshold=args.cluster_max,
        threshold_step=args.cluster_step,
    )
    if isinstance(clusters, int) or getattr(clusters, "size", 0) == 0:
        ocprint.print_warning(
            "Clustering did not converge or returned no labels; using the first pose as representative."
        )
        rep_mol2 = mol2_list[0]
    else:
        meds = ocrmsd.get_medoids(df, clusters, onlyBiggest=True)
        if not meds:
            ocprint.print_warning(
                "No medoid found from clusters; using the first pose as representative."
            )
        rep_mol2 = meds[0] if meds else mol2_list[0]

    rep_path = outdir / "representative.mol2"
    # Copy to preserve source files
    import shutil
    shutil.copyfile(rep_mol2, rep_path)

    # Rescoring (representative only)
    rescoring: Dict[str, Dict[str, float]] = {}
    # VINA
    if "vina" in ctx:
        from OCDocker.Docking.Vina import run_rescore as v_rescore, get_rescore_log_paths as v_logs, read_rescore_logs as v_read  # type: ignore
        src = mol2_map.get(rep_mol2)
        if src and src.endswith('.pdbqt'):
            for sf in ("vina", "vinardo"):
                v_rescore(ctx["vina"]["conf"], src, ctx["vina"]["dir"], sf, splitLigand=False, overwrite=True)
            data = v_read(v_logs(ctx["vina"]["dir"]), onlyBest=True)
            vals: Dict[str, float] = {}
            for k, v in data.items():
                try:
                    for _, vv in v.items():
                        vals[k] = float(vv if not isinstance(vv, (list, tuple)) else vv[0])
                except Exception:
                    pass
            rescoring["vina"] = vals
    # SMINA
    if "smina" in ctx:
        from OCDocker.Docking.Smina import run_rescore as s_rescore, get_rescore_log_paths as s_logs, read_rescore_logs as s_read  # type: ignore
        src = mol2_map.get(rep_mol2)
        if src and src.endswith('.pdbqt'):
            for sf in ("vina", "vinardo", "dkoes_scoring", "old_scoring_dkoes", "fast_dkoes", "ad4_scoring"):
                s_rescore(ctx["smina"]["conf"], src, ctx["smina"]["dir"], sf, splitLigand=False, overwrite=True)
            data = s_read(s_logs(ctx["smina"]["dir"]), onlyBest=True)
            vals: Dict[str, float] = {}
            for k, v in data.items():
                try:
                    for _, vv in v.items():
                        vals[k] = float(vv if not isinstance(vv, (list, tuple)) else vv[0])
                except Exception:
                    pass
            rescoring["smina"] = vals
    # PLANTS
    if "plants" in ctx:
        from OCDocker.Docking.PLANTS import write_rescoring_config_file, run_rescore as p_rescore, get_binding_site  # type: ignore
        pose_list = outdir / "pose_list_single.txt"
        pose_list.write_text(str(rep_path) + "\n")
        # Extract center/radius from the box
        center, radius = get_binding_site(str(box_path))  # type: ignore
        for sf in ("chemplp", "plp", "plp95"):
            conf_sf = outdir / f"conf_plants_rescore_{sf}.txt"
            write_rescoring_config_file(str(conf_sf), ctx["plants"]["prep_rec"], str(pose_list), ctx["plants"]["dir"], center[0], center[1], center[2], radius, scoringFunction=sf)
            p_rescore(str(conf_sf), str(pose_list), ctx["plants"]["dir"], ctx["plants"]["prep_rec"], sf, center[0], center[1], center[2], radius, overwrite=True)

    # Write summary
    summ = {
        "job": name,
        "engines": engines,
        "representative_pose": str(rep_path),
        "rescoring": rescoring,
    }
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))

    if args.store_db:
        try:
            from OCDocker.DB.Models.Complexes import Complexes  # type: ignore
            Complexes.insert_or_update({"name": name})
        except Exception as e:
            print(f"Warning: failed to store to DB: {e}")

    print(f"Pipeline finished. Representative pose: {rep_path}")
    return 0

def cmd_console(args: argparse.Namespace) -> int:
    """Open an interactive console with OCDockerConsole namespace.

    Respects global flags by bootstrapping environment first.
    """
    # Bootstrap env to ensure Initialise is safe to import
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Import console module and open interactive session with its namespace
    try:
        import OCDockerConsole as occ  # type: ignore
    except Exception as e:
        print(f"Failed to import OCDockerConsole: {e}")
        return 1

    print("Launching OCDocker Console. Press Ctrl-D to exit.")
    try:
        import code
        # Expose console namespace without dunders
        local_ns = {k: v for k, v in vars(occ).items() if not k.startswith('__')}
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
            except Exception:
                pass
        except Exception:
            pass
        # Avoid printing the console banner twice: it's already printed on import
        code.interact(banner="", local=local_ns)
        # Save history (best-effort)
        try:
            if 'readline' in sys.modules:
                sys.modules['readline'].write_history_file(os.path.expanduser('~/.ocdocker_console_history'))
        except Exception:
            pass
    except Exception as e:
        print(f"Interactive console exited with error: {e}")
        return 1
    return 0

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
