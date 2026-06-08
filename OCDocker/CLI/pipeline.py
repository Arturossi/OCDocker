#!/usr/bin/env python3
"""Multi-engine pipeline CLI command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.CLI.common import (
    _bootstrap_ocdocker_env,
    _db_dependencies_available,
    _preparse_global_args,
    _print_optional_dependency_hint,
    _require_file,
)
from OCDocker.CLI import workflow as cli_workflow

_box_sort_key = cli_workflow._box_sort_key
_ensure_mol2_poses = cli_workflow._ensure_mol2_poses
_list_boxes = cli_workflow._list_boxes
_select_pipeline_representative_medoid = cli_workflow._select_pipeline_representative_medoid
_wait_for_rescore_logs_ready = cli_workflow._wait_for_rescore_logs_ready
_collect_log_file_signatures = cli_workflow._collect_log_file_signatures
_collect_numeric_descriptors = cli_workflow._collect_numeric_descriptors
_store_pipeline_results_in_db = cli_workflow._store_pipeline_results_in_db
_flatten_rescoring_to_complex_payload = cli_workflow._flatten_rescoring_to_complex_payload
_map_score_to_complex_column = cli_workflow._map_score_to_complex_column
_is_integer_descriptor_name = cli_workflow._is_integer_descriptor_name
_to_numeric = cli_workflow._to_numeric
_rescoring_data_has_numeric_scores = cli_workflow._rescoring_data_has_numeric_scores

LOGGER = oclogging.get_logger("cli")

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

    if args.store_db:
        db_ok, db_exc = _db_dependencies_available()
        if not db_ok and db_exc is not None:
            return _print_optional_dependency_hint(
                feature="database storage (--store-db)",
                extra="db",
                exc=db_exc,
            )

    # Bootstrap env
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

    # Domain imports
    try:
        import OCDocker.Ligand as ocl
        import OCDocker.Receptor as ocr
        import OCDocker.Docking.Vina as ocvina
        import OCDocker.Docking.Smina as ocsmina
        import OCDocker.Docking.Gnina as ocgnina
        import OCDocker.Docking.PLANTS as ocplants
        import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
        import OCDocker.Toolbox.Printing as ocprint
        import OCDocker.Processing.Preprocessing.RMSDClustering as ocrmsd
        import pandas as pd
        import numpy as np
        import json
    except ModuleNotFoundError as exc:
        extra = _suggest_extra_for_missing_module(getattr(exc, "name", ""))
        return _print_optional_dependency_hint(
            feature="pipeline docking workflow",
            extra=extra,
            exc=exc,
        )

    base_outdir = Path(args.outdir).resolve()
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
    engines = [e for e in engines if e in ("vina", "smina", "gnina", "plants")]
    if not engines:
        print("No valid engine provided. Use --engines vina,smina,gnina,plants")
        return 1

    # Get rescoring engines (default to same as docking engines if not specified)
    rescoring_engines = engines
    if args.rescoring_engines:
        rescoring_engines = [e.strip().lower() for e in args.rescoring_engines.split(",") if e.strip()]
        # Validate rescoring engines
        valid_rescoring = {"vina", "smina", "gnina", "plants", "oddt"}
        invalid_rescoring = [e for e in rescoring_engines if e not in valid_rescoring]
        if invalid_rescoring:
            print(f"Error: invalid rescoring engines: {', '.join(invalid_rescoring)}. Valid options: vina, smina, gnina, plants, oddt")
            return 2
        # Filter to only valid engines
        rescoring_engines = [e for e in rescoring_engines if e in valid_rescoring]
        if not rescoring_engines:
            print("Error: no valid rescoring engines specified. Valid options: vina, smina, gnina, plants, oddt")
            return 2

    # Validate required binaries are available
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

    missing = []
    for e in engines:
        if e == "vina" and not _exists_exe(_vina_bin):
            missing.append("vina")
        elif e == "smina" and not _exists_exe(_smina_bin):
            missing.append("smina")
        elif e == "gnina" and not _exists_exe(_gnina_bin):
            missing.append("gnina")
        elif e == "plants" and not _exists_exe(_plants_bin):
            missing.append("plants")
    if missing:
        print(f"Error: missing engine binaries: {', '.join(missing)}. Check paths in OCDocker.cfg or PATH.")
        return 2

    def _run_pipeline_for_box(box_path: Path, outdir: Path, box_label: Optional[str]) -> int:
        outdir.mkdir(parents=True, exist_ok=True)
        all_poses: List[str] = []
        pose_engine_map: Dict[str, str] = {}  # Map pose path to engine name
        ctx: Dict[str, Dict[str, str]] = {}
        engine_errors: Dict[str, str] = {}
        import os as _os

        for eng in engines:
            r: Any
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
                elif eng == "gnina":
                    conf = e_dir / "conf_gnina.conf"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
                    log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
                    r = ocgnina.Gnina(str(conf), str(box_path), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"GNINA {name}", overwrite_config=True)
                    # Only prepare receptor/ligand if they don't exist
                    if not (_os.path.isfile(str(prep_r)) and _os.path.getsize(str(prep_r)) > 0):
                        rc = r.run_prepare_receptor(); rc = rc[0] if isinstance(rc, tuple) else rc
                        if rc != 0:
                            engine_errors[eng] = f"Receptor preparation failed with code {rc}"
                            ocprint.print_warning(f"Gnina receptor preparation failed. Continuing with other engines...")
                            continue
                    if not (_os.path.isfile(str(prep_l)) and _os.path.getsize(str(prep_l)) > 0):
                        rc = r.run_prepare_ligand(); rc = rc[0] if isinstance(rc, tuple) else rc
                        if rc != 0:
                            engine_errors[eng] = f"Ligand preparation failed with code {rc}"
                            ocprint.print_warning(f"Gnina ligand preparation failed. Continuing with other engines...")
                            continue
                    rc = r.run_docking(); rc = rc[0] if isinstance(rc, tuple) else rc
                    if rc != 0:
                        engine_errors[eng] = f"Docking failed with code {rc}"
                        ocprint.print_warning(f"Gnina docking failed. Continuing with other engines...")
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
        mol2_engine_map = ocrmsd.build_pose_engine_map(mol2_list, pose_engine_map, mol2_map)
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
            pose_engine_map=mol2_engine_map,
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
                rep_mol2, representative_info = _select_pipeline_representative_medoid(df, clusters, meds)
                clustering_info["representative_selection"] = "medoid_of_largest_cluster"
                if representative_info.get("tie_breaker") == "lowest_intra_cluster_medoid_distance":
                    clustering_info["representative_tie_breaker"] = representative_info["tie_breaker"]
                    clustering_info["tied_largest_cluster_ids"] = representative_info["largest_cluster_ids"]
                    clustering_info["representative_intra_cluster_distance_sum"] = representative_info["selected_medoid_sum_distance"]
                    clustering_info["representative_intra_cluster_distance_mean"] = representative_info["selected_medoid_mean_distance"]
                clustering_info["medoids"] = [str(m) for m in meds]
                clustering_info["representative_pose"] = str(rep_mol2)
                # Find which cluster the representative belongs to
                rep_idx = mol2_list.index(rep_mol2)
                rep_cluster_id = int(representative_info.get("selected_cluster_id", clusters[rep_idx]))
                clustering_info["representative_cluster_id"] = rep_cluster_id
                clustering_info["representative_cluster_size"] = cluster_sizes.get(rep_cluster_id, 0)

        # Get the original pose path for the representative
        rep_original = mol2_map.get(rep_mol2, rep_mol2)
        rep_engine = pose_engine_map.get(rep_original, None)

        # Convert representative to appropriate format for each engine's rescoring
        # Vina/Smina need PDBQT, PLANTS needs MOL2
        rep_pdbqt: Optional[Union[str, Path]] = None
        rep_mol2_final: Optional[Union[str, Path]] = None

        import OCDocker.Toolbox.Conversion as occonversion
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
            from OCDocker.Docking.Vina import run_rescore as v_rescore, get_rescore_log_paths as v_logs, read_rescore_logs as v_read
            if rep_pdbqt and Path(rep_pdbqt).exists():
                # Get scoring functions from config
                vina_sfs = config.vina.scoring_functions if config.vina.scoring_functions else ["vina"]
                for sf in vina_sfs:
                    try:
                        v_rescore(ctx["vina"]["conf"], str(rep_pdbqt), ctx["vina"]["dir"], sf, splitLigand=False, overwrite=True)
                    except Exception as e:
                        ocprint.print_warning(f"Vina rescoring with {sf} failed: {e}. Continuing with other scoring functions...")
                try:
                    log_paths, data, logs_ready = _wait_for_rescore_logs_ready(
                        v_logs,
                        v_read,
                        ctx["vina"]["dir"],
                    )
                    if not log_paths:
                        ocprint.print_warning(f"No Vina rescoring log files found in {ctx['vina']['dir']}. Check if rescoring completed successfully.")
                        # Debug: list files in directory
                        if Path(ctx["vina"]["dir"]).exists():
                            files = list(Path(ctx["vina"]["dir"]).glob("*"))
                            ocprint.print_warning(f"Files in Vina directory: {[f.name for f in files]}")
                    else:
                        if not logs_ready:
                            ocprint.print_warning("Vina rescoring logs were not parse-ready within 2.0 seconds. Continuing with best available data.")
                        ocprint.printv(f"Found Vina rescoring log files: {log_paths}")
                        if not data:
                            data = v_read(log_paths, onlyBest=True)
                        if not data:
                            ocprint.print_warning(f"Vina rescoring log files found but no data extracted. Log paths: {log_paths}")
                        else:
                            vina_vals: Dict[str, float] = {}
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
                                        vina_vals[clean_key] = float(v)
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
                                            vina_vals[clean_key] = numeric_val
                                except (ValueError, TypeError, KeyError) as e:
                                    ocprint.print_warning(f"Failed to parse Vina rescoring value for {k}: {e}. Value type: {type(v)}, value: {v}")
                            if vina_vals:
                                rescoring["vina"] = vina_vals
                            else:
                                ocprint.print_warning(f"Vina rescoring data found but no valid values extracted. Data structure: {data}")
                except Exception as e:
                    ocprint.print_warning(f"Failed to read Vina rescoring results: {e}")
                    import traceback
                    ocprint.print_warning(f"Traceback: {traceback.format_exc()}")
        # SMINA
        if "smina" in rescoring_engines:
            from OCDocker.Docking.Smina import run_rescore as s_rescore, get_rescore_log_paths as s_logs, read_rescore_logs as s_read
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
                        import OCDocker.Docking.Smina as ocsmina
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
                        log_paths, data, logs_ready = _wait_for_rescore_logs_ready(
                            s_logs,
                            s_read,
                            ctx["smina"]["dir"],
                        )
                        if not log_paths:
                            ocprint.print_warning(f"No Smina rescoring log files found in {ctx['smina']['dir']}")
                            # Debug: list files in directory
                            if Path(ctx["smina"]["dir"]).exists():
                                files = list(Path(ctx["smina"]["dir"]).glob("*"))
                                ocprint.print_warning(f"Files in Smina directory: {[f.name for f in files]}")
                        else:
                            if not logs_ready:
                                ocprint.print_warning("Smina rescoring logs were not parse-ready within 2.0 seconds. Continuing with best available data.")
                            ocprint.printv(f"Found Smina rescoring log files: {log_paths}")
                            if not data:
                                data = s_read(log_paths, onlyBest=True)
                            smina_vals: Dict[str, float] = {}
                            # Data structure: Dict[str, float] (read_rescoring_log returns float, not list)
                            # Key format: "rescoring_{scoring_function}_{pose_number}" or "smina_{scoring_function}_rescoring"
                            for k, v in data.items():
                                try:
                                    # v is a float (from read_rescoring_log)
                                    if isinstance(v, (int, float)):
                                        # Normalize key: extract scoring function and create clean key
                                        # Keys can be: "smina_vinardo_rescoring", "rescoring_vina_1", "rescoring_dkoes_scoring_1", etc.
                                        smina_sf_name: Optional[str] = None
                                        if k.startswith("smina_") and k.endswith("_rescoring"):
                                            # Format: "smina_{scoring_function}_rescoring"
                                            smina_sf_name = k.replace("smina_", "").replace("_rescoring", "")
                                            clean_key = f"smina_{smina_sf_name}"
                                        elif k.startswith("rescoring_"):
                                            # Format: "rescoring_{scoring_function}_{pose_number}"
                                            parts = k.replace("rescoring_", "").split("_")
                                            if len(parts) >= 1:
                                                # Handle multi-part scoring function names like "dkoes_scoring"
                                                # Try to match against known scoring functions
                                                smina_sf_name = None
                                                for known_sf in smina_sfs:
                                                    # Check if the key starts with this scoring function
                                                    if "_".join(parts[:len(known_sf.split("_"))]) == known_sf:
                                                        smina_sf_name = known_sf
                                                        break
                                                if not smina_sf_name and parts:
                                                    # Fallback: use first part
                                                    smina_sf_name = parts[0]
                                                clean_key = f"smina_{smina_sf_name}" if smina_sf_name else k
                                            else:
                                                clean_key = k
                                        else:
                                            clean_key = k
                                        smina_vals[clean_key] = float(v)
                                    elif isinstance(v, list) and len(v) > 0:
                                        # Handle list case (shouldn't happen but just in case)
                                        smina_vals[k] = float(v[0] if not isinstance(v[0], (list, tuple)) else v[0][0])
                                except (ValueError, TypeError, KeyError) as e:
                                    ocprint.print_warning(f"Failed to parse Smina rescoring value for {k}: {e}")
                            if smina_vals:
                                rescoring["smina"] = smina_vals
                            else:
                                ocprint.print_warning(f"Smina rescoring data found but no valid values extracted. Data structure: {data}")
                    except Exception as e:
                        ocprint.print_warning(f"Failed to read Smina rescoring results: {e}")
        # GNINA
        if "gnina" in rescoring_engines:
            from OCDocker.Docking.Gnina import run_rescore as g_rescore, get_rescore_log_paths as g_logs, read_rescore_logs as g_read
            if rep_pdbqt and Path(rep_pdbqt).exists():
                # If gnina wasn't docked, we can still use the shared prepared artifacts
                if "gnina" not in ctx:
                    gnina_dir = outdir / "gninaFiles"
                    gnina_dir.mkdir(parents=True, exist_ok=True)
                    gnina_conf = gnina_dir / "conf_gnina.conf"
                    prep_r = outdir / "prepared_receptor.pdbqt"
                    prep_l = outdir / "prepared_ligand.pdbqt"
                    gnina_obj = ocgnina.Gnina(
                        str(gnina_conf),
                        str(box_path),
                        receptor,
                        str(prep_r),
                        ligand,
                        str(prep_l),
                        str(gnina_dir / f"{name}.log"),
                        str(gnina_dir / f"{name}.pdbqt"),
                        name=f"GNINA {name}",
                        overwrite_config=True,
                    )
                    _ = gnina_obj
                    ctx["gnina"] = {"conf": str(gnina_conf), "dir": str(gnina_dir)}

                gnina_default_scoring = str(getattr(config.gnina, "scoring", "default") or "default").strip() or "default"
                gnina_scoring_functions = getattr(config.gnina, "scoring_functions", None)
                if not isinstance(gnina_scoring_functions, list) or not gnina_scoring_functions:
                    gnina_scoring_functions = [gnina_default_scoring]
                gnina_scoring_functions = [str(sf).strip() for sf in gnina_scoring_functions if str(sf).strip()]

                gnina_cnn_models = getattr(config.gnina, "cnn_models", None)
                if not isinstance(gnina_cnn_models, list) or not gnina_cnn_models:
                    gnina_cnn_models = [str(getattr(config.gnina, "cnn", "default") or "default")]
                gnina_cnn_models = [str(model).strip() for model in gnina_cnn_models if str(model).strip()]

                for sf_txt in gnina_scoring_functions:
                    if not sf_txt:
                        continue
                    try:
                        g_rescore(
                            ctx["gnina"]["conf"],
                            str(rep_pdbqt),
                            ctx["gnina"]["dir"],
                            sf_txt,
                            splitLigand = False,
                            overwrite = True,
                            disable_cnn = True,
                        )
                    except Exception as e:
                        ocprint.print_warning(f"Gnina rescoring with empirical scoring function '{sf_txt}' failed: {e}. Continuing with other rescoring functions...")

                for cnn_model in gnina_cnn_models:
                    try:
                        g_rescore(
                            ctx["gnina"]["conf"],
                            str(rep_pdbqt),
                            ctx["gnina"]["dir"],
                            gnina_default_scoring,
                            splitLigand = False,
                            overwrite = True,
                            cnn_model = cnn_model,
                            disable_cnn = False,
                        )
                    except Exception as e:
                        ocprint.print_warning(f"Gnina rescoring with CNN model '{cnn_model}' failed: {e}. Continuing with other CNN models...")

                try:
                    log_paths, data, logs_ready = _wait_for_rescore_logs_ready(
                        g_logs,
                        g_read,
                        ctx["gnina"]["dir"],
                    )
                    if not log_paths:
                        ocprint.print_warning(f"No Gnina rescoring log files found in {ctx['gnina']['dir']}")
                        if Path(ctx["gnina"]["dir"]).exists():
                            files = list(Path(ctx["gnina"]["dir"]).glob("*"))
                            ocprint.print_warning(f"Files in Gnina directory: {[f.name for f in files]}")
                    else:
                        if not logs_ready:
                            ocprint.print_warning("Gnina rescoring logs were not parse-ready within 2.0 seconds. Continuing with best available data.")
                        ocprint.printv(f"Found Gnina rescoring log files: {log_paths}")
                        if not data:
                            data = g_read(log_paths, onlyBest=True)
                        gnina_vals: Dict[str, float] = {}
                        for k, v in data.items():
                            try:
                                if isinstance(v, (int, float)):
                                    gnina_sf_name: Optional[str] = None
                                    gnina_cnn_name: Optional[str] = None
                                    if k.startswith("gnina_cnn_") and k.endswith("_rescoring"):
                                        gnina_cnn_name = k.replace("gnina_cnn_", "").replace("_rescoring", "")
                                        clean_key = f"gnina_cnn_{gnina_cnn_name}"
                                    elif k.startswith("gnina_") and k.endswith("_rescoring"):
                                        gnina_sf_name = k.replace("gnina_", "").replace("_rescoring", "")
                                        clean_key = f"gnina_{gnina_sf_name}"
                                    elif k.startswith("rescoring_cnn_"):
                                        cnn_suffix = k.replace("rescoring_cnn_", "", 1)
                                        for known_cnn in sorted(gnina_cnn_models, key=len, reverse=True):
                                            known_prefix = f"{known_cnn}_"
                                            if cnn_suffix.startswith(known_prefix):
                                                gnina_cnn_name = known_cnn
                                                break
                                        if not gnina_cnn_name and "_" in cnn_suffix:
                                            gnina_cnn_name = cnn_suffix.rsplit("_", 1)[0]
                                        clean_key = f"gnina_cnn_{gnina_cnn_name}" if gnina_cnn_name else k
                                    elif k.startswith("rescoring_"):
                                        parts = k.replace("rescoring_", "").split("_")
                                        if len(parts) >= 1:
                                            gnina_sf_name = None
                                            for known_sf in gnina_scoring_functions:
                                                if "_".join(parts[:len(str(known_sf).split("_"))]) == str(known_sf):
                                                    gnina_sf_name = str(known_sf)
                                                    break
                                            if not gnina_sf_name and parts:
                                                gnina_sf_name = parts[0]
                                            clean_key = f"gnina_{gnina_sf_name}" if gnina_sf_name else k
                                        else:
                                            clean_key = k
                                    else:
                                        clean_key = k
                                    gnina_vals[clean_key] = float(v)
                                elif isinstance(v, list) and len(v) > 0:
                                    gnina_vals[k] = float(v[0] if not isinstance(v[0], (list, tuple)) else v[0][0])
                            except (ValueError, TypeError, KeyError) as e:
                                ocprint.print_warning(f"Failed to parse Gnina rescoring value for {k}: {e}")

                        if gnina_vals:
                            rescoring["gnina"] = gnina_vals
                        else:
                            ocprint.print_warning(f"Gnina rescoring data found but no valid values extracted. Data structure: {data}")
                except Exception as e:
                    ocprint.print_warning(f"Failed to read Gnina rescoring results: {e}")

        # PLANTS
        if "plants" in ctx and "plants" in rescoring_engines:
            from OCDocker.Docking.PLANTS import write_rescoring_config_file, run_rescore as p_rescore, get_binding_site
            pose_list = outdir / "pose_list_single.txt"
            # Use MOL2 format for PLANTS rescoring
            plants_rep = str(rep_mol2_final) if rep_mol2_final and Path(rep_mol2_final).exists() else str(rep_path)
            pose_list.write_text(plants_rep + "\n")
            # Extract center/radius from the box
            binding_site_result = get_binding_site(str(box_path))
            binding_site: Optional[tuple[tuple[float, float, float], float]] = None
            if isinstance(binding_site_result, int):
                ocprint.print_warning(f"Failed to read binding site from {box_path}. Skipping PLANTS rescoring.")
            else:
                binding_site = binding_site_result
            if binding_site is None:
                ocprint.print_warning(f"Skipping PLANTS rescoring for '{box_path}' due to missing binding site.")
            else:
                center, radius = binding_site
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
                    from OCDocker.Docking.PLANTS import read_log as plants_read_log
                    plants_rescoring_data: Dict[str, float] = {}
                    for sf in plants_sfs:
                        # Each scoring function has its own directory: run_{scoring_function}
                        ranking_file = Path(ctx["plants"]["dir"]) / f"run_{sf}" / "bestranking.csv"
                        if ranking_file.exists():
                            try:
                                log_data = plants_read_log(str(ranking_file), onlyBest=True)
                                if log_data:
                                    # PLANTS returns Dict[int, Dict[str, float]] where first int is pose number
                                    for _, scores in log_data.items():
                                        # scores is Dict[str, float]
                                        for score_type_code, score_value in scores.items():
                                            key = f"plants_{sf}"
                                            if key not in plants_rescoring_data:
                                                try:
                                                    if isinstance(score_value, list):
                                                        if len(score_value) != 1:
                                                            continue
                                                        plants_rescoring_data[key] = float(score_value[0])
                                                    else:
                                                        plants_rescoring_data[key] = float(score_value)
                                                except (TypeError, ValueError):
                                                    continue
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
                from OCDocker.Rescoring.ODDT import run_oddt, df_to_dict
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
                                if isinstance(oddt_dict, int):
                                    ocprint.print_warning(f"ODDT results conversion returned error code: {oddt_dict}")
                                elif oddt_dict:
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
                                    ocprint.print_warning(
                                        f"ODDT rescoring completed but no valid scores extracted. Dict keys: "
                                        f"{list(oddt_dict.keys()) if isinstance(oddt_dict, dict) else 'None'}"
                                    )
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
                stored, stored_name, ignored_keys = _store_pipeline_results_in_db(
                    job_name = name,
                    receptor = receptor,
                    ligand = ligand,
                    rescoring = rescoring,
                    box_label = box_label,
                )
                if stored:
                    print(f"Stored pipeline data in database row '{stored_name}'.")
                    if ignored_keys:
                        print(
                            "Warning: some score keys could not be mapped to Complexes columns and were skipped: "
                            + ", ".join(ignored_keys)
                        )
                else:
                    print("Warning: failed to store pipeline data in DB (upsert returned False).")
            except Exception as e:
                print(f"Warning: failed to store to DB: {e}")

        print(f"Pipeline finished. Representative pose: {rep_path}")
        return 0


    boxes = _list_boxes(ligand_path.parent, box_path, args.all_boxes)
    if args.all_boxes and not boxes:
        print("Warning: no box*.pdb files found. Skipping pipeline.")
        return 2

    if args.all_boxes:
        overall_rc = 0
        use_multi_boxes = len(boxes) > 1
        for box in boxes:
            box_id = box.stem
            box_outdir = base_outdir / box_id if use_multi_boxes else base_outdir
            rc = _run_pipeline_for_box(box, box_outdir, box_id if use_multi_boxes else None)
            if rc != 0:
                overall_rc = rc
        return overall_rc

    return _run_pipeline_for_box(box_path, base_outdir, None)



def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    p_pipe = sub.add_parser(
        "pipeline",
        description=(
            "Run multi-engine docking with RMSD clustering and representative pose selection.\n\n"
            "This command performs a complete workflow:\n"
            "  1. Runs docking with multiple engines (Vina, Smina, Gnina, PLANTS, or any combination)\n"
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
        "--all-boxes",
        action="store_true",
        help="Use all box*.pdb files found in the ligand directory (and the --box directory). Results are written under <outdir>/boxN/."
    )
    p_pipe.add_argument(
        "--engines",
        default="vina,smina,plants",
        help="Comma-separated list of docking engines to use. Options: 'vina', 'smina', 'gnina', 'plants', or any combination (e.g., 'vina,smina' or 'vina,gnina,plants'). Default: vina,smina,plants"
    )
    p_pipe.add_argument(
        "--rescoring-engines",
        "--rescore-engines",  # Alias for convenience
        dest="rescoring_engines",
        default=None,
        help="Comma-separated list of engines to use for rescoring. Options: 'vina', 'smina', 'gnina', 'plants', 'oddt', or any combination. If not specified, uses the same engines as --engines. Can be different from docking engines (e.g., dock with 'vina,plants' but rescore with 'vina,smina,gnina,oddt')."
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
        help="Store run data in the database (Receptors, Ligands, Complexes) including available descriptors and supported program scores. Requires database to be configured and accessible, and optional DB deps installed (`pip install \"ocdocker[db]\"`)."
    )
    p_pipe.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for external docking tools. Overrides the OCDOCKER_TIMEOUT environment variable. If a tool exceeds this time, the process will be terminated."
    )
    p_pipe.set_defaults(func=cmd_pipeline)

