#!/usr/bin/env python3

"""
OCDocker CLI
============

Interface de linha de comando integrada para o projeto OCDocker.

Comandos principais
- version: exibe a versão da biblioteca.
- init-config: cria rapidamente um arquivo `OCDocker.cfg` a partir do exemplo.
- vs: executa docking + (opcional) rescoring para um único par receptor/ligante/caixa
      usando Vina, Smina ou PLANTS.
- shap: repassa para o CLI de SHAP (OCScore) já existente.
- pipeline: fluxo completo multi‑motor: docking em múltiplos motores, clusterização por RMSD,
            seleção de pose representativa, rescoring focado e exportação do resultado.

Opções globais
- --conf, --multiprocess, --update-databases, --output-level, --overwrite:
  compatíveis com OCDocker.Initialise e usadas para bootstrap do ambiente.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def _preparse_global_args(argv: list[str]) -> argparse.Namespace:
    """Parse only global OCDocker.Initialise-compatible args.

    This allows importing OCDocker modules that eagerly parse args.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--version", action="store_true")
    p.add_argument("--multiprocess", action="store_true", default=True)
    p.add_argument("-u", "--update-databases", dest="update", action="store_true", default=False)
    p.add_argument("--conf", dest="config_file", type=str)
    p.add_argument("--output-level", dest="output_level", type=int, default=1)
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=False)
    # ignore unknowns here
    ns, _unknown = p.parse_known_args(argv)
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

    sub = parser.add_subparsers(dest="command", required=True)

    # init-config
    p_init = sub.add_parser("init-config", help="Interactive creation of OCDocker.cfg")
    p_init.set_defaults(func=cmd_init_config)

    # version
    p_ver = sub.add_parser("version", help="Print OCDocker version")
    p_ver.set_defaults(func=cmd_version)

    # vs (virtual screening para uma única entrada)
    p_vs = sub.add_parser("vs", help="Executa docking + rescoring para um receptor/ligante/caixa")
    p_vs.add_argument("--engine", choices=["vina", "smina", "plants"], default="vina", help="Motor de docking")
    p_vs.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDB)")
    p_vs.add_argument("--ligand", required=True, help="Path to ligand file (smi/sdf/mol2/pdbqt)")
    p_vs.add_argument("--box", required=True, help="Path to box file (PDB with REMARK center/size)")
    p_vs.add_argument("--name", help="Job name (defaults to ligand stem)")
    p_vs.add_argument("--outdir", default="./ocdocker_out", help="Output directory")
    p_vs.add_argument("--skip-rescore", action="store_true", help="Skip rescoring phase")
    p_vs.add_argument("--skip-split", action="store_true", help="Skip pose splitting (quando aplicável)")
    p_vs.set_defaults(func=cmd_vs)

    # shap passthrough (reuses existing module)
    p_shap = sub.add_parser("shap", help="Run SHAP analysis (OCScore)")
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

    # pipeline (multi‑motor + clusterização + rescoring)
    p_pipe = sub.add_parser("pipeline", help="Executa docking (vina/smina/plants), clusteriza por RMSD, escolhe a pose representativa e aplica rescoring")
    p_pipe.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDB)")
    p_pipe.add_argument("--ligand", required=True, help="Path to ligand file (smi/sdf/mol2/pdbqt)")
    p_pipe.add_argument("--box", required=True, help="Path to box file (PDB with REMARK center/size)")
    p_pipe.add_argument("--engines", default="vina,smina,plants", help="Lista separada por vírgulas: vina,smina,plants")
    p_pipe.add_argument("--name", help="Job name (defaults to ligand stem)")
    p_pipe.add_argument("--outdir", default="./ocdocker_out", help="Output directory")
    p_pipe.add_argument("--cluster-min", type=float, default=10.0, help="Threshold mínimo para clusterização")
    p_pipe.add_argument("--cluster-max", type=float, default=20.0, help="Threshold máximo para clusterização")
    p_pipe.add_argument("--cluster-step", type=float, default=0.1, help="Passo de busca do threshold")
    p_pipe.add_argument("--store-db", action="store_true", help="Armazena metadados no banco (Complexes)")
    p_pipe.set_defaults(func=cmd_pipeline)

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
    # Import to access version string
    _bootstrap_ocdocker_env(_preparse_global_args(sys.argv[1:]))
    from OCDocker.Initialise import ocVersion  # type: ignore
    print(ocVersion)
    return 0

def cmd_vs(args: argparse.Namespace) -> int:
    """Executa um docking simples com o motor escolhido.

    Fluxo: prepara receptor/ligante, executa docking, divide poses (quando aplicável)
    e, se solicitado, aplica rescoring.
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
    # Engine-specific file namespace
    files_dir = outdir / f"{eng}Files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Build IO paths
    if eng in ("vina", "smina"):
        conf_path = files_dir / ("conf_vina.txt" if eng == "vina" else "conf_smina.txt")
        prep_rec = outdir / "prepared_receptor.pdbqt"
        prep_lig = outdir / "prepared_ligand.pdbqt"
        log_path = files_dir / f"{name}.log"
        out_pose = files_dir / f"{name}.pdbqt"
    else:  # plants
        conf_path = files_dir / "conf_plants.txt"
        prep_rec = outdir / "prepared_receptor.mol2"
        prep_lig = outdir / "prepared_ligand.mol2"
        log_path = files_dir / f"{name}.log"
        out_pose = files_dir  # diretório de saída do PLANTS

    # Create domain objects
    receptor = ocr.Receptor(str(args.receptor), name=f"{name}_receptor")
    ligand = ocl.Ligand(str(args.ligand), name=f"{name}_ligand")
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
    rc = runner.run_prepare_receptor()
    if isinstance(rc, tuple):
        rc = rc[0]
    if rc != 0:
        return int(rc)

    rc = runner.run_prepare_ligand()
    if isinstance(rc, tuple):
        rc = rc[0]
    if rc != 0:
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
    """Garante uma lista de poses no formato MOL2, convertendo quando necessário.

    Retorna a lista de caminhos .mol2 e um mapeamento mol2->original.
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
    """Fluxo completo multi‑motor + clusterização + rescoring e export.

    1) Executa docking nos motores selecionados.
    2) Converte poses para MOL2, clusteriza por RMSD e pega o medoide do maior cluster.
    3) Aplica rescoring apenas para a pose representativa.
    4) Salva representative.mol2 e summary.json (com resultados de rescoring).
    5) (Opcional) Armazena metadados mínimos no banco.
    """

    # Bootstrap env
    globals_ns = _preparse_global_args(sys.argv[1:])
    _bootstrap_ocdocker_env(globals_ns)

    # Imports de domínio
    import OCDocker.Ligand as ocl  # type: ignore
    import OCDocker.Receptor as ocr  # type: ignore
    import OCDocker.Docking.Vina as ocvina  # type: ignore
    import OCDocker.Docking.Smina as ocsmina  # type: ignore
    import OCDocker.Docking.PLANTS as ocplants  # type: ignore
    import OCDocker.Toolbox.MoleculeProcessing as ocmolproc  # type: ignore
    import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsd  # type: ignore
    import pandas as pd  # type: ignore
    import json

    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.ligand).stem

    receptor = ocr.Receptor(str(args.receptor), name=f"{name}_receptor")
    ligand = ocl.Ligand(str(args.ligand), name=f"{name}_ligand")

    engines = [e.strip().lower() for e in args.engines.split(',') if e.strip()]
    engines = [e for e in engines if e in ("vina", "smina", "plants")]
    if not engines:
        print("Nenhum motor válido informado. Use --engines vina,smina,plants")
        return 1

    all_poses: List[str] = []
    ctx: Dict[str, Dict[str, str]] = {}

    for eng in engines:
        e_dir = outdir / f"{eng}Files"; e_dir.mkdir(parents=True, exist_ok=True)
        if eng == "vina":
            conf = e_dir / "conf_vina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
            log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
            r = ocvina.Vina(str(conf), str(args.box), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"VINA {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            _ = r.split_poses(str(e_dir))
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
        elif eng == "smina":
            conf = e_dir / "conf_smina.txt"; prep_r = outdir / "prepared_receptor.pdbqt"; prep_l = outdir / "prepared_ligand.pdbqt"
            log = e_dir / f"{name}.log"; outp = e_dir / f"{name}.pdbqt"
            r = ocsmina.Smina(str(conf), str(args.box), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"SMINA {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            _ = r.split_poses(str(e_dir))
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir)}
        else:
            conf = e_dir / "conf_plants.txt"; prep_r = outdir / "prepared_receptor.mol2"; prep_l = outdir / "prepared_ligand.mol2"
            log = e_dir / f"{name}.log"; outp = e_dir
            r = ocplants.PLANTS(str(conf), str(args.box), receptor, str(prep_r), ligand, str(prep_l), str(log), str(outp), name=f"PLANTS {name}", overwriteConfig=True)
            for fn in (r.run_prepare_receptor, r.run_prepare_ligand, r.run_docking):
                rc = fn(); rc = rc[0] if isinstance(rc, tuple) else rc
                if rc != 0: return int(rc)
            all_poses.extend(r.get_docked_poses())
            ctx[eng] = {"conf": str(conf), "dir": str(e_dir), "prep_rec": str(prep_r)}

    if not all_poses:
        print("Nenhuma pose foi gerada.")
        return 2

    # Converte para MOL2 e clusteriza por RMSD
    mol2_dir = outdir / "poses_mol2"
    mol2_list, mol2_map = _ensure_mol2_poses(all_poses, mol2_dir)
    rmsd = ocmolproc.get_rmsd_matrix(mol2_list)
    df = pd.DataFrame(rmsd).loc[mol2_list, mol2_list]
    clusters = ocrmsd.cluster_rmsd(df, min_distance_threshold=args.cluster_min, max_distance_threshold=args.cluster_max, threshold_step=args.cluster_step)
    if isinstance(clusters, int) or getattr(clusters, "size", 0) == 0:
        rep_mol2 = mol2_list[0]
    else:
        meds = ocrmsd.get_medoids(df, clusters, onlyBiggest=True)
        rep_mol2 = meds[0] if meds else mol2_list[0]

    rep_path = outdir / "representative.mol2"
    # Copiar para preservar fontes
    import shutil
    shutil.copyfile(rep_mol2, rep_path)

    # Rescoring (apenas da representativa)
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
        # Extrai centro/raio do box
        center, radius = get_binding_site(str(args.box))  # type: ignore
        for sf in ("chemplp", "plp", "plp95"):
            conf_sf = outdir / f"conf_plants_rescore_{sf}.txt"
            write_rescoring_config_file(str(conf_sf), ctx["plants"]["prep_rec"], str(pose_list), ctx["plants"]["dir"], center[0], center[1], center[2], radius, scoringFunction=sf)
            p_rescore(str(conf_sf), str(pose_list), ctx["plants"]["dir"], ctx["plants"]["prep_rec"], sf, center[0], center[1], center[2], radius, overwrite=True)

    # Grava sumário
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
            print(f"Aviso: falha ao armazenar no banco: {e}")

    print(f"Pipeline concluído. Pose representativa: {rep_path}")
    return 0

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
