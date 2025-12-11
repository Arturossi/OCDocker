
from __future__ import annotations
import json
import argparse
from typing import Optional
from .Runner import run_shap_analysis
from .Studies import StudyHandles


def build_argparser() -> argparse.ArgumentParser:
    '''Build command-line argument parser for SHAP analysis.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all SHAP analysis command-line arguments.
    '''
    
    p = argparse.ArgumentParser(description="Run SHAP analysis for OCScore NeuralNet.")
    p.add_argument("--storage", required=True)
    p.add_argument("--ao_study", required=True)
    p.add_argument("--nn_study", required=True)
    p.add_argument("--seed_study", required=True)
    p.add_argument("--mask_study", required=True)
    p.add_argument("--df_path", required=True)
    p.add_argument("--base_models", required=True)
    p.add_argument("--study_number", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--explainer", default="deep", choices=["deep","kernel"])
    p.add_argument("--background_size", type=int)
    p.add_argument("--eval_size", type=int)
    p.add_argument("--stratify_by", nargs="*")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_csv", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    '''Main entry point for SHAP analysis command-line interface.
    
    Parameters
    ----------
    argv : Optional[list[str]], optional
        Command-line arguments to parse. If None, uses sys.argv. Default is None.
    '''
    
    args = build_argparser().parse_args(argv)
    studies = StudyHandles(
        ao_study_name=args.ao_study,
        nn_study_name=args.nn_study,
        seed_study_name=args.seed_study,
        mask_study_name=args.mask_study,
        storage=args.storage,
    )
    out = run_shap_analysis(
        studies=studies,
        df_path=args.df_path,
        base_models_folder=args.base_models,
        study_number=args.study_number,
        out_dir=args.out_dir,
        background_size=args.background_size,
        eval_size=args.eval_size,
        explainer=args.explainer,
        stratify_by=args.stratify_by,
        seed=args.seed,
        save_csv=not args.no_csv,
    )
    print(json.dumps(out.__dict__, indent=2))

if __name__ == "__main__":
    raise SystemExit(main())
