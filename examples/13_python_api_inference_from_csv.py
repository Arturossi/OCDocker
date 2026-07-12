#!/usr/bin/env python3
"""
Example: OCScore Inference from CSV

This example demonstrates how to run OCScore model inference using feature data
loaded from a CSV file. The script:

1. Loads OCDocker configuration to enforce `reference_column_order`
2. Loads input CSV data and keeps the original table for output
3. Loads model artifacts (model, optional mask, optional scaler)
4. Runs `ocscoring.get_score(...)` for inference
5. Exports an output CSV preserving original rows/columns plus `OCSCORE`

Usage:
    # Recommended: pass config path explicitly
    python examples/13_python_api_inference_from_csv.py \
        --csv-path /path/to/features.csv \
        --model-name OCScore \
        --config-path /path/to/OCDocker.cfg \
        --output-csv /path/to/scored.csv

    # Alternative: use OCDOCKER_CONFIG environment variable
    export OCDOCKER_CONFIG=/path/to/OCDocker.cfg
    python examples/13_python_api_inference_from_csv.py \
        --csv-path /path/to/features.csv \
        --model-name OCScore
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Allow running the example directly from source checkout without installation.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import OCDocker.Config as occonfig
import OCDocker.OCScore.Scoring as ocscoring
import OCDocker.OCScore.Utils.IO as ocscoreio


def _resolve_model_path(model_name: str, models_dir: str, model_path: str | None) -> str:
    """Resolve the model path from an explicit path or common model-name patterns."""

    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    candidates = [
        os.path.join(models_dir, f"{model_name}.pt"),
        os.path.join(models_dir, f"{model_name}.pth"),
        os.path.join(models_dir, f"{model_name}.pkl"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find model for '{model_name}' in {models_dir}. "
        "Tried .pt, .pth, and .pkl."
    )


def _resolve_config_path(config_path: str | None) -> str | None:
    """Resolve config path from argument/env/common local locations."""

    if config_path:
        candidate = os.path.abspath(config_path)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate

    env_cfg = os.getenv("OCDOCKER_CONFIG", "").strip()
    if env_cfg:
        env_candidate = os.path.abspath(env_cfg)
        if os.path.isfile(env_candidate):
            return env_candidate

    repo_cfg = os.path.join(_parent_dir, "OCDocker.cfg")
    if os.path.isfile(repo_cfg):
        return os.path.abspath(repo_cfg)

    cwd_cfg = os.path.abspath("OCDocker.cfg")
    if os.path.isfile(cwd_cfg):
        return cwd_cfg

    return None


def _ensure_reference_order_config(config_path: str | None) -> str | None:
    """Load config into OCDocker singleton so get_score can enforce column order."""

    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return None

    os.environ["OCDOCKER_CONFIG"] = resolved
    loaded = occonfig.OCDockerConfig.from_config_file(resolved)
    occonfig.set_config(loaded)
    return resolved


def main() -> None:
    '''Run OCScore inference on a feature CSV and print the resulting scores.'''

    parser = argparse.ArgumentParser(description="Run OCScore inference from a CSV file.")
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to input CSV file containing features for inference.",
    )
    parser.add_argument(
        "--model-name",
        default="OCScore",
        help="Model name used to locate model/mask/scaler files (default: OCScore).",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing model artifacts. Defaults to OCScore_models.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit model path. If omitted, inferred from --model-name in --models-dir.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to OCDocker config file (.cfg/.yml). Needed to enforce reference_column_order.",
    )
    parser.add_argument(
        "--scaler-path",
        default=None,
        help="Optional scaler path. If omitted, tries <model_name>_scaler.pkl in models dir.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable mask loading and run inference without a mask.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU when available (PyTorch models).",
    )
    parser.add_argument(
        "--output-csv",
        default="predictions_from_csv.csv",
        help="Output CSV file for predictions (default: predictions_from_csv.csv).",
    )
    parser.add_argument(
        "--score-column",
        default="OCSCORE",
        help="Column name to store predictions in the output CSV (default: OCSCORE).",
    )
    parser.add_argument(
        "--disable-reference-order",
        action="store_true",
        help="Disable enforcement of reference_column_order (not recommended).",
    )
    parser.add_argument(
        "--invert-conditionally",
        dest="invert_conditionally",
        action="store_true",
        default=False,
        help="Invert VINA/SMINA/PLANTS-like columns before inference (default: enabled).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {args.csv_path}")

    enforce_reference_order = not args.disable_reference_order
    loaded_config_path = _ensure_reference_order_config(args.config_path)

    if enforce_reference_order:
        cfg = occonfig.get_config()
        if not cfg.paths.reference_column_order:
            raise ValueError(
                "reference_column_order is not set in the active config. "
                "Pass --config-path /path/to/OCDocker.cfg (or set OCDOCKER_CONFIG)."
            )
        print(f"Using OCDocker config: {loaded_config_path if loaded_config_path else 'active runtime config'}")

    # Keep a full copy for output (same input rows/columns + OCScore column).
    input_df = pd.read_csv(args.csv_path)
    # Use OCScore loader for inference input (drops rows with invalid NaNs).
    scoring_df = ocscoreio.load_data(args.csv_path)

    if scoring_df.empty:
        raise ValueError("No valid rows found for inference after CSV preprocessing.")

    models_dir = os.path.abspath(args.models_dir) if args.models_dir else ocscoreio.get_models_dir()
    model_path = _resolve_model_path(args.model_name, models_dir, args.model_path)

    mask = None
    if not args.no_mask:
        try:
            mask = ocscoreio.load_mask(args.model_name, models_dir=models_dir)
            print(f"Loaded mask for model '{args.model_name}'.")
        except FileNotFoundError:
            print(f"Mask file not found for model '{args.model_name}'. Continuing without mask.")

    scaler_path = args.scaler_path
    if scaler_path is None:
        candidate_scaler = os.path.join(models_dir, f"{args.model_name}_scaler.pkl")
        if os.path.isfile(candidate_scaler):
            scaler_path = candidate_scaler

    if scaler_path:
        print(f"Using scaler: {scaler_path}")
    else:
        print("No scaler provided/found. Inference will fit a scaler on the prediction data.")

    predictions = ocscoring.get_score(
        model_path=model_path,
        data=scoring_df,
        mask=mask,
        scaler_path=scaler_path,
        use_gpu=args.use_gpu,
        enforce_reference_column_order=enforce_reference_order,
        invert_conditionally=args.invert_conditionally,
    )

    if "predicted_score" not in predictions.columns:
        raise ValueError("Prediction output does not contain 'predicted_score' column.")

    # Build output with the same rows/columns as input plus an OCScore column.
    # Rows removed during preprocessing keep NaN in the score column.
    output_df = input_df.copy()
    output_df[args.score_column] = np.nan
    output_df.loc[scoring_df.index, args.score_column] = predictions["predicted_score"].to_numpy()

    print(f"Input shape: {input_df.shape}")
    print(f"Output shape: {output_df.shape}")
    print(f"Predictions assigned to {predictions['predicted_score'].notna().sum()} rows.")
    print(output_df[[args.score_column]].head())

    if args.output_csv:
        output_df.to_csv(args.output_csv, index=False)
        print(f"Saved output CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
