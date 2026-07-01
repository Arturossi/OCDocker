#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: generate selected OCScore DUDEz SHAP dependence plots from existing
SHAP artifacts.

The script does not retrain models and does not recompute SHAP values. It loads
existing ``shap_values.csv`` files, aligns the DUDEz feature matrix to the saved
validation split, and writes dependence plots for requested features.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.Analysis.SHAP.Plots import save_dependence_plots


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "OCSCORE_OUTPUT_DIR",
        "/data/hd4tb/OCDocker/data/ocdb2/OCScore/output",
    )
)

DEFAULT_FEATURES = [
    "ligand_PMI2",
    "ligand_PMI1",
    "ligand_BertzCT",
    "ligand_AUTOCORR2D_96",
    "ligand_AUTOCORR2D_93",
    "ligand_TPSA",
    "ligand_EState_VSA6",
    "ligand_SMR_VSA6",
]

DEFAULT_POLICIES = [
    "full",
    "ligand_plus_scoring_function_no_plants",
    "ligand_plus_scoring_function_no_pmi",
    "ligand_plus_scoring_function_no_pmi_no_plants",
    "ligand_plus_scoring_function_no_pmi_no_autocorr2d",
    "ligand_plus_scoring_function_no_pmi_no_shape_size_no_autocorr2d_no_vsa",
    "ligand_only",
    "scoring_function_only",
    "shape_only",
]


# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    '''Parse command-line arguments.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    '''

    parser = argparse.ArgumentParser(
        description="Generate selected DUDEz SHAP dependence plots from existing OCScore artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="OCScore output directory containing train/ and export/",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policies to process. Use 'full' or 'full_ocscore' for the root model.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature names for SHAP dependence plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    return parser.parse_args(argv)


def _policy_paths(output_root: Path, policy: str) -> tuple[Path, Path, Path]:
    '''Resolve analysis, train, and SHAP directories for a policy.

    Parameters
    ----------
    output_root : Path
        OCScore output directory.
    policy : str
        Policy label.

    Returns
    -------
    tuple[Path, Path, Path]
        Analysis JSON path, train directory, and SHAP directory.
    '''

    if policy in {"full", "full_ocscore"}:
        return (
            output_root / "export" / "best_model_analysis_complete.json",
            output_root / "train",
            output_root / "export" / "dudez" / "shap",
        )
    return (
        output_root / "export" / "ablations" / policy / "best_model_analysis_complete.json",
        output_root / "train" / "ablations" / policy,
        output_root / "export" / "ablations" / policy / "dudez" / "shap",
    )


def _load_validation_feature_matrix(
        dudez_csv: Path,
        split_npz: Path,
    ) -> pd.DataFrame:
    '''Load DUDEz feature rows aligned to the saved validation split.

    Parameters
    ----------
    dudez_csv : Path
        DUDEz modeling matrix.
    split_npz : Path
        Saved split index archive.

    Returns
    -------
    pd.DataFrame
        Validation feature matrix.
    '''

    validation_indices = np.load(split_npz)["validation_indices"]
    return pd.read_csv(dudez_csv).iloc[validation_indices].reset_index(drop=True)


## Public ##


def main(argv: Sequence[str] | None = None) -> int:
    '''Run the SHAP dependence plotting example.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit code.
    '''

    args = _parse_args(argv)
    output_root = args.output_root
    for policy in args.policies:
        analysis_path, train_dir, shap_dir = _policy_paths(output_root, policy)
        shap_csv = shap_dir / "shap_values.csv"
        dudez_csv = train_dir / "modeling_dudez.csv"
        if not analysis_path.exists() or not shap_csv.exists() or not dudez_csv.exists():
            print(f"[SKIP] {policy}: missing analysis, SHAP CSV, or DUDEz matrix")
            continue

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        dudez_best = Path(analysis.get("best_model_paths", {}).get("dudez", ""))
        split_npz = dudez_best / "split_indices.npz"
        if not split_npz.exists():
            print(f"[SKIP] {policy}: missing split indices")
            continue

        shap_values = pd.read_csv(shap_csv)
        feature_matrix = _load_validation_feature_matrix(dudez_csv, split_npz)
        written = save_dependence_plots(
            shap_values=shap_values,
            feature_matrix=feature_matrix,
            feature_names=list(shap_values.columns),
            requested_features=args.features,
            output_dir=shap_dir,
            policy=f"{policy}_dudez",
            dpi=args.dpi,
        )
        print(f"[DONE] {policy}: skipped={written.get('skipped_features', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
