#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: plot per-descriptor-family SHAP importance composition, averaged
across replicas, for a curated subset of feature-ablation policies.

The script does not retrain models and does not recompute SHAP values. It
reads the per-replica SHAP exports already written under
``export/replica_analysis/{full,ablations/<policy>}/replica_XXX/<task>/shap/
shap_values.csv`` by ``OCScore.Analysis.SHAP.ExportRunner.run_export_shap_analysis``,
aggregates family-level importance across replicas with
``OCScore.Analysis.SHAP.Dominance.aggregate_family_composition``, and plots the
result with ``OCScore.Analysis.SHAP.DominancePlots.save_family_composition_stacked_plot``.

Usage:

    python examples/23_ocscore_shap_family_composition_plot.py \\
        --output-root /path/to/OCScore/output \\
        --figures-dir /path/to/plots
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
import pandas as pd
from matplotlib import font_manager

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.Analysis.SHAP.Dominance import aggregate_family_composition
from OCDocker.OCScore.Analysis.SHAP.DominancePlots import save_family_composition_stacked_plot
from OCDocker.OCScore.Utils.FeaturePolicy import FULL_OCSCORE_POLICY_NAME

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
DEFAULT_ABLATION_SUMMARY_CSV = DEFAULT_OUTPUT_ROOT / "train" / "ablations" / "ablation_summary.csv"

# Coarse 4-family view used for the paper figure: PMI carved out first (the
# shortcut of interest), then receptor, then non-ligand scoring-tool
# descriptors, then a "ligand_other" catch-all for the remaining ligand
# descriptors. Order matters: assign_feature_families keeps the first match.
DEFAULT_FAMILY_SPEC = {
    "ligand_PMI": ["ligand_PMI*"],
    "receptor": ["receptor_*"],
    "scoring_function": ["plants_*", "vina_*", "smina_*", "gnina_*", "oddt_*"],
    "ligand_other": ["ligand_*"],
}

# Sized ~1:1 with the width it is embedded at in the paper (A4, ABNT margins =>
# ~16 cm of text width), so the point sizes inside the figure survive the embed
# instead of being shrunk by ~40% as they would at the library's wider default.
PAPER_FIGSIZE = (6.3, 4.6)

# The paper renders this figure in Portuguese; the library ships English defaults
# for title/xlabel but falls back to raw family keys (e.g. "ligand_PMI") when no
# family_labels are given, so both localized string sets live here, in the caller,
# and never in the library.
PT_TEXT = {
    "title": "Composição da importância SHAP por família de descritores",
    "xlabel": "Importância SHAP relativa (%, média entre as réplicas)",
    "family_labels": {
        "ligand_PMI": "PMI do ligante",
        "ligand_other": "Demais descritores do ligante",
        "receptor": "Descritores do receptor",
        "scoring_function": "Funções de pontuação",
    },
}
EN_TEXT = {
    "title": "SHAP importance composition by descriptor family",
    "xlabel": "Relative SHAP importance (%, mean across replicas)",
    "family_labels": {
        "ligand_PMI": "Ligand PMI",
        "ligand_other": "Other ligand descriptors",
        "receptor": "Receptor descriptors",
        "scoring_function": "Scoring functions",
    },
}

# Curated subset illustrating the pattern discussed in the paper: PMI present
# and dominant, PMI removed but dominance relocated to one remaining
# descriptor, dominance genuinely distributed, and a single-family baseline.
DEFAULT_POLICIES = [
    "ligand_plus_scoring_function",
    FULL_OCSCORE_POLICY_NAME,
    "ligand_only",
    "ligand_plus_scoring_function_no_shape_core",
    "no_shape_core_no_receptor_length_pair",
    "ligand_plus_scoring_function_no_pmi",
    "ligand_plus_scoring_function_no_shape_size",
    "no_ligand_shape_size",
    "scoring_function_only",
]


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
        description="Plot per-family SHAP importance composition for a curated set of ablation policies.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="OCScore output directory containing train/ and export/ (default: %(default)s).",
    )
    parser.add_argument(
        "--ablation-summary-csv",
        type=Path,
        default=DEFAULT_ABLATION_SUMMARY_CSV,
        help="Ablation summary CSV, used only to derive the Table-3/4 rank numbers (default: %(default)s).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policies to plot, top-to-bottom.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dudez",
        help="Evaluation task subdirectory to read SHAP exports from (default: %(default)s).",
    )
    parser.add_argument(
        "--n-replicas",
        type=int,
        default=5,
        help="Number of replicas per policy (default: %(default)s).",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "pt"),
        default="en",
        help="Figure language. The library ships English defaults; 'pt' passes the "
             "paper's Portuguese strings from this script (default: %(default)s).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        required=True,
        help="Output directory for the plot and its underlying CSV.",
    )
    return parser.parse_args(argv)


def _use_paper_font() -> None:
    '''Match the figure typeface to the paper's (Arial-metric) body font.

    The library deliberately does not touch rcParams, so a caller that embeds the
    figure in a typeset document sets the font itself. Falls back silently to the
    matplotlib default when the font is not installed.
    '''

    for family in ("Arial", "Liberation Sans"):
        if any(f.name == family for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = family
            return


def _rank_labels(ablation_summary_csv: Path) -> dict[str, str]:
    '''Rank every policy by mean DUDEz test BEDROC, matching the paper's Table 3/4 numbering.

    Parameters
    ----------
    ablation_summary_csv : Path
        Ablation summary CSV with ``feature_policy_name`` and ``dudez_test_bedroc_mean`` columns.

    Returns
    -------
    dict[str, str]
        Policy name to zero-padded rank label (``"01"``, ``"02"``, ...).
    '''

    summary_df = pd.read_csv(ablation_summary_csv).sort_values("dudez_test_bedroc_mean", ascending=False)
    return {
        str(policy): f"{rank:02d}"
        for rank, policy in enumerate(summary_df["feature_policy_name"], start=1)
    }


## Public ##

def main(argv: Sequence[str] | None = None) -> int:
    '''Run the SHAP family composition plotting example.

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
    composition = aggregate_family_composition(
        args.output_root / "export",
        args.policies,
        task=args.task,
        n_replicas=args.n_replicas,
        family_spec=DEFAULT_FAMILY_SPEC,
    )

    policy_labels = _rank_labels(args.ablation_summary_csv) if args.ablation_summary_csv.is_file() else None

    _use_paper_font()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_family_composition_stacked_plot(
        composition,
        args.policies,
        str(args.figures_dir),
        policy_labels=policy_labels,
        figsize=PAPER_FIGSIZE,
        legend_ncol=2,
        file_stem="figura3_shap_familia",
        **(PT_TEXT if args.lang == "pt" else EN_TEXT),
    )
    print(f"Wrote {artifacts['family_composition_png']}")
    print(f"Wrote {artifacts['family_composition_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
