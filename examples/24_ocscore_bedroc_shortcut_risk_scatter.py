#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: scatter mean DUDEz test BEDROC against SHAP shortcut risk (the
maximum share of total SHAP importance concentrated in a single feature,
across replicas) for every feature-ablation policy.

The script does not retrain models and does not recompute SHAP values. Per-
policy mean BEDROC is read from ``ablation_summary.csv``; per-policy shortcut
risk is aggregated from the per-replica SHAP exports under
``export/replica_analysis/{full,ablations/<policy>}/replica_XXX/<task>/shap/
shap_values.csv`` with ``OCScore.Analysis.SHAP.Dominance.aggregate_dominant_feature_risk``,
then plotted with ``OCScore.Analysis.Plotting.Stats.plot_bedroc_vs_shortcut_risk_scatter``.

Usage:

    python examples/24_ocscore_bedroc_shortcut_risk_scatter.py \\
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

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot
from OCDocker.OCScore.Analysis.SHAP.Dominance import aggregate_dominant_feature_risk
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

# The final recommended configuration and its two alternatives (Seção 4.5):
# lowest shortcut risk among configurations statistically tied with the top
# of the BEDROC ranking.
DEFAULT_GOOD_POLICIES = [
    "ligand_plus_scoring_function_no_shape_size",
    "ligand_plus_scoring_function_no_pmi",
    "no_ligand_shape_size",
]

# The three highest raw-BEDROC configurations, discarded because their
# performance is explained by a shortcut (PMI, or a secondary one such as
# ligand_BertzCT once PMI is removed) rather than a genuine advantage (Seção 4.5).
DEFAULT_BAD_POLICIES = [
    "ligand_plus_scoring_function",
    "ligand_plus_scoring_function_no_shape_core",
    "ligand_plus_scoring_function_no_plants",
]

# Below this maximum single-feature share (across replicas), the explanation
# was considered genuinely distributed (Seção 4.4).
DEFAULT_RISK_THRESHOLD_PCT = 20.0


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
        description="Scatter mean DUDEz BEDROC against SHAP shortcut risk for every ablation policy.",
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
        help="Ablation summary CSV with feature_policy_name and dudez_test_bedroc_mean (default: %(default)s).",
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
        "--reference-policy",
        type=str,
        default=FULL_OCSCORE_POLICY_NAME,
        help="Reference feature-policy name (default: %(default)s).",
    )
    parser.add_argument(
        "--good-policies",
        nargs="*",
        default=DEFAULT_GOOD_POLICIES,
        help="Policies to highlight as recommended/alternatives.",
    )
    parser.add_argument(
        "--bad-policies",
        nargs="*",
        default=DEFAULT_BAD_POLICIES,
        help="Policies to highlight as discarded due to shortcut risk.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=DEFAULT_RISK_THRESHOLD_PCT,
        help="Shortcut-risk guide line, in percent (default: %(default)s).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        required=True,
        help="Output directory for the plot.",
    )
    return parser.parse_args(argv)


## Public ##

def main(argv: Sequence[str] | None = None) -> int:
    '''Run the BEDROC-vs-shortcut-risk scatter example.

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
    summary_df = pd.read_csv(args.ablation_summary_csv)
    policies = summary_df.sort_values("dudez_test_bedroc_mean", ascending=False)["feature_policy_name"].tolist()
    rank_labels = {policy: f"{rank:02d}" for rank, policy in enumerate(policies, start=1)}

    risk_df = aggregate_dominant_feature_risk(
        args.output_root / "export",
        policies,
        task=args.task,
        n_replicas=args.n_replicas,
        reference_policy_name=args.reference_policy,
    )

    plot_df = risk_df.merge(
        summary_df[["feature_policy_name", "dudez_test_bedroc_mean"]].rename(
            columns={"feature_policy_name": "policy", "dudez_test_bedroc_mean": "bedroc_mean"}
        ),
        on="policy",
        how="left",
    )
    plot_df["rank_label"] = plot_df["policy"].map(rank_labels)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    ocstatplot.plot_bedroc_vs_shortcut_risk_scatter(
        plot_df,
        reference_policy=args.reference_policy,
        good_policies=args.good_policies,
        bad_policies=args.bad_policies,
        risk_threshold=args.risk_threshold,
        bedroc_column="bedroc_mean",
        risk_column="top1_pct_max",
        label_column="rank_label",
        metric_label="BEDROC",
        output_dir=str(args.figures_dir),
    )
    output_png = args.figures_dir / "ablation_bedroc_vs_shortcut_risk_scatter.png"
    print(f"Wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
