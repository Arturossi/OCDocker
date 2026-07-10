#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: test whether feature-ablation policies significantly change DUDEz test
BEDROC relative to the full OCScore feature set.

Replica ``i`` uses the same PDBbind/DUDEz data split for every feature policy
(``seed = base_seed + replica_index`` controls the split; see
``OCScore.Optimization.Protocol`` / ``StagedTrainProtocol``), so policies are
compared with a paired test on matched replica seeds, Holm-corrected across
all policies tested against the reference.

Usage:

    python examples/22_ocscore_ablation_bedroc_significance.py \\
        --ablation-summary-csv /path/to/train/ablations/ablation_summary.csv \\
        --output-csv /path/to/ablation_bedroc_significance.csv \\
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.Analysis.AblationSignificance import AblationSignificanceConfig
from OCDocker.OCScore.Analysis.AblationSignificance import PAIRED_TEST_METHODS
from OCDocker.OCScore.Analysis.AblationSignificance import build_ablation_bedroc_significance_table
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
        description="Test feature-ablation policies for a significant DUDEz BEDROC change vs the reference policy.",
    )
    parser.add_argument(
        "--ablation-summary-csv",
        type=Path,
        default=DEFAULT_ABLATION_SUMMARY_CSV,
        help="Ablation summary CSV listing one row per feature policy (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-policy",
        type=str,
        default=FULL_OCSCORE_POLICY_NAME,
        help="Reference feature-policy name (default: %(default)s).",
    )
    parser.add_argument(
        "--metric-column",
        type=str,
        default="dudez_test_bedroc",
        help="Per-replica metric column to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=PAIRED_TEST_METHODS,
        default="paired_ttest",
        help="Paired significance test (default: %(default)s).",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        default="holm",
        help="Multiple-comparisons correction passed to statsmodels.stats.multitest.multipletests (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <ablation-summary-csv dir>/ablation_bedroc_significance.csv).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="If set, write a significance bar chart PNG to this directory.",
    )
    return parser.parse_args(argv)


## Public ##

def main(argv: Sequence[str] | None = None) -> int:
    '''Run the ablation BEDROC significance example.

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
    config = AblationSignificanceConfig(
        reference_policy=args.reference_policy,
        metric_column=args.metric_column,
        method=args.method,
        correction_method=args.correction_method,
        alpha=args.alpha,
    )
    significance_df = build_ablation_bedroc_significance_table(
        args.ablation_summary_csv,
        config=config,
    )

    output_csv = args.output_csv or args.ablation_summary_csv.parent / "ablation_bedroc_significance.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    significance_df.to_csv(output_csv, index=False)
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(significance_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")

    print(f"\n=== {args.metric_column} vs {args.reference_policy} ({args.method}, {args.correction_method}-corrected) ===\n")
    print(
        significance_df[
            ["policy", "n_pairs", "reference_mean", "policy_mean", "mean_diff", "pvalue", "pvalue_corrected", "direction"]
        ].to_string(index=False)
    )

    if args.figures_dir:
        import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot

        figures_path = Path(args.figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        ocstatplot.plot_ablation_bedroc_significance_bars(
            significance_df,
            reference_policy=args.reference_policy,
            metric_label=args.metric_column.upper(),
            output_dir=str(figures_path),
            alpha=args.alpha,
        )
        print(f"Wrote plot to {figures_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
