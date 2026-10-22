#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: scatter mean DUDEz validation BEDROC against SHAP shortcut risk (the
maximum share of total SHAP importance concentrated in a single feature,
across replicas) for every feature-ablation policy.

The script does not retrain models and does not recompute SHAP values. Mean
validation BEDROC is read from ``ablation_summary.csv`` and eligibility is
computed with the paper's paired, Holm-corrected validation test. Per-policy
shortcut risk is aggregated from the per-replica SHAP exports under
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

import matplotlib
import pandas as pd
from matplotlib import font_manager

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot
from OCDocker.OCScore.Analysis.AblationSignificance import AblationSignificanceConfig
from OCDocker.OCScore.Analysis.AblationSignificance import build_ablation_bedroc_significance_table
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
VALIDATION_MEAN_COLUMN = "dudez_validation_primary_metric_mean"
VALIDATION_REPLICA_COLUMN = "dudez_best_validation_metric"
DEFAULT_ALPHA = 0.05

# Below this maximum single-feature share (across replicas), the explanation
# was considered genuinely distributed (Seção 4.4).
DEFAULT_RISK_THRESHOLD_PCT = 20.0
RISK_RULE_DECIMALS = 1

# Manual (dx, dy) point-label offsets, in points.  Every label gets its own
# position so that the dense validation cluster remains readable at the size
# used in the paper; the plotting helper draws subtle leaders back to points.
DEFAULT_LABEL_OFFSETS = {
    "ligand_plus_scoring_function": (-28, -4),  # #01
    "ligand_plus_scoring_function_no_shape_core": (13, -3),  # #02
    "ligand_plus_scoring_function_no_shape_size": (-18, 9),  # #03 (recommended)
    "ligand_plus_scoring_function_no_plants": (12, 7),  # #04
    "ligand_plus_scoring_function_no_pmi": (12, -14),  # #05
    "no_shape_core_no_receptor_surface_counts": (13, 7),  # #06
    "ligand_plus_scoring_function_no_pmi_no_autocorr2d": (-34, 9),  # #07
    "ligand_plus_scoring_function_no_pmi_no_plants": (-32, -8),  # #08
    "ligand_plus_scoring_function_no_shape_size_no_autocorr2d": (24, 11),  # #09
    "ligand_plus_scoring_function_clean_receptor": (13, -13),  # #10
    "no_shape_core_no_receptor_length_pair": (12, 7),  # #11
    "no_pmi": (13, 8),  # #12
    "no_shape_core_no_receptor_surface_size": (13, -10),  # #13
    "full_ocscore": (13, 7),  # #14
    "no_shape_core": (13, 9),  # #15
    "no_ligand_shape_size": (-32, -22),  # #16
    "ligand_only": (-28, -8),  # #17
    "ligand_plus_scoring_function_no_pmi_no_shape_size_no_autocorr2d_no_vsa": (15, 7),  # #18
    "scoring_function_only": (-28, 7),  # #19
    "no_scoring_function": (12, 7),  # #20
    "receptor_plus_scoring_function": (10, 7),  # #21
    "shape_only": (10, 7),  # #22
}

# The paper renders this figure in Portuguese; the library ships generic English
# defaults (e.g. "BEDROC (test)"), which do not match this figure's validation-set
# axis. Both localized string sets live here, in the caller, and never in the library.
PT_TEXT = {
    "title": "BEDROC de validação vs. risco de atalho (22 configurações)",
    "xlabel": "BEDROC (validação, DUDEz)",
    "ylabel": "Risco de atalho\n(máx. % da importância SHAP em um único atributo)",
    "highlight_note": "recomendada",
}
PT_LEGEND = {
    "reference": "Modelo completo (full_ocscore, referência)",
    "discarded": "Elegíveis: sem piora detectável ($p_{\\mathrm{Holm}}\\geq0{,}05$)\nDescartadas: risco $\\geq$ 20%",
    "retained": "Elegíveis: sem piora detectável ($p_{\\mathrm{Holm}}\\geq0{,}05$)\nMantidas: risco < 20%",
    "other": "Não elegíveis: piora significativa\n($p_{\\mathrm{Holm}}<0{,}05$)",
}
EN_TEXT = {
    "title": "Validation BEDROC vs. shortcut risk (22 configurations)",
    "xlabel": "BEDROC (validation, DUDEz)",
    "ylabel": "Shortcut risk\n(max. % of SHAP importance in a single feature)",
    "highlight_note": "recommended",
}
EN_LEGEND = {
    "reference": "Full model (full_ocscore, reference)",
    "discarded": "Eligible: no detectable degradation ($p_{\\mathrm{Holm}}\\geq0.05$)\nDiscarded: risk $\\geq$ 20%",
    "retained": "Eligible: no detectable degradation ($p_{\\mathrm{Holm}}\\geq0.05$)\nRetained: risk < 20%",
    "other": "Not eligible: significant degradation\n($p_{\\mathrm{Holm}}<0.05$)",
}

# The configuration recommended in the paper; called out with an arrow.
RECOMMENDED_POLICY = "ligand_plus_scoring_function_no_shape_size"


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
        default=None,
        help="Policies to highlight as retained. Derived from the shortcut-risk rule when omitted.",
    )
    parser.add_argument(
        "--bad-policies",
        nargs="*",
        default=None,
        help="Policies to highlight as discarded. Derived from the shortcut-risk rule when omitted.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=DEFAULT_RISK_THRESHOLD_PCT,
        help="Shortcut-risk guide line, in percent (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Holm-corrected validation significance threshold (default: %(default)s).",
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
        help="Output directory for the plot.",
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
    # These identifiers are fixed throughout the paper from the descending test
    # BEDROC ordering. The scatter axis below intentionally uses validation BEDROC.
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
        summary_df[["feature_policy_name", VALIDATION_MEAN_COLUMN]].rename(
            columns={"feature_policy_name": "policy", VALIDATION_MEAN_COLUMN: "bedroc_mean"}
        ),
        on="policy",
        how="left",
    )
    plot_df["rank_label"] = plot_df["policy"].map(rank_labels)

    significance_df = build_ablation_bedroc_significance_table(
        args.ablation_summary_csv,
        config=AblationSignificanceConfig(
            reference_policy=args.reference_policy,
            metric_column=VALIDATION_REPLICA_COLUMN,
            method="paired_ttest",
            correction_method="holm",
            alpha=args.alpha,
        ),
    )
    eligibility_df = significance_df[["policy", "pvalue_corrected", "reject_null"]].copy()
    eligibility_df["eligible"] = eligibility_df["pvalue_corrected"].notna() & ~eligibility_df["reject_null"]
    plot_df = plot_df.merge(eligibility_df[["policy", "eligible"]], on="policy", how="left")
    plot_df["eligible"] = plot_df["eligible"].fillna(False)
    # The paper reports shortcut risk to one decimal place and treats #05's
    # 19.983868% as 20.0%, exactly at the operational cutoff.
    plot_df["risk_for_rule_pct"] = plot_df["top1_pct_max"].round(RISK_RULE_DECIMALS)

    derived_good, derived_bad = ocstatplot.classify_policies_by_eligibility_and_shortcut_risk(
        plot_df,
        reference_policy=args.reference_policy,
        eligibility_column="eligible",
        risk_threshold=args.risk_threshold,
        risk_column="risk_for_rule_pct",
    )
    good_policies = args.good_policies if args.good_policies is not None else derived_good
    bad_policies = args.bad_policies if args.bad_policies is not None else derived_bad
    print(f"Retained ({len(good_policies)}): {', '.join(good_policies)}")
    print(f"Discarded ({len(bad_policies)}): {', '.join(bad_policies)}")

    text = PT_TEXT if args.lang == "pt" else EN_TEXT
    legend = PT_LEGEND if args.lang == "pt" else EN_LEGEND

    _use_paper_font()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    ocstatplot.plot_bedroc_vs_shortcut_risk_scatter(
        plot_df,
        reference_policy=args.reference_policy,
        good_policies=good_policies,
        bad_policies=bad_policies,
        show_rule_geometry=False,
        risk_threshold=args.risk_threshold,
        bedroc_column="bedroc_mean",
        risk_column="top1_pct_max",
        label_column="rank_label",
        highlight_policy=RECOMMENDED_POLICY,
        metric_label="BEDROC",
        legend_labels=legend,
        label_offsets=DEFAULT_LABEL_OFFSETS,
        output_dir=str(args.figures_dir),
        **text,
    )
    output_png = args.figures_dir / "ablation_bedroc_vs_shortcut_risk_scatter.png"
    print(f"Wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
