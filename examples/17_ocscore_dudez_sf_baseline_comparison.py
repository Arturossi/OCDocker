#!/usr/bin/env python3

# Description
###############################################################################
"""
Compare DUDEz screening separability for individual scoring functions vs an exported OCScore model.

Uses the same receptor-held-out row indices saved in the DUDEz ``best_model/split_indices.npz``
bundle and the same grouped metrics as staged Optuna (``evaluate_screening_metrics``),
plus OCScore calibration metrics (Brier, log loss, ECE) with optional Platt/isotonic
fitting on the validation split.

Usage:

    python examples/17_ocscore_dudez_sf_baseline_comparison.py \\
        --reduction-archive /path/to/ocdocker.tar.gz \\
        --export-dir /path/to/replica_000/dudez/best_model \\
        --output-csv /path/to/dudez_sf_baseline_comparison.csv
"""

# Imports
###############################################################################
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from OCDocker.OCScore.Optimization import ModelExport as ocexport
from OCDocker.OCScore.Analysis.Metrics.Calibration import DEFAULT_CALIBRATION_METHOD
from OCDocker.OCScore.Analysis.Metrics.Calibration import ProbabilityCalibrator
from OCDocker.OCScore.Analysis.Metrics.Calibration import calibration_metric_names
from OCDocker.OCScore.Analysis.Metrics.Calibration import merge_calibration_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics_by_group
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_scoring_functions_by_group
from OCDocker.OCScore.Optimization.StagedOptuna import derive_dudez_labels
from OCDocker.OCScore.Analysis.Plotting.CrossValidationPlots import save_baseline_comparison_figures
from OCDocker.OCScore.Analysis.Plotting.CrossValidationPlots import save_calibration_reliability_figures
from OCDocker.OCScore.Analysis.Plotting.CrossValidationPlots import save_per_target_figures
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_descriptor_aggregates_by_group
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import evaluate_sf_consensus_by_group
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import DESCRIPTOR_AGGREGATE_SCORER_TYPE
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import SF_CONSENSUS_SCORER_TYPE
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import format_descriptor_aggregate_scorer
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import format_sf_consensus_scorer
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import row_aggregate_feature_scores
from OCDocker.OCScore.Utils.DescriptorAggregateBaselines import row_aggregate_sf_scores
from OCDocker.OCScore.Utils.FeatureReduction import DEFAULT_SCORING_PATTERNS, split_descriptor_blocks

# License
###############################################################################
"""
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""


# Constants
###############################################################################
REDUCED_DATASET_NAME = "reduced_dataset.csv"
SELECTED_FEATURES_NAME = "selected_features.json"
DATASET_COLUMN_CANDIDATES = ["dataset", "source", "db"]
DUDEZ_DATASET_VALUES = {"dudez", "dude-z", "dude_z"}
DEFAULT_GROUP_COLUMN = "receptor"
DEFAULT_KIND_COLUMN = "kind"
REPORT_METRICS = (
    "BEDROC",
    "ROC-AUC",
    "PR-AUC",
    "EF1%",
    "NDCG@1%",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "TP",
    "FP",
    "TN",
    "FN",
    *calibration_metric_names(include_calibrated=True),
)


# Functions
###############################################################################
## Private ##

def _load_json_from_tar(tar: tarfile.TarFile, member_name: str) -> Any:
    member = tar.getmember(member_name)
    payload = tar.extractfile(member)
    if payload is None:
        raise FileNotFoundError(member_name)
    return json.load(payload)


def load_dudez_and_selected_features(reduction_source: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load reduced DUDEz rows and selected feature names from a tar archive or directory."""

    if reduction_source.is_dir():
        reduced_path = reduction_source / REDUCED_DATASET_NAME
        selected_path = reduction_source / SELECTED_FEATURES_NAME
        reduced = pd.read_csv(reduced_path, low_memory=False)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        with tarfile.open(reduction_source, "r:*") as tar:
            reduced = pd.read_csv(io.BytesIO(tar.extractfile(REDUCED_DATASET_NAME).read()), low_memory=False)
            selected = _load_json_from_tar(tar, SELECTED_FEATURES_NAME)

    source_column = next((column for column in DATASET_COLUMN_CANDIDATES if column in reduced.columns), None)
    if source_column is None:
        raise ValueError(
            f"{REDUCED_DATASET_NAME!r} requires one dataset/source column from {DATASET_COLUMN_CANDIDATES}."
        )
    normalized = reduced[source_column].astype(str).str.strip().str.lower()
    dudez_df = reduced[normalized.isin(DUDEZ_DATASET_VALUES)].copy()
    if dudez_df.empty:
        raise ValueError("No DUDEz rows found in the reduced dataset.")

    if isinstance(selected, dict):
        selected = selected.get("selected_features", selected.get("features"))
    selected_features = [str(item) for item in selected]
    return dudez_df, selected_features


def surviving_scoring_columns(selected_features: Sequence[str]) -> list[str]:
    """Return scoring-function columns that survived feature reduction.

    Uses the same block detection as feature reduction (``Complexes.allDescriptors``
    plus ``DEFAULT_SCORING_PATTERNS``, including ``oddt_``).
    """

    blocks = split_descriptor_blocks(
        selected_features,
        scoring_patterns=DEFAULT_SCORING_PATTERNS,
        use_scoring_model_descriptors=True,
    )
    scoring_columns = list(blocks.scoring)
    if not scoring_columns:
        raise ValueError(
            "No scoring-function columns found among selected features. "
            f"Unmatched selected columns: {blocks.unmatched}"
        )
    return scoring_columns


def infer_higher_is_better(scores: np.ndarray, labels: np.ndarray) -> bool:
    """Infer whether larger raw scores favor actives on a split."""

    mask = np.isfinite(scores)
    if int(mask.sum()) < 10:
        return False
    active_mean = float(np.nanmean(scores[mask & (labels == 1)]))
    decoy_mean = float(np.nanmean(scores[mask & (labels == 0)]))
    if not np.isfinite(active_mean) or not np.isfinite(decoy_mean):
        return False
    return active_mean >= decoy_mean


def predict_ocscore_logits(model: torch.nn.Module, device: torch.device, features: np.ndarray) -> np.ndarray:
    """Return classifier logits for a feature matrix."""

    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy().reshape(-1)


def evaluate_scorer(
        scorer_name: str,
        scorer_type: str,
        split_name: str,
        labels: np.ndarray,
        groups: np.ndarray,
        scores: np.ndarray,
        higher_is_better: bool,
        *,
        ocscore_logits: Optional[np.ndarray] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
    ) -> dict[str, Any]:
    """Evaluate one scorer on one split and return a flat result row."""

    nan_fraction = float(np.mean(~np.isfinite(scores)))
    metrics = evaluate_screening_metrics(
        labels,
        scores,
        groups=groups,
        higher_is_better=higher_is_better,
    )
    row: dict[str, Any] = {
        "scorer": scorer_name,
        "scorer_type": scorer_type,
        "split": split_name,
        "higher_is_better": bool(higher_is_better),
        "nan_fraction": nan_fraction,
        "n_rows": int(len(labels)),
        "n_groups_used": metrics.get("n_groups_used"),
        "ranking_metrics_valid": metrics.get("ranking_metrics_valid"),
    }
    for metric_name in REPORT_METRICS:
        if metric_name in metrics:
            row[metric_name] = metrics.get(metric_name)
    if scorer_type == "model" and ocscore_logits is not None:
        merge_calibration_metrics(
            row,
            labels,
            ocscore_logits,
            calibrator=calibrator,
        )
    return row


def _rows_from_group_metric_frame(
        group_df: pd.DataFrame,
        *,
        split_name: str,
        scorer: str,
        scorer_type: str,
        metric_names: Sequence[str],
        labels: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
    ) -> list[dict[str, Any]]:
    '''Expand a per-group metrics frame into long-format export rows.'''

    if group_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_values = None if groups is None else np.asarray(groups).reshape(-1)
    for _, record in group_df.iterrows():
        entry: dict[str, Any] = {
            "split": split_name,
            "group": str(record["group"]),
            "scorer": scorer,
            "scorer_type": scorer_type,
        }
        for metric_name in metric_names:
            if metric_name in record and pd.notna(record[metric_name]):
                entry[metric_name] = float(record[metric_name])
        if (
            scorer_type == "model"
            and calibrator is not None
            and labels is not None
            and logits is not None
            and group_values is not None
        ):
            mask = group_values.astype(str) == str(record["group"])
            if int(mask.sum()) > 0:
                merge_calibration_metrics(
                    entry,
                    labels[mask],
                    logits[mask],
                    calibrator=calibrator,
                )
        rows.append(entry)
    return rows


def build_per_target_comparison_table(
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
        export_dir: Path,
        group_column: str,
        kind_column: str,
        device: str,
        calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    ) -> pd.DataFrame:
    '''Evaluate every scorer on each receptor (validation and test splits).'''

    bundle = ocexport.load_exported_model(export_dir, device=device)
    split_indices = bundle["split_indices"]
    if not split_indices:
        raise FileNotFoundError(f"Missing split indices in export bundle: {export_dir}")

    labels = derive_dudez_labels(dudez_df, kind_column=kind_column)
    if group_column not in dudez_df.columns:
        raise ValueError(f"DUDEz group column is missing: {group_column!r}")
    groups_all = dudez_df[group_column].to_numpy()

    feature_matrix = dudez_df[list(selected_features)].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    if scaler is not None:
        feature_matrix = scaler.transform(feature_matrix).astype(np.float32)
    sf_columns = surviving_scoring_columns(selected_features)

    split_map = {
        "validation": np.asarray(split_indices["validation_indices"], dtype=np.int64),
        "test": np.asarray(split_indices["test_indices"], dtype=np.int64),
    }
    calibrator: Optional[ProbabilityCalibrator] = None
    if calibration_method != "none":
        val_idx = split_map["validation"]
        val_logits = predict_ocscore_logits(
            bundle["model"],
            bundle["device"],
            feature_matrix[val_idx],
        )
        calibrator = ProbabilityCalibrator.fit(
            labels[val_idx],
            val_logits,
            method=calibration_method,
            scores_are_logits=True,
        )
    elif bundle.get("calibrator") is not None:
        calibrator = bundle["calibrator"]

    ocscore_name = f"OCScore (trial {bundle['summary'].get('trial_number')})"
    rows: list[dict[str, Any]] = []

    for split_name, row_idx in split_map.items():
        y_split = labels[row_idx]
        g_split = groups_all[row_idx]
        logits = predict_ocscore_logits(bundle["model"], bundle["device"], feature_matrix[row_idx])

        oc_group_df = evaluate_screening_metrics_by_group(
            y_split,
            logits,
            g_split,
            higher_is_better=True,
            metric_names=REPORT_METRICS,
        )
        rows.extend(
            _rows_from_group_metric_frame(
                oc_group_df,
                split_name=split_name,
                scorer=ocscore_name,
                scorer_type="model",
                metric_names=REPORT_METRICS,
                labels=y_split,
                logits=logits,
                groups=g_split,
                calibrator=calibrator,
            )
        )

        if sf_columns:
            orientations = {
                column: infer_higher_is_better(
                    dudez_df[column].to_numpy(dtype=float)[row_idx],
                    y_split,
                )
                for column in sf_columns
            }
            sf_group_df = evaluate_scoring_functions_by_group(
                dudez_df,
                row_idx,
                labels,
                groups_all,
                sf_columns,
                metric_names=REPORT_METRICS,
                column_higher_is_better=orientations,
            )
            if not sf_group_df.empty:
                for column in sf_columns:
                    column_df = sf_group_df.loc[sf_group_df["scorer"] == column].drop(
                        columns=["scorer", "scorer_type"],
                        errors="ignore",
                    )
                    rows.extend(
                        _rows_from_group_metric_frame(
                            column_df,
                            split_name=split_name,
                            scorer=column,
                            scorer_type="sf",
                            metric_names=REPORT_METRICS,
                        )
                    )

        desc_group_frames = evaluate_descriptor_aggregates_by_group(
            feature_matrix,
            row_idx,
            labels,
            groups_all,
            metric_names=REPORT_METRICS,
            infer_higher_is_better=infer_higher_is_better,
        )
        for scorer_name, group_df in desc_group_frames.items():
            rows.extend(
                _rows_from_group_metric_frame(
                    group_df,
                    split_name=split_name,
                    scorer=scorer_name,
                    scorer_type=DESCRIPTOR_AGGREGATE_SCORER_TYPE,
                    metric_names=REPORT_METRICS,
                )
            )

        sf_consensus_frames = evaluate_sf_consensus_by_group(
            dudez_df,
            row_idx,
            sf_columns,
            labels,
            groups_all,
            metric_names=REPORT_METRICS,
            infer_higher_is_better=infer_higher_is_better,
        )
        for scorer_name, group_df in sf_consensus_frames.items():
            rows.extend(
                _rows_from_group_metric_frame(
                    group_df,
                    split_name=split_name,
                    scorer=scorer_name,
                    scorer_type=SF_CONSENSUS_SCORER_TYPE,
                    metric_names=REPORT_METRICS,
                )
            )

    return pd.DataFrame(rows)


def build_comparison_table(
        dudez_df: pd.DataFrame,
        selected_features: Sequence[str],
        export_dir: Path,
        group_column: str,
        kind_column: str,
        device: str,
        include_export_summary: bool,
        calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    ) -> Tuple[
        pd.DataFrame,
        Optional[ProbabilityCalibrator],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
    """Build the full baseline comparison table and optional OCScore calibrator."""

    bundle = ocexport.load_exported_model(export_dir, device=device)
    split_indices = bundle["split_indices"]
    if not split_indices:
        raise FileNotFoundError(f"Missing split indices in export bundle: {export_dir}")

    labels = derive_dudez_labels(dudez_df, kind_column=kind_column)
    if group_column not in dudez_df.columns:
        raise ValueError(f"DUDEz group column is missing: {group_column!r}")
    groups_all = dudez_df[group_column].to_numpy()

    feature_matrix = dudez_df[list(selected_features)].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    if scaler is not None:
        feature_matrix = scaler.transform(feature_matrix).astype(np.float32)
    sf_columns = surviving_scoring_columns(selected_features)
    if not sf_columns:
        raise ValueError("No scoring-function columns found among selected features.")

    rows: list[dict[str, Any]] = []
    split_map = {
        "validation": np.asarray(split_indices["validation_indices"], dtype=np.int64),
        "test": np.asarray(split_indices["test_indices"], dtype=np.int64),
    }
    calibrator: Optional[ProbabilityCalibrator] = None
    if calibration_method != "none":
        val_idx = split_map["validation"]
        val_logits = predict_ocscore_logits(
            bundle["model"],
            bundle["device"],
            feature_matrix[val_idx],
        )
        calibrator = ProbabilityCalibrator.fit(
            labels[val_idx],
            val_logits,
            method=calibration_method,
            scores_are_logits=True,
        )
    elif bundle.get("calibrator") is not None:
        calibrator = bundle["calibrator"]

    ocscore_logits_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}

    for split_name, row_idx in split_map.items():
        y_split = labels[row_idx]
        g_split = groups_all[row_idx]
        logits = predict_ocscore_logits(bundle["model"], bundle["device"], feature_matrix[row_idx])
        ocscore_logits_by_split[split_name] = logits
        labels_by_split[split_name] = y_split
        rows.append(
            evaluate_scorer(
                scorer_name=f"OCScore (trial {bundle['summary'].get('trial_number')})",
                scorer_type="model",
                split_name=split_name,
                labels=y_split,
                groups=g_split,
                scores=logits,
                higher_is_better=True,
                ocscore_logits=logits,
                calibrator=calibrator,
            )
        )

        for column in sf_columns:
            raw_scores = dudez_df[column].to_numpy(dtype=float)[row_idx]
            if float(np.mean(np.isfinite(raw_scores))) <= 0.0:
                continue
            rows.append(
                evaluate_scorer(
                    scorer_name=column,
                    scorer_type="sf",
                    split_name=split_name,
                    labels=y_split,
                    groups=g_split,
                    scores=raw_scores,
                    higher_is_better=infer_higher_is_better(raw_scores, y_split),
                )
            )

        split_features = feature_matrix[row_idx]
        for agg_name, agg_scores in row_aggregate_feature_scores(split_features).items():
            if float(np.mean(np.isfinite(agg_scores))) <= 0.0:
                continue
            rows.append(
                evaluate_scorer(
                    scorer_name=format_descriptor_aggregate_scorer(agg_name),
                    scorer_type=DESCRIPTOR_AGGREGATE_SCORER_TYPE,
                    split_name=split_name,
                    labels=y_split,
                    groups=g_split,
                    scores=agg_scores,
                    higher_is_better=infer_higher_is_better(agg_scores, y_split),
                )
            )

        for agg_name, agg_scores in row_aggregate_sf_scores(dudez_df, sf_columns, row_idx).items():
            if float(np.mean(np.isfinite(agg_scores))) <= 0.0:
                continue
            rows.append(
                evaluate_scorer(
                    scorer_name=format_sf_consensus_scorer(agg_name),
                    scorer_type=SF_CONSENSUS_SCORER_TYPE,
                    split_name=split_name,
                    labels=y_split,
                    groups=g_split,
                    scores=agg_scores,
                    higher_is_better=infer_higher_is_better(agg_scores, y_split),
                )
            )

    if include_export_summary:
        summary = bundle["summary"]
        for split_name, metrics_key in (("validation", "validation_metrics"), ("test", "test_metrics")):
            block = summary.get(metrics_key) or {}
            row = {
                "scorer": "OCScore (exported summary)",
                "scorer_type": "reference",
                "split": split_name,
                "higher_is_better": True,
                "nan_fraction": 0.0,
                "n_rows": float("nan"),
                "n_groups_used": block.get("n_groups_used"),
                "ranking_metrics_valid": block.get("ranking_metrics_valid"),
            }
            for metric_name in REPORT_METRICS:
                row[metric_name] = block.get(metric_name)
            rows.append(row)

    return pd.DataFrame(rows), calibrator, ocscore_logits_by_split, labels_by_split


def _print_test_ranking(table: pd.DataFrame) -> None:
    '''Print test-split summaries (full metrics are always in the CSV/JSON).'''

    test_rows = table[
        (table["split"] == "test")
        & (
            table["scorer_type"].isin(
                ["model", "sf", DESCRIPTOR_AGGREGATE_SCORER_TYPE, SF_CONSENSUS_SCORER_TYPE]
            )
        )
    ].copy()
    if test_rows.empty:
        return
    test_rows = test_rows.sort_values("BEDROC", ascending=False)

    ranking_columns = [
        column
        for column in ("scorer", "BEDROC", "ROC-AUC", "PR-AUC", "EF1%", "NDCG@1%")
        if column in test_rows.columns
    ]
    classification_columns = [
        column
        for column in (
            "scorer",
            "Precision",
            "Recall",
            "F1",
            "MCC",
            "TP",
            "FP",
            "TN",
            "FN",
        )
        if column in test_rows.columns
    ]
    meta_columns = [
        column
        for column in ("scorer", "higher_is_better", "ranking_metrics_valid", "n_groups_used")
        if column in test_rows.columns
    ]

    print("\n=== Test split — ranking (macro mean over receptors) ===\n")
    print(test_rows[ranking_columns].to_string(index=False))

    if len(classification_columns) > 1:
        print(
            "\n=== Test split — classification "
            "(Youden J threshold; TP/FP/TN/FN pooled over all test rows) ===\n"
        )
        print(test_rows[classification_columns].to_string(index=False))

    print(
        "\n=== Test split — score orientation & ranking validity ===\n"
        "higher_is_better: whether raw scores were treated as larger-is-active "
        "(False = inverted before metrics, typical for binding energies).\n"
        "ranking_metrics_valid: 1.0 when BEDROC/EF/NDCG are defined "
        "(both classes, non-constant scores per receptor).\n"
    )
    print(test_rows[meta_columns].to_string(index=False))

    calibration_columns = [
        column
        for column in (
            "scorer",
            "Brier",
            "Log-loss",
            "ECE",
            "Brier_calibrated",
            "Log-loss_calibrated",
            "ECE_calibrated",
        )
        if column in test_rows.columns
    ]
    model_rows = test_rows[test_rows["scorer_type"] == "model"]
    if len(calibration_columns) > 1 and not model_rows.empty:
        print("\n=== Test split — OCScore calibration (sigmoid vs Platt/isotonic on validation) ===\n")
        print(model_rows[calibration_columns].to_string(index=False))


## Public ##

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DUDEz OCScore vs individual scoring-function baselines on saved splits.",
    )
    parser.add_argument(
        "--reduction-archive",
        type=str,
        required=True,
        help="Feature-reduction tar archive or directory (reduced_dataset.csv + selected_features.json).",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help="Exported DUDEz best_model/ directory containing split_indices.npz.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (default: <export-dir>/../dudez_sf_baseline_comparison.csv).",
    )
    parser.add_argument("--group-column", type=str, default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--kind-column", type=str, default=DEFAULT_KIND_COLUMN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-export-summary-row",
        action="store_true",
        help="Do not append reference rows from best_trial_summary.json.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="If set, write baseline comparison PNG plots to this directory.",
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        choices=("none", "platt", "isotonic"),
        default=DEFAULT_CALIBRATION_METHOD,
        help="Post-hoc OCScore calibration fit on validation logits (default: platt).",
    )
    parser.add_argument(
        "--no-calibration-plots",
        action="store_true",
        help="Skip reliability diagrams even when --figures-dir is set.",
    )
    args = parser.parse_args()

    reduction_source = Path(args.reduction_archive)
    export_dir = Path(args.export_dir)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else export_dir.parent / "dudez_sf_baseline_comparison.csv"
    )

    dudez_df, selected_features = load_dudez_and_selected_features(reduction_source)
    table, calibrator, ocscore_logits_by_split, labels_by_split = build_comparison_table(
        dudez_df=dudez_df,
        selected_features=selected_features,
        export_dir=export_dir,
        group_column=args.group_column,
        kind_column=args.kind_column,
        device=args.device,
        include_export_summary=not args.no_export_summary_row,
        calibration_method=args.calibrate,
    )
    per_target_table = build_per_target_comparison_table(
        dudez_df=dudez_df,
        selected_features=selected_features,
        export_dir=export_dir,
        group_column=args.group_column,
        kind_column=args.kind_column,
        device=args.device,
        calibration_method=args.calibrate,
    )
    if calibrator is not None:
        calibrator_path = export_dir / ocexport.CALIBRATOR_FILENAME
        joblib.dump(calibrator, calibrator_path)
        print(f"Wrote calibrator {calibrator_path}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    per_target_csv = output_csv.with_name("dudez_sf_baseline_per_target.csv")
    per_target_table.to_csv(per_target_csv, index=False)
    per_target_json = per_target_csv.with_suffix(".json")
    per_target_json.write_text(per_target_table.to_json(orient="records", indent=2), encoding="utf-8")

    sf_columns = surviving_scoring_columns(selected_features)
    families = {
        "vina": [c for c in sf_columns if c.startswith("vina_")],
        "gnina": [c for c in sf_columns if c.startswith("gnina_")],
        "smina": [c for c in sf_columns if c.startswith("smina_")],
        "plants": [c for c in sf_columns if c.startswith("plants_")],
        "oddt": [c for c in sf_columns if c.startswith("oddt_")],
    }
    print(f"Scoring functions evaluated ({len(sf_columns)} total):")
    for family, columns in families.items():
        if columns:
            print(f"  {family} ({len(columns)}): {', '.join(columns)}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    print(f"Wrote {per_target_csv}")
    print(f"Wrote {per_target_json}")
    if args.figures_dir:
        figures_path = Path(args.figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        plot_paths = save_baseline_comparison_figures(
            output_csv,
            figures_dir=figures_path,
            split="test",
        )
        for label, path in plot_paths.items():
            print(f"Wrote plot {label}: {path}")
        if not per_target_table.empty:
            pt_plot_paths = save_per_target_figures(
                per_target_table,
                figures_path,
                split="test",
            )
            for label, path in pt_plot_paths.items():
                print(f"Wrote plot {label}: {path}")
            val_pt_paths = save_per_target_figures(
                per_target_table,
                figures_path,
                split="validation",
            )
            for label, path in val_pt_paths.items():
                print(f"Wrote plot {label}: {path}")
        if not args.no_calibration_plots and ocscore_logits_by_split:
            for split_name, logits in ocscore_logits_by_split.items():
                rel_paths = save_calibration_reliability_figures(
                    labels_by_split[split_name],
                    logits,
                    figures_path,
                    split=split_name,
                    calibrator=calibrator,
                )
                for label, path in rel_paths.items():
                    print(f"Wrote plot {label} ({split_name}): {path}")
    _print_test_ranking(table)


if __name__ == "__main__":
    main()
