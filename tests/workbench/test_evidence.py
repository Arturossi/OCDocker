#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench OCScore evidence discovery.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_evidence_index
from OCDocker.Workbench import resolve_evidence_asset
from OCDocker.Workbench import write_model

# License
###############################################################################
"""OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Functions
###############################################################################
## Private ##


def _write_evidence_workspace(tmp_path):
    '''Write an adopted Workbench run with OCScore evidence files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.

    Returns
    -------
    pathlib.Path
        Workbench root.
    '''

    source_root = tmp_path / "source" / "output"
    train_source = source_root / "train"
    train_source.mkdir(parents=True)
    (train_source / "baselines_per_fold.csv").write_text(
        "baseline,baseline_family,split,BEDROC,ROC-AUC,replica\n"
        "vina,scoring_function,validation,0.25,0.70,replica_000\n"
        "vina,scoring_function,validation,0.35,0.80,replica_001\n",
        encoding="utf-8",
    )
    optuna_dir = train_source / "replica_001" / "dudez"
    optuna_dir.mkdir(parents=True)
    (optuna_dir / "dudez_optuna_trials.csv").write_text(
        "number,value,state\n0,0.50,COMPLETE\n1,0.60,COMPLETE\n2,0.58,COMPLETE\n",
        encoding="utf-8",
    )
    shap_dir = source_root / "export" / "dudez" / "shap"
    shap_dir.mkdir(parents=True)
    (shap_dir / "shap_values.csv").write_text(
        "feature_a,feature_b\n1.0,-2.0\n3.0,4.0\n",
        encoding="utf-8",
    )
    (shap_dir / "shap_feature_importance.png").write_bytes(b"png")

    workspace = tmp_path / "runs"
    run_dir = workspace / "train"
    run_dir.mkdir(parents=True)
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="train",
            spec_type="ocscore_study",
            name="train",
            status="completed",
            workspace=train_source,
            metadata={"adopted": True, "source_path": str(train_source)},
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(run_id="train", status="completed", metrics={"auc": 0.9}),
    )
    return workspace


## Public ##


def test_build_evidence_index_discovers_performance_optuna_and_shap(tmp_path) -> None:
    '''Evidence discovery parses known OCScore outputs without executing jobs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    workspace = _write_evidence_workspace(tmp_path)

    evidence = build_evidence_index(workspace, max_depth=2, source_depth=4, max_csv_rows=20)

    assert evidence.result_manifest_count == 1
    assert evidence.kind_counts["performance"] == 1
    assert evidence.kind_counts["optimization"] == 1
    assert evidence.kind_counts["shap"] == 2
    assert evidence.performance_points
    assert any(point["metric_name"] == "BEDROC" and point["value"] == 0.3 for point in evidence.performance_points)
    assert [point["best_value"] for point in evidence.optimization_points] == [0.5, 0.6, 0.6]
    assert evidence.shap_features[0]["feature"] == "feature_b"
    assert evidence.shap_features[0]["mean_abs_shap"] == 3.0


def test_resolve_evidence_asset_allows_discovered_image(tmp_path) -> None:
    '''Evidence asset resolution only exposes discovered image evidence files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    workspace = _write_evidence_workspace(tmp_path)
    image_path = tmp_path / "source" / "output" / "export" / "dudez" / "shap" / "shap_feature_importance.png"
    csv_path = image_path.parent / "shap_values.csv"
    outside_path = tmp_path / "outside.png"
    outside_path.write_bytes(b"png")

    resolved, content_type = resolve_evidence_asset(workspace, image_path, max_depth=2, source_depth=4)

    assert resolved == image_path
    assert content_type == "image/png"
    with pytest.raises(ValueError, match="supported image file"):
        resolve_evidence_asset(workspace, csv_path, max_depth=2, source_depth=4)
    with pytest.raises(ValueError, match="not part of a discovered image evidence source"):
        resolve_evidence_asset(workspace, outside_path, max_depth=2, source_depth=4)
