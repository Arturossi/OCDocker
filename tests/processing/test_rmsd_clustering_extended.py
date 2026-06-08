#!/usr/bin/env python3

# Description
###############################################################################
'''
Additional fallback-path tests for Processing.Preprocessing.RMSDClustering.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.Error as ocerror
import OCDocker.Processing.Preprocessing.RMSDClustering as ocrmsdclust

# License
###############################################################################
'''
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
'''

# Classes
###############################################################################


class _TwoStepAgglomerative:
    outputs = []

    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        _ = data
        if _TwoStepAgglomerative.outputs:
            return np.asarray(_TwoStepAgglomerative.outputs.pop(0), dtype=int)
        return np.array([], dtype=int)


class _AlwaysUniqueAgglomerative:
    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        return np.arange(len(data), dtype=int)


class _SingleClusterAgglomerative:
    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        return np.zeros(len(data), dtype=int)


# Functions
###############################################################################
## Private ##


def _matrix_df(size: int) -> pd.DataFrame:
    labels = [f"pose{i}" for i in range(size)]
    base = np.abs(np.subtract.outer(np.arange(size), np.arange(size))).astype(float)
    return pd.DataFrame(base, index=labels, columns=labels)


## Public ##


@pytest.mark.order(365)
def test_cluster_rmsd_uses_last_result_when_unique_clusters_appear_after_tie(monkeypatch):
    _TwoStepAgglomerative.outputs = [
        np.array([0, 0, 1, 1], dtype=int),
        np.array([0, 1, 2, 3], dtype=int),
    ]
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _TwoStepAgglomerative)

    clusters = ocrmsdclust.cluster_rmsd(
        _matrix_df(4),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot="",
    )

    assert isinstance(clusters, np.ndarray)
    assert clusters.tolist() == [0, 0, 1, 1]


@pytest.mark.order(366)
def test_cluster_rmsd_non_converged_plot_exception_logs_warning(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _AlwaysUniqueAgglomerative)
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(
        ocrmsdclust.plt,
        "subplots",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced subplots failure")),
    )

    out_plot = tmp_path / "non_converged_plot.png"
    rc = ocrmsdclust.cluster_rmsd(
        _matrix_df(3),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot=str(out_plot),
    )

    assert rc == ocerror.ErrorCode.CLUSTER_NOT_CONVERGED
    assert any("Failed to generate plot for non-converged clustering" in msg for msg in warnings)


@pytest.mark.order(367)
def test_cluster_rmsd_accepts_dict_input_and_returns_clusters(monkeypatch):
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SingleClusterAgglomerative)
    data_dict = {
        "poseA": {"poseA": 0.0, "poseB": 1.0},
        "poseB": {"poseA": 1.0, "poseB": 0.0},
    }
    clusters = ocrmsdclust.cluster_rmsd(
        data_dict,  # type: ignore[arg-type]
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot="",
    )
    assert isinstance(clusters, np.ndarray)
    assert clusters.tolist() == [0, 0]


@pytest.mark.order(411)
def test_get_medoids_with_int_clusters_returns_empty_list():
    medoids = ocrmsdclust.get_medoids(_matrix_df(2), 7)  # type: ignore[arg-type]
    assert medoids == []


@pytest.mark.order(412)
def test_cluster_rmsd_pose_engine_partial_path_mapping_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SingleClusterAgglomerative)

    df = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["poseA", "poseB"],
        columns=["poseA", "poseB"],
    )
    output_plot = tmp_path / "engine_partial_map.png"
    clusters = ocrmsdclust.cluster_rmsd(
        df,
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot=str(output_plot),
        pose_engine_map={
            "/tmp/vina/poseA.pdbqt": "vina",
            "/tmp/smina/poseB.pdbqt": "smina",
        },
    )

    assert isinstance(clusters, np.ndarray)
    assert clusters.tolist() == [0, 0]
    assert output_plot.exists()


@pytest.mark.order(413)
def test_cluster_rmsd_gnina_mol2_engine_map_colors_plot(monkeypatch, tmp_path):
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SingleClusterAgglomerative)

    df = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["/tmp/gnina_ligand_1.mol2", "/tmp/plants_ligand_2.mol2"],
        columns=["/tmp/gnina_ligand_1.mol2", "/tmp/plants_ligand_2.mol2"],
    )
    output_plot = tmp_path / "gnina_plants.png"
    clusters = ocrmsdclust.cluster_rmsd(
        df,
        max_distance_threshold=5.0,
        min_distance_threshold=1.0,
        threshold_step=1.0,
        outputPlot=str(output_plot),
        pose_engine_map=ocrmsdclust.build_pose_engine_map(list(df.index)),
    )

    assert isinstance(clusters, np.ndarray)
    assert output_plot.exists()
    assert ocrmsdclust.DEFAULT_ENGINE_COLORS["gnina"] == "#785EF0"
    assert ocrmsdclust.DEFAULT_ENGINE_COLORS["plants"] == "#009E73"
    assert ocrmsdclust.DEFAULT_ENGINE_COLORS["smina"] == "#E69F00"
    assert ocrmsdclust.DEFAULT_ENGINE_COLORS["vina"] == "#0072B2"
