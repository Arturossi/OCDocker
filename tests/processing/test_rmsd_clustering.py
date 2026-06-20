#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.Preprocessing.RMSDClustering.
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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


class _AlwaysUniqueAgglomerative:
    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        return np.arange(len(data))


class _SequencedAgglomerative:
    outputs = []

    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        if _SequencedAgglomerative.outputs:
            nxt = _SequencedAgglomerative.outputs.pop(0)
            return np.asarray(nxt, dtype=int)
        return np.arange(len(data), dtype=int)


# Functions
###############################################################################
## Private ##

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]],
        index=["poseA", "poseB", "poseC"],
        columns=["poseA", "poseB", "poseC"],
    )


def _matrix_df(size: int) -> pd.DataFrame:
    labels = [f"pose{i}" for i in range(size)]
    base = np.abs(np.subtract.outer(np.arange(size), np.arange(size))).astype(float)
    return pd.DataFrame(base, index=labels, columns=labels)


## Public ##

@pytest.mark.order(144)
def test_cluster_rmsd_invalid_threshold_order_returns_value_error():
    rc = ocrmsdclust.cluster_rmsd(
        _sample_df(),
        max_distance_threshold=1.0,
        min_distance_threshold=2.0,
    )
    assert rc == ocerror.ErrorCode.VALUE_ERROR


@pytest.mark.order(145)
def test_build_pose_engine_map_from_mol2_filenames():
    mol2_paths = [
        "/tmp/poses/gnina_ligand_split_1.mol2",
        "/tmp/poses/vina_ligand_split_2.mol2",
    ]
    engine_map = ocrmsdclust.build_pose_engine_map(mol2_paths)

    assert engine_map == {
        "/tmp/poses/gnina_ligand_split_1.mol2": "gnina",
        "/tmp/poses/vina_ligand_split_2.mol2": "vina",
    }


@pytest.mark.order(146)
def test_build_pose_engine_map_uses_original_pose_paths():
    mol2_paths = ["/tmp/poses/smina_ligand.mol2"]
    source_map = {"/dock/smina/ligand.pdbqt": "smina"}
    aliases = {"/tmp/poses/smina_ligand.mol2": "/dock/smina/ligand.pdbqt"}

    engine_map = ocrmsdclust.build_pose_engine_map(mol2_paths, source_map, aliases)

    assert engine_map == {"/tmp/poses/smina_ligand.mol2": "smina"}


@pytest.mark.order(147)
def test_cluster_rmsd_unsupported_algorithm_returns_error():
    rc = ocrmsdclust.cluster_rmsd(_sample_df(), algorithm="unsupported")
    assert rc == ocerror.ErrorCode.UNSUPPORTED_CLUSTERING_ALGORITHM


@pytest.mark.order(146)
def test_cluster_rmsd_single_row_returns_single_cluster(monkeypatch):
    warnings = []
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda message: warnings.append(message))
    data = pd.DataFrame([[0.0]], index=["only"], columns=["only"])
    clusters = ocrmsdclust.cluster_rmsd(data)
    assert isinstance(clusters, np.ndarray)
    assert clusters.tolist() == [0.0]
    assert warnings


@pytest.mark.order(147)
def test_cluster_rmsd_non_converged_path_returns_cluster_not_converged(monkeypatch):
    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _AlwaysUniqueAgglomerative)
    rc = ocrmsdclust.cluster_rmsd(
        _sample_df(),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot="",
    )
    assert rc == ocerror.ErrorCode.CLUSTER_NOT_CONVERGED


@pytest.mark.order(148)
def test_get_medoids_handles_invalid_clusters_and_returns_expected_representatives():
    data = _sample_df()
    assert ocrmsdclust.get_medoids(data, np.array([], dtype=int)) == []
    assert ocrmsdclust.get_medoids(data, np.array([-1, 0, 0])) == []

    clusters = np.array([0, 0, 1])
    biggest = ocrmsdclust.get_medoids(data, clusters, onlyBiggest=True)
    assert len(biggest) == 1
    assert biggest[0] in {"poseA", "poseB"}

    all_medoids = ocrmsdclust.get_medoids(data, clusters, onlyBiggest=False)
    assert len(all_medoids) == 2
    assert set(all_medoids).issubset({"poseA", "poseB", "poseC"})


@pytest.mark.order(149)
def test_get_medoids_accepts_dict_input():
    data_dict = {
        "poseA": {"poseA": 0.0, "poseB": 1.0},
        "poseB": {"poseA": 1.0, "poseB": 0.0},
    }
    medoids = ocrmsdclust.get_medoids(data_dict, np.array([0, 0]), onlyBiggest=True)
    assert medoids == ["poseA"]


@pytest.mark.order(182)
def test_cluster_rmsd_generates_consensus_plot_and_labels_file(tmp_path):
    output_plot = tmp_path / "consensus.png"
    clusters = ocrmsdclust.cluster_rmsd(
        _sample_df(),
        max_distance_threshold=5.0,
        min_distance_threshold=0.5,
        threshold_step=0.5,
        outputPlot=str(output_plot),
        molecule_name="LigandX",
        pose_engine_map={"poseA": "vina", "poseB": "smina", "poseC": "plants"},
        engine_colors={"vina": "red", "smina": "blue", "plants": "green"},
    )

    labels_file = tmp_path / "consensus_labels.txt"
    assert isinstance(clusters, np.ndarray)
    assert clusters.shape[0] == 3
    assert output_plot.exists()
    assert labels_file.exists()
    assert "Representative" in labels_file.read_text(encoding="utf-8")


@pytest.mark.order(183)
def test_cluster_rmsd_uses_fallback_plot_when_main_plot_generation_fails(monkeypatch, tmp_path):
    output_plot = tmp_path / "fallback.png"
    warnings = []
    original_dendrogram = ocrmsdclust.sch.dendrogram
    calls = {"count": 0}

    def _flaky_dendrogram(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("forced plotting error")
        return original_dendrogram(*args, **kwargs)

    monkeypatch.setattr(ocrmsdclust.sch, "dendrogram", _flaky_dendrogram)
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda msg: warnings.append(msg))

    clusters = ocrmsdclust.cluster_rmsd(
        _sample_df(),
        max_distance_threshold=5.0,
        min_distance_threshold=0.5,
        threshold_step=0.5,
        outputPlot=str(output_plot),
    )

    assert isinstance(clusters, np.ndarray)
    assert output_plot.exists()
    assert calls["count"] >= 2
    assert any("Generated fallback plot" in msg for msg in warnings)


@pytest.mark.order(247)
def test_cluster_rmsd_non_converged_with_output_plot_generates_plot(monkeypatch, tmp_path):
    output_plot = tmp_path / "non_converged.png"
    warnings = []

    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _AlwaysUniqueAgglomerative)
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocrmsdclust.cluster_rmsd(
        _sample_df(),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot=str(output_plot),
        molecule_name="LigY",
    )
    assert rc == ocerror.ErrorCode.CLUSTER_NOT_CONVERGED
    assert output_plot.exists()
    assert any("Generated plot for failed clustering" in msg for msg in warnings)


@pytest.mark.order(248)
def test_cluster_rmsd_uses_last_result_fallback_cluster_when_loop_does_not_converge(monkeypatch):
    warnings = []
    _SequencedAgglomerative.outputs = [
        np.array([0, 0, 1, 1], dtype=int),
        np.array([0, 0, 1, 1], dtype=int),
    ]

    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SequencedAgglomerative)
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda msg: warnings.append(msg))

    clusters = ocrmsdclust.cluster_rmsd(
        _matrix_df(4),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot="",
    )
    assert isinstance(clusters, np.ndarray)
    assert clusters.tolist() == [0, 0, 1, 1]
    assert any("did not fully converge" in msg for msg in warnings)


@pytest.mark.order(249)
def test_cluster_rmsd_plotting_branch_for_up_to_twelve_clusters(monkeypatch, tmp_path):
    output_plot = tmp_path / "clusters_up_to_twelve.png"
    _SequencedAgglomerative.outputs = [
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
        np.arange(9, dtype=int),
    ]

    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SequencedAgglomerative)
    monkeypatch.setattr(ocrmsdclust, "silhouette_score", lambda *_a, **_k: 0.42)

    clusters = ocrmsdclust.cluster_rmsd(
        _matrix_df(9),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot=str(output_plot),
        molecule_name="LigZ",
    )
    assert isinstance(clusters, np.ndarray)
    assert clusters.shape[0] == 9
    assert output_plot.exists()


@pytest.mark.order(250)
def test_cluster_rmsd_plotting_branch_for_more_than_twelve_clusters_and_leaf_warning(monkeypatch, tmp_path):
    output_plot = tmp_path / "clusters_over_twelve.png"
    warnings = []
    _SequencedAgglomerative.outputs = [
        np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int),
        np.arange(13, dtype=int),
    ]

    original_dendrogram = ocrmsdclust.sch.dendrogram

    def _dendrogram_missing_leaf(*args, **kwargs):
        out = original_dendrogram(*args, **kwargs)
        if out.get("leaves"):
            out["leaves"] = out["leaves"][:-1]
        return out

    monkeypatch.setattr(ocrmsdclust, "AgglomerativeClustering", _SequencedAgglomerative)
    monkeypatch.setattr(ocrmsdclust, "silhouette_score", lambda *_a, **_k: 0.51)
    monkeypatch.setattr(ocrmsdclust.sch, "dendrogram", _dendrogram_missing_leaf)
    monkeypatch.setattr(ocrmsdclust.ocprint, "print_warning", lambda msg: warnings.append(msg))

    clusters = ocrmsdclust.cluster_rmsd(
        _matrix_df(13),
        max_distance_threshold=2.0,
        min_distance_threshold=1.0,
        threshold_step=0.5,
        outputPlot=str(output_plot),
    )
    assert isinstance(clusters, np.ndarray)
    assert clusters.shape[0] == 13
    assert output_plot.exists()
    assert any("Dendrogram shows" in msg for msg in warnings)
