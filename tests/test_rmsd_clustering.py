#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.Preprocessing.RmsdClustering.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd
import pytest

import OCDocker.Error as ocerror
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust

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


class _AlwaysUniqueAgglomerative:
    def __init__(self, n_clusters=None, distance_threshold=None):
        _ = (n_clusters, distance_threshold)

    def fit_predict(self, data):
        return np.arange(len(data))


# Functions
###############################################################################
## Private ##

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]],
        index=["poseA", "poseB", "poseC"],
        columns=["poseA", "poseB", "poseC"],
    )


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
