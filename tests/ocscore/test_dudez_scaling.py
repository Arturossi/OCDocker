#!/usr/bin/env python3

# Description
###############################################################################
'''Tests for DUDEz scaling policies in staged OCScore.'''

# Imports
###############################################################################
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from OCDocker.OCScore.Utils.DUDEzScaling import DUDEzScalingConfig
from OCDocker.OCScore.Utils.DUDEzScaling import scale_dudez_features

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


def _toy_matrix(n: int = 30, n_features: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, n_features)).astype(np.float32)


@pytest.mark.order(420)
def test_default_scaling_requires_pdbbind_scaler():
    X = _toy_matrix()
    features = [f"f{i}" for i in range(X.shape[1])]
    with pytest.raises(ValueError, match="pdbbind_scaler"):
        scale_dudez_features(
            X,
            train_idx=np.arange(0, 18),
            val_idx=np.arange(18, 24),
            test_idx=np.arange(24, 30),
            config=DUDEzScalingConfig(),
            selected_features=features,
        )


@pytest.mark.order(421)
def test_pdbbind_scaler_reused_on_dudez_splits():
    X = _toy_matrix()
    features = [f"f{i}" for i in range(X.shape[1])]
    train_idx = np.arange(0, 18)
    val_idx = np.arange(18, 24)
    test_idx = np.arange(24, 30)
    pdbbind_scaler = StandardScaler()
    pdbbind_scaler.fit(X[train_idx])
    pdbbind_scaler.feature_names_in_ = np.array(features, dtype=object)

    X_train, X_val, X_test, metadata, dudez_scaler = scale_dudez_features(
        X,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=DUDEzScalingConfig(strategy="pdbbind_scaler", strict=True),
        selected_features=features,
        pdbbind_scaler=pdbbind_scaler,
    )
    assert dudez_scaler is None
    assert metadata["pdbbind_scaler_reused"] is True
    expected = pdbbind_scaler.transform(X[train_idx])
    np.testing.assert_allclose(X_train, expected, rtol=1e-5, atol=1e-5)
@pytest.mark.order(422)
def test_dudez_train_scaler_fits_train_only():
    X = _toy_matrix()
    train_idx = np.arange(0, 18)
    val_idx = np.arange(18, 24)
    test_idx = np.arange(24, 30)
    features = [f"f{i}" for i in range(X.shape[1])]
    _, _, _, metadata, scaler = scale_dudez_features(
        X,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=DUDEzScalingConfig(strategy="dudez_train_scaler", strict=True),
        selected_features=features,
    )
    assert scaler is not None
    assert metadata["dudez_scaler_fit_scope"] == "dudez_train"
