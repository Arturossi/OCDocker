#!/usr/bin/env python3

# Description
###############################################################################
'''Explicit DUDEz feature scaling policies for staged OCScore transfer.

PDBbind regression fits a train-only ``StandardScaler``. DUDEz screening must
apply a documented scaling strategy so transferred encoders see compatible inputs.
'''

# Imports
###############################################################################
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

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

DUDEzScalingStrategy = Literal[
    "pdbbind_scaler", "dudez_train_scaler", "none_prestandardized"
]
DUDEZ_SCALING_STRATEGIES: tuple[DUDEzScalingStrategy, ...] = (
    "pdbbind_scaler",
    "dudez_train_scaler",
    "none_prestandardized",
)


@dataclass
class DUDEzScalingConfig:
    """Configuration for DUDEz feature scaling during staged screening.

    Parameters
    ----------
    strategy : str
        Scaling strategy name.
    strict : bool, optional
        When True, forbid implicit unscaled features during transfer, by default True.
    """

    strategy: DUDEzScalingStrategy = "pdbbind_scaler"
    strict: bool = True


def _validate_feature_order(
    scaler: StandardScaler,
    selected_features: Sequence[str],
) -> None:
    '''Ensure scaler feature names match selected features when available.'''

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is None:
        if scaler.n_features_in_ != len(selected_features):
            raise ValueError(
                f"Scaler expects {scaler.n_features_in_} features but got {len(selected_features)}."
            )
        return
    expected = [str(name) for name in scaler_features]
    actual = [str(name) for name in selected_features]
    if expected != actual:
        raise ValueError(
            "Feature order mismatch between scaler and selected features. "
            f"Scaler order (first 5): {expected[:5]}; selected (first 5): {actual[:5]}"
        )


def _transform_with_feature_names(
    scaler: StandardScaler,
    X: np.ndarray,
    selected_features: Sequence[str],
) -> np.ndarray:
    '''Transform arrays as named columns when the fitted scaler expects names.'''

    if getattr(scaler, "feature_names_in_", None) is None:
        return scaler.transform(X)
    import pandas as pd

    return scaler.transform(
        pd.DataFrame(X, columns=[str(name) for name in selected_features])
    )


def scale_dudez_features(
    X: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: DUDEzScalingConfig,
    selected_features: Sequence[str],
    pdbbind_scaler: Optional[StandardScaler] = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Optional[StandardScaler]
]:
    '''Scale DUDEz feature matrices according to ``config``.

    Parameters
    ----------
    X : np.ndarray
        Full DUDEz feature matrix before splitting.
    train_idx : np.ndarray
        Training row indices.
    val_idx : np.ndarray
        Validation row indices.
    test_idx : np.ndarray
        Test row indices.
    config : DUDEzScalingConfig
        Scaling policy.
    selected_features : Sequence[str]
        Feature column order used for modeling.
    pdbbind_scaler : StandardScaler | None, optional
        Train-fitted PDBbind scaler from the regression stage.

    Returns
    -------
    tuple
        Scaled train/val/test arrays, metadata dict, and optional fitted DUDEz scaler.
    '''

    strategy = config.strategy
    metadata: dict[str, Any] = {
        "strategy": strategy,
        "strict": bool(config.strict),
        "feature_names": [str(name) for name in selected_features],
        "n_features": len(selected_features),
        "pdbbind_scaler_reused": False,
        "dudez_scaler_fit_scope": None,
    }

    if strategy == "none_prestandardized":
        metadata["notes"] = [
            "Caller asserts DUDEz features were pre-standardized upstream; no scaler applied."
        ]
        return (
            X[train_idx].astype(np.float32),
            X[val_idx].astype(np.float32),
            X[test_idx].astype(np.float32),
            metadata,
            None,
        )

    if strategy == "pdbbind_scaler":
        if pdbbind_scaler is None:
            raise ValueError(
                "pdbbind_scaler strategy requires a fitted PDBbind StandardScaler."
            )
        _validate_feature_order(pdbbind_scaler, selected_features)
        metadata["pdbbind_scaler_reused"] = True
        metadata["scaler_fit_scope"] = "pdbbind_train"
        return (
            _transform_with_feature_names(
                pdbbind_scaler, X[train_idx], selected_features
            ).astype(np.float32),
            _transform_with_feature_names(
                pdbbind_scaler, X[val_idx], selected_features
            ).astype(np.float32),
            _transform_with_feature_names(
                pdbbind_scaler, X[test_idx], selected_features
            ).astype(np.float32),
            metadata,
            None,
        )

    if strategy == "dudez_train_scaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_val = scaler.transform(X[val_idx]).astype(np.float32)
        X_test = scaler.transform(X[test_idx]).astype(np.float32)
        metadata["dudez_scaler_fit_scope"] = "dudez_train"
        metadata["scaler_fit_scope"] = "dudez_train"
        return X_train, X_val, X_test, metadata, scaler

    raise ValueError(f"Unsupported DUDEz scaling strategy {strategy!r}.")


def scaling_config_to_dict(config: DUDEzScalingConfig) -> dict[str, Any]:
    '''Serialize scaling config for JSON artifacts.'''

    return asdict(config)


__all__ = [
    "DUDEZ_SCALING_STRATEGIES",
    "DUDEzScalingConfig",
    "DUDEzScalingStrategy",
    "scale_dudez_features",
    "scaling_config_to_dict",
]
