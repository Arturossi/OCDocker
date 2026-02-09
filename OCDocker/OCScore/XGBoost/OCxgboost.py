#!/usr/bin/env python3

# Description
###############################################################################
''' Module to run the Extreme Gradient Boost algorithm. 

It is imported as:

import OCDocker.OCScore.XGBoost.OCxgboost as OCxgboost
'''

# Imports
###############################################################################

import numpy as np

from typing import Any, Optional

from xgboost import XGBRegressor

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


# Functions
###############################################################################
## Private ##

## Public ##

def run_xgboost(
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_test : np.ndarray,
        y_test : np.ndarray,
        params : Optional[dict[str, Any]] = None,
        verbose : bool = False
    ) -> tuple[XGBRegressor, float]:
    '''
    A function to train an XGBoost model and calculate the AUC score.

    Parameters
    ----------
    X_train : np.ndarray
        The training dataset.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The test dataset.
    y_test : np.ndarray
        The test labels.
    params : dict, optional
        The hyperparameters for the XGBoost model. Default is None (treated as an empty dictionary).
    verbose : bool, optional
        Whether to print the training logs. Default is False.

    Returns
    -------
    model : XGBRegressor
        The trained XGBoost model.
    roc_auc : float
        The AUC score of the trained model.
    '''

    if params is None:
        params = {}

    # Create the XGBoost model
    model = XGBRegressor(**params)

    # Train the model
    model.fit(
        X_train, 
        y_train, 
        eval_set = [(X_test, y_test)],
        verbose = verbose
    )

    # Get the AUC score
    evals_result = model.evals_result()
    metric = evals_result["validation_0"][params["eval_metric"].lower()][-1]

    # Return the trained model and the metric
    return model, metric
