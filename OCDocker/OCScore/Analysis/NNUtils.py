#!/usr/bin/env python3

# Description
###############################################################################
'''
This module provides utility functions for analyzing autoencoder (AE)-based
feature importance and neural network representations, including permutation
importance evaluation.

Usage:

import OCDocker.OCScore.Analysis.NNUtils as ocnnutils
'''

# Imports
###############################################################################

import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# Methods
###############################################################################

def run_ae_feature_importance(
    ae_model: Any,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    features: list[str],
    n_repeats: int = 30,
    save_dir: str = "plots",
    prefix: str = "AE"
) -> pd.DataFrame:
    '''
    Run permutation-based feature importance on an AE + NN pipeline.

    Parameters
    ----------
    ae_model : torch.nn.Module
        Trained model that outputs predictions from latent representations.
    X_valid : np.ndarray
        Validation feature matrix (already encoded if AE is separate).
    y_valid : np.ndarray
        Ground-truth RMSE or AUC values.
    features : list[str]
        Names of original features.
    n_repeats : int
        Number of permutation repetitions.
    save_dir : str
        Directory to save the barplot output.
    prefix : str
        Filename prefix for saved figure.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with feature importances.
    '''
    
    print("Running AE-based permutation importance...")

    if not hasattr(ae_model, 'predict'):
        raise ValueError("Model must implement a 'predict' method")


    def score_fn(X: np.ndarray) -> float:
        ''' Score function for the permutation importance.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        float
            The score.
        '''

        # Convert the input data to a tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = ae_model.predict(X_tensor).detach().numpy()
        return -mean_squared_error(y_valid, preds)

    result = permutation_importance(
        estimator=None,
        X=X_valid,
        y=y_valid,
        scoring=score_fn,
        n_repeats=n_repeats,
        random_state=42
    )

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title(f'Permutation Importance ({prefix})')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_permutation_importance.png'))
    plt.close()

    print("AE feature importance analysis completed.")
    return importance_df
