#!/usr/bin/env python3

# Description
###############################################################################
'''
Data loading and preparation for SHAP analysis.

Usage:

from OCDocker.OCScore.Analysis.SHAP.Data import load_and_prepare_data
'''

# Imports
###############################################################################
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List

import OCDocker.OCScore.Utils.Data as ocscoredata

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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

LOGGER = logging.getLogger("OCScore.SHAP.data")

# Classes
###############################################################################
@dataclass
class DataHandles:
    """Data container for SHAP analysis datasets.
    
    Attributes
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    y_val : np.ndarray
        Validation target values.
    feature_names : List[str]
        List of feature column names.
    """
    
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_val: np.ndarray
    feature_names: List[str]

    
# Functions
###############################################################################
## Private ##

## Public ##

def load_and_prepare_data(
    df_path: str,
    base_models_folder: str,
    study_number: int,
    use_pca: bool = False,
    use_pdb_train: bool = True,
    random_seed: int = 42,
) -> DataHandles:
    '''Load and prepare datasets for SHAP analysis.
    
    Parameters
    ----------
    df_path : str
        Path to the main dataframe file.
    base_models_folder : str
        Base path to the models folder.
    study_number : int
        Study number identifier.
    use_pca : bool, optional
        Whether to use PCA-transformed features. Default is False.
    use_pdb_train: bool, optional
        Whether to use PDBbind training data. Default is True.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    DataHandles
        Container with train/val/test feature matrices, validation targets, and feature names.
    '''
    
    LOGGER.info("Loading dataframes and preprocessed features")
    _df_dudez, _df_pdbbind, score_columns = ocscoredata.preprocess_df(df_path)

    data = ocscoredata.load_data(
        base_models_folder=base_models_folder,
        storage_id=study_number,
        df_path=df_path,
        optimization_type="NN",
        no_scores=False,
        only_scores=False,
        use_PCA=use_pca,
        use_pdb_train=use_pdb_train,
        random_seed=random_seed,
    )

    X_train = ocscoredata.invert_values_conditionally(data['X_train'])
    X_test = ocscoredata.invert_values_conditionally(data['X_test'])
    X_val = ocscoredata.invert_values_conditionally(data['X_val'])
    y_val = data['y_val'].values

    feature_names = list(X_train.columns)
    return DataHandles(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_val=y_val,
        feature_names=feature_names,
    )
