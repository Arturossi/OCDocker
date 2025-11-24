#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage scoring and prediction in OCDocker in the context of
scoring functions.

They are imported as:

import OCDocker.OCScore.Scoring as ocscoring
'''

# Imports
###############################################################################

import os
from typing import Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import OCDocker.OCScore.Utils.IO as ocscoreio
import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.Error as ocerror

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################


def get_score(
    model_path: str,
    data: Optional[Union[pd.DataFrame, str]] = None,
    pca_model: Optional[Union[str, PCA]] = None,
    mask: Optional[Union[list, np.ndarray]] = None,
    score_columns_list: list[str] = ["SMINA", "VINA", "ODDT", "PLANTS"],
    scaler: str = "standard",
    invert_conditionally: bool = True,
    normalize: bool = True,
    no_scores: bool = False,
    only_scores: bool = False,
    columns_to_skip_pca: Optional[list[str]] = None,
    serialization_method: str = "joblib"
) -> Union[pd.DataFrame, np.ndarray]:
    ''' Get scores by loading a model and applying the same preprocessing pipeline.
    
    This function loads a trained model and applies it to input data following
    the same preprocessing pipeline used during training. The data can be provided
    as a DataFrame or read from a database.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    data : pd.DataFrame | str, optional
        Input data. Can be:
        - A pandas DataFrame with the features
        - A string path to a CSV file
        - None to read from database (requires DB setup)
        Default is None.
    pca_model : str | PCA, optional
        Path to the PCA model file or a PCA model object. If provided, PCA
        transformation will be applied. If None, no PCA is used.
        Default is None.
    mask : list | np.ndarray, optional
        Feature mask array of 0s and 1s to filter features before prediction.
        Length should match the number of features after preprocessing.
        1 means keep the feature, 0 means remove it.
        Default is None (no masking applied).
    score_columns_list : list[str], optional
        List of score column prefixes to identify score columns. 
        Default is ["SMINA", "VINA", "ODDT", "PLANTS"].
    scaler : str, optional
        Scaler to use for normalization. Options are "standard" or "minmax".
        Default is "standard".
    invert_conditionally : bool, optional
        Whether to invert values conditionally (for VINA, SMINA, PLANTS columns).
        Default is True.
    normalize : bool, optional
        Whether to normalize the data. Default is True.
    no_scores : bool, optional
        If True, remove score columns from the data. Default is False.
    only_scores : bool, optional
        If True, keep only score columns and metadata. Default is False.
    columns_to_skip_pca : list[str], optional
        List of columns to skip during PCA transformation. If None, defaults to
        metadata columns: ["receptor", "ligand", "name", "type", "db"].
        Default is None.
    serialization_method : str, optional
        Serialization method used to save the model. Options are "joblib" or "pickle".
        Default is "joblib".
    
    Returns
    -------
    pd.DataFrame | np.ndarray
        Predicted scores. Returns a DataFrame if input was a DataFrame (preserving
        metadata columns), otherwise returns a numpy array.
    
    Raises
    ------
    FileNotFoundError
        If the model file or PCA model file is not found.
    ValueError
        If data is None and database is not available, or if invalid parameters are provided.
    '''
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        ocerror.Error.file_not_exist(f"Model file not found: {model_path}") # type: ignore
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model - IO module now handles format detection automatically
    model = ocscoreio.load_object(model_path, serialization_method="auto")
    
    # Set PyTorch models to eval mode for inference
    if hasattr(model, 'eval'):
        model.eval()
    
    # Load or prepare the data
    if data is None:
        # Try to read from database
        try:
            import OCDocker.Initialise as init
            from OCDocker.DB.Models.Complexes import Complexes
            
            # Check if session is available
            if not hasattr(init, 'session') or init.session is None:
                ocerror.Error.session_not_created("Database session not available. Please provide data or initialize the database.") # type: ignore
                raise ValueError("Database session not available. Please provide data or initialize the database.")
            
            # Read from database
            with init.session() as s:
                # Query all complexes
                complexes = s.query(Complexes).all()
                
                # Convert to DataFrame
                data_list = []
                for complex_obj in complexes:
                    row = {}
                    # Get all descriptor columns
                    for desc in Complexes.allDescriptors:
                        desc_attr = desc.lower()
                        if hasattr(complex_obj, desc_attr):
                            value = getattr(complex_obj, desc_attr)
                            # Only add non-None values
                            if value is not None:
                                row[desc] = value
                    # Add metadata if available
                    if hasattr(complex_obj, 'ligand') and complex_obj.ligand:
                        if hasattr(complex_obj.ligand, 'name'):
                            row['ligand'] = complex_obj.ligand.name
                    if hasattr(complex_obj, 'receptor') and complex_obj.receptor:
                        if hasattr(complex_obj.receptor, 'name'):
                            row['receptor'] = complex_obj.receptor.name
                    # Add db column if not present (for compatibility with preprocessing)
                    if 'db' not in row:
                        row['db'] = 'UNKNOWN'
                    data_list.append(row)
                
                data = pd.DataFrame(data_list)
                
                if data.empty:
                    ocerror.Error.data_not_found("No data found in database.") # type: ignore
                    raise ValueError("No data found in database.")
        
        except (ImportError, AttributeError) as e:
            ocerror.Error.data_not_found(f"Failed to read from database: {e}. Please provide data as DataFrame or file path.") # type: ignore
            raise ValueError(f"Failed to read from database: {e}. Please provide data as DataFrame or file path.")
    
    # If data is a string, treat it as a file path
    if isinstance(data, str):
        if not os.path.isfile(data):
            ocerror.Error.file_not_exist(f"Data file not found: {data}") # type: ignore
            raise FileNotFoundError(f"Data file not found: {data}")
        data = ocscoreio.load_data(data)
    
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        ocerror.Error.value_error("Data must be a pandas DataFrame or a path to a CSV file.") # type: ignore
        raise ValueError("Data must be a pandas DataFrame or a path to a CSV file.")
    
    # Store original data structure for return format
    original_data = data.copy()
    is_dataframe = True
    
    # Identify score columns
    if score_columns_list:
        score_columns = data.filter(regex=f"^({'|'.join(score_columns_list)})").columns.to_list()
    else:
        score_columns = []
    
    # Apply preprocessing pipeline (similar to preprocess_df)
    # Handle score columns
    if no_scores:
        # Remove score columns
        if score_columns:
            data = data.drop(columns=score_columns, errors='ignore')
    elif only_scores:
        # Keep only score columns and metadata
        metadata_cols = ["receptor", "ligand", "name", "type", "db"]
        columns_to_keep = [col for col in metadata_cols if col in data.columns] + score_columns
        data = ocscoredata.remove_other_columns(data, columns_to_keep, inplace=False)
    
    # Invert values conditionally
    if invert_conditionally:
        data = ocscoredata.invert_values_conditionally(data, inplace=False)
    
    # Normalize data
    if normalize:
        data = ocscoredata.norm_data(data, scaler=scaler, inplace=False)
    
    # Apply PCA if pca_model is provided
    if pca_model is not None:
        # Set default columns to skip PCA
        if columns_to_skip_pca is None:
            columns_to_skip_pca = ["receptor", "ligand", "name", "type", "db"]
            if score_columns:
                columns_to_skip_pca.extend(score_columns)
        
        # Apply PCA
        data = ocscoredata.apply_pca(
            data, 
            pca_model, 
            columns_to_skip_pca=columns_to_skip_pca, 
            inplace=False
        )
    
    # Prepare features for prediction (exclude metadata columns)
    metadata_cols = ["receptor", "ligand", "name", "type", "db", "experimental"]
    feature_cols = [col for col in data.columns if col not in metadata_cols]
    X = data[feature_cols].values
    
    # Apply mask if provided
    if mask is not None:
        # Convert mask to numpy array if it's a list
        mask = np.asarray(mask, dtype=bool)
        
        # Validate mask length
        if len(mask) != X.shape[1]:
            ocerror.Error.value_error(f"Mask length ({len(mask)}) does not match number of features ({X.shape[1]}).") # type: ignore
            raise ValueError(f"Mask length ({len(mask)}) does not match number of features ({X.shape[1]}).")
        
        # Apply mask to filter features
        X = X[:, mask]
    
    # Make predictions
    try:
        # Try to use predict method (for sklearn, xgboost, etc.)
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
        # Try to use forward method (for PyTorch models)
        elif hasattr(model, 'forward'):
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).cpu().numpy()
                # Flatten if needed
                if predictions.ndim > 1 and predictions.shape[1] == 1:
                    predictions = predictions.flatten()
        else:
            ocerror.Error.value_error("Model does not have a predict or forward method.") # type: ignore
            raise ValueError("Model does not have a predict or forward method.")
    except Exception as e:
        ocerror.Error.value_error(f"Error during prediction: {e}") # type: ignore
        raise ValueError(f"Error during prediction: {e}")
    
    # Return results in appropriate format
    if is_dataframe:
        # Create result DataFrame with metadata if available
        result = original_data[metadata_cols].copy() if any(col in original_data.columns for col in metadata_cols) else pd.DataFrame()
        result['predicted_score'] = predictions
        return result
    else:
        return predictions

