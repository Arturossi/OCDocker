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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    scaler: Optional[str] = "standard",
    scaler_path: Optional[str] = None,
    invert_conditionally: bool = True,
    normalize: bool = True,
    no_scores: bool = False,
    only_scores: bool = False,
    columns_to_skip_pca: Optional[list[str]] = None,
    serialization_method: str = "auto",
    use_gpu: bool = True
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
        Scaler to use for normalization if scaler_path is not provided. 
        Options are "standard" or "minmax". If scaler_path is provided, this is ignored.
        Default is "standard".
    scaler_path : str, optional
        Path to a saved scaler file (saved with joblib/pickle). If provided, the saved
        scaler will be loaded and used instead of creating a new one. This ensures
        the same scaling parameters from training are applied. Default is None.
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
    loaded_obj = ocscoreio.load_object(model_path, serialization_method="auto")
    
    # Handle different model formats
    # If loaded object is a dict, it might be a state_dict or a dict containing the model
    if isinstance(loaded_obj, dict):
        # Check if it's a state_dict (PyTorch model weights)
        if 'state_dict' in loaded_obj or any(key.startswith(('layer', 'fc', 'linear', 'encoder', 'decoder')) for key in loaded_obj.keys()):
            # This is likely a state_dict, but we need the model class to load it
            # For now, raise an error asking for the model object
            ocerror.Error.value_error("Model file contains a state_dict (weights only), not a complete model. Please load the model class first, then load_state_dict().") # type: ignore
            raise ValueError("Model file contains a state_dict (weights only), not a complete model. Please load the model class first, then load_state_dict().")
        elif 'model' in loaded_obj:
            # Dict contains the model under 'model' key
            model = loaded_obj['model']
        elif 'network' in loaded_obj:
            # Dict contains the model under 'network' key
            model = loaded_obj['network']
        else:
            # Try to find any value that looks like a model object
            for key, value in loaded_obj.items():
                if hasattr(value, 'predict') or hasattr(value, 'forward'):
                    model = value
                    break
            else:
                ocerror.Error.value_error(f"Model file contains a dict but no model object found. Keys: {list(loaded_obj.keys())}") # type: ignore
                raise ValueError(f"Model file contains a dict but no model object found. Keys: {list(loaded_obj.keys())}")
    else:
        # Loaded object is the model itself
        model = loaded_obj
    
    # Set PyTorch models to eval mode for inference
    if hasattr(model, 'eval'):
        model.eval()
    
    # Determine device for PyTorch models
    import torch
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Move PyTorch model to the correct device
    if hasattr(model, 'to'):
        model = model.to(device)
        # Also update model.device if it exists
        if hasattr(model, 'device'):
            model.device = device
    
    # Fix mask attribute if it's stored as dict/list instead of tensor
    # This can happen when models are saved/loaded
    def fix_mask_attribute(obj, device=None):
        """Recursively fix mask attributes in model and nested modules."""
        if hasattr(obj, 'mask'):
            if obj.mask is None:
                obj.mask = []
            elif isinstance(obj.mask, dict):
                # Extract array from dict
                mask_value = None
                for key in ['mask', 'array', 'feature_mask', 'ablation_mask']:
                    if key in obj.mask:
                        mask_value = obj.mask[key]
                        break
                if mask_value is None:
                    for v in obj.mask.values():
                        if isinstance(v, (list, np.ndarray, torch.Tensor)):
                            mask_value = v
                            break
                if mask_value is None:
                    obj.mask = []
                elif isinstance(mask_value, torch.Tensor):
                    obj.mask = mask_value.float()
                elif isinstance(mask_value, np.ndarray):
                    obj.mask = torch.from_numpy(mask_value).float()
                elif isinstance(mask_value, list):
                    obj.mask = torch.tensor(mask_value, dtype=torch.float32) if mask_value else []
                else:
                    obj.mask = []
                
                # Move to device
                if isinstance(obj.mask, torch.Tensor):
                    target_device = device if device else (obj.device if hasattr(obj, 'device') else torch.device('cpu'))
                    obj.mask = obj.mask.to(target_device)
            elif isinstance(obj.mask, (list, np.ndarray)) and not isinstance(obj.mask, torch.Tensor):
                # Convert list/array to tensor
                if isinstance(obj.mask, np.ndarray):
                    obj.mask = torch.from_numpy(obj.mask).float()
                elif isinstance(obj.mask, list):
                    obj.mask = torch.tensor(obj.mask, dtype=torch.float32) if obj.mask else []
                
                # Move to device
                if isinstance(obj.mask, torch.Tensor):
                    target_device = device if device else (obj.device if hasattr(obj, 'device') else torch.device('cpu'))
                    obj.mask = obj.mask.to(target_device)
        
        # Also check nested modules
        if hasattr(obj, 'modules'):
            for module in obj.modules():
                if module is not obj:  # Avoid infinite recursion
                    fix_mask_attribute(module, device)
    
    # Fix mask in the model and all nested modules
    device = model.device if hasattr(model, 'device') else None
    fix_mask_attribute(model, device)
    
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
        # Try to load scaler from file if scaler_path is provided
        scaler_obj = None
        if scaler_path is not None:
            if not os.path.isfile(scaler_path):
                ocerror.Error.file_not_exist(f"Scaler file not found: {scaler_path}") # type: ignore
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            try:
                scaler_obj = ocscoreio.load_object(scaler_path, serialization_method="auto")
                # Verify it's a scaler object
                if not isinstance(scaler_obj, (StandardScaler, MinMaxScaler)):
                    ocerror.Error.value_error(f"File {scaler_path} does not contain a valid scaler object.") # type: ignore
                    raise ValueError(f"File {scaler_path} does not contain a valid scaler object. Got {type(scaler_obj)}")
            except Exception as e:
                ocerror.Error.value_error(f"Failed to load scaler from {scaler_path}: {e}") # type: ignore
                raise ValueError(f"Failed to load scaler from {scaler_path}: {e}")
        
        # Use the loaded scaler or the scaler string
        if scaler_obj is not None:
            data = ocscoredata.norm_data(data, scaler=scaler_obj, inplace=False)
        else:
            # Create a new scaler (this will fit on the prediction data - not ideal but backward compatible)
            result = ocscoredata.norm_data(data, scaler=scaler, inplace=False)
            # Handle tuple return (DataFrame, scaler) when fitting new scaler
            if isinstance(result, tuple):
                data = result[0]
            else:
                data = result
    
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
    # NOTE: If the model has its own mask (e.g., PyTorch DynamicNN), we should NOT apply
    # the external mask here, as the model will apply its own mask in the forward pass.
    # The external mask parameter is for models that don't have built-in masking.
    model_has_mask = hasattr(model, 'mask') and model.mask is not None and len(model.mask) > 0
    
    if mask is not None and not model_has_mask:
        # Only apply external mask if model doesn't have its own mask
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
            
            # Ensure mask is properly formatted before forward pass
            # This is critical - the mask must be a tensor, not a dict/list
            # Also ensure it's on the same device as the model
            if hasattr(model, 'mask'):
                if model.mask is None:
                    # Set to empty list if None (DynamicNN expects list or tensor)
                    model.mask = []
                elif not isinstance(model.mask, torch.Tensor):
                    # Convert mask to tensor if it's not already
                    if isinstance(model.mask, dict):
                        # Extract from dict - try common keys first
                        mask_value = None
                        for key in ['mask', 'array', 'feature_mask', 'ablation_mask']:
                            if key in model.mask:
                                mask_value = model.mask[key]
                                break
                        
                        # If not found, try to find first array-like value
                        if mask_value is None:
                            for v in model.mask.values():
                                if isinstance(v, (list, np.ndarray, torch.Tensor)):
                                    mask_value = v
                                    break
                        
                        # If still None, try first value
                        if mask_value is None and model.mask:
                            first_val = list(model.mask.values())[0]
                            if isinstance(first_val, (list, np.ndarray, torch.Tensor)):
                                mask_value = first_val
                            else:
                                mask_value = []
                    elif isinstance(model.mask, (list, np.ndarray)):
                        mask_value = model.mask
                    else:
                        mask_value = []
                    
                    # Convert to tensor
                    if isinstance(mask_value, torch.Tensor):
                        model.mask = mask_value.float()
                    elif isinstance(mask_value, np.ndarray):
                        model.mask = torch.from_numpy(mask_value).float()
                    elif isinstance(mask_value, list):
                        if len(mask_value) > 0:
                            model.mask = torch.tensor(mask_value, dtype=torch.float32)
                        else:
                            model.mask = []
                    else:
                        model.mask = []
            
            # Final safety check: ensure mask is not a dict before forward pass
            # This prevents the "dict * int" error in DynamicNN.forward()
            if hasattr(model, 'mask') and isinstance(model.mask, dict):
                # If still a dict after all conversion attempts, set to empty list
                # This will prevent the multiplication error
                model.mask = []
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                # Move input tensor to the same device as the model
                X_tensor = X_tensor.to(device)
                
                # Ensure mask is on the same device as the input tensor
                if hasattr(model, 'mask') and isinstance(model.mask, torch.Tensor):
                    model.mask = model.mask.to(device)
                
                predictions = model(X_tensor).cpu().numpy()
                # Flatten if needed
                if predictions.ndim > 1 and predictions.shape[1] == 1:
                    predictions = predictions.flatten()
        else:
            ocerror.Error.value_error("Model does not have a predict or forward method.") # type: ignore
            raise ValueError("Model does not have a predict or forward method.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        ocerror.Error.value_error(f"Error during prediction: {e}\n{error_details}") # type: ignore
        raise ValueError(f"Error during prediction: {e}\n{error_details}")
    
    # Return results in appropriate format
    if is_dataframe:
        # Create result DataFrame with metadata if available
        # Only include metadata columns that actually exist in the original data
        available_metadata_cols = [col for col in metadata_cols if col in original_data.columns]
        if available_metadata_cols:
            result = original_data[available_metadata_cols].copy()
        else:
            result = pd.DataFrame()
        result['predicted_score'] = predictions
        return result
    else:
        return predictions

