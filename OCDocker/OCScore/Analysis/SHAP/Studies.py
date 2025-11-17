
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Union
import numpy as np
import optuna
import logging

LOGGER = logging.getLogger("OCScore.SHAP.studies")

@dataclass
class StudyHandles:
    '''Container for Optuna study handles and storage information.
    
    Attributes
    ----------
    ao_study_name : str
        Name of the autoencoder optimization study.
    nn_study_name : str
        Name of the neural network optimization study.
    seed_study_name : str
        Name of the random seed optimization study.
    mask_study_name : str
        Name of the feature mask optimization study.
    storage : str
        Storage path/URL for Optuna studies.
    '''
    
    ao_study_name: str
    nn_study_name: str
    seed_study_name: str
    mask_study_name: str
    storage: str

@dataclass
class BestSelections:
    '''Container for best parameters selected from Optuna studies.
    
    Attributes
    ----------
    autoencoder_params : Dict[str, Union[int, float, str, bool]]
        Best autoencoder parameters.
    nn_params : Dict[str, Union[int, float, str, bool]]
        Best neural network parameters.
    seed : int
        Best random seed.
    mask : np.ndarray
        Best feature mask as a binary array.
    '''
    
    autoencoder_params: Dict[str, Union[int, float, str, bool]]
    nn_params: Dict[str, Union[int, float, str, bool]]
    seed: int
    mask: np.ndarray


def select_best_from_studies(handles: StudyHandles) -> BestSelections:
    '''Select best parameters from multiple Optuna optimization studies.
    
    Parameters
    ----------
    handles : StudyHandles
        Container with study names and storage information.
    
    Returns
    -------
    BestSelections
        Container with best parameters from all studies (autoencoder, neural network, seed, mask).
    '''
    
    LOGGER.info("Loading Optuna studies from storage")
    ao_study = optuna.load_study(study_name=handles.ao_study_name, storage=handles.storage)
    nn_study = optuna.load_study(study_name=handles.nn_study_name, storage=handles.storage)
    seed_study = optuna.load_study(study_name=handles.seed_study_name, storage=handles.storage)
    mask_study = optuna.load_study(study_name=handles.mask_study_name, storage=handles.storage)

    # Autoencoder
    ao_df = ao_study.trials_dataframe()
    ao_df = ao_df[ao_df['state'] == 'COMPLETE']
    best_ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])
    best_ao_trial = ao_study.trials[int(best_ao_df.iloc[0].number)]
    autoencoder_params = best_ao_trial.params

    # Neural network topology
    nn_df = nn_study.trials_dataframe()
    nn_df = nn_df[nn_df['state'] == 'COMPLETE']
    nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']
    best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])
    best_nn_trial = nn_study.trials[int(best_nn_df.iloc[0].number)]
    nn_params = best_nn_trial.params

    # Seed
    seed_df = seed_study.trials_dataframe()
    seed_df = seed_df[seed_df['state'] == 'COMPLETE']
    best_seed_df = seed_df.sort_values(by=['value', 'user_attrs_AUC'], ascending=[True, False])
    best_seed_trial = seed_study.trials[int(best_seed_df.iloc[0].number)]
    seed = int(best_seed_trial.user_attrs['random_seed'])

    # Mask
    mask_df = mask_study.trials_dataframe()
    mask_df = mask_df[mask_df['state'] == 'COMPLETE']
    best_mask_df = mask_df.sort_values(by=['value', 'user_attrs_AUC'], ascending=[True, False])
    best_mask_trial = mask_study.trials[int(best_mask_df.iloc[0].number)]
    mask = np.array([int(x) for x in best_mask_trial.user_attrs['Feature_Mask']])

    return BestSelections(
        autoencoder_params=autoencoder_params,
        nn_params=nn_params,
        seed=seed,
        mask=mask,
    )
