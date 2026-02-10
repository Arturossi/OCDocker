#!/usr/bin/env python3

# Description
###############################################################################
''' Module with a helper to perform the optimization of the future DNN pipeline.

It is imported as:

import OCDocker.OCScore.Optimization.future.DNN as ocdnn_future
'''

# Imports
###############################################################################

import optuna

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from typing import Union

import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.Toolbox.Printing as ocprint

from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer

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

def _future_worker(
        pid: int,
        storage_id: int,
        X_pdb_train,
        y_pdb_train,
        X_pdb_test,
        y_pdb_test,
        dude_train: dict,
        dude_val: dict,
        storage: str,
        encoder_params: dict | None,
        random_seed: int,
        use_gpu: bool,
        verbose: bool,
        n_trials: int,
        study_name: str,
        future_config: dict
    ) -> None:
    if verbose:
        ocprint.printv(f"[FutureNN] Process {pid} starting optimization")

    local_config = dict(future_config) if future_config else {}
    local_config["dude_train_data"] = dude_train
    local_config["dude_val_data"] = dude_val

    trainer = DNNOptimizer(
        X_pdb_train,
        y_pdb_train,
        X_pdb_test,
        y_pdb_test,
        X_validation=None,
        y_validation=None,
        storage=storage,
        encoder_params=encoder_params,
        random_seed=random_seed + pid,
        use_gpu=use_gpu,
        verbose=verbose,
        future_config=local_config
    )

    trainer.optimize(
        direction="maximize",
        n_trials=n_trials,
        study_name=study_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        n_jobs=1
    )

    if verbose:
        ocprint.printv(f"[FutureNN] Process {pid} finished optimization")


def _prepare_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=drop_cols, errors="ignore")


def _split_dude_by_target(X: Union[pd.DataFrame, list[pd.DataFrame]], y: np.ndarray, targets: np.ndarray, val_fraction: float, random_seed: int) -> tuple[dict, dict]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_seed)
    idx_train, idx_val = next(splitter.split(np.arange(len(y)), y, groups=targets))

    def _sel(data, idx):
        if isinstance(data, list):
            return [d.iloc[idx] for d in data]
        return data.iloc[idx]

    train = {
        "X": _sel(X, idx_train),
        "y": y[idx_train],
        "targets": targets[idx_train]
    }
    val = {
        "X": _sel(X, idx_val),
        "y": y[idx_val],
        "targets": targets[idx_val]
    }

    return train, val


## Public ##

def optimize_NN_future(
        df_path: str,
        storage_id: int,
        base_models_folder: str,
        data: dict = {},
        storage: str = "sqlite:///NN_optimization.db",
        use_pdb_train: bool = True,
        no_scores: bool = False,
        only_scores: bool = False,
        use_PCA: bool = False,
        best_ao_params: Union[dict, None] = None,
        pca_type: int = 80,
        pca_model: Union[str, PCA] = "",
        encoder_dims: tuple[int, int] = (16, 256),
        autoencoder: bool = True,
        multiencoder: bool = False,
        run_autoencoder_optimization: bool = True,
        num_processes_autoencoder: int = 8,
        total_trials_autoencoder: int = 2000,
        run_NN_optimization: bool = True,
        num_processes_NN: int = 4,
        total_trials_NN: int = 50,
        explained_variance: float = 0.95,
        random_seed: int = 42,
        load_if_exists: bool = True,
        use_gpu: bool = True,
        parallel_backend: str = "joblib",
        verbose: bool = False,
        use_future: bool = True,
        future_config: dict | None = None
    ) -> None:
    '''Optimize the future DNN pipeline.

    Parameters
    ----------
    df_path : str
        Path to the dataset.
    storage_id : int
        Storage ID.
    base_models_folder : str
        Base folder for models.
    data : dict, optional
        Preloaded data dict (ignored in future pipeline if empty).
    use_future : bool, optional
        If False, fallback to the current pipeline. Default True.

    Notes
    -----
    Example usage:
        optimize_NN_future(df_path, 1, "./models", total_trials_NN=20)
    '''

    if not use_future:
        # Fallback to current pipeline
        import OCDocker.OCScore.Optimization.DNN as ocdnn
        return ocdnn.optimize_NN(
            df_path=df_path,
            storage_id=storage_id,
            base_models_folder=base_models_folder,
            data=data,
            storage=storage,
            use_pdb_train=use_pdb_train,
            no_scores=no_scores,
            only_scores=only_scores,
            use_PCA=use_PCA,
            best_ao_params=best_ao_params,
            pca_type=pca_type,
            pca_model=pca_model,
            encoder_dims=encoder_dims,
            autoencoder=autoencoder,
            multiencoder=multiencoder,
            run_autoencoder_optimization=run_autoencoder_optimization,
            num_processes_autoencoder=num_processes_autoencoder,
            total_trials_autoencoder=total_trials_autoencoder,
            run_NN_optimization=run_NN_optimization,
            num_processes_NN=num_processes_NN,
            total_trials_NN=total_trials_NN,
            explained_variance=explained_variance,
            random_seed=random_seed,
            load_if_exists=load_if_exists,
            use_gpu=use_gpu,
            parallel_backend=parallel_backend,
            verbose=verbose
        )

    if multiencoder:
        ocprint.print_warning("Multiencoder is not supported in the future pipeline. Disabling it.")
        multiencoder = False

    # Load data
    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path, invert_conditionally=True)

    # Filter columns
    if no_scores:
        dudez_data = dudez_data.drop(columns=score_columns, errors="ignore")
        pdbbind_data = pdbbind_data.drop(columns=score_columns, errors="ignore")
        study_prefix = "NoScores_"
    elif only_scores:
        ocscoredata.remove_other_columns(dudez_data, ["receptor", "ligand", "name", "type", "db"] + score_columns, inplace=True)
        ocscoredata.remove_other_columns(pdbbind_data, ["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns, inplace=True)
        study_prefix = "ScoreOnly_"
    else:
        study_prefix = ""

    if use_PCA:
        if pca_model == "":
            # Fit PCA on PDBbind features if no model path provided
            pdb_features = _prepare_features(pdbbind_data, ["receptor", "ligand", "name", "type", "db", "experimental"])
            if isinstance(pca_type, int) and pca_type <= 100:
                pca_model = PCA(n_components=pca_type / 100.0, svd_solver="full")
            else:
                pca_model = PCA(n_components=int(pca_type))
            pca_model.fit(pdb_features.values)
        ocscoredata.apply_pca(pdbbind_data, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns, inplace=True)
        ocscoredata.apply_pca(dudez_data, pca_model, columns_to_skip_pca=["receptor", "ligand", "name", "type", "db"] + score_columns, inplace=True)
        study_prefix = f"PCA{pca_type}_" + study_prefix

    # Prepare PDBbind features/labels
    X_pdb = _prepare_features(pdbbind_data, ["receptor", "ligand", "name", "type", "db", "experimental"])
    y_pdb = pdbbind_data["experimental"].values

    # Prepare DUDE features/labels/targets
    y_dude = dudez_data["type"].map({"ligand": 1, "decoy": 0}).values
    targets = dudez_data["receptor"].values
    X_dude = _prepare_features(dudez_data, ["receptor", "ligand", "name", "type", "db", "experimental"])

    # Train/test split for PDBbind
    X_pdb_train, X_pdb_test, y_pdb_train, y_pdb_test = ocscoredata.split_dataset(
        X_pdb,
        y_pdb,
        test_size=0.25,
        random_state=random_seed
    )

    # Split DUDE by target for train/val
    val_fraction = 0.2
    if future_config and "data" in future_config and "dude_validation_fraction" in future_config["data"]:
        val_fraction = float(future_config["data"]["dude_validation_fraction"])

    dude_train, dude_val = _split_dude_by_target(X_dude, y_dude, targets, val_fraction, random_seed)

    # Build study name
    study_name = f"{study_prefix}Future_NN_Optimization_{storage_id}"

    if not run_NN_optimization:
        return None

    # Respect use_pdb_train flag by disabling stage1/energy anchoring if requested
    if not use_pdb_train:
        future_config = future_config or {}
        future_config.setdefault("stage1", {})
        future_config.setdefault("stage2", {})
        future_config["stage1"]["enabled"] = False
        future_config["stage2"]["lambda_energy"] = 0.0
        future_config["stage2"]["lambda_recon"] = 0.0

    # Trials per process
    n_trials = max(1, total_trials_NN // max(1, num_processes_NN))

    if parallel_backend == "joblib":
        Parallel(n_jobs=num_processes_NN)(
            delayed(_future_worker)(
                pid,
                storage_id,
                X_pdb_train,
                y_pdb_train,
                X_pdb_test,
                y_pdb_test,
                dude_train,
                dude_val,
                storage,
                best_ao_params,
                random_seed,
                use_gpu,
                verbose,
                n_trials,
                study_name,
                future_config or {}
            ) for pid in range(num_processes_NN)
        )
    elif parallel_backend == "multiprocessing":
        with Pool(num_processes_NN) as pool:
            pool.starmap(_future_worker, [(
                pid,
                storage_id,
                X_pdb_train,
                y_pdb_train,
                X_pdb_test,
                y_pdb_test,
                dude_train,
                dude_val,
                storage,
                best_ao_params,
                random_seed,
                use_gpu,
                verbose,
                n_trials,
                study_name,
                future_config or {}
            ) for pid in range(num_processes_NN)])
    else:
        raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

    return None


# Alias
optimize = optimize_NN_future
