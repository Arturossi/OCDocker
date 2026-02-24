#!/usr/bin/env python3

# Description
###############################################################################
'''
Parallel worker utilities for Optuna-based model optimization.

Usage:

import OCDocker.OCScore.Utils.Workers as ocscoreworkers
'''

# Imports
###############################################################################

import optuna
import os
import time

import numpy as np

from optuna.samplers import TPESampler
from typing import Any, Callable, Optional, TypeVar, Union

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.OCScore.Dimensionality.AutoencoderOptimizer import AutoencoderOptimizer
from OCDocker.OCScore.Dimensionality.GeneticAlgorithm import GeneticAlgorithm
from OCDocker.OCScore.DNN.DNNOptimizer import DNNOptimizer
from OCDocker.OCScore.Transformer.TransOptimizer import TransOptimizer
from OCDocker.OCScore.XGBoost.XGBoostOptimizer import XGBoostOptimizer

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

_T = TypeVar("_T")


def _is_transient_optuna_storage_error(exc: Exception) -> bool:
    '''Detect transient Optuna storage contention/migration race errors.

    Parameters
    ----------
    exc : Exception
        Raised exception.

    Returns
    -------
    bool
        True when the error is likely transient and safe to retry.
    '''

    msg = str(exc).lower()

    sqlite_lock = "database is locked" in msg
    sqlite_schema_race = (
        ("alembic_version" in msg and ("already exists" in msg or "unique constraint failed" in msg))
        or ("table study_" in msg and "already exists" in msg)
        or ("table trial_" in msg and "already exists" in msg)
        or ("table studies" in msg and "already exists" in msg)
    )

    # Common transactional lock/contention markers for network DB backends.
    rdb_lock = (
        "deadlock detected" in msg
        or "lock wait timeout exceeded" in msg
        or "could not serialize access due to concurrent update" in msg
        or "serialization failure" in msg
        or "could not obtain lock on row" in msg
        or "(1213" in msg  # MySQL deadlock
        or "(1205" in msg  # MySQL lock wait timeout
        or "40p01" in msg  # PostgreSQL deadlock SQLSTATE
        or "40001" in msg  # PostgreSQL serialization failure SQLSTATE
    )

    # Bootstrap/migration DDL races that can occur when multiple workers hit
    # create_study() at once on fresh storage (MySQL/PostgreSQL included).
    rdb_schema_race = (
        # PostgreSQL-style relation races
        ("relation " in msg and "already exists" in msg and (
            "alembic_version" in msg
            or "study_" in msg
            or "trial_" in msg
            or "studies" in msg
        ))
        # Generic duplicate-key races during bootstrap/create-study
        or (
            (
                "duplicate key value violates unique constraint" in msg
                or "unique constraint failed" in msg
                or "duplicate entry" in msg
                or "(1062" in msg  # MySQL duplicate entry
            )
            and (
                "alembic_version" in msg
                or "studies" in msg
                or "study_" in msg
                or "trial_" in msg
            )
        )
        # MySQL table-exists bootstrap race
        or (("(1050" in msg) and ("study_" in msg or "trial_" in msg or "alembic_version" in msg))
    )

    return sqlite_lock or sqlite_schema_race or rdb_lock or rdb_schema_race


def _run_optuna_op_with_retry(
    operation: Callable[[], _T],
    *,
    pid: int,
    verbose: bool,
    context: str,
) -> _T:
    '''Run an Optuna operation with retry on transient storage contention.

    The retry logic is intentionally scoped to Optuna storage race/lock errors
    to avoid masking real training/logic failures.
    '''

    try:
        retries = max(0, int(os.getenv("OCDOCKER_OPTUNA_LOCK_RETRIES", "6")))
    except (TypeError, ValueError):
        retries = 6

    try:
        base_delay = max(0.0, float(os.getenv("OCDOCKER_OPTUNA_LOCK_BASE_DELAY", "0.15")))
    except (TypeError, ValueError):
        base_delay = 0.15

    try:
        max_delay = max(base_delay, float(os.getenv("OCDOCKER_OPTUNA_LOCK_MAX_DELAY", "3.0")))
    except (TypeError, ValueError):
        max_delay = 3.0

    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient_optuna_storage_error(exc) or attempt >= retries:
                raise

            delay = min(max_delay, base_delay * (2 ** attempt))
            # Deterministic micro-jitter to avoid synchronized retries.
            delay += ((pid % 13) + 1) * 0.01

            if verbose:
                ocprint.print_warning(
                    f"Transient Optuna storage contention during '{context}' "
                    f"(attempt {attempt + 1}/{retries + 1}). Retrying in {delay:.2f}s. "
                    f"Error: {type(exc).__name__}: {exc}"
                )
            time.sleep(delay)

    # Unreachable, loop always returns or raises.
    raise RuntimeError(f"Unexpected retry loop termination for context '{context}'.")

## Public ##

def AEworker(
        pid : int,
        id : int,
        X_train : np.ndarray,
        X_test : np.ndarray,
        X_val : np.ndarray,
        encoding_dims : tuple,
        storage : str,
        models_folder : str,
        random_seed : int = 42,
        use_gpu : bool = True,
        verbose : bool = False,
        direction : str = "minimize",
        n_trials : int = 250,
        load_if_exists : bool = True,
        n_jobs : int = 1,
        study_name : str = "Autoencoder_Optimization"
    ) -> optuna.study.Study:
    ''' Autoencoder optimization worker function.

    This function is used to run the optimization of an autoencoder model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Testing data.
    X_val : np.ndarray
        Validation data.
    encoding_dims : tuple
        Tuple with the encoding dimensions.
    storage : str
        Storage string.
    models_folder : str
        Folder to save the models.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    direction : str, optional
        Optimization direction. The default is "minimize".
    n_trials : int, optional
        Number of trials. The default is 250.
    load_if_exists : bool, optional
        Load if exists. The default is True.
    n_jobs : int, optional
        Number of jobs. The default is 1.
    study_name : str, optional
        Study name. The default is "Autoencoder_Optimization".

    Returns
    -------
    study : optuna.study.Study
        Study object.
    '''

    if verbose:
        ocprint.printv(f"Process {pid} starting optimization")

    # Initialize the trainer
    trainer = AutoencoderOptimizer(
        X_train,
        X_test,
        X_val,
        encoding_dims,
        storage,
        models_folder,
        random_seed = random_seed,
        use_gpu = use_gpu,
        verbose = verbose
    )

    study = None

    # Run optimization
    study = _run_optuna_op_with_retry(
        lambda: trainer.optimize(
            direction = direction,
            n_trials = n_trials,
            study_name = f"{study_name}_{id}",
            load_if_exists = load_if_exists,
            sampler = TPESampler(),
            n_jobs = n_jobs
        ),
        pid=pid,
        verbose=verbose,
        context=f"{study_name}_{id}",
    )

    if verbose:
        ocprint.printv(f"Process {pid} has completed the optimization")

    return study


def GAWorker(
        pid : int,
        id : int,
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_test : np.ndarray,
        y_test : np.ndarray,
        X_validation : Union[np.ndarray, None] = None,
        y_validation : Union[np.ndarray, None] = None,
        storage : str = "sqlite:///GA.db",
        best_params : Optional[dict] = None,
        n_trials : int = 100,
        study_name : str = "GA_Feature_Selection",
        random_state : int = 42,
        use_gpu : bool = True,
        verbose : bool = False,
        n_jobs : int = 1
    ) -> tuple[optuna.study.Study, dict, float]:
    ''' Feature selection worker function using Genetic Algorithms.

    This function is used to run the optimization of a feature selection model in
    a separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_validation : Union[np.ndarray, None], optional
        Validation data. The default is None.
    y_validation : Union[np.ndarray, None], optional
        Validation labels. The default is None.
    storage : str, optional
        Storage string. The default is "sqlite:///GA.db".
    best_params : dict, optional
        Best parameters. The default is None (treated as an empty dict).
    algorithm : str, optional
        Algorithm. The default is "ga".
    n_trials : int, optional
        Number of trials. The default is 100.
    study_name : str, optional
        Study name. The default is "GA_Feature_Selection".
    random_state : int, optional
        Random state. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.

    Returns
    -------
    study : optuna.study.Study
        Study object.
    best_features : list
        Best features.
    best_score : float
        Best score.
    '''

    if best_params is None:
        best_params = {}

    if verbose:
        ocprint.printv(f"Process {pid} starting optimization")

    # Create the GeneticAlgorithm object
    evo = GeneticAlgorithm(X_train, y_train, X_test, y_test, X_validation = X_validation, y_validation = y_validation, storage = storage, xgboost_params = best_params, use_gpu = use_gpu, random_state = random_state, verbose = verbose)

    # Run the optimization
    study, best_features, best_score = _run_optuna_op_with_retry(
        lambda: evo.optimize(
            study_name = f"{study_name}_{id}",
            direction = "minimize",
            n_trials = n_trials,
            n_jobs = n_jobs
        ),
        pid=pid,
        verbose=verbose,
        context=f"{study_name}_{id}",
    )

    if verbose:
        ocprint.printv(f"Process {pid} has completed the optimization")

    return study, best_features, best_score


def NNAblationworker(
        pid : int,
        id : int,
        X_train : np.ndarray, y_train : np.ndarray,
        X_test : np.ndarray, y_test : np.ndarray,
        X_val : np.ndarray, y_val : np.ndarray,
        mask : Union[list[np.ndarray], np.ndarray],
        storage : str,
        network_params : dict[str, Any],
        encoder_params : Union[dict, None] = None,
        output_size : int = 1,
        random_seed : int = 42,
        use_gpu : bool = True,
        verbose : bool = False,
        load_if_exists : bool = True,
        n_jobs : int = 1,
        study_name : str = "NN_Ablation_Optimization"
    ) ->  None:
    ''' Neural network optimization worker function.

    This function is used to run the optimization of a neural network model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_val : np.ndarray
        Validation data.
    y_val : np.ndarray
        Validation labels.
    mask : list[Union[int, bool]]
        Mask list.
    storage : str
        Storage string.
    network_params : dict[str, Any]
        Network parameters.
    encoder_params : Union[dict, None], optional
        Encoder parameters. The default is None.
    output_size : int, optional
        Output size. The default is 1.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    '''

    if verbose:
        ocprint.printv(f"Process {pid} starting ablation")

    # If mask is a list of np.ndarrays
    if isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        for m in mask:
            # Initialize the trainer
            trainer = DNNOptimizer(
                X_train, y_train,
                X_test, y_test,
                X_val, y_val,
                mask = m,
                storage = storage,
                encoder_params = encoder_params,
                output_size = output_size,
                random_seed = random_seed,
                use_gpu = use_gpu,
                verbose = verbose,
            )

            # Run optimization
            _run_optuna_op_with_retry(
                lambda: trainer.ablate(
                    network_params = network_params,
                    n_trials = 1,
                    study_name = f"{study_name}_{id}",
                    load_if_exists = load_if_exists,
                    n_jobs = n_jobs
                ),
                pid=pid,
                verbose=verbose,
                context=f"{study_name}_{id}",
            )

        if verbose:
            ocprint.printv(f"Process {pid} has completed the ablation")

    elif isinstance(mask, np.ndarray):
        # Initialize the trainer
        trainer = DNNOptimizer(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            mask = mask,
            storage = storage,
            encoder_params = encoder_params,
            output_size = output_size,
            random_seed = random_seed,
            use_gpu = use_gpu,
            verbose = verbose,
        )

        # Run optimization
        _run_optuna_op_with_retry(
            lambda: trainer.ablate(
                network_params = network_params,
                n_trials = 1,
                study_name = f"{study_name}_{id}",
                load_if_exists = load_if_exists,
                n_jobs = n_jobs
            ),
            pid=pid,
            verbose=verbose,
            context=f"{study_name}_{id}",
        )

        if verbose:
            ocprint.printv(f"Process {pid} has completed the ablation")
    else:
        raise ValueError("Mask must be a list of np.ndarrays of zeros, ones, or boleans or a np.ndarray of zeros, ones, or booleans")

    return None


def NNSeedAblationworker(
        pid : int,
        id : int,
        X_train : np.ndarray, y_train : np.ndarray,
        X_test : np.ndarray, y_test : np.ndarray,
        X_val : np.ndarray, y_val : np.ndarray,
        mask : np.ndarray,
        storage : str,
        network_params : dict[str, Any],
        random_seeds : Union[list[int], int],
        encoder_params : Union[dict, None] = None,
        output_size : int = 1,
        use_gpu : bool = True,
        verbose : bool = False,
        load_if_exists : bool = True,
        n_jobs : int = 1,
        study_name : str = "NN_Seed_Ablation_Optimization"
    ) ->  None:
    ''' Neural network optimization worker function.

    This function is used to run the optimization of a neural network model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_val : np.ndarray
        Validation data.
    y_val : np.ndarray
        Validation labels.
    mask : list[Union[int, bool]]
        Mask list.
    storage : str
        Storage string.
    network_params : dict[str, Any]
        Network parameters.
    random_seeds : list[int] | int
        Random seed list to ablate.
    encoder_params : Union[dict, None], optional
        Encoder parameters. The default is None.
    output_size : int, optional
        Output size. The default is 1.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    '''

    if verbose:
        ocprint.printv(f"Process {pid} starting ablation")

    # If random_seeds is a list of ints
    if isinstance(random_seeds, list) and isinstance(random_seeds[0], int):
        for random_seed in random_seeds:
            # Initialize the trainer
            trainer = DNNOptimizer(
                X_train, y_train,
                X_test, y_test,
                X_val, y_val,
                mask = mask,
                storage = storage,
                encoder_params = encoder_params,
                output_size = output_size,
                random_seed = random_seed,
                use_gpu = use_gpu,
                verbose = verbose,
            )

            # Run optimization
            _run_optuna_op_with_retry(
                lambda: trainer.ablate(
                    network_params = network_params,
                    n_trials = 1,
                    study_name = f"{study_name}_{id}",
                    load_if_exists = load_if_exists,
                    n_jobs = n_jobs
                ),
                pid=pid,
                verbose=verbose,
                context=f"{study_name}_{id}",
            )

        if verbose:
            ocprint.printv(f"Process {pid} has completed the ablation")

    elif isinstance(random_seeds, int):
        # Initialize the trainer
        trainer = DNNOptimizer(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            mask = mask,
            storage = storage,
            encoder_params = encoder_params,
            output_size = output_size,
            random_seed = random_seeds,
            use_gpu = use_gpu,
            verbose = verbose,
        )

        # Run optimization
        _run_optuna_op_with_retry(
            lambda: trainer.ablate(
                network_params = network_params,
                n_trials = 1,
                study_name = f"{study_name}_{id}",
                load_if_exists = load_if_exists,
                n_jobs = n_jobs
            ),
            pid=pid,
            verbose=verbose,
            context=f"{study_name}_{id}",
        )

        if verbose:
            ocprint.printv(f"Process {pid} has completed the ablation")
    else:
        raise ValueError("Seeds must be a list of ints or an int")

    return None


def NNworker(
        pid : int,
        id : int,
        X_train : np.ndarray, y_train : np.ndarray,
        X_test : np.ndarray, y_test : np.ndarray,
        X_val : np.ndarray, y_val : np.ndarray,
        storage : str,
        encoder_params : Union[dict, None] = None,
        output_size : int = 1,
        random_seed : int = 42,
        use_gpu : bool = True,
        verbose : bool = False,
        direction : str = "minimize",
        n_trials : int = 250,
        load_if_exists : bool = True,
        n_jobs : int = 1,
        study_name : str = "NN_Optimization"
    ) ->  None:
    ''' Neural network optimization worker function.

    This function is used to run the optimization of a neural network model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_val : np.ndarray
        Validation data.
    y_val : np.ndarray
        Validation labels.
    storage : str
        Storage string.
    encoder_params : Union[dict, None], optional
        Encoder parameters. The default is None.
    output_size : int, optional
        Output size. The default is 1.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    '''

    if True:
        ocprint.printv(f"Process {pid} starting optimization")

    # Initialize the trainer
    trainer = DNNOptimizer(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        storage = storage,
        encoder_params = encoder_params,
        output_size = output_size,
        random_seed = random_seed,
        use_gpu = use_gpu,
        verbose=verbose
    )

    # Run optimization
    _run_optuna_op_with_retry(
        lambda: trainer.optimize(
            direction = direction,
            n_trials = n_trials,
            study_name = f"{study_name}_{id}",
            load_if_exists = load_if_exists,
            sampler = TPESampler(),
            n_jobs = n_jobs
        ),
        pid=pid,
        verbose=verbose,
        context=f"{study_name}_{id}",
    )

    # Setup unique to this instance, potentially using instance_id to differentiate setups
    if verbose:
        ocprint.printv(f"Process {pid} has completed the optimization")

    return None


def Transworker(
            pid : int,
            id : int,
            X_train : np.ndarray, y_train : np.ndarray,
            X_test : np.ndarray, y_test : np.ndarray,
            X_val : np.ndarray, y_val : np.ndarray,
            storage : str,
            output_size : int = 1,
            random_seed : int = 42,
            use_gpu : bool = True,
            verbose : bool = False,
            direction : str = "minimize",
            n_trials : int = 250,
            load_if_exists : bool = True,
            n_jobs : int = 1,
            study_name : str = "Trans_Optimization"
        ) -> None:
        ''' Transformer optimization worker function.

        This function is used to run the optimization of a transformer model in a
        separate process. It is used to parallelize the optimization process.

        Parameters
        ----------
        pid : int
            Process ID.
        id : int
            Instance ID.
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training labels.
        X_test : np.ndarray
            Testing data.
        y_test : np.ndarray
            Testing labels.
        X_val : np.ndarray
            Validation data.
        y_val : np.ndarray
            Validation labels.
        storage : str
            Storage string.
        output_size : int, optional
            Output size. The default is 1.
        random_seed : int, optional
            Random seed. The default is 42.
        use_gpu : bool, optional
            Use GPU. The default is True.
        verbose : bool, optional
            Verbose. The default is False.
        direction : str, optional
            Optimization direction. The default is "minimize".
        n_trials : int, optional
            Number of trials. The default is 250.
        load_if_exists : bool, optional
            Load if exists. The default is True.
        n_jobs : int, optional
            Number of jobs. The default is 1.
        study_name : str, optional
            Study name. The default
        None
        '''

        if verbose:
            ocprint.printv(f"Process {pid} starting optimization")

        # Initialize the trainer
        trainer = TransOptimizer(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            storage,
            output_size = output_size,
            random_seed = random_seed,
            use_gpu = use_gpu,
            verbose=verbose
        )

        # Run optimization
        _run_optuna_op_with_retry(
            lambda: trainer.optimize(
                direction = direction,
                n_trials = n_trials,
                study_name = f"{study_name}_{id}",
                load_if_exists = load_if_exists,
                sampler = TPESampler(),
                n_jobs = n_jobs
            ),
            pid=pid,
            verbose=verbose,
            context=f"{study_name}_{id}",
        )

        if verbose:
            ocprint.printv(f"Process {pid} has compleated optimization")


def XGBworker(
        pid : int, id : int,
        X_train : np.ndarray,
        X_test : np.ndarray,
        X_val : np.ndarray,
        y_train : np.ndarray,
        y_test : np.ndarray,
        y_val : np.ndarray,
        storage : str,
        random_seed : int = 42,
        use_gpu : bool = True,
        verbose : bool = False,
        n_trials : int = 250,
        load_if_exists : bool = True,
        n_jobs : int = 10,
        study_name : str = "XGB_Optimization",
        early_stopping_rounds : int = 50,
        params : Optional[dict] = None
    ) -> optuna.study.Study:
    ''' XGBoost optimization worker function.

    This function is used to run the optimization of an XGBoost model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Testing data.
    X_val : np.ndarray
        Validation data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.
    y_val : np.ndarray
        Validation labels.
    storage : str
        Storage string.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    n_trials : int, optional
        Number of trials. The default is 250.
    load_if_exists : bool, optional
        Load if exists. The default is True.
    n_jobs : int, optional
        Number of jobs. The default is 10.
    study_name : str, optional
        Study name. The default is "XGB_Optimization".
    early_stopping_rounds : int, optional
        Early stopping rounds. The default is 50.
    params : dict, optional
        Parameters. The default is None (treated as an empty dict).

    Returns
    -------
    study_pre : optuna.study.Study
        Study object.
    '''

    if params is None:
        params = {}

    if verbose:
        ocprint.printv(f"Process {pid} starting optimization")

    # Set direction based on X_val
    direction = "maximize" if X_val is None else "minimize"

    # Create the XGBoostOptimizer object
    xgb = XGBoostOptimizer(
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        storage = storage,
        params = params,
        use_gpu = use_gpu,
        early_stopping_rounds = early_stopping_rounds,
        random_state = random_seed,
        verbose = verbose
    )

    # Run the pre-optimization for XGBoost
    study_pre = _run_optuna_op_with_retry(
        lambda: xgb.optimize(
            direction = direction,
            n_trials = n_trials,
            n_jobs = n_jobs,
            study_name = f"{study_name}_{id}",
            load_if_exists = load_if_exists,
        ),
        pid=pid,
        verbose=verbose,
        context=f"{study_name}_{id}",
    )

    if verbose:
        ocprint.printv(f"Process {pid} has completed the optimization")

    return study_pre
