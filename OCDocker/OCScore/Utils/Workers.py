#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage I/O operations in OCDocker in the context of scoring 
functions.

They are imported as:

import OCDocker.OCScore.Utils.Workers as ocscoreworkers
'''

# Imports
###############################################################################


import time

from AutoencoderOptimizer import AutoencoderOptimizer
from NNOptimizer import NNOptimizer
from optuna.samplers import TPESampler

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Methods
###############################################################################

def AOworker(
        pid, 
        id,
        X_train,
        X_test, 
        X_val,
        encoding_dims,
        storage,
        models_folder,
        random_seed = 42,
        use_gpu = True, 
        verbose = False, 
        direction = "minimize", 
        n_trials = 250, 
        load_if_exists = True, 
        n_jobs = 10, 
        study_name = "Autoencoder_Optimization"
    ):
    
    print(f"Process {pid} starting optimization")

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
    
    for sampler_name, sampler in [("TPE", TPESampler())]:#, ("CMA", CmaEsSampler())]:
        # Run optimization
        study = trainer.optimize(
                direction = direction, 
                n_trials = n_trials, 
                study_name = f"{study_name}_{id}_{sampler_name}", 
                load_if_exists = load_if_exists, 
                sampler = sampler, 
                n_jobs = n_jobs
        )
        print(f"Process {id} completed {sampler_name} optimization")

    return study

def NNworker(
        pid, id,
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        encoder_params = None,
        output_size = 1, 
        random_seed = 42,
        use_gpu = True, 
        verbose = False, 
        direction = "minimize", 
        n_trials = 250, 
        load_if_exists = True, 
        n_jobs = 10, 
        study_name = "NN_Optimization"
    ):
    print(f"Process {pid} starting optimization")

    # Sleep pid seconds before starting
    time.sleep(pid)

    # Initialize the trainer
    trainer = NNOptimizer(
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        encoder_params,
        output_size = output_size, 
        random_seed = random_seed,
        use_gpu = use_gpu, 
        verbose=verbose
    )

    for sampler_name, sampler in [("TPE", TPESampler())]:#, ("CMA", CmaEsSampler())]:
        # Run optimization
        trainer.optimize(
            direction = direction, 
            n_trials = n_trials, 
            study_name = f"{study_name}_{id}_{sampler_name}", 
            load_if_exists = load_if_exists, 
            sampler = sampler, 
            n_jobs = n_jobs
        )
        print(f"Process {id} completed {sampler_name} optimization")
