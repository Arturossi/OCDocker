#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the feature selection using the Genetic Algorithm. 

It is imported as:

from OCDocker.OCScore.Dimensionality.GeneticAlgorithm import GeneticAlgorithm
'''

# Imports
###############################################################################

import optuna

import cupy as cp
import numpy as np
import pandas as pd

from numpy.random import default_rng
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from typing import Any, Union

import OCDocker.OCScore.XGBoost.OCxgboost as OCxgboost
import OCDocker.Toolbox.Printing as ocprint

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


class GeneticAlgorithm:
    '''
    A class to optimize the feature selection for XGBoost using a genetic algorithm.
    '''


    def __init__(self, 
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            xgboost_params: dict,
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            storage: str = "sqlite:///GA.db",
            evolution_params: Union[dict, None] = None,
            use_gpu: bool = False,
            early_stopping_rounds : int = 20,
            random_state: int = 42,
            fixed_features_index: Union[list, None] = None,
            verbose: bool = False
        ) -> None:
        '''
        Constructor for the GeneticAlgorithm class.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame | pd.Series
            The full training dataset.
        y_train : np.ndarray | pd.DataFrame | pd.Series
            The training labels.
        X_test : np.ndarray | pd.DataFrame | pd.Series
            The full test dataset.
        y_test : np.ndarray | pd.DataFrame | pd.Series
            The test labels.
        xgboost_params : dict
            The hyperparameters for the XGBoost model.
        X_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation dataset and labels. Default is None.
        y_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation labels. Default is None.
        evolution_params : dict, None, optional
            The hyperparameters for the genetic algorithm. Default is None.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model.
        random_state : int, optional
            The random state for the XGBoost model. Default is 42.
        fixed_features_index : list, None, optional
            The indexes of the scores to be used for the evaluation. Default is None.
        '''

        # Set the class variables converting to numpy arrays

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.xgboost_params = xgboost_params
        self.X_validation = np.asarray(X_validation) if X_validation is not None else None
        self.y_validation = np.asarray(y_validation) if y_validation is not None else None
        self.evolution_params = evolution_params if evolution_params is not None else {}
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.fixed_features_index = fixed_features_index if fixed_features_index is not None else []
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.direction = None
        self.use_gpu = use_gpu

        if use_gpu:
            self.xgboost_params['device'] = 'cuda'
            self.X_train = cp.asarray(self.X_train)
            self.y_train = cp.asarray(self.y_train)
            self.X_test = cp.asarray(self.X_test)
            self.y_test = cp.asarray(self.y_test)
        
        if "tree_method" not in xgboost_params:
            self.xgboost_params["tree_method"] = "hist"

        if "objective" not in xgboost_params:
            self.xgboost_params["objective"] = "reg:squarederror"

        if "booster" not in xgboost_params:
            self.xgboost_params["booster"] = "gbtree"

        if "eval_metric" not in xgboost_params:
            if self.X_validation is not None:
                self.xgboost_params["eval_metric"] = 'rmse'
            else:
                self.xgboost_params["eval_metric"] = 'auc'

        if "random_state" not in xgboost_params:
            self.xgboost_params["random_state"] = self.random_state
        

        self.storage = storage









    def fitness(self, individual: list) -> tuple:
        '''
        A function to calculate the fitness of a set of features represented by an individual.

        Parameters
        ----------
        individual : list
            A binary list representing the inclusion (1) or exclusion (0) of each feature.

        Returns
        -------
        tuple
            The metric score of the selected features and the model.
        '''

        # Determine which features to include based on the individual's genes
        selected_features_indices = np.where(individual)[0]

        # Filter the datasets to include only the selected features
        X_train_filtered = self.X_train[:, selected_features_indices]
        X_test_filtered = self.X_test[:, selected_features_indices]

        if self.use_gpu:
            X_train_filtered = cp.asarray(X_train_filtered)
            X_test_filtered = cp.asarray(X_test_filtered)

        # Train the model and get the AUC score
        model, metric = OCxgboost.run_xgboost(X_train_filtered, self.y_train, X_test_filtered, self.y_test, params = self.xgboost_params, verbose = self.verbose) # type: ignore

        # Return the metric score and the model

        return metric, model








    def initialize_population(self, number_of_features: int, population_size: int) -> np.ndarray:
        '''
        A function to initialize the population for the genetic algorithm.

        Parameters
        ----------
        number_of_features : int
            The number of features in the dataset.
        population_size : int
            The size of the population.

        Returns
        -------
        np.ndarray
            The initialized population.
        '''

        # Create the initial population with a random selection of True/False for each feature
        population = self.rng.choice([False, True], size=(population_size, number_of_features))

        # For each individual in the population, ensure that fixed features are always included
        # and at least one feature is True
        for individual in population:
            # Ensure fixed features are set to True
            for index in self.fixed_features_index:
                individual[index] = True
            
            # Check if at least one feature is True, if not, randomly select one (non-fixed, if possible) to set to True
            if not individual.any():
                # Attempt to choose a non-fixed feature if possible
                non_fixed_indices = [i for i in range(number_of_features) if i not in self.fixed_features_index]
                if non_fixed_indices:
                    random_index = self.rng.choice(non_fixed_indices)
                else:
                    # If all features are fixed, choose from all features
                    random_index = self.rng.integers(0, number_of_features)
                individual[random_index] = True

        assert all(individual.shape[0] == number_of_features for individual in population), "Inconsistent gene size detected in initialize pop."


        return population








    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        '''
        A function to perform tournament selection for the genetic algorithm.

        Parameters
        ----------
        population : np.ndarray
            The current population.
        fitnesses : np.ndarray
            The fitness scores of the population.
        tournament_size : int, optional
            The size of the tournament. Default is 3.

        Returns
        -------
        np.ndarray
            The selected individual.
        '''

        # Select a random subset of the population
        selected_indices = self.rng.choice(range(len(population)), size = tournament_size, replace = False)

        # Get the fitness scores of the selected individuals
        selected_fitnesses = fitnesses[selected_indices]

        # If the direction in study is minimize
        if self.direction == "minimize":
            # Get the individual with the lowest fitness score
            winner_index = selected_indices[np.argmin(selected_fitnesses)]
        else:
            # Get the individual with the highest fitness score
            winner_index = selected_indices[np.argmax(selected_fitnesses)]

        # Return the selected individual

        return population[winner_index]








    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        '''
        A function to perform crossover for the genetic algorithm.
        
        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The child individual.
        '''

        # Select a random crossover point
        crossover_point = self.rng.integers(low = 0, high = len(parent1))

        # Create the child individual by combining the parents
        child = np.hstack([parent1[:crossover_point], parent2[crossover_point:]])

        # Return the child individual

        return child








    def mutation(self, individual: np.ndarray, mutation_rate: float = 0.05) -> np.ndarray:
        '''
        A function to perform mutation for the genetic algorithm.

        Parameters
        ----------
        individual : np.ndarray
            The individual to be mutated.
        mutation_rate : float, optional
            The mutation rate. Default is 0.05.

        Returns
        -------
        np.ndarray
            The mutated individual.
        '''

        # Perform mutation for each feature in the individual
        for i in range(len(individual)):
            # If it is a score column, do not mutate
            if i in self.fixed_features_index:
                continue
            # If the mutation rate is less than the mutation rate, flip the feature
            if self.rng.random() < mutation_rate:
                # Flip the feature
                individual[i] = not individual[i]

        # Return the mutated individual

        return individual








    def genetic_algorithm(self, trial_params: dict, trial: Any) -> tuple[np.ndarray, OCxgboost.XGBRegressor, float, Union[None, float]]:
        '''
        A function to perform the genetic algorithm for feature selection.

        Parameters
        ----------
        number_of_generations : int
            The number of generations.
        population_size : int
            The size of the population.
        mutation_rate : float
            The mutation rate.

        Returns
        -------
        np.ndarray
            The selected features.
        XGBRegressor
            The model.
        float
            The score of the selected features.
        Union[None, float]
            The AUC score of the selected features. If the validation dataset is not provided, None is returned.
        '''

        # Get the total number of features
        number_of_features = self.X_train.shape[1]

        # Initialize the population
        population = self.initialize_population(number_of_features, trial_params['population_size'])

        # Initialize the best score and the best individual
        if self.direction == "minimize":
            best_score = np.inf
        else:
            best_score = 0

        best_individual = None
        best_score2 = None
        best_model = None

        # Perform the genetic algorithm for the specified number of generations
        for generation in tqdm(range(trial_params['number_of_generations'])):
            # Create lists to store the fitness scores and the models
            fitnesses = []
            models = []

            # For each individual in the population
            for individual in population:
                # Get the fitness score and the model
                f = self.fitness(individual)

                # Append the fitness score and the model to the lists
                fitnesses.append(f[0])
                models.append(f[1])

                # If verbose, print the fitness score and the number of features
                if self.verbose:
                    ocprint.printv(f"{f[0]} - {len(individual.nonzero()[0])} - {f[1].n_features_in_}")
                
            # Convert to numpy arrays
            fitnesses = np.array(fitnesses)
            models = np.array(models)

            # If the direction in study is minimize
            if self.direction == "minimize":
                # Get the best score in the current generation
                best_score_in_generation = np.min(fitnesses)

                # Get the index of the best score
                best_score_index = np.argmin(fitnesses)

                # Set the has_better_score flag to best_score_in_generation < best_score
                has_better_score = best_score_in_generation < best_score
            else:
                # Get the best score in the current generation
                best_score_in_generation = np.max(fitnesses)

                # Get the index of the best score
                best_score_index = np.argmax(fitnesses)

                # Set the has_better_score flag to best_score_in_generation > best_score
                has_better_score = best_score_in_generation > best_score

            if self.verbose:
                ocprint.printv(f"pop: {population}")
                ocprint.printv(f"argmin: {np.argmin(fitnesses)}")
                ocprint.printv(f"argmax: {np.argmax(fitnesses)}")
                ocprint.printv(f"best_score: {best_score_in_generation}")
                ocprint.printv(f"best_score_index: {best_score_index}")

            # Report the best score in the current generation
            trial.report(best_score_in_generation, generation)

            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            # If the best score in the current generation is better than the best score so far, update the best score and the best individual
            if has_better_score:
                # Update the best score and the best individual
                best_score = best_score_in_generation

                # Get the best individual
                best_individual = population[best_score_index]

                # Get the best model (loaded from pickle file)
                best_model = models[best_score_index] 

                # If the validation dataset is provided (if X_validation is not None y_validation is not None as well)
                if self.X_validation is not None:
                    # Filter the validation dataset to include only the selected features
                    X_validation_filtered = self.X_validation[:, best_individual.nonzero()[0]]

                    if self.use_gpu:
                        X_validation_filtered = cp.asarray(X_validation_filtered)
                    
                    # Predict the validation dataset
                    y_pred = best_model.predict(X_validation_filtered)

                    if self.use_gpu:
                        # Take the cupy array to numpy
                        y_pred = cp.asnumpy(y_pred)

                    # Get the AUC score of the validation dataset (self.validation is not none here because of the if statement above)
                    fpr, tpr, _ = roc_curve(self.y_validation, y_pred) # type: ignore

                    # Calculate the AUC score
                    best_score2 = auc(fpr, tpr)

                    if self.verbose:
                        # Print the AUC score
                        ocprint.printv(f"Generation {generation}:\nBest score = {best_score}\nBest score AUC = {best_score2}")
                elif self.verbose:
                    # Print the best score
                    ocprint.printv(f"Generation {generation}: Best score = {best_score}")
            
            # Create a new population
            new_population = []

            # Perform crossover and mutation to create the new population using the current population pairs by pairs
            for _ in range(trial_params['population_size'] // 2):
                # Select the parents using tournament selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = None

                # Ensure that parent2 is different from parent1
                while parent2 is None or np.array_equal(parent2, parent1):
                    parent2 = self.tournament_selection(population, fitnesses)
                
                # Perform crossover and mutation to create 2 children
                child1 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, trial_params['mutation_rate'])

                child2 = self.crossover(parent2, parent1)
                child2 = self.mutation(child2, trial_params['mutation_rate'])

                assert parent1.shape[0] == parent2.shape[0], "Inconsistent gene size parents mismatch."
                assert child1.shape[0] == child2.shape[0], "Inconsistent gene size childs mismatch."
                assert parent1.shape[0] == child1.shape[0], "Inconsistent gene size parent/child mismatch."

                # Add the children to the new population
                new_population.extend([child1, child2])

            # Update the population
            population = np.array(new_population)

            assert all(individual.shape[0] == number_of_features for individual in population), "Inconsistent gene size detected in entire population."

        # Return the best individual and the best score

        return best_individual, best_model, best_score, best_score2 # type: ignore








    def objective(self, trial: optuna.Trial) -> float:
        '''
        The objective function for the Optuna optimization.
        
        Parameters
        ----------
        trial : optuna.Trial
            The trial object.
            
        Returns
        -------
        float
            The AUC score of the selected features.
        '''

        # Create a local copy of evolution_params for this trial to prevent side-effects
        trial_params = self.evolution_params.copy()

        # Get the hyperparameters for the genetic algorithm
        if "number_of_generations" not in trial_params:
            trial_params["number_of_generations"] = trial.suggest_int('number_of_generations', 20, 100)

        if "population_size" not in trial_params:
            trial_params["population_size"] = trial.suggest_int('population_size', 20, 200)

        if "mutation_rate" not in trial_params:
            trial_params["mutation_rate"] = trial.suggest_float('mutation_rate', 0.01, 0.2)

        # Add the early stopping rounds to the trial_params
        trial_params['early_stopping_rounds'] = self.early_stopping_rounds

        # Perform the genetic algorithm
        best_individual, model, best_score, best_score2 = self.genetic_algorithm(trial_params, trial)

        # Pickle the best individual
        trial.set_user_attr('best_individual', ''.join([str(int(i)) for i in best_individual.tolist()]))

        # If the validation dataset is provided
        if self.X_validation is not None:
            # Set the best AUC score as a user attribute
            trial.set_user_attr('best_AUC', best_score2)

        # Return the AUC score

        return best_score








    def optimize(self, direction: str = "maximize", n_trials: int = 100,  n_jobs: int = 1, study_name: str = "Genetic Algorithm for descriptor optimization", load_if_exists: bool = True, verbose: bool = False) -> tuple[optuna.study.Study, dict, float]:
        '''
        A function to optimize the feature selection using the genetic algorithm using Optuna.

        Parameters
        ----------
        direction : str, optional
            The direction of the optimization. Default is "maximize".
        n_trials : int, optional
            The number of trials. Default is 100.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is 1.
        study_name : str, optional
            The name of the study. Default is "Genetic Algorithm for descriptor optimization".
        load_if_exists : bool, optional
            Whether to load the study if it exists. Default is True.
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Returns
        -------
        optuna.study.Study
            The Optuna study object.
        dict
            The best hyperparameters.
        float
            The best score.
        '''

        # Set the direction
        self.direction = direction

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction = direction, study_name = study_name, storage = self.storage, load_if_exists = load_if_exists)

        # Optimize the objective function
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)

        # Get the best hyperparameters and the best score
        best_params = study.best_params
        best_score = study.best_value

        if verbose:
            ocprint.printv(f"Best score: {best_score}")
            ocprint.printv(f"Best hyperparameters: {best_params}")

            # If the validation dataset is provided, print the best AUC
            if self.X_validation is not None:
                ocprint.printv(f"Best AUC: {study.best_trial.user_attrs['best_AUC']}")
        
        return study, best_params, best_score

# Methods
###############################################################################
