import optuna

import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from numpy.random import default_rng
from optuna.samplers import CmaEsSampler, TPESampler
from sklearn.metrics import auc, roc_curve
from typing import Union

#from OCDocker.Initialise import *
from sqlalchemy.engine.url import URL

import OCxgboost

class EvolutionaryFeatureSelectorCustom:
    """
    A class to optimize the feature selection for XGBoost using a genetic algorithm.
    """

    def __init__(self, X_train: Union[np.ndarray, pd.DataFrame, pd.Series], y_train: Union[np.ndarray, pd.DataFrame, pd.Series], X_test: Union[np.ndarray, pd.DataFrame, pd.Series], y_test: Union[np.ndarray, pd.DataFrame, pd.Series], xgboost_params: dict, evolution_params: dict = {}, use_gpu: bool = False, early_stopping_rounds : int = 20, random_state: int = 42, fixed_features_index: list = [], verbose: bool = False) -> None:
        '''
        Constructor for the EvolutionaryFeatureSelector class.

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
        params : dict
            The hyperparameters for the XGBoost model.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model.
        random_state : int, optional
            The random state for the XGBoost model. Default is 42.
        fixed_features_index : list, optional
            The indexes of the scores to be used for the evaluation. Default is an empty list.
        '''

        # Set the class variables converting to numpy arrays
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.xgboost_params = xgboost_params
        self.evolution_params = evolution_params
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.fixed_features_index = fixed_features_index
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

        if use_gpu:
            self.xgboost_params['device'] = 'cuda'
        
        if "tree_method" not in xgboost_params:
            self.xgboost_params["tree_method"] = "hist"
        if "objective" not in xgboost_params:
            self.xgboost_params["objective"] = "reg:squarederror"
        if "booster" not in xgboost_params:
            self.xgboost_params["booster"] = "gbtree"
        if "eval_metric" not in xgboost_params:
            self.xgboost_params["eval_metric"] = "auc"
        if "random_state" not in xgboost_params:
            self.xgboost_params["random_state"] = self.random_state
        
        # Set the storage string for the study
        self.storage = str(URL.create(
            drivername = 'mysql+pymysql',
            username   = "ocdocker",
            password   = "@Kp3sRv9t@",
            host       = "localhost",
            port       = "3306",
            database   = "feature_selection"
        ))

    def fitness_function(self, features: list) -> float:
        '''
        A function to calculate the fitness of a set of features.

        Parameters
        ----------
        features : list
            The indices of the features to be used.
        trial_params : dict
            The hyperparameters for the XGBoost model.

        Returns
        -------
        float
            The AUC score of the XGBoost model using the selected features.
        '''

        # Select the columns from the full dataset based on the features index
        filtered_X_train = self.X_train[:, features]
        filtered_X_test = self.X_test[:, features]

        # Train the model and get the AUC score
        _, roc_auc = OCxgboost.run_xgboost(filtered_X_train, self.y_train, filtered_X_test, self.y_test, self.xgboost_params, self.verbose) # type: ignore

        # Return the AUC score
        return roc_auc

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

    def genetic_algorithm(self, trial_params: dict) -> tuple[np.ndarray, float]:
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
        float
            The AUC score of the selected features.
        '''

        # Get the total number of features
        number_of_features = self.X_train.shape[1]

        # Initialize the population
        population = self.initialize_population(number_of_features, trial_params['population_size'])

        # Initialize the best score and the best individual
        best_score = 0
        best_individual = None

        # Perform the genetic algorithm for the specified number of generations
        for generation in tqdm(range(trial_params['number_of_generations'])):
            # Calculate the fitness scores of the population
            fitnesses = np.array([self.fitness_function(individual.nonzero()[0]) for individual in population])

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

                # Add the children to the new population
                new_population.extend([child1, child2])

            # Update the population
            population = np.array(new_population)

            # Get the best score in the current generation
            best_score_in_generation = np.max(fitnesses)

            # If the best score in the current generation is better than the best score so far, update the best score and the best individual
            if best_score_in_generation > best_score:
                # Update the best score and the best individual
                best_score = best_score_in_generation
                # Get the best individual
                best_individual = population[np.argmax(fitnesses)]
                # Print the best score
                print(f"Generation {generation}: Best score = {best_score}")

        # Return the best individual and the best score
        return best_individual, best_score

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
        best_individual, best_score = self.genetic_algorithm(trial_params)

        # Pickle the best individual
        trial.set_user_attr('best_individual', best_individual)

        # Return the AUC score
        return best_score

    def optimize(self, direction: str = "maximize", n_trials: int = 100,  n_jobs: int = 1, study_name: str = "Genetic Algorithm for descriptor optimization", load_if_exists: bool = True) -> tuple[optuna.study.Study, dict, float]:
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
        storage : str, optional
            The storage for the study. Default is "sqlite:///example.db".
        load_if_exists : bool, optional
            Whether to load the study if it exists. Default is True.

        Returns
        -------
        optuna.study.Study
            The Optuna study object.
        dict
            The best hyperparameters.
        float
            The best AUC score.
        '''

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction = direction, study_name = study_name, storage = self.storage, load_if_exists = load_if_exists)

        # Optimize the objective function
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)

        # Get the best hyperparameters and the best score
        best_params = study.best_params
        best_score = study.best_value

        print(f"Best AUC score: {best_score}")
        print(f"Best hyperparameters: {best_params}")

        return study, best_params, best_score
