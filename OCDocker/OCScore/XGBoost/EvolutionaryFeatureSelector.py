import optuna

import cupy as cp
import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from numpy.random import default_rng
from optuna.samplers import CmaEsSampler, TPESampler
from sklearn.metrics import auc, roc_curve
from typing import Union
from urllib.parse import quote_plus

#from OCDocker.Initialise import *

import OCxgboost

class EvolutionaryFeatureSelector:
    """
    A class to optimize the feature selection for XGBoost using an evolutionary algorithm.
    """

    def __init__(self,
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            xgboost_params: dict,
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            evolution_params: dict = {},
            algorithm: str = "cmaes",
            early_stopping_rounds : int = 20,
            use_gpu: bool = False,
            random_state: int = 42,
            fixed_features_index: list = [],
            verbose: bool = False
        ) -> None:
        '''
        Constructor for the EvolutionaryAlgorithmOptimizer class.

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
        evolution_params : dict, optional
            The hyperparameters for the evolutionary algorithm. Default is an empty dictionary.
        algorithm : str, optional
            The algorithm to be used for the evolutionary algorithm. Default is "cmaes".
        early_stopping_rounds : int, optional
            The number of early stopping rounds for the XGBoost model. Default is 50.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model.
        random_state : int, optional
            The random state for the XGBoost model. Default is 42.
        fixed_features_index : list, optional
            The indexes of the features to be always set as true. Default is an empty list (all features can be turned off).
        verbose : bool, optional
            Whether to print the training logs. Default is False.
        '''

        # Set the class variables converting to numpy arrays
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.xgboost_params = xgboost_params
        self.X_validation = np.asarray(X_validation)
        self.y_validation = np.asarray(y_validation)
        self.evolution_params = evolution_params
        self.early_stopping_rounds = early_stopping_rounds
        self.algorithm = algorithm
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.fixed_features_index = fixed_features_index
        self.verbose = verbose

        # If use_gpu is True, set the device to 'cuda'
        if use_gpu:
            self.xgboost_params['device'] = 'cuda'
        
        if "tree_method" not in xgboost_params:
            xgboost_params["tree_method"] = "hist"

        if "objective" not in xgboost_params:
            xgboost_params["objective"] = "reg:squarederror"

        if "booster" not in xgboost_params:
            xgboost_params["booster"] = "gbtree"

        if "random_state" not in xgboost_params:
            xgboost_params["random_state"] = self.random_state

        if "eval_metric" not in xgboost_params:
            if self.X_validation is not None:
                xgboost_params["eval_metric"] = 'rmse'
            else:
                xgboost_params["eval_metric"] = 'auc'
        
        # Set the storage string for the study
        self.storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
    
    def __create_individual(self) -> list:
        ''' 
        A function to create an individual for the genetic algorithm.

        Returns
        -------
        list
            A binary list representing the inclusion (1) or exclusion (0) of each feature.
        '''

        # Create a binary list representing the inclusion (1) or exclusion (0) of each feature
        individual = [1 if i in self.fixed_features_index else bool(self.rng.random() < 0.5) for i in range(self.X_train.shape[1])]

        # While individual is all False, create a new one to prevent all features being turned off
        while not any(individual):
            print("All features are turned off, creating a new individual...")
            individual = [1 if i in self.fixed_features_index else bool(self.rng.random() < 0.5) for i in range(self.X_train.shape[1])]

        # Return the individual
        return individual

    def __custom_mutFlipBit(self, individual: list, indpb: float, fixed_features_index: list) -> tuple[list]:
        '''
        Custom mutation that flips bits with a probability of indpb, excluding fixed features.

        Parameters
        ----------
        individual : list
            The individual to be mutated.
        indpb : float
            The probability of mutation.
        fixed_features_index : list
            The indexes of the features to be always set as true.

        Returns
        -------
        tuple[list]
            The mutated individual.
        '''

        # For each feature
        for i in range(len(individual)):
            # If the feature is not fixed
            if i not in fixed_features_index:
                # Flip the bit with a probability of indpb
                if self.rng.random() < indpb:
                    individual[i] = type(individual[i])(not individual[i])
        
        return individual,

    def fitness(self, individual: list) -> tuple:
        """
        A function to calculate the fitness of a set of features represented by an individual.

        Parameters
        ----------
        individual : list
            A binary list representing the inclusion (1) or exclusion (0) of each feature.

        Returns
        -------
        tuple
            The AUC score of the selected features.
        """

        # Determine which features to include based on the individual's genes
        selected_features_indices = [i for i, use_feature in enumerate(individual) if use_feature]

        # Filter the datasets to include only the selected features
        X_train_filtered = self.X_train[:, selected_features_indices]
        X_test_filtered = self.X_test[:, selected_features_indices]

        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Train the model and get the AUC score
            model, rmse = OCxgboost.run_xgboost(X_train_filtered, self.y_train, X_test_filtered, self.y_test, params = self.xgboost_params, verbose = self.verbose) # type: ignore

            # Filtrate the validation dataset
            X_validation_filtered = self.X_validation[:, selected_features_indices]

            # Predict the validation dataset
            y_pred = model.predict(X_validation_filtered)

            # Get the AUC score of the validation dataset
            fpr, tpr, _ = roc_curve(self.y_validation, y_pred) # type: ignore

            # Calculate the AUC score
            roc_auc = auc(fpr, tpr)

            # Return the AUC score and the RMSE
            return rmse, roc_auc
        else:
            # Use the provided XGBoost function to train the model and get the AUC score
            _, roc_auc = OCxgboost.run_xgboost(
                X_train_filtered, 
                self.y_train, 
                X_test_filtered, 
                self.y_test, 
                self.xgboost_params, 
                verbose = self.verbose
            )

            # Return the AUC score as a tuple
            return roc_auc,

    def objective_GA(self, trial: optuna.Trial) -> float:
        '''
        The objective function for the Optuna optimization using the Genetic Algorithm.
        
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

        # Check if the evolution_params are already defined, if not, suggest them
        if "population_size" not in trial_params:
            trial_params['population_size'] = trial.suggest_int('population_size', 10, 30)
        if "n_generations" not in trial_params:
            trial_params['n_generations'] = trial.suggest_int('n_generations', 10, 50)
        if "cxpb" not in trial_params:
            trial_params['cxpb'] = trial.suggest_float('cxpb', 0.1, 0.9)
        if "mutpb" not in trial_params:
            trial_params['mutpb'] = trial.suggest_float('mutpb', 0.1, 0.9)
        if "indpb" not in trial_params:
            trial_params['indpb'] = trial.suggest_float('indpb', 0.05, 0.5)
        if "tournsize" not in trial_params:
            trial_params['tournsize'] = trial.suggest_int('tournsize', 2, max(2, int(trial_params['population_size'] * 0.5)))

        # Set the early stopping rounds
        self.xgboost_params['early_stopping_rounds'] = self.early_stopping_rounds

        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Check if the DEAP creator classes are already defined to prevent errors
            if "FitnessMax" not in dir(creator):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

            if "Individual" not in dir(creator):
                creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            # Check if the DEAP creator classes are already defined to prevent errors
            if "FitnessMax" not in dir(creator):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))

            if "Individual" not in dir(creator):
                creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create the toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self.rng.integers, 0, 1)
        toolbox.register("individual", tools.initIterate, creator.Individual, self.__create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.__custom_mutFlipBit, indpb = trial_params['indpb'], fixed_features_index = self.fixed_features_index)
        toolbox.register("select", tools.selTournament, tournsize = trial_params['tournsize'])

        # Register the statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", min)

        # Create the population
        population = toolbox.population(n = trial_params['population_size'])

        # Initialize the hall of fame
        hof = tools.HallOfFame(1)

        # Execute the evolution
        _, logbook = algorithms.eaSimple(
            population, 
            toolbox, 
            cxpb = trial_params['cxpb'], 
            mutpb = trial_params['mutpb'], 
            ngen = trial_params['n_generations'], 
            stats = stats, 
            halloffame = hof,
            verbose = self.verbose
        )

        # Get the best individual
        best_individual = hof[0]

        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Return the AUC score
            best_score_rmse, best_score_auc = best_individual.fitness.values[0]
            # Logging AUC and best individual's feature indices
            trial.set_user_attr("AUC", best_score_auc)
            # Logging AUC and best individual's feature indices
            trial.report(best_score_rmse, trial.number)
        else:
            best_score_auc = best_individual.fitness.values[0]
            # Logging AUC and best individual's feature indices
            trial.report(best_score_auc, trial.number)

        # Logging the best individual's feature indices
        selected_features_indices = [i for i, use_feature in enumerate(best_individual) if use_feature]
        trial.set_user_attr("selected_features_indices", selected_features_indices)
        trial.set_user_attr("selected_features_length", len(selected_features_indices))

        # Logging statistics from logbook
        stats_to_log = ['max', 'avg', 'std', 'min']
        for stat in stats_to_log:
            trial.set_user_attr(f"{stat}_AUC", logbook.select(stat)[-1])

        # If the validation dataset is provided, return the RMSE score
        if self.X_validation is not None:
            return best_score_rmse
        
        # Return the AUC score
        return best_score_auc

    def objective_CMA(self, trial: optuna.Trial) -> float:
        """
        Objective function to optimize using Optuna, evaluating the performance of feature selection for an XGBoost model using CMA-ES from DEAP.

        Parameters
        ----------
        trial : optuna.Trial
            The trial object from Optuna.

        Returns
        -------
        float
            The AUC score of the model trained with the selected features.
        """

        # Create a local copy of evolution_params for this trial to prevent side-effects
        trial_params = self.evolution_params.copy()

        # Check if the evolution_params are already defined, if not, suggest them
        if "sigma" not in trial_params:
            trial_params["sigma"] = trial.suggest_float("sigma", 0.5, 2.0) # Step size
        if "lambda_" not in trial_params:
            trial_params["lambda_"] = trial.suggest_int("lambda_", 10, 100) # Number of offspring
        if "n_gen" not in trial_params:
            trial_params["n_gen"] = trial.suggest_int("n_gen", 10, 100) # Number of generations
        if "population_size" not in trial_params:
            trial_params['population_size'] = trial.suggest_int('population_size', 10, 30)
        
        # Generate individual with Optuna
        individual = []
        for i in range(self.X_train.shape[1]):
            if i in self.fixed_features_index:  # Check if the feature is fixed
                individual.append(1)  # Fixed features are always included
            else:
                # Suggest the inclusion of each feature
                feature_inclusion = trial.suggest_int(f"feature_inclusion_{i}", 0, 1)
                individual.append(feature_inclusion)

        # Filtrate the X and y datasets
        selected_features_indices = [i for i, use_feature in enumerate(individual) if use_feature]
        X_train_filtered = self.X_train[:, selected_features_indices]
        X_test_filtered = self.X_test[:, selected_features_indices]
        
        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Filtrate the validation dataset
            x_validation_filtered = self.X_validation[:, selected_features_indices]

            # Train the model and get the RMSE score
            model, rmse = OCxgboost.run_xgboost(X_train_filtered, self.y_train, X_test_filtered, self.y_test, params = self.xgboost_params, verbose = self.verbose) # type: ignore

            # Predict the validation dataset
            y_pred = model.predict(x_validation_filtered)

            # Get the False Positive Rate and True Positive Rate
            fpr, tpr, _ = roc_curve(self.y_validation, y_pred)

            # Calculate the AUC score
            roc_auc = auc(fpr, tpr)

            # Logging AUC and best individual's feature indices
            trial.add_user_attr("AUC", roc_auc)

            # Return the RMSE score
            return rmse
        
        # Run xgboost with the selected features
        _, roc_auc = OCxgboost.run_xgboost(X_train_filtered, self.y_train, X_test_filtered, self.y_test, self.xgboost_params, self.verbose)

        return roc_auc

    def optimize(self, direction: str = "maximize", n_trials: int = 1000,  n_jobs: int = 1, study_name: str = "Evolutionary Algorithm for OCDocker", load_if_exists: bool = True) -> tuple[optuna.study.Study, dict, float]:
        '''
        A function to optimize the feature selection using the evolutionary algorithm using Optuna.

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

        Returns
        -------
        optuna.study.Study
            The Optuna study object.
        dict
            The best hyperparameters.
        float
            The best AUC score.
        '''

        # If the algorithm is CMA-ES, use the objective_CMA function, otherwise use the objective_GA function
        if self.algorithm == "cmaes":
            # Create the Sampler
            sampler = CmaEsSampler(seed = self.random_state)

            # Set the objective function
            oFunc = self.objective_CMA
        else:
            # Create the Sampler
            sampler = TPESampler(seed = self.random_state)

            # Set the objective function
            oFunc = self.objective_GA
        
        # Create an Optuna study
        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler
        )

        study.optimize(oFunc, n_trials = n_trials, n_jobs = n_jobs)

        # Get the best hyperparameters and the best score
        best_params = study.best_params
        best_score = study.best_value

        print(f"Best AUC score: {best_score}")
        print(f"Best hyperparameters: {best_params}")

        return study, best_params, best_score
