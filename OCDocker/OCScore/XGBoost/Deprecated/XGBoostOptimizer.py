import optuna

import numpy as np
import pandas as pd

from deap import base, cma, creator, tools, algorithms
from numpy.random import default_rng
from optuna.samplers import CmaEsSampler, TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from typing import Union
from urllib.parse import quote_plus

#from OCDocker.Initialise import *

#import OCDocker.OCScore.XGBoost.OCxgboost as OCxgboost
import XGBoost.OCxgboost as OCxgboost

class XGBoostOptimizer:
    def __init__(self, 
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            storage: str = "sqlite:///pre_xgboost.db",
            params: dict = {},
            early_stopping_rounds : int = 20,
            use_gpu = False,
            random_state: int = 42,
            verbose: bool = False
        ):
        """
        Initializes the PreXGBoostOptimizer with training data and configuration.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame | pd.Series
            The training dataset.
        y_train : np.ndarray | pd.DataFrame | pd.Series
            The training labels.
        X_test : np.ndarray | pd.DataFrame | pd.Series
            The test dataset.
        y_test : np.ndarray | pd.DataFrame | pd.Series
            The test labels.
        X_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation dataset and labels. Default is None.
        y_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation labels. Default is None.
        params : dict, optional
            The hyperparameters for the XGBoost model. Default is an empty dictionary.
        early_stopping_rounds : int, optional
            The number of early stopping rounds for the XGBoost model. Default is 50.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model. Default is False.
        load_if_exists : bool, optional
            Whether to load the study if it exists. Default is True.
        random_state : int, optional
            The random state for reproducibility. Default is 42.
        verbose : bool, optional
            Whether to print the training logs. Default is False.
        """

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)

        # If the validation dataset is provided, convert it to numpy arrays
        if X_validation is not None and y_validation is not None:
            self.X_validation = np.asarray(X_validation)
            self.y_validation = np.asarray(y_validation)

            # If the use_gpu flag is set
            if use_gpu:
                import cupy as cp

                # Send the validation data to the GPU
                self.X_validation = cp.asarray(self.X_validation)
                self.y_validation = cp.asarray(self.y_validation)
        else:
            self.X_validation = None
            self.y_validation = None

        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu

        # If use_gpu is True, set the device to 'cuda'
        if use_gpu:
            self.params['device'] = 'cuda'

        # Set the storage string for the study
        self.storage = storage

    def objective(self, trial):
        """
        The objective function for Optuna optimization to tune XGBoost hyperparameters.

        Parameters
        ----------
        trial : optuna.trial._trial.Trial
            A single trial object which suggests hyperparameters.

        Returns
        -------
        float | tuple[float, float]
            The AUC of the model as a result of the suggested hyperparameters. If the validation dataset is provided, returns a tuple of AUC and RMSE.
        """

        # Create a local copy of params for this trial to prevent side-effects
        trial_params = self.params.copy()

        # Check if the hyperparameters are already in the params dictionary, if not, suggest them
        if "max_depth" not in trial_params:
            trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 10)

        if "learning_rate" not in trial_params:
            trial_params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)

        if "n_estimators" not in trial_params:
            trial_params["n_estimators"] = trial.suggest_int("n_estimators", 75, 125)

        if "subsample" not in trial_params:
            trial_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        if "colsample_bytree" not in trial_params:
            trial_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        if "reg_alpha" not in trial_params:
            trial_params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 1.0)

        if "reg_lambda" not in trial_params:
            trial_params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 1.0)

        if "min_child_weight" not in trial_params:
            trial_params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)

        if "gamma" not in trial_params:
            trial_params["gamma"] = trial.suggest_float("gamma", 0.0, 1.0)

        if "tree_method" not in trial_params:
            trial_params["tree_method"] = "hist"

        if "objective" not in trial_params:
            trial_params["objective"] = "reg:squarederror"

        if "booster" not in trial_params:
            trial_params["booster"] = "gbtree"

        if "random_state" not in trial_params:
            trial_params["random_state"] = self.random_state

        if "eval_metric" not in trial_params:
            if self.X_validation is not None:
                trial_params["eval_metric"] = 'rmse'
            else:
                trial_params["eval_metric"] = 'auc'

            # Set validation for pruning based on AUC
            pruning_callback = XGBoostPruningCallback(trial, f"validation_0-{ trial_params['eval_metric'] }")

            # Add the pruning callback to the trial_params
            trial_params['callbacks'] = [ pruning_callback ]

        # Add the early stopping rounds to the trial_params
        trial_params['early_stopping_rounds'] = self.early_stopping_rounds

        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Train the model and get the AUC score
            model, metric = OCxgboost.run_xgboost(self.X_train, self.y_train, self.X_test, self.y_test, params = trial_params, verbose = self.verbose) # type: ignore

            # Predict the validation dataset
            y_pred = model.predict(self.X_validation)

            # If the use_gpu flag is set
            if self.use_gpu:
                # Convert the predictions to numpy arrays
                y_validation_np = self.y_validation.get() # type: ignore
            else:
                y_validation_np = self.y_validation


            # Get the AUC score of the validation dataset
            fpr, tpr, _ = roc_curve(y_validation_np, y_pred) # type: ignore

            # Calculate the AUC score
            roc_auc = auc(fpr, tpr)

            # Save the AUC score as a user attribute
            trial.set_user_attr("AUC", roc_auc)
        
        else:
            # Train the model and get the AUC score
            _, metric = OCxgboost.run_xgboost(self.X_train, self.y_train, self.X_test, self.y_test, params = trial_params, verbose = self.verbose) # type: ignore
    
        # Return the trained AUC score
        return metric

    def optimize(self, direction: str = "minimize", n_trials: int = 1000,  n_jobs: int = 1, study_name: str = "XGBoost pre-optimization", load_if_exists: bool = True) -> optuna.study.Study:
        """
        Optimizes XGBoost hyperparameters using Optuna.

        Parameters
        ----------
        directions : str | list, optional
            The direction of the optimization. Default is "maximize".
        n_trials : int, optional
            The number of trials for Optuna optimization. Default is 100.

        Returns
        -------
        optuna.study.Study
            The Optuna study object.
        n_trials : int, optional
            The number of trials for Optuna optimization. Default is 1000.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is 1.
        study_name : str, optional
            The name of the study. Default is "XGBoost pre-optimization".
        dict
            The best hyperparameters.
        float
            The best AUC score.
        """


        # Create the Sampler
        sampler = TPESampler(seed = self.random_state)

        # Create an Optuna study
        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler
        )

        # Optimize the objective function
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs) # type: ignore

        # Get the best hyperparameters and the best score
        best_params = study.best_params
        best_score = study.best_value

        print(f"Best score: { best_score }")
        print(f"Best hyperparameters: {best_params}")

        return study

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
        if "eval_metric" not in xgboost_params:
            xgboost_params["eval_metric"] = "auc"
        if "random_state" not in xgboost_params:
            xgboost_params["random_state"] = self.random_state
        
        # Set the storage string for the study
        self.storage = str(URL.create(
            drivername = 'mysql+pymysql',
            username   = "ocdocker",
            password   = "@Kp3sRv9t@",
            host       = "%",
            port       = "3306",
            database   = "feature_selection"
        ))
    
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

        # Use the provided XGBoost function to train the model and get the AUC score
        _, roc_auc = OCxgboost.run_xgboost(
            X_train_filtered, 
            self.y_train, 
            X_test_filtered, 
            self.y_test, 
            self.xgboost_params, 
            verbose = self.verbose
        )

        # Check if the validation dataset is provided



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

        # Return the AUC score
        best_score = best_individual.fitness.values[0]

        # Logging evolution parameters and results
        evolution_params = ['population_size', 'n_generations', 'cxpb', 'mutpb', 'indpb', 'tournsize']
        for param in evolution_params:
            trial.set_user_attr(param, trial_params[param])

        # Logging AUC and best individual's feature indices
        trial.set_user_attr("AUC", best_score)

        # Logging the best individual's feature indices
        selected_features_indices = [i for i, use_feature in enumerate(best_individual) if use_feature]
        trial.set_user_attr("selected_features_indices", selected_features_indices)
        trial.set_user_attr("selected_features_length", len(selected_features_indices))

        # Logging statistics from logbook
        stats_to_log = ['max', 'avg', 'std', 'min']
        for stat in stats_to_log:
            trial.set_user_attr(f"{stat}_AUC", logbook.select(stat)[-1])

        # Return the AUC score
        return best_score

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

        # Run xgboost with the selected features
        _, roc_auc = OCxgboost.run_xgboost(X_train_filtered, self.y_train, X_test_filtered, self.y_test, self.xgboost_params, self.verbose)

        # Report the AUC score
        trial.report(roc_auc, trial.number)

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
        self.storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"

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


