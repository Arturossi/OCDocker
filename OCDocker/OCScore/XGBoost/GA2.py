import numpy as np
import optuna

from numpy.random import default_rng
from tqdm import tqdm
from xgboost import XGBRegressor

class GeneticAlgorithmOptimizer:
    """
    A class to optimize the feature selection for XGBoost using a genetic algorithm.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, labels: np.ndarray, params: dict, use_gpu: bool = False, random_state: int = 42, score_columns_index: list = []) -> None:
        '''
        Constructor for the GeneticAlgorithmOptimizer class.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame | pd.Series
            The full training dataset.
        y_train : np.ndarray | pd.DataFrame | pd.Series
            The training labels.
        X_test : np.ndarray | pd.DataFrame | pd.Series
            The full test dataset.
        labels : np.ndarray | pd.DataFrame | pd.Series
            The test labels.
        params : dict
            The hyperparameters for the XGBoost model.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model.
        random_state : int, optional
            The random state for the XGBoost model. Default is 42.
        score_columns_index : list, optional
            The indexes of the scores to be used for the evaluation. Default is an empty list.
        '''

        # If the input is a pandas DataFrame or Series, convert to numpy array
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        if hasattr(labels, 'values'):
            labels = labels.values

        # Set the class variables converting to numpy arrays
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.labels = labels
        self.params = params
        self.random_state = random_state
        self.rng = default_rng(random_state)
        self.score_columns_index = score_columns_index

        if use_gpu:
            self.params['device'] = 'cuda'

    def fitness_function(self, features: list, verbose: bool = False) -> float:
        '''
        A function to calculate the fitness of a set of features.

        Parameters
        ----------
        features : list
            The indices of the features to be used.
        verbose : bool, optional
            Whether to print the training logs. Default is False.

        Returns
        -------
        float
            The AUC score of the XGBoost model using the selected features.
        '''

        # Select the columns from the full dataset based on the features index
        X_train = self.X_train[:, features]
        X_test = self.X_test[:, features]

        # Use the provided XGBoost function to train the model and get the AUC score
        _, roc_auc = self.run_xgboost(X_train, self.y_train, X_test, self.labels, verbose = verbose)

        # Return the AUC score
        return roc_auc

    def run_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, labels: np.ndarray, verbose: bool = False) -> tuple[XGBRegressor, float]:
        '''
        A function to train an XGBoost model and calculate the AUC score.

        Parameters
        ----------
        X_train : np.ndarray
            The training dataset.
        y_train : np.ndarray
            The training labels.
        X_test : np.ndarray
            The test dataset.
        labels : np.ndarray
            The test labels.
        verbose : bool, optional
            Whether to print the training logs. Default is False.

        Returns
        -------
        model : XGBRegressor
            The trained XGBoost model.
        roc_auc : float
            The AUC score of the trained model.
        '''

        # Create the XGBoost model
        model = XGBRegressor(
            objective = 'reg:squarederror',
            booster = 'gbtree',
            tree_method = 'hist',
            eval_metric = 'auc',
            random_state = self.random_state,
            **self.params
        )

        # Train the model and get the AUC score
        model.fit(X_train, y_train, eval_set=[(X_test, labels)], verbose = verbose)

        # Get the AUC score
        evals_result = model.evals_result()
        roc_auc = evals_result['validation_0']['auc'][-1]

        # Return the trained model and the AUC score
        return model, roc_auc

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

        # Create the initial population
        population = self.rng.choice([False, True], size = (population_size, number_of_features))

        # For each individual in the population, ensure that scores are always included
        for individual in population:
            for index in self.score_columns_index:
                individual[index] = True

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
            if i in self.score_columns_index:
                continue
            # If the mutation rate is less than the mutation rate, flip the feature
            if self.rng.random() < mutation_rate:
                # Flip the feature
                individual[i] = not individual[i]

        # Return the mutated individual
        return individual

    def genetic_algorithm(self, number_of_generations: int, population_size: int, mutation_rate: float) -> tuple[np.ndarray, float]:
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
        population = self.initialize_population(number_of_features, population_size)

        # Initialize the best score and the best individual
        best_score = 0
        best_individual = None

        # Perform the genetic algorithm for the specified number of generations
        for generation in tqdm(range(number_of_generations), desc = "Genetic Algorithm"):
            # Calculate the fitness scores of the population
            fitnesses = np.array([self.fitness_function(individual.nonzero()[0]) for individual in population])

            # Create a new population
            new_population = []

            # Perform crossover and mutation to create the new population using the current population pairs by pairs
            for _ in range(population_size // 2):

                # Select the parents using tournament selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = None

                # Ensure that parent2 is different from parent1
                while parent2 is None or np.array_equal(parent2, parent1):
                    parent2 = self.tournament_selection(population, fitnesses)
                
                # Perform crossover and mutation to create 2 children
                child1 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, mutation_rate)

                child2 = self.crossover(parent2, parent1)
                child2 = self.mutation(child2, mutation_rate)

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

        # Get the hyperparameters for the genetic algorithm
        number_of_generations = trial.suggest_int('number_of_generations', 20, 100)
        population_size = trial.suggest_int('population_size', 20, 200)
        mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.2)

        # Perform the genetic algorithm
        best_individual, best_score = self.genetic_algorithm(number_of_generations, population_size, mutation_rate)

        # Pickle the best individual
        trial.set_user_attr('best_individual', best_individual)
        
        # Return the AUC score
        return best_score

    def optimize(self, direction: str = "maximize", n_trials: int = 100, study_name: str = "Genetic Algorithm for descriptor optimization", storage: str = "sqlite:///example.db", load_if_exists: bool = True) -> None:
        '''
        A function to optimize the feature selection using the genetic algorithm using Optuna.

        Parameters
        ----------
        direction : str, optional
            The direction of the optimization. Default is "maximize".
        n_trials : int, optional
            The number of trials. Default is 100.
        study_name : str, optional
            The name of the study. Default is "Genetic Algorithm for descriptor optimization".
        storage : str, optional
            The storage for the study. Default is "sqlite:///example.db".
        load_if_exists : bool, optional
            Whether to load the study if it exists. Default is True.
        '''

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction = direction, study_name = study_name, storage = storage, load_if_exists = load_if_exists)

        # Optimize the objective function
        study.optimize(self.objective, n_trials = n_trials)

        return study
