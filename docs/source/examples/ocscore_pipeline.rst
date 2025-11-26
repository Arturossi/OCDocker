OCScore Pipeline Examples
=========================

These examples demonstrate how to train and use OCScore models for consensus scoring.

Training Models from Database
------------------------------

Example of training machine learning models using data from the OCDocker database:

.. literalinclude:: ../../../examples/11_python_api_train_model_from_db.py
   :language: python
   :caption: Training model from database

This script demonstrates:

* Loading data from the OCDocker database
* Finding best hyperparameters across multiple Optuna studies
* Training DNN or XGBoost models
* Saving trained models and masks
* Using the full preprocessing pipeline

Complete OCScore Pipeline
--------------------------

Complete end-to-end pipeline to obtain OCScore results from scratch:

.. literalinclude:: ../../../examples/12_python_api_complete_ocscore_pipeline.py
   :language: python
   :caption: Complete OCScore pipeline

This script demonstrates:

* Receptor and ligand preparation
* Multi-engine docking (Vina, PLANTS)
* Pose clustering to find representative poses
* Rescoring with multiple scoring functions (ODDT, PLANTS, Vina, SMINA)
* Feature extraction (receptor and ligand descriptors)
* Model inference using trained OCScore model
* Automatic mapping of rescoring results to database column names
* Multiprocessing support for processing multiple ligands

Configuration
-------------

The complete pipeline script includes a configuration section at the top where you can customize:

* Receptor and ligand paths
* Model paths and names
* Preprocessing settings
* Output file paths
* Multiprocessing settings

Example configuration:

.. code-block:: python

   # Receptor configuration
   RECEPTOR_PATH = "/path/to/receptor.pdb"
   RECEPTOR_NAME = "Receptor"
   
   # Ligand configuration
   LIGAND_PATHS = [
       "/path/to/ligand1",
       "/path/to/ligand2",
   ]
   
   # Model configuration
   MODEL_NAME = "OCScore"
   MODELS_DIR = "OCScore_models"
   
   # Multiprocessing
   N_JOBS = 4  # Number of parallel jobs
   USE_MULTIPROCESSING = True

