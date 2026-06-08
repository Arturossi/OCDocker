OCScore Pipeline Examples
=========================

These examples demonstrate how to train and use OCScore models for consensus scoring.

Complete OCScore Pipeline
--------------------------

Complete end-to-end pipeline to obtain OCScore results from scratch:

.. literalinclude:: ../../../examples/11_python_api_complete_ocscore_pipeline.py
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

Inference from CSV
------------------

Example of OCScore inference loading features directly from a CSV file:

.. literalinclude:: ../../../examples/13_python_api_inference_from_csv.py
   :language: python
   :caption: OCScore inference from CSV

This script demonstrates:

* Loading input features from a ``.csv`` file
* Loading OCDocker config so ``reference_column_order`` is enforced
* Resolving model artifacts from ``OCScore_models`` (or a custom directory)
* Optional mask and scaler loading
* Running model inference and exporting an output CSV with original rows/columns plus ``OCSCORE``

Example command:

.. code-block:: bash

   python examples/13_python_api_inference_from_csv.py \
       --csv-path /path/to/features.csv \
       --model-name OCScore \
       --config-path /path/to/OCDocker.cfg \
       --output-csv /path/to/scored.csv

Staged pipeline CLI
-------------------

Examples 14–16 are also available as ``ocdocker ocscore`` subcommands (requires ``pip install "ocdocker[ml]"``):

.. code-block:: bash

   ocdocker ocscore reduce --pdbbind-archive ... --dudez-archive ... --output-dir ...
   ocdocker ocscore train --protocol development --raw-input-dir ... --output-dir ...
   ocdocker ocscore score --export-dir ... --raw-archive ... --output-csv ...

See :doc:`../usage` for the full command list.

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
