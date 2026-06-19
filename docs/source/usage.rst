Usage
=====

This section summarizes common CLI tasks, options, and environment variables.

CLI overview
------------

.. code-block:: bash

   ocdocker --help
   ocdocker <command> --help

Commands
--------

- Optional dependencies: see :doc:`optional_dependencies` for the install cheat
  sheet and full extra reference. Minimal commands:

  .. code-block:: bash

     pip install "ocdocker[docking]"    # vs / pipeline
     pip install "ocdocker[db]"         # --store-db
     pip install "ocdocker[ml]"         # OCScore ML

- vs: Dock a single receptor/ligand/box with one engine (vina/smina/plants)

  .. code-block:: bash

     ocdocker vs \\
       --engine vina \\
       --receptor path/to/receptor.pdb \\
       --ligand path/to/ligand.smi \\
       --box path/to/box0.pdb \\
       --timeout 600 \\
       --store-db

- pipeline: Multi-engine docking + clustering + rescoring

  .. code-block:: bash

     ocdocker pipeline \\
       --receptor path/to/receptor.pdb \\
       --ligand path/to/ligand.sdf \\
       --box path/to/box0.pdb \\
       --engines vina,smina,plants \\
       --outdir runs/exp1 \\
       --timeout 900 \\
       --store-db

- ocscore: Staged OCScore ML pipeline (raw preparation -> train-only feature reduction -> Optuna -> export tools)

  Requires ``pip install "ocdocker[ml]"``.

  **Full step-by-step replication guide:** :doc:`ocscore_replication` (also `OCSCORE_REPLICATION.md` in the repo root).

  **Bundled protocol YAML and feature-policy data:** :doc:`OCDocker.OCScore.Protocols`.

  Minimal commands:

  .. code-block:: bash

     # Raw input preparation - inputs: .csv, directory, or tar.gz (PDBbind.csv / DUDEz.csv / pipeline_results.csv)
     ocdocker ocscore reduce \\
       --pdbbind-archive path/to/PDBbind.csv \\
       --dudez-archive path/to/DUDEz.csv \\
       --output-dir path/to/raw_prepare

     # Staged Optuna (development protocol)
     ocdocker ocscore train \\
       --protocol development \\
       --raw-input-dir path/to/raw_prepare \\
       --output-dir path/to/optuna_out

     # Feature-policy ablation; bundled .yml policies live in OCDocker/OCScore/Protocols/Ablations/
     ocdocker ocscore train \
       --protocol production \
       --raw-input-dir path/to/raw_prepare \
       --feature-policy ligand_plus_scoring_function_no_pmi \
       --feature-policy ligand_plus_scoring_function_no_plants \
       --feature-policy ligand_plus_scoring_function_no_shape_size_no_autocorr2d \
       --feature-policy ligand_plus_scoring_function_clean_receptor \
       --output-dir path/to/new_ablations

     # Same workflow through the full-pipeline shell runner; the policies above
     # are listed in examples/18_run_full_pipeline.sh::FEATURE_POLICY_ABLATIONS.
     ./examples/18_run_full_pipeline.sh

     # Score new pipeline data with an exported best_model bundle
     ocdocker ocscore score \\
       --export-dir path/to/replica_000/dudez_optuna/best_model \\
       --raw-archive path/to/new_pipeline.csv \\
       --output-csv path/to/predictions.csv

     ocdocker ocscore --help

- console: Interactive console with tab-completion and history

  Launch via ``ocdocker console`` or ``python -m OCDocker.Console``.
  Importing :mod:`OCDocker.Console` is side-effect-free. API reference:
  :doc:`OCDocker.Console`.

  .. code-block:: bash

     ocdocker console --conf OCDocker.cfg
     ocdocker console --conf OCDocker.cfg --ipython

  Inside the console:

  .. code-block:: pycon

     >>> print_args()                 # environment overview
     >>> print_args('all')            # all sections
     >>> print_args('vina')           # also: smina, plants, gnina, oddt, db, paths

- doctor: Diagnostics for binaries/deps/DB

  .. code-block:: bash

     ocdocker doctor --conf OCDocker.cfg

- init-config: Create a starter ``OCDocker.cfg`` or ``OCDocker.yml`` from the example

  .. code-block:: bash

     ocdocker init-config --conf OCDocker.cfg
     # or:
     ocdocker init-config --conf OCDocker.yml

- workbench: Validate specs, preflight specs, build run bundles, prepare launch plans, export publication scaffolds, emit starter templates and JSON Schemas, and serve a read-only strict OCScore dashboard over output roots with direct baseline replicas and ``ablation/`` or ``ablations/`` studies. The dashboard reports curated metrics as sortable ablation-table columns, replica status, ablation summaries, dataset/role/metric-filtered figure previews, explicit test/validation/combined generated metric-delta, rank, and replica-stability plots with SVG/CSV export, and separated model-comparison versus selected-model figure sections without launching or stopping runs.

  .. code-block:: bash

     ocdocker workbench template ocscore_study --output study.yml
     ocdocker workbench validate study.yml
     ocdocker workbench check study.yml
     ocdocker workbench build study.yml runs/run-001 --run-id run-001
     ocdocker workbench launch-plan runs/run-001 --script-output runs/run-001/run.sh
     ocdocker workbench export runs/run-001/run_manifest.yml exports/run-001
     # Serve an OCScore output root shaped as train/replica_* plus train/ablations/<study>/replica_*.
     ocdocker workbench serve /data/hd4tb/OCDocker/data/ocdb2/OCScore/output --host 127.0.0.1 --port 8765
     # Open http://127.0.0.1:8765/app after forwarding the port over SSH.
     # Dashboard UI sources: OCDocker/Workbench/static/ (index.html, app.css, app.js).
     # Old adopted Workbench run_manifest.yml smoke folders are reported as unsupported.
     # The dashboard reads /api/ocscore-workspace and stays read-only.
     ocdocker workbench schema ocscore_study --output ocscore_study.schema.json
     ocdocker workbench plan study.yml --run-id run-001 --output plan.json

- manifest: Generate reproducibility manifest JSON with version metadata

  .. code-block:: bash

     ocdocker manifest --conf OCDocker.cfg --output reproducibility_manifest.json

  Programmatic API:

  .. code-block:: python

     import OCDocker.Toolbox.Reproducibility as ocrepro
     manifest = ocrepro.generate_reproducibility_manifest(include_python_packages=False)
     _ = ocrepro.write_reproducibility_manifest("reproducibility_manifest.json")

- script: Run a Python script with OCDocker pre-loaded
  (requires explicit trust opt-in)

  .. code-block:: bash

     ocdocker script --conf OCDocker.cfg --allow-unsafe-exec script.py --arg1 value

- version: Print installed version

  .. code-block:: bash

     ocdocker version

Global options
--------------

All commands accept the following global options:

- ``--conf``: path to ``OCDocker.cfg`` or ``OCDocker.yml``
- ``--multiprocess``: enable multiprocessing for compatible tasks
- ``--no-multiprocess``: disable multiprocessing for compatible tasks
- ``--update-databases``: run DB updates at startup
- ``--output-level``: control log level (0-5)
- ``--overwrite``: allow overwriting outputs when applicable
- ``--log-file``: write logs to a file
- ``--no-stdout-log``: disable logging to stdout
- ``--threads``: scheduler-provided worker count; also accepts ``OCDOCKER_THREADS`` or ``SNAKEMAKE_THREADS``
- ``--tmp-dir``: job-local temporary directory; also accepts ``OCDOCKER_TMP_DIR``


Scheduler and Snakemake usage
-----------------------------

For workflow managers, prefer explicit per-job resources instead of host autodetection:

.. code-block:: bash

   ocdocker \
     --threads 4 \
     --tmp-dir tmp/sample_001 \
     pipeline \
     --receptor input/sample_001/receptor.pdbqt \
     --ligand input/sample_001/ligand.pdbqt \
     --box input/sample_001/box.txt \
     --outdir results/sample_001 \
     --engines vina,smina,plants \
     --strict-engines \
     --done-marker results/sample_001/done.json \
     --log-file logs/sample_001.log \
     --no-stdout-log

``--strict-engines`` makes the command fail if any requested docking engine fails.
``--done-marker`` writes a small JSON completion marker atomically after
``summary.json`` is written. The example Snakefile is available at
``examples/19_Snakefile_ocdocker_pipeline.smk`` and can run either native
``ocdocker`` or the Docker wrapper ``ocd`` through ``--config ocdocker_command=ocd``.
The Snakefile also declares ``examples/envs/ocdocker.yml`` for ``--use-conda`` runs.

For stage-level scheduling, use ``examples/20_Snakefile_ocdocker_granular_pipeline.smk``. It calls ``ocdocker pipeline prepare``, per-engine ``dock``, ``collect``, ``cluster``, ``rescore``, and ``export`` as separate Snakemake rules.

Bootstrap & environment
-----------------------

- Imports are side-effect-free. CLI/application code calls explicit bootstrap before runtime state is used.
- Environment variables:

  - ``OCDOCKER_CONFIG``: config file path
  - ``OCDOCKER_DB_BACKEND`` / ``DB_BACKEND``: select backend (``postgresql``, ``mysql``, ``sqlite``)
  - ``OCDOCKER_SQLITE_PATH``: explicit SQLite database file path
  - ``OCDOCKER_TIMEOUT``: default timeout (seconds) for external tools
  - ``OCDOCKER_SKIP_ODDT``: skip importing ODDT during bootstrap
  - ``OCDOCKER_ALLOW_SCRIPT_EXEC``: allow trusted in-process script execution
  - ``OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION``: allow trusted pickle/joblib/torch deserialization

- For trusted scripts that need deserialization, use:

  .. code-block:: python

     from OCDocker.Toolbox.Security import allow_unsafe_runtime
     allow_unsafe_runtime(deserialization=True, script_exec=False)

Database storage with ``--store-db`` explicitly initializes DB access and creates tables. Missing PostgreSQL/MySQL databases are created only through explicit setup intent; SQLite remains the recommended backend for tests and quick local runs.

See :doc:`OCDocker.Initialise` for details.

Further reading
---------------

- :doc:`OCDocker`
- :doc:`OCDocker.DB`
- :doc:`OCDocker.Docking`
- :doc:`OCDocker.Error`
- :doc:`OCDocker.Initialise`
- :doc:`OCDocker.Ligand`
- :doc:`OCDocker.OCScore`
- :doc:`OCDocker.Processing`
- :doc:`OCDocker.Receptor`
- :doc:`OCDocker.Rescoring`
- :doc:`OCDocker.Toolbox`
