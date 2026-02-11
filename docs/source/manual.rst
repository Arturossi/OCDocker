Manual
======

Overview
--------

This manual describes the main workflows, inputs and outputs, configuration, and data layout
for OCDocker. For full API reference see :doc:`modules` and the package pages
(for example :doc:`OCDocker.Docking` and :doc:`OCDocker.OCScore`).

Core workflow
-------------

1. Configure external tools and databases in ``OCDocker.cfg`` (or set ``OCDOCKER_CONFIG``).
2. Prepare receptor, ligand, and a binding box.
3. Dock with ``vs`` (single engine) or ``pipeline`` (multi-engine).
4. Optionally rescore, analyze, and store metadata.

Key concepts
------------

- Receptor: a protein structure (usually ``.pdb``) represented by ``OCDocker.Receptor``.
- Ligand: a small molecule (``.smi``, ``.sdf``, ``.mol2``, or ``.pdbqt``) represented by ``OCDocker.Ligand``.
- Box: a PDB file defining the search space via ``REMARK`` lines with center and dimensions.
- Preparation:
  - Vina/Smina use MGLTools to prepare ``.pdbqt`` (fallback to OpenBabel when enabled).
  - PLANTS uses SPORES to prepare ``.mol2``.
- Rescoring: Vina/Smina/PLANTS scoring functions and optional ODDT models.

Inputs
------

Receptor
~~~~~~~~

- Preferred: ``.pdb``.
- Optional: pre-prepared receptor files (``.pdbqt`` for Vina/Smina, ``.mol2`` for PLANTS).

Ligand
~~~~~~

- ``.smi``, ``.sdf``, ``.mol2``, or ``.pdbqt`` are accepted by the CLI.
- Ligands are prepared automatically when needed.

Box file format
~~~~~~~~~~~~~~~

The docking box is a PDB file with ``REMARK`` lines for center and dimensions. A minimal example:

.. code-block:: text

   HEADER    CORNERS OF BOX
   REMARK    CENTER (X Y Z)      12.345  23.456  34.567
   REMARK    DIMENSIONS (X Y Z)  20.000  20.000  20.000

Boxes can be created programmatically via ``Ligand.create_box()`` or in preprocessing pipelines.

Outputs
-------

Single engine (``ocdocker vs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CLI creates engine-specific folders next to the ligand directory (``vinaFiles``, ``sminaFiles``,
or ``plantsFiles``) and writes:

- ``conf_*.txt``: generated docking config
- ``*.log``: docking logs
- ``*.pdbqt`` or PLANTS output directories
- ``prepare_receptor.log`` and ``prepare_ligand.log`` for preparation steps
- optional rescoring logs and pose splits

Pipeline (``ocdocker pipeline``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outputs are written under ``--outdir`` and typically include:

- ``vinaFiles/``, ``sminaFiles/``, ``plantsFiles/`` (engine runs)
- ``prepared_receptor.*`` and ``prepared_ligand.*``
- ``poses_mol2/`` (poses converted to MOL2)
- ``rmsd_matrix.csv`` and ``clustering_dendrogram.png``
- ``cluster_assignments.csv`` and ``clustering_info.json`` (when clustering succeeds)
- ``representative.mol2`` (selected pose)
- ``summary.json`` (rescoring summary)
- ``oddt_rescoring/`` (if ODDT rescoring is enabled)

Some files appear only when the corresponding step is enabled.

Configuration
-------------

OCDocker reads configuration from ``OCDocker.cfg``. Use:

.. code-block:: bash

   ocdocker init-config --conf OCDocker.cfg

Key sections in ``OCDocker.cfg`` (see ``OCDocker.cfg.example``):

- Database: ``HOST``, ``USER``, ``PASSWORD``, ``DATABASE``, ``OPTIMIZEDB``, ``PORT``
- SQLite: ``USE_SQLITE`` and ``SQLITE_PATH``
- External tools: ``vina``, ``smina``, ``plants``, ``spores``, ``pythonsh``,
  ``prepare_ligand``, ``prepare_receptor``, ``obabel``, ``oddt``
- Engine defaults: ``vina_*``, ``smina_*``, ``plants_*``
- Data directories: ``ocdb`` (datasets), ``pca`` (PCA models)

Environment variables
---------------------

Common:

- ``OCDOCKER_CONFIG``: path to ``OCDocker.cfg``
- ``OCDOCKER_USE_SQLITE``: use SQLite backend instead of MySQL
- ``OCDOCKER_SQLITE_PATH``: explicit SQLite file path
- ``OCDOCKER_NO_AUTO_BOOTSTRAP``: disable import-time bootstrap
- ``OCDOCKER_TIMEOUT``: default timeout (seconds) for external tools

Advanced (debugging subprocesses):

- ``OCDOCKER_SUBPROCESS_TAIL_LINES``: number of log tail lines to include on errors
- ``OCDOCKER_DEBUG_SUBPROCESS``: include stdout tail and env snapshot in failure reports
- ``OCDOCKER_RAISE_SUBPROCESS``: raise exceptions instead of returning error codes
- ``OCDOCKER_SKIP_ODDT``: skip importing ODDT during bootstrap
- ``OCDOCKER_ALLOW_SCRIPT_EXEC``: allow trusted in-process script execution
- ``OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION``: allow trusted pickle/joblib/torch deserialization

CLI
---

Main commands (see ``ocdocker <command> --help``):

- ``vs``: single-engine docking with optional rescoring of all poses
- ``pipeline``: multi-engine docking, RMSD clustering, representative selection, rescoring
- ``shap``: OCScore SHAP analysis
- ``console``: interactive console with OCDocker pre-loaded
- ``script``: run a Python script with OCDocker pre-loaded
  (requires ``--allow-unsafe-exec`` or ``OCDOCKER_ALLOW_SCRIPT_EXEC=1``)
- ``doctor``: diagnostics (binaries, deps, DB)
- ``manifest``: generate reproducibility manifest (versions/runtime/tooling)
- ``init-config``: create a starter config file
- ``version``: print installed version

Global options:

- ``--conf``, ``--multiprocess``, ``--no-multiprocess``, ``--update-databases``
- ``--output-level``, ``--overwrite``, ``--log-file``, ``--no-stdout-log``

Programmatic manifest API:

.. code-block:: python

   import OCDocker.Toolbox.Reproducibility as ocrepro
   manifest = ocrepro.generate_reproducibility_manifest(include_python_packages=False)
   _ = ocrepro.write_reproducibility_manifest("reproducibility_manifest.json")

Trusted runtime helper
~~~~~~~~~~~~~~~~~~~~~~

For trusted scripts that need deserialization gates enabled at runtime:

.. code-block:: python

   from OCDocker.Toolbox.Security import allow_unsafe_runtime
   allow_unsafe_runtime(deserialization=True, script_exec=False)

Examples
~~~~~~~~

.. code-block:: bash

   ocdocker vs \
     --engine vina \
     --receptor path/to/receptor.pdb \
     --ligand path/to/ligand.sdf \
     --box path/to/box0.pdb \
     --timeout 600 \
     --store-db

.. code-block:: bash

   ocdocker pipeline \
     --receptor path/to/receptor.pdb \
     --ligand path/to/ligand.sdf \
     --box path/to/box0.pdb \
     --engines vina,smina,plants \
     --rescoring-engines vina,smina,oddt \
     --timeout 900 \
     --store-db

Python API
----------

Minimal example using Vina:

.. code-block:: python

   import OCDocker.Receptor as ocr
   import OCDocker.Ligand as ocl
   import OCDocker.Docking.Vina as ocvina

   receptor = ocr.Receptor("receptor.pdb", name="receptor")
   ligand = ocl.Ligand("ligand.sdf", name="ligand")

   vina = ocvina.Vina(
       "conf_vina.txt",
       "box0.pdb",
       receptor,
       "prepared_receptor.pdbqt",
       ligand,
       "prepared_ligand.pdbqt",
       "vina.log",
       "vina_output.pdbqt",
       name="vina_run",
       overwrite_config=True,
   )

   vina.run_prepare_receptor()
   vina.run_prepare_ligand()
   vina.run_docking()

Data layout for screening sets
------------------------------

OCDocker supports a simple folder layout for virtual screening datasets:

.. code-block:: text

   receptor/
     compounds/
       candidates/
         molecule_1/
         molecule_2/
       decoys/
         molecule_A/
         molecule_B/
       ligands/
         molecule_a/
         molecule_b/

- ``receptor``: receptor structure (``.pdb``)
- ``candidates``: unknown binders (typical screening set)
- ``decoys``: negative controls for evaluation
- ``ligands``: known actives (training/validation)

Rescoring and OCScore
---------------------

- Engine rescoring: Vina/Smina/PLANTS scoring functions are configured in ``OCDocker.cfg``.
- ODDT rescoring: optional ML-based scoring via ``OCDocker.Rescoring.ODDT``.
- OCScore: training, optimization, and analysis pipelines for consensus scoring.
  See :doc:`OCDocker.OCScore` and the examples under :doc:`examples`.

Database and persistence
------------------------

- Default backend is MySQL; SQLite can be enabled for local development.
- Use ``--store-db`` in CLI commands to store minimal metadata in the database.
- Database schemas are defined under :doc:`OCDocker.DB` and related model pages.

Diagnostics and troubleshooting
-------------------------------

- Run ``ocdocker doctor`` to validate configuration, binaries, Python deps, and DB access.
- If docking tools fail, confirm paths in ``OCDocker.cfg`` and review preparation logs.

Further reading
---------------

- :doc:`usage`
- :doc:`examples`
- :doc:`OCDocker`
- :doc:`OCDocker.Docking`
- :doc:`OCDocker.Processing`
- :doc:`OCDocker.Rescoring`
- :doc:`OCDocker.OCScore`
