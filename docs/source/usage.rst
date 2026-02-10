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

- shap: Delegate to OCScore SHAP CLI

  .. code-block:: bash

     ocdocker shap --help

- console: Interactive console with tab-completion and history

  .. code-block:: bash

     ocdocker console --conf OCDocker.cfg

- doctor: Diagnostics for binaries/deps/DB

  .. code-block:: bash

     ocdocker doctor --conf OCDocker.cfg

- init-config: Create a starter ``OCDocker.cfg`` from the example

  .. code-block:: bash

     ocdocker init-config --conf OCDocker.cfg

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

- ``--conf``: path to ``OCDocker.cfg``
- ``--multiprocess``: enable multiprocessing for compatible tasks
- ``--no-multiprocess``: disable multiprocessing for compatible tasks
- ``--update-databases``: run DB updates at startup
- ``--output-level``: control log level (0-5)
- ``--overwrite``: allow overwriting outputs when applicable
- ``--log-file``: write logs to a file
- ``--no-stdout-log``: disable logging to stdout

Bootstrap & environment
-----------------------

- Auto-bootstrap happens on first import outside docs/tests.
- Environment variables:

  - ``OCDOCKER_CONFIG``: config file path
  - ``OCDOCKER_NO_AUTO_BOOTSTRAP``: disable auto-bootstrap on import
  - ``OCDOCKER_USE_SQLITE``: opt-in SQLite backend (local file), instead of MySQL
  - ``OCDOCKER_SQLITE_PATH``: explicit SQLite database file path
  - ``OCDOCKER_TIMEOUT``: default timeout (seconds) for external tools
  - ``OCDOCKER_SKIP_ODDT``: skip importing ODDT during bootstrap
  - ``OCDOCKER_ALLOW_SCRIPT_EXEC``: allow trusted in-process script execution
  - ``OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION``: allow trusted pickle/joblib/torch deserialization

- For trusted scripts that need deserialization, use:

  .. code-block:: python

     from OCDocker.Toolbox.Security import allow_unsafe_runtime
     allow_unsafe_runtime(deserialization=True, script_exec=False)

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
