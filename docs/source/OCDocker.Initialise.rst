OCDocker.Initialise module
==========================

Overview
--------

``OCDocker.Initialise`` bootstraps the runtime environment when called explicitly: it reads the configuration
file, prepares directories, sets up database connections, and ensures ODDT models are
available. Importing this module does not bootstrap, connect to databases, create files, or print the banner.

Environment variables
---------------------

- ``OCDOCKER_CONFIG``: path to ``OCDocker.cfg`` or ``OCDocker.yml`` (if omitted, a local config file is auto-detected)
- ``OCDOCKER_DB_BACKEND`` / ``DB_BACKEND``: select backend (``postgresql``, ``mysql``, ``sqlite``)
- ``OCDOCKER_SQLITE_PATH``: explicit SQLite database file path (used when backend is ``sqlite``)
- ``OCDOCKER_TIMEOUT``: default timeout (seconds) for external tools

Explicit bootstrap
------------------

Call ``bootstrap`` from CLI/application code before using initialized runtime globals:

.. code-block:: python

   from OCDocker.Initialise import bootstrap
   import argparse
   bootstrap(
       argparse.Namespace(
           multiprocess=True,
           update=False,
           config_file='OCDocker.cfg',
           output_level=2,
           overwrite=False,
       ),
       init_db=True,
       create_db_if_missing=False,
   )

Set ``create_db_if_missing=True`` only when the caller intentionally wants to create missing PostgreSQL/MySQL databases.

API
---

.. automodule:: OCDocker.Initialise
   :members:
   :undoc-members:
   :show-inheritance:
