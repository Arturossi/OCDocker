OCDocker.Initialise module
==========================

Overview
--------

``OCDocker.Initialise`` bootstraps the runtime environment: it reads the configuration
file, prepares directories, sets up database connections, and ensures ODDT models are
available. By default, OCDocker auto‑bootstraps on first import outside of docs/tests.

Environment variables
---------------------

- ``OCDOCKER_CONFIG``: path to ``OCDocker.cfg`` or ``OCDocker.yml`` (if omitted, a local config file is auto-detected)
- ``OCDOCKER_NO_AUTO_BOOTSTRAP``: set to disable auto‑bootstrap on import
- ``OCDOCKER_DB_BACKEND`` / ``DB_BACKEND``: select backend (``postgresql``, ``mysql``, ``sqlite``)
- ``OCDOCKER_SQLITE_PATH``: explicit SQLite database file path (used when backend is ``sqlite``)
- ``OCDOCKER_TIMEOUT``: default timeout (seconds) for external tools

Explicit bootstrap
------------------

If you disable auto‑bootstrap or need custom arguments, call ``bootstrap`` manually:

.. code-block:: python

   from OCDocker.Initialise import bootstrap
   import argparse
   bootstrap(argparse.Namespace(
       multiprocess=True,
       update=False,
       config_file='OCDocker.cfg',
       output_level=2,
       overwrite=False,
   ))

API
---

.. automodule:: OCDocker.Initialise
   :members:
   :undoc-members:
   :show-inheritance:
