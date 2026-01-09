Installation
============

This section provides instructions on how to install OCDocker.

Prerequisites
-------------

Make sure you have the following prerequisites installed:

- Python (>=3.10)
- pip (or conda/mamba)

Quickstart (minimal, SQLite)
----------------------------

If you want the fastest path without setting up MySQL, use SQLite (local file DB) as the default backend:

1. Ensure the system dependencies are installed (see :ref:`system-dependencies`).
2. Install OCDocker (see below).
3. Enable SQLite when running commands:

   .. code-block:: bash

      export OCDOCKER_USE_SQLITE=1
      ocdocker doctor

SQLite is recommended for quick experiments and development. MySQL is optional and only needed for multi-user or long-running database workflows.

.. _system-dependencies:

System dependencies
-------------------

Before installing OCDocker, you must install the following system packages on Ubuntu/Debian systems:

.. code-block:: bash

   sudo apt-get install openbabel libopenbabel-dev swig

These packages are required for building and using OpenBabel Python bindings, which are essential for OCDocker's molecular processing capabilities.

Installing OCDocker
-------------------

To install OCDocker, follow these steps:

1. Ensure the system dependencies are installed (see :ref:`system-dependencies`).

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-repository/OCDocker.git
      cd OCDocker

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

4. Install OCDocker and dependencies (pip):

   .. code-block:: bash

      pip install .

   Or install from PyPI:

   .. code-block:: bash

      pip install ocdocker

   Or use conda (recommended for binary deps like RDKit/OpenBabel):

   .. code-block:: bash

      mamba install arturossi/label/prealpha::ocdocker

   **Note:** Even when using conda, ensure the system packages are installed first for optimal compatibility (see :ref:`system-dependencies`).
