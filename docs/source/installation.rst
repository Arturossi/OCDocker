Installation
============

This section provides instructions on how to install OCDocker.

Prerequisites
-------------

Make sure you have the following prerequisites installed:

- Python (>=3.11)
- Conda (Miniconda/Anaconda) with mamba
- pip (inside the conda environment)

Quickstart (minimal, SQLite)
----------------------------

If you want the fastest path without setting up MySQL, use SQLite (local file DB) as the default backend:

1. Ensure the system dependencies are installed (see :ref:`system-dependencies`).
2. Create/activate a conda env with Python 3.11 (prefer mamba) and install OCDocker with pip.
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

   sudo apt-get install openbabel libopenbabel-dev swig cmake g++

These packages are required for building and using OpenBabel Python bindings, which are essential for OCDocker's molecular processing capabilities.

Installing OCDocker
-------------------

To install OCDocker, follow these steps:

1. Ensure the system dependencies are installed (see :ref:`system-dependencies`).

2. Install mamba (if not already installed):

   .. code-block:: bash

      conda install -n base -c conda-forge mamba

3. Create and activate a conda environment:

   .. code-block:: bash

      mamba create -n ocdocker python=3.11 -y
      conda activate ocdocker

4. Install OCDocker with pip (choose one option):

   Option A: install from PyPI (recommended for users):

   .. code-block:: bash

      pip install ocdocker

   Option B: install from source (recommended for development):

   .. code-block:: bash

      git clone https://github.com/Arturossi/OCDocker.git
      cd OCDocker
      pip install .

Optional: build the Sphinx documentation
-----------------------------------------

If you want to build docs locally in the same ``ocdocker`` conda environment:

Option A (conda/mamba):

.. code-block:: bash

   mamba install -n ocdocker -c conda-forge sphinx sphinx-argparse furo sphinx-rtd-theme myst-parser
   make -C docs html

Option B (pip extras):

.. code-block:: bash

   pip install -e ".[docs]"
   make -C docs html
