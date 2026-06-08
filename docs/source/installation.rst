Installation
============

This section provides instructions on how to install OCDocker.

Prerequisites
-------------

Make sure you have the following prerequisites installed:

- Python (>=3.11)
- Conda (Miniconda/Anaconda) with mamba
- pip (inside the conda environment)
- NVIDIA driver/runtime compatible with CUDA 12.8 (required to run Gnina CUDA builds)

Quickstart (minimal, SQLite)
----------------------------

If you want the fastest path without setting up PostgreSQL/MySQL, use SQLite (local file DB) as the backend:

1. Ensure the system dependencies are installed (see :ref:`system-dependencies`).
2. Create/activate a conda env with Python 3.11 (prefer mamba) and install OCDocker with pip.
3. Enable SQLite when running commands:

   .. code-block:: bash

      export OCDOCKER_DB_BACKEND=sqlite
      ocdocker doctor

SQLite is recommended for quick experiments and development. PostgreSQL is the default backend, and MySQL is optional for alternative server-based workflows.

.. _system-dependencies:

System dependencies
-------------------

Before installing OCDocker, you must install the following system packages on Ubuntu/Debian systems:

.. code-block:: bash

   sudo apt-get install openbabel libopenbabel-dev swig cmake g++

These packages are required for building and using OpenBabel Python bindings, which are essential for OCDocker's molecular processing capabilities.

Gnina (CUDA 12.8)
-----------------

OCDocker expects the Gnina CUDA 12.8 build. To run this binary reliably, ensure:

- NVIDIA driver is compatible with CUDA 12.8
- cuDNN 9 runtime is available on the system

Step-by-step installation:

.. code-block:: bash

   mkdir -p gnina
   wget -O gnina/gnina.1.3.2.cuda12.8 \
     https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8
   chmod +x gnina/gnina.1.3.2.cuda12.8
   sudo install -m 0755 gnina/gnina.1.3.2.cuda12.8 /usr/bin/gnina

Verify:

.. code-block:: bash

   gnina --version

Note: ``install.sh`` already installs this Gnina CUDA 12.8 binary automatically.

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

   ``pip install ocdocker`` installs the minimal core (CLI bootstrap, config I/O,
   logging, and serialization helpers). Feature-specific dependencies are optional
   extras — see :doc:`optional_dependencies` for the full cheat sheet, per-extra
   package lists, and command mapping.

   Quick install commands:

   .. code-block:: bash

      pip install ocdocker                      # minimal core
      pip install "ocdocker[docking]"           # vs / pipeline
      pip install "ocdocker[db]"                # --store-db
      pip install "ocdocker[ml]"                # OCScore ML
      pip install "ocdocker[analysis]"          # plots / statistics
      pip install "ocdocker[all]"               # all runtime stacks
      pip install -e ".[all,dev]"               # typical developer setup

   Typical combinations (details in :doc:`optional_dependencies`):

   - Single-engine docking: ``ocdocker[docking]``
   - Multi-engine pipeline with dendrogram output: ``ocdocker[docking,analysis]``
   - OCScore replication: ``ocdocker[ml]`` (often ``ocdocker[all]`` for full parity)

   Option B: install from source (recommended for development):

   .. code-block:: bash

      git clone https://github.com/Arturossi/OCDocker.git
      cd OCDocker
      # Minimal core
      pip install -e .

      # Typical developer install
      pip install -e ".[all,dev]"

Optional: build the Sphinx documentation
-----------------------------------------

If you want to build docs locally in the same ``ocdocker`` conda environment:

Option A (conda/mamba):

.. code-block:: bash

   mamba install -n ocdocker -c conda-forge sphinx sphinx-argparse furo sphinx-rtd-theme myst-parser
   make -C docs html

Option B (pip extras):

.. code-block:: bash

   pip install -e ".[full]"
   make -C docs html
