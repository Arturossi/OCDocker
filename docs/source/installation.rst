Installation
============

This section provides instructions on how to install OCDocker.

Prerequisites
-------------

Make sure you have the following prerequisites installed:

- Python (>=3.9)
- pip (or conda/mamba)

Installing OCDocker
--------------------

To install OCDocker, follow these steps:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-repository/OCDocker.git
      cd OCDocker

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

3. Install OCDocker and dependencies (pip):

   .. code-block:: bash

      pip install .

   Or install from PyPI:

   .. code-block:: bash

      pip install ocdocker

   Or use conda (recommended for binary deps like RDKit/OpenBabel):

   .. code-block:: bash

      mamba install arturossi/label/prealpha::ocdocker
