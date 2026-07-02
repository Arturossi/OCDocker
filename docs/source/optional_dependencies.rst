Optional dependencies
=====================

Brief overview
--------------

OCDocker splits Python dependencies into a **minimal core** and **optional extras**.
Install only what your workflow needs; add extras when a command or module requires
heavier libraries.

**Cheat sheet**

.. code-block:: bash

   pip install ocdocker                      # minimal core
   pip install "ocdocker[docking]"           # vs / pipeline
   pip install "ocdocker[db]"                # --store-db
   pip install "ocdocker[ml]"                # OCScore ML
   pip install "ocdocker[analysis]"          # plots / statistics
   pip install "ocdocker[workflow]"          # Snakemake
   pip install "ocdocker[all]"               # all runtime stacks
   pip install "ocdocker[full]"              # all runtime + docs build
   pip install "ocdocker[dev]"               # pytest, mypy, ruff
   pip install -e ".[all,dev]"               # typical developer setup

**Typical combinations**

- Single-engine docking → ``ocdocker[docking]``
- Multi-engine pipeline with dendrogram PNG → ``ocdocker[docking,analysis]``
- OCScore replication → ``ocdocker[ml]`` (often ``ocdocker[all]`` for full parity)
- Editable development → ``pip install -e ".[all,dev]"``

If a command fails with ``ModuleNotFoundError``, the CLI suggests the matching
extra (for example ``pip install "ocdocker[ml]"`` for missing ``torch``).

See :doc:`installation` for conda/system prerequisites and :doc:`usage` for CLI
workflows that depend on each extra.

.. _optional-deps-design:

Design and rationale
--------------------

Why a lightweight core?
~~~~~~~~~~~~~~~~~~~~~~~

``pip install ocdocker`` should work quickly in CI, headless servers, and
environments where you only need configuration, manifests, or doctor checks.
Scientific stacks (RDKit, PyTorch, matplotlib) are large, platform-sensitive, and
not needed to import the base package or run ``ocdocker --help``.

The core therefore includes only utilities shared across many code paths:

.. list-table:: Core dependencies (always installed)
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Role in OCDocker
   * - ``configargparse``
     - CLI/config parsing helpers
   * - ``joblib``
     - Serialization for masks and sklearn-style artifacts in OCScore I/O
   * - ``packaging``
     - Version and distribution metadata
   * - ``pydantic``, ``pydantic-settings``
     - Structured configuration (reserved for typed settings)
   * - ``pyyaml``
     - YAML configuration files (``OCDocker.yml``)
   * - ``requests``
     - HTTP downloads and remote resources
   * - ``rich``
     - Optional enhanced console logging (falls back to stdlib logging)
   * - ``tqdm``
     - Progress bars in processing and download utilities

Everything else is grouped by **workflow** into pip extras defined in
``pyproject.toml``. Extras can overlap (``numpy`` appears in ``docking``, ``ml``,
and ``analysis``); pip deduplicates installed packages.

Lazy imports and CLI hints
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional modules are imported only when you run the matching command or import
the matching submodule. If a dependency is missing, the CLI prints a short hint
such as::

   Error: missing optional dependency 'torch' required for OCScore pipeline.
   Install with: pip install "ocdocker[ml]"

Plotting in RMSD clustering uses lazy ``matplotlib`` imports; clustering itself
runs with ``[docking]`` only, while dendrogram PNG output needs ``[analysis]``.

.. _optional-deps-extras:

Extra reference
---------------

Runtime extras
~~~~~~~~~~~~~~

``docking``
   Chemistry and the numeric stack for virtual screening and pipeline clustering.

   * **Chemistry:** ``rdkit``, ``openbabel-wheel``, ``biopython``, ``spyrmsd``
   * **Numeric / clustering:** ``numpy``, ``pandas``, ``scipy``, ``scikit-learn``

   Required for ``ocdocker vs``, ``ocdocker pipeline``, and Python imports such as
   ``OCDocker.Ligand`` and ``OCDocker.Receptor``.

``db``
   SQLAlchemy and database drivers for PostgreSQL and MySQL.

   * ``sqlalchemy``, ``sqlalchemy-utils``, ``psycopg``, ``pymysql``

   Required when using ``--store-db`` or ``OCDocker.DB`` against server backends.
   SQLite is recommended for development/testing/small local runs; PostgreSQL/MySQL are recommended for persistent,
   concurrent, or long-running workflows. Missing DB drivers report the install hint ``pip install "ocdocker[db]"``.

``ml``
   OCScore machine-learning pipeline (PyTorch, XGBoost, Optuna).

   * **Frameworks:** ``torch``, ``torchaudio``, ``torchvision``, ``xgboost``
   * **Tuning / viz:** ``optuna``, ``optuna-dashboard``, ``optuna-integration``,
     ``torchsummary``, ``torchviz``, ``visualtorch``
   * **Shared numeric stack:** ``numpy``, ``pandas``, ``scipy``, ``scikit-learn``

   Required for ``ocdocker ocscore`` and ``OCDocker.OCScore`` training/export paths.

``analysis``
   Plotting, statistics, explainability, and extended scientific I/O.

   * **Plotting:** ``matplotlib``, ``seaborn``, ``graphviz``
   * **Statistics / explainability:** ``statsmodels``, ``pingouin``, ``dcor``, ``lime``, ``shap``
   * **Graphs / imaging:** ``networkx``, ``rustworkx``, ``scikit-image``,
     ``h5py``, ``imageio``, ``pillow``, ``tifffile``
   * **Notebook / symbolic:** ``ipython``, ``sympy``, ``mpmath``, ``gmpy2``
   * **Acceleration / utilities:** ``numba``, ``llvmlite``, ``fsspec``
   * **Numeric (standalone analysis installs):** ``numpy``, ``pandas``, ``scipy``

   Required for OCScore plotting/SHAP visualization, statistical reports, and
   pipeline dendrogram PNGs (``clustering_dendrogram.png``). Clustering math
   itself lives in ``[docking]``; plots live in ``[analysis]``.

``workflow``
   Snakemake integration and logging plugins.

   * ``snakemake``, ``snakemake-logger-plugin-snkmt``

``cloud``
   Optional cloud storage backends.

   * ``boto3``, ``google-cloud-storage``, ``dropbox``

``gpu``
   GPU array acceleration (non-macOS).

   * ``cupy-cuda11x`` (platform marker excludes Darwin)

``api``
   FastAPI/uvicorn stack for the Workbench HTTP API.

   * ``fastapi``, ``uvicorn``

   Required for ``ocdocker workbench serve``. The Workbench browser dashboard
   and JSON API are unavailable without this extra.

``mcp``
   MCP server exposing the Workbench API to LLM clients (Claude Code, Claude
   Desktop, ...) over stdio.

   * ``mcp``

   Required for ``ocdocker mcp serve``. Requires a running
   ``ocdocker workbench serve`` instance to connect to.

Aggregate and tooling extras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``all``
   Union of runtime extras: ``docking`` + ``db`` + ``ml`` + ``analysis`` +
   ``workflow`` + ``cloud`` + ``gpu`` + ``api`` + ``mcp``. Use when you want
   every runtime feature without documentation build tools.

``full``
   ``all`` plus ``docs`` (Sphinx, Furo, MyST, themes). Use to build this
   documentation locally with ``make -C docs html``.

``docs``
   Sphinx documentation build only (subset of ``full``).

``dev``
   Developer tooling: ``pytest``, ``pytest-cov``, ``pytest-order``, ``coverage``,
   ``mypy``, ``ruff``, ``pre-commit``, ``detect-secrets``. Combine with runtime
   extras, e.g. ``pip install -e ".[all,dev]"``.

``build``
   PyPI release tooling: ``build``, ``twine``, ``wheel``, ``setuptools``.

.. _optional-deps-command-map:

Command and feature map
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - User action
     - Recommended install
     - Notes
   * - ``import OCDocker`` / ``ocdocker version``
     - core only
     - No scientific stack required
   * - ``ocdocker doctor`` (basic)
     - core; add extras for full dep checks
     - Doctor reports missing optional imports
   * - ``ocdocker vs``
     - ``[docking]``
     - RDKit/OpenBabel at ligand import time
   * - ``ocdocker pipeline`` (clustering only)
     - ``[docking]``
     - RMSD clustering uses sklearn/scipy
   * - ``ocdocker pipeline`` (dendrogram PNG)
     - ``[docking,analysis]``
     - Matplotlib loaded lazily when plotting
   * - ``ocdocker vs|pipeline --store-db``
     - add ``[db]``
     - SQLAlchemy + driver for your backend
   * - ``ocdocker ocscore …``
     - ``[ml]``
     - PyTorch/Optuna/XGBoost stack
   * - OCScore plots / SHAP exports
     - ``[ml,analysis]`` or ``[all]``
     - Seaborn/matplotlib in analysis modules
   * - Snakemake workflows
     - ``[workflow]``
     - External ``snakemake`` CLI still required on PATH
   * - ``ocdocker workbench serve``
     - ``[api]``
     - FastAPI/uvicorn; dashboard is unavailable without it
   * - ``ocdocker mcp serve``
     - ``[mcp]``
     - Connects to a running ``ocdocker workbench serve``; needs ``[api]`` there too
   * - Local Sphinx docs
     - ``[full]`` or ``[docs]`` + runtime as needed
     - See :doc:`installation`
   * - Running the test suite
     - ``[all,dev]`` (typical CI layout)
     - Tests import docking/db/ml modules

Combining extras
~~~~~~~~~~~~~~~~

List multiple extras in one install command::

   pip install "ocdocker[docking,db,ml]"

Editable installs from a git checkout use the same syntax::

   pip install -e ".[docking,analysis]"

.. _optional-deps-migration:

Upgrading from older installs
-----------------------------

Earlier releases installed many scientific packages as **core** dependencies.
Current releases move them into extras:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Previously in core
     - Now install via
   * - ``numpy``, ``pandas``, ``scipy``, ``scikit-learn``
     - ``[docking]`` and/or ``[ml]``
   * - ``matplotlib``, ``seaborn``, ``statsmodels``, ``shap``, …
     - ``[analysis]``
   * - ``torch``, ``optuna``, ``xgboost``, …
     - ``[ml]`` (unchanged extra name)
   * - ``rdkit``, ``openbabel-wheel``, …
     - ``[docking]`` (unchanged)
   * - ``sqlalchemy``, …
     - ``[db]`` (unchanged)

If scripts break after upgrading, install the extras your workflow used implicitly
before, or switch to ``[all]`` for parity with a full historical environment.

``requirements.txt`` in the repository root mirrors **core only**; optional stacks
are selected via pip extras, not a monolithic requirements file. Packaged YAML
protocols and Workbench static assets are declared under
``[tool.setuptools.package-data]`` in ``pyproject.toml`` (see
:doc:`OCDocker.OCScore.Protocols`).

Troubleshooting
---------------

**``ModuleNotFoundError`` after upgrade**
   Install the extra suggested in the CLI message. For Python API use, import the
   submodule only after installing the matching extra.

**Pipeline runs but no dendrogram PNG**
   Install ``[analysis]``. Without matplotlib, clustering completes but plot
   generation is skipped with a warning.

**``ocdocker --help`` works but docking fails**
   Expected: help does not require RDKit. Run ``pip install "ocdocker[docking]"``.

**CI / reproducible environments**
   Pin extras explicitly in your environment file, e.g.
   ``pip install "ocdocker[all,dev]"``, rather than relying on transitive pins
   from an old monolithic core.

**Source of truth**
   Extras and version constraints are defined in ``pyproject.toml`` under
   ``[project.dependencies]`` and ``[project.optional-dependencies]``.
   Packaging tests in ``tests/cli/test_packaging_metadata.py`` keep
   ``requirements.txt`` aligned with the core list.

Contributor formatting rules (line length 120, multiline TOML, long Python
list constants) are in :doc:`development`.
