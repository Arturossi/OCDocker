Development conventions
=======================

This page summarizes repository conventions for contributors. See also
``CONTRIBUTING.md`` in the repository root.

Formatting and readability
--------------------------

OCDocker favors **reviewable diffs** over compressed single-line blobs.

Python
~~~~~~

- Target **Black-compatible** formatting at **line length 120** (matches
  ``[tool.ruff] line-length`` in ``pyproject.toml``).
- Format touched modules only, for example:

  .. code-block:: bash

     python -m black --line-length 120 OCDocker/Config.py

- Long default lists (for example ``reference_column_order`` in
  ``OCDocker/Config.py``) belong in **module-level constants** with one element
  per line, following the ``GNINA_DEFAULT_*`` pattern—not inline mega-literals.

Packaging metadata
~~~~~~~~~~~~~~~~~~

- ``pyproject.toml``: multiline TOML arrays; **one dependency per line** in
  ``[project.dependencies]`` and ``[project.optional-dependencies]``.
- ``requirements.txt``: mirrors core ``[project.dependencies]`` only; one
  requirement per line. Optional stacks use pip extras (see
  :doc:`optional_dependencies`).
- ``[tool.setuptools.package-data]`` in ``pyproject.toml`` ships bundled OCScore
  protocol YAML (``OCDocker/OCScore/Protocols/``) and Workbench static assets
  (``OCDocker/Workbench/static/``). See :doc:`OCDocker.OCScore.Protocols`.

Config and tooling files
~~~~~~~~~~~~~~~~~~~~~~~~

- ``pytest.ini``: standard INI; one option per line where practical.
- ``.pre-commit-config.yaml``: block YAML with normal indentation.

Verification
~~~~~~~~~~~~

After formatting-only edits, run:

.. code-block:: bash

   python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
   python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"
   python -m compileall OCDocker/Config.py
   pytest tests/cli/test_packaging_metadata.py tests/core/test_config.py -q

Formatting-only PRs must not change dependency versions, config semantics,
docking logic, or OCScore protocol behavior.

Interactive console
-------------------

Console logic lives in :mod:`OCDocker.Console`. Keep imports side-effect-free: no
banner, bootstrap, or REPL at import time. API reference: :doc:`OCDocker.Console`.
User entrypoints: :doc:`usage`.
