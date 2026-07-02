Changelog
=========

Unreleased
----------

MCP server for LLM-driven OCDocker orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ocdocker mcp serve`` runs a Model Context Protocol server (:mod:`OCDocker.MCP`,
new ``mcp`` optional extra) over stdio for LLM clients such as Claude Code and
Claude Desktop. It is a thin adapter over a running ``ocdocker workbench serve``
API: workspace inspection, ablation design preview/plan, protocol similarity,
and job listing/logs are always-available read tools; ``run_job`` and
``cancel_job`` require the Workbench job bearer token and an explicit
``confirm=True`` from the calling LLM, so a job is never launched from one
ambiguous instruction. See :doc:`optional_dependencies`.

Workbench API can launch and track jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ocdocker workbench serve`` now exposes ``/api/jobs*`` endpoints that launch,
list, poll, tail logs for, and cancel ``vs``, ``pipeline``, ``ocscore train``,
and ``ocscore reduce`` runs as tracked local subprocesses
(:mod:`OCDocker.Workbench.Jobs`). Job state is persisted under
``<served root>/.ocdocker-jobs/`` and survives an API restart. Job-execute
endpoints require a bearer token, auto-generated on first run at
``~/.config/ocdocker/workbench_token`` or set via ``OCDOCKER_WORKBENCH_TOKEN``
(:mod:`OCDocker.Workbench.Auth`); all existing read-only inspection endpoints
are unchanged and remain unauthenticated.

Workbench API migrated to FastAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ocdocker workbench serve`` now runs on FastAPI/uvicorn instead of the stdlib
``http.server``, behind the new ``api`` optional extra
(``pip install "ocdocker[api]"``). All existing endpoints, response shapes, and
the ``/app`` browser dashboard are unchanged; the server now also exposes an
OpenAPI schema at ``/api/openapi.json``. See :doc:`optional_dependencies` and
:doc:`usage`.

Console and CLI separation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The interactive console lives in :mod:`OCDocker.Console`. Use ``ocdocker console``
or ``python -m OCDocker.Console`` to start the REPL; built-in ``help`` and ``exit``
commands are supported. The root ``OCDockerConsole.py`` wrapper was removed.
See :doc:`OCDocker.Console` and :doc:`usage`.

Packaging and optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OCDocker now ships a **minimal core** install; scientific, ML, and plotting stacks
are optional pip extras (``docking``, ``db``, ``ml``, ``analysis``, ``workflow``,
``all``, ``full``, ``dev``). See :doc:`optional_dependencies` for the cheat sheet
and full reference.

Formatting and readability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Priority config and Python modules use expanded multiline formatting (line length
120, one dependency per line in ``pyproject.toml``). Long config defaults such as
``reference_column_order`` live in module-level constants in ``OCDocker/Config.py``.
See :doc:`development` for contributor formatting rules.

OCScore feature-reduction API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added a granular feature-reduction API in
``OCDocker.OCScore.Utils.FeatureReduction`` for descriptor datasets. The new API
keeps feature-reduction behavior in reusable Python functions and dataclasses,
with ``run_feature_reduction_protocol`` provided only as an orchestration helper.

Highlights:

* descriptor block detection for receptor, ligand, and scoring-function columns
* Ligand/Receptor descriptor metadata support with configurable pattern fallback
* missing-row removal with row-level and block-level reports before data loss
* block-wise constant, near-constant, duplicate, and correlation filtering
* cross-block correlation and Ridge CV predictability diagnostics
* opt-in parallel Ridge CV diagnostics through ``CrossBlockDiagnosticsConfig.n_jobs``
* opt-in orchestration progress logging through ``FeatureReductionConfig.verbose``
* disabled-by-default conservative cross-block filtering
* reproducibility protocol and stable report filenames

Compatibility and rewiring:

* Existing OCScore training, DNN, autoencoder, SHAP, and downstream evaluation
  paths are not automatically rewired to call this API.
* ``OCDocker.OCScore.Utils.IO.load_data`` is not changed; the new orchestration
  helper reads raw CSV input directly so missing rows can be reported before
  removal.
* This is additive public API. If released publicly, it fits a minor version bump
  rather than a major version bump.
