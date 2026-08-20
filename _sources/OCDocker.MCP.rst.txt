OCDocker.MCP package
=====================

MCP (`Model Context Protocol <https://modelcontextprotocol.io/>`_) server
exposing the OCDocker Workbench API to LLM clients (Claude Code, Claude
Desktop, or any other MCP client) over stdio. It is a thin adapter: every tool
call is translated into an HTTP request against an already-running
``ocdocker workbench serve`` API and the JSON response is returned as-is — no
business logic is duplicated here. See :mod:`OCDocker.Workbench.Server` for
the underlying API and :mod:`OCDocker.Workbench.Jobs` for job execution.

.. note::
   Nearly all of the actual tool implementations are Python closures defined
   *inside* :func:`OCDocker.MCP.Server.build_ocdocker_mcp_server` (required by
   how the ``mcp`` SDK's ``@server.tool()`` decorator registers tools against
   one running server instance). Sphinx's ``automodule`` only documents
   module-level names, so it cannot see these tools or their docstrings — the
   full reference below is maintained by hand and is the authoritative list.

Requirements and setup
-----------------------

Requires the ``mcp`` optional extra (``mcp`` package, pulls in ``httpx``) and
a separately running Workbench API (the ``api`` extra). This server does
**not** start ``ocdocker workbench serve`` itself — it only talks to one that
is already running.

.. code-block:: bash

   pip install "ocdocker[api,mcp]"

   # Terminal 1: start the Workbench API for one served OCScore/workspace root.
   ocdocker workbench serve /path/to/OCScore/output --port 8765

   # Terminal 2 (or launched directly by an MCP client, see below):
   ocdocker mcp serve --workbench-api-url http://127.0.0.1:8765

``--workbench-api-url`` defaults to ``http://127.0.0.1:8765``
(:data:`OCDocker.MCP.Server.DEFAULT_WORKBENCH_API_URL`, derived from
:data:`OCDocker.Workbench.Server.DEFAULT_WORKBENCH_API_HOST` /
:data:`OCDocker.Workbench.Server.DEFAULT_WORKBENCH_API_PORT`) and rarely needs
to be set explicitly unless the Workbench API is bound to a non-default host
or port.

Transport
---------

**stdio only.** The MCP client is expected to spawn ``ocdocker mcp serve`` as
a subprocess and communicate over its stdin/stdout, which is how every
example below is wired. There is no HTTP/SSE transport in this version —
adding one later is a small, isolated change since the tool implementations
never touch the transport layer directly (they only call the Workbench HTTP
API via ``httpx``).

Connecting a client
--------------------

Any MCP-compatible client that supports stdio servers works. For Claude
Desktop or Claude Code, add an entry like this to the client's MCP server
configuration (e.g. ``claude_desktop_config.json``):

.. code-block:: json

   {
     "mcpServers": {
       "ocdocker-workbench": {
         "command": "ocdocker",
         "args": ["mcp", "serve", "--workbench-api-url", "http://127.0.0.1:8765"]
       }
     }
   }

The server identifies itself to the client as
``ocdocker-workbench`` (:data:`OCDocker.MCP.Server.MCP_SERVER_NAME`) and
advertises this instruction string
(:data:`OCDocker.MCP.Server.MCP_SERVER_INSTRUCTIONS`) so the model
understands the confirmation-gating contract before it calls anything:

    Tools for inspecting an OCDocker Workbench workspace (OCScore studies,
    ablations, jobs) and for designing, launching, and monitoring
    vs/pipeline/ocscore_train/ocscore_reduce jobs. Read/plan/preview tools
    are always safe to call. run_job and cancel_job execute real, possibly
    long-running work: call plan_job first, show the plan to the user, and
    only call run_job with confirm=True after they agree. Never set
    confirm=True without an explicit go-ahead from the user for that
    specific job.

Tool reference
---------------

Nineteen tools are registered, in two groups. All tools return the parsed JSON
body of the underlying Workbench API response (a plain ``dict``); tool
functions never raise a raw ``httpx`` exception — every failure surfaces as
:class:`OCDocker.MCP.Server.OCDockerMCPError` (see `Error handling`_ below).

.. list-table:: Read and plan tools (no confirmation, no bearer token)
   :header-rows: 1
   :widths: 22 46 32

   * - Tool
     - Parameters
     - Workbench API endpoint
   * - ``get_health``
     - *(none)*
     - ``GET /health``
   * - ``get_workspace``
     - *(none)*
     - ``GET /api/ocscore-workspace``
   * - ``get_ablation_design_context``
     - *(none)*
     - ``GET /api/ablation-design``
   * - ``preview_ablation_design``
     - ``policy: dict``
     - ``POST /api/ablation-design/preview``
   * - ``plan_ablation_design``
     - ``policy: dict``
     - ``POST /api/ablation-design/plan``
   * - ``get_vs_design_context``
     - ``input_dir: str | None``
     - ``GET /api/vs-design``
   * - ``preview_vs_design``
     - ``draft: dict``
     - ``POST /api/vs-design/preview``
   * - ``plan_vs_design``
     - ``draft: dict``
     - ``POST /api/vs-design/plan``
   * - ``get_vs_campaign_context``
     - ``input_dir: str | None``
     - ``GET /api/vs-campaign``
   * - ``preview_vs_campaign``
     - ``manifest: list[dict]``
     - ``POST /api/vs-campaign/preview``
   * - ``plan_vs_campaign``
     - ``manifest: list[dict]``, ``outdir: str | None``
     - ``POST /api/vs-campaign/plan``
   * - ``get_protocol_similarity``
     - ``metric: str | None``, ``reference: str | None``
     - ``GET /api/ablation-protocol-similarity``
   * - ``list_jobs``
     - *(none)*
     - ``GET /api/jobs``
   * - ``get_job``
     - ``job_id: str``
     - ``GET /api/jobs/{job_id}``
   * - ``get_job_logs``
     - ``job_id: str``, ``lines: int = 80``
     - ``GET /api/jobs/{job_id}/logs``
   * - ``get_campaign_progress``
     - ``job_id: str``
     - ``GET /api/jobs/{job_id}/campaign-progress``
   * - ``plan_job``
     - ``kind: WorkbenchJobKind``, ``args: list[str] | None``, ``cwd: str | None``, ``manifest: list[dict] | None``, ``engine: str = "shell"``, ``cores: int = 4``, ``results_dir: str | None``
     - ``POST /api/jobs/plan``

.. list-table:: Execute tools (confirmation-gated, bearer-token-authenticated)
   :header-rows: 1
   :widths: 22 46 32

   * - Tool
     - Parameters
     - Workbench API endpoint
   * - ``run_job``
     - ``kind: WorkbenchJobKind``, ``args: list[str] | None``, ``cwd: str | None``, ``manifest: list[dict] | None``, ``engine: str = "shell"``, ``cores: int = 4``, ``results_dir: str | None``, ``confirm: bool = False``
     - ``POST /api/jobs/plan`` (confirm=False) or ``POST /api/jobs`` (confirm=True)
   * - ``cancel_job``
     - ``job_id: str``, ``confirm: bool = False``
     - ``GET /api/jobs/{job_id}`` (confirm=False) or ``POST /api/jobs/{job_id}/cancel`` (confirm=True)

``WorkbenchJobKind`` is the literal type ``"vs" | "pipeline" | "ocscore_train" |
"ocscore_reduce" | "vs_campaign"`` (:data:`OCDocker.Workbench.Models.WorkbenchJobKind`)
— it selects which ``ocdocker`` subcommand prefix a job runs (``ocdocker vs ...``,
``ocdocker pipeline ...``, ``ocdocker ocscore train ...``, ``ocdocker ocscore
reduce ...``), except ``"vs_campaign"``, which runs a generated shell script
covering every row of ``manifest`` instead (see `Batch campaigns`_ below).
``args`` are extra CLI flags appended after that prefix — or, for
``"vs_campaign"``, common flags appended to every row's command — e.g.
``["--protocol", "production.yml", "--output-dir", "runs/run-001"]``.

Read and plan tools
~~~~~~~~~~~~~~~~~~~~

``get_health()``
   Check whether the Workbench API is reachable, and its
   ``read_only``/``dashboard_model`` status. Returns the same payload as
   ``GET /health`` on the Workbench API (``ok``, ``service``, ``api_version``,
   ``root``, ``read_only``, ``dashboard_model``).

``get_workspace()``
   Return the served OCScore workspace summary: baseline study, ablation
   studies, per-replica status, and curated metrics. Same payload as
   ``GET /api/ocscore-workspace``
   (:class:`OCDocker.Workbench.Models.WorkbenchOCScoreWorkspace`).

``get_ablation_design_context()``
   Return the bundled and workspace-discovered ablation feature policies plus
   defaults for designing a new one. Same payload as ``GET /api/ablation-design``.

``preview_ablation_design(policy)``
   Preview which features a draft ablation feature-policy would keep or
   exclude, without writing anything. ``policy`` is a JSON object matching
   the Workbench Design tab's request body: a ``policy`` block
   (``include_features``, ``include_patterns``, ``exclude_features``,
   ``exclude_patterns``, ``allow_missing_excludes``, ...) plus optional
   identity/source fields (``name``, ``description``, ``feature_source``).
   See :func:`OCDocker.Workbench.AblationDesign.preview_ablation_design` for
   the exact body contract.

``plan_ablation_design(policy)``
   Generate the policy YAML text and the ``ocdocker ocscore train`` command
   string for a draft ablation. Same ``policy`` body shape as
   ``preview_ablation_design``. This only plans — it does not write the
   policy file or launch training; use ``run_job`` with
   ``kind="ocscore_train"`` for that once the design is finalized. See
   :func:`OCDocker.Workbench.AblationDesign.plan_ablation_design`.

``get_vs_design_context(input_dir=None)``
   Discover receptor/ligand/box candidates for designing a single-target
   ``vs``/``pipeline`` run. Best-effort, depth-limited scan by
   filename/extension heuristics (receptor: ``.pdb``/``.pdbqt`` with
   "receptor" in the name; ligand: ``.smi``/``.sdf``/``.mol2``/``.pdbqt``
   with "ligand" in the name or parent directory; box: ``.pdb``/``.txt``
   starting with "box") — there is no single fixed input layout in
   OCDocker, so results are candidates to choose from, never a
   requirement. ``input_dir`` optionally narrows the scan to one
   subdirectory of the served root instead of the whole workspace. Same
   payload as ``GET /api/vs-design``. See
   :func:`OCDocker.Workbench.VSDesign.discover_vs_design_candidates`.

``preview_vs_design(draft)``
   Validate one draft single-target VS design without running anything.
   ``draft`` is a JSON object with ``kind`` (``"vs"`` or ``"pipeline"``),
   ``receptor``, ``ligand``, ``box`` (paths — absolute, or relative to the
   served root), plus kind-specific fields: ``engine`` for ``"vs"``, or
   ``engines``/``rescoring_engines`` (lists) for ``"pipeline"``. Checks
   path existence, extension sanity, and engine names against
   :data:`OCDocker.Workbench.Models.VALID_DOCKING_ENGINES`/
   :data:`OCDocker.Workbench.Models.VALID_RESCORING_ENGINES`. Covers exactly
   one receptor, one ligand, one box; for many samples in one job, see
   `Batch campaigns`_ below. Same payload as
   ``POST /api/vs-design/preview``. See
   :func:`OCDocker.Workbench.VSDesign.preview_vs_design`.

``plan_vs_design(draft)``
   Build the exact ``ocdocker vs``/``pipeline`` argv for a valid draft
   design. Same ``draft`` body shape as ``preview_vs_design`` — call that
   first and show the user any errors/warnings. On success returns
   ``{"kind", "args", "cwd", "shell_command"}``, ready to hand directly to
   ``run_job`` (as ``kind``/``args``/``cwd``) to actually launch it,
   subject to the same ``confirm=True`` gate as every other execute tool.
   Raises :class:`~OCDocker.MCP.Server.OCDockerMCPError` if the draft is
   invalid. Same payload as ``POST /api/vs-design/plan``. See
   :func:`OCDocker.Workbench.VSDesign.plan_vs_design`.

Batch campaigns
~~~~~~~~~~~~~~~~

A campaign runs many receptor/ligand/box samples as **one** tracked job — no
per-row confirmations. Its manifest reuses the same row shape
:func:`discover_vs_campaign_candidates` produces: ``sample``, ``row_kind``
(``"vs"`` or ``"pipeline"`` — rows may mix both), ``receptor``, ``ligand``,
``box``, ``engines`` (list), optional ``rescoring_engines`` (list).

Two execution engines, chosen via ``engine`` on ``plan_vs_campaign``/
``run_job``/``plan_job``:

- ``engine="shell"`` (default, no extra dependency): a generated POSIX shell
  script (:func:`OCDocker.Workbench.Jobs.build_campaign_script`) runs one
  ``ocdocker vs``/``pipeline`` invocation per row sequentially, logging
  ``[sample i/N] <name>`` markers before each, **continuing past a failing
  row**, and exiting non-zero only if any row failed.
- ``engine="snakemake"`` (requires the ``mcp``/``workflow`` extra's
  Snakemake dependency): a bundled multi-sample Snakefile
  (:func:`OCDocker.Workbench.Jobs.build_campaign_snakemake_command`,
  ``OCDocker/Workbench/Snakefiles/vs_campaign.smk``) runs every row through
  real Snakemake DAG orchestration — parallel via ``cores``, resumable via
  ``--rerun-incomplete``, also continuing past a failing row
  (``--keep-going``).

Either way the job's own ``completed``/``failed`` status reflects the whole
campaign. ``results_dir``, if given, is a shared **base** directory — every
row still writes to its own ``<results_dir>/<sample>`` (a literal shared
``--outdir`` across every row would make samples overwrite each other, since
``ocdocker vs``/``pipeline`` write straight under ``--outdir`` with no
nesting of their own).

``get_vs_campaign_context(input_dir=None)``
   Discover a draft multi-sample manifest from an ``input/{sample}/...``
   layout — one subdirectory per sample, matching the convention used by
   ``examples/19_Snakefile_ocdocker_pipeline.smk`` and
   ``examples/20_Snakefile_ocdocker_granular_pipeline.smk``. When
   ``input_dir`` is omitted, ``<served root>/input`` is used automatically if
   present, otherwise the served root itself. Best-effort — a workspace not
   organized this way returns an empty manifest with an explanatory issue,
   not an error; a manifest can also be hand-authored directly for
   ``preview_vs_campaign``. Same payload as ``GET /api/vs-campaign``. See
   :func:`OCDocker.Workbench.VSDesign.discover_vs_campaign_candidates`.

``preview_vs_campaign(manifest)``
   Validate a draft manifest without running anything: path existence,
   engine names, and duplicate sample names, per row (errors/warnings are
   prefixed ``Row i (sample): ...``). Same payload as
   ``POST /api/vs-campaign/preview``. See
   :func:`OCDocker.Workbench.VSDesign.preview_vs_campaign`.

``plan_vs_campaign(manifest, outdir=None)``
   Build the ``vs_campaign`` job payload for a valid manifest — call
   ``preview_vs_campaign`` first and show the user any errors/warnings,
   especially for large manifests. Accepts the same ``engine``/``cores``
   settings described above (via the request body's ``engine``/``cores``
   fields; not yet exposed as direct MCP tool parameters — pass them through
   ``run_job``/``plan_job`` instead when actually launching). On success
   returns ``{"kind": "vs_campaign", "engine", "manifest", "args",
   "results_dir", "cwd", "shell_command"}`` (plus ``"cores"`` for
   ``engine="snakemake"``), ready to hand directly to ``run_job`` (as
   ``kind``/``manifest``/``args``/``engine``/``cores``/``results_dir``/``cwd``)
   to launch the whole batch. Raises
   :class:`~OCDocker.MCP.Server.OCDockerMCPError` if the manifest is
   invalid or ``engine`` is unrecognized. Same payload as
   ``POST /api/vs-campaign/plan``. See
   :func:`OCDocker.Workbench.VSDesign.plan_vs_campaign`.

``get_campaign_progress(job_id)``
   Return structured per-sample progress for a tracked job, parsed live from
   its own log text — no Snakemake internals, no extra process. Returns
   ``{"engine": "snakemake" | "shell" | "unknown", "overall": {...} | None,
   "samples": {name: {"status": "pending" | "running" | "done" | "failed"}}}``.
   Degrades to ``engine: "unknown"`` for a non-``vs_campaign`` job or an
   unrecognized log format — never an error. ``engine="snakemake"`` campaigns
   get reliable per-sample status; ``engine="shell"`` campaigns only reliable
   *aggregate* success/failure counts, since the shell loop never echoes an
   individual row's outcome (see
   :func:`OCDocker.Workbench.CampaignProgress.parse_campaign_progress`).
   Same payload as ``GET /api/jobs/{job_id}/campaign-progress``.

``get_protocol_similarity(metric=None, reference=None)``
   Return pairwise Jaccard feature-similarity across ablation protocols,
   clustered, with an optional outcome-metric overlay and reference-protocol
   diff. ``metric`` selects which metric to overlay on clusters; ``reference``
   selects the protocol used for the added/removed feature diff (defaults to
   ``full_ocscore`` server-side when omitted). Same payload as
   ``GET /api/ablation-protocol-similarity``.

``list_jobs()``
   List every tracked Workbench job — every ``vs``/``pipeline``/
   ``ocscore_train``/``ocscore_reduce`` run ever launched through this
   Workbench API instance, most recent first. Returns ``{"jobs": [...]}``
   where each entry is a
   :class:`OCDocker.Workbench.Models.WorkbenchJobRecord`.

``get_job(job_id)``
   Return one tracked job's current status (``defined`` / ``running`` /
   ``completed`` / ``failed`` / ``cancelled``), command, PID, and exit code.
   Raises :class:`~OCDocker.MCP.Server.OCDockerMCPError` if ``job_id`` is
   unknown.

``get_job_logs(job_id, lines=80)``
   Return a bounded tail of a job's stdout and stderr
   (``{"stdout": {...}, "stderr": {...}}``, each a
   :class:`OCDocker.Workbench.Models.RunLogFilePreview`). ``lines`` caps how
   many trailing lines are returned per stream.

``plan_job(kind, args=None, cwd=None, manifest=None, engine="shell", cores=4, results_dir=None)``
   Preview the exact command a job **would** run — ``{"kind", "command", "cwd"}``
   — without launching it and without any side effects. Always call this (or
   ``run_job`` with ``confirm=False``, which does the same thing) before
   ``run_job(..., confirm=True)`` and show the resulting command to the user.
   ``manifest`` is required for ``kind="vs_campaign"`` (see `Batch
   campaigns`_) and ignored for every other kind. ``engine``, ``cores``, and
   ``results_dir`` are also only meaningful for ``kind="vs_campaign"``.

Execute tools
~~~~~~~~~~~~~

``run_job(kind, args=None, cwd=None, manifest=None, engine="shell", cores=4, results_dir=None, confirm=False)``
   Launch a tracked job as a background subprocess on the machine running the
   Workbench API. ``manifest`` is required for ``kind="vs_campaign"`` (build
   one with ``plan_vs_campaign`` first) and ignored otherwise. ``engine``
   (``"shell"`` or ``"snakemake"``), ``cores`` (Snakemake ``--cores``), and
   ``results_dir`` (shared base output directory, per-sample-nested) are
   also only meaningful for ``kind="vs_campaign"`` — see `Batch campaigns`_.

   - ``confirm=False`` (the default): **launches nothing.** Returns
     ``{"launched": false, "message": "...", "plan": {...}}`` — identical to
     calling ``plan_job`` — so the caller can inspect and show the command
     before committing to it.
   - ``confirm=True``: actually calls ``POST /api/jobs`` with the Workbench
     job bearer token attached, and returns
     ``{"launched": true, "job": {...}}`` where ``job`` is the tracked
     :class:`~OCDocker.Workbench.Models.WorkbenchJobRecord` (status
     ``"running"``, PID, log paths).

   Only pass ``confirm=True`` after the user has explicitly agreed to run
   that specific command — never infer consent from an ambiguous or general
   instruction. See `Authentication`_ for how the bearer token is resolved.

``cancel_job(job_id, confirm=False)``
   Cancel a running tracked job (sends ``SIGTERM``, escalating to
   ``SIGKILL`` after a grace period on the Workbench API side).

   - ``confirm=False`` (the default): **cancels nothing.** Returns
     ``{"cancelled": false, "message": "...", "job": {...}}`` with the job's
     current status, so the caller can confirm this is the right job before
     cancelling it.
   - ``confirm=True``: actually calls ``POST /api/jobs/{job_id}/cancel``
     with the bearer token attached, and returns
     ``{"cancelled": true, "job": {...}}`` with the job's post-cancel status.

   Cancelling an already-finished job is a no-op on the Workbench API side —
   the returned record simply reflects its existing terminal status.

The confirm two-step pattern
------------------------------

``run_job`` and ``cancel_job`` are deliberately **not** split into separate
``plan_*``/``run_*`` tool pairs per job kind — they are single tools with a
``confirm`` flag, because the underlying command construction
(:meth:`OCDocker.Workbench.Jobs.JobManager.plan`) is generic across all four
job kinds and duplicating it per kind would only add tools without adding
precision. The pattern is:

1. The LLM calls ``plan_job`` (or ``run_job``/``cancel_job`` with
   ``confirm`` omitted) and shows the resulting command/target to the user.
2. Only after the user explicitly agrees to that specific action does the
   LLM call ``run_job``/``cancel_job`` again with ``confirm=True``.

This mirrors the "ask before hard-to-reverse, resource-consuming actions"
principle: an LLM cannot launch a multi-hour docking or training job, or
cancel one, from a single ambiguous instruction — a job is only ever
launched or cancelled from an explicit, second, confirmed tool call.

Authentication
----------------

Read/plan tools require no authentication (they map to the Workbench API's
unauthenticated read-only endpoints). ``run_job`` and ``cancel_job`` with
``confirm=True`` attach ``Authorization: Bearer <token>``, where the token is
resolved by :func:`OCDocker.Workbench.Auth.resolve_workbench_job_token` — the
**same function**, and therefore the same token, that
``ocdocker workbench serve`` itself uses to gate its execute endpoints (see
:doc:`OCDocker.Workbench.Auth`). Resolution order:

1. the ``OCDOCKER_WORKBENCH_TOKEN`` environment variable, if set;
2. the token file at ``~/.config/ocdocker/workbench_token``, auto-generated
   (mode 600) on first use by either the MCP server or the Workbench API,
   whichever runs first.

Because both processes resolve the token identically, there is nothing to
configure by hand as long as they run as the same user on the same machine.
Confirm-gated calls made with ``confirm=False`` never resolve or need a
token at all — only the actual execute request does.

Error handling
----------------

Every tool call that fails — a network error reaching the Workbench API, or
an HTTP error status from it — raises
:class:`OCDocker.MCP.Server.OCDockerMCPError` with a message of the form
``"{method} {path} failed ({status_code}): {detail}"`` (or, for a
connection failure, ``"Could not reach Workbench API at {url}: {error}"``).
Tool functions do not catch this themselves; the ``mcp`` SDK's protocol
handler catches it at the MCP request-dispatch layer and turns it into a
standard MCP tool-error result (``isError: true`` with the message as the
content), which the calling LLM sees as a normal, recoverable tool failure —
not a server crash. This was verified end-to-end over the real stdio
transport, not just at the Python-call level.

Submodules
----------

.. toctree::
   :maxdepth: 4

   OCDocker.MCP.Server

Module contents
---------------

.. automodule:: OCDocker.MCP
   :members:
   :undoc-members:
   :show-inheritance:
