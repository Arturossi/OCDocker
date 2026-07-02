OCDocker.Workbench package
==========================

The Workbench package defines declarative campaign, study, ablation, run, and
result models used by GUI and automation layers. The active browser dashboard is
now centered on a strict OCScore output layout: baseline replicas live directly
under the served root, or under a ``train/`` child, and ablation studies live
under ``ablation/`` or ``ablations/``. Generic adopted Workbench manifest folders
are reported as unsupported instead of being treated as OCScore runs. The
dashboard exposes curated metrics as sortable ablation-table columns, replica
status, ablation summaries, dataset/role/metric-filtered figure previews,
separate test, validation, and combined generated metric-delta/rank/stability
plots with SVG/CSV export, and separated model-comparison versus selected-model
figure sections through ``/api/ocscore-workspace``. The dashboard **Design**
tab composes custom feature-policy ablations interactively: clone bundled
policies, preview kept/excluded features against workspace metadata, and generate
read-only YAML plus ``ocdocker ocscore train`` command plans through
``/api/ablation-design``. ``/api/vs-design`` mirrors this for single-target
``vs``/``pipeline`` docking runs: discover receptor/ligand/box candidates,
validate a draft selection, and preview the exact launch command (one
receptor/ligand/box per draft — no multi-compound batch screening yet).
Job-execute endpoints (``/api/jobs*``) can launch, track, and cancel ``vs``,
``pipeline``, ``ocscore train``, and ``ocscore reduce`` runs as local
subprocesses; these endpoints require a bearer token while inspection
endpoints remain read-only and unauthenticated.

Submodules
----------

.. toctree::
   :maxdepth: 4

   OCDocker.Workbench.Adoption
   OCDocker.Workbench.Ablation
   OCDocker.Workbench.AblationDesign
   OCDocker.Workbench.AblationProtocolSimilarity
   OCDocker.Workbench.Artifacts
   OCDocker.Workbench.Auth
   OCDocker.Workbench.Bundle
   OCDocker.Workbench.Comparison
   OCDocker.Workbench.Decision
   OCDocker.Workbench.Evidence
   OCDocker.Workbench.Export
   OCDocker.Workbench.IO
   OCDocker.Workbench.Jobs
   OCDocker.Workbench.Launch
   OCDocker.Workbench.Leaderboard
   OCDocker.Workbench.MetricsMatrix
   OCDocker.Workbench.Logs
   OCDocker.Workbench.Models
   OCDocker.Workbench.OCScoreLayout
   OCDocker.Workbench.OptunaDashboard
   OCDocker.Workbench.Overview
   OCDocker.Workbench.Planner
   OCDocker.Workbench.Plots
   OCDocker.Workbench.Preflight
   OCDocker.Workbench.Report
   OCDocker.Workbench.Registry
   OCDocker.Workbench.RunDetail
   OCDocker.Workbench.Results
   OCDocker.Workbench.Schema
   OCDocker.Workbench.Server
   OCDocker.Workbench.Status
   OCDocker.Workbench.Templates
   OCDocker.Workbench.VSDesign
   OCDocker.Workbench.Web
