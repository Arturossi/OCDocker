OCDocker.Workbench package
==========================

The Workbench package defines declarative campaign, study, ablation, run, and
result models used by future GUI and automation layers. Current helpers build prepared run bundles, adopt existing output directories into separate Workbench manifest workspaces, compare adopted OCScore ablations, discover read-only OCScore evidence such as performance tables, Optuna traces, SHAP exports, and analysis figures, export publication scaffolds, plan
commands, serialize specs, emit starter templates and JSON Schemas, preflight specs, prepare launch plans, inventory workspaces, index artifacts, compare result runs, build dashboard overviews, inspect run status, preview logs, build aggregate run drill-downs, rank result metrics, build metric matrices, build metric catalogs, Pareto fronts, plot-ready payloads, and composed analysis reports, serve a local read-only API and embedded browser dashboard, and summarize result artifacts; they do not execute Snakemake
or OCScore runs.

Submodules
----------

.. toctree::
   :maxdepth: 4

   OCDocker.Workbench.Adoption
   OCDocker.Workbench.Ablation
   OCDocker.Workbench.Artifacts
   OCDocker.Workbench.Bundle
   OCDocker.Workbench.Comparison
   OCDocker.Workbench.Decision
   OCDocker.Workbench.Evidence
   OCDocker.Workbench.Export
   OCDocker.Workbench.IO
   OCDocker.Workbench.Launch
   OCDocker.Workbench.Leaderboard
   OCDocker.Workbench.MetricsMatrix
   OCDocker.Workbench.Logs
   OCDocker.Workbench.Models
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
   OCDocker.Workbench.Web
