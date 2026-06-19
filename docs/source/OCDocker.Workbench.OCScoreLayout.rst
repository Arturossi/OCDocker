OCDocker.Workbench.OCScoreLayout module
========================================

Strict OCScore workspace discovery
----------------------------------

``OCDocker.Workbench.OCScoreLayout`` is the supported dashboard data source for
OCScore result control and comparison. It intentionally scans the OCScore output
layout directly instead of building a generic gallery from every possible metric
or artifact file.

The expected root layout is::

   output/
     replica_1/
     replica_2/
     replica_3/
     replica_4/
     replica_5/
     ablation/
       no_shape/
         replica_1/
         replica_2/
         replica_3/
         replica_4/
         replica_5/

The scanner also accepts zero-padded replica names such as ``replica_001`` and
``ablations/`` as an alias for ``ablation/`` so existing OCScore outputs can be
inspected without being moved. If the served path is the broader OCScore output directory
that contains ``train/``, the scanner automatically uses ``train/`` when that is
where the replicas and ablations live.

The dashboard keeps a small curated metric set: BEDROC, ROC AUC, PR AUC, EF 1%,
EF 5%, best validation metric, RMSE, MAE, and R2. Extra metric names can be
requested explicitly through the API, but random numeric columns are not promoted
by default.

Useful entry points
-------------------

.. code-block:: bash

   ocdocker workbench serve /path/to/ocscore/output --host 127.0.0.1 --port 8765

The local API exposes the strict payload at ``/api/ocscore-workspace``. The
embedded browser dashboard at ``/app`` renders baseline replicas, ablation
studies, study-level replica status, curated metric summaries, generated
decision plots, and labeled figures. Figure discovery tags known datasets
(``dudez``, ``pdbbind``, ``casf``, ``dekois``, and ``lit_pcba``), common roles
such as SHAP beeswarm, SHAP importance, CV mean/std, CV heatmap, per-target
validation, architecture, and performance plots, and curated metric names when
they appear in figure filenames. The dashboard filters previews by dataset, role,
figure metric, and artifact group so model-comparison plots stay separate from
selected-model diagnostics. Generated decision plots expose an explicit result
scope selector for test metrics, validation metrics, or both together, and can be
exported as SVG plus CSV for paper-ready review. The dashboard still caps large
galleries and keeps generated comparison plots visible even when no source image
files are present. It is read-only at this stage.

API
---

.. automodule:: OCDocker.Workbench.OCScoreLayout
   :members:
   :undoc-members:
   :show-inheritance:
