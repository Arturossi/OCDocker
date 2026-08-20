OCDocker.Workbench.Ablation module
==================================

.. automodule:: OCDocker.Workbench.Ablation
   :members:
   :undoc-members:
   :show-inheritance:

OCScore ablation comparison
---------------------------

The ablation analysis helper scans an adopted Workbench workspace, detects
``train/ablations/<policy>`` source paths recorded by adoption, selects a
non-ablation reference run such as ``train`` when possible, and compares each
policy against that reference. The analysis is read-only and does not launch,
stop, copy, or modify OCScore jobs or outputs.

.. code-block:: bash

   ocdocker workbench ablations workbench-runs --metric auc:max --metric validation.loss:min
   ocdocker workbench ablations workbench-runs --baseline train --candidate no_shape_core

The same payload is exposed by the local API at ``/api/ablations`` and by the
browser dashboard decision view at ``/app``. The dashboard renders the ablation table, a
centered delta bar chart for the selected rank metric, and a metric-direction
heatmap for the selected decision metrics.

Expanded feature-set similarity across protocols—Jaccard on policy-expanded
columns, family rollups, and cluster overlays—is provided separately by
:mod:`OCDocker.Workbench.AblationProtocolSimilarity` at
``/api/ablation-protocol-similarity``.
