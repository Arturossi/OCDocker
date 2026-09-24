OCDocker.Workbench.AblationProtocolSimilarity module
====================================================

.. automodule:: OCDocker.Workbench.AblationProtocolSimilarity
   :members:
   :undoc-members:
   :show-inheritance:

Ablation protocol similarity
----------------------------

This helper compares OCScore ablation **protocols** by their **expanded**
feature sets rather than by raw YAML include/exclude tokens. Each bundled or
workspace policy is resolved with
:func:`OCDocker.OCScore.Utils.FeaturePolicy.apply_feature_policy` against a
shared candidate feature universe discovered from replica
``feature_policy_metadata.json`` or, as a fallback, from ``raw_prepare``
tables via :mod:`OCDocker.Workbench.AblationDesign`.

Wildcard patterns such as ``ligand_*`` therefore expand to every matching
column before Jaccard similarity, family rollups, reference diffs, and
hierarchical clustering are computed. Clustering uses feature similarity only;
an optional comparison metric overlays mean outcomes per cluster without
changing the cluster assignment.

The read-only payload is exposed by the local API at
``GET /api/ablation-protocol-similarity`` and rendered in the **Protocol
similarity** zone of the Ablation dashboard tab (heatmap, family grid, cluster
summaries, and reference diffs). Outcome-oriented ablation tables remain in
:mod:`OCDocker.Workbench.Ablation`.

.. code-block:: bash

   curl 'http://127.0.0.1:8765/api/ablation-protocol-similarity?metric=auc:max&reference=full_ocscore&include_catalog_only=true'

Query parameters:

* ``metric`` — optional comparison metric for cluster mean overlays (same syntax
  as :func:`OCDocker.Workbench.Ablation.parse_ablation_metric`).
* ``reference`` — reference policy for add/remove diffs (defaults to
  ``full_ocscore`` when available).
* ``include_catalog_only`` — defaults to ``true``; when ``false``, restrict to executed workspace folders (API/CLI). The dashboard always loads the full catalog and filters client-side.
