OCDocker.Workbench.Adoption module
==================================

.. automodule:: OCDocker.Workbench.Adoption
   :members:
   :undoc-members:
   :show-inheritance:

OCScore ablation adoption
-------------------------

The adoption scanner is read-only. It writes Workbench manifests only to the
destination workspace and leaves the original OCScore output tree untouched.
When the scanner reaches an OCScore ``ablations/`` container, each direct
feature-policy directory below it is inspected as an adopted run, even when the
regular ``--max-depth`` value would otherwise stop at the parent train folder.

This allows an existing train output such as ``train/ablations/no_shape_core``
to appear as a separate Workbench run in the dashboard:

.. code-block:: bash

   ocdocker workbench adopt-plan OCScore/output/train --max-depth 0 --require-metrics --output adoption_plan.json
   ocdocker workbench adopt OCScore/output/train workbench-runs --max-depth 0 --require-metrics
   ocdocker workbench serve workbench-runs --host 127.0.0.1 --port 8765

The same discovery also works from a parent output root when the train folder is
inside the requested depth.

Use ``--require-metrics`` when the source tree also contains exported
placeholder directories. This keeps the adopted workspace focused on completed
result folders and prevents metricless placeholders from consuming duplicate
run ids.

