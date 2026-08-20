OCDocker.Workbench.Evidence module
==================================

.. automodule:: OCDocker.Workbench.Evidence
   :members:
   :undoc-members:
   :show-inheritance:

OCScore evidence discovery
--------------------------

The evidence helper scans adopted Workbench source paths and discovers produced
OCScore evidence such as performance CSV files, Optuna trial traces, SHAP value
exports, SHAP figures, cross-validation tables, and analysis figures. The scan is
read-only: it does not launch jobs, stop jobs, mutate OCScore output directories,
or regenerate plots.

.. code-block:: bash

   ocdocker workbench evidence workbench-runs --source-depth 5
   ocdocker workbench serve workbench-runs --host 127.0.0.1 --port 8765

The local API exposes the payload at ``/api/evidence``. The Workbench browser
dashboard can use the same payload to render performance, optimization, and SHAP previews
when those evidence files are present in adopted OCScore outputs. Raster figure
previews are served through ``/api/evidence-asset`` only when the requested image
belongs to a discovered SHAP or analysis figure under an adopted source path.
