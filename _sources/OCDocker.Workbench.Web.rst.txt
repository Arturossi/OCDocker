OCDocker.Workbench.Web module
==============================

Browser assets
--------------

The strict OCScore dashboard UI is shipped as static files under
``OCDocker/Workbench/static/``:

- ``index.html`` served at ``/app``
- ``app.css`` served at ``/app.css``
- ``app.js`` served at ``/app.js``

``build_workbench_web_asset()`` reads those files from disk at request time.
The files are included in the wheel/sdist through ``pyproject.toml`` package
data for ``OCDocker.Workbench``. Brand images are served from repository or
package asset paths when present:

- ``/app-favicon.png``
- ``/app-brand-logo.png``

API
---

.. automodule:: OCDocker.Workbench.Web
   :members:
   :undoc-members:
   :show-inheritance:
