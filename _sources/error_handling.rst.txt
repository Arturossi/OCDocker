Error Handling
==============

OCDocker uses a standardized error reporting model for operational/user-facing
failures, alongside standard Python exceptions for programming errors.

Full guide
----------

- :download:`Error Handling Guide (Markdown) <../ERROR_HANDLING.md>`

Quick guidance
--------------

- Use ``OCDocker.Error`` helpers for recoverable, user-facing operational errors
  (files, configuration, validation, docking workflow failures).
- Use standard exceptions (for example ``TypeError``/``ValueError``) for
  programming errors and invalid internal assumptions.
- Keep error handling patterns consistent within each module.

Related pages
-------------

- :doc:`OCDocker.Error`
- :doc:`manual`
- :doc:`usage`
