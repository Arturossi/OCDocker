Bundled OCScore protocol data
================================

Training protocols and feature-ablation policies ship inside the ``ocdocker``
wheel/sdist through ``pyproject.toml`` package data:

.. code-block:: toml

   [tool.setuptools.package-data]
   "OCDocker.OCScore" = ["Protocols/*.yml", "Protocols/Ablations/*.yml"]

Training protocols
------------------

YAML files under ``OCDocker/OCScore/Protocols/`` define replica counts, Optuna
trial budgets, split policy, reporting artifacts, and optional ablation blocks.
The train CLI resolves bundled names (for example ``development``,
``production``, ``smoke-test``) via
:func:`OCDocker.OCScore.Optimization.StagedTrainProtocol.resolve_protocol_path`.

Feature-policy ablations
------------------------

YAML files under ``OCDocker/OCScore/Protocols/Ablations/`` constrain descriptor
pools before train-only feature reduction. Policies are discovered through
:mod:`OCDocker.OCScore.Utils.FeaturePolicy` and referenced from
``ocdocker ocscore train --feature-policy``.
