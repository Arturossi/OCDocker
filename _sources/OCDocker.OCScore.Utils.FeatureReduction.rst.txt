OCDocker.OCScore.Utils.FeatureReduction module
==============================================

Granular feature-reduction utilities for OCScore descriptor datasets. The
module keeps core behavior in reusable functions and dataclasses; the
``run_feature_reduction_protocol`` helper only orchestrates those functions.

API Surface
-----------

The public surface is intentionally limited to descriptor block handling,
missing-row filtering, column-quality filters, correlation diagnostics, result
composition, protocol generation, and output writing. Thin wrappers that only
expose one branch of another function's behavior are kept private; for example,
pattern-only block detection is handled by ``split_descriptor_blocks`` with the
metadata flags disabled instead of a second public function. Existing OCScore
training, DNN, autoencoder, and downstream metric code is not rewired to call
this module automatically. Training pipelines can consume the selected feature
list later.

The documented public function surface is:

* ``validate_descriptor_frame``
* ``split_descriptor_blocks`` and ``summarize_blocks``
* ``drop_rows_with_missing_values``
* ``find_constant_features``, ``find_near_constant_features``,
  ``find_duplicate_features``, and ``apply_feature_drops``
* ``compute_intra_block_correlations`` and ``filter_correlated_features``
* ``compute_cross_block_correlations``, ``compute_cross_block_predictability``,
  and ``filter_cross_block_redundant_features``
* ``compose_selected_features`` and ``build_reduced_dataframe``
* ``build_feature_reduction_protocol`` and ``write_feature_reduction_outputs``
* ``run_feature_reduction_protocol``

Rewiring Notes
--------------

* ``OCDocker.OCScore.Utils.IO.load_data`` is not changed because it currently
  drops rows with missing values before detailed row-level reporting. The new
  orchestration helper uses raw ``pandas.read_csv`` when an input path is
  supplied, so missing-row removal can be reported reproducibly.
* Ligand and receptor descriptor metadata are read from ``Ligand.allDescriptors``
  and ``Receptor.allDescriptors`` when those classes can be imported. Pattern
  matching remains the fallback.
* Scoring descriptors use ``Complexes.allDescriptors`` when available and fall
  back to configurable scoring-function prefixes.
* Cross-block filtering is disabled by default. Cross-block diagnostics do not
  drop features unless filtering is explicitly enabled.

Versioning Notes
----------------

This is an additive API module. It does not remove or change existing public
behavior. A minor version bump is appropriate if this API is released as a new
feature; a patch version is only appropriate if it remains internal or
undocumented.

API Reference
-------------

Configuration and result classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.DescriptorBlocks
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.BlockDetectionConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.MissingRowsConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.ColumnQualityConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.IntraBlockCorrelationConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.CrossBlockDiagnosticsConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.CrossBlockFilteringConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.FeatureReductionConfig
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.MissingRowsResult
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.CorrelationReport
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.CorrelationFilterResult
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.CrossBlockFilterResult
   :no-index:

.. autoclass:: OCDocker.OCScore.Utils.FeatureReduction.FeatureReductionResult
   :no-index:

Public functions
~~~~~~~~~~~~~~~~

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.validate_descriptor_frame

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.split_descriptor_blocks

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.summarize_blocks

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.drop_rows_with_missing_values

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.find_constant_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.find_near_constant_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.find_duplicate_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.apply_feature_drops

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.compute_intra_block_correlations

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.filter_correlated_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.compute_cross_block_correlations

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.compute_cross_block_predictability

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.filter_cross_block_redundant_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.compose_selected_features

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.build_reduced_dataframe

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.build_feature_reduction_protocol

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.write_feature_reduction_outputs

.. autofunction:: OCDocker.OCScore.Utils.FeatureReduction.run_feature_reduction_protocol
