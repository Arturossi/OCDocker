# Test suite inventory

Generated: 2026-05-30 (plan 008, U1). Updated after U2–U7 triage (2026-05-30).

**U2 deletions (removed from tree):** `test_cli_utils_edge_cases.py`, `test_db_package_init.py`, `test_postprocessing_init.py`, `test_io_edge_cases.py`.

**Branch-coverage satellites (U3):** kept — primary docking/preparation modules exceed 600 lines.

| Module | Tests | Tier | Action |
|--------|------:|------|--------|
| `tests/cli/test_cli_core_branches.py` | 25 | **A** | Keep |
| `tests/cli/test_cli_doctor.py` | 3 | **A** | Keep |
| `tests/cli/test_cli_init_version.py` | 2 | **A** | Keep |
| `tests/cli/test_cli_manifest.py` | 4 | **A** | Keep |
| `tests/cli/test_cli_ocscore.py` | 9 | **A** | Keep |
| `tests/cli/test_cli_parse_vs.py` | 2 | **A** | Keep |
| `tests/cli/test_cli_pipeline_db_mapping.py` | 3 | **A** | Keep |
| `tests/cli/test_cli_utilities.py` | 9 | **A** | Keep |
| ~~`tests/cli/test_cli_utils_edge_cases.py`~~ | — | **B** | **Deleted** — merged into test_cli_utilities.py |
| `tests/cli/test_packaging_metadata.py` | 4 | **A** | Keep |
| `tests/core/legacy/test_legacy_impact.py` | 5 | **C** | Gate (legacy marker) |
| `tests/core/legacy/test_legacy_stattests_security.py` | 6 | **C** | Gate (legacy marker) |
| `tests/core/test_config.py` | 10 | **A** | Keep |
| `tests/core/test_error.py` | 6 | **A** | Keep |
| `tests/core/test_exception_handling.py` | 10 | **A** | Keep |
| `tests/core/test_initialise.py` | 1 | **A** | Keep |
| `tests/core/test_initialise_config_parser.py` | 10 | **A** | Keep |
| `tests/core/test_initialise_import_behavior.py` | 1 | **A** | Keep |
| `tests/core/test_ligand.py` | 28 | **A** | Keep |
| `tests/core/test_ligand_receptor_extended.py` | 15 | **A** | Keep |
| `tests/core/test_ligand_receptor_regression.py` | 10 | **A** | Keep |
| `tests/core/test_molecule_processing_extended.py` | 12 | **A** | Keep |
| `tests/core/test_receptor.py` | 37 | **A** | Keep |
| `tests/db/test_basedb.py` | 10 | **A** | Keep |
| `tests/db/test_db_connection_leaks.py` | 8 | **A** | Keep |
| `tests/db/test_db_export_extended.py` | 3 | **A** | Keep |
| `tests/db/test_db_export_sqlite.py` | 1 | **A** | Keep |
| `tests/db/test_db_extended.py` | 3 | **A** | Keep |
| `tests/db/test_db_integration.py` | 9 | **A** | Keep |
| `tests/db/test_db_models_base.py` | 5 | **A** | Keep |
| `tests/db/test_db_models_structure.py` | 3 | **A** | Keep |
| ~~`tests/db/test_db_package_init.py`~~ | — | **D** | **Deleted** |
| `tests/db/test_dbminimal.py` | 6 | **A** | Keep |
| `tests/db/test_dudez.py` | 7 | **A** | Keep |
| `tests/db/test_pdbbind.py` | 10 | **A** | Keep |
| `tests/docking/test_basevinalike_branch_coverage.py` | 2 | **B** | Keep (primary >600 lines) |
| `tests/docking/test_docking_log_readers.py` | 2 | **A** | Keep |
| `tests/docking/test_gnina_branch_coverage.py` | 16 | **B** | Keep (primary >600 lines) |
| `tests/docking/test_gnina_rescore.py` | 3 | **A** | Keep |
| `tests/docking/test_plants.py` | 44 | **A** | Keep |
| `tests/docking/test_plants_branch_coverage.py` | 5 | **B** | Keep (primary >600 lines) |
| `tests/docking/test_plants_prepare.py` | 1 | **A** | Keep |
| `tests/docking/test_plants_utilities.py` | 7 | **A** | Keep |
| `tests/docking/test_pose_index.py` | 1 | **A** | Keep |
| `tests/docking/test_smina.py` | 40 | **A** | Keep |
| `tests/docking/test_smina_branch_coverage.py` | 4 | **B** | Keep (primary >600 lines) |
| `tests/docking/test_smina_logs.py` | 2 | **A** | Keep |
| `tests/docking/test_smina_prepare.py` | 1 | **A** | Keep |
| `tests/docking/test_smina_utilities.py` | 5 | **A** | Keep |
| `tests/docking/test_split_and_convert.py` | 4 | **A** | Keep |
| `tests/docking/test_vina.py` | 40 | **A** | Keep |
| `tests/docking/test_vina_branch_coverage.py` | 14 | **B** | Keep (primary >600 lines) |
| `tests/docking/test_vina_logs.py` | 3 | **A** | Keep |
| `tests/docking/test_vina_prepare.py` | 1 | **A** | Keep |
| `tests/docking/test_vina_smina_configs.py` | 2 | **A** | Keep |
| `tests/docking/test_vinalike_parsing_edge_cases.py` | 19 | **A** | Keep |
| `tests/examples/test_ocscore_exported_model_tools_score.py` | 2 | **A** | Protect |
| `tests/examples/test_ocscore_staged_optuna_from_reduction_example.py` | 1 | **A** | Protect |
| `tests/integration/test_coverage_batch_regression.py` | 7 | **B** | Review per-test |
| `tests/integration/test_integration_docking_workflow.py` | 8 | **A** | Protect |
| `tests/ocscore/legacy/test_legacy_analysis_core.py` | 10 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_autoencoder_future.py` | 2 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_autoencoder_optimizer.py` | 4 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_dimensionality_pca.py` | 5 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_dnn_future_embeddings.py` | 1 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_dnn_optimizer.py` | 17 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_future_autoencoder_trainer_utilities.py` | 3 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_future_dimensionality_utils_losses.py` | 3 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_future_dnn_optimizer.py` | 4 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_genetic_algorithm.py` | 2 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_impactplots.py` | 5 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_mutable_defaults.py` | 1 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_nnutils.py` | 3 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_optimization_runtime.py` | 19 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_performance_evaluation.py` | 7 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_rankingmetrics.py` | 5 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_shap_integration.py` | 14 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_simple_consensus.py` | 5 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_study_parser.py` | 6 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_study_processing.py` | 4 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_trans_optimizer.py` | 2 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_transformer_optimization_extended.py` | 2 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_utils_data_evaluation.py` | 7 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_utils_plotting.py` | 2 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_workers_extended.py` | 9 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_workers_smoke.py` | 5 | **C** | Gate (legacy marker) |
| `tests/ocscore/legacy/test_legacy_xgboost_optimizer.py` | 3 | **C** | Gate (legacy marker) |
| `tests/ocscore/test_cross_validation_plots.py` | 6 | **A** | Keep |
| `tests/ocscore/test_descriptor_aggregate_baselines.py` | 6 | **A** | Keep |
| `tests/ocscore/test_dudez_split.py` | 24 | **A** | Protect |
| `tests/ocscore/test_export_inference.py` | 7 | **A** | Protect |
| `tests/ocscore/test_export_shap.py` | 4 | **A** | Protect |
| `tests/ocscore/test_model_cross_validation.py` | 9 | **A** | Protect |
| `tests/ocscore/test_ocscore_calibration_metrics.py` | 3 | **A** | Keep |
| `tests/ocscore/test_ocscore_dnn_future.py` | 1 | **A** | Keep |
| `tests/ocscore/test_ocscore_dnn_future_core.py` | 5 | **A** | Keep |
| `tests/ocscore/test_ocscore_feature_reduction.py` | 12 | **A** | Protect |
| `tests/ocscore/test_ocscore_io_security.py` | 6 | **A** | Protect |
| `tests/ocscore/test_ocscore_metricsplots.py` | 2 | **A** | Keep |
| `tests/ocscore/test_ocscore_model_export.py` | 2 | **A** | Keep |
| `tests/ocscore/test_ocscore_optuna_search_space.py` | 10 | **A** | Keep |
| `tests/ocscore/test_ocscore_plotting_colouring.py` | 2 | **A** | Keep |
| `tests/ocscore/test_ocscore_plotting_stats.py` | 8 | **A** | Keep |
| `tests/ocscore/test_ocscore_ranking_metrics.py` | 4 | **A** | Keep |
| `tests/ocscore/test_ocscore_replicated_protocol.py` | 5 | **A** | Protect |
| `tests/ocscore/test_ocscore_scoring.py` | 41 | **A** | Keep |
| `tests/ocscore/test_ocscore_screening_metrics.py` | 14 | **A** | Keep |
| `tests/ocscore/test_ocscore_staged_optuna_protocol.py` | 24 | **A** | Protect |
| `tests/ocscore/test_ocscore_utils_data.py` | 5 | **A** | Keep |
| `tests/ocscore/test_ocscore_utils_io.py` | 12 | **A** | Keep |
| `tests/ocscore/test_ocscore_utils_io_extended.py` | 11 | **A** | Keep |
| `tests/ocscore/test_optuna_storage.py` | 2 | **A** | Keep |
| `tests/ocscore/test_pdbbind_split.py` | 7 | **A** | Protect |
| `tests/ocscore/test_pipeline_archive_io.py` | 8 | **A** | Protect |
| `tests/processing/test_dock_parallel_error_handling.py` | 1 | **A** | Keep |
| `tests/processing/test_dock_utilities.py` | 40 | **A** | Keep |
| `tests/processing/test_garbage_collection.py` | 5 | **A** | Keep |
| `tests/processing/test_get_docked_poses.py` | 1 | **A** | Keep |
| `tests/processing/test_postprocessing_digest.py` | 15 | **A** | Keep |
| ~~`tests/processing/test_postprocessing_init.py`~~ | — | **D** | **Deleted** |
| `tests/processing/test_preparation_branch_coverage.py` | 7 | **B** | Keep (primary >600 lines) |
| `tests/processing/test_preparation_strategy.py` | 14 | **A** | Keep |
| `tests/processing/test_preprocessing_prepare.py` | 24 | **A** | Keep |
| `tests/processing/test_rmsd.py` | 2 | **A** | Keep |
| `tests/processing/test_rmsd_clustering.py` | 14 | **A** | Keep |
| `tests/processing/test_rmsd_clustering_extended.py` | 6 | **A** | Keep |
| `tests/rescoring/test_oddt_utilities.py` | 42 | **A** | Keep |
| `tests/toolbox/test_basetools.py` | 1 | **A** | Keep |
| `tests/toolbox/test_constants.py` | 5 | **A** | Keep |
| `tests/toolbox/test_conversion.py` | 2 | **A** | Keep |
| `tests/toolbox/test_downloading.py` | 6 | **A** | Keep |
| `tests/toolbox/test_filesfolders.py` | 7 | **A** | Keep |
| `tests/toolbox/test_filesfolders_edge_cases.py` | 2 | **B** | Review merge |
| `tests/toolbox/test_io.py` | 1 | **A** | Keep |
| ~~`tests/toolbox/test_io_edge_cases.py`~~ | — | **B** | **Deleted** — duplicate of test_io.py |
| `tests/toolbox/test_logging_branches.py` | 7 | **A** | Keep |
| `tests/toolbox/test_logging_edge_cases.py` | 2 | **B** | Review merge |
| `tests/toolbox/test_printing.py` | 10 | **A** | Keep |
| `tests/toolbox/test_printing_edge_cases.py` | 2 | **B** | Review merge |
| `tests/toolbox/test_reproducibility.py` | 1 | **A** | Keep (delegation test removed) |
| `tests/toolbox/test_running.py` | 3 | **A** | Keep |
| `tests/toolbox/test_running_deep.py` | 10 | **A** | Keep |
| `tests/toolbox/test_running_parallel_paths.py` | 2 | **A** | Keep |
| `tests/toolbox/test_security_path_traversal.py` | 8 | **A** | Protect |
| `tests/toolbox/test_toolbox_security.py` | 5 | **A** | Keep |
| `tests/toolbox/test_validation.py` | 16 | **A** | Keep |
| `tests/toolbox/test_validation_edge_cases.py` | 3 | **B** | Review merge |

## Totals

- Modules: 147
- Tests by tier action: A=939 B=66 C=161 D=4

