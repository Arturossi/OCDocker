# OCDocker Examples

This directory contains examples demonstrating how to use OCDocker for molecular docking, virtual screening, and analysis.

## Table of Contents

- [CLI Examples](#cli-examples)
- [Python API Examples](#python-api-examples)
- [Getting Started](#getting-started)

## CLI Examples

### 1. Basic Docking (`01_cli_basic_docking.sh`)
Simple examples of running docking with the CLI:
- Basic Vina docking
- Docking with database storage
- Docking without rescoring
Run: `bash examples/01_cli_basic_docking.sh`

### 2. Multi-Engine Pipeline (`02_cli_multi_engine_pipeline.sh`)
Examples of running multi-engine docking pipelines:
- Pipeline with Vina, Smina, and PLANTS
- Pipeline with custom timeouts
- Pipeline with subset of engines
Run: `bash examples/02_cli_multi_engine_pipeline.sh`

### 3. Diagnostics (`03_cli_diagnostics.sh`)
Examples of checking your installation:
- Running diagnostics
- Creating configuration files
- Checking version
Run: `bash examples/03_cli_diagnostics.sh`

### 4. Interactive Console (`04_cli_console.py`)
Guide to using the interactive console for step-by-step workflows.
Run: `ocdocker console --conf OCDocker.cfg`

### 5. Running Scripts (`12_cli_script_example.py`)
Example of running Python scripts with OCDocker libraries pre-loaded:
- Using the `ocdocker script` command
- Accessing pre-loaded OCDocker modules
- Working with script arguments
- Example workflow structure
Run: `ocdocker script --conf OCDocker.cfg --allow-unsafe-exec examples/12_cli_script_example.py`
(`--allow-unsafe-exec` is required for trusted in-process script execution).

## Python API Examples

### 6. Vina Docking (`05_python_api_vina.py`)
Complete example of using Vina programmatically:
- Creating receptor and ligand objects
- Preparing structures
- Running docking
- Reading results
- Rescoring
Run: `python examples/05_python_api_vina.py`

### 7. Smina Docking (`06_python_api_smina.py`)
Example of using Smina for docking and rescoring:
- Smina docking workflow
- Using Smina for rescoring only (after Vina docking)
Run: `python examples/06_python_api_smina.py`

### 8. PLANTS Docking (`07_python_api_plants.py`)
Example of using PLANTS (uses MOL2 format):
- PLANTS-specific preparation
- Docking with PLANTS
- Rescoring with PLANTS
Run: `python examples/07_python_api_plants.py`

### 9. ODDT Rescoring (`08_python_api_rescoring_oddt.py`)
Example of using ODDT for rescoring:
- Running ODDT rescoring on docked poses
- Converting results to different formats
Run: `python examples/08_python_api_rescoring_oddt.py`

### 10. RMSD Clustering (`09_python_api_clustering.py`)
Example of clustering poses from multiple engines:
- Combining poses from different engines
- Calculating RMSD matrix
- Clustering and finding medoids
Run: `python examples/09_python_api_clustering.py`

### 11. Complete Workflow (`10_python_api_complete_workflow.py`)
End-to-end workflow example:
- Multi-engine docking
- Pose clustering
- Rescoring
- Results analysis
Run: `python examples/10_python_api_complete_workflow.py`

### 12. Complete OCScore Pipeline (`11_python_api_complete_ocscore_pipeline.py`)
Complete end-to-end pipeline to obtain OCScore results from scratch:
- Receptor and ligand preparation
- Multi-engine docking (Vina, PLANTS)
- Pose clustering to find representative poses
- Rescoring with multiple scoring functions (ODDT, PLANTS, Vina, SMINA)
- Feature extraction (receptor and ligand descriptors)
- Model inference using trained OCScore model
- Automatic mapping of rescoring results to database column names
Run: `python examples/11_python_api_complete_ocscore_pipeline.py`

### 13. CSV Inference (`13_python_api_inference_from_csv.py`)
Example of OCScore inference loading features directly from a `.csv` file:
- Resolving model artifacts from `OCScore_models` (or custom directory)
- Loading OCDocker config to enforce `reference_column_order`
- Optional mask loading
- Optional scaler loading for consistent preprocessing
- Running `ocscoring.get_score(...)` with `data=<csv_path>`
- Writing an output CSV with the original rows/columns plus an `OCSCORE` column
Run: `python examples/13_python_api_inference_from_csv.py --csv-path /path/to/features.csv --model-name OCScore --config-path /path/to/OCDocker.cfg --output-csv /path/to/scored.csv`

### 14. Shared PDBbind + DUDEz Feature Reduction (`14_feature_reduction_pdbbind_dudez.py`)
Example of unsupervised shared feature reduction from two pipeline result archives:
- Loads pipeline tables from PDBbind and DUDEz inputs (`.csv`, directory, or `.tar.gz` with `pipeline_results.csv`, `PDBbind.csv`, or `DUDEz.csv`)
- Preserves `experimental` for PDBbind and creates `NaN` DUDEz affinity values
- Preserves DUDEz `kind` and creates a `label` column for later classification
- Merges raw unreduced PDBbind and DUDEz pipeline tables into `merged_input_dataset.csv`
- Writes separate `raw_pdbbind.csv` and `raw_dudez.csv` plus `prepare_manifest.json`
- Does **not** fit feature reduction (train-only reduction happens in `ocscore train`)
Run: `python examples/14_feature_reduction_pdbbind_dudez.py --pdbbind-archive /path/to/pdbbind.tar.gz --dudez-archive /path/to/DUDEz.tar.gz --output-dir /path/to/raw_prepare`

CLI: `ocdocker ocscore reduce` (same flags; requires `pip install "ocdocker[ml]"`)

### 15. Staged OCScore Optuna from raw inputs (`15_ocscore_staged_optuna_from_reduction.py`)
Example of starting the unified staged Optuna modeling protocol from raw unreduced inputs:
- Loads `merged_input_dataset.csv` (or separate raw PDBbind/DUDEz CSVs)
- Creates a fixed outer split, fits train-only feature reduction on PDBbind train rows
- Runs PDBbind RMSE-only regression Optuna, transfers the feature extractor, then runs DUDEz screening Optuna
- Keeps PDBbind and DUDEz checkpoints, metrics, studies, and protocol logs separate
Run: `ocdocker ocscore train --protocol development --raw-input-dir /path/to/raw_prepare --output-dir /path/to/optuna_output`

For production validation (protocol files, leakage guards, provenance), see [docs/ocscore-production-protocol.md](../docs/ocscore-production-protocol.md) and `ocdocker ocscore train --protocol production --raw-input-dir ... --output-dir ...`.

CLI: `ocdocker ocscore train` (`--protocol`, `--raw-input-dir` or `--merged-input` or `--pdbbind-input` + `--dudez-input`, `--output-dir`)

### 16. OCScore exported model tools (`16_ocscore_exported_model_tools.py`)

Tools for validating, loading, cross-validating, plotting, rendering architecture diagrams, running SHAP on, and **scoring raw pipeline archives** with exported `best_model/` bundles from example 15.

CLI: `ocdocker ocscore validate|load|retrain|cross-validate|plot|architecture-plot|shap|score` (same flags per subcommand). SHAP on exports: `ocdocker ocscore shap`.

**Inference kit (portable scoring):**
- `best_model/` export directory (weights, architecture, `feature_metadata.json`, optional `scaler.joblib`)
- Raw pipeline input (`.csv`, directory, or tar) containing `pipeline_results.csv`, `PDBbind.csv`, or `DUDEz.csv` with the export's selected feature columns
- For DUDEz transfer models: linked PDBbind `best_model/` (or pass `--pdbbind-export-dir`)

Scoring uses the **frozen `selected_features` list** from the export bundle. It does not re-run feature reduction or refit scalers on new data. Optional `feature_reduction_protocol.json` from example 14 is audit metadata only.

```bash
python examples/16_ocscore_exported_model_tools.py score \
  --export-dir /path/to/replica_000/pdbbind/best_model \
  --raw-archive /path/to/new_pipeline.tar.gz \
  --output-csv /path/to/predictions.csv
```

Architecture figures can be generated from an export bundle or a manual JSON/YAML architecture file:

```bash
ocdocker ocscore architecture-plot \
  --export-dir /path/to/replica_000/pdbbind/best_model \
  --output-dir /path/to/figures \
  --formats png
```

Architecture figures are compact main-network-only by default. Use `--show-decoder` to include the auxiliary reconstruction branch. In `18_run_full_pipeline.sh`, set `ARCHITECTURE_PLOT_INCLUDE_DECODER=true` for the same behavior.

Other subcommands: `validate`, `load`, `retrain`, `cross-validate`, `plot`, `architecture-plot`, `shap`.

### 17. DUDEz SF baseline comparison (`17_ocscore_dudez_sf_baseline_comparison.py`)

Compares OCScore to individual scoring functions, **descriptor aggregates** (`desc_mean`, … over all model input features), and **SF consensus** (`sf_mean`, … across Vina/Gnina/Smina/… columns only) on saved validation/test splits. It writes pooled metrics (`dudez_sf_baseline_comparison.csv`) and **per-receptor** metrics (`dudez_sf_baseline_per_target.csv`). With `--figures-dir`, it also emits per-target heatmaps, boxplots, and OCScore win-count charts.

**Cross-validation outputs** (under `<export_dir>/cross_validation/` or `--output-dir`):
- `cross_validation_results.json` — fold metrics and diagnostics (including entity-overlap warnings when present)
- `cross_validation_per_target_metrics.csv` — per-receptor BEDROC/ROC-AUC (and scoring-function baselines) for DUDEz receptor-grouped CV

## Getting Started

### Prerequisites

1. **Install OCDocker**: Follow the installation instructions in the main README.md
2. **Configure OCDocker**: Create or copy `OCDocker.cfg` configuration file
3. **Install Docking Engines**: Ensure Vina, Smina, and/or PLANTS are installed and configured

### Running CLI Examples

Make the scripts executable and run them:

```bash
chmod +x examples/*.sh
./examples/01_cli_basic_docking.sh
```

**Note**: Update the file paths in the examples to match your system.

### Running Python Examples

Activate your conda environment and run:

```bash
conda activate ocdocker
python examples/05_python_api_vina.py
```

**Note**: Update the file paths in the examples to match your test data location.

### Configuration

Before running examples, ensure you have:

1. A valid `OCDocker.cfg` file (use `ocdocker init-config` to create one)
2. Test data files (receptor, ligands, box files)
3. Required docking engines installed and configured

### Environment Variables

You can set these environment variables to customize behavior:

- `OCDOCKER_CONFIG`: Path to your `OCDocker.cfg` file
- `OCDOCKER_DB_BACKEND` / `DB_BACKEND`: Select backend (`postgresql`, `mysql`, `sqlite`)
- `OCDOCKER_TIMEOUT`: Default timeout for external tools (seconds)
- `OCDOCKER_NO_AUTO_BOOTSTRAP`: Disable auto-bootstrap on import (set to `1`)

### File Structure

OCDocker expects a specific file structure for organizing docking data:

```
receptor/
└── compounds/
    ├── candidates/
    │   └── molecule_1/
    ├── decoys/
    │   └── molecule_A/
    └── ligands/
        └── molecule_a/
```

See the main README.md for more details on the file structure.

## Tips

1. **Start Simple**: Begin with example 1 (basic CLI docking) or example 5 (basic Python API)
2. **Check Diagnostics**: Use `ocdocker doctor` to verify your installation before running examples
3. **Use Test Data**: The examples reference `test_files/` directory - ensure you have test data or update paths
4. **Read Logs**: Check log files in output directories for detailed information
5. **Database Storage**: Use `--store-db` flag to persist results in the database

## Troubleshooting

### Common Issues

1. **Binary not found**: Run `ocdocker doctor` to check if docking engines are properly configured
2. **Configuration errors**: Verify your `OCDocker.cfg` file has correct paths
3. **File format errors**: Ensure input files are in the correct format (PDB for receptor, SMILES/SDF for ligands)
4. **Timeout errors**: Increase timeout values if docking takes longer than expected

### Getting Help

- Run `ocdocker doctor` for diagnostics
- Check the main README.md for detailed documentation
- Review log files in output directories
- Use the interactive console (`ocdocker console`) for debugging

## License

These examples are part of OCDocker and are subject to the same license terms.
See the LICENSE file in the main directory for details.
