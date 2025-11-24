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

### 2. Multi-Engine Pipeline (`02_cli_multi_engine_pipeline.sh`)
Examples of running multi-engine docking pipelines:
- Pipeline with Vina, Smina, and PLANTS
- Pipeline with custom timeouts
- Pipeline with subset of engines

### 3. Diagnostics (`03_cli_diagnostics.sh`)
Examples of checking your installation:
- Running diagnostics
- Creating configuration files
- Checking version

### 4. Interactive Console (`04_cli_console.py`)
Guide to using the interactive console for step-by-step workflows.

## Python API Examples

### 5. Vina Docking (`05_python_api_vina.py`)
Complete example of using Vina programmatically:
- Creating receptor and ligand objects
- Preparing structures
- Running docking
- Reading results
- Rescoring

### 6. Smina Docking (`06_python_api_smina.py`)
Example of using Smina for docking and rescoring:
- Smina docking workflow
- Using Smina for rescoring only (after Vina docking)

### 7. PLANTS Docking (`07_python_api_plants.py`)
Example of using PLANTS (uses MOL2 format):
- PLANTS-specific preparation
- Docking with PLANTS
- Rescoring with PLANTS

### 8. ODDT Rescoring (`08_python_api_rescoring_oddt.py`)
Example of using ODDT for rescoring:
- Running ODDT rescoring on docked poses
- Converting results to different formats

### 9. RMSD Clustering (`09_python_api_clustering.py`)
Example of clustering poses from multiple engines:
- Combining poses from different engines
- Calculating RMSD matrix
- Clustering and finding medoids

### 10. Complete Workflow (`10_python_api_complete_workflow.py`)
End-to-end workflow example:
- Multi-engine docking
- Pose clustering
- Rescoring
- Results analysis

### 11. Train Model from Database (`11_python_api_train_model_from_db.py`)
Example of training machine learning models using data from the OCDocker database:
- Loading data from database
- Finding best hyperparameters across multiple Optuna studies
- Training DNN or XGBoost models
- Saving trained models and masks
- Using the full preprocessing pipeline

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

Activate the OCDocker environment and run:

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
- `OCDOCKER_USE_SQLITE`: Use SQLite instead of MySQL (set to `1`)
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

