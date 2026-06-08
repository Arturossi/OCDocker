# Legacy OCDocker Examples

These scripts use legacy OCScore APIs (pre-staged Optuna studies, legacy optimization modules, or experimental AE→DNN trainers under `Dimensionality.legacy/` and `DNN.future/`).

For the current OCScore pipeline, start with:

- `ocdocker ocscore reduce` (or `examples/14_feature_reduction_pdbbind_dudez.py`)
- `ocdocker ocscore train` (or `examples/15_ocscore_staged_optuna_from_reduction.py`)
- `ocdocker ocscore validate|…|shap|score` (or `examples/16_ocscore_exported_model_tools.py`)

## Scripts

### 01. Train Model from Database (`01_python_api_train_model_from_db.py`)

Trains DNN or XGBoost models using database or CSV data and legacy four-study Optuna hyperparameter search.

Run: `python examples/legacy/01_python_api_train_model_from_db.py --help`

### 02. Future AE → DNN Pipeline (`02_python_api_future_ae_dnn.py`)

Demonstrates the experimental Autoencoder embedding path and future DNN optimizer (requires PyTorch).

Run: `python examples/legacy/02_python_api_future_ae_dnn.py`
