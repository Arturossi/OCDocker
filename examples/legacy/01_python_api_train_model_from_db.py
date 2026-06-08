#!/usr/bin/env python3
"""
Legacy Example 01: Train Model from Database (pre-staged Optuna)

This example demonstrates how to train machine learning models (DNN or XGBoost) using data
from the OCDocker database. The script:

1. Reads data from the OCDocker database (or CSV file)
2. Applies the full preprocessing pipeline (outlier removal, normalization, PCA, etc.)
3. Finds the best hyperparameters across multiple Optuna studies
4. Trains a model using the best hyperparameters
5. Saves the trained model and feature mask to OCScore_models directory

The script can work with:
- Database data (using --from_db flag)
- CSV files (using --df_path flag)
- Multiple Optuna studies (finds best trial across all studies)
- DNN or XGBoost models
- Custom preprocessing options (PCA, feature selection, etc.)

Usage:
    # Train XGBoost model from CSV
    python examples/legacy/01_python_api_train_model_from_db.py --model_type XGB --model_name my_model --df_path data.csv
    
    # Train DNN model from database with GPU
    python examples/legacy/01_python_api_train_model_from_db.py --from_db --model_type DNN --model_name my_model --use_gpu
    
    # Train DNN model using best trial from multiple Optuna studies
    python examples/legacy/01_python_api_train_model_from_db.py --from_db --model_type DNN --studies NN_Optimization_6 NN_Optimization_7 NN_Optimization_8 NN_Optimization_9 NN_Optimization_10 --use_gpu
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd

from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Any, List, Optional
from urllib.parse import quote_plus, urlparse, urlunparse

# Add parent directory to path to allow importing OCDocker
# This allows the script to be run from any directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import OCDocker modules
import OCDocker.Initialise as init
import OCDocker.OCScore.Optimization.legacy.XGBoost as ocxgb
import OCDocker.OCScore.Utils.legacy.Data as ocscoredata
import OCDocker.OCScore.Utils.IO as ocscoreio

from OCDocker.DB.DB import export_db_to_csv
from OCDocker.DB.Models import Complexes, Ligands, Receptors
from OCDocker.OCScore.Optimization.legacy.models.dnn.DNNOptimizer import DNNOptimizer


def _normalize_db_backend(raw_backend: str) -> str:
    '''Normalize backend aliases used by config and environment variables.
    
    Parameters
    ----------
    raw_backend : str
        Raw backend string from config or environment variable (e.g., "postgresql", "mysql", "sqlite")

    Returns
    -------
    str
        Normalized backend string ("postgresql", "mysql", or "sqlite")
    '''

    backend = str(raw_backend).strip().lower()
    if backend in ('postgresql', 'postgres', 'pgsql'):
        return 'postgresql'
    if backend in ('mysql', 'mariadb'):
        return 'mysql'
    if backend in ('sqlite', 'sqlite3'):
        return 'sqlite'
    return 'postgresql'


def _sqlite_storage_for_model(model_type: str) -> str:
    '''Return a local SQLite storage URL for Optuna studies.

    Parameters
    ----------
    model_type : str
        Type of model (e.g., "DNN", "XGB") to determine which SQLite file to use

    Returns
    -------
    str
        SQLite storage URL (e.g., "sqlite:///NN_optimization.db")
    '''

    mt = str(model_type).upper()
    if mt in ('DNN', 'NN'):
        return "sqlite:///NN_optimization.db"
    if mt == 'XGB':
        return "sqlite:///XGB_optimization.db"
    return "sqlite:///model_optimization.db"


def _build_storage_url_from_config(config: Any, model_type: str) -> str:
    '''Build Optuna storage URL from OCDocker config and backend settings.
    
    Parameters
    ----------
    config : Any
        OCDocker config object (after bootstrap)
    model_type : str
        Type of model (e.g., "DNN", "XGB") to determine which SQLite file to use if backend is SQLite

    Returns
    -------
    str
        Optuna storage URL (e.g., "postgresql+psycopg://user:***@host:port/db" or "sqlite:///NN_optimization.db")
    '''

    backend_env = os.getenv('OCDOCKER_DB_BACKEND', '') or os.getenv('DB_BACKEND', '')
    backend_cfg = getattr(config.database, 'backend', '')
    backend = _normalize_db_backend(backend_env or backend_cfg or 'postgresql')

    if backend == 'sqlite':
        return _sqlite_storage_for_model(model_type)

    user = config.database.user
    password = config.database.password
    host = config.database.host
    port = config.database.port
    db = config.database.optimizedb  # OPTIMIZEDB stores Optuna studies

    if backend == 'mysql':
        driver = "mysql+pymysql"
    else:
        driver = "postgresql+psycopg"

    return f"{driver}://{user}:{quote_plus(password)}@{host}:{port}/{db}"


def load_data_from_database(session: Session, methodology: Optional[str] = None) -> pd.DataFrame:
    ''' Load data from the database and convert to DataFrame format expected by the pipeline.
    
    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        Database session
    methodology : str, optional
        Filter by methodology (e.g., 'NN', 'XGB', 'Trans', 'AE', 'PCA', 'PCA80', 'PCA85', 'PCA90', 'PCA95', 'GA', 'ScoreOnly', 'NoScores').
        If None, fetches all data. Filters by complex name pattern that matches the methodology.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with data from database, formatted for the training pipeline.
    '''
    
    # Query complexes with methodology filter if provided
    if methodology:
        # Filter by complex name containing the methodology pattern
        # Methodology can be: NN, XGB, Trans, AE, PCA, PCA80, PCA85, PCA90, PCA95, GA, ScoreOnly, NoScores
        merged_data = session.query(Complexes.Complexes, Ligands.Ligands, Receptors.Receptors)\
            .join(Ligands.Ligands, Ligands.Ligands.id == Complexes.Complexes.ligand_id)\
            .join(Receptors.Receptors, Receptors.Receptors.id == Complexes.Complexes.receptor_id)\
            .filter(Complexes.Complexes.name.like(f'%{methodology}%'))\
            .all()
        
        # Convert to DataFrame format manually
        result = []
        for complex_obj, ligand, receptor in merged_data:
            merged_entry = {
                'name': complex_obj.name,
                **{key: value for key, value in complex_obj.__dict__.items() 
                   if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name', 'ligand_id', 'receptor_id']},
                **{key: value for key, value in ligand.__dict__.items() 
                   if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name']},
                **{key: value for key, value in receptor.__dict__.items() 
                   if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name']},
                'receptor': receptor.name,
                'ligand': ligand.name.split('_')[-1] if ligand.name else None
            }
            result.append(merged_entry)
        
        # Get column order
        complex_columns = [c.name for c in Complexes.Complexes.__table__.columns 
                          if c.name not in ['created_at', 'modified_at', 'id', 'name', 'ligand_id', 'receptor_id']]
        ligand_columns = [c.name for c in Ligands.Ligands.__table__.columns 
                         if c.name not in ['created_at', 'modified_at', 'id', 'name']]
        receptor_columns = [c.name for c in Receptors.Receptors.__table__.columns 
                           if c.name not in ['created_at', 'modified_at', 'id', 'name']]
        
        column_order = ['name'] + complex_columns + receptor_columns + ligand_columns + ['receptor', 'ligand']
        result = [{col: entry.get(col, None) for col in column_order} for entry in result]
        
        # Drop rows with missing values
        result = [entry for entry in result if all(value is not None for value in entry.values())]
        
        df = pd.DataFrame(result)
    else:
        # Export all data from database (OCSCORE can be empty - it's the target we predict)
        print("Exporting data from database...")
        result = export_db_to_csv(
            session=session,
            output_format='dataframe',
            drop_na=False  # Don't drop NA - OCSCORE will be empty and that's expected
        )
        
        # Ensure we have a DataFrame
        if result is None:
            raise ValueError("Failed to export data from database")
        
        df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
        
        if df.empty:
            raise ValueError("Database is empty. No complexes found in the database.")
        
        print(f"Exported {len(df)} rows from database")
        
        # Drop OCSCORE column if it exists and is all NaN (it's the target, not a feature)
        if 'OCSCORE' in df.columns:
            ocscore_na_count = df['OCSCORE'].isna().sum()
            if ocscore_na_count == len(df):
                print("Dropping OCSCORE column (all NaN - this is the target variable, not a feature)")
                df = df.drop(columns=['OCSCORE'])
            else:
                print(f"Warning: OCSCORE column has {len(df) - ocscore_na_count} non-NaN values. This should typically be empty for training data.")
    
    # Check if 'name' column exists
    if 'name' not in df.columns:
        print(f"Warning: 'name' column not found in DataFrame. Available columns: {list(df.columns)}")
        # Try to find an alternative column name
        name_col = None
        for col in ['name', 'complex_name', 'Name', 'complexName']:
            if col in df.columns:
                name_col = col
                break
        
        if name_col is None:
            raise ValueError(f"Could not find 'name' column in DataFrame. Available columns: {list(df.columns)}")
        
        # Rename the column to 'name' for consistency
        df = df.rename(columns={name_col: 'name'})
    
    # Add 'db' column based on name pattern (if not already present)
    if 'db' not in df.columns:
        # Check if name ends with 'ligand' for PDBbind, otherwise DUDEz
        df['db'] = df['name'].apply(
            lambda x: 'PDBBIND' if str(x).endswith('ligand') else 'DUDEZ'
        )
    
    # Add 'type' column for DUDEz data (ligand vs decoy)
    if 'type' not in df.columns:
        df['type'] = df['name'].apply(
            lambda x: 'ligand' if 'ligand' in str(x).lower() else 'decoy'
        )
        # For PDBbind, set type to 'ligand'
        df.loc[df['db'] == 'PDBBIND', 'type'] = 'ligand'
    
    # Add 'experimental' column
    # For PDBbind: use experimental binding affinity (if available) or a placeholder
    # For DUDEz: use type (1 for ligand, 0 for decoy)
    if 'experimental' not in df.columns:
        df['experimental'] = df.apply(
            lambda row: 1.0 if row['type'] == 'ligand' else 0.0,
            axis=1
        )
        # If there's a binding affinity column, use it for PDBbind
        # You may need to adjust this based on your actual database schema
        if 'binding_affinity' in df.columns:
            df.loc[df['db'] == 'PDBBIND', 'experimental'] = df.loc[df['db'] == 'PDBBIND', 'binding_affinity']
    
    # Always return DataFrame (we'll handle CSV saving separately if needed)
    return df


def get_best_nn_trial_from_study(
    study_name: str,
    storage: str,
    selection_metric: str = "combined"
) -> dict[str, Any]:
    '''Return best complete NN trial from one study.

    Parameters
    ----------
    study_name : str
        Optuna NN study name.
    storage : str
        Optuna storage URL.
    selection_metric : str, optional
        One of ``combined``, ``rmse``, ``auc``.
        - ``combined``: minimize RMSE - AUC
        - ``rmse``: minimize RMSE
        - ``auc``: maximize AUC
    '''

    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    study_df = study.trials_dataframe()
    study_df = study_df[study_df["state"] == "COMPLETE"].copy()
    if study_df.empty:
        raise ValueError(f"No complete trials found in {study_name}")

    if "value" not in study_df.columns:
        raise ValueError(f"Study {study_name} does not contain trial value (RMSE)")
    study_df = study_df.dropna(subset=["value"]).copy()
    if study_df.empty:
        raise ValueError(f"Study {study_name} has no complete trials with RMSE")

    if "user_attrs_AUC" in study_df.columns:
        study_df["user_attrs_AUC"] = pd.to_numeric(study_df["user_attrs_AUC"], errors="coerce")
    else:
        study_df["user_attrs_AUC"] = np.nan

    study_df["combined_metric"] = study_df["value"] - study_df["user_attrs_AUC"]

    metric = str(selection_metric).lower().strip()
    if metric == "rmse":
        best_row = study_df.sort_values(
            by=["value", "user_attrs_AUC"],
            ascending=[True, False]
        ).iloc[0]
    elif metric == "auc":
        auc_df = study_df.dropna(subset=["user_attrs_AUC"]).copy()
        if auc_df.empty:
            raise ValueError(f"Study {study_name} has no complete trials with AUC")
        best_row = auc_df.sort_values(
            by=["user_attrs_AUC", "value"],
            ascending=[False, True]
        ).iloc[0]
    else:
        combined_df = study_df.dropna(subset=["user_attrs_AUC"]).copy()
        if combined_df.empty:
            raise ValueError(f"Study {study_name} has no complete trials with AUC required for combined metric")
        best_row = combined_df.sort_values(
            by=["combined_metric", "value", "user_attrs_AUC"],
            ascending=[True, True, False]
        ).iloc[0]
    best_trial = study.trials[int(best_row["number"])]

    return {
        "study_name": study_name,
        "trial": best_trial,
        "rmse": float(best_row["value"]),
        "auc": float(best_row["user_attrs_AUC"]),
        "combined_metric": float(best_row["combined_metric"])
    }


def get_best_ao_trial_from_study(study_name: str, storage: str) -> dict[str, Any]:
    '''Return best complete AO trial from one study using RMSE then val RMSE.'''

    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    study_df = study.trials_dataframe()
    study_df = study_df[study_df["state"] == "COMPLETE"].copy()
    if study_df.empty:
        raise ValueError(f"No complete trials found in {study_name}")

    study_df = study_df.dropna(subset=["value"]).copy()
    if study_df.empty:
        raise ValueError(f"Study {study_name} has no complete trials with RMSE")

    if "user_attrs_val_rmse" in study_df.columns:
        study_df["user_attrs_val_rmse"] = pd.to_numeric(study_df["user_attrs_val_rmse"], errors="coerce").fillna(np.inf)
    else:
        study_df["user_attrs_val_rmse"] = np.inf

    best_row = study_df.sort_values(
        by=["value", "user_attrs_val_rmse"],
        ascending=[True, True]
    ).iloc[0]
    best_trial = study.trials[int(best_row["number"])]

    return {
        "study_name": study_name,
        "trial": best_trial,
        "rmse": float(best_row["value"]),
        "val_rmse": float(best_row["user_attrs_val_rmse"])
    }


def evaluate_trained_dnn_model(model: Any, data: dict) -> dict[str, float]:
    '''Evaluate trained DNN using RMSE on X_test and AUC on X_val.'''

    import torch
    from sklearn.metrics import roc_auc_score

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(np.asarray(data["X_test"]), dtype=torch.float32, device=device)
        test_pred = model(X_test_tensor).detach().cpu().numpy().reshape(-1)

    y_test_np = np.asarray(data["y_test"]).reshape(-1)
    rmse = float(np.sqrt(np.mean((test_pred - y_test_np) ** 2)))

    auc = float("nan")
    auc_adjusted = float("nan")
    if data.get("X_val") is not None and data.get("y_val") is not None:
        with torch.no_grad():
            X_val_tensor = torch.tensor(np.asarray(data["X_val"]), dtype=torch.float32, device=device)
            val_pred = model(X_val_tensor).detach().cpu().numpy().reshape(-1)
        y_val_np = np.asarray(data["y_val"]).reshape(-1)
        unique_classes = np.unique(y_val_np[~np.isnan(y_val_np)])
        if unique_classes.size >= 2:
            auc = float(roc_auc_score(y_val_np, val_pred))
            auc_adjusted = float(max(auc, 1.0 - auc))

    return {"rmse": rmse, "auc": auc, "auc_adjusted": auc_adjusted}


def _json_sanitize(value: Any) -> Any:
    '''Convert nested values to JSON-serializable builtin types.'''

    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    '''Compute SHA256 checksum for a file path.'''

    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_head_info(repo_dir: str) -> dict[str, Optional[str]]:
    '''Collect git HEAD and dirty status (best-effort).'''

    head = None
    dirty = None
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        dirty_state = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        dirty = "yes" if dirty_state else "no"
    except Exception:
        pass
    return {"head": head, "dirty": dirty}


def main():
    parser = argparse.ArgumentParser(description='Train model from database or CSV file')
    parser.add_argument('--from_db', action='store_true',
                        help='Read data from database instead of CSV file')
    parser.add_argument('--methodology', type=str, default=None,
                        help='Filter by methodology (e.g., NN, XGB, Trans, AE, PCA, PCA80, PCA85, PCA90, PCA95, GA, ScoreOnly, NoScores). Only used with --from_db')
    parser.add_argument('--df_path', type=str, default=None,
                        help='Path to CSV file (required if --from_db is not set)')
    parser.add_argument('--model_type', type=str, choices=['XGB', 'DNN'], default='XGB',
                        help='Type of model to train (XGB or DNN)')
    parser.add_argument('--model_name', type=str, default='OCScore',
                        help='Name for the saved model and mask (default: OCScore)')
    parser.add_argument('--storage_id', type=int, default=1,
                        help='Storage ID for the study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (default: fetched from config or model type)')
    parser.add_argument('--studies', type=str, nargs='+', default=None,
                        help='List of Optuna study names to use (e.g., "NN_Optimization_6" "NN_Optimization_7"). Default: AE with NN studies (6-10)')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA preprocessing')
    parser.add_argument('--pca_model', type=str, default="",
                        help='Path to PCA model file')
    parser.add_argument('--pca_type', type=int, default=95,
                        help='PCA variance percentage')
    parser.add_argument('--no_scores', action='store_true',
                        help='Exclude score columns')
    parser.add_argument('--only_scores', action='store_true',
                        help='Use only score columns')
    parser.add_argument('--use_pdb_train', action='store_true', default=True,
                        help='Use PDBbind for training')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--invert_conditionally', action='store_true', default=True,
                        help='Invert score values conditionally (default: True)')
    parser.add_argument('--no_invert', action='store_true',
                        help='Disable conditional inversion of scores')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize data (default: True)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable data normalization')
    parser.add_argument('--scaler', type=str, choices=['standard', 'minmax'], default='standard',
                        help='Scaler type for normalization (default: standard)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving models and masks. If not provided, uses default OCScore_models directory.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--pair_by_major', action='store_true',
                        help='For each major N, pair best AO_Optimization_N + NN_Optimization_N, train one model per N, and keep the best performer.')
    parser.add_argument('--major_numbers', type=int, nargs='+', default=[6, 7, 8, 9, 10],
                        help='Major IDs used by --pair_by_major (default: 6 7 8 9 10)')
    parser.add_argument('--pair_select_by', type=str, choices=['combined', 'rmse', 'auc'], default='combined',
                        help='Selection metric for --pair_by_major: combined=min(RMSE-AUC), rmse=min(RMSE), auc=max(AUC).')
    
    args = parser.parse_args()
    
    # Get default studies if not provided (AE with NN: studies 6-10)
    if args.studies is None:
        try:
            from OCDocker.OCScore.Analysis.legacy.PerformanceEvaluation import get_all_lists
            _, ao_nn_list_len, _ = get_all_lists()
            # AE with NN studies are NN_Optimization_6 through NN_Optimization_10
            args.studies = [f"NN_Optimization_{i}" for i in range(6, 6 + ao_nn_list_len)]
            print(f"Using default AE with NN studies: {args.studies}")
        except ImportError:
            # Fallback if function not available
            args.studies = [f"NN_Optimization_{i}" for i in range(6, 11)]
            print(f"Using fallback AE with NN studies: {args.studies}")
    
    # Get default storage if not provided - read from config after bootstrap
    if args.storage is None:
        # We'll set this after bootstrap when config is available
        args.storage = None  # Will be set after bootstrap
    
    print(f"Training with studies: {args.studies}")

    # Shared preprocessing flags (applies to both DB and CSV paths)
    invert_conditionally = args.invert_conditionally and not args.no_invert
    normalize = args.normalize and not args.no_normalize
    
    # Check if we need database or CSV
    if args.from_db:
        # Initialize OCDocker (this sets up database connection)
        # Pass a namespace to bootstrap to avoid parsing sys.argv (we handle our own args)
        print("Initializing OCDocker...")
        import OCDocker.Error as ocerror
        bootstrap_ns = argparse.Namespace(
            multiprocess=True,
            update=False,
            config_file=os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg'),
            output_level=ocerror.ReportLevel.WARNING,
            overwrite=False
        )
        init.bootstrap(bootstrap_ns)
        
        # Get database session
        if not hasattr(init, 'session') or init.session is None:
            print("ERROR: Database session not available. Please ensure database is configured.")
            sys.exit(1)
        
        # Get storage from config if not provided
        if args.storage is None:
            from OCDocker.Config import get_config
            config = get_config()
            args.storage = _build_storage_url_from_config(config, args.model_type)
            print(f"Using storage from config: {mask_password_in_url(args.storage)}")
        
        # Prepare data from database
        with init.session() as session:
            data = prepare_data_from_db(
                session=session,
                storage_id=args.storage_id,
                optimization_type=args.model_type,
                pca_model=args.pca_model,
                use_PCA=args.use_pca,
                pca_type=args.pca_type,
                no_scores=args.no_scores,
                only_scores=args.only_scores,
                use_pdb_train=args.use_pdb_train,
                random_seed=args.random_seed,
                invert_conditionally=invert_conditionally,
                normalize=normalize,
                scaler=args.scaler,
                methodology=args.methodology
            )
    else:
        # Read from CSV file
        if args.df_path is None:
            print("ERROR: Either --from_db or --df_path must be provided.")
            sys.exit(1)
        
        if not os.path.exists(args.df_path):
            print(f"ERROR: CSV file not found: {args.df_path}")
            sys.exit(1)
        
        # Get storage from config if not provided (for CSV mode, we still need storage for Optuna)
        if args.storage is None:
            # Try to bootstrap to get config (but don't require database connection)
            try:
                import OCDocker.Error as ocerror
                bootstrap_ns = argparse.Namespace(
                    multiprocess=True,
                    update=False,
                    config_file=os.getenv('OCDOCKER_CONFIG', 'OCDocker.cfg'),
                    output_level=ocerror.ReportLevel.WARNING,
                    overwrite=False
                )
                init.bootstrap(bootstrap_ns)
                from OCDocker.Config import get_config
                config = get_config()
                args.storage = _build_storage_url_from_config(config, args.model_type)
                print(f"Using storage from config: {mask_password_in_url(args.storage)}")
            except Exception as e:
                # Fallback to SQLite if config read fails
                print(f"Warning: Could not read config for storage, using SQLite fallback: {e}")
                args.storage = _sqlite_storage_for_model(args.model_type)
                print(f"Using fallback storage: {mask_password_in_url(args.storage)}")
        
        # Use existing load_data function for CSV
        # Note: load_data requires base_models_folder, so we use default (not used for anything)
        base_models_folder = ocscoreio.get_models_dir()
        data = ocscoredata.load_data(
            base_models_folder=base_models_folder,
            storage_id=args.storage_id,
            df_path=args.df_path,
            optimization_type=args.model_type,
            pca_model=args.pca_model,
            no_scores=args.no_scores,
            only_scores=args.only_scores,
            use_PCA=args.use_pca,
            pca_type=args.pca_type,
            use_pdb_train=args.use_pdb_train,
            random_seed=args.random_seed,
            invert_conditionally=invert_conditionally,
            normalize=normalize,
            scaler=args.scaler,
            enforce_reference_order=True
        )
    
    print(f"Data prepared: Train={data['X_train'].shape}, Test={data['X_test'].shape}")
    if data.get('X_val') is not None:
        print(f"Validation: {data['X_val'].shape}")

    # CSV pipeline compatibility: load_data() may create `<models_dir>/<model_type>_<storage_id>`
    # even though script 11 does not use that directory for outputs.
    # Remove it only when it's empty to avoid leaving clutter like `DNN_1`.
    if (not args.from_db) and isinstance(data, dict):
        csv_models_folder = data.get("models_folder")
        if isinstance(csv_models_folder, str):
            try:
                if os.path.isdir(csv_models_folder) and len(os.listdir(csv_models_folder)) == 0:
                    os.rmdir(csv_models_folder)
                    if args.verbose:
                        print(f"Removed unused empty folder: {csv_models_folder}")
            except OSError:
                # Keep silent if folder is non-empty or cannot be removed.
                pass

    selected_nn_trial_obj: Optional[Any] = None
    selected_ao_params: Optional[dict[str, Any]] = None
    selection_context: dict[str, Any] = {}
    final_eval_summary: Optional[dict[str, Any]] = None

    pairwise_summary_df = None
    if args.pair_by_major:
        if args.model_type != 'DNN':
            print("ERROR: --pair_by_major is only supported with --model_type DNN.")
            sys.exit(1)

        major_numbers = sorted(set(args.major_numbers))
        if len(major_numbers) == 0:
            print("ERROR: No major numbers provided for --pair_by_major.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("PAIRWISE MAJOR SELECTION (AO_N + NN_N)")
        print(f"{'='*60}")
        print(f"Majors: {major_numbers}")
        print("Selecting the best major from study metrics, then training only that pair.")
        print(f"{'='*60}\n")

        pairwise_candidates: list[dict[str, Any]] = []
        for major in major_numbers:
            nn_study_name = f"NN_Optimization_{major}"
            ao_study_name = f"AO_Optimization_{major}"

            print(f"\n{'-'*60}")
            print(f"Major {major}: selecting best AO + best NN")
            print(f"  AO study: {ao_study_name}")
            print(f"  NN study: {nn_study_name}")
            print(f"{'-'*60}")

            try:
                nn_best = get_best_nn_trial_from_study(
                    nn_study_name,
                    args.storage,
                    selection_metric=args.pair_select_by
                )
                ao_best = get_best_ao_trial_from_study(ao_study_name, args.storage)
            except Exception as e:
                print(f"  Warning: skipping major {major} due to study loading error: {e}")
                continue

            print(
                f"  Selected NN trial {nn_best['trial'].number}: "
                f"RMSE={nn_best['rmse']:.4f}, AUC={nn_best['auc']:.4f}, Combined={nn_best['combined_metric']:.4f}"
            )
            print(
                f"  Selected AO trial {ao_best['trial'].number}: "
                f"RMSE={ao_best['rmse']:.4f}, Val_RMSE={ao_best['val_rmse']:.4f}"
            )

            pairwise_candidates.append({
                "major": major,
                "nn_study": nn_study_name,
                "nn_trial": int(nn_best["trial"].number),
                "ao_study": ao_study_name,
                "ao_trial": int(ao_best["trial"].number),
                "nn_optuna_rmse": float(nn_best["rmse"]),
                "nn_optuna_auc": float(nn_best["auc"]),
                "nn_optuna_combined_metric": float(nn_best["combined_metric"]),
                "ao_optuna_rmse": float(ao_best["rmse"]),
                "ao_optuna_val_rmse": float(ao_best["val_rmse"]),
                "nn_trial_obj": nn_best["trial"],
                "ao_params": ao_best["trial"].params
            })

        if len(pairwise_candidates) == 0:
            print("ERROR: No pairwise study candidates were found.")
            sys.exit(1)

        if args.pair_select_by == "rmse":
            sort_key = lambda x: (x["nn_optuna_rmse"], x["ao_optuna_rmse"], x["ao_optuna_val_rmse"])
            sort_label = "NN Optuna RMSE (ascending)"
        elif args.pair_select_by == "auc":
            sort_key = lambda x: (-x["nn_optuna_auc"], x["nn_optuna_rmse"], x["nn_optuna_combined_metric"])
            sort_label = "NN Optuna AUC (descending)"
        else:
            sort_key = lambda x: (x["nn_optuna_combined_metric"], x["nn_optuna_rmse"], -x["nn_optuna_auc"])
            sort_label = "NN Optuna RMSE - AUC (ascending)"

        pairwise_candidates = sorted(pairwise_candidates, key=sort_key)
        best_pair = pairwise_candidates[0]
        selected_nn_trial_obj = best_pair["nn_trial_obj"]
        selected_ao_params = best_pair["ao_params"]
        selection_context = {
            "mode": "pair_by_major",
            "pair_select_by": args.pair_select_by,
            "selected_major": int(best_pair["major"]),
            "selected_nn_study": best_pair["nn_study"],
            "selected_nn_trial": int(best_pair["nn_trial"]),
            "selected_ao_study": best_pair["ao_study"],
            "selected_ao_trial": int(best_pair["ao_trial"]),
            "selected_optuna_metrics": {
                "nn_optuna_rmse": float(best_pair["nn_optuna_rmse"]),
                "nn_optuna_auc": float(best_pair["nn_optuna_auc"]),
                "nn_optuna_combined_metric": float(best_pair["nn_optuna_combined_metric"]),
                "ao_optuna_rmse": float(best_pair["ao_optuna_rmse"]),
                "ao_optuna_val_rmse": float(best_pair["ao_optuna_val_rmse"])
            }
        }

        display_cols = [
            "major", "nn_trial", "ao_trial",
            "nn_optuna_rmse", "nn_optuna_auc", "nn_optuna_combined_metric",
            "ao_optuna_rmse", "ao_optuna_val_rmse"
        ]
        print(f"\n{'='*60}")
        print(f"PAIRWISE CANDIDATES (sorted by {sort_label})")
        print(f"{'='*60}")
        print(
            pd.DataFrame(
                [{k: v for k, v in row.items() if k in display_cols} for row in pairwise_candidates]
            )[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print(f"{'='*60}")
        print(f"Selected major: {best_pair['major']}")
        print(f"Selected NN study/trial: {best_pair['nn_study']} / {best_pair['nn_trial']}")
        print(f"Selected AO study/trial: {best_pair['ao_study']} / {best_pair['ao_trial']}")
        print(f"Selected NN Optuna RMSE: {best_pair['nn_optuna_rmse']:.4f}")
        print(f"Selected NN Optuna AUC: {best_pair['nn_optuna_auc']:.4f}")
        print(f"Selected NN Optuna Combined: {best_pair['nn_optuna_combined_metric']:.4f}")
        print(f"Selection criterion: {args.pair_select_by}")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print("TRAINING SELECTED BEST PAIR")
        print(f"{'='*60}")
        model, mask = train_dnn_model(
            data=data,
            storage=args.storage,
            storage_id=int(best_pair["major"]),
            model_name=args.model_name,
            optimization_type=args.model_type,
            use_PCA=args.use_pca,
            pca_type=args.pca_type,
            no_scores=args.no_scores,
            only_scores=args.only_scores,
            study_name=str(best_pair["nn_study"]),
            use_gpu=args.use_gpu,
            verbose=args.verbose,
            best_trial=best_pair["nn_trial_obj"],
            encoder_params_override=best_pair["ao_params"],
            random_seed=args.random_seed
        )

        best_eval = evaluate_trained_dnn_model(model, data)
        best_eval_auc = best_eval["auc"]
        best_eval_combined = best_eval["rmse"] - best_eval_auc if not np.isnan(best_eval_auc) else float("inf")
        print(
            f"Selected pair evaluation: RMSE={best_eval['rmse']:.4f}, "
            f"AUC={best_eval_auc:.4f}, AUC_adj={best_eval['auc_adjusted']:.4f}, "
            f"Combined={best_eval_combined:.4f}"
        )
        final_eval_summary = {
            "rmse": float(best_eval["rmse"]),
            "auc": float(best_eval_auc),
            "auc_adjusted": float(best_eval["auc_adjusted"]),
            "combined_metric": float(best_eval_combined)
        }

        pairwise_summary_rows = []
        for row in pairwise_candidates:
            summary_row = {k: v for k, v in row.items() if k not in {"nn_trial_obj", "ao_params"}}
            selected = int(row["major"] == best_pair["major"])
            summary_row["selected"] = selected
            summary_row["selection_criterion"] = args.pair_select_by
            if selected:
                summary_row["eval_rmse"] = float(best_eval["rmse"])
                summary_row["eval_auc"] = float(best_eval_auc)
                summary_row["eval_auc_adjusted"] = float(best_eval["auc_adjusted"])
                summary_row["eval_combined_metric"] = float(best_eval_combined)
            else:
                summary_row["eval_rmse"] = np.nan
                summary_row["eval_auc"] = np.nan
                summary_row["eval_auc_adjusted"] = np.nan
                summary_row["eval_combined_metric"] = np.nan
            pairwise_summary_rows.append(summary_row)
        pairwise_summary_df = pd.DataFrame(pairwise_summary_rows)
    else:
        # Find the best trial across ALL studies
        print(f"\n{'='*60}")
        print(f"FINDING BEST TRIAL ACROSS ALL STUDIES")
        print(f"{'='*60}")

        import optuna
        import re

        all_trials = []  # List of (study_name, trial, combined_metric)

        for study_name in args.studies:
            print(f"Loading study: {study_name}...")
            try:
                study = optuna.load_study(study_name=study_name, storage=args.storage)
                study_df = study.trials_dataframe()
                study_df = study_df[study_df['state'] == 'COMPLETE']

                if len(study_df) == 0:
                    print(f"  Warning: No complete trials in {study_name}, skipping...")
                    continue

                # Compute combined metric (RMSE - AUC) for all trials
                study_df['combined_metric'] = study_df['value'] - study_df['user_attrs_AUC']

                # Add all trials from this study
                for _, row in study_df.iterrows():
                    trial = study.trials[row['number']]
                    all_trials.append({
                        'study_name': study_name,
                        'trial': trial,
                        'combined_metric': row['combined_metric'],
                        'rmse': row['value'],
                        'auc': row['user_attrs_AUC']
                    })

                print(f"  Found {len(study_df)} complete trials")
            except Exception as e:
                print(f"  Error loading {study_name}: {e}")
                continue

        if len(all_trials) == 0:
            print("ERROR: No trials found in any study!")
            sys.exit(1)

        # Find the best trial (lowest combined_metric = RMSE - AUC)
        best_trial_info = min(all_trials, key=lambda x: x['combined_metric'])
        best_study_name = best_trial_info['study_name']
        best_trial = best_trial_info['trial']
        selected_nn_trial_obj = best_trial
        selection_context = {
            "mode": "global_best_trial",
            "selected_study": best_study_name,
            "selected_trial": int(best_trial.number),
            "selected_optuna_metrics": {
                "rmse": float(best_trial_info["rmse"]),
                "auc": float(best_trial_info["auc"]),
                "combined_metric": float(best_trial_info["combined_metric"])
            }
        }

        print(f"\n{'='*60}")
        print(f"BEST TRIAL FOUND")
        print(f"{'='*60}")
        print(f"Study: {best_study_name}")
        print(f"Trial number: {best_trial.number}")
        print(f"RMSE: {best_trial_info['rmse']:.4f}")
        print(f"AUC: {best_trial_info['auc']:.4f}")
        print(f"Combined metric (RMSE - AUC): {best_trial_info['combined_metric']:.4f}")
        print(f"Total trials evaluated: {len(all_trials)}")
        print(f"{'='*60}\n")

        # Extract storage_id from best study name
        match = re.search(r'_(\d+)$', best_study_name)
        if match:
            study_storage_id = int(match.group(1))
        else:
            study_storage_id = args.storage_id  # Fallback to provided storage_id

        # Train only ONE model using the best trial
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL WITH BEST PIPELINE")
        print(f"{'='*60}")
        print(f"Using best trial from: {best_study_name}")
        print(f"{'='*60}\n")

        if args.model_type == 'XGB':
            model, mask = train_xgboost_model(
                data=data,
                storage=args.storage,
                storage_id=study_storage_id,
                model_name=args.model_name,  # Use base name, not study-specific
                optimization_type=args.model_type,
                use_PCA=args.use_pca,
                pca_type=args.pca_type,
                no_scores=args.no_scores,
                only_scores=args.only_scores,
                study_name=best_study_name,  # Pass best study name for context
                use_gpu=args.use_gpu,
                verbose=args.verbose
            )
        else:  # DNN
            # Extract AO study numbers from NN study names (for finding best AE)
            # Extract numbers from study names like "NN_Optimization_6" -> [6, 7, 8, 9, 10]
            ao_study_numbers = []
            for study_name in args.studies:
                match = re.search(r'_(\d+)$', study_name)
                if match:
                    study_num = int(match.group(1))
                    # Only include if it's in the AE with NN range (typically 6-10, but be flexible)
                    if study_num >= 6:
                        ao_study_numbers.append(study_num)

            # Remove duplicates and sort
            ao_study_numbers = sorted(list(set(ao_study_numbers))) if ao_study_numbers else None

            model, mask = train_dnn_model(
                data=data,
                storage=args.storage,
                storage_id=study_storage_id,
                model_name=args.model_name,  # Use base name, not study-specific
                optimization_type=args.model_type,
                use_PCA=args.use_pca,
                pca_type=args.pca_type,
                no_scores=args.no_scores,
                only_scores=args.only_scores,
                study_name=best_study_name,  # Pass best study name for context
                use_gpu=args.use_gpu,
                verbose=args.verbose,
                best_trial=best_trial,  # Pass the best trial found across all studies
                ao_study_numbers=ao_study_numbers,  # Pass AO study numbers to search for best AE
                random_seed=args.random_seed
            )
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    
    # Save model and mask
    # Use custom output directory if provided, otherwise use default
    if args.output_dir:
        models_dir = os.path.abspath(args.output_dir)
        # Create directory if it doesn't exist
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            print(f"Created output directory: {models_dir}")
    else:
        models_dir = ocscoreio.get_models_dir()
    
    print(f"Saving model and mask to: {models_dir}")
    
    # Determine file extension based on model type
    # PyTorch models should use .pt, others use .pkl
    import torch
    if isinstance(model, torch.nn.Module):
        model_ext = ".pt"
        model_path = os.path.join(models_dir, f"{args.model_name}{model_ext}")
        # Use IO module to save PyTorch model
        ocscoreio.save_object(model, model_path, serialization_method="torch")
        print(f"Saved PyTorch model to: {model_path}")
    else:
        # XGBoost or other sklearn-style models
        model_ext = ".pkl"
        model_path = os.path.join(models_dir, f"{args.model_name}{model_ext}")
        # Use IO module to save model
        ocscoreio.save_object(model, model_path, serialization_method="joblib")
        print(f"Saved model (joblib) to: {model_path}")
    
    mask_path = ocscoreio.save_mask(mask, name=args.model_name, models_dir=models_dir)
    
    # Save scaler if normalization was used
    scaler_path = None
    if data.get('scaler') is not None:
        scaler_path = os.path.join(models_dir, f"{args.model_name}_scaler.pkl")
        ocscoreio.save_object(data['scaler'], scaler_path, serialization_method="joblib")
        print(f"Saved scaler to: {scaler_path}")

    pairwise_summary_path = None
    if pairwise_summary_df is not None:
        pairwise_summary_path = os.path.join(models_dir, f"{args.model_name}_pairwise_major_results.csv")
        pairwise_summary_df.to_csv(pairwise_summary_path, index=False)
        print(f"Saved pairwise major summary to: {pairwise_summary_path}")

    # Save reproducibility/statistics artifact for this training run.
    stats_path = os.path.join(models_dir, f"{args.model_name}_run_stats.json")
    input_data_info: dict[str, Any] = {"from_db": bool(args.from_db)}
    if args.df_path is not None:
        input_path = os.path.abspath(args.df_path)
        input_data_info["df_path"] = input_path
        if os.path.isfile(input_path):
            try:
                st = os.stat(input_path)
                input_data_info["df_size_bytes"] = int(st.st_size)
                input_data_info["df_mtime_epoch"] = float(st.st_mtime)
                input_data_info["df_sha256"] = _sha256_file(input_path)
            except OSError:
                pass

    selected_trial_info = None
    if selected_nn_trial_obj is not None:
        trial_value = selected_nn_trial_obj.value
        selected_trial_info = {
            "number": int(selected_nn_trial_obj.number),
            "value": None if trial_value is None else float(trial_value),
            "params": _json_sanitize(dict(selected_nn_trial_obj.params)),
            "user_attrs": _json_sanitize(dict(selected_nn_trial_obj.user_attrs))
        }

    pairwise_summary_records = None
    if pairwise_summary_df is not None:
        pairwise_summary_records = _json_sanitize(
            pairwise_summary_df.astype(object).where(pd.notna(pairwise_summary_df), None).to_dict(orient="records")
        )

    git_info = _git_head_info(_parent_dir)
    torch_env = {}
    try:
        import torch as _torch
        torch_env = {
            "torch_version": _torch.__version__,
            "cuda_available": bool(_torch.cuda.is_available()),
            "cuda_device_count": int(_torch.cuda.device_count()) if _torch.cuda.is_available() else 0,
            "cudnn_benchmark": bool(getattr(_torch.backends.cudnn, "benchmark", False)),
            "cudnn_deterministic": bool(getattr(_torch.backends.cudnn, "deterministic", False))
        }
    except Exception:
        torch_env = {}

    run_stats = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": os.path.abspath(__file__),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "args": _json_sanitize(vars(args)),
        "storage_url": args.storage,
        "storage_url_masked": mask_password_in_url(args.storage) if args.storage else None,
        "input_data": _json_sanitize(input_data_info),
        "data_shapes": {
            "X_train": list(data["X_train"].shape),
            "X_test": list(data["X_test"].shape),
            "X_val": None if data.get("X_val") is None else list(data["X_val"].shape),
            "y_train": int(len(data["y_train"])),
            "y_test": int(len(data["y_test"])),
            "y_val": None if data.get("y_val") is None else int(len(data["y_val"]))
        },
        "selection_context": _json_sanitize(selection_context),
        "selected_nn_trial": _json_sanitize(selected_trial_info),
        "selected_ao_params": _json_sanitize(selected_ao_params),
        "final_eval": _json_sanitize(final_eval_summary),
        "pairwise_summary_records": pairwise_summary_records,
        "artifacts": {
            "model_path": model_path,
            "mask_path": mask_path,
            "scaler_path": scaler_path,
            "pairwise_summary_path": pairwise_summary_path
        },
        "mask_info": {
            "shape": list(mask.shape) if hasattr(mask, "shape") else None,
            "active_features": int(np.sum(mask)) if mask is not None else None
        },
        "git": git_info,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            **torch_env
        }
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(run_stats), f, indent=2, sort_keys=True)
    print(f"Saved run statistics to: {stats_path}")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Mask saved to: {mask_path}")
    if scaler_path:
        print(f"Scaler saved to: {scaler_path}")
    if pairwise_summary_path:
        print(f"Pairwise summary saved to: {pairwise_summary_path}")
    print(f"Run stats saved to: {stats_path}")
    print(f"\nTo use this model:")
    print(f"  import OCDocker.OCScore.Scoring as ocscoring")
    print(f"  import OCDocker.OCScore.Utils.IO as ocscoreio")
    print(f"  import pandas as pd")
    print(f"  ")
    print(f"  # Load your data (DataFrame, CSV file path, or None for database)")
    print(f"  # Option 1: From CSV file")
    print(f"  your_data = pd.read_csv('your_data.csv')")
    print(f"  # Option 2: From DataFrame (already loaded)")
    print(f"  # your_data = your_dataframe")
    print(f"  # Option 3: From database (set data=None)")
    print(f"  ")
    print(f"  # Load mask (use models_dir if using custom directory)")
    if args.output_dir:
        print(f"  mask = ocscoreio.load_mask('{args.model_name}', models_dir='{models_dir}')")
    else:
        print(f"  mask = ocscoreio.load_mask('{args.model_name}')  # Uses default directory")
        print(f"  # Or with custom directory:")
        print(f"  # mask = ocscoreio.load_mask('{args.model_name}', models_dir='/path/to/custom/dir')")
    print(f"  ")
    print(f"  # Get predictions")
    if scaler_path:
        print(f"  scores = ocscoring.get_score(")
        print(f"      model_path='{model_path}',")
        print(f"      data=your_data,  # DataFrame, CSV file path (str), or None for database")
        print(f"      mask=mask,")
        print(f"      scaler_path='{scaler_path}'  # IMPORTANT: Use the saved scaler")
        print(f"  )")
    else:
        print(f"  scores = ocscoring.get_score(")
        print(f"      model_path='{model_path}',")
        print(f"      data=your_data,  # DataFrame, CSV file path (str), or None for database")
        print(f"      mask=mask")
        print(f"  )")


def mask_password_in_url(url: str) -> str:
    '''Mask password in a database URL for secure printing.
    
    Parameters
    ----------
    url : str
        Database URL (e.g., postgresql+psycopg://user:<db_password>@host:port/db)
    
    Returns
    -------
    str
        URL with password masked (e.g., postgresql+psycopg://user:***@host:port/db)
    '''
    
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with ***
            masked_netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=masked_netloc))
        return url
    except Exception:
        # If parsing fails, just return the URL (might be SQLite path)
        return url


def prepare_data_from_db(
        session,
        storage_id: int,
        optimization_type: str = "XGB",
        pca_model: str = "",
        use_PCA: bool = False,
        pca_type: int = 95,
        no_scores: bool = False,
        only_scores: bool = False,
        use_pdb_train: bool = True,
        random_seed: int = 42,
        invert_conditionally: bool = True,
        normalize: bool = True,
        scaler: str = "standard",
        methodology: Optional[str] = None
    ) -> dict:
    '''
    Prepare data from database using the full preprocessing pipeline.
    
    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        Database session
    storage_id : int
        Storage ID for the study
    optimization_type : str
        Type of optimization (XGB, NN, etc.)
    pca_model : str
        Path to PCA model (if using PCA)
    use_PCA : bool
        Whether to use PCA
    pca_type : int
        PCA variance percentage
    no_scores : bool
        Whether to exclude score columns
    only_scores : bool
        Whether to use only score columns
    use_pdb_train : bool
        Whether to use PDBbind for training
    random_seed : int
        Random seed
    invert_conditionally : bool
        Whether to invert the conditionally
    normalize : bool
        Whether to normalize the data
    scaler : str
        The scaler to use
    methodology : str
        The methodology to use

    Returns
    -------
    dict
        Dictionary with preprocessed data (X_train, X_test, y_train, y_test, etc.)
    '''
    
    # Load data from database
    print(f"Loading data from database{f' (methodology: {methodology})' if methodology else ''}...")
    db_data = load_data_from_database(session, methodology=methodology)
    
    # Save to temporary CSV for compatibility with preprocess_df
    temp_csv_path = "/tmp/ocscore_db_data.csv"
    db_data.to_csv(temp_csv_path, index=False)
    
    print(f"Data loaded: {len(db_data)} rows")
    
    # Use preprocess_df exactly as in existing load_data function (mimics optimize_NN/optimize_XGB pattern)
    # Request scaler if normalization is enabled (needed for consistent scaling during prediction)
    if normalize:
        dudez_data, pdbbind_data, score_columns, fitted_scaler = ocscoredata.preprocess_df(
            file_name=temp_csv_path,
            score_columns_list=["SMINA", "VINA", "ODDT", "PLANTS"],  # Default score columns
            outliers_columns_list=None,  # No outlier removal by default
            scaler=scaler,
            invert_conditionally=invert_conditionally,
            normalize=normalize,
            return_scaler=True  # Get the fitted scaler to save it
        )
    else:
        dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(
            file_name=temp_csv_path,
            score_columns_list=["SMINA", "VINA", "ODDT", "PLANTS"],  # Default score columns
            outliers_columns_list=None,  # No outlier removal by default
            scaler=scaler,
            invert_conditionally=invert_conditionally,
            normalize=normalize
        )
        fitted_scaler = None
    
    # Handle score columns
    if no_scores:
        dudez_data = dudez_data.drop(columns=score_columns, errors='ignore')
        pdbbind_data = pdbbind_data.drop(columns=score_columns, errors='ignore')
        study_name = f"NoScores_{optimization_type}_Optimization"
    elif only_scores:
        metadata_cols = ["receptor", "ligand", "name", "type", "db"]
        columns_to_keep = [col for col in metadata_cols if col in dudez_data.columns] + score_columns
        dudez_data = ocscoredata.remove_other_columns(dudez_data, columns_to_keep, inplace=False)
        columns_to_keep = [col for col in metadata_cols + ["experimental"] if col in pdbbind_data.columns] + score_columns
        pdbbind_data = ocscoredata.remove_other_columns(pdbbind_data, columns_to_keep, inplace=False)
        study_name = f"ScoreOnly_{optimization_type}_Optimization"
    else:
        study_name = f"{optimization_type}_Optimization"
    
    # Apply PCA if needed
    if use_PCA and pca_model:
        columns_to_skip_pca = ["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns
        pdbbind_data = ocscoredata.apply_pca(
            pdbbind_data, 
            pca_model, 
            columns_to_skip_pca=columns_to_skip_pca, 
            inplace=False
        )
        if use_pdb_train:
            columns_to_skip_pca = ["receptor", "ligand", "name", "type", "db"] + score_columns
            dudez_data = ocscoredata.apply_pca(
                dudez_data, 
                pca_model, 
                columns_to_skip_pca=columns_to_skip_pca, 
                inplace=False
            )
        study_name = f"PCA{pca_type}_{study_name}"
    
    # CRITICAL: Reorder columns to match reference column order from config
    # This ensures the column order matches the training data order, which is essential
    # for proper mask application and model inference consistency.
    # Do this AFTER all transformations (score column handling, PCA, etc.) but BEFORE splitting
    print("Reordering columns to match reference column order from config...")
    dudez_data = ocscoredata.reorder_columns_to_match_data_order(
        dudez_data,
        data_source=None,  # Uses config.reference_column_order by default
        keep_extra_columns=True,  # Keep any extra columns (e.g., PCA components) that might exist
        fill_missing_columns=False  # Don't add missing columns as NaN
    )
    pdbbind_data = ocscoredata.reorder_columns_to_match_data_order(
        pdbbind_data,
        data_source=None,  # Uses config.reference_column_order by default
        keep_extra_columns=True,  # Keep any extra columns (e.g., PCA components) that might exist
        fill_missing_columns=False  # Don't add missing columns as NaN
    )
    
    # Split data
    if use_pdb_train:
        X_train_df = pdbbind_data.drop(
            columns=["receptor", "ligand", "name", "type", "db", "experimental"],
            errors="ignore"
        )
        X_train, X_test, y_train, y_test = ocscoredata.split_dataset(
            X_train_df,
            pdbbind_data["experimental"],
            test_size=0.25,
            random_state=random_seed
        )
        X_val_df = dudez_data.drop(
            columns=["receptor", "ligand", "name", "type", "db", "experimental"],
            errors="ignore"
        )
        X_val = X_val_df
        y_val = dudez_data["type"].map({"ligand": 1, "decoy": 0})
        # Get feature column names from DataFrame before converting to numpy
        feature_columns = X_train_df.columns.tolist()
    else:
        X_train_df = dudez_data.drop(
            columns=["receptor", "ligand", "name", "type", "db", "experimental"],
            errors="ignore"
        )
        X_train = X_train_df
        y_train = dudez_data["experimental"]
        X_test_df = dudez_data.drop(
            columns=["receptor", "ligand", "name", "type", "db", "experimental"],
            errors="ignore"
        )
        X_test = X_test_df
        y_test = dudez_data["type"].map({"ligand": 1, "decoy": 0})
        X_val = None
        y_val = None
        # Get feature column names from DataFrame before converting to numpy
        feature_columns = X_train_df.columns.tolist()
    
    # Convert to numpy arrays for compatibility with existing code
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val if X_val is not None else None
    
    # Add storage_id to study name to match the pattern: {prefix}_Optimization_{storage_id}
    study_name_with_id = f"{study_name}_{storage_id}"
    
    data = {
        "study_name": study_name_with_id,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "feature_columns": feature_columns,  # Add feature column names for mask application
        "scaler": fitted_scaler  # Add fitted scaler (None if normalization is disabled)
    }
    
    # Clean up temporary file if we created it
    if temp_csv_path and os.path.exists(temp_csv_path) and temp_csv_path.startswith('/tmp/'):
        try:
            os.remove(temp_csv_path)
        except:
            pass
    
    return data


def train_dnn_model(
        data: dict,
        storage: str,
        storage_id: int,
        model_name: str,
        optimization_type: str = "NN",
        use_PCA: bool = False,
        pca_type: int = 95,
        no_scores: bool = False,
        only_scores: bool = False,
        study_name: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        verbose: bool = True,
        best_trial: Optional[Any] = None,  # Optional: if provided, use this trial instead of loading from study
        ao_study_numbers: Optional[List[int]] = None,  # Optional: list of AO study numbers to search (e.g., [6, 7, 8, 9, 10])
        encoder_params_override: Optional[dict[str, Any]] = None,
        random_seed: int = 42
    ) -> tuple:
    '''
    Train a DNN model.

    Parameters
    ----------
    data : dict
        The data dictionary.
    storage : str
        The storage to use.
    storage_id : int
        The storage ID to use.
    model_name : str
        The name of the model.
    optimization_type : str
        The optimization type.
    use_PCA : bool
        If True, use PCA.
    pca_type : int
        The PCA type to use.
    no_scores : bool
        If True, don't use scores.
    only_scores : bool
        If True, only use scores.
    study_name : str
        The name of the study.
    mask : np.ndarray
        The mask to use.
    use_gpu : bool
        If True, use the GPU.
    verbose : bool
        If True, print the output.
    encoder_params_override : dict[str, Any], optional
        If provided, use these AO parameters directly instead of searching AO studies.
    random_seed : int, optional
        Random seed used for final DNN retraining.
    
    Returns
    -------
    tuple
        (model, mask) - trained model and feature mask
    '''
    
    print("Training DNN model...")
    
    # Extract study number from study name
    import re
    match = re.search(r'_(\d+)$', study_name)
    study_num = int(match.group(1)) if match else None
    
    # Fetch mask from ablation study and autoencoder params for AE with NN studies (6-10)
    encoder_params = encoder_params_override
    if mask is None:
        # Check if this is an AE with NN study (studies 6-10)
        if study_num and 6 <= study_num <= 10:
            import optuna
            if encoder_params is None:
                print(f"Fetching best autoencoder params across ALL AO studies...")

                # Find the best AE across ALL AO studies (not just the one matching the NN study number)
                # This ensures we use the globally best autoencoder, not just the one from the matching study
                # Use provided ao_study_numbers or default to [6, 7, 8, 9, 10] for AE with NN studies
                if ao_study_numbers is None:
                    ao_study_numbers = [6, 7, 8, 9, 10]  # Default: All AE with NN study numbers

                all_ao_trials = []

                for ao_num in ao_study_numbers:
                    ao_study_name = f"AO_Optimization_{ao_num}"
                    try:
                        ao_study = optuna.load_study(study_name=ao_study_name, storage=storage)
                        ao_df = ao_study.trials_dataframe()
                        ao_df = ao_df[ao_df['state'] == 'COMPLETE']

                        if len(ao_df) > 0:
                            # Sort by value (RMSE) and validation RMSE
                            ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])
                            best_ao_trial = ao_study.trials[ao_df.iloc[0].number]
                            all_ao_trials.append({
                                'study_name': ao_study_name,
                                'study_number': ao_num,
                                'trial': best_ao_trial,
                                'value': best_ao_trial.value,
                                'val_rmse': best_ao_trial.user_attrs.get('val_rmse', float('inf'))
                            })
                            if verbose:
                                print(f"  Found best trial in {ao_study_name}: RMSE={best_ao_trial.value:.4f}, Val_RMSE={best_ao_trial.user_attrs.get('val_rmse', 'N/A')}")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not load autoencoder study {ao_study_name}: {e}")

                # Find the best AE trial across all studies
                if len(all_ao_trials) > 0:
                    # Sort by value (RMSE) first, then by validation RMSE
                    best_ao_info = min(all_ao_trials, key=lambda x: (x['value'], x['val_rmse']))
                    encoder_params = best_ao_info['trial'].params
                    if verbose:
                        print(f"\nSelected best autoencoder from: {best_ao_info['study_name']}")
                        print(f"  Trial number: {best_ao_info['trial'].number}")
                        print(f"  RMSE: {best_ao_info['value']:.4f}")
                        print(f"  Val RMSE: {best_ao_info['val_rmse']:.4f}")
                        print(f"  (Selected from {len(all_ao_trials)} AO studies)")
                else:
                    if verbose:
                        print(f"Warning: No autoencoder studies found. Training without autoencoder.")
            elif verbose:
                print("Using provided autoencoder params (encoder_params_override).")
            
            # Load ablation study to get mask
            ablation_study_name = "NN_Ablation_Optimization_1"
            try:
                ablation_study = optuna.load_study(study_name=ablation_study_name, storage=storage)
                ablation_df = ablation_study.trials_dataframe()
                
                # Filter to only complete trials
                ablation_df = ablation_df[ablation_df['state'] == 'COMPLETE']
                ablation_df = ablation_df.reset_index(drop=True)
                
                # Rename columns for clarity
                ablation_df = ablation_df.rename(columns={
                    'value': 'RMSE',
                    'user_attrs_Feature_Mask': 'Feature_Mask',
                    'user_attrs_AUC': 'AUC'
                })
                
                # Compute score (RMSE - AUC) and get best
                ablation_df['score'] = ablation_df['RMSE'] - ablation_df['AUC']
                best_ablation_df = ablation_df.sort_values(by=['score'], ascending=[True])
                
                # Get mask from best ablation trial
                mask_str = best_ablation_df.iloc[0]['Feature_Mask']
                
                # Convert mask string to numpy array of 0s and 1s
                mask_from_study = np.array([int(x) for x in mask_str])
                
                # The ablation study creates masks for SFs (scoring functions) only
                # The mask stored is a full-length mask where only SF positions are modified
                # We need to identify SF columns in the current data and extract/apply the mask correctly
                
                # Get feature column names (if available)
                feature_columns = data.get('feature_columns')
                
                # Create full mask (all 1s)
                mask = np.ones(data['X_train'].shape[1], dtype=int)
                
                if len(mask_from_study) == 16:
                    # This is an SF-only mask (16 values for 16 SFs)
                    # We need to find SF column indices in current data and apply mask there
                    if feature_columns is not None:
                        # Identify SF columns (VINA*, SMINA*, ODDT*, PLANTS*)
                        import re
                        sf_pattern = re.compile(r'^(VINA|SMINA|ODDT|PLANTS)', re.IGNORECASE)
                        sf_indices = [i for i, col in enumerate(feature_columns) if sf_pattern.match(str(col))]
                        
                        if len(sf_indices) == 16:
                            # Apply the 16-element mask to SF positions
                            for idx, mask_val in zip(sf_indices, mask_from_study):
                                mask[idx] = mask_val
                            print(f"Applied 16-element SF mask to {len(sf_indices)} SF features.")
                        else:
                            print(f"Warning: Found {len(sf_indices)} SF features, but mask has 16 elements.")
                            print(f"  SF features found: {sf_indices[:10]}..." if len(sf_indices) > 10 else f"  SF features: {sf_indices}")
                            print(f"  Using default mask (all 1s).")
                    else:
                        print(f"Warning: Cannot identify SF positions - feature column names not available.")
                        print(f"  Using default mask (all 1s).")
                elif len(mask_from_study) == data['X_train'].shape[1]:
                    # This is a full-length mask that matches current data shape
                    # Extract only SF positions if we have feature names
                    if feature_columns is not None:
                        import re
                        sf_pattern = re.compile(r'^(VINA|SMINA|ODDT|PLANTS)', re.IGNORECASE)
                        sf_indices = [i for i, col in enumerate(feature_columns) if sf_pattern.match(str(col))]
                        
                        if len(sf_indices) == 16:
                            # Create mask: all 1s, but apply SF mask values to SF positions
                            for idx, sf_idx in enumerate(sf_indices):
                                mask[sf_idx] = mask_from_study[sf_idx]
                            print(f"Extracted SF mask from full mask and applied to {len(sf_indices)} SF features.")
                        else:
                            # Use full mask as-is
                            mask = mask_from_study
                            print(f"Using full-length mask from ablation study ({len(mask_from_study)} features).")
                    else:
                        # Use full mask as-is
                        mask = mask_from_study
                        print(f"Using full-length mask from ablation study ({len(mask_from_study)} features).")
                else:
                    # Mask size doesn't match - ablation study was run on different data
                    print(f"Warning: Mask from ablation study has {len(mask_from_study)} elements,")
                    print(f"  but current data has {data['X_train'].shape[1]} features.")
                    print(f"  The ablation study was likely run on different data.")
                    print(f"  Using default mask (all 1s).")
                
                if verbose:
                    print(f"Loaded mask from ablation study: {ablation_study_name}")
                    print(f"Original mask shape: {mask_from_study.shape}")
                    print(f"Applied mask shape: {mask.shape}, sum: {mask.sum()}")
                    print(f"Mask (first 50 values): {mask[:50] if len(mask) > 50 else mask}")
            except Exception as e:
                print(f"Warning: Could not load mask from ablation study: {e}")
                print("Using default mask (all 1s)")
                mask = np.ones(data['X_train'].shape[1], dtype=int)
        else:
            # For other studies, use default mask (all 1s)
            mask = np.ones(data['X_train'].shape[1], dtype=int)
    
    # Print mask information
    print(f"\n{'='*60}")
    print(f"Using mask:")
    print(f"  Shape: {mask.shape}")
    print(f"  Sum (active features): {mask.sum()}/{len(mask)}")
    print(f"  Source: {'Ablation study' if study_num and 6 <= study_num <= 10 and mask.sum() < len(mask) else 'Default (all 1s)'}")
    
    # Identify which scoring functions are kept/removed
    feature_columns = data.get('feature_columns')
    sf_indices = []
    if feature_columns is not None:
        import re
        sf_pattern = re.compile(r'^(VINA|SMINA|ODDT|PLANTS)', re.IGNORECASE)
        sf_indices = [i for i, col in enumerate(feature_columns) if sf_pattern.match(str(col))]
        
        # Print mask preview for SF values only (n = number of SFs)
        if len(sf_indices) > 0:
            sf_mask_values = [mask[idx] for idx in sf_indices]
            print(f"  SF mask values ({len(sf_indices)} SFs): {sf_mask_values}")
        else:
            # Fallback: print first n values where n is a reasonable default
            n = min(16, len(mask))
            print(f"  Mask preview (first {n}): {mask[:n]}")
    
    if feature_columns is not None and len(sf_indices) > 0:
        print(f"\n  Scoring Functions (SFs) status:")
        kept_sfs = []
        removed_sfs = []
        for sf_idx in sf_indices:
            sf_name = feature_columns[sf_idx]
            if mask[sf_idx] == 1:
                kept_sfs.append(sf_name)
            else:
                removed_sfs.append(sf_name)
        
        print(f"    Kept ({len(kept_sfs)}/{len(sf_indices)}): {', '.join(kept_sfs) if kept_sfs else 'None'}")
        if removed_sfs:
            print(f"    Removed ({len(removed_sfs)}/{len(sf_indices)}): {', '.join(removed_sfs)}")
        else:
            print(f"    Removed (0/{len(sf_indices)}): None")
    
    print(f"{'='*60}\n")
    
    # Create optimizer
    optimizer = DNNOptimizer(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        X_validation=data.get('X_val'),
        y_validation=data.get('y_val'),
        mask=mask,
        storage=storage,
        output_size=1,
        random_seed=random_seed,
        use_gpu=use_gpu,
        verbose=verbose
    )
    
    # Use provided study_name (must be provided - we're using existing studies)
    if study_name is None:
        raise ValueError(f"study_name must be provided when using existing studies")
    
    # Use provided best_trial or load from study
    import optuna
    if best_trial is None:
        # Load existing study and find best trial
        study = optuna.load_study(study_name=study_name, storage=storage)
        if verbose:
            print(f"Loaded study: {study_name}")
        
        # Get best trial using combined metric (RMSE - AUC) as in your scripts
        nn_df = study.trials_dataframe()
        nn_df = nn_df[nn_df['state'] == 'COMPLETE']
        nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']
        best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])
        best_trial = study.trials[best_nn_df.iloc[0].number]
        
        if verbose:
            print(f"Best trial: {best_trial.number}, RMSE: {best_trial.value:.4f}, AUC: {best_trial.user_attrs.get('AUC', 'N/A')}, Combined: {best_trial.value - best_trial.user_attrs.get('AUC', 0):.4f}")
    else:
        # Use provided best_trial
        if verbose:
            print(f"Using provided best trial: {best_trial.number}, RMSE: {best_trial.value:.4f}, AUC: {best_trial.user_attrs.get('AUC', 'N/A')}, Combined: {best_trial.value - best_trial.user_attrs.get('AUC', 0):.4f}")
    
    # Build and train final model
    from OCDocker.OCScore.Optimization.legacy.models.dnn.DNNOptimizer import NeuralNet
    neural = NeuralNet(
        data['X_train'].shape[1],
        1,
        encoder_params=encoder_params,  # Use autoencoder params if available (for AE with NN)
        nn_params=best_trial.params,
        random_seed=random_seed,
        use_gpu=use_gpu,
        verbose=verbose,
        mask=mask
    )
    
    # Train the model using the best hyperparameters
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Training model with best hyperparameters from trial {best_trial.number}...")
    print(f"  Epochs: {best_trial.params.get('epochs', 'N/A')}")
    print(f"  Batch size: {best_trial.params.get('batch_size', 'N/A')}")
    print(f"  Learning rate: {best_trial.params.get('lr', 'N/A')}")
    print(f"  Optimizer: {best_trial.params.get('optimizer', 'N/A')}")
    print(f"  Random seed: {random_seed}")
    print(f"  Using GPU: {use_gpu}")
    print(f"  Training samples: {data['X_train'].shape[0]}")
    print(f"  Test samples: {data['X_test'].shape[0]}")
    if data.get('X_val') is not None:
        print(f"  Validation samples: {data['X_val'].shape[0]}")
    print(f"{'='*60}\n")
    
    # Train the model - this will run the full training loop
    neural.train_model(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        X_validation=data.get('X_val'),
        y_validation=data.get('y_val')
    )
    
    # Get the trained model (now with trained weights!)
    model = neural.NN
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"  Model trained using best hyperparameters from Optuna trial {best_trial.number}")
    print(f"  Best trial metrics (from Optuna optimization):")
    print(f"    RMSE: {best_trial.value:.4f}")
    auc_value = best_trial.user_attrs.get('AUC', None)
    if auc_value is not None:
        print(f"    AUC: {auc_value:.4f}")
    combined_metric = best_trial.value - (auc_value if auc_value is not None else 0)
    print(f"    Combined metric (RMSE - AUC): {combined_metric:.4f}")
    print(f"{'='*60}\n")
    
    return model, mask


def train_xgboost_model(
        data: dict,
        storage: str,
        storage_id: int,
        model_name: str,
        optimization_type: str = "XGB",
        use_PCA: bool = False,
        pca_type: int = 95,
        no_scores: bool = False,
        only_scores: bool = False,
        study_name: Optional[str] = None,
        use_gpu: bool = False,
        verbose: bool = True
    ) -> tuple:
    '''
    Train an XGBoost model.
    
    Parameters
    ----------
    data : dict
        The data dictionary.
    storage : str
        The storage to use.
    storage_id : int
        The storage ID to use.
    model_name : str
        The name of the model.
    optimization_type : str
        The optimization type.
    use_PCA : bool
        If True, use PCA.
    pca_type : int
        The PCA type to use.
    no_scores : bool
        If True, don't use scores.
    only_scores : bool
        If True, only use scores.
    study_name : str
        The name of the study.
    use_gpu : bool
        If True, use the GPU.
    verbose : bool
        If True, print the output.

    Returns
    -------
    tuple
        (model, mask) - trained model and feature mask
    '''
    
    print("Training XGBoost model...")
    
    # Import XGBoost optimizer
    from OCDocker.OCScore.Optimization.legacy.models.xgboost.XGBoostOptimizer import XGBoostOptimizer
    
    # Create optimizer
    optimizer = XGBoostOptimizer(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        X_validation=data.get('X_val'),
        y_validation=data.get('y_val'),
        storage=storage,
        use_gpu=use_gpu,
        verbose=verbose
    )
    
    # Use provided study_name or build from methodology parameters
    import optuna
    if study_name is None:
        if use_PCA:
            study_name = f"PCA{pca_type}_{optimization_type}_Optimization_{storage_id}"
        elif only_scores:
            study_name = f"ScoreOnly_{optimization_type}_Optimization_{storage_id}"
        elif no_scores:
            study_name = f"NoScores_{optimization_type}_Optimization_{storage_id}"
        else:
            study_name = f"{optimization_type}_Optimization_{storage_id}"
    
    # Load existing study (must exist - we're using best trial only)
    study = optuna.load_study(study_name=study_name, storage=storage)
    if verbose:
        print(f"Loaded study: {study_name}")
    
    # Get best trial from existing study (no new optimization)
    best_trial = study.best_trial
    if verbose:
        print(f"Best trial: {best_trial.number}, RMSE: {best_trial.value:.4f}")
    best_params = best_trial.params
    
    # Train final model with best parameters
    import OCDocker.OCScore.Optimization.legacy.models.xgboost.OCxgboost as OCxgboost
    model, _ = OCxgboost.run_xgboost(
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test'],
        params=best_params,
        verbose=verbose
    )
    
    # Create mask (all 1s for XGBoost, as it handles feature selection internally)
    mask = np.ones(data['X_train'].shape[1], dtype=int)
    
    print(f"XGBoost training completed. Best RMSE: {best_trial.value:.4f}")
    
    return model, mask


if __name__ == "__main__":
    main()
