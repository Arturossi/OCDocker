#!/usr/bin/env python3

# Description
###############################################################################
''' This module provides functions to evaluate the performance of docking methods
    using various metrics, statistical tests, and visualizations.

It is imported as:

import OCDocker.OCScore.Analysis.PerformanceEvaluation as ocperf
'''

# Imports
###############################################################################

from typing import List
import os
import pandas as pd

import OCDocker.OCScore.Analysis.Correlation as occorrana
import OCDocker.OCScore.Analysis.NNUtils as ocnnutils
import OCDocker.OCScore.Analysis.Plotting.Colouring as ocstatcolour
import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot
import OCDocker.OCScore.Analysis.StatTests as ocstat
import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.Evaluation as ocseval
import OCDocker.OCScore.Utils.SimpleConsensus as ocsimple
import OCDocker.OCScore.Utils.StudyParser as ocstudy

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################


def get_all_lists() -> tuple[list[str], int, int]:
    '''
    Generate lists of study names for different optimization methods.

    WARNING: This function is hardcoded for specific study names and may be removed in the future.
    TODO: Replace with a more dynamic approach to fetch study names.
    
    Returns
    -------
    tuple[list[str], int, int]
        A tuple containing:
        - A list of study names for all optimizations.
        - The length of the autoencoder NN list.
        - The length of the genetic algorithm XGB list.
    '''

    #region lists
    # Plain NN
    plain_nn_list = [
        "NN_Optimization_1",
        "NN_Optimization_2",
        "NN_Optimization_3",
        "NN_Optimization_4",
        "NN_Optimization_5"
    ]
    # With autoencoder
    ao_nn_list = [
        "NN_Optimization_6",
        "NN_Optimization_7",
        "NN_Optimization_8",
        "NN_Optimization_9",
        "NN_Optimization_10"
    ]
    # With PCA 80 of variance
    pca80_nn_list = [
        "PCA80_NN_Optimization_11",
        "PCA80_NN_Optimization_12",
        "PCA80_NN_Optimization_13",
        "PCA80_NN_Optimization_14",
        "PCA80_NN_Optimization_15"
    ]
    # With PCA 85 of variance
    pca85_nn_list = [
        "PCA85_NN_Optimization_16",
        "PCA85_NN_Optimization_17",
        "PCA85_NN_Optimization_18",
        "PCA85_NN_Optimization_19",
        "PCA85_NN_Optimization_20"
    ]
    # With PCA 90 of variance
    pca90_nn_list = [
        "PCA90_NN_Optimization_21",
        "PCA90_NN_Optimization_22",
        "PCA90_NN_Optimization_23",
        "PCA90_NN_Optimization_24",
        "PCA90_NN_Optimization_25"
    ]
    # With PCA 95
    pca95_nn_list = [
        "PCA95_NN_Optimization_26",
        "PCA95_NN_Optimization_27",
        "PCA95_NN_Optimization_28",
        "PCA95_NN_Optimization_29",
        "PCA95_NN_Optimization_30",
    ]
    # Score only
    scoreonly_nn_list = [
        "ScoreOnly_NN_Optimization_31",
        "ScoreOnly_NN_Optimization_32",
        "ScoreOnly_NN_Optimization_33",
        "ScoreOnly_NN_Optimization_34",
        "ScoreOnly_NN_Optimization_35"
    ]
    # No Scores 
    noscores_nn_list = [
        "NoScores_NN_Optimization_36",
        "NoScores_NN_Optimization_37",
        "NoScores_NN_Optimization_38",
        "NoScores_NN_Optimization_39",
        "NoScores_NN_Optimization_40"
    ]

    # Plain XGB
    plain_xgb_list = [
        "XGB_Optimization_1",
        "XGB_Optimization_2",
        "XGB_Optimization_3",
        "XGB_Optimization_4",
        "XGB_Optimization_5"
    ]
    # With Genetic Algorithm
    ga_xgb_list = [
        "XGB_Optimization_6",
        "XGB_Optimization_7",
        "XGB_Optimization_8",
        "XGB_Optimization_9",
        "XGB_Optimization_10"
    ]
    # With PCA 80
    pca80_xgb_list = [
        "PCA80_XGB_Optimization_11",
        "PCA80_XGB_Optimization_12",
        "PCA80_XGB_Optimization_13",
        "PCA80_XGB_Optimization_14",
        "PCA80_XGB_Optimization_15"
    ]
    # With PCA 85
    pca85_xgb_list = [
        "PCA85_XGB_Optimization_16",
        "PCA85_XGB_Optimization_17",
        "PCA85_XGB_Optimization_18",
        "PCA85_XGB_Optimization_19",
        "PCA85_XGB_Optimization_20"
    ]
    # With PCA 90
    pca90_xgb_list = [
        "PCA90_XGB_Optimization_21",
        "PCA90_XGB_Optimization_22",
        "PCA90_XGB_Optimization_23",
        "PCA90_XGB_Optimization_24",
        "PCA90_XGB_Optimization_25"
    ]
    # With PCA 95
    pca95_xgb_list = [
        "PCA95_XGB_Optimization_26",
        "PCA95_XGB_Optimization_27",
        "PCA95_XGB_Optimization_28",
        "PCA95_XGB_Optimization_29",
        "PCA95_XGB_Optimization_30",
    ]
    # Score only
    scoreonly_xgb_list = [
        "ScoreOnly_XGB_Optimization_31",
        "ScoreOnly_XGB_Optimization_32",
        "ScoreOnly_XGB_Optimization_33",
        "ScoreOnly_XGB_Optimization_34",
        "ScoreOnly_XGB_Optimization_35"
    ]
    # No Scores 
    noscores_xgb_list = [
        "NoScores_XGB_Optimization_36",
        "NoScores_XGB_Optimization_37",
        "NoScores_XGB_Optimization_38",
        "NoScores_XGB_Optimization_39",
        "NoScores_XGB_Optimization_40"
    ]

    # Plain Transformers
    plain_trans_list = [
        "Trans_Optimization_1",
        "Trans_Optimization_2",
        "Trans_Optimization_3",
        "Trans_Optimization_4",
        "Trans_Optimization_5"
    ]
    # With PCA 80
    pca80_trans_list = [
        "PCA80_Trans_Optimization_6",
        "PCA80_Trans_Optimization_7",
        "PCA80_Trans_Optimization_8",
        "PCA80_Trans_Optimization_9",
        "PCA80_Trans_Optimization_10"
    ]
    # With PCA 85
    pca85_trans_list = [
        "PCA85_Trans_Optimization_11",
        "PCA85_Trans_Optimization_12",
        "PCA85_Trans_Optimization_13",
        "PCA85_Trans_Optimization_14",
        "PCA85_Trans_Optimization_15"
    ]
    # With PCA 90
    pca90_trans_list = [
        "PCA90_Trans_Optimization_16",
        "PCA90_Trans_Optimization_17",
        "PCA90_Trans_Optimization_18",
        "PCA90_Trans_Optimization_19",
        "PCA90_Trans_Optimization_20"
    ]
    # With PCA 95
    pca95_trans_list = [
        "PCA95_Trans_Optimization_21",
        "PCA95_Trans_Optimization_22",
        "PCA95_Trans_Optimization_23",
        "PCA95_Trans_Optimization_24",
        "PCA95_Trans_Optimization_25"
    ]
    # Score only
    scoreonly_trans_list = [
        "ScoreOnly_Trans_Optimization_31",
        "ScoreOnly_Trans_Optimization_32",
        "ScoreOnly_Trans_Optimization_33",
        "ScoreOnly_Trans_Optimization_34",
        "ScoreOnly_Trans_Optimization_35"
    ]
    # No Scores 
    noscores_trans_list = [
        "NoScores_Trans_Optimization_36",
        "NoScores_Trans_Optimization_37",
        "NoScores_Trans_Optimization_38",
        "NoScores_Trans_Optimization_39",
        "NoScores_Trans_Optimization_40"
    ]
    #endregion

    # Concatenate all the lists
    snames = plain_nn_list + ao_nn_list + pca80_nn_list + pca85_nn_list + pca90_nn_list + pca95_nn_list + scoreonly_nn_list + noscores_nn_list \
        + plain_xgb_list + ga_xgb_list + pca80_xgb_list + pca85_xgb_list + pca90_xgb_list + pca95_xgb_list + scoreonly_xgb_list + noscores_xgb_list \
        + plain_trans_list + pca80_trans_list + pca85_trans_list + pca90_trans_list + pca95_trans_list + scoreonly_trans_list + noscores_trans_list
    
    return snames, len(ao_nn_list), len(ga_xgb_list)


def setup_dirs() -> None:
    '''Ensure the output directories for plots and CSVs exist.'''
    
    # Skip directory creation during Sphinx documentation builds
    if os.environ.get('OC_BUILD_DOCS') == '1':
        return

    os.makedirs('plots', exist_ok=True)
    os.makedirs('csvs', exist_ok=True)


def load_combined_metrics(df_path: str, metrics: list[str] = ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis']) -> pd.DataFrame:
    '''
    Load DUDEz and PDBbind data, compute evaluation metrics, and combine with consensus scores.

    Parameters
    ----------
    df_path : str
        Path to the compressed dataframe file (usually OCDocker.csv.gz).

    Returns
    -------
    pd.DataFrame
        Combined dataframe with AUC, RMSE, and consensus-derived metrics.
    metrics : list[str], optional
        List of metrics to calculate. Default is ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis']
        If empty, all metrics will be calculated.
    '''

    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)

    # Compute performance metrics
    dudez_metrics = ocseval.compute_auc(dudez_data, "ligand", score_columns, "type")
    pdbbind_metrics = ocseval.compute_rmse(pdbbind_data, score_columns, "experimental")

    docking_metrics = pd.merge(dudez_metrics, pdbbind_metrics, on="score_column")
    docking_metrics["Methodology"] = "Raw Scoring Function"

    simple_consensus = ocsimple.perform_simple_consensus(df_path, threshold=1.2, metrics=metrics, verbose=False)

    # Process the label as the only one metric which is being calculated, otherwise use a generic name "Simple consensus"
    if metrics and len(metrics) == 1:
        simple_consensus["Methodology"] = f"{metrics[0].capitalize()} consensus"
    else:
        simple_consensus["Methodology"] = "Simple consensus"
    
    simple_consensus["score_column"] = simple_consensus.index
    simple_consensus.reset_index(drop=True, inplace=True)

    final_metrics = pd.concat([docking_metrics, simple_consensus], axis=0)
    final_metrics["combined_metric"] = final_metrics["RMSE"] - final_metrics["AUC"]
    final_metrics.rename(columns={"score_column": "study_name"}, inplace=True)
    final_metrics.reset_index(drop=True, inplace=True)

    return final_metrics


def get_feature_matrix(df_path: str) -> pd.DataFrame:
    '''
    Load and return the feature matrix for PCA from the dataset.

    Parameters
    ----------
    df_path : str
        Path to the input dataset (e.g., OCDocker.csv.gz).

    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix excluding non-descriptor columns.
    '''

    df = pd.read_csv(df_path, compression='infer')

    score_columns = df.filter(regex=f"^({'|'.join(['SMINA', 'VINA', 'ODDT', 'PLANTS'])})").columns.to_list()

    # Columns to exclude (metadata or targets)
    exclude_cols = [
        'receptor', 'ligand', 'name', 'type', 'db', 'experimental', 'label',
        'Methodology', 'study_name', 'RMSE', 'AUC', 'combined_metric'
    ] + score_columns

    # Drop score columns if they follow a known pattern (optional)
    score_cols = [col for col in df.columns if col.lower().startswith('score')]
    exclude_cols.extend(score_cols)

    feature_df = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')

    return feature_df


def run_full_analysis(
        df_path: str,
        base_path: str,
        storage_str: str,
        trials_list: List[int],
        output_dir: str = "plots",
        palette_colour: str = "glasbey",
        rmse_threshold: float = 1.5,
        feature_analysis: bool = True,
        plot_summary: bool = True
    ) -> None:
    '''
    Run the full evaluation and visualization pipeline.

    Parameters
    ----------
    df_path : str
        Path to the input metrics CSV file.
    base_path : str
        Base directory containing model folders and PCA files.
    storage_str : str
        SQLAlchemy-compatible DB connection string.
    trials_list : List[int]
        List of trial counts to run the evaluation for (e.g. [1, 5, 10, 100]).
    palette_colour : str
        Color palette for plots (default is "glasbey").
    feature_analysis : bool
        If True, perform PCA and AE importance analysis.
    plot_summary : bool
        If True, generate plots summarizing performance.
    '''

    setup_dirs()

    final_metrics = load_combined_metrics(df_path, metrics = ["mean"])

    for n_trials in trials_list:
        print(f"Running analysis for top {n_trials} trials")

        snames, nn_len, xgb_len = get_all_lists()
        nn_ae_start = 5 * n_trials
        nn_ae_end = nn_ae_start + (n_trials * nn_len)
        xgb_ga_start = 45 * n_trials
        xgb_ga_end = xgb_ga_start + (n_trials * xgb_len)

        results_df_rmse, results_df_auc, results_df = ocstudy.analyze_studies(snames, storage=storage_str, n_trials=n_trials)

        for i, df in enumerate([results_df_rmse, results_df_auc, results_df]):
            df.loc[nn_ae_start:nn_ae_end - 1, 'study_type'] = 'NN + AE'
            df.loc[xgb_ga_start:xgb_ga_end - 1, 'study_type'] = 'XGB + GA'

            merged = pd.concat([
                df.rename(columns={
                    'study_name': 'study_name',
                    'study_type': 'Methodology',
                    'rmse': 'RMSE',
                    'auc': 'AUC',
                    'best_combined_value': 'RMSE',
                    'best_combined_auc': 'AUC',
                    'best_combined_metric': 'combined_metric'
                }),
                final_metrics
            ])

            if i == 0:
                filtered_df_rmse = merged[merged['RMSE'] <= rmse_threshold].reset_index(drop=True)
            if i == 1:
                filtered_df_auc = merged[merged['AUC'] >= 0.5].reset_index(drop=True)
            elif i == 2:
                filtered_df = merged[merged['RMSE'] <= rmse_threshold].reset_index(drop=True)

        colour_mapping = ocstatcolour.set_color_mapping(filtered_df, palette_colour)
            
        if plot_summary:
            ocstatplot.plot_combined_metric_scatter(filtered_df, n_trials, colour_mapping, output_dir=output_dir)
            ocstatplot.plot_boxplots(filtered_df, n_trials, colour_mapping, output_dir=output_dir)
            #plot_barplots2(filtered_df, n_trials, colour_mapping, output_dir=output_dir)
            ocstatplot.plot_scatterplot(filtered_df, filtered_df_rmse, filtered_df_auc, n_trials, colour_mapping, output_dir=output_dir)

        ocstat.run_statistical_tests(filtered_df, n_trials, colour_mapping, output_dir=output_dir)
        occorrana.correlation_analysis(results_df, final_metrics, n_trials, error_threshold=1.5) # Não sei o quanto útil isso é

        if feature_analysis:
            pca_dir = os.path.join(base_path, "models")
            ocstat.run_pca_analysis(
                data_matrix=get_feature_matrix(df_path),
                models_dir=pca_dir,
                output_dir=output_dir,
                n_trials=n_trials,
                n_features=20,
                #colour_mapping=colour_mapping
            )
            
            # Optionally call AE importance
            # TODO: Replace this mock with your real AE model and inputs
            # Example: ocnnutils.run_ae_feature_importance(ae_model, X_valid, y_valid, features)

    print("Full analysis completed.")

'''
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
df_path: str = f"{base_path}/OCDocker.csv.gz"
trials_list = [100]#[1, 5, 10, 50, 100, 500]

from urllib.parse import quote_plus

user = "ocdocker"
password = "@Kp3sRv9t@"
host = "localhost"
port = 3306
db = "optimization"

# Set the storage
storage_str = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db}"

palette_colour = "glasbey"
output_dir = "plots"

run_full_analysis(
    df_path=df_path,
    base_path=base_path,
    storage_str=storage_str,
    trials_list=trials_list,
    output_dir=output_dir,
    palette_colour=palette_colour,
    rmse_threshold=1.5,
    feature_analysis=True,
    plot_summary=True
)
'''
