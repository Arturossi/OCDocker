#!/usr/bin/env python3
import sys
import os
import logging
import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from urllib.parse import quote_plus

from OCDocker.Initialise import *
import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.Evaluation as ocseval
import OCDocker.OCScore.Utils.SimpleConsensus as ocsimple
import OCDocker.OCScore.Utils.StudyParser as ocstudy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "user": "ocdocker",
    "password": "@Kp3sRv9t@",
    "host": "localhost",
    "port": 3306,
    "db": "optimization"
}

logger.info("Starting the analysis of the results.")

# Paths
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
df_path: str = f"{base_path}/OCDocker.csv.gz"

def load_data(df_path: str):
    """
    Load the DataFrames using ocscoredata preprocess_df function
    """
    logger.info(f"Reading the input file: '{df_path}'")
    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)
    return dudez_data, pdbbind_data, score_columns

def compute_metrics(dudez_data, pdbbind_data, score_columns):
    """
    Compute AUC and RMSE metrics for the data.
    """
    logger.info("Computing the metrics for the results.")
    dudez_metrics = ocseval.compute_auc(dudez_data, "ligand", score_columns, "type")
    pdbbind_metrics = ocseval.compute_rmse(pdbbind_data, score_columns, "experimental")
    docking_metrics = dudez_metrics.merge(pdbbind_metrics, on="score_column")
    docking_metrics["Methodology"] = "Raw Scoring Function"
    return docking_metrics

def concatenate_lists(*args):
    """
    Concatenate multiple lists into one.
    """
    combined = []
    for study_list in args:
        combined += study_list
    return combined

# Define study lists
plain_nn_list = ["NN_Optimization_1", "NN_Optimization_2", "NN_Optimization_3", "NN_Optimization_4", "NN_Optimization_5"]
ao_nn_list = ["NN_Optimization_6", "NN_Optimization_7", "NN_Optimization_8", "NN_Optimization_9", "NN_Optimization_10"]
pca80_nn_list = ["PCA80_NN_Optimization_11", "PCA80_NN_Optimization_12", "PCA80_NN_Optimization_13", "PCA80_NN_Optimization_14", "PCA80_NN_Optimization_15"]
pca85_nn_list = ["PCA85_NN_Optimization_16", "PCA85_NN_Optimization_17", "PCA85_NN_Optimization_18", "PCA85_NN_Optimization_19", "PCA85_NN_Optimization_20"]
pca90_nn_list = ["PCA90_NN_Optimization_21", "PCA90_NN_Optimization_22", "PCA90_NN_Optimization_23", "PCA90_NN_Optimization_24", "PCA90_NN_Optimization_25"]
pca95_nn_list = ["PCA95_NN_Optimization_26", "PCA95_NN_Optimization_27", "PCA95_NN_Optimization_28", "PCA95_NN_Optimization_29", "PCA95_NN_Optimization_30"]
scoreonly_nn_list = ["ScoreOnly_NN_Optimization_31", "ScoreOnly_NN_Optimization_32", "ScoreOnly_NN_Optimization_33", "ScoreOnly_NN_Optimization_34", "ScoreOnly_NN_Optimization_35"]
noscores_nn_list = ["NoScores_NN_Optimization_36", "NoScores_NN_Optimization_37", "NoScores_NN_Optimization_38", "NoScores_NN_Optimization_39", "NoScores_NN_Optimization_40"]
plain_xgb_list = ["XGB_Optimization_1", "XGB_Optimization_2", "XGB_Optimization_3", "XGB_Optimization_4", "XGB_Optimization_5"]
ga_xgb_list = ["XGB_Optimization_6", "XGB_Optimization_7", "XGB_Optimization_8", "XGB_Optimization_9", "XGB_Optimization_10"]
pca80_xgb_list = ["PCA80_XGB_Optimization_11", "PCA80_XGB_Optimization_12", "PCA80_XGB_Optimization_13", "PCA80_XGB_Optimization_14", "PCA80_XGB_Optimization_15"]
pca85_xgb_list = ["PCA85_XGB_Optimization_16", "PCA85_XGB_Optimization_17", "PCA85_XGB_Optimization_18", "PCA85_XGB_Optimization_19", "PCA85_XGB_Optimization_20"]
pca90_xgb_list = ["PCA90_XGB_Optimization_21", "PCA90_XGB_Optimization_22", "PCA90_XGB_Optimization_23", "PCA90_XGB_Optimization_24", "PCA90_XGB_Optimization_25"]
pca95_xgb_list = ["PCA95_XGB_Optimization_26", "PCA95_XGB_Optimization_27", "PCA95_XGB_Optimization_28", "PCA95_XGB_Optimization_29", "PCA95_XGB_Optimization_30"]
scoreonly_xgb_list = ["ScoreOnly_XGB_Optimization_31", "ScoreOnly_XGB_Optimization_32", "ScoreOnly_XGB_Optimization_33", "ScoreOnly_XGB_Optimization_34", "ScoreOnly_XGB_Optimization_35"]
noscores_xgb_list = ["NoScores_XGB_Optimization_36", "NoScores_XGB_Optimization_37", "NoScores_XGB_Optimization_38", "NoScores_XGB_Optimization_39", "NoScores_XGB_Optimization_40"]
plain_trans_list = ["Trans_Optimization_1", "Trans_Optimization_2", "Trans_Optimization_3", "Trans_Optimization_4", "Trans_Optimization_5"]
pca80_trans_list = ["PCA80_Trans_Optimization_6", "PCA80_Trans_Optimization_7", "PCA80_Trans_Optimization_8", "PCA80_Trans_Optimization_9", "PCA80_Trans_Optimization_10"]
pca85_trans_list = ["PCA85_Trans_Optimization_11", "PCA85_Trans_Optimization_12", "PCA85_Trans_Optimization_13", "PCA85_Trans_Optimization_14", "PCA85_Trans_Optimization_15"]
pca90_trans_list = ["PCA90_Trans_Optimization_16", "PCA90_Trans_Optimization_17", "PCA90_Trans_Optimization_18", "PCA90_Trans_Optimization_19", "PCA90_Trans_Optimization_20"]
pca95_trans_list = ["PCA95_Trans_Optimization_21", "PCA95_Trans_Optimization_22", "PCA95_Trans_Optimization_23", "PCA95_Trans_Optimization_24", "PCA95_Trans_Optimization_25"]
scoreonly_trans_list = ["ScoreOnly_Trans_Optimization_31", "ScoreOnly_Trans_Optimization_32", "ScoreOnly_Trans_Optimization_33", "ScoreOnly_Trans_Optimization_34", "ScoreOnly_Trans_Optimization_35"]
noscores_trans_list = ["NoScores_Trans_Optimization_36", "NoScores_Trans_Optimization_37", "NoScores_Trans_Optimization_38", "NoScores_Trans_Optimization_39", "NoScores_Trans_Optimization_40"]

# Concatenate all the study names
snames = concatenate_lists(
    plain_nn_list, ao_nn_list, pca80_nn_list, pca85_nn_list, pca90_nn_list, pca95_nn_list, scoreonly_nn_list, noscores_nn_list,
    plain_xgb_list, ga_xgb_list, pca80_xgb_list, pca85_xgb_list, pca90_xgb_list, pca95_xgb_list, scoreonly_xgb_list, noscores_xgb_list,
    plain_trans_list, pca80_trans_list, pca85_trans_list, pca90_trans_list, pca95_trans_list, scoreonly_trans_list, noscores_trans_list
)

# Set up storage path for MySQL database
storage = f"mysql+pymysql://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['db']}"

def fetch_results(storage, snames):
    """
    Fetch results from the database for the given study names.
    """
    logger.info("Fetching the results from the database.")
    results_df = ocstudy.analyze_studies(snames, storage=storage)
    return results_df

def preprocess_results(results_df):
    """
    Preprocess and clean up the results dataframe, fix study types, and compute the combined metrics.
    """
    logger.info("Preprocessing the results.")
    results_df.loc[25:49, 'study_type'] = 'NN + AE'
    results_df.loc[225:249, 'study_type'] = 'XGB + GA'
    return results_df

def compute_combined_metrics(final_metrics, results_df):
    """
    Compute combined metrics (RMSE - AUC) for the results.
    """
    logger.info("Computing the combined metrics for RMSE and AUC.")
    # Merge the metrics and add combined metric
    best_rmse_df = results_df[["study_name", "study_type", "best_rmse_number", "best_rmse_value", "best_rmse_auc"]]
    best_auc_df = results_df[["study_name", "study_type", "best_auc_number", "best_auc_value", "best_auc"]]
    best_combined_df = results_df[["study_name", "study_type", "best_combined_number", "best_combined_metric", "best_combined_value", "best_combined_auc"]]

    # Add simple consensus to the results
    final_metrics["Methodology"] = "Simple consensus"
    best_rmse_df = pd.concat([best_rmse_df, final_metrics], axis=0)
    best_auc_df = pd.concat([best_auc_df, final_metrics], axis=0)
    best_combined_df = pd.concat([best_combined_df, final_metrics], axis=0)

    return best_rmse_df, best_auc_df, best_combined_df

def plot_scatter(df, x_col, y_col, color_mapping, output_path, shape_col=None):
    """
    General function to plot scatterplot for visual comparison.
    """
    logger.info("Plotting scatterplot for visual comparison.")
    
    # Drop duplicate rows to avoid reindexing issues
    df = df.drop_duplicates(subset=[x_col, y_col, 'Methodology'])

    plt.figure(figsize=(20, 8))

    # Plot with unique labels
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='Methodology', palette=color_mapping)
    
    if shape_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='Methodology', palette=color_mapping, marker='*')
    
    plt.savefig(output_path)
    plt.close()
    
def main():
    # Load data
    dudez_data, pdbbind_data, score_columns = load_data(df_path)

    # Compute metrics
    docking_metrics = compute_metrics(dudez_data, pdbbind_data, score_columns)

    # Compute simple consensus
    simple_docking_consensus = ocsimple.perform_simple_consensus(df_path, threshold=1.2, verbose=False)
    simple_docking_consensus["Methodology"] = "Simple consensus"
    simple_docking_consensus["score_column"] = simple_docking_consensus.index
    simple_docking_consensus.reset_index(drop=True, inplace=True)

    # Fetch results from the database
    results_df = fetch_results(storage, snames)

    # Preprocess results
    results_df = preprocess_results(results_df)

    # Compute combined metrics and add final metrics
    final_metrics = pd.concat([docking_metrics, simple_docking_consensus], axis=0)
    best_rmse_df, best_auc_df, best_combined_df = compute_combined_metrics(final_metrics, results_df)

    # Plot results
    plot_scatter(best_rmse_df, "RMSE", "AUC", sns.color_palette(cc.glasbey, n_colors=best_combined_df['Methodology'].nunique()), "plots/Experiments.png")

if __name__ == "__main__":
    main()
