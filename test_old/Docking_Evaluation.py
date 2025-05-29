#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

import os
import math

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns

from typing import Union
from urllib.parse import quote_plus

from OCDocker.Initialise import *

import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.Evaluation as ocseval
import OCDocker.OCScore.Utils.SimpleConsensus as ocsimple
import OCDocker.OCScore.Utils.StudyParser as ocstudy

# Bar plot with significance annotations
def plot_bar_with_significance(data, metric, y_col='diff'):
    # Calculate means by Methodology for bar plot
    means = pd.concat([data[['A', 'mean(A)']].rename(columns={'A': 'Methodology', 'mean(A)': y_col}),
                       data[['B', 'mean(B)']].rename(columns={'B': 'Methodology', 'mean(B)': y_col})], ignore_index=True)
    means = means.groupby('Methodology').mean().reset_index()

    # Count significant differences for each methodology
    significant_counts = data[data['pval'] < 0.05].groupby('A').size().reindex(means['Methodology'], fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Methodology', y=y_col, data=means, ci='sd', capsize=0.2)
    plt.xticks(rotation=90)
    plt.title(f"{metric} Means with Significant Pairwise Differences")

    # Annotate each bar with the count of significant differences
    for i, count in enumerate(significant_counts):
        if count > 0:
            plt.text(i, means.iloc[i][y_col] + 0.01, f"*{count}", ha='center', color='red')

    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"plots/games_howell_barplot_{metric}.png")

# Heatmap for p-values
def plot_heatmap(data, title, metric):
    methodologies = sorted(set(data['A']).union(set(data['B'])))
    p_matrix = np.ones((len(methodologies), len(methodologies)))  # Initialize with 1's for clarity in the heatmap
    
    # Fill in the matrix with p-values
    for _, row in data.iterrows():
        i, j = methodologies.index(row['A']), methodologies.index(row['B'])
        p_matrix[i, j] = row['pval']
        p_matrix[j, i] = row['pval']  # Make it symmetric

    # Custom annotation function to format numbers
    def custom_fmt(x):
        if x >= 0.01:       # Display as regular decimal if larger than 0.01
            return f"{x:.2f}"
        elif x < 0.01:      # Display in scientific notation for small values
            return f"{x:.2e}".replace("e+00", "")  # Remove "e+00" if present
        return str(x)       # Default case, should not occur

    # Create annotations with custom format as strings
    annot_matrix = np.vectorize(custom_fmt)(p_matrix)

    plt.figure(figsize=(16, 14))  # Larger figure for readability
    ax = sns.heatmap(p_matrix, annot=annot_matrix, xticklabels=methodologies, yticklabels=methodologies, 
                     cmap="coolwarm", cbar_kws={'label': 'p-value'}, annot_kws={"size": 8, "rotation": 45}, fmt="")  # Blank fmt
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/games_howell_heatmap_{metric}.png")

def load_data(df_path: str) -> pd.DataFrame:
    # Load the DataFrames
    dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)

    print("Computing the metrics for the results.")

    # Compute the metrics
    dudez_metrics = ocseval.compute_auc(dudez_data, "ligand", score_columns, "type")
    pdbbind_metrics = ocseval.compute_rmse(pdbbind_data, score_columns, "experimental")

    # Merge the metrics using score_column as the key for joining
    docking_metrics = dudez_metrics.merge(pdbbind_metrics, on = "score_column")

    # Set the Methodology as Raw Scoring Function
    docking_metrics["Methodology"] = "Raw Scoring Function"

    # Compute the simple consensus
    simple_docking_consensus = ocsimple.perform_simple_consensus(df_path, threshold = 1.2, verbose = False)

    # Set the Methodology as Raw Scoring Function
    simple_docking_consensus["Methodology"] = "Simple consensus"

    # Set the column score_column with the index values
    simple_docking_consensus["score_column"] = simple_docking_consensus.index

    # Reset the index
    simple_docking_consensus.reset_index(drop = True, inplace = True)

    # Concatenate the metrics vertically
    final_metrics = pd.concat([docking_metrics, simple_docking_consensus], axis = 0)

    # Compute the RMSE - AUC (score)
    final_metrics["combined_metric"] = final_metrics["RMSE"] - final_metrics["AUC"]

    # Reset the index
    final_metrics.reset_index(drop = True, inplace = True)

    # Rename the score_column to study_name
    final_metrics.rename(columns = {"score_column": "study_name"}, inplace = True)

    return final_metrics

def get_all_lists() -> tuple[list[str], int, int]:
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

def get_study_data(
        snames, 
        storage, 
        error_threshold = 1.5, 
        nn_ae_start = None, 
        nn_ae_end = None, 
        xgb_ga_start = None, 
        xgb_ga_end = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float, float, float, float]:
    # Fetch the results
    results_df = ocstudy.analyze_studies(snames, storage = storage, n_trials=n_trials)

    print("Preprocessing the results.")

    if nn_ae_start and nn_ae_end:
        if nn_ae_start >= nn_ae_end:
            raise ValueError("The start index for 'NN + AE' must be less than the end index.")
        # Fix the study type for 'NN + AE'
        results_df.loc[nn_ae_start:nn_ae_end - 1, 'study_type'] = 'NN + AE'
    
    if xgb_ga_start and xgb_ga_end:
        if xgb_ga_start >= xgb_ga_end:
            raise ValueError("The start index for 'XGB + GA' must be less than the end index.")
        # Fix the study type for 'XGB + GA'
        results_df.loc[xgb_ga_start:xgb_ga_end - 1, 'study_type'] = 'XGB + GA'

    # Separate results by 3 evaluation metrics (Error (Smallest Error), Error (Biggest AUC), Error (Smallest Error - AUC))
    best_rmse_df = results_df[["study_name", "study_type", "best_rmse_number", "best_rmse_value", "best_rmse_auc"]]
    best_auc_df = results_df[["study_name", "study_type", "best_auc_number", "best_auc_value", "best_auc"]]
    best_combined_df = results_df[["study_name", "study_type", "best_combined_number", "best_combined_metric", "best_combined_value", "best_combined_auc"]]

    ## Rename the columns ##

    # study_type -> Methodology
    best_rmse_df.rename(columns = {"study_type": "Methodology", "best_rmse_number": "Experiment", "best_rmse_value": "RMSE", "best_rmse_auc": "AUC"}, inplace = True)
    best_auc_df.rename(columns = {"study_type": "Methodology", "best_auc_number": "Experiment", "best_auc_value": "RMSE", "best_auc": "AUC"}, inplace = True)
    best_combined_df.rename(columns = {"study_type": "Methodology", "best_combined_number": "Experiment", "best_combined_metric": "combined_metric", "best_combined_value": "RMSE", "best_combined_auc": "AUC"}, inplace = True)

    ## Set the combined_metric for rmse and auc ##

    best_rmse_df["combined_metric"] = best_rmse_df["RMSE"] - best_rmse_df["AUC"]
    best_auc_df["combined_metric"] = best_auc_df["RMSE"] - best_auc_df["AUC"]

    ## Add the simple consensus to the results ##

    best_rmse_df = pd.concat([best_rmse_df, final_metrics], axis = 0)
    best_auc_df = pd.concat([best_auc_df, final_metrics], axis = 0)
    best_combined_df = pd.concat([best_combined_df, final_metrics], axis = 0)

    ## Get the minimum and maximum values for AUC and Error ##

    # Get the Error range
    min_error = min([best_rmse_df['RMSE'].min(), best_auc_df['RMSE'].min(), best_combined_df['RMSE'].min()])
    max_error = max([best_rmse_df['RMSE'].max(), best_auc_df['RMSE'].max(), best_combined_df['RMSE'].max()])
    #max_error = 1.0

    # Compute the new AUCs
    best_rmse_df['AUC New'] = best_rmse_df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)
    best_auc_df['AUC New'] = best_auc_df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)
    best_combined_df['AUC New'] = best_combined_df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)

    # Reindex the dataframes
    best_rmse_df.reset_index(drop = True, inplace = True)
    best_auc_df.reset_index(drop = True, inplace = True)
    best_combined_df.reset_index(drop = True, inplace = True)

    # Get the AUC range
    min_auc = min([best_rmse_df['AUC New'].min(), best_auc_df['AUC New'].min(), best_combined_df['AUC New'].min()])
    max_auc = max([best_rmse_df['AUC New'].max(), best_auc_df['AUC New'].max(), best_combined_df['AUC New'].max()])

    error_range = max_error - min_error
    auc_range = max_auc - min_auc

    # Get the rows with error greater than the error_threshold
    best_rmse_df_filtered = best_rmse_df[best_rmse_df['RMSE'] <= error_threshold]
    best_auc_df_filtered = best_auc_df[best_auc_df['RMSE'] <= error_threshold]
    best_combined_df_filtered = best_combined_df[best_combined_df['RMSE'] <= error_threshold]

    return best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered, results_df, min_auc, max_auc, min_error, max_error, error_range, auc_range

def setup_dirs() -> None:
    # If the plots and csvs folder does not exist, create it
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('csvs'):
        os.makedirs('csvs')
    return None

def set_color_mapping(df, palette_colour = "glasbey") -> dict[str, str]:
    print("Setting the pallette, alpha, and error threshold for the plots.")

    # Palette
    #palette_colour = "Set2"
    #palette_colour = "Set3"
    #palette_colour = "tab10"
    #palette_colour = "tab20"
    #palette_colour = "colorblind"
    #palette_colour = "pastel"
    #palette_colour = "bright"
    #palette_colour = "dark"
    #palette_colour = "deep"
    #palette_colour = "muted"
    #palette_colour = "viridis"
    if palette_colour == "glasbey":
        palette_colour = sns.color_palette(cc.glasbey, n_colors=df['Methodology'].nunique())

    # Create a color mapping for methodologies
    color_mapping = {
        method: color for method, color in zip(
            df['Methodology'].unique(), 
            sns.color_palette(palette_colour, n_colors=df['Methodology'].nunique())
        )
    }

    return color_mapping, palette_colour

def get_ae_xgb_indices(n_ae: int, n_xgb: int, n_trials: int) -> tuple[int, int, int, int]:
    # Calculate the index ranges dynamically
    nn_ae_start = 5 * n_trials  # Start index for 'NN + AE'
    nn_ae_end = nn_ae_start + (n_trials * n_ae)
    xgb_ga_start = 45 * n_trials  # Start index for 'XGB + GA'
    xgb_ga_end = xgb_ga_start + (n_trials * n_xgb)

    return nn_ae_start, nn_ae_end, xgb_ga_start, xgb_ga_end

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_single_scatterplot(best_combined_df_filtered, n_trials, color_mapping, min_auc, max_auc, min_error, max_error, error_range, auc_range, alpha=0.9):
    # Create the figure
    plt.figure(figsize=(10, 8))

    # Prepare the data by adding a new column indicating AUC category
    best_combined_df_filtered['AUC_category'] = best_combined_df_filtered["AUC"].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')
    # Make 1 - AUC for AUC < 0.5
    best_combined_df_filtered.loc[best_combined_df_filtered['AUC_category'] == '< 0.5', "AUC"] = 1 - best_combined_df_filtered["AUC"]

    # Plot AUC >= 0.5 with circles
    sns.scatterplot(
        data=best_combined_df_filtered[best_combined_df_filtered['AUC_category'] == '>= 0.5'], 
        x="RMSE", 
        y="AUC",
        hue='Methodology', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha, 
        marker='o'
    )

    # Plot AUC < 0.5 with stars
    '''sns.scatterplot(
        data=best_combined_df_filtered[best_combined_df_filtered['AUC_category'] == '< 0.5'], 
        x="RMSE", 
        y="AUC",
        hue='Methodology', 
        legend=False, 
        palette=color_mapping,
        alpha=alpha,  
        marker='*', 
        s=100
    )'''

    # Set title, axis labels, and limits
    plt.title('Error vs. AUC (Smallest Error - AUC)')
    plt.xlabel('Error')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

    # Set x and y limits based on the range of RMSE and AUC
    error_range = best_combined_df_filtered["RMSE"].max() - best_combined_df_filtered["RMSE"].min()
    plt.xlim(best_combined_df_filtered["RMSE"].min() - error_range * 0.1, best_combined_df_filtered["RMSE"].max() + error_range * 0.1)
    plt.ylim(min_auc - auc_range * 0.1, max_auc + auc_range * 0.1)

    # Add the legends
    '''shape_labels = ['AUC >= 0.5 (= AUC)', 'AUC < 0.5 (= 1-AUC)']
    shape_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='AUC >= 0.5 (= AUC)', markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], marker='*', color='w', label='AUC < 0.5 (= 1-AUC)', markerfacecolor='gray', markersize=10)
    ]'''

    color_labels = best_combined_df_filtered['Methodology'].unique().tolist()
    color_handles = [plt.Line2D([0], [0], color=color_mapping[method], lw=4) for method in color_labels]

    # Place legends
    #plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.1, 0.02), ncol=1, title='AUC')
    plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, title='Methodology')

    # Adjust layout to accommodate legend spacing
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    
    # Save the plot
    plt.savefig(f'plots/Experiments_Smallest_Error_AUC_{n_trials}.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    return None

def plot_scatterplot(best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered, n_trials, color_mapping, min_auc, max_auc, min_error, max_error, error_range, auc_range, alpha = 0.9):
    # Plotting with the chosen palette and adjustments for marker and transparency
    plt.figure(figsize=(20, 8))

    for i, (metric, df) in enumerate([('RMSE', best_rmse_df_filtered), ('AUC', best_auc_df_filtered), ('RMSE-AUC', best_combined_df_filtered)]):
        plt.subplot(1, 3, i+1)

        # Prepare the data by adding a new column indicating AUC category
        df['AUC_category'] = df["AUC"].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')

        # Make 1 - AUC for AUC < 0.5
        df.loc[df['AUC_category'] == '< 0.5', "AUC"] = 1 - df["AUC"]

        # Plot the df_auc_ge_05 normally
        sns.scatterplot(
            data=df[df['AUC_category'] == '>= 0.5'], 
            x="RMSE", 
            y="AUC",
            hue='Methodology', 
            legend=False, 
            palette=color_mapping,
            alpha=alpha,  # Adjusting transparency
            marker='o', # You can change markers for each method if needed
        )

        # Now plot the df_auc_lt_05 with a different marker (star)
        sns.scatterplot(
            data=df[df['AUC_category'] == '< 0.5'], 
            x="RMSE", 
            y="AUC",
            hue='Methodology', 
            legend=False, 
            palette=color_mapping,
            alpha=alpha,  # Adjusting transparency
            marker='*', # You can change markers for each method if needed
            s=100,
        )

        # Set the title (smallest error, biggest auc, smallest error - auc) according to the metric
        if metric == 'RMSE':
            plt.title(f'Error vs. AUC (Smallest Error)')
        elif metric == 'AUC':
            plt.title(f'Error vs. AUC (Biggest AUC)')
        else:
            plt.title(f'Error vs. AUC (Smallest Error - AUC)')
                
        #plt.xlim(min_error - error_range * 0.1, max_error + error_range * 0.1)
        # Set as minimum value of x-axis the minimum value of the error minus 10% of the error range and the maximum value of x-axis the maximum value of the error plus 10% of the error range for each plot
        error_range = df["RMSE"].max() - df["RMSE"].min()
        plt.xlim(df["RMSE"].min() - error_range * 0.1, df["RMSE"].max() + error_range * 0.1)
        #plt.ylim(-0.1, 1.1)
        plt.ylim(min_auc - auc_range * 0.1, max_auc + auc_range * 0.1)
        plt.xlabel('Error')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

    # Extend the space under the plot to add the legend
    plt.subplots_adjust(bottom=0.4)

    # First legend for the shapes
    shape_labels = ['AUC >= 0.5 (= AUC)', 'AUC < 0.5 (= 1-AUC)']
    shape_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='AUC >= 0.5 (= AUC)', markerfacecolor='gray', markersize=10), # type: ignore
        plt.Line2D([0], [0], marker='*', color='w', label='AUC < 0.5 (= 1-AUC)', markerfacecolor='gray', markersize=10) # type: ignore
    ]

    # Second legend for the colors (Methodology)
    color_labels = df['Methodology'].unique().tolist()
    color_handles = [plt.Line2D([0], [0], color=color_mapping[method], lw=4) for method in color_labels] # type: ignore

    # Place the AUC shape legend at the bottom left
    plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.26, 0.03), ncol=1, title='AUC')

    # Place the Methodology color legend at the bottom center
    plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.57, 0.03), ncol=4, title='Methodology')

    # Use tight_layout to adjust the spacing, but leave the space for the legends under the plot
    plt.tight_layout(rect=[0, 0.22, 1, 1])

    plt.savefig(f'plots/Experiments_{n_trials}.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    return None

def separate_dfs(best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered, to_remove = []) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create three new dataframes, one for Error (Smallest Error), one for Error (Biggest AUC), and one for Error (Smallest Error - AUC)
    df_error_menor_erro = best_rmse_df_filtered[['Experiment', 'Methodology', 'RMSE']].copy()
    df_error_maior_auc = best_auc_df_filtered[['Experiment', 'Methodology', 'RMSE']].copy()
    df_error_menor_erro_auc = best_combined_df_filtered[['Experiment', 'Methodology', 'RMSE']].copy()

    # Add the metric name to each dataframe in Methodology (except for Raw Scoring Function and Simple consensus)
    df_error_menor_erro['Methodology'] = df_error_menor_erro['Methodology'].apply(lambda x: f"{x} (Smallest Error)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
    df_error_maior_auc['Methodology'] = df_error_maior_auc['Methodology'].apply(lambda x: f"{x} (Biggest AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
    df_error_menor_erro_auc['Methodology'] = df_error_menor_erro_auc['Methodology'].apply(lambda x: f"{x} (Smallest Error - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

    # Concatenate the three dataframes
    df_error_concat = pd.concat([df_error_menor_erro, df_error_maior_auc, df_error_menor_erro_auc])

    # Do the same for AUC
    df_auc_menor_erro = best_rmse_df_filtered[['Experiment', 'Methodology', 'AUC']].copy()
    df_auc_maior_auc = best_auc_df_filtered[['Experiment', 'Methodology', 'AUC']].copy()
    df_auc_menor_erro_auc = best_combined_df_filtered[['Experiment', 'Methodology', 'AUC']].copy()

    # Add the metric name to each dataframe in Methodology (except for Raw Scoring Function and Simple consensus)
    df_auc_menor_erro.loc[:, 'Methodology'] = df_auc_menor_erro['Methodology'].apply(lambda x: f"{x} (Smallest Error)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
    df_auc_maior_auc.loc[:, 'Methodology'] = df_auc_maior_auc['Methodology'].apply(lambda x: f"{x} (Biggest AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)
    df_auc_menor_erro_auc.loc[:, 'Methodology'] = df_auc_menor_erro_auc['Methodology'].apply(lambda x: f"{x} (Smallest Error - AUC)" if x not in ['Raw Scoring Function', 'Simple consensus'] else x)

    # Concatenate the three dataframes
    df_auc_concat = pd.concat([df_auc_menor_erro, df_auc_maior_auc, df_auc_menor_erro_auc])

    # Sort the concatenated dataframes by Methodology
    df_error_concat.sort_values('Methodology', inplace=True)
    df_auc_concat.sort_values('Methodology', inplace=True)

    # Put the Raw Scoring Function and Simple consensus at the beginning of the dataframes
    df_error_concat = pd.concat([df_error_concat[df_error_concat['Methodology'] == 'Raw Scoring Function'], df_error_concat[df_error_concat['Methodology'] == 'Simple consensus'], df_error_concat[df_error_concat['Methodology'] != 'Raw Scoring Function'], df_error_concat[df_error_concat['Methodology'] != 'Simple consensus']])
    df_auc_concat = pd.concat([df_auc_concat[df_auc_concat['Methodology'] == 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Methodology'] == 'Simple consensus'], df_auc_concat[df_auc_concat['Methodology'] != 'Raw Scoring Function'], df_auc_concat[df_auc_concat['Methodology'] != 'Simple consensus']])

    for m in to_remove:
        df_error_concat = df_error_concat[~df_error_concat['Methodology'].str.startswith(m)]
        df_auc_concat = df_auc_concat[~df_auc_concat['Methodology'].str.startswith(m)]

    return df_error_concat, df_auc_concat

def plot_boxplot_violinplot_clean(df_error_concat: pd.DataFrame, df_auc_concat: pd.DataFrame, n_trials: int, palette_colour: str, mean_rmse: float, mean_auc: float) -> None:
    # Set the font size
    plt.rcParams['font.size'] = 10

    # Set the metrics
    metrics = ['(Smallest Error)', '(Biggest AUC)', '(Smallest Error - AUC)']

    for metric in metrics:
        # Filter the dataframes and exclude "Simple consensus"
        aux_df_error_concat = df_error_concat[
            (df_error_concat['Methodology'].str.endswith(metric, na=False)) |
            (df_error_concat['Methodology'] == 'Raw Scoring Function')
        ]
        aux_df_auc_concat = df_auc_concat[
            (df_auc_concat['Methodology'].str.endswith(metric, na=False)) |
            (df_auc_concat['Methodology'] == 'Raw Scoring Function')
        ]

        # Set the aux metric (Remove parentheses from the metric)
        aux_metric = metric.replace('(', '').replace(')', '')

        # Remove the metric string (with its previous space) from the Methodology column
        aux_df_error_concat['Methodology'] = aux_df_error_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))
        aux_df_auc_concat['Methodology'] = aux_df_auc_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))

        # Remake the color mapping for the concatenated dataframes
        color_mapping_error = {
            method: color for method, color in zip(
                aux_df_error_concat['Methodology'].unique(), 
                sns.color_palette(palette_colour, 
                n_colors=aux_df_error_concat['Methodology'].nunique())
            )
        }
        color_mapping_auc = {
            method: color for method, color in zip(
                aux_df_auc_concat['Methodology'].unique(), 
                sns.color_palette(palette_colour, 
                n_colors=aux_df_auc_concat['Methodology'].nunique())
            )
        }

        for plot_type in ['boxplot', 'violin']:
            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

            for plot, ax in zip(['RMSE', 'AUC'], [ax1, ax2]):
                data = aux_df_error_concat if plot == 'RMSE' else aux_df_auc_concat
                color_mapping = color_mapping_error if plot == 'RMSE' else color_mapping_auc

                if plot_type == 'boxplot':
                    sns.boxplot(
                        data=data, 
                        x='Methodology', 
                        y=plot, 
                        palette=color_mapping,
                        showfliers=False,
                        ax=ax
                    )
                else:
                    sns.violinplot(
                        data=data, 
                        x='Methodology', 
                        y=plot, 
                        palette=color_mapping,
                        ax=ax
                    )

                if plot == 'RMSE':
                    ax.axhline(mean_rmse, color='red', linestyle='--', label='Mean RMSE')
                else: # Is AUC
                    ax.axhline(mean_auc, color='blue', linestyle='--', label='Mean AUC')

                # Add minor grid lines
                ax.grid(True)
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

                # Get the positions of the boxes
                box_positions = range(len(data['Methodology'].unique()))

                # Highlight the Raw Scoring Function methodology
                if 'Raw Scoring Function' in data['Methodology'].unique():
                    method_pos = data['Methodology'].unique().tolist().index('Raw Scoring Function')
                    ax.axvspan(method_pos - 0.5, method_pos + 0.5, color='cyan', alpha=0.3)

                # Add shaded areas for methodologies that start with NN, XGB, and Transformer
                nn_methods = [method for method in data['Methodology'].unique() if method.startswith('NN')]
                xgb_methods = [method for method in data['Methodology'].unique() if method.startswith('XGB')]
                transformer_methods = [method for method in data['Methodology'].unique() if method.startswith('Transformer')]

                for method_group, color in zip([nn_methods, xgb_methods, transformer_methods], ['lightblue', 'lightgreen', 'lightcoral']):
                    for method in method_group:
                        method_pos = data['Methodology'].unique().tolist().index(method)
                        ax.axvspan(method_pos - 0.5, method_pos + 0.5, color=color, alpha=0.3)
                
                # Add the legend to the entire figure
                ax.legend()

            fig.suptitle(f'{aux_metric} for the {n_trials} best training results', fontsize=16)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
            plt.tight_layout()
            plt.savefig(f'plots/Experiments_{plot_type}_{aux_metric}_{n_trials}.png', bbox_inches='tight')

    plt.close('all')

def plot_boxplot_violinplot(df_error_concat: pd.DataFrame, df_auc_concat: pd.DataFrame, n_trials: int, palette_colour: str) -> None:
    # Set the font size
    plt.rcParams['font.size'] = 10 # type: ignore

    # Set the metrics
    metrics = ['(Smallest Error)', '(Biggest AUC)', '(Smallest Error - AUC)']

    for metric in metrics:
        # Filter the dataframes
        aux_df_error_concat = df_error_concat[df_error_concat['Methodology'].str.endswith(metric, na=False) | (df_auc_concat['Methodology'] == 'Raw Scoring Function') | (df_auc_concat['Methodology'] == 'Simple consensus')]
        aux_df_auc_concat = df_auc_concat[df_auc_concat['Methodology'].str.endswith(metric, na=False) | (df_auc_concat['Methodology'] == 'Raw Scoring Function') | (df_auc_concat['Methodology'] == 'Simple consensus')]

        # Set the aux metric (Remove parentheses from the metric)
        aux_metric = metric.replace('(', '').replace(')', '')

        # Remove the metric string (with its previous space) from the Methodology column
        aux_df_error_concat.loc[:, 'Methodology'] = aux_df_error_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))
        aux_df_auc_concat.loc[:, 'Methodology'] = aux_df_auc_concat['Methodology'].apply(lambda x: x.replace(f' {metric}', ''))

        # Remake the color mapping for the concatenated dataframes
        color_mapping_error = {
            method: color for method, color in zip(
                aux_df_error_concat['Methodology'].unique(), 
                sns.color_palette(palette_colour, 
                n_colors=aux_df_error_concat['Methodology'].nunique())
            )
        }
        color_mapping_auc = {
            method: color for method, color in zip(
                aux_df_auc_concat['Methodology'].unique(), 
                sns.color_palette(palette_colour, 
                n_colors=aux_df_auc_concat['Methodology'].nunique())
            )
        }

        for plot_type in ['boxplot', 'violin']:
            plt.close('all')

            plt.figure(figsize=(15, 30))  # Adjust the size of the entire figure

            # Create subplots with shared x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10)) # type: ignore

            for i, plot in enumerate(['RMSE', 'AUC']):
                ax = ax1 if plot == 'RMSE' else ax2
                if plot_type == 'boxplot':
                    sns.boxplot(
                        data=aux_df_error_concat if plot == 'RMSE' else aux_df_auc_concat, 
                        x='Methodology', 
                        y=plot, 
                        palette=color_mapping_error if plot == 'RMSE' else color_mapping_auc,
                        showfliers=False,
                        ax=ax,
                        hue='Methodology',
                        legend=False
                    )
                else:
                    sns.violinplot(
                        data=aux_df_error_concat if plot == 'RMSE' else aux_df_auc_concat, 
                        x='Methodology', 
                        y=plot, 
                        palette=color_mapping_error if plot == 'RMSE' else color_mapping_auc,
                        ax=ax,
                        hue='Methodology',
                        legend=False
                    )

                ax.grid(True)
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')
                
                # Get the positions of the boxes
                box_positions = range(len(aux_df_error_concat['Methodology'].unique()))
                
                # Add shaded area between the markers
                ax.axvspan(
                    box_positions[aux_df_error_concat['Methodology'].unique().tolist().index('Raw Scoring Function')] - 0.5, 
                    box_positions[aux_df_error_concat['Methodology'].unique().tolist().index('Simple consensus')] - 0.5, 
                    color='cyan', alpha=0.3
                )
                ax.axvspan(
                    box_positions[df_error_concat['Methodology'].unique().tolist().index('Raw Scoring Function')] + 0.5, 
                    box_positions[df_error_concat['Methodology'].unique().tolist().index('Simple consensus')] + 0.5, 
                    color='lime', alpha=0.3
                )

                # Add shaded areas for methodologies that start with NN, XGB, and Transformer
                nn_methods = [method for method in aux_df_error_concat['Methodology'].unique() if method.startswith('NN')]
                xgb_methods = [method for method in aux_df_error_concat['Methodology'].unique() if method.startswith('XGB')]
                transformer_methods = [method for method in aux_df_error_concat['Methodology'].unique() if method.startswith('Transformer')]

                for method_group, color in zip([nn_methods, xgb_methods, transformer_methods], ['lightblue', 'lightgreen', 'lightcoral']):
                    for method in method_group:
                        method_pos = aux_df_error_concat['Methodology'].unique().tolist().index(method)
                        ax.axvspan(method_pos - 0.5, method_pos + 0.5, color=color, alpha=0.3)

            # Add Title to the entire figure
            fig.suptitle(f'{aux_metric} for the {n_trials} best training results', fontsize=16) # type: ignore
            
            # Rotate x-axis labels for both subplots
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

            # Use tight_layout to adjust the spacing
            plt.tight_layout()

            plt.savefig(f'plots/Experiments_{plot_type}_{aux_metric}_{n_trials}.png', bbox_inches='tight')

    plt.close('all')

def plot_barplots(best_rmse_df_filtered: pd.DataFrame, best_auc_df_filtered: pd.DataFrame, best_combined_df_filtered: pd.DataFrame, n_trials: int, color_mapping: dict) -> None:
    # Define the plotting information
    plotting_info = [
        ('RMSE', 'RMSE', 'RMSE', True), 
        ('AUC', 'AUC', 'AUC', False), 
        ('RMSE-AUC', 'combined_metric', 'Combined Metric', True)
    ]

    for metric_name, y_column, ylabel, ascending in plotting_info:
        plt.figure(figsize=(20, 8))  # Reset figure for each plot type

        for i, (metric, df) in enumerate([
                ('RMSE', best_rmse_df_filtered), 
                ('AUC', best_auc_df_filtered), 
                ('RMSE-AUC', best_combined_df_filtered)
            ]):
            plt.subplot(1, 3, i+1)
            
            # Calculate the mean values for each methodology and sort by the mean
            df_means = df.groupby('Methodology')[y_column].mean().reset_index()
            df_sorted = df.merge(df_means, on='Methodology', suffixes=('', '_mean'))
            df_sorted = df_sorted.sort_values(by=f'{y_column}_mean', ascending=ascending)
            
            # Sort the color mapping according to the sorted methodologies
            sorted_methodologies = df_sorted['Methodology'].unique()
            color_mapping_sorted = {method: color_mapping[method] for method in sorted_methodologies}

            sns.barplot(
                data=df_sorted, 
                x='Methodology', 
                y=y_column, 
                palette=color_mapping_sorted, 
                hue='Methodology',
                hue_order=sorted_methodologies,  # Keep color order intact
                dodge=False
            )
            
            plt.title(f'{metric} for the best {n_trials} training results')
            plt.xticks(rotation=90)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

        # Add the title to the entire figure
        plt.suptitle(ylabel, fontsize=16)

        # Use tight_layout to adjust the spacing
        plt.tight_layout()

        # Save each figure to a separate file
        plt.savefig(f'plots/Experiments_{y_column}_barplot_{n_trials}.png', bbox_inches='tight')

        plt.close('all')

    return None

def correlation_analysis(results_df: pd.DataFrame, final_metrics: pd.DataFrame, n_trials: int, error_threshold: float = 1.5) -> None:
    # Create the dataframe from the results_df with the columns Methodology, RMSE, and AUC (RMSE will be the best_combined_value, AUC will be the best_combined_auc and Methodology will be the study_type)
    corr_data_df = results_df[['study_name', 'study_type', 'best_combined_value', 'best_combined_auc', 'best_combined_metric']].copy()
    corr_data_df.rename(columns={'study_type': 'Methodology', 'best_combined_value': 'RMSE', 'best_combined_auc': 'AUC', 'best_combined_metric': 'combined_metric'}, inplace=True)

    # Create the dataframe with data for the correlation plot (concatenation between results_df and final_metrics)
    corr_data_df = pd.concat([corr_data_df, final_metrics], axis=0)

    # Filter according to the error threshold
    corr_data_df_filtered = corr_data_df[corr_data_df['RMSE'] <= error_threshold]

    # Compute the correlation between RMSE and AUC for each methodology and the entire dataset
    correlation_df = pd.DataFrame(columns=['Methodology', 'Correlation'])
    correlation_df.loc[0] = ['All', corr_data_df_filtered['RMSE'].corr(corr_data_df_filtered['AUC'])]

    for methodology in corr_data_df_filtered['Methodology'].unique():
        correlation_df.loc[len(correlation_df)] = [
            methodology,
            best_rmse_df_filtered[best_rmse_df_filtered['Methodology'] == methodology]['RMSE'].corr(
                best_rmse_df_filtered[best_rmse_df_filtered['Methodology'] == methodology]['AUC']
            )
        ]

    # Sort the data by the correlation values from highest to lowest
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    # Assign a color to All in the color mapping (new color)
    color_mapping['All'] = (0.5, 0.5, 0.5)

    print("Saving the correlation plot...")

    # Plot the correlation between RMSE and AUC for each methodology and the entire dataset in a barplot
    plt.figure(figsize=(20, 8))

    sns.barplot(data=correlation_df, x='Methodology', y='Correlation', palette=color_mapping, hue='Methodology', dodge=False)

    plt.title(f'Correlation between RMSE and AUC for the {n_trials} best training results')
    plt.xticks(rotation=90)
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

    # Use tight_layout to adjust the spacing
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(f'plots/Experiments_Correlation_barplot_{n_trials}.png', bbox_inches='tight')

    plt.close('all')

    return None

def plot_bar_with_significance_metrics(df, metrics = list[str], n_columns: int = 4):
    # Group the data by methodology, excluding "Simple Consensus"
    methodology_groups = df[df["Methodology"] != "Simple consensus"].groupby("Methodology")

    # Loop through each metric
    for metric in metrics:
        print(f"Metric: {metric}")
        
        # Determine the number of rows needed
        num_methods = len(methodology_groups)
        rows = math.ceil(num_methods / n_columns)
        
        # Initialize a multi-plot figure
        fig, axes = plt.subplots(nrows=rows, ncols=n_columns, figsize=(5 * n_columns, 5 * rows))
        axes = axes.flatten() # Flatten to easily iterate, regardless of row/column structure
        
        # List to collect data for variance and homogeneity tests
        metric_data_by_methodology = []

        # Loop through each methodology group and plot in the respective subplot
        for i, (method, group) in enumerate(methodology_groups):
            print(f"Methodology: {method}")

            # Extract the metric data for testing
            metric_data = group[metric].dropna()
            metric_data_by_methodology.append(metric_data)

            # Normality Tests
            shapiro_test = stats.shapiro(metric_data)
            anderson_test = stats.anderson(metric_data, dist='norm')
            print(f"  Shapiro-Wilk Test p-value for {method}: {shapiro_test.pvalue}")
            print(f"  Anderson-Darling Test statistic for {method}: {anderson_test.statistic}")

            # Plot QQ plot on the appropriate subplot
            stats.probplot(metric_data, dist="norm", plot=axes[i])
            axes[i].set_title(f"{method} - {metric}")

        # Remove any empty subplots if the total number of methodologies is not a perfect multiple of n_columns
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j]) # type: ignore

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"plots/multiplot_qq_{metric}_{n_trials}.png")
        plt.close('all')
        
        # Variance for each methodology
        variances = methodology_groups[metric].var()
        print(f"Variance for each methodology ({metric}):")
        print(variances)
        
        # Levene's Test for homogeneity of variances
        levene_test = stats.levene(*metric_data_by_methodology)
        print(f"Levene's Test statistic for {metric}:", levene_test.statistic)
        print(f"Levene's Test p-value for {metric}:", levene_test.pvalue)

        # Bartlett's Test for homogeneity of variances (use only if data is normally distributed)
        bartlett_test = stats.bartlett(*metric_data_by_methodology)
        print(f"Bartlett's Test statistic for {metric}:", bartlett_test.statistic)
        print(f"Bartlett's Test p-value for {metric}:", bartlett_test.pvalue)
    
    return None

def perform_welch_anova_and_games_howell_posthoc_tests(df, n_trials):

    # Welch's ANOVA for AUC
    welch_anova_auc = pg.welch_anova(dv='AUC', between='Methodology', data=df)
    print("Welch's ANOVA for AUC:")
    print(welch_anova_auc)
    welch_anova_auc.to_csv(f"csvs/welch_anova_auc_{n_trials}.csv", index=False)

    # Welch's ANOVA for RMSE
    welch_anova_rmse = pg.welch_anova(dv='RMSE', between='Methodology', data=df)
    print("Welch's ANOVA for RMSE:")
    print(welch_anova_rmse)
    welch_anova_rmse.to_csv(f"csvs/welch_anova_rmse_{n_trials}.csv", index=False)

    # Games-Howell
    ################

    # Games-Howell post-hoc test for AUC
    games_howell_auc = pg.pairwise_gameshowell(dv='AUC', between='Methodology', data=df)
    print("Games-Howell post-hoc test results for AUC:")
    print(games_howell_auc)

    # Games-Howell post-hoc test for RMSE
    games_howell_rmse = pg.pairwise_gameshowell(dv='RMSE', between='Methodology', data=df)
    print("\nGames-Howell post-hoc test results for RMSE:")
    print(games_howell_rmse)

    # Optionally, save the results to CSV for review
    games_howell_auc.to_csv(f"csvs/games_howell_posthoc_AUC_{n_trials}.csv", index=False)
    games_howell_rmse.to_csv(f"csvs/games_howell_posthoc_RMSE_{n_trials}.csv", index=False)
    print("\nGames-Howell post-hoc results saved as CSV files.")
    
    # Bar plot for AUC/RMSE means with significance annotations
    plot_bar_with_significance(games_howell_auc, metric="AUC", y_col="diff")
    plot_bar_with_significance(games_howell_rmse, metric="RMSE", y_col="diff")

    # Heatmap for p-values from Games-Howell for AUC/RMSE
    plot_heatmap(games_howell_auc, "Games-Howell p-values for AUC", metric="AUC")
    plot_heatmap(games_howell_rmse, "Games-Howell p-values for RMSE", metric="RMSE")
    
    return None

print("Starting the analysis of the results.")

base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
df_path: str = f"{base_path}/OCDocker.csv.gz"

print(f"Reading the input file: '{df_path}'")

# Load the data
final_metrics = load_data(df_path)

print("Setting the lists of the studies to be analyzed.")
snames, nn_len, xgb_len = get_all_lists()

# Fetch all the studies results
user = "ocdocker"
password = "@Kp3sRv9t@"
host = "localhost"
port = 3306
db = "optimization"

# Set the storage
storage = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db}"

# Setup dirs
setup_dirs()

#n_trials = 10

for n_trials in [1, 5, 10, 50, 100, 500]: # TODO: Check the behaviour for 50, 100, and 500 trials (it is strange)
    print(f"Recovering the indexes for the NN-AE and XGB-GA methodologies for {n_trials} trials.")
    nn_ae_start, nn_ae_end, xgb_ga_start, xgb_ga_end = get_ae_xgb_indices(nn_len, xgb_len, n_trials)

    print(f"Fetching the results for the {n_trials} best training results.")
    best_rmse_df_filtered, \
    best_auc_df_filtered, \
    best_combined_df_filtered, \
    results_df, \
    min_auc, \
    max_auc, \
    min_error, \
    max_error, \
    error_range, \
    auc_range = get_study_data(
        snames, 
        storage, 
        nn_ae_start = nn_ae_start, 
        nn_ae_end = nn_ae_end, 
        xgb_ga_start = xgb_ga_start, 
        xgb_ga_end = xgb_ga_end
    )

    # Set the color mapping
    color_mapping, palette_colour = set_color_mapping(best_combined_df_filtered)

    print("Plotting the scatterplot to allow visual comparison of the methodologies for AUC and RMSE")

    plot_scatterplot(best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered, n_trials, color_mapping, min_auc, max_auc, min_error, max_error, error_range, auc_range, alpha = 0.9)
    plot_single_scatterplot(best_combined_df_filtered, n_trials, color_mapping, min_auc, max_auc, min_error, max_error, error_range, auc_range, alpha = 0.9)

    print("Processing the dataframes to create some plots.")

    df_error_concat, df_auc_concat = separate_dfs(best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered)
    
    print(f"Plotting the barplot and violinplot to allow visual comparison of the methodologies for AUC and RMSE")

    #plot_boxplot_violinplot(df_error_concat, df_auc_concat, n_trials, palette_colour = palette_colour)
    plot_boxplot_violinplot_clean(df_error_concat, df_auc_concat, n_trials, palette_colour = palette_colour, mean_rmse = final_metrics[final_metrics['study_name'] == 'mean']['RMSE'].iloc[0], mean_auc = final_metrics[final_metrics['study_name'] == 'mean']['AUC'].iloc[0])
    
    print("Plotting the RMSE, AUC, and combined metric barplots...")

    plot_barplots(best_rmse_df_filtered, best_auc_df_filtered, best_combined_df_filtered, n_trials, color_mapping)

    print("Performing the correlation calculations...")

    correlation_analysis(results_df, final_metrics, n_trials, error_threshold = 1.5)

    print("Performing the normality tests...")

    plot_bar_with_significance_metrics(best_combined_df_filtered, metrics = ["AUC", "RMSE"], n_columns = 4)

    print("Performing the Welch's ANOVA and Games-Howell post-hoc tests...")

    perform_welch_anova_and_games_howell_posthoc_tests(best_combined_df_filtered, n_trials)

# PCA Feature Importance Analysis
# TODO: Organize this

import optuna
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to load the saved PCA model
def load_pca_model(pickle_file):
    with open(pickle_file, 'rb') as f:
        pca_model = pickle.load(f)
    return pca_model

# Function to compute and visualize variable importance
def compute_variable_importance(pca_model, data, pca_number, n_most_important=None, min_importance=None):
    # Step 1: Get loadings (how much each feature contributes to each principal component)
    loadings = pca_model.components_  # No need to transpose here
    
    # Step 2: Get explained variance
    explained_variance = pca_model.explained_variance_ratio_
    
    # Step 3: Calculate squared loadings weighted by explained variance
    squared_loadings = np.square(loadings)  # Square of the loadings

    # Reshape explained_variance to be broadcast correctly across the loadings
    weighted_importance = squared_loadings * explained_variance[:, np.newaxis]
    
    # Step 4: Sum across components to get the total importance of each variable
    total_importance = np.sum(weighted_importance, axis=0)
    
    # Step 5: Create a DataFrame for easier visualization
    importance_df = pd.DataFrame({'Feature': data.columns, 'Importance': total_importance})
    
    # Step 6: Apply filtering based on n_most_important or min_importance
    if min_importance is not None:
        importance_df = importance_df[importance_df['Importance'] > min_importance]
    
    if n_most_important is not None:
        importance_df = importance_df.nlargest(n_most_important, 'Importance')
    
    # Sort by importance for better visualization
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Step 7: Plot the variable importance
    plt.figure(figsize=(10,6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Variable Importance based on PCA {pca_number}')
    plt.gca().invert_yaxis()  # To display the most important feature at the top
    plt.tight_layout()
    plt.savefig(f'plots/variable_importance_{pca_number}.png')
    
    return importance_df

# Function to analyze and visualize variable importance
def analyze_pca_model(pca_model, data, pca_number):
    # Step 1: Get loadings (how much each feature contributes to each principal component)
    loadings = pca_model.components_.T  # Transpose to get features in rows
    
    # Step 2: Create a DataFrame for easy interpretation
    components_df = pd.DataFrame(loadings, 
                                 columns=[f'PC{i+1}' for i in range(loadings.shape[1])], 
                                 index=data.columns)
    
    # Step 3: Plot the loadings
    components_df.plot(kind='bar', figsize=(10,6))
    plt.title('Feature Importance for Each Principal Component')
    plt.ylabel('Loading Value')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(f'plots/feature_importance_{pca_number}.png')
    
    # Step 4: Print explained variance
    explained_variance = pca_model.explained_variance_ratio_
    print("Explained variance ratio by each component:", explained_variance)
    
    # Plot the explained variance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance)+1), explained_variance)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.savefig(f'plots/explained_variance_{pca_number}.png')

# Filter the pdbbind data to remove unwanted columns ["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns
pdbbind_data_filter = pdbbind_data.drop(columns=["receptor", "ligand", "name", "type", "db", "experimental"] + score_columns)

# Optional filters
n_most_important = 20  # Show top n most important features (set to None to show all)
min_importance = None  # Show features with importance greater than x (set to None to show all)

for pca_type in [80, 85, 90, 95]:
    # Path to save the PCA model
    pca_model_file = f"{pca_path}/pca{pca_type}.pkl"
    
    # Load the model
    pca_model = load_pca_model(pca_model_file)
    
    # Compute and display the variable importance with filters
    variable_importance_df = compute_variable_importance(pca_model, pdbbind_data_filter, pca_type,
                                                         n_most_important=n_most_important, 
                                                         min_importance=min_importance)

# After a minucious analysis of the performance of the experiments, we have decided to carry on the Autoencoder + Neural Network methodology to the next phase of the project.
# We will now proceed with the ablative analysis of the Autoencoder + Neural Network methodology to understand the impact of each component in the final performance of the model.
# Firstly we will perform the analysis of the variable importance of the input features in the Autoencoder alone

# Create the autoencoder studies variable to set which are the studies that we want to analyze
autoencoder_studies = [
    "AO_Optimization_6",
    "AO_Optimization_7",
    "AO_Optimization_8",
    "AO_Optimization_9",
    "AO_Optimization_10"
]

# To get the best result, we should look into the best study from the final optimization results (NN)
autoencoder_nn_studies = [
    "NN_Optimization_6",
    "NN_Optimization_7",
    "NN_Optimization_8",
    "NN_Optimization_9",
    "NN_Optimization_10"
]

# Find the best combination of hyperparameters for the Autoencoder from the optimization results (fetch the best study from the optimization results)
autoencoder_optimization_results = ocstudy.analyze_studies(autoencoder_nn_studies, storage=storage, n_trials=1)

# Good, now we get the number of the best study (the one with the smallest 'best_combined_value')
best_study_name = autoencoder_optimization_results.loc[autoencoder_optimization_results['best_combined_value'].idxmin()]['study_name']
best_study_number = int(best_study_name.split('_')[-1])

# Get the best AO study with the same number
best_ao_study_name = f"AO_Optimization_{best_study_number}"

# Fetch its data from the database
best_ao_study = optuna.load_study(study_name = best_ao_study_name, storage = storage)

# Get the best trial (smallest value, there is no AUC here)
best_ao_params = best_ao_study.best_params


# Autoencoder Feature Importance Analysis
# TODO: Organize this

# Imports
from OCDocker.OCScore.Dimensionality.AutoencoderOptimizer import Autoencoder, AutoencoderDataset, DataLoader
import math
import torch
import random

def parse_activation_func(encoder_activation_str, best_ao_params):
    activation_functions = [torch.nn.GELU, torch.nn.LeakyReLU, torch.nn.Mish, torch.nn.ReLU, torch.nn.SELU, torch.nn.Identity]
    activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']

    if encoder_activation_str == 'LeakyReLU':
        pre_encoder_params = {
            f'negative_slope_encoder': best_ao_params['negative_slope_encoder']
        }
        encoder_params = {k.replace('_encoder', ''): v for k, v in pre_encoder_params.items()}
        encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)](**encoder_params)
    elif encoder_activation_str == 'GELU':
        pre_encoder_params = {
            f'approximate_encoder': best_ao_params['approximate_encoder']
        }
        encoder_params = {k.replace('_encoder', ''): v for k, v in pre_encoder_params.items()}
        encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)](**encoder_params)
    else:
        encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)]()

    return encoder_activation_fn

def set_random_seed(random_seed = 42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Set the seed for CPU
    torch.manual_seed(random_seed)
    # Set the seed for GPU
    torch.cuda.manual_seed_all(random_seed)
    
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_autoencoder(model, criterion, loader = None):
    set_random_seed()
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, _ in loader: # type: ignore
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            total_loss += loss.item()

    average_loss = total_loss / len(loader) # type: ignore

    rmse = np.sqrt(average_loss)
    
    return rmse

def train_autoencoder(model, optimizer, criterion, clip_grad, epochs):
    # Set the best validation and training rmse to infinity
    best_validation_rmse = np.inf
    best_train_rmse = np.inf

    model.train()
    for _ in range(epochs):

        running_loss = 0.0

        for data, _ in train_loader: # type: ignore
            optimizer.zero_grad()
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Clip the gradients
            optimizer.step()

            running_loss += loss.item()
        
        average_loss = running_loss / len(train_loader) # type: ignore
        rmse = np.sqrt(average_loss)

        # Validation phase
        if validation_loader is not None:
            val_rmse = evaluate_autoencoder(model, criterion, validation_loader)

            # Check for improvement
            if val_rmse < best_validation_rmse:
                best_train_rmse = rmse
                best_validation_rmse = val_rmse

        print(f'Test Loss: {average_loss}')
        print(f'Test RMSE: {rmse}')
        
        return best_train_rmse, best_validation_rmse, model

# Separate the X and y data
data = ocscoredata.load_data(
    base_models_folder = f"{base_path}/models",
    storage_id = best_study_number,
    df_path = df_path,
    optimization_type = "NN"
)

# Set the explained variance as 0.95
explained_variance: float = 0.95

# Compute the singular values for AO_X_train
singular_values = np.linalg.svd(data['X_train'], compute_uv = False)

# Compute the explained variance ratio
explained_variance_ratio = singular_values**2 / np.sum(singular_values**2)

# Compute the cumulative explained variance ratio
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Compute the number of components that explain 95% of the variance
n_components = np.argmax(cumulative_explained_variance_ratio >= explained_variance) + 1

# Get the number of dimensions for the encoding layer and round up to the nearest power of 2 + 1
encoding_dims = ( # Size should be the same size or smaller than the number of features to explain the desired variance
    max(2 ** math.ceil(math.log2(n_components / 2) - 1), 4), # Minimum value
    n_components
)

activation_function_0_encoder = parse_activation_func(best_ao_params['activation_function_0_encoder'], best_ao_params)
activation_function_0_decoder = parse_activation_func(best_ao_params['activation_function_0_decoder'], best_ao_params)

best_ao_model = Autoencoder(
    data['X_train'].shape[1], 
    best_ao_params['n_units_layer_0_encoder'], 
    activation_function_0_encoder, 
    activation_function_0_decoder
).to(torch.device('cuda'))

# Choose the optimizer
optimizer_name = best_ao_params['optimizer']
weight_decay = best_ao_params['weight_decay']
lr = best_ao_params['lr']
clip_grad = best_ao_params['clip_grad']
epochs = best_ao_params['epochs']
random_seed = 42

optimizer = getattr(torch.optim, optimizer_name)(best_ao_model.parameters(), lr = lr, weight_decay = weight_decay)

criterion = torch.nn.MSELoss()

train_loader = DataLoader(
    dataset = AutoencoderDataset(torch.tensor(np.asarray(data['X_train']), dtype=torch.float32).to(torch.device('cuda'))), 
    batch_size = best_ao_params['batch_size'],
    shuffle = True
)

test_loader = DataLoader(
    dataset = AutoencoderDataset(torch.tensor(np.asarray(data['X_test']), dtype=torch.float32).to(torch.device('cuda'))), 
    batch_size = best_ao_params['batch_size']
)

validation_loader = DataLoader(
    dataset = AutoencoderDataset(torch.tensor(np.asarray(data['X_val']), dtype=torch.float32).to(torch.device('cuda'))), 
    batch_size = best_ao_params['batch_size']
)

best_train_rmse, best_validation_rmse, model = train_autoencoder(best_ao_model, optimizer, criterion, clip_grad, epochs) # type: ignore

# To get the permutation_importance
from tqdm import tqdm

# Define a function to calculate reconstruction error
def compute_reconstruction_error(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    criterion = torch.nn.MSELoss()  # Ensure the criterion is defined

    with torch.no_grad():
        for data, _ in data_loader:  # Unpack the tuple
            data = data.to(torch.device('cuda'))  # Move data to the appropriate device
            reconstructed = model(data)  # Forward pass
            loss = criterion(reconstructed, data)  # MSE Loss
            total_loss += loss.item()
    
    return np.sqrt(total_loss / len(data_loader))

# Get the validation data
validation_data = torch.tensor(np.asarray(data['X_val']), dtype=torch.float32).to(torch.device('cuda'))

# Calculate the original reconstruction error
original_error = compute_reconstruction_error(best_ao_model, validation_loader)

# Calculate permutation importance
def permutation_importance_custom(model, data, original_error, n_repeats=30):
    importances = []
    for i in tqdm(range(data.shape[1]), desc='Calculating Permutation Importance'):
        permuted_data = data.clone()  # Clone the data for each feature permutation
        save_column = permuted_data[:, i].clone()  # Save original column
        permuted_errors = []
        
        for _ in tqdm(range(n_repeats), desc='Permutation Iteration', leave=False):
            # Permute the feature in the cloned data
            permuted_data[:, i] = permuted_data[torch.randperm(permuted_data.size(0)), i]

            # Calculate error with permuted feature
            permuted_error = compute_reconstruction_error(model, DataLoader(
                dataset=AutoencoderDataset(permuted_data),
                batch_size=best_ao_params['batch_size'],
                shuffle=False
            ))
            permuted_errors.append(permuted_error)

        # Calculate importance as the difference between permuted and original errors
        importance = np.mean(permuted_errors) - original_error
        importances.append(importance)

    return np.array(importances)

# Compute permutation importance
importances = permutation_importance_custom(best_ao_model, validation_data.cpu(), original_error)

# Get the feature names
feature_names = list(data['X_val'].columns)

# Pair feature names with their importance scores
feature_importances = list(zip(feature_names, importances))

# Sort the features by importance if needed
feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)
sorted_feature_names, sorted_importances = zip(*feature_importances_sorted)

# Print or analyze the sorted feature importances
for feature, importance in feature_importances_sorted:
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# Optionally visualize the importances
import matplotlib.pyplot as plt


def filter_and_visualize_importances(feature_names, importances, n_best=None, threshold=None):
    """
    Filter and visualize feature importances based on specified criteria.

    Parameters:
    - feature_names: List of feature names.
    - importances: List of corresponding importances.
    - n_best: Integer, number of top features to return. If None, include all.
    - threshold: Float, minimum importance value to include. If None, include all.

    Returns:
    - filtered_feature_names: List of feature names that meet the criteria.
    - filtered_importances: List of importances corresponding to the filtered features.
    """
    # Zip and sort the feature importances
    feature_importances_sorted = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Apply filtering
    filtered_importances = []
    filtered_feature_names = []

    for feature, importance in feature_importances_sorted:
        if (threshold is None or importance >= threshold) and (n_best is None or len(filtered_importances) < n_best):
            filtered_feature_names.append(feature)
            filtered_importances.append(importance)

    # Optionally visualize the filtered importances
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_feature_names, filtered_importances)
    plt.title('Filtered Permutation Importance of Features')
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Save the plot
    plt.tight_layout()  # Ensures the plot fits well
    plt.savefig('plots/filtered_permutation_importance.png')

    return filtered_feature_names, filtered_importances

# Example usage
filtered_names, filtered_importances = filter_and_visualize_importances(
    feature_names=sorted_feature_names,
    importances=sorted_importances,
    n_best=20,
    threshold=None
)

# Neural Network Autoencoder visualization

import optuna
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from OCDocker.OCScore.DNN.DNNOptimizer import CustomDataset, DNNOptimizer, NeuralNet
import OCDocker.OCScore.Utils.Data as ocscoredata
import math
import torch
import random

# Define the names
ao_study_name = f"AO_Optimization_{study_number}"
nn_study_name = f"NN_Optimization_{study_number}"
ablation_study_name = f"NN_Ablation_Optimization_1"

# Load the study
ao_study = optuna.load_study(study_name = ao_study_name, storage = storage)
ao_df = ao_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
ao_df = ao_df[ao_df['state'] == 'COMPLETE']

#best_ao_df = ao_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])
best_ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])

# Recreate the autoencoder object for the best trial based on the best_ao_df
best_ao_trial = best_ao_df.iloc[0]

# Select the trial by the best_ao_trial number
best_ao_trial = ao_study.trials[best_ao_trial.number]

# Pick the params from the best_ao_trial
autoencoder_params = best_ao_trial.params

# Load the study
ablation_study = optuna.load_study(study_name = ablation_study_name, storage = storage)
ablation_df = ablation_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
ablation_df = ablation_df[ablation_df['state'] == 'COMPLETE']

# Reset data index
ablation_df = ablation_df.reset_index(drop=True)

# Rename the columns
# value is the RMSE
# user_attrs_Feature_Mask is the Feature Mask
# user_attrs_AUC is the AUC
ablation_df = ablation_df.rename(columns={
        'value': 'RMSE',
        'user_attrs_Feature_Mask': 'Feature_Mask',
        'user_attrs_AUC': 'AUC'
    }
)

# Compute the score (RMSE - AUC)
ablation_df['score'] = ablation_df['RMSE'] - ablation_df['AUC']

best_ablation_df = ablation_df.sort_values(by=['score'], ascending=[True])

# Pick the user_attrs_Feature_Mask (Feature_Mask) from the best_ablation_trial
mask = best_ablation_df.iloc[0]['Feature_Mask']

# Convert the mask to a numpy array of 0s and 1s
mask = np.array([int(x) for x in mask])

## Define the Topology
##########################

# Load the study
nn_study = optuna.load_study(study_name = nn_study_name, storage = storage)
nn_df = nn_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
nn_df = nn_df[nn_df['state'] == 'COMPLETE']

nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']

best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])

best_nn_trial = best_nn_df.iloc[0]

best_nn_trial = nn_study.trials[best_nn_trial.number]

nn_params = best_nn_trial.params

base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
base_models_folder: str = f"{base_path}/models"

# Load the data
data = ocscoredata.load_data(
        base_models_folder = base_models_folder,
        storage_id = study_number,
        df_path = df_path,
        optimization_type = "NN",
        no_scores = False,
        only_scores = False,
        use_PCA = False,
        use_pdb_train = True,
        random_seed = 42
    )

# Build the entire model
neural2 = DNNOptimizer(
            data['X_train'], data['y_train'], 
            data['X_test'], data['y_test'], 
            data['X_val'], data['y_val'], 
            mask = mask,
            storage = storage,
            encoder_params = autoencoder_params,
            output_size = 1, 
            random_seed = 42,
            use_gpu = True, 
            verbose = False,
        )

neural = NeuralNet(
        data["X_train"].shape[1], 
        1, 
        autoencoder_params,
        nn_params,
        random_seed = 42,
        use_gpu = False,
        verbose = False,
        mask = mask
    )

model = neural.NN
from torch.utils.data import Dataset, DataLoader

loader = DataLoader(
    dataset = CustomDataset(data['X_train'], data['y_train']), # type: ignore
    batch_size = batch_size,
    shuffle = True
)

X_train = torch.tensor(np.asarray(data["X_train"]), dtype=torch.float32).to("cuda")
X_train = torch.tensor(np.asarray(data["X_train"]), dtype=torch.float32).to("cpu")

y = model(X_train)

import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("model_topology")

print("\nLogging model graph to TensorBoard...")
writer = SummaryWriter("runs/model")  # Specify log directory
writer.add_graph(model, X_train)
writer.close()

from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn

color_map: dict = defaultdict(dict)
color_map[nn.Linear]["fill"] = "#98FB98"
color_map[nn.ReLU]["fill"] = "#FFA07A"
color_map[nn.Sigmoid]["fill"] = "#FFD700"
color_map[nn.Tanh]["fill"] = "#87CEFA"
color_map[nn.Dropout]["fill"] = "#FF6347"
color_map[nn.BatchNorm1d]["fill"] = "#FFA500"
color_map[nn.BatchNorm2d]["fill"] = "#FFA500"
color_map[nn.BatchNorm3d]["fill"] = "#FFA500"
color_map[nn.Conv1d]["fill"] = "#FF4500"
color_map[nn.Conv2d]["fill"] = "#FF4500"
color_map[nn.Conv3d]["fill"] = "#FF4500"
color_map[nn.MaxPool1d]["fill"] = "#FFD700"
color_map[nn.MaxPool2d]["fill"] = "#FFD700"
color_map[nn.MaxPool3d]["fill"] = "#FFD700"
color_map[nn.AvgPool1d]["fill"] = "#FFD700"
color_map[nn.AvgPool2d]["fill"] = "#FFD700"
color_map[nn.AvgPool3d]["fill"] = "#FFD700"
color_map[nn.AdaptiveMaxPool1d]["fill"] = "#FFD700"
color_map[nn.AdaptiveMaxPool2d]["fill"] = "#FFD700"
color_map[nn.AdaptiveMaxPool3d]["fill"] = "#FFD700"
color_map[nn.AdaptiveAvgPool1d]["fill"] = "#FFD700"
color_map[nn.AdaptiveAvgPool2d]["fill"] = "#FFD700"
color_map[nn.AdaptiveAvgPool3d]["fill"] = "#FFD700"
color_map["output"]["fill"] = "blue"

img = visualtorch.graph_view(model, (data["X_train"].shape), color_map=color_map, to_file="test.png")