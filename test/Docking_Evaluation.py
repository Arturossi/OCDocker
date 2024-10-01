#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

import os

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

base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
df_path: str = f"{base_path}/OCDocker.csv.gz"

'''
# Load the DataFrame
df = pd.read_csv(df_path)

# Get the score columns based on the regex below
score_columns_regex = r"(VINA|SMINA|ODDT|PLANTS).*"
score_columns = [col for col in df.columns if re.match(score_columns_regex, col)]

# Normalize the columns
normalized_df = ocscoredata.norm_data(df[score_columns + ["experimental", ]], scaler = "standard", inplace = False)
'''

# Load the DataFrames
dudez_data, pdbbind_data, score_columns = ocscoredata.preprocess_df(df_path)

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
#endregion

# Fetch all the studies results
user = "ocdocker"
password = "@Kp3sRv9t@"
host = "localhost"
port = 3306
db = "optimization"

# Concatenate all the lists
snames = plain_nn_list + ao_nn_list + pca80_nn_list + pca85_nn_list + pca90_nn_list + pca95_nn_list + plain_xgb_list + ga_xgb_list + pca80_xgb_list + pca85_xgb_list + pca90_xgb_list + pca95_xgb_list + plain_trans_list + pca80_trans_list + pca85_trans_list + pca90_trans_list + pca95_trans_list

# Set the storage
storage = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db}"

# Fetch the results
results_df = ocstudy.analyze_studies(snames, storage = storage)

# Fix the study type of NN + AE and XGB + GA (we know that it has been set wrong)
results_df.loc[25:49, 'study_type'] = 'NN + AE'
results_df.loc[175:199, 'study_type'] = 'XGB + GA'

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

## Start plotting the results ##

# If the plots folder does not exist, create it
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plotting with the chosen palette and adjustments for marker and transparency
plt.figure(figsize=(20, 8))

# Palette
#palette_colour = "Set2"
#palette_colour = "Set3"
#palette_colour = "tab10"
palette_colour = "tab20"
#palette_colour = "colorblind"
#palette_colour = "pastel"
#palette_colour = "bright"
#palette_colour = "dark"
#palette_colour = "deep"
#palette_colour = "muted"
#palette_colour = "viridis"
#palette_colour = sns.color_palette(cc.glasbey, n_colors=best_combined_df['Methodology'].nunique())

# Set alpha value
alpha = 0.9

# Create a color mapping for methodologies
color_mapping = {method: color for method, color in zip(best_combined_df['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=best_combined_df['Methodology'].nunique()))}

# Set the error threshold
error_threshold = 1.5

# Get the rows with error greater than the error_threshold
best_rmse_df_filtered = best_rmse_df[best_rmse_df['RMSE'] <= error_threshold]
best_auc_df_filtered = best_auc_df[best_auc_df['RMSE'] <= error_threshold]
best_combined_df_filtered = best_combined_df[best_combined_df['RMSE'] <= error_threshold]

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
    plt.Line2D([0], [0], marker='o', color='w', label='AUC >= 0.5 (= AUC)', markerfacecolor='gray', markersize=10),
    plt.Line2D([0], [0], marker='*', color='w', label='AUC < 0.5 (= 1-AUC)', markerfacecolor='gray', markersize=10)
]

# Second legend for the colors (Methodology)
color_labels = df['Methodology'].unique().tolist()
color_handles = [plt.Line2D([0], [0], color=color_mapping[method], lw=4) for method in color_labels]

# Place the AUC shape legend at the bottom left
plt.figlegend(handles=shape_handles, labels=shape_labels, loc='lower left', bbox_to_anchor=(0.26, 0.03), ncol=1, title='AUC')

# Place the Methodology color legend at the bottom center
plt.figlegend(handles=color_handles, labels=color_labels, loc='lower center', bbox_to_anchor=(0.57, 0.03), ncol=4, title='Methodology')

# Use tight_layout to adjust the spacing, but leave the space for the legends under the plot
plt.tight_layout(rect=[0, 0.22, 1, 1])

plt.savefig('plots/Experiments.png', bbox_inches='tight', dpi=300)
#plt.show()
plt.close('all')

"""
# Create a boxplot for each method for the three metrics for Error and AUC
plt.figure(figsize=(20, 8))

for i, plot in enumerate(['Error (Smallest Error - AUC)', 'Error (Biggest AUC)', 'Error (Smallest Error - AUC)']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(
        data=df, 
        x='Methodology', 
        y=plot, 
        palette=color_mapping,
        showfliers=False,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{plot}')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_boxplot.png', bbox_inches='tight')
"""

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

# Remove all the methods that start with any of the following strings (empty list means no methods will be removed)
to_remove = []

for m in to_remove:
    df_error_concat = df_error_concat[~df_error_concat['Methodology'].str.startswith(m)]
    df_auc_concat = df_auc_concat[~df_auc_concat['Methodology'].str.startswith(m)]

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
    color_mapping_error = {method: color for method, color in zip(aux_df_error_concat['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_error_concat['Methodology'].nunique()))}
    color_mapping_auc = {method: color for method, color in zip(aux_df_auc_concat['Methodology'].unique(), sns.color_palette(palette_colour, n_colors=aux_df_auc_concat['Methodology'].nunique()))}

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

        # Add Title to the entire figure
        fig.suptitle(f'{aux_metric}', fontsize=16) # type: ignore
        
        # Rotate x-axis labels for both subplots
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        # Use tight_layout to adjust the spacing
        plt.tight_layout()

        plt.savefig(f'plots/Experiments_{plot_type}_{aux_metric}_concat.png', bbox_inches='tight')

plt.close('all')

# Make bar plots for the error and AUC for each metric (3 bars for each method in the same plot)
plt.figure(figsize=(20, 8))

for i, (metric, df) in enumerate([('RMSE', best_rmse_df_filtered), ('AUC', best_auc_df_filtered), ('RMSE-AUC', best_combined_df_filtered)]):
    plt.subplot(1, 3, i+1)
    sns.barplot(
        data=df, 
        x='Methodology', 
        y="RMSE", 
        palette=color_mapping,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{metric}')
    plt.xticks(rotation=90)
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Add the title to the entire figure
plt.suptitle('RMSE', fontsize=16)

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_rmse_barplot.png', bbox_inches='tight')

plt.close('all')

plt.figure(figsize=(20, 8))

for i, (metric, df) in enumerate([('RMSE', best_rmse_df_filtered), ('AUC', best_auc_df_filtered), ('RMSE-AUC', best_combined_df_filtered)]):
    plt.subplot(1, 3, i+1)
    sns.barplot(
        data=df, 
        x='Methodology', 
        y='AUC', 
        palette=color_mapping,
        hue='Methodology',
        legend=False
    )
    plt.title(f'{metric}')
    plt.xticks(rotation=90)
    plt.ylabel('AUC')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='darkgray')

# Add the title to the entire figure
plt.suptitle('AUC', fontsize=16)

# Use tight_layout to adjust the spacing
plt.tight_layout()

plt.savefig('plots/Experiments_auc_barplot.png', bbox_inches='tight')
