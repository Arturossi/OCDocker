#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the Transformer parameters model
using Optuna.

It is imported as:
import OCDocker.OCScore.Utils.StudyParser as ocstudy
'''

# Imports
###############################################################################

import optuna
import pandas as pd

import OCDocker.Toolbox.Printing as ocprint

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


def parse_study_type(
        name : str,
        autoencoder : bool = False,
        genetic_algorithm : bool = False,
        multiple_autoencoders : bool = False
    ) -> str:
    ''' Parse the study type from the study name.

    Parameters
    ----------
    name : str
        The name of the study.
    autoencoder : bool, optional
        Whether the study is an autoencoder study. Default is False.
    genetic_algorithm : bool, optional
        Whether the study is a genetic algorithm study. Default is False.
    multiple_autoencoders : bool, optional
        Whether the study is a multiple autoencoders study. Default is False.

    Returns
    -------
    str
        The study type.
    '''

    # Determine the dimensional method
    if autoencoder:
        dimensional = "AE"
    elif genetic_algorithm:
        dimensional = "GA"
    elif multiple_autoencoders:
        dimensional = "MAE"
    elif "PCA95" in name:
        dimensional = "PCA95"
    elif "PCA90" in name:
        dimensional = "PCA90"
    elif "PCA85" in name:
        dimensional = "PCA85"
    elif "PCA80" in name:
        dimensional = "PCA80"
    elif "ScoreOnly" in name:
        dimensional = "Scores Only"
    elif "NoScores" in name:
        dimensional = "No Scores"
    else:
        dimensional = ""

    # Determine the ML method
    if "XGB" in name or "XGBoost" in name:
        ml_method = "XGB"
    elif "NN" in name:
        ml_method = "NN"
    elif "Trans" in name:
        ml_method = "Transformer"
    else:
        ml_method = ""

    # Combine dimensional and ML method
    if dimensional:
        return f"{ml_method} + {dimensional}"
    else:
        return ml_method


def analyze_studies_old(
        snames : list[str],
        storage : str,
        n_trials : int = 5,
        verbose : bool = False
    ) -> pd.DataFrame:
    ''' Analyze the studies and get the n best trials. 
    
    Parameters
    ----------
    snames : list[str]
        The list of study names.
    storage : str
        The storage string for the database.
    n_trials : int
        The number of trials to get.
    verbose : bool
        Whether to print the results.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the results.
    '''

    # Create an empty list to store the results
    results = []

    # Define the previous studies flags
    autoencoder = False
    genetic_algorithm = False
    multiple_autoencoders = False

    # Iterate over the study names
    for sname in snames:
        if verbose:
            print(f"\nStudy: {sname}")

        if "AO" in sname:
            if "LIG" in sname or "REC" in sname or "SF" in sname:
                multiple_autoencoders = True
            else:
                autoencoder = True
            # Ignore the study (not needed)
            continue
        elif "feature_selection" in sname or "Feature selection" in sname:
            genetic_algorithm = True
            # Ignore the study (not needed)
            continue
        elif "Pre_" in sname or "pre-" in sname:
            # Ignore the study (not needed)
            continue

        try:
            # Load the study
            study = optuna.load_study(study_name = sname, storage = storage)
        except Exception as e:
            print(f"Error loading study {sname}: {e}")
            continue

        # Get the trials dataframe
        df = study.trials_dataframe()

        # Filter the trials that are complete
        df = df[df['state'] == 'COMPLETE']

        # Filter repeated trials (same value and user_attrs_AUC) (just in case)
        df = df.drop_duplicates(subset=['value', 'user_attrs_AUC'])

        # Calculate the combined metric
        df['combined_metric'] = df['value'] - df['user_attrs_AUC']

        # Convert the number to int
        df['number'] = df['number'].astype(int)

        # Sort the trials by RMSE, AUC and combined metric
        best_rmse_df = df.sort_values(by=['value'], ascending=[True])
        best_auc_df = df.sort_values(by=['user_attrs_AUC'], ascending=[False])
        best_df = df.sort_values(by=['combined_metric'], ascending=[True])

        # Get the study type from the name
        study_type = parse_study_type(sname, autoencoder, genetic_algorithm, multiple_autoencoders)

        autoencoder = False
        genetic_algorithm = False
        multiple_autoencoders = False

        # If n_trials are -1 or bigger than the len of the df, get all the trials
        if n_trials == -1 or n_trials > len(df):
            n_trials = len(df)

        # Get the n best trials
        for i in range(0, n_trials):
            # Append the results to the list
            result = {
                'study_name': sname,
                'study_type': study_type,
                'total_trials': len(df),
                'best_rmse_number': best_rmse_df['number'].iloc[i],
                'best_rmse_value': best_rmse_df['value'].iloc[i],
                'best_rmse_auc': best_rmse_df['user_attrs_AUC'].iloc[i],
            }

            if "Ablation" in sname:
                result['best_rmse_features'] = best_rmse_df['user_attrs_Feature_Mask'].iloc[i]

            result.update({
                'best_auc_number': best_auc_df['number'].iloc[i],
                'best_auc_value': best_auc_df['value'].iloc[i],
                'best_auc': best_auc_df['user_attrs_AUC'].iloc[i],
            })

            if "Ablation" in sname:
                result['best_auc_features'] = best_auc_df['user_attrs_Feature_Mask'].iloc[i]

            result.update({
                'best_combined_number': best_df['number'].iloc[i],
                'best_combined_metric': best_df['combined_metric'].iloc[i],
                'best_combined_value': best_df['value'].iloc[i],
                'best_combined_auc': best_df['user_attrs_AUC'].iloc[i]
            })

            if "Ablation" in sname:
                result['best_combined_features'] = best_df['user_attrs_Feature_Mask'].iloc[i]

            results.append(result)
                
            if verbose:
                ocprint.printv(f"{len(df)}\t{best_rmse_df['number'].iloc[i]}\t{best_rmse_df['value'].iloc[i]}\t{best_rmse_df['user_attrs_AUC'].iloc[i]}\t{best_auc_df['number'].iloc[i]}\t{best_auc_df['value'].iloc[i]}\t{best_auc_df['user_attrs_AUC'].iloc[i]}\t{best_df['number'].iloc[i]}\t{best_df['combined_metric'].iloc[i]}\t{best_df['user_attrs_AUC'].iloc[i]}")

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Return the DataFrame
    return results_df, results_df_auc, results_df_rmse



import optuna
import pandas as pd


def analyze_studies(
    snames: list[str],
    storage: str,
    n_trials: int = 5,
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    For each study, load trials, filter COMPLETE + dedupe,
    compute combined_metric = RMSE - AUC, then pull out
    top-n by RMSE (smallest), top-n by AUC (largest),
    and top-n by combined_metric (smallest).
    Ablation studies also get a 'features' column.
    Returns three DataFrames: df_rmse, df_auc, df_combined.
    '''

    rmse_results = []
    auc_results = []
    combined_results = []

    for sname in snames:
        if verbose:
            print(f"Loading {sname}")
        # skip unwanted studies
        if any(tag in sname for tag in ("AO", "LIG", "REC", "SF",
                                         "feature_selection", "Feature selection",
                                         "Pre_", "pre-")):
            continue

        try:
            study = optuna.load_study(study_name=sname, storage=storage)
        except Exception as e:
            print(f"Could not load {sname}: {e}")
            continue

        df = study.trials_dataframe()
        df = df[df.state == "COMPLETE"].drop_duplicates(subset=["value", "user_attrs_AUC"])
        df["combined_metric"] = df.value - df.user_attrs_AUC
        df["number"] = df.number.astype(int)

        take = len(df) if (n_trials == -1 or n_trials > len(df)) else n_trials

        top_rmse     = df.nsmallest(take, "value")
        top_auc      = df.nlargest(take, "user_attrs_AUC")
        top_combined = df.nsmallest(take, "combined_metric")

        study_type = parse_study_type(sname, False, False, False)
        is_ablation = "Ablation" in sname

        for _, row in top_rmse.iterrows():
            entry = {
                "study_name": sname,
                "study_type": study_type,
                "trial": row.number,
                "rmse": row.value,
                "auc": row.user_attrs_AUC
            }
            if is_ablation:
                entry["features"] = row.user_attrs_Feature_Mask
            rmse_results.append(entry)

        for _, row in top_auc.iterrows():
            entry = {
                "study_name": sname,
                "study_type": study_type,
                "trial": row.number,
                "rmse": row.value,
                "auc": row.user_attrs_AUC
            }
            if is_ablation:
                entry["features"] = row.user_attrs_Feature_Mask
            auc_results.append(entry)

        for _, row in top_combined.iterrows():
            entry = {
                "study_name": sname,
                "study_type": study_type,
                "trial": row.number,
                "combined_metric": row.combined_metric,
                "rmse": row.value,
                "auc": row.user_attrs_AUC
            }
            if is_ablation:
                entry["features"] = row.user_attrs_Feature_Mask
            combined_results.append(entry)

    df_rmse     = pd.DataFrame(rmse_results)
    df_auc      = pd.DataFrame(auc_results)
    df_combined = pd.DataFrame(combined_results)

    return df_rmse, df_auc, df_combined

'''
# Example usage:
snames = [
    'XGBoost optimization',
    'NN_Optimization',
    'NN_Optimization_2',
    'NN_Optimization_3_TPE',
    'NN_Optimization_5_TPE',
    'NN_Optimization_9_TPE',
    'NN_Optimization_10_TPE',
    'NN_Optimization_11_TPE',
    'XGB_Optimization_2',
    'NN_Optimization_12_TPE',
    'NN_Optimization_14_TPE',
    'NN_Optimization_15_TPE',
    'NN_Optimization_16_TPE',
    'NN_Optimization_17_TPE',
    'NN_Optimization_18_TPE',
    'NN_Optimization_19_TPE',
    'NN_Optimization_20_TPE',
    'NN_Optimization_21_TPE',
    'NN_Optimization_22_TPE',
    'NN_Optimization_23_TPE',
    'XGB_Optimization_24',
    'XGB_Optimization_25',
    'Trans_Optimization_26_TPE',
    'XGB_Optimization_27',
    'XGB_Optimization_28',
    'Trans_Optimization_29_TPE',
    'Trans_Optimization_30_TPE',
    'Trans_Optimization_31_TPE',
    'Trans_Optimization_32_TPE',
    'Trans_Optimization_33_TPE',
    'PCA95_NN_Optimization_34_TPE',
    'PCA95_NN_Optimization_35_TPE',
    'PCA95_NN_Optimization_36_TPE',
    'PCA95_NN_Optimization_37_TPE',
    'PCA95_NN_Optimization_38_TPE',
    'PCA95_NN_Optimization_39_TPE',
    'PCA90_NN_Optimization_40_TPE',
    'PCA90_NN_Optimization_41_TPE',
    'PCA90_NN_Optimization_42_TPE',
    'PCA90_NN_Optimization_43_TPE',
    'PCA90_NN_Optimization_44_TPE',
    'PCA90_NN_Optimization_45_TPE',
    'PCA85_NN_Optimization_46_TPE',
    'PCA85_NN_Optimization_47_TPE',
    'PCA85_NN_Optimization_48_TPE',
    'PCA85_NN_Optimization_49_TPE',
    'PCA85_NN_Optimization_50_TPE',
    'PCA85_NN_Optimization_51_TPE',
    'PCA80_NN_Optimization_52_TPE',
    'PCA80_NN_Optimization_53_TPE',
    'PCA80_NN_Optimization_54_TPE',
    'PCA80_NN_Optimization_55_TPE',
    'PCA80_NN_Optimization_56_TPE',
    'PCA80_NN_Optimization_57_TPE',
    'ScoreOnly_NN_Optimization_58_TPE',
    'ScoreOnly_NN_Optimization_59_TPE',
    'ScoreOnly_NN_Optimization_60_TPE',
    'ScoreOnly_NN_Optimization_61_TPE',
    'ScoreOnly_NN_Optimization_62_TPE',
    'ScoreOnly_XGB_Optimization_63',
    'ScoreOnly_XGB_Optimization_64',
    'ScoreOnly_NN_Optimization_65_TPE',
    'ScoreOnly_XGB_Optimization_66',
    'ScoreOnly_XGB_Optimization_67',
    'ScoreOnly_XGB_Optimization_68',
    'ScoreOnly_XGB_Optimization_69',
    'PCA95_XGB_Optimization_70',
    'PCA95_XGB_Optimization_71',
    'PCA95_XGB_Optimization_72',
    'PCA95_XGB_Optimization_73',
    'PCA95_XGB_Optimization_74',
    'PCA95_XGB_Optimization_75',
    'PCA90_XGB_Optimization_76',
    'PCA90_XGB_Optimization_77',
    'PCA90_XGB_Optimization_78',
    'PCA90_XGB_Optimization_79',
    'PCA90_XGB_Optimization_80',
    'PCA90_XGB_Optimization_81',
    'PCA85_XGB_Optimization_82',
    'PCA85_XGB_Optimization_83',
    'PCA85_XGB_Optimization_84',
    'PCA85_XGB_Optimization_85',
    'PCA85_XGB_Optimization_86',
    'PCA85_XGB_Optimization_87',
    'PCA80_XGB_Optimization_88',
    'PCA80_XGB_Optimization_89',
    'PCA80_XGB_Optimization_90',
    'PCA80_XGB_Optimization_91',
    'PCA80_XGB_Optimization_92',
    'PCA80_XGB_Optimization_93',
    'PCA95_Trans_Optimization_94_TPE',
    'PCA95_Trans_Optimization_95_TPE',
    'PCA95_Trans_Optimization_96_TPE',
    'PCA95_Trans_Optimization_97_TPE',
    'PCA95_Trans_Optimization_98_TPE',
    'PCA95_Trans_Optimization_99_TPE',
    'PCA90_Trans_Optimization_100_TPE',
    'PCA90_Trans_Optimization_101_TPE',
    'PCA90_Trans_Optimization_102_TPE',
    'PCA90_Trans_Optimization_103_TPE',
    'PCA90_Trans_Optimization_104_TPE',
    'PCA90_Trans_Optimization_105_TPE',
    'PCA85_Trans_Optimization_106_TPE',
    'PCA85_Trans_Optimization_107_TPE',
    'PCA85_Trans_Optimization_108_TPE',
    'PCA85_Trans_Optimization_109_TPE',
    'PCA85_Trans_Optimization_110_TPE',
    'PCA85_Trans_Optimization_111_TPE',
    'PCA80_Trans_Optimization_112_TPE',
    'PCA80_Trans_Optimization_113_TPE',
    'PCA80_Trans_Optimization_114_TPE',
    'PCA80_Trans_Optimization_115_TPE',
    'PCA80_Trans_Optimization_116_TPE',
    'PCA80_Trans_Optimization_117_TPE',
    'NoScores_XGB_Optimization_118',
    'NoScores_XGB_Optimization_119',
    'NoScores_XGB_Optimization_120',
    'NoScores_XGB_Optimization_121',
    'NoScores_XGB_Optimization_122',
    'NoScores_XGB_Optimization_123',
    'NoScores_Trans_Optimization_124_TPE',
    'NoScores_Trans_Optimization_125_TPE',
    'NoScores_Trans_Optimization_126_TPE',
    'NoScores_Trans_Optimization_127_TPE',
    'NoScores_Trans_Optimization_128_TPE',
    'NoScores_Trans_Optimization_129_TPE'
]

snames = [
    'XGBoost pre-optimization', 'Feature selection Custom GA', 'XGBoost optimization',
    'NN_Optimization',
    'NN_Optimization_2',
    'NN_Optimization_3_TPE',
    'NN_Optimization_5_TPE',
    'AO_Optimization_9_TPE', 'NN_Optimization_9_TPE',
    'AO_Optimization_10_TPE', 'NN_Optimization_10_TPE',
    'AO_Optimization_11_TPE', 'NN_Optimization_11_TPE',
    'Pre_XGB_Optimization_2', 'feature_selection_2', 'XGB_Optimization_2',
    'AO_Optimization_SF_12_TPE', 'AO_Optimization_LIG_12_TPE', 'AO_Optimization_REC_12_TPE', 'NN_Optimization_12_TPE',
    'AO_Optimization_LIG_14_TPE', 'AO_Optimization_REC_14_TPE', 'NN_Optimization_14_TPE',
    'AO_Optimization_LIG_15_TPE', 'AO_Optimization_REC_15_TPE', 'NN_Optimization_15_TPE',
    'AO_Optimization_16_TPE', 'NN_Optimization_16_TPE',
    'AO_Optimization_17_TPE', 'NN_Optimization_17_TPE',
    'AO_Optimization_18_TPE', 'NN_Optimization_18_TPE',
    'AO_Optimization_19_TPE', 'NN_Optimization_19_TPE',
    'AO_Optimization_LIG_20_TPE', 'AO_Optimization_REC_20_TPE', 'NN_Optimization_20_TPE',
    'AO_Optimization_LIG_21_TPE', 'AO_Optimization_REC_21_TPE', 'NN_Optimization_21_TPE',
    'AO_Optimization_LIG_22_TPE', 'AO_Optimization_REC_22_TPE', 'NN_Optimization_22_TPE',
    'NN_Optimization_23_TPE',
    'Pre_XGB_Optimization_24', 'feature_selection_24', 'XGB_Optimization_24',
    'Pre_XGB_Optimization_25', 'feature_selection_25', 'XGB_Optimization_25',
    'Trans_Optimization_26_TPE',
    'Pre_XGB_Optimization_27', 'feature_selection_27', 'XGB_Optimization_27',
    'Pre_XGB_Optimization_28', 'feature_selection_28', 'XGB_Optimization_28',
    'Trans_Optimization_29_TPE',
    'Trans_Optimization_30_TPE',
    'Trans_Optimization_31_TPE',
    'Trans_Optimization_32_TPE',
    'Trans_Optimization_33_TPE',
    'PCA95_NN_Optimization_34_TPE',
    'PCA95_NN_Optimization_35_TPE',
    'PCA95_NN_Optimization_36_TPE',
    'PCA95_NN_Optimization_37_TPE',
    'PCA95_NN_Optimization_38_TPE',
    'PCA95_NN_Optimization_39_TPE',
    'PCA90_NN_Optimization_40_TPE',
    'PCA90_NN_Optimization_41_TPE',
    'PCA90_NN_Optimization_42_TPE',
    'PCA90_NN_Optimization_43_TPE',
    'PCA90_NN_Optimization_44_TPE',
    'PCA90_NN_Optimization_45_TPE',
    'PCA85_NN_Optimization_46_TPE',
    'PCA85_NN_Optimization_47_TPE',
    'PCA85_NN_Optimization_48_TPE',
    'PCA85_NN_Optimization_49_TPE',
    'PCA85_NN_Optimization_50_TPE',
    'PCA85_NN_Optimization_51_TPE',
    'PCA80_NN_Optimization_52_TPE',
    'PCA80_NN_Optimization_53_TPE',
    'PCA80_NN_Optimization_54_TPE',
    'PCA80_NN_Optimization_55_TPE',
    'PCA80_NN_Optimization_56_TPE',
    'PCA80_NN_Optimization_57_TPE',
    'ScoreOnly_NN_Optimization_58_TPE',
    'ScoreOnly_NN_Optimization_59_TPE',
    'ScoreOnly_NN_Optimization_60_TPE',
    'ScoreOnly_NN_Optimization_61_TPE',
    'ScoreOnly_NN_Optimization_62_TPE',
    'ScoreOnly_XGB_Optimization_63',
    'ScoreOnly_XGB_Optimization_64',
    'ScoreOnly_NN_Optimization_65_TPE',
    'ScoreOnly_XGB_Optimization_66',
    'ScoreOnly_XGB_Optimization_67',
    'ScoreOnly_XGB_Optimization_68',
    'ScoreOnly_XGB_Optimization_69',
    'PCA95_XGB_Optimization_70',
    'PCA95_XGB_Optimization_71',
    'PCA95_XGB_Optimization_72',
    'PCA95_XGB_Optimization_73',
    'PCA95_XGB_Optimization_74',
    'PCA95_XGB_Optimization_75',
    'PCA90_XGB_Optimization_76',
    'PCA90_XGB_Optimization_77',
    'PCA90_XGB_Optimization_78',
    'PCA90_XGB_Optimization_79',
    'PCA90_XGB_Optimization_80',
    'PCA90_XGB_Optimization_81',
    'PCA85_XGB_Optimization_82',
    'PCA85_XGB_Optimization_83',
    'PCA85_XGB_Optimization_84',
    'PCA85_XGB_Optimization_85',
    'PCA85_XGB_Optimization_86',
    'PCA85_XGB_Optimization_87',
    'PCA80_XGB_Optimization_88',
    'PCA80_XGB_Optimization_89',
    'PCA80_XGB_Optimization_90',
    'PCA80_XGB_Optimization_91',
    'PCA80_XGB_Optimization_92',
    'PCA80_XGB_Optimization_93',
    'PCA95_Trans_Optimization_94_TPE',
    'PCA95_Trans_Optimization_95_TPE',
    'PCA95_Trans_Optimization_96_TPE',
    'PCA95_Trans_Optimization_97_TPE',
    'PCA95_Trans_Optimization_98_TPE',
    'PCA95_Trans_Optimization_99_TPE',
    'PCA90_Trans_Optimization_100_TPE',
    'PCA90_Trans_Optimization_101_TPE',
    'PCA90_Trans_Optimization_102_TPE',
    'PCA90_Trans_Optimization_103_TPE',
    'PCA90_Trans_Optimization_104_TPE',
    'PCA90_Trans_Optimization_105_TPE',
    'PCA85_Trans_Optimization_106_TPE',
    'PCA85_Trans_Optimization_107_TPE',
    'PCA85_Trans_Optimization_108_TPE',
    'PCA85_Trans_Optimization_109_TPE',
    'PCA85_Trans_Optimization_110_TPE',
    'PCA85_Trans_Optimization_111_TPE',
    'PCA80_Trans_Optimization_112_TPE',
    'PCA80_Trans_Optimization_113_TPE',
    'PCA80_Trans_Optimization_114_TPE',
    'PCA80_Trans_Optimization_115_TPE',
    'PCA80_Trans_Optimization_116_TPE',
    'PCA80_Trans_Optimization_117_TPE',
    'NoScores_XGB_Optimization_118',
    'NoScores_XGB_Optimization_119',
    'NoScores_XGB_Optimization_120',
    'NoScores_XGB_Optimization_121',
    'NoScores_XGB_Optimization_122',
    'NoScores_XGB_Optimization_123',
    'NoScores_Trans_Optimization_124_TPE',
    'NoScores_Trans_Optimization_125_TPE',
    'NoScores_Trans_Optimization_126_TPE',
    'NoScores_Trans_Optimization_127_TPE',
    'NoScores_Trans_Optimization_128_TPE',
    'NoScores_Trans_Optimization_129_TPE',
    'NoScores_NN_Optimization_130_TPE',
    'NoScores_NN_Optimization_131_TPE',
    'NoScores_NN_Optimization_132_TPE',
    'NoScores_NN_Optimization_133_TPE',
    'NoScores_NN_Optimization_134_TPE',
    'NoScores_NN_Optimization_135_TPE'
]
user = "ocdocker"
password = "@Kp3sRv9t@"
host = "localhost"
port = 3306
db = "optimization"

results_df = analyze_studies(snames, storage=f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db}")
#print(results_df)
df = results_df.copy()
'''
