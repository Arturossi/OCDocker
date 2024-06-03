#!/usr/bin/env python3

# Description
###############################################################################
""" Module to perform the optimization of the Transformer parameters model
using Optuna."""

# Imports
###############################################################################

import optuna
import pandas as pd

from urllib.parse import quote_plus

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Methods
###############################################################################

def analyze_studies(snames: list[str], storage: str, n_trials: int = 5, verbose: bool = False) -> pd.DataFrame:
    ''' Function to analyze the studies and get the best trials. 
    
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

    # Iterate over the study names
    for sname in snames:
        if verbose:
            print(f"\nStudy: {sname}")

        # Load the study
        study = optuna.load_study(study_name = sname, storage = storage)

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

        # Get the n best trials
        for i in range(0, n_trials):
            # Append the results to the list
            results.append({
                'study_name': sname,
                'total_trials': len(df),
                'best_rmse_number': best_rmse_df['number'].iloc[i],
                'best_rmse_value': best_rmse_df['value'].iloc[i],
                'best_rmse_auc': best_rmse_df['user_attrs_AUC'].iloc[i],
                'best_auc_number': best_auc_df['number'].iloc[i],
                'best_auc_value': best_auc_df['value'].iloc[i],
                'best_auc': best_auc_df['user_attrs_AUC'].iloc[i],
                'best_combined_number': best_df['number'].iloc[i],
                'best_combined_metric': best_df['combined_metric'].iloc[i],
                'best_combined_value': best_df['value'].iloc[i],
                'best_combined_auc': best_df['user_attrs_AUC'].iloc[i]
            })

            if verbose:
                print(f"{len(df)}\t{best_rmse_df['number'].iloc[i]}\t{best_rmse_df['value'].iloc[i]}\t{best_rmse_df['user_attrs_AUC'].iloc[i]}\t{best_auc_df['number'].iloc[i]}\t{best_auc_df['value'].iloc[i]}\t{best_auc_df['user_attrs_AUC'].iloc[i]}\t{best_df['number'].iloc[i]}\t{best_df['combined_metric'].iloc[i]}\t{best_df['user_attrs_AUC'].iloc[i]}")

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Return the DataFrame
    return results_df

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
    'XGB_Optimization_27'
    'XGB_Optimization_28'
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
    'PCA80_Trans_Optimization_117_TPE'
]

user = "ocdocker"
password = "@Kp3sRv9t@"
host = "localhost"
port = 3306
db = "optimization"

results_df = analyze_studies(snames, storage=f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db}")
print(results_df)
